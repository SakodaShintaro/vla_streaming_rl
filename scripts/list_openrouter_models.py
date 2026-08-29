# SPDX-License-Identifier: MIT
"""List the OpenRouter models that meet what ZeroShotVLMAgent requires.

The agent sends an image plus chat history and expects text back, with
``temperature`` and ``max_tokens`` honoured, so a model is usable only if it
accepts image input, emits text output, and supports both parameters. The
catalogue endpoint needs no API key.
"""

import argparse
import json
import urllib.request

MODELS_URL = "https://openrouter.ai/api/v1/models"
REQUIRED_PARAMETERS = ("temperature", "max_tokens")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sort",
        type=str,
        default="id",
        choices=["id", "price", "context"],
        help="id: alphabetical. price: cheapest input first. context: longest first.",
    )
    parser.add_argument(
        "--max_input_price",
        type=float,
        default=float("inf"),
        help="Keep models at or below this input price in USD per million tokens.",
    )
    parser.add_argument(
        "--min_context",
        type=int,
        default=0,
        help="Keep models with at least this context length.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Keep models whose slug contains this substring (e.g. 'gemini').",
    )
    parser.add_argument(
        "--include_batch",
        type=int,
        default=0,
        choices=[0, 1],
        help="Include ':batch' endpoints, which are not usable for a live rollout.",
    )
    return parser.parse_args()


def fetch_models() -> list[dict]:
    with urllib.request.urlopen(MODELS_URL, timeout=30) as response:
        return json.loads(response.read())["data"]


def is_usable(model: dict, include_batch: bool) -> bool:
    architecture = model["architecture"]
    supported = model["supported_parameters"]
    return (
        "image" in architecture["input_modalities"]
        and "text" in architecture["output_modalities"]
        and all(parameter in supported for parameter in REQUIRED_PARAMETERS)
        and (include_batch or not model["id"].endswith(":batch"))
    )


def price_per_mtok(model: dict, key: str) -> float:
    return float(model["pricing"][key]) * 1e6


def main() -> None:
    args = parse_args()

    models = [m for m in fetch_models() if is_usable(m, bool(args.include_batch))]
    models = [
        m
        for m in models
        if price_per_mtok(m, "prompt") <= args.max_input_price
        and m["context_length"] >= args.min_context
        and args.query in m["id"]
    ]

    sort_keys = {
        "id": lambda m: m["id"],
        "price": lambda m: price_per_mtok(m, "prompt"),
        "context": lambda m: -m["context_length"],
    }
    models.sort(key=sort_keys[args.sort])

    print(f"{'model':<48} {'context':>10} {'$in/Mtok':>10} {'$out/Mtok':>10}")
    for model in models:
        print(
            f"{model['id']:<48} {model['context_length']:>10,} "
            f"{price_per_mtok(model, 'prompt'):>10.3f} "
            f"{price_per_mtok(model, 'completion'):>10.3f}"
        )
    print(f"\n{len(models)} models match (image input, text output, temperature + max_tokens).")


if __name__ == "__main__":
    main()
