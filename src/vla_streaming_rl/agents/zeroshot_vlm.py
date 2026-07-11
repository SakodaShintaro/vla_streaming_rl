# SPDX-License-Identifier: MIT
import re
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import StoppingCriteria, StoppingCriteriaList

from vla_streaming_rl.networks.modules.vlm_backbone import load_model

PERCEPTION_OPEN = "<perception>\n"
REASONING_CLOSE = "</reasoning>"
ANSWER_OPEN = "\n<answer>"
ANSWER_CLOSE = "</answer>"

_NUMBER_RE = re.compile(r"[+-]?\d*\.\d+|[+-]?\d+")


class _StopOnTokenSequence(StoppingCriteria):
    """Stop when the most recently generated tokens match a fixed target sequence —
    used to halt generation as soon as the model emits the requested closing tag."""

    def __init__(self, target_ids: list[int]) -> None:
        self.target_ids = torch.as_tensor(target_ids, dtype=torch.long)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        n = self.target_ids.shape[0]
        if input_ids.shape[1] < n:
            return False
        tail = input_ids[:, -n:]
        target = self.target_ids.to(tail.device)
        return bool((tail == target).all(dim=1).all().item())


def _fallback_parse(text: str, action_dim: int) -> tuple[np.ndarray, bool]:
    """Pick the LAST ``action_dim`` numeric tokens from ``text``. Used when the
    env's label-strict regex fails."""
    matches = _NUMBER_RE.findall(text)
    if len(matches) < action_dim:
        return np.zeros(action_dim, dtype=np.float32), False
    values = np.array([float(m) for m in matches[-action_dim:]], dtype=np.float32)
    return np.clip(values, -1.0, 1.0).reshape(1, action_dim), True


class ZeroShotVLMAgent:
    """Zero-shot VLM controller following the Odysseus structured CoT protocol.

    At each step the current observation is sent to a chat-tuned VLM as a user
    turn; the model is prompted to produce three XML-style sections in order:

      ``<perception>...</perception>`` — describe the visual state of the screen,
      grounding nearby obstacles, agent location, and interactive elements.

      ``<reasoning>...</reasoning>`` — lay out a step-by-step strategy that
      justifies the next action.

      ``<answer>...</answer>`` — emit the textual action that the env's
      ``parse_action_text`` callback can decode into a continuous action vector.

    The most recent ``seq_len`` turns from the current episode (image, assistant
    response, reward observed AFTER that response) are kept in a FIFO and
    prepended to the chat as in-context history. Generation runs in two stages:
    first ``<perception>`` is pre-filled and the model emits up to and including
    ``</reasoning>``; then ``<answer>`` is pre-filled and the model emits the
    action text up to ``</answer>``."""

    def __init__(
        self,
        *,
        model_id: str,
        parse_action_text: Callable[[str], tuple[np.ndarray, bool]],
        action_dim: int,
        format_hint: str,
        seq_len: int,
        max_new_tokens: int,
    ) -> None:
        self.device = "cuda"
        self.model, self.processor = load_model(model_id, use_lora=False, device=self.device)
        self.model.eval()

        self.parse_action_text = parse_action_text
        self.action_dim = action_dim
        self.format_hint = format_hint
        self.seq_len = seq_len
        self.max_new_tokens = max_new_tokens
        self._eos_token_id = self.processor.tokenizer.eos_token_id
        pad_id = self.processor.tokenizer.pad_token_id
        self._pad_token_id = pad_id if pad_id is not None else self._eos_token_id
        tok = self.processor.tokenizer
        self._reasoning_close_ids = tok.encode(REASONING_CLOSE, add_special_tokens=False)
        self._answer_open_ids = tok.encode(ANSWER_OPEN, add_special_tokens=False)
        self._answer_close_ids = tok.encode(ANSWER_CLOSE, add_special_tokens=False)

        # Reserve budget for the answer stage (action text + closing tag); the
        # remainder of max_new_tokens goes to perception + reasoning.
        self._answer_reserve = action_dim * 4 + 12 + len(self._answer_close_ids)
        self._reasoning_budget = max(
            1, max_new_tokens - self._answer_reserve - len(self._reasoning_close_ids)
        )

        self.task_prompt = ""
        self.history: deque[tuple[Image.Image, str, float | None]] = deque(maxlen=seq_len)
        self.last_action = np.zeros(action_dim, dtype=np.float32)
        self._action_count = 0

    def reset(self, task_prompt: str) -> None:
        self.task_prompt = task_prompt
        self.history.clear()
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        self._action_count = 0

    @torch.inference_mode()
    def select_action(self, obs: dict[str, Any], prev_reward: float) -> tuple[np.ndarray, dict]:
        has_prev_reward = self._action_count > 0
        if has_prev_reward and self.history:
            past_image, past_response, _ = self.history[-1]
            self.history[-1] = (past_image, past_response, prev_reward)

        current_image = self._obs_to_pil(obs["image"])
        messages = self._build_messages(
            current_image,
            current_reward=prev_reward if has_prev_reward else None,
        )

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text = text + PERCEPTION_OPEN
        images, _ = process_vision_info(messages)

        inputs = self.processor(text=text, images=images, return_tensors="pt", padding=True)
        inputs.pop("token_type_ids", None)
        inputs = {
            k: v.to(self.device).to(torch.bfloat16)
            if v.dtype.is_floating_point
            else v.to(self.device)
            for k, v in inputs.items()
        }

        prompt_len = inputs["input_ids"].shape[1]

        stage1_out = self._run_reasoning(inputs)
        reasoning_text = self.processor.tokenizer.decode(
            stage1_out[0, prompt_len:], skip_special_tokens=False
        )

        answer_open = torch.tensor(
            [self._answer_open_ids], device=self.device, dtype=stage1_out.dtype
        )
        stage2_input_ids = torch.cat([stage1_out, answer_open], dim=-1)
        answer_emit = self._emit_answer(inputs, stage2_input_ids)
        action_text = answer_emit.split(ANSWER_CLOSE, 1)[0]

        action, parse_used = self._parse_continuous_action(action_text)

        # ``response_text`` is fed back into the chat history and MUST be the
        # actual decoded text of the model's output (so re-tokenizing it on the
        # next turn restores the same token ids — preserving the in-context
        # property). The closing ``</answer>`` is appended unconditionally so
        # the stored message is well-formed even if stage 2 ran out of budget.
        response_text = (
            PERCEPTION_OPEN + reasoning_text + ANSWER_OPEN + action_text + ANSWER_CLOSE
        ).strip()

        if self.seq_len > 0:
            self.history.append((current_image, response_text, None))
        self.last_action = action
        self._action_count += 1

        return action, {"text": response_text, "parse_used": parse_used}

    # ------------------------------------------------------------------
    # Generation stages
    # ------------------------------------------------------------------

    def _run_reasoning(self, inputs: dict) -> torch.LongTensor:
        """Stage 1: emit ``<perception>...</perception><reasoning>...</reasoning>``
        until ``</reasoning>`` is observed (or the budget is hit, in which case
        the closing tag is force-injected). Returns the full ``input_ids``
        tensor extended with the perception + reasoning + closure."""
        stop = StoppingCriteriaList([_StopOnTokenSequence(self._reasoning_close_ids)])
        prompt_len = inputs["input_ids"].shape[1]
        out = self.model.generate(
            **inputs,
            max_new_tokens=self._reasoning_budget,
            do_sample=False,
            num_beams=1,
            eos_token_id=self._eos_token_id,
            pad_token_id=self._pad_token_id,
            stopping_criteria=stop,
        )
        emitted = self.processor.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=False)
        if REASONING_CLOSE not in emitted:
            closure = torch.tensor([self._reasoning_close_ids], device=self.device, dtype=out.dtype)
            out = torch.cat([out, closure], dim=-1)
        return out

    def _emit_answer(self, inputs: dict, action_input_ids: torch.LongTensor) -> str:
        """Stage 2: free-form text generation for the ``<answer>`` content,
        stopping at ``</answer>``. Returns the decoded continuation only
        (excluding the pre-filled ``<answer>`` opener)."""
        stop = StoppingCriteriaList([_StopOnTokenSequence(self._answer_close_ids)])
        stage2_inputs = {**inputs}
        stage2_inputs["input_ids"] = action_input_ids
        stage2_inputs["attention_mask"] = torch.ones_like(action_input_ids)
        out = self.model.generate(
            **stage2_inputs,
            max_new_tokens=self._answer_reserve,
            do_sample=False,
            num_beams=1,
            eos_token_id=self._eos_token_id,
            pad_token_id=self._pad_token_id,
            stopping_criteria=stop,
        )
        new_tokens = out[0, action_input_ids.shape[1] :]
        return self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Action decoding
    # ------------------------------------------------------------------

    def _parse_continuous_action(self, action_text: str) -> tuple[np.ndarray, str]:
        action_array, parse_strict = self.parse_action_text(action_text)
        if parse_strict and len(action_array) > 0:
            return action_array[0].astype(np.float32), "strict"
        action_array, parse_loose = _fallback_parse(action_text, self.action_dim)
        if parse_loose:
            return action_array[0].astype(np.float32), "fallback"
        return self.last_action, "failed"

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------

    def _build_messages(
        self, current_image: Image.Image, current_reward: float | None
    ) -> list[dict]:
        system_prompt = "\n\n".join(p for p in (self.task_prompt, self.format_hint) if p)
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        # The reward shown before user turn i is the one stored on history[i-1] —
        # i.e. the reward observed AFTER history[i-1]'s action. The oldest entry
        # in the FIFO has no preceding entry we still remember, so its reward
        # prefix is dropped.
        for i in range(len(self.history)):
            past_image, past_response, _ = self.history[i]
            reward_prefix = self.history[i - 1][2] if i > 0 else None
            messages.append(
                {"role": "user", "content": self._build_user_content(past_image, reward_prefix)}
            )
            messages.append({"role": "assistant", "content": past_response})
        messages.append(
            {"role": "user", "content": self._build_user_content(current_image, current_reward)}
        )
        return messages

    @staticmethod
    def _build_user_content(image: Image.Image, reward: float | None) -> list[dict]:
        content: list[dict] = []
        if reward is not None:
            content.append({"type": "text", "text": f"Previous reward: {reward:+.3f}"})
        content.append({"type": "image", "image": image})
        return content

    @staticmethod
    def _obs_to_pil(obs: np.ndarray) -> Image.Image:
        img = (obs.transpose(1, 2, 0) * 255).astype(np.uint8)
        return Image.fromarray(img)
