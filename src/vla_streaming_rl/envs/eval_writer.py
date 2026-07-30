# SPDX-License-Identifier: MIT
"""Write Bench2Drive-eval-compatible artifacts from inside the RL training loop.

Mirrors what ``leaderboard_evaluator.py`` produces during Bench2Drive's
standard eval scripts so that the same downstream
tools (``Bench2Drive/tools/merge_route_json.py``,
``efficiency_smoothness_benchmark.py``) can compute Driving Score,
Success Rate, Efficiency, and Comfort directly from this run's output.

The training is Test Time Training: each scenario contributes a fully-
fledged eval entry *while the policy is being updated* on the same
rollout, so eval artifacts must be flushed at the end of every episode
(not in a separate post-hoc evaluation pass).

Output layout (matches what ``efficiency_smoothness_benchmark.py``
expects via ``read_from_json(filepath, metric_dir)``):

    {output_root}/
      eval_res/
        {index:03d}_res.json      # one StatisticsManager dump per scenario
        merged.json               # produced by finalize_all() after route 220
        debug_{index:03d}.txt     # StatisticsManager debug endpoint (unused, but required)
      eval_viz/
        {save_name}/
          metric_info.json        # per-step ego physics (acceleration, etc.)
"""

import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import carla
import numpy as np
from leaderboard.utils.statistics_manager import StatisticsManager
from srunner.scenarioconfigs.route_scenario_configuration import (
    RouteScenarioConfiguration,
)
from srunner.scenariomanager.timer import GameTime


def _vector_to_list(vector, rotation: bool = False) -> list[float]:
    """Mirror ``autonomous_agent.AutonomousAgent.get_metric_info``'s helper.

    Bench2Drive's comfort benchmark reads ``rotation`` as ``[roll, pitch,
    yaw]`` and the rest as ``[x, y, z]``; deviating breaks the downstream
    numerics silently.
    """
    if rotation:
        return [float(vector.roll), float(vector.pitch), float(vector.yaw)]
    return [float(vector.x), float(vector.y), float(vector.z)]


def _load_weather_table(weather_xml: Path) -> list[tuple[str, dict[str, str]]]:
    """Pre-parse ``leaderboard/data/weather.xml`` into (id, attrs) tuples.

    The legacy lookup in ``leaderboard_evaluator.get_weather_id`` re-parses
    this file on every route; we read it once per training run.
    """
    tree = ET.parse(str(weather_xml))
    table: list[tuple[str, dict[str, str]]] = []
    for case in tree.getroot().findall("case"):
        weather_id = case.get("weather_id")
        # The first <weather> element in a <case> is the reference for the
        # start-of-route weather, which is what RouteScenarioConfiguration
        # exposes as ``config.weather[0][1]``.
        attrs = {k: v for k, v in case[0].items() if k != "route_percentage"}
        table.append((weather_id, attrs))
    return table


def _resolve_weather_id(
    weather_table: list[tuple[str, dict[str, str]]],
    weather: carla.WeatherParameters,
) -> str | None:
    """Reverse-lookup a CARLA weather struct against ``weather.xml`` cases."""
    for wid, attrs in weather_table:
        if all(str(getattr(weather, k)) == v for k, v in attrs.items()):
            return wid
    return None


class Bench2DriveEvalWriter:
    """Per-route StatisticsManager + per-step metric_info.json writer.

    Lifecycle (one cycle per scenario):
        begin_episode(route_scenario, config, scenario_index)
        record_step(vehicle)   # called every env.step
        ...
        end_episode(failure_message="")

    After the final scenario, ``finalize_all()`` computes the merged
    summary so ``merge_route_json.py`` does not need to be re-run.
    """

    def __init__(self, output_root: Path, weather_xml: Path):
        self.output_root = Path(output_root)
        self.eval_res_dir = self.output_root / "eval_res"
        self.eval_viz_dir = self.output_root / "eval_viz"
        self.eval_res_dir.mkdir(parents=True, exist_ok=True)
        self.eval_viz_dir.mkdir(parents=True, exist_ok=True)
        self._weather_xml = Path(weather_xml)
        self._weather_table = _load_weather_table(self._weather_xml)

        # Episode-scoped state. None outside of an episode.
        self._stats: StatisticsManager | None = None
        self._metric_records: dict[int, dict[str, list[float]]] | None = None
        self._step: int = 0
        self._save_name: str | None = None
        self._scenario_index: int | None = None
        self._wall_start: float = 0.0
        self._game_start: float = 0.0

    def begin_episode(
        self,
        route_scenario,
        config: RouteScenarioConfiguration,
        scenario_index: int,
    ) -> None:
        """Open a fresh per-route ``_res.json`` and start the metric_info buffer."""
        if self._stats is not None:
            raise RuntimeError(
                "begin_episode called twice without end_episode — likely a "
                "missing finalize on the previous episode"
            )

        # One ``StatisticsManager`` per route (mirrors Bench2Drive's eval
        # scripts: leaderboard_evaluator is invoked once per route file, so
        # each output JSON has exactly one entry in ``records``).
        endpoint = self.eval_res_dir / f"{scenario_index:03d}_res.json"
        debug_endpoint = self.eval_res_dir / f"debug_{scenario_index:03d}.txt"
        stats = StatisticsManager(str(endpoint), str(debug_endpoint))
        stats.save_progress(route_index=0, total_routes=1)
        stats.save_entry_status("Started")

        # repetition_index is set by leaderboard's RouteIndexer (not by the
        # bare RouteParser we use); default to 0 so route_name still
        # matches the "RouteScenario_<id>_rep0" convention.
        repetition_index = getattr(config, "repetition_index", 0)
        route_name = f"{config.name}_rep{repetition_index}"
        scenario_name = config.scenario_configs[0].name if config.scenario_configs else "NoScenario"
        town_name = str(config.town)
        weather_id = _resolve_weather_id(self._weather_table, config.weather[0][1])
        save_name = (
            f"{route_name}_{town_name}_{scenario_name}_{weather_id}_"
            f"{datetime.now().strftime('%m_%d_%H_%M_%S')}"
        )
        stats.create_route_data(
            route_id=route_name,
            scenario_name=scenario_name,
            weather_id=weather_id,
            save_name=save_name,
            town_name=town_name,
            index=0,
        )
        stats.set_scenario(route_scenario)

        self._stats = stats
        self._metric_records = {}
        self._step = 0
        self._save_name = save_name
        self._scenario_index = scenario_index
        self._wall_start = time.time()
        self._game_start = GameTime.get_time()

    def record_step(self, vehicle: carla.Vehicle) -> None:
        """Capture per-step ego physics (matches ``get_metric_info``).

        Called once per ``env.step``; values are indexed by step id and
        flushed as a single JSON at ``end_episode``.
        """
        if self._metric_records is None:
            return  # eval not active (e.g. between episodes)
        transform = vehicle.get_transform()
        self._metric_records[self._step] = {
            "acceleration": _vector_to_list(vehicle.get_acceleration()),
            "angular_velocity": _vector_to_list(vehicle.get_angular_velocity()),
            "forward_vector": _vector_to_list(transform.get_forward_vector()),
            "right_vector": _vector_to_list(transform.get_right_vector()),
            "location": _vector_to_list(transform.location),
            "rotation": _vector_to_list(transform.rotation, rotation=True),
        }
        self._step += 1

    def end_episode(self, failure_message: str = "") -> dict[str, float]:
        """Flush ``_res.json`` + ``metric_info.json`` and return summary scores.

        ``failure_message`` is forwarded to ``compute_route_statistics``;
        leave empty to let StatisticsManager derive the status from the
        scenario criteria (collision / off-route / timeout etc.).
        """
        if self._stats is None:
            raise RuntimeError("end_episode without a prior begin_episode")

        duration_system = time.time() - self._wall_start
        duration_game = max(GameTime.get_time() - self._game_start, 0.0)
        self._stats.compute_route_statistics(
            route_index=0,
            duration_time_system=duration_system,
            duration_time_game=duration_game,
            failure_message=failure_message,
        )
        self._stats.compute_global_statistics()
        self._stats.save_entry_status("Finished")
        # remove_scenario before write so the JSON does not retain a live
        # reference to the (about-to-be-cleaned-up) RouteScenario object.
        self._stats.remove_scenario()
        self._stats.write_statistics()

        # Per-step physics → metric_info.json. Bench2Drive's
        # efficiency_smoothness_benchmark.read_from_json reads
        # ``{metric_dir}/{save_name}/metric_info.json``.
        viz_dir = self.eval_viz_dir / self._save_name
        viz_dir.mkdir(parents=True, exist_ok=True)
        with open(viz_dir / "metric_info.json", "w") as f:
            json.dump(self._metric_records, f)

        # Pull a small summary back for wandb logging.
        record = self._stats._results.checkpoint.records[0]
        summary = {
            "score_composed": float(record.scores.get("score_composed", 0.0)),
            "score_route": float(record.scores.get("score_route", 0.0)),
            "score_penalty": float(record.scores.get("score_penalty", 1.0)),
            "num_infractions": int(record.num_infractions),
            "status": record.status,
        }

        self._stats = None
        self._metric_records = None
        self._save_name = None
        self._scenario_index = None
        return summary

    def finalize_all(self) -> dict[str, float] | None:
        """After the final scenario, write ``eval_res/merged.json``.

        Mirrors ``Bench2Drive/tools/merge_route_json.py`` so the same
        downstream consumers (Driving Score, Success Rate) can read this
        directly without invoking the merge script.

        Returns the aggregate ``{driving_score, success_rate, eval_num,
        comfort_rate, driving_efficiency}`` dict, or ``None`` if there are
        no eval results to merge.
        """
        from glob import glob

        files = sorted(glob(str(self.eval_res_dir / "*_res.json")))
        if not files:
            return None

        merged_records = []
        driving_score = []
        success_num = 0
        for path in files:
            with open(path) as fh:
                data = json.load(fh)
            for rd in data["_checkpoint"]["records"]:
                if rd["status"] == "Failed - Agent crashed":
                    continue
                rd.pop("index", None)
                merged_records.append(rd)
                driving_score.append(rd["scores"]["score_composed"])
                if rd["status"] in ("Completed", "Perfect"):
                    # min_speed_infractions does not count as a failure
                    # (matches merge_route_json.py).
                    if all(
                        not v or k == "min_speed_infractions" for k, v in rd["infractions"].items()
                    ):
                        success_num += 1

        if not driving_score:
            return None

        merged = {
            "_checkpoint": {
                "records": sorted(merged_records, key=lambda d: d["route_id"], reverse=True)
            },
            "driving score": sum(driving_score) / len(driving_score),
            "success rate": success_num / len(driving_score),
            "eval num": len(driving_score),
        }

        # Efficiency + Comfort require the per-step metric_info.json
        # files; compute them inline so the trainer log can report all
        # four headline numbers at the end of the 220-scenario sweep.
        efficiency, comfort = self._compute_efficiency_and_comfort(merged_records)
        if efficiency is not None:
            merged["driving efficiency"] = efficiency
        if comfort is not None:
            merged["driving comfort"] = comfort

        with open(self.eval_res_dir / "merged.json", "w") as f:
            json.dump(merged, f, indent=4)

        return {
            "driving_score": merged["driving score"],
            "success_rate": merged["success rate"],
            "eval_num": float(merged["eval num"]),
            **({"driving_efficiency": efficiency} if efficiency is not None else {}),
            **({"driving_comfort": comfort} if comfort is not None else {}),
        }

    def _compute_efficiency_and_comfort(
        self, records: list[dict]
    ) -> tuple[float | None, float | None]:
        """Inline port of ``efficiency_smoothness_benchmark.py``.

        Bench2Drive's ``tools/`` ships without ``__init__.py``, so the
        file is loaded by path (derived from ``weather_xml``'s standard
        layout: ``<B2D_ROOT>/leaderboard/data/weather.xml`` →
        ``<B2D_ROOT>/tools/efficiency_smoothness_benchmark.py``).
        """
        import importlib.util
        import re

        bench_tool = self._weather_xml.parents[2] / "tools" / "efficiency_smoothness_benchmark.py"
        if not bench_tool.exists():
            return None, None
        spec = importlib.util.spec_from_file_location("efficiency_smoothness_benchmark", bench_tool)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        seg_compute_comfort_metric = mod.seg_compute_comfort_metric

        driving_efficiency: list[float] = []
        comfort_res: list[float] = []
        for record in records:
            metric_path = self.eval_viz_dir / record["save_name"] / "metric_info.json"
            if not metric_path.exists():
                continue
            with open(metric_path) as fh:
                json_data = json.load(fh)
            tmp = {
                k: []
                for k in (
                    "acceleration",
                    "angular_velocity",
                    "forward_vector",
                    "right_vector",
                    "location",
                    "rotation",
                )
            }
            for _, v in json_data.items():
                for k in tmp:
                    tmp[k].append(v[k])
            tmp = {k: np.array(v) for k, v in tmp.items()}
            comfort_res.append(seg_compute_comfort_metric(**tmp))

            speed_msgs = record["infractions"].get("min_speed_infractions", [])
            if speed_msgs:
                pcts = []
                for msg in speed_msgs:
                    m = re.search(r"\b\d+\.?\d*%", msg)
                    if m is None:
                        continue
                    pct = float(m.group().rstrip("%"))
                    if pct > 1000:
                        continue
                    pcts.append(pct)
                if pcts:
                    driving_efficiency.append(sum(pcts) / len(pcts))

        efficiency = (
            sum(driving_efficiency) / len(driving_efficiency) if driving_efficiency else None
        )
        comfort = sum(comfort_res) / len(comfort_res) if comfort_res else None
        return efficiency, comfort
