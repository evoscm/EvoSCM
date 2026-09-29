import argparse
import json
from pathlib import Path
import yaml
from scienceagent.release import require_world, run_session, save_json


def prepare_sessions(config, output_dir, baseline_dir=None):
    if config.get("method") not in ("baseline", "evoscm"):
        raise ValueError("Invalid method")
    if baseline_dir and config["method"] != "evoscm":
        raise ValueError("Baseline matching requires method evoscm")
    if not config.get("worlds") or not config.get("seeds"):
        raise ValueError("Worlds and seeds must be nonempty")
    if len(set(config["worlds"])) != len(config["worlds"]) or len(
        set(config["seeds"])
    ) != len(config["seeds"]):
        raise ValueError("Duplicate worlds or seeds")
    sessions = []
    for world in config["worlds"]:
        require_world(world)
        for seed in config["seeds"]:
            output = Path(output_dir) / f"{world}_seed{seed}.json"
            if output.exists():
                raise FileExistsError(output)
            settings = {k: v for k, v in config.items() if k not in ("worlds", "seeds")}
            if baseline_dir:
                control = json.loads(
                    (Path(baseline_dir) / f"{world}_seed{seed}.json").read_text()
                )
                if (
                    control["world"] != world
                    or control["noise_seed"] != seed
                    or control.get("method") != "baseline"
                    or control.get("engine") != "nbody"
                ):
                    raise ValueError("Control identity mismatch")
                for field in (
                    "model",
                    "judge_model",
                    "noise_frac",
                    "max_rounds",
                    "max_tokens",
                    "context_max_chars",
                ):
                    if field not in control:
                        raise ValueError(f"Missing control setting: {field}")
                    if field in settings and settings[field] != control[field]:
                        raise ValueError(f"Paired protocol mismatch: {field}")
                    settings[field] = control[field]
                budget = control["simulator_episodes_used"]
                if (
                    isinstance(budget, bool)
                    or not isinstance(budget, int)
                    or budget < 0
                ):
                    raise ValueError("Invalid control episode count")
                settings["max_episodes"] = budget
            sessions.append((world, seed, output, settings))
    return sessions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-dir")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    sessions = prepare_sessions(config, args.output_dir, args.baseline_dir)
    for world, seed, output, settings in sessions:
        result = run_session(world=world, seed=seed, **settings)
        if (
            args.baseline_dir
            and result["simulator_episodes_used"] > settings["max_episodes"]
        ):
            raise RuntimeError("Simulator budget exceeded")
        save_json(output, result)
        print(world, seed, result["evaluation_error"] or "completed", flush=True)


if __name__ == "__main__":
    main()
