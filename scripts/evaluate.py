import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from scienceagent.release import WORLD_VARIANCES, require_world, save_json


def summarize(paths):
    groups = defaultdict(list)
    explanations, finite_errors = [], []
    nonfinite, missing, episodes = 0, 0, 0
    seen = set()
    for path in paths:
        row = json.loads(Path(path).read_text())
        world = row["world"]
        require_world(world)
        key = (world, row["noise_seed"])
        if key in seen:
            raise ValueError(f"Duplicate session: {key}")
        seen.add(key)
        result = row.get("evaluation") or {}
        raw_score = (result.get("explanation") or {}).get("score")
        raw_error = result.get("mean_pos_error")
        score = float(raw_score) if raw_score is not None else 0.0
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid explanation score in {path}")
        error = (
            float(raw_error) / WORLD_VARIANCES[world]
            if raw_error is not None
            else math.inf
        )
        if raw_score is None or raw_error is None:
            missing += 1
        explanations.append(score)
        if math.isfinite(error) and error >= 0:
            finite_errors.append(error)
        else:
            nonfinite += 1
        groups[world].append(math.isfinite(error) and 0 <= error < 0.1 and score >= 0.9)
        episodes += int(row.get("simulator_episodes_used") or 0)
    if not seen:
        raise ValueError("No session files")
    pass_at = {}
    for k in range(1, 6):
        if any(len(v) < k for v in groups.values()):
            pass_at[str(k)] = None
            continue
        values = []
        for passed in groups.values():
            n, c = len(passed), sum(passed)
            values.append(
                1 - math.comb(n - c, k) / math.comb(n, k) if n - c >= k else 1.0
            )
        pass_at[str(k)] = sum(values) / len(values)
    return {
        "sessions": len(seen),
        "worlds": len(groups),
        "explanation_mean": sum(explanations) / len(explanations),
        "normalized_mse_finite_geometric_mean": math.exp(
            sum(math.log(max(x, 1e-14)) for x in finite_errors) / len(finite_errors)
        )
        if finite_errors
        else None,
        "finite_mse_count": len(finite_errors),
        "nonfinite_mse_count": nonfinite,
        "missing_evaluation_count": missing,
        "pass_at_k": pass_at,
        "simulator_episodes_total": episodes,
        "simulator_episodes_mean": episodes / len(seen),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize(sorted(Path(args.input_dir).glob("*_seed*.json")))
    save_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
