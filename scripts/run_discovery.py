import argparse
import os
from scienceagent.release import WORLD_VARIANCES, run_session, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", choices=sorted(WORLD_VARIANCES), default="yukawa")
    parser.add_argument("--method", choices=["baseline", "evoscm"], default="evoscm")
    parser.add_argument("--model", default=os.environ.get("EVOSCM_MODEL", "gpt-5.5"))
    parser.add_argument("--judge-model", default="claude-opus-4-6")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--noise-frac", type=float)
    parser.add_argument("--max-rounds", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--scm-max-candidates", type=int, default=8)
    parser.add_argument("--context-max-chars", type=int)
    parser.add_argument("--output", required=True)
    args = vars(parser.parse_args())
    output = args.pop("output")
    from pathlib import Path

    if Path(output).exists():
        parser.error("Output already exists")
    save_json(output, run_session(**args))


if __name__ == "__main__":
    main()
