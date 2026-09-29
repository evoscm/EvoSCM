import argparse
import json
from pathlib import Path
from scienceagent import llm_client
from scienceagent.release import ROOT, evaluate_law, require_world, save_json
from scienceagent.transfer import (
    _assert_no_forbidden_keys,
    _build_prompt,
    _freeze_scm,
    _parse_response,
)
from scienceagent.worlds import get_world


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--source", required=True)
    export.add_argument("--output", required=True)
    run = sub.add_parser("run")
    run.add_argument("--scm", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--judge-model", default="claude-opus-4-6")
    run.add_argument("--max-tokens", type=int, default=8192)
    run.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Output already exists")
    if args.command == "export":
        source = json.loads(Path(args.source).read_text())
        require_world(source["world"])
        frozen = _freeze_scm(source, Path(args.source).name)
        save_json(
            args.output,
            {
                "world": source["world"],
                "noise_seed": source["noise_seed"],
                "source_model": source["model"],
                "source_simulator_episodes": source["simulator_episodes_used"],
                "scm": frozen,
            },
        )
        return
    artifact = json.loads(Path(args.scm).read_text())
    require_world(artifact["world"])
    _assert_no_forbidden_keys(artifact["scm"])
    config = get_world(artifact["world"], engine="nbody")
    interface = {
        "mission": config["mission"],
        "instructions": (ROOT / config["instructions"]).read_text(),
        "law_stub": config["law_stub"],
    }
    reply = llm_client.complete(
        args.model,
        [{"role": "user", "content": _build_prompt(interface, artifact["scm"])}],
        system="Compile supplied SCM knowledge into executable scientific code.",
        max_tokens=args.max_tokens,
    )
    law, explanation = _parse_response(reply)
    evaluation = evaluate_law(
        artifact["world"],
        law,
        explanation,
        args.judge_model,
        config["executor"],
    )
    save_json(
        args.output,
        {
            "world": artifact["world"],
            "noise_seed": artifact["noise_seed"],
            "model": args.model,
            "judge_model": args.judge_model,
            "source_model": artifact["source_model"],
            "source_simulator_episodes": artifact["source_simulator_episodes"],
            "simulator_episodes_used": 0,
            "final_law": law,
            "final_explanation": explanation,
            "evaluation": evaluation,
            "evaluation_error": "ExplanationEvaluationFailed"
            if (evaluation.get("explanation") or {}).get("score") is None
            else None,
        },
    )


if __name__ == "__main__":
    main()
