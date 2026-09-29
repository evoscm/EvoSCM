import json
import math
import tempfile
from pathlib import Path
from .agent import DiscoveryAgent
from .evaluator import (
    CircleEvaluator,
    DarkMatterEvaluator,
    EtherEvaluator,
    Evaluator,
    ExplanationJudge,
    HubbleEvaluator,
    ThreeSpeciesEvaluator,
    _extract_training_trajectories,
)
from .scm_pipeline import SCMPipeline
from .trajectory_logger import TrajectoryLogger, make_run_id
from .worlds import get_world

WORLD_VARIANCES = {
    "gravity": 4.283,
    "yukawa": 5.677,
    "fractional": 5.721,
    "circle": 6.596,
    "three_species": 28.717,
    "dark_matter": 63.303,
    "ether": 21.259,
    "hubble": 41.189,
    "oscillator": 6.332,
    "extra_dimensions": 4.248,
    "coulomb_easy": 4.244,
}
ROOT = Path(__file__).resolve().parents[2]


def require_world(world):
    if world not in WORLD_VARIANCES:
        raise ValueError(f"Unsupported world: {world}")


def evaluator_for(world, executor):
    require_world(world)
    classes = {
        "circle": CircleEvaluator,
        "three_species": ThreeSpeciesEvaluator,
        "dark_matter": DarkMatterEvaluator,
        "ether": EtherEvaluator,
        "hubble": HubbleEvaluator,
    }
    return classes.get(world, Evaluator)(executor)


def evaluate_law(world, law, explanation, judge_model, executor, training=None):
    config = get_world(world, engine="nbody")
    kwargs = {"verbose": False}
    if training is not None and world not in ("three_species", "dark_matter"):
        kwargs["training_trajectories"] = training
    evaluation = evaluator_for(world, executor).evaluate(law, **kwargs)
    evaluation["explanation"] = ExplanationJudge(judge_model=judge_model).score(
        agent_explanation=explanation,
        optimal_explanation=config["optimal_explanation"],
        rubric=config["explanation_rubric"],
        verbose=False,
    )
    return evaluation


def run_session(
    world,
    model,
    judge_model=None,
    method="evoscm",
    seed=0,
    noise_frac=0.05,
    max_rounds=16,
    max_tokens=8192,
    max_episodes=None,
    context_max_chars=None,
):
    require_world(world)
    if method not in ("baseline", "evoscm"):
        raise ValueError("method must be baseline or evoscm")
    if max_episodes is not None and max_episodes < 0:
        raise ValueError("max_episodes must be nonnegative")
    judge_model = judge_model or "claude-opus-4-6"
    noise_std = noise_frac * math.sqrt(WORLD_VARIANCES[world])
    config = get_world(world, engine="nbody", noise_std=noise_std, noise_seed=seed)
    pipeline = (
        SCMPipeline(
            world=world,
            mission=config["mission"],
            strict=True,
            observation_noise_std=noise_std,
            public_experiment_format=config["experiment_format"],
        )
        if method == "evoscm"
        else None
    )
    with tempfile.TemporaryDirectory(prefix="evoscm-") as temporary:
        logger = TrajectoryLogger(
            world=world,
            executor=config["executor"],
            csv_path=Path(temporary) / "trajectories.csv",
            run_id=make_run_id(model),
        )
        agent = DiscoveryAgent(
            model=model,
            executor=config["executor"],
            mission=config["mission"],
            max_tokens=max_tokens,
            verbose=False,
            system_prompt_path=str(ROOT / config["system_prompt"]),
            instructions_path=str(ROOT / config["instructions"]),
            law_stub=config["law_stub"],
            experiment_format=config["experiment_format"],
            trajectory_logger=logger,
            scm_pipeline=pipeline,
            max_rounds=max_rounds,
            max_simulator_episodes=max_episodes,
            context_max_chars=context_max_chars,
        )
        law = agent.run()
        evaluation = None
        error = None
        if law:
            try:
                evaluation = evaluate_law(
                    world,
                    law,
                    agent.discovered_explanation,
                    judge_model,
                    config["executor"],
                    _extract_training_trajectories(agent.conversation_log),
                )
                if (evaluation.get("explanation") or {}).get("score") is None:
                    error = "ExplanationEvaluationFailed"
            except Exception as exc:
                error = type(exc).__name__
        else:
            error = "NoFinalLaw"
        return {
            "world": world,
            "model": model,
            "judge_model": judge_model,
            "method": method,
            "engine": "nbody",
            "noise_seed": seed,
            "noise_std": noise_std,
            "noise_frac": noise_frac,
            "max_rounds": max_rounds,
            "max_tokens": max_tokens,
            "context_max_chars": context_max_chars,
            "max_simulator_episodes": max_episodes,
            "simulator_episodes_used": agent.simulator_episodes_used,
            "final_law": law,
            "final_explanation": agent.discovered_explanation,
            "evaluation": evaluation,
            "evaluation_error": error,
            "rounds": agent.conversation_log,
            "scm_pipeline": pipeline.to_dict() if pipeline else None,
        }


def save_json(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x),
        )
        + "\n"
    )
