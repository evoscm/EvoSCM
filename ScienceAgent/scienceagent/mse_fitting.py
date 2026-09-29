from __future__ import annotations
import argparse
import functools
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Callable, Optional
import numpy as np
from scienceagent.evaluator import (
    _compile_fit_parameters,
    _compile_law,
    _dark_matter_loss,
    _ether_loss,
    _fit_law_parameters,
    _three_species_loss,
    _two_particle_loss,
    _circle_loss,
    _validate_fit_spec,
    clean_law_source,
)
from scienceagent.load_trajectories import (
    ExperimentTrajectory,
    default_csv_path,
    load_trajectories,
)

_LOSS_FNS: dict[str, Callable] = {
    "gravity": _two_particle_loss,
    "yukawa": _two_particle_loss,
    "fractional": _two_particle_loss,
    "oscillator": _two_particle_loss,
    "extra_dimensions": _two_particle_loss,
    "circle": _circle_loss,
    "three_species": _three_species_loss,
    "dark_matter": _dark_matter_loss,
    "ether": _ether_loss,
    "hubble": _ether_loss,
    "coulomb_easy": _two_particle_loss,
}


def redact_world_leak(
    msg: str, csv_path: Optional[os.PathLike | str], world: str
) -> str:
    s = str(msg)
    needles: list[str] = []
    if csv_path is not None:
        p = Path(csv_path)
        needles.extend([str(p), str(p.resolve()), p.name])
    if world:
        needles.append(world)
    for needle in needles:
        if needle:
            s = s.replace(needle, "<redacted>")
    return s


def fit_law(
    law_source: str,
    world: str,
    csv_path: Optional[os.PathLike | str] = None,
    run_id: Optional[str] = None,
    training_samples: Optional[list[dict]] = None,
) -> dict:
    result = {
        "loss_before": None,
        "loss_after": None,
        "fitted_params": {},
        "declared_params": {},
        "n_training": 0,
        "training_mode": None,
        "error": None,
    }
    loss_fn = _LOSS_FNS.get(world)
    if loss_fn is None:
        result[
            "error"
        ] = f"world '{world}' has no continuous parameters to fit; supported worlds: {sorted(_LOSS_FNS)}"
        return result
    if training_samples is not None:
        training = [
            sample
            for sample in training_samples
            if isinstance(sample, dict)
            and isinstance(sample.get("input"), dict)
            and isinstance(sample.get("output"), dict)
        ]
        result["training_mode"] = (
            "paired_conversation"
            if any(
                (isinstance(sample.get("_paired_factual"), dict) for sample in training)
            )
            else "conversation"
        )
    else:
        if not run_id:
            result["error"] = "run_id is required so the fit only sees this run's data"
            return result
        csv_path = Path(csv_path) if csv_path else default_csv_path(world)
        if not csv_path.exists():
            result[
                "error"
            ] = "trajectory CSV not found for this run; run at least one successful <run_experiment> first"
            return result
        try:
            experiments = load_trajectories(csv_path, run_id=run_id)
        except Exception as e:
            result[
                "error"
            ] = f"failed to load training data: {redact_world_leak(e, csv_path, world)}"
            return result
        if not experiments:
            result["error"] = f"no training trajectories logged yet for run_id={run_id}"
            return result
        try:
            training = _experiments_to_training(world, experiments)
        except Exception as e:
            result[
                "error"
            ] = "failed to reshape CSV into training samples: " + redact_world_leak(
                e, csv_path, world
            )
            return result
        result["training_mode"] = "trajectory_csv"
    result["n_training"] = len(training)
    if not training:
        result["error"] = "no usable training samples after CSV reshape"
        return result
    try:
        discovered_law = _compile_law(law_source)
    except Exception as e:
        result["error"] = "compile_error: " + redact_world_leak(e, csv_path, world)
        return result
    fit_fn = _compile_fit_parameters(law_source)
    fit_spec_list: list = []
    if fit_fn is not None:
        try:
            raw_spec = fit_fn()
            fit_spec_list = _validate_fit_spec(raw_spec)
        except Exception as e:
            result["error"] = "invalid_fit_parameters: " + redact_world_leak(
                e, csv_path, world
            )
            return result
        result["declared_params"] = {
            name: {"init": init, "bounds": list(bounds)}
            for name, init, bounds in fit_spec_list
        }
    init_kwargs = {name: init for name, init, _ in fit_spec_list}
    init_law = (
        functools.partial(discovered_law, **init_kwargs)
        if init_kwargs
        else discovered_law
    )
    try:
        loss_before = float(loss_fn(init_law, training))
    except Exception:
        loss_before = float("inf")
    result["loss_before"] = _finite_or_none(loss_before)
    if not fit_spec_list:
        result["loss_after"] = result["loss_before"]
        result["fitted_params"] = {}
        return result
    try:
        fitted = _fit_law_parameters(discovered_law, fit_spec_list, training, loss_fn)
    except Exception as e:
        result["error"] = "optimizer_failure: " + redact_world_leak(e, csv_path, world)
        result["fitted_params"] = init_kwargs
        result["loss_after"] = result["loss_before"]
        return result
    bound_law = functools.partial(discovered_law, **fitted)
    try:
        loss_after = float(loss_fn(bound_law, training))
    except Exception:
        loss_after = float("inf")
    result["fitted_params"] = fitted
    result["loss_after"] = _finite_or_none(loss_after)
    return result


def _finite_or_none(x: float) -> Optional[float]:
    return None if not np.isfinite(x) else float(x)


_PACK_FNS: dict[str, Callable[[ExperimentTrajectory], dict]] = {}


def _experiments_to_training(
    world: str, experiments: list[ExperimentTrajectory]
) -> list[dict]:
    pack = _PACK_FNS.get(world)
    if pack is not None:
        return [pack(e) for e in experiments]
    if world == "circle":
        return [_one_circle_sample(e) for e in experiments]
    if world in ("ether", "hubble"):
        return [_one_ether_sample(e) for e in experiments]
    if world in ("three_species", "dark_matter"):
        return [_one_full_state_sample(e) for e in experiments]
    return [_one_two_particle_sample(e) for e in experiments]


def _one_two_particle_sample(exp: ExperimentTrajectory) -> dict:
    times_obs = exp.times[1:].tolist()
    pos2_obs = exp.positions[1:, 1, :].tolist()
    vel2_obs = exp.velocities[1:, 1, :].tolist()
    pos2_init = exp.initial_positions[1].tolist()
    vel2_init = exp.initial_velocities[1].tolist()
    input_case = {
        "p1": exp.params.get("p1"),
        "p2": exp.params.get("p2"),
        "pos2": pos2_init,
        "velocity2": vel2_init,
        "measurement_times": times_obs,
    }
    if exp.params.get("start_time") is not None:
        input_case["start_time"] = exp.params["start_time"]
    return {
        "input": input_case,
        "output": {
            "measurement_times": times_obs,
            "pos2": pos2_obs,
            "velocity2": vel2_obs,
        },
    }


def _one_circle_sample(exp: ExperimentTrajectory) -> dict:
    times_obs = exp.times[1:].tolist()
    positions_obs = exp.positions[1:].tolist()
    velocities_obs = exp.velocities[1:].tolist()
    return {
        "input": {
            "ring_radius": exp.params.get("ring_radius"),
            "initial_tangential_velocity": exp.params.get(
                "initial_tangential_velocity"
            ),
            "measurement_times": times_obs,
        },
        "output": {
            "measurement_times": times_obs,
            "positions": positions_obs,
            "velocities": velocities_obs,
        },
    }


def _one_full_state_sample(exp: ExperimentTrajectory) -> dict:
    times_obs = exp.times[1:].tolist()
    positions_obs = exp.positions[1:].tolist()
    return {
        "input": {
            "init_positions": exp.initial_positions.tolist(),
            "init_velocities": exp.initial_velocities.tolist(),
            "measurement_times": times_obs,
        },
        "output": {"measurement_times": times_obs, "positions": positions_obs},
    }


def _one_ether_sample(exp: ExperimentTrajectory) -> dict:
    times_obs = exp.times[1:].tolist()
    positions_obs = exp.positions[1:].tolist()
    masses = exp.masses
    if masses is None or np.isnan(masses).any():
        masses = np.ones(exp.n_particles, dtype=float)
    return {
        "input": {
            "init_positions": exp.initial_positions.tolist(),
            "init_velocities": exp.initial_velocities.tolist(),
            "init_masses": masses.tolist(),
            "measurement_times": times_obs,
        },
        "output": {"measurement_times": times_obs, "positions": positions_obs},
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit a candidate law against a run's trajectory CSV."
    )
    parser.add_argument("world", help="World name (e.g. gravity, yukawa, circle).")
    parser.add_argument("run_id", help="run_id to filter the CSV by.")
    parser.add_argument(
        "law_file",
        help="Path to a Python file defining discovered_law (and optionally fit_parameters).",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Override the trajectory CSV path (default: results/trajectories/<world>.csv).",
    )
    args = parser.parse_args(argv)
    try:
        law_source = Path(args.law_file).read_text()
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    result = fit_law(
        law_source=law_source,
        world=args.world,
        csv_path=args.csv_path,
        run_id=args.run_id,
    )
    print(json.dumps(result, indent=2, default=float))
    return 0 if result["error"] is None else 1


if __name__ == "__main__":
    sys.exit(main())
