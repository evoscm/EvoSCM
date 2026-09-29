import functools
import inspect
import re
import time
import numpy as np
from typing import Callable, Optional
from scienceagent.executor import (
    SimulationExecutor,
    ThreeSpeciesExecutor,
    DarkMatterExecutor,
    NBodyEtherExecutor,
    NBodyHubbleExecutor,
)

MAX_FIT_PARAMETERS = 5
FIT_MAXITER = 50
FIT_TIME_BUDGET_S = 180.0
LAW_CALL_TIMEOUT_S = 10.0
FIT_MAX_TRAJECTORIES = 4
FIT_MAX_TIMES_PER_TRAJ = 5


class _FitTimeBudgetExceeded(Exception):
    pass


class _LawCallTimeout(Exception):
    pass


def _wrap_with_timeout(fn: Callable, timeout_s: float = LAW_CALL_TIMEOUT_S) -> Callable:
    import signal
    import threading

    if not hasattr(signal, "SIGALRM"):
        return fn
    if threading.current_thread() is not threading.main_thread():
        return fn

    def _handler(signum, frame):
        raise _LawCallTimeout(f"discovered_law exceeded {timeout_s:g}s")

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
        try:
            return fn(*args, **kwargs)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

    return wrapper


_DEFAULT_TEST_CASES = [
    {
        "p1": 1.0,
        "p2": 1.0,
        "pos2": [3.0, 0.0],
        "velocity2": [0.0, 0.5],
        "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    },
    {
        "p1": 2.0,
        "p2": 1.0,
        "pos2": [5.0, 0.0],
        "velocity2": [0.0, 0.0],
        "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    },
    {
        "p1": 1.0,
        "p2": 2.0,
        "pos2": [-4.0, 2.0],
        "velocity2": [0.3, -0.3],
        "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    },
]


class Evaluator:
    def __init__(self, executor: SimulationExecutor, test_cases: list[dict] = None):
        self.executor = executor
        self.test_cases = test_cases or _DEFAULT_TEST_CASES

    def evaluate(
        self,
        law_source: str,
        verbose: bool = True,
        training_trajectories: Optional[list] = None,
    ) -> dict:
        discovered_law = _compile_law(law_source)
        discovered_law, fit_info = _maybe_fit(
            law_source,
            discovered_law,
            training_trajectories,
            _two_particle_loss,
            verbose,
        )
        with self.executor.noise_disabled():
            ground_truths = self.executor.run(self.test_cases)
        per_case_errors = []
        all_errors = []
        trajectories = []
        for i, (case, gt) in enumerate(zip(self.test_cases, ground_truths)):
            gt_pos1 = np.asarray(gt["pos1"])
            gt_pos2 = np.asarray(gt["pos2"])
            try:
                pred_traj = []
                case_errors = []
                vel = list(case["velocity2"])
                for j, t in enumerate(case["measurement_times"]):
                    p2_out, v2_out = discovered_law(
                        pos1=[0.0, 0.0],
                        pos2=case["pos2"],
                        p1=case["p1"],
                        p2=case["p2"],
                        velocity2=case["velocity2"],
                        duration=t,
                    )
                    p2_out = np.asarray(p2_out)
                    if p2_out.ndim == 2:
                        p2_out = p2_out[-1]
                    pred_traj.append(p2_out.tolist())
                    diff = p2_out - np.asarray(gt_pos2[j])
                    err = float(np.dot(diff, diff))
                    case_errors.append(err)
                mean_err = float(np.mean(case_errors))
                per_case_errors.append(mean_err)
                all_errors.extend(case_errors)
                trajectories.append(
                    {
                        "case": i + 1,
                        "times": case["measurement_times"],
                        "p1": case["p1"],
                        "p2": case["p2"],
                        "gt1": gt_pos1.tolist(),
                        "gt": gt_pos2.tolist(),
                        "pred": pred_traj,
                        "error": mean_err,
                    }
                )
                if verbose:
                    print(f"  Case {i + 1}: mean_particle_mse = {mean_err:.4f}")
            except Exception as e:
                if verbose:
                    print(f"  Case {i + 1}: ERROR — {e}")
                per_case_errors.append(float("inf"))
                all_errors.append(float("inf"))
                trajectories.append(
                    {
                        "case": i + 1,
                        "times": case["measurement_times"],
                        "p1": case["p1"],
                        "p2": case["p2"],
                        "gt1": gt_pos1.tolist(),
                        "gt": gt_pos2.tolist(),
                        "pred": None,
                        "error": float("inf"),
                    }
                )
        mean_total = float(np.mean(all_errors)) if all_errors else float("inf")
        max_total = float(np.max(all_errors)) if all_errors else float("inf")
        passed = mean_total < 0.01
        if verbose:
            print(f"\n  Mean particle MSE: {mean_total:.4f}")
            print(f"  Max  particle MSE: {max_total:.4f}")
            print(f"  Result: {('PASS' if passed else 'FAIL')}")
        return {
            "mean_pos_error": mean_total,
            "max_pos_error": max_total,
            "per_case": per_case_errors,
            "passed": passed,
            "trajectories": trajectories,
            "fit": fit_info,
        }


_CIRCLE_TEST_CASES = [
    {
        "ring_radius": 5.0,
        "initial_tangential_velocity": 0.3,
        "measurement_times": [2.0, 4.0, 6.0, 8.0, 10.0],
    }
]


class CircleEvaluator:
    def __init__(self, executor, test_cases: list[dict] = None):
        self.executor = executor
        self.test_cases = test_cases or _CIRCLE_TEST_CASES

    def evaluate(
        self,
        law_source: str,
        verbose: bool = True,
        training_trajectories: Optional[list] = None,
    ) -> dict:
        discovered_law = _compile_law(law_source)
        discovered_law, fit_info = _maybe_fit(
            law_source, discovered_law, training_trajectories, _circle_loss, verbose
        )
        with self.executor.noise_disabled():
            ground_truths = self.executor.run(self.test_cases)
        per_case_errors = []
        all_errors = []
        trajectories = []
        for i, (case, gt) in enumerate(zip(self.test_cases, ground_truths)):
            gt_positions = np.asarray(gt["positions"])
            gt_velocities = np.asarray(gt["velocities"])
            init_pos = gt_positions[0] if len(gt_positions) > 0 else None
            zero_result = self.executor.run(
                [{**case, "measurement_times": [case["measurement_times"][0]]}]
            )
            ring_radius = float(case.get("ring_radius", 5.0))
            v_tang = float(case.get("initial_tangential_velocity", 0.0))
            angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
            ring_pos = np.column_stack(
                [ring_radius * np.cos(angles), ring_radius * np.sin(angles)]
            )
            init_positions = np.vstack([[[0.0, 0.0]], ring_pos]).tolist()
            ring_vel = np.column_stack(
                [-v_tang * np.sin(angles), v_tang * np.cos(angles)]
            )
            init_velocities = np.vstack([[[0.0, 0.0]], ring_vel]).tolist()
            try:
                pred_traj = []
                case_errors = []
                for j, t in enumerate(case["measurement_times"]):
                    pos_out = discovered_law(
                        positions=init_positions, velocities=init_velocities, duration=t
                    )
                    pos_out = np.asarray(pos_out)
                    pred_traj.append(pos_out.tolist())
                    diff = pos_out - gt_positions[j]
                    errs = np.sum(diff * diff, axis=-1)
                    case_errors.append(float(np.mean(errs)))
                    all_errors.extend(errs.tolist())
                mean_err = float(np.mean(case_errors))
                per_case_errors.append(mean_err)
                trajectories.append(
                    {
                        "case": i + 1,
                        "ring_radius": case["ring_radius"],
                        "v_tang": case["initial_tangential_velocity"],
                        "times": case["measurement_times"],
                        "gt": gt_positions.tolist(),
                        "pred": pred_traj,
                        "error": mean_err,
                    }
                )
                if verbose:
                    print(
                        f"  Case {i + 1} (r={case['ring_radius']}, v_t={case['initial_tangential_velocity']}): mean_particle_mse = {mean_err:.4f}"
                    )
            except Exception as e:
                if verbose:
                    print(f"  Case {i + 1}: ERROR — {e}")
                per_case_errors.append(float("inf"))
                all_errors.append(float("inf"))
                trajectories.append(
                    {
                        "case": i + 1,
                        "ring_radius": case["ring_radius"],
                        "v_tang": case["initial_tangential_velocity"],
                        "times": case["measurement_times"],
                        "gt": gt_positions.tolist() if gt_positions is not None else [],
                        "pred": None,
                        "error": float("inf"),
                    }
                )
        mean_total = float(np.mean(all_errors)) if all_errors else float("inf")
        max_total = float(np.max(all_errors)) if all_errors else float("inf")
        passed = mean_total < 0.25
        if verbose:
            print(f"\n  Mean particle MSE (all particles): {mean_total:.4f}")
            print(f"  Max  particle MSE:                 {max_total:.4f}")
            print(f"  Result: {('PASS' if passed else 'FAIL')}")
        return {
            "mean_pos_error": mean_total,
            "max_pos_error": max_total,
            "per_case": per_case_errors,
            "passed": passed,
            "trajectories": trajectories,
            "fit": fit_info,
        }


_THREE_SPECIES_TEST_CASES = [
    {
        "probe_positions": [[5, 0], [0, 5], [-5, 0], [0, -5], [7, 7]],
        "probe_velocities": [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0],
    },
    {
        "probe_positions": [[3, 3], [-3, 3], [-3, -3], [3, -3], [0, 0]],
        "probe_velocities": [[0.2, 0], [0, 0.2], [-0.2, 0], [0, -0.2], [0, 0]],
        "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0],
    },
]


class ThreeSpeciesEvaluator:
    def __init__(self, executor: ThreeSpeciesExecutor, test_cases: list[dict] = None):
        self.executor = executor
        self.test_cases = test_cases or _THREE_SPECIES_TEST_CASES

    def evaluate(self, law_source: str, verbose: bool = True) -> dict:
        discovered_law = _compile_law(law_source)
        with self.executor.noise_disabled():
            ground_truths = self.executor.run(self.test_cases)
        per_case_errors = []
        all_errors = []
        trajectories = []
        for i, (case, gt) in enumerate(zip(self.test_cases, ground_truths)):
            gt_positions = np.asarray(gt["positions"])
            bg_init = np.asarray(gt["background_initial_positions"])
            probe_pos = np.asarray(case["probe_positions"])
            probe_vel = np.asarray(case["probe_velocities"])
            init_positions = np.vstack([bg_init, probe_pos]).tolist()
            init_velocities = np.vstack(
                [np.zeros((self.executor.N_BACKGROUND, 2)), probe_vel]
            ).tolist()
            try:
                pred_traj = []
                case_errors = []
                for j, t in enumerate(case["measurement_times"]):
                    pos_out = discovered_law(
                        positions=init_positions, velocities=init_velocities, duration=t
                    )
                    pos_out = np.asarray(pos_out)
                    pred_traj.append(pos_out.tolist())
                    diff = pos_out - gt_positions[j]
                    errs = np.sum(diff * diff, axis=-1)
                    case_errors.append(float(np.mean(errs)))
                    all_errors.extend(errs.tolist())
                mean_err = float(np.mean(case_errors))
                per_case_errors.append(mean_err)
                trajectories.append(
                    {
                        "case": i + 1,
                        "times": case["measurement_times"],
                        "gt": gt_positions.tolist(),
                        "pred": pred_traj,
                        "error": mean_err,
                    }
                )
                if verbose:
                    print(f"  Case {i + 1}: mean_particle_mse = {mean_err:.4f}")
            except Exception as e:
                if verbose:
                    print(f"  Case {i + 1}: ERROR -- {e}")
                per_case_errors.append(float("inf"))
                all_errors.append(float("inf"))
                trajectories.append(
                    {
                        "case": i + 1,
                        "times": case["measurement_times"],
                        "gt": gt_positions.tolist(),
                        "pred": None,
                        "error": float("inf"),
                    }
                )
        mean_total = float(np.mean(all_errors)) if all_errors else float("inf")
        max_total = float(np.max(all_errors)) if all_errors else float("inf")
        passed = mean_total < 0.25
        if verbose:
            print(f"\n  Mean particle MSE (all particles): {mean_total:.4f}")
            print(f"  Max  particle MSE:                 {max_total:.4f}")
            print(f"  Result: {('PASS' if passed else 'FAIL')}")
        return {
            "mean_pos_error": mean_total,
            "max_pos_error": max_total,
            "per_case": per_case_errors,
            "passed": passed,
            "trajectories": trajectories,
        }


_DARK_MATTER_TEST_CASES = [
    {
        "probe_positions": [[12, 0], [0, 14], [-11, 0], [0, -13], [10, 10]],
        "probe_velocities": [[0, 2.0], [-2.0, 0], [0, -1.5], [2.0, 0], [-1.5, 1.5]],
        "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    },
    {
        "probe_positions": [[9, 5], [-7, 10], [-10, -6], [6, -11], [0, 15]],
        "probe_velocities": [
            [0.5, -2.0],
            [2.0, 0.5],
            [-0.5, 2.0],
            [-2.0, -0.5],
            [2.5, 0],
        ],
        "visible_velocity_sign": -1.0,
        "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    },
]


class DarkMatterEvaluator:
    def __init__(self, executor: DarkMatterExecutor, test_cases: list[dict] = None):
        self.executor = executor
        self.test_cases = test_cases or _DARK_MATTER_TEST_CASES

    def evaluate(self, law_source: str, verbose: bool = True) -> dict:
        discovered_law = _compile_law(law_source)
        with self.executor.noise_disabled():
            ground_truths = self.executor.run(self.test_cases)
        full_truths = self.executor.run_full(self.test_cases)
        per_case_errors = []
        all_errors = []
        trajectories = []
        n_vis = self.executor.N_VISIBLE
        probe_slice = slice(n_vis, n_vis + self.executor.N_PROBES)
        for i, (case, gt, gt_full) in enumerate(
            zip(self.test_cases, ground_truths, full_truths)
        ):
            gt_positions = np.asarray(gt["positions"])
            bg_init = np.asarray(gt["background_initial_positions"])
            vis_vel_sign = float(case.get("visible_velocity_sign", 1.0))
            vis_vel = vis_vel_sign * self.executor._visible_velocities
            probe_pos = np.asarray(case["probe_positions"])
            probe_vel = np.asarray(case["probe_velocities"])
            init_positions = np.vstack([bg_init, probe_pos]).tolist()
            init_velocities = np.vstack([vis_vel, probe_vel]).tolist()
            try:
                pred_traj = []
                case_errors = []
                for j, t in enumerate(case["measurement_times"]):
                    pos_out = discovered_law(
                        positions=init_positions, velocities=init_velocities, duration=t
                    )
                    pos_out = np.asarray(pos_out)
                    pred_traj.append(pos_out.tolist())
                    diff = pos_out[probe_slice] - gt_positions[j, probe_slice]
                    errs = np.sum(diff * diff, axis=-1)
                    case_errors.append(float(np.mean(errs)))
                    all_errors.extend(errs.tolist())
                mean_err = float(np.mean(case_errors))
                per_case_errors.append(mean_err)
                trajectories.append(
                    {
                        "case": i + 1,
                        "times": case["measurement_times"],
                        "gt": gt_positions.tolist(),
                        "gt_full": gt_full["positions"],
                        "field_snapshots": gt_full["field_snapshots"],
                        "dark_initial": gt_full["dark_initial_positions"],
                        "pred": pred_traj,
                        "error": mean_err,
                    }
                )
                if verbose:
                    print(f"  Case {i + 1}: mean_probe_mse = {mean_err:.4f}")
            except Exception as e:
                if verbose:
                    print(f"  Case {i + 1}: ERROR -- {e}")
                per_case_errors.append(float("inf"))
                all_errors.append(float("inf"))
                trajectories.append(
                    {
                        "case": i + 1,
                        "times": case["measurement_times"],
                        "gt": gt_positions.tolist(),
                        "gt_full": gt_full["positions"],
                        "field_snapshots": gt_full["field_snapshots"],
                        "dark_initial": gt_full["dark_initial_positions"],
                        "pred": None,
                        "error": float("inf"),
                    }
                )
        mean_total = float(np.mean(all_errors)) if all_errors else float("inf")
        max_total = float(np.max(all_errors)) if all_errors else float("inf")
        passed = mean_total < 0.25
        if verbose:
            print(f"\n  Mean probe MSE (probes only): {mean_total:.4f}")
            print(f"  Max  probe MSE:               {max_total:.4f}")
            print(f"  Result: {('PASS' if passed else 'FAIL')}")
        return {
            "mean_pos_error": mean_total,
            "max_pos_error": max_total,
            "per_case": per_case_errors,
            "passed": passed,
            "trajectories": trajectories,
        }


_ETHER_TEST_CASES = [
    {
        "probe_positions": [[15, 0], [0, 18], [-15, 0], [0, -16], [12, 12]],
        "probe_velocities": [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        "probe_masses": [1.0, 2.0, 4.0, 1.0, 2.0],
        "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    },
    {
        "probe_positions": [[10, 0], [0, 10], [-10, 0], [0, -10], [9, 9]],
        "probe_velocities": [[0, 2.8], [-2.8, 0], [0, -2.8], [2.8, 0], [-2.0, 2.0]],
        "probe_masses": [1.0, 1.0, 1.0, 1.0, 1.0],
        "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    },
]


class EtherEvaluator:
    PROBE_SLICE = slice(21, 26)

    def __init__(self, executor: NBodyEtherExecutor, test_cases: list[dict] = None):
        self.executor = executor
        self.test_cases = test_cases or _ETHER_TEST_CASES

    def evaluate(
        self,
        law_source: str,
        verbose: bool = True,
        training_trajectories: Optional[list] = None,
        **kwargs,
    ) -> dict:
        discovered_law = _compile_law(law_source)
        discovered_law, fit_info = _maybe_fit(
            law_source, discovered_law, training_trajectories, _ether_loss, verbose
        )
        with self.executor.noise_disabled():
            ground_truths = self.executor.run(self.test_cases)
        per_case_errors = []
        all_errors = []
        trajectories = []
        bg_pos = np.asarray(self.executor._bg_positions_rel)
        bg_vel = np.asarray(self.executor._bg_velocities)
        bg_mass = np.asarray(self.executor._bg_masses)
        for i, (case, gt) in enumerate(zip(self.test_cases, ground_truths)):
            gt_positions = np.asarray(gt["positions"])
            probe_pos = np.asarray(case["probe_positions"])
            probe_vel = np.asarray(case["probe_velocities"])
            probe_mass = np.asarray(
                case.get(
                    "probe_masses",
                    [self.executor.DEFAULT_PROBE_MASS] * self.executor.N_PROBES,
                ),
                dtype=float,
            )
            init_positions = np.vstack([bg_pos, probe_pos]).tolist()
            init_velocities = np.vstack([bg_vel, probe_vel]).tolist()
            init_masses = np.concatenate([bg_mass, probe_mass]).tolist()
            try:
                pred_traj = []
                case_errors = []
                for j, t in enumerate(case["measurement_times"]):
                    pos_out = discovered_law(
                        positions=init_positions,
                        velocities=init_velocities,
                        masses=init_masses,
                        duration=float(t),
                    )
                    pos_out = np.asarray(pos_out)
                    pred_traj.append(pos_out.tolist())
                    diff = pos_out[self.PROBE_SLICE] - gt_positions[j, self.PROBE_SLICE]
                    errs = np.sum(diff * diff, axis=-1)
                    case_errors.append(float(np.mean(errs)))
                    all_errors.extend(errs.tolist())
                mean_err = float(np.mean(case_errors))
                per_case_errors.append(mean_err)
                trajectories.append(
                    {
                        "case": i + 1,
                        "times": case["measurement_times"],
                        "gt": gt_positions.tolist(),
                        "pred": pred_traj,
                        "error": mean_err,
                    }
                )
                if verbose:
                    print(f"  Case {i + 1}: mean_probe_mse = {mean_err:.4f}")
            except Exception as e:
                if verbose:
                    print(f"  Case {i + 1}: ERROR -- {e}")
                per_case_errors.append(float("inf"))
                all_errors.append(float("inf"))
                trajectories.append(
                    {
                        "case": i + 1,
                        "times": case["measurement_times"],
                        "gt": gt_positions.tolist(),
                        "pred": None,
                        "error": float("inf"),
                    }
                )
        mean_total = float(np.mean(all_errors)) if all_errors else float("inf")
        max_total = float(np.max(all_errors)) if all_errors else float("inf")
        passed = mean_total < 0.25
        if verbose:
            print(f"\n  Mean probe MSE (probes only): {mean_total:.4f}")
            print(f"  Max  probe MSE:               {max_total:.4f}")
            print(f"  Result: {('PASS' if passed else 'FAIL')}")
        return {
            "mean_pos_error": mean_total,
            "max_pos_error": max_total,
            "per_case": per_case_errors,
            "passed": passed,
            "trajectories": trajectories,
            "fit": fit_info,
        }


_HUBBLE_TEST_CASES = [
    {
        "probe_positions": [[6, 0], [0, 10], [15, 0], [-15, 0], [0, 18]],
        "probe_velocities": [[0, 2.48], [-1.72, 0], [0, 0], [0, 0], [0, 0]],
        "probe_masses": [1.0, 2.0, 4.0, 1.0, 2.0],
        "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    },
    {
        "probe_positions": [[10, 0], [0, 10], [-10, 0], [0, -10], [16, 0]],
        "probe_velocities": [[0, 1.72], [-1.72, 0], [0, -1.72], [1.72, 0], [0, 0]],
        "probe_masses": [1.0, 1.0, 1.0, 1.0, 1.0],
        "measurement_times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    },
]


class HubbleEvaluator:
    PROBE_SLICE = slice(21, 26)

    def __init__(self, executor, test_cases: list[dict] = None):
        self.executor = executor
        self.test_cases = test_cases or _HUBBLE_TEST_CASES

    def evaluate(
        self,
        law_source: str,
        verbose: bool = True,
        training_trajectories: Optional[list] = None,
        **kwargs,
    ) -> dict:
        discovered_law = _compile_law(law_source)
        discovered_law, fit_info = _maybe_fit(
            law_source, discovered_law, training_trajectories, _ether_loss, verbose
        )
        with self.executor.noise_disabled():
            ground_truths = self.executor.run(self.test_cases)
        per_case_errors = []
        all_errors = []
        trajectories = []
        bg_pos = np.asarray(self.executor._bg_positions_rel)
        bg_vel = np.asarray(self.executor._bg_velocities)
        bg_mass = np.asarray(self.executor._bg_masses)
        for i, (case, gt) in enumerate(zip(self.test_cases, ground_truths)):
            gt_positions = np.asarray(gt["positions"])
            probe_pos = np.asarray(case["probe_positions"])
            probe_vel = np.asarray(case["probe_velocities"])
            probe_mass = np.asarray(
                case.get(
                    "probe_masses",
                    [self.executor.DEFAULT_PROBE_MASS] * self.executor.N_PROBES,
                ),
                dtype=float,
            )
            init_positions = np.vstack([bg_pos, probe_pos]).tolist()
            init_velocities = np.vstack([bg_vel, probe_vel]).tolist()
            init_masses = np.concatenate([bg_mass, probe_mass]).tolist()
            try:
                pred_traj = []
                case_errors = []
                for j, t in enumerate(case["measurement_times"]):
                    pos_out = discovered_law(
                        positions=init_positions,
                        velocities=init_velocities,
                        masses=init_masses,
                        duration=float(t),
                    )
                    pos_out = np.asarray(pos_out)
                    pred_traj.append(pos_out.tolist())
                    diff = pos_out[self.PROBE_SLICE] - gt_positions[j, self.PROBE_SLICE]
                    errs = np.sum(diff * diff, axis=-1)
                    case_errors.append(float(np.mean(errs)))
                    all_errors.extend(errs.tolist())
                mean_err = float(np.mean(case_errors))
                per_case_errors.append(mean_err)
                trajectories.append(
                    {
                        "case": i + 1,
                        "times": case["measurement_times"],
                        "gt": gt_positions.tolist(),
                        "pred": pred_traj,
                        "error": mean_err,
                    }
                )
                if verbose:
                    print(f"  Case {i + 1}: mean_probe_mse = {mean_err:.4f}")
            except Exception as e:
                if verbose:
                    print(f"  Case {i + 1}: ERROR -- {e}")
                per_case_errors.append(float("inf"))
                all_errors.append(float("inf"))
                trajectories.append(
                    {
                        "case": i + 1,
                        "times": case["measurement_times"],
                        "gt": gt_positions.tolist(),
                        "pred": None,
                        "error": float("inf"),
                    }
                )
        mean_total = float(np.mean(all_errors)) if all_errors else float("inf")
        max_total = float(np.max(all_errors)) if all_errors else float("inf")
        passed = mean_total < 1.0
        if verbose:
            print(f"\n  Mean probe MSE (probes only): {mean_total:.4f}")
            print(f"  Max  probe MSE:               {max_total:.4f}")
            print(f"  Result: {('PASS' if passed else 'FAIL')}")
        return {
            "mean_pos_error": mean_total,
            "max_pos_error": max_total,
            "per_case": per_case_errors,
            "passed": passed,
            "trajectories": trajectories,
            "fit": fit_info,
        }


def clean_law_source(source: str) -> str:
    import re as _re

    source = _re.sub("^```[a-zA-Z]*\\n?", "", source.strip(), flags=_re.MULTILINE)
    source = source.replace("```", "")
    lines = source.splitlines()
    code_start = next(
        (
            i
            for i, l in enumerate(lines)
            if l.startswith("def ") or l.startswith("import ") or l.startswith("from ")
        ),
        0,
    )
    return "\n".join(lines[code_start:])


def _compile_law(source: str) -> Callable:
    source = clean_law_source(source)
    namespace = {}
    exec(compile(source, "<discovered_law>", "exec"), namespace)
    if "discovered_law" not in namespace:
        raise ValueError("Source does not define a function named `discovered_law`")
    return _wrap_with_timeout(namespace["discovered_law"])


def _compile_fit_parameters(source: str) -> Optional[Callable]:
    source = clean_law_source(source)
    namespace = {}
    try:
        exec(compile(source, "<fit_parameters>", "exec"), namespace)
    except Exception:
        return None
    return namespace.get("fit_parameters")


def _extract_training_trajectories(conversation_log: list) -> list:
    training = []
    if not conversation_log:
        return training
    for entry in conversation_log:
        action = entry.get("action")
        if action in ("experiment", "scm_experiment"):
            inputs = entry.get("experiment_input")
            outputs = entry.get("experiment_output")
            if not isinstance(inputs, list) or not isinstance(outputs, list):
                continue
            for inp, out in zip(inputs, outputs):
                if not isinstance(inp, dict) or not isinstance(out, dict):
                    continue
                training.append({"input": inp, "output": out})
            continue
        if action != "counterfactual":
            continue
        paired = entry.get("experiment_output")
        if not isinstance(paired, dict):
            continue
        factual = paired.get("factual")
        if isinstance(factual, dict):
            factual_input = factual.get("input")
            factual_outputs = factual.get("output")
            if isinstance(factual_input, dict) and isinstance(factual_outputs, list):
                for factual_output in factual_outputs:
                    if isinstance(factual_output, dict):
                        training.append(
                            {"input": factual_input, "output": factual_output}
                        )
        for branch in paired.get("counterfactuals", []):
            if not isinstance(branch, dict):
                continue
            branch_input = branch.get("input")
            branch_outputs = branch.get("output")
            if not isinstance(branch_input, dict) or not isinstance(
                branch_outputs, list
            ):
                continue
            for branch_output in branch_outputs:
                if isinstance(branch_output, dict):
                    training.append(
                        {
                            "input": branch_input,
                            "output": branch_output,
                            "_paired_factual": {
                                "input": factual.get("input"),
                                "output": factual_outputs[0]
                                if isinstance(factual_outputs, list)
                                and factual_outputs
                                and isinstance(factual_outputs[0], dict)
                                else None,
                            },
                            "_paired_branch_id": branch.get("id"),
                        }
                    )
    return training


def _subsample_training(
    training: list,
    max_trajectories: int = FIT_MAX_TRAJECTORIES,
    max_times: int = FIT_MAX_TIMES_PER_TRAJ,
) -> list:
    if not training:
        return training
    paired_training = [
        sample
        for sample in training
        if isinstance(sample.get("_paired_factual"), dict)
        and isinstance(sample["_paired_factual"].get("input"), dict)
        and isinstance(sample["_paired_factual"].get("output"), dict)
    ]
    if paired_training:
        training = paired_training
    n_traj = len(training)
    if n_traj > max_trajectories:
        idx = np.linspace(0, n_traj - 1, max_trajectories).round().astype(int).tolist()
        training = [training[i] for i in idx]
    pruned = []
    for sample in training:
        in_dict = sample.get("input", {}) or {}
        out_in = sample.get("output", {}) or {}
        times = list(
            out_in.get("measurement_times", in_dict.get("measurement_times", []))
        )
        if not times or len(times) <= max_times:
            pruned.append(sample)
            continue
        sel = sorted(
            set(np.linspace(0, len(times) - 1, max_times).round().astype(int).tolist())
        )
        out = dict(out_in)
        out["measurement_times"] = [times[i] for i in sel]
        for key in (
            "positions",
            "pos1",
            "pos2",
            "velocities",
            "velocity1",
            "velocity2",
        ):
            arr = out.get(key)
            if arr is None or len(arr) != len(times):
                continue
            out[key] = [arr[i] for i in sel]
        pruned_sample = {**sample, "input": in_dict, "output": out}
        paired = sample.get("_paired_factual")
        if isinstance(paired, dict) and isinstance(paired.get("output"), dict):
            paired_output_in = paired["output"]
            paired_times = list(
                paired_output_in.get(
                    "measurement_times",
                    (paired.get("input") or {}).get("measurement_times", []),
                )
            )
            if len(paired_times) == len(times):
                paired_output = dict(paired_output_in)
                paired_output["measurement_times"] = [
                    paired_times[index] for index in sel
                ]
                for key in (
                    "positions",
                    "pos1",
                    "pos2",
                    "velocities",
                    "velocity1",
                    "velocity2",
                ):
                    arr = paired_output.get(key)
                    if arr is not None and len(arr) == len(paired_times):
                        paired_output[key] = [arr[index] for index in sel]
                pruned_sample["_paired_factual"] = {**paired, "output": paired_output}
        pruned.append(pruned_sample)
    return pruned


def _call_two_particle_law(law: Callable, case: dict, duration: float):
    kwargs = {
        "pos1": [0.0, 0.0],
        "pos2": case["pos2"],
        "p1": case["p1"],
        "p2": case["p2"],
        "velocity2": case["velocity2"],
        "duration": float(duration),
    }
    if "start_time" in case:
        try:
            parameters = inspect.signature(law).parameters.values()
            accepts_start_time = any(
                (
                    parameter.name == "start_time"
                    or parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters
                )
            )
        except (TypeError, ValueError):
            accepts_start_time = False
        if accepts_start_time:
            kwargs["start_time"] = float(case["start_time"])
    return law(**kwargs)


def _two_particle_loss(law: Callable, training: list) -> float:
    total_sq = 0.0
    count = 0
    for sample in training:
        case = sample["input"]
        out = sample["output"]
        obs_pos2 = np.asarray(out.get("pos2", []))
        times = out.get("measurement_times", case.get("measurement_times", []))
        if obs_pos2.ndim != 2 or len(times) == 0:
            continue
        paired = sample.get("_paired_factual")
        if isinstance(paired, dict):
            factual_case = paired.get("input")
            factual_out = paired.get("output")
            if isinstance(factual_case, dict) and isinstance(factual_out, dict):
                factual_pos2 = np.asarray(factual_out.get("pos2", []))
                factual_times = factual_out.get(
                    "measurement_times", factual_case.get("measurement_times", [])
                )
                if factual_pos2.ndim == 2 and len(factual_times):
                    factual_by_time = {
                        float(time_value): factual_pos2[index]
                        for index, time_value in enumerate(factual_times)
                    }
                    for t, branch_obs in zip(times, obs_pos2):
                        factual_obs = factual_by_time.get(float(t))
                        if factual_obs is None:
                            continue
                        branch_pred, _ = _call_two_particle_law(law, case, float(t))
                        factual_pred, _ = _call_two_particle_law(
                            law, factual_case, float(t)
                        )
                        branch_pred = np.asarray(branch_pred)
                        factual_pred = np.asarray(factual_pred)
                        if branch_pred.ndim == 2:
                            branch_pred = branch_pred[-1]
                        if factual_pred.ndim == 2:
                            factual_pred = factual_pred[-1]
                        predicted_effect = branch_pred - factual_pred
                        observed_effect = np.asarray(branch_obs) - np.asarray(
                            factual_obs
                        )
                        diff = predicted_effect - observed_effect
                        branch_pos0 = np.asarray(case["pos2"], dtype=float)
                        factual_pos0 = np.asarray(factual_case["pos2"], dtype=float)
                        branch_vel0 = np.asarray(case["velocity2"], dtype=float)
                        factual_vel0 = np.asarray(
                            factual_case["velocity2"], dtype=float
                        )
                        inertial_effect = (
                            branch_pos0
                            - factual_pos0
                            + (branch_vel0 - factual_vel0) * float(t)
                        )
                        causal_effect = observed_effect - inertial_effect
                        effect_scale = max(
                            float(np.dot(causal_effect, causal_effect)), 1e-08
                        )
                        total_sq += float(np.dot(diff, diff)) / effect_scale
                        count += 1
                    continue
        for t, obs in zip(times, obs_pos2):
            pred, _ = _call_two_particle_law(law, case, float(t))
            pred = np.asarray(pred)
            if pred.ndim == 2:
                pred = pred[-1]
            diff = pred - np.asarray(obs)
            total_sq += float(np.dot(diff, diff))
            count += 1
    if count == 0:
        return float("inf")
    return total_sq / count


def _circle_loss(law: Callable, training: list) -> float:
    total_sq = 0.0
    count = 0
    for sample in training:
        case = sample["input"]
        out = sample["output"]
        obs_positions = np.asarray(out.get("positions", []))
        times = out.get("measurement_times", case.get("measurement_times", []))
        if obs_positions.ndim != 3 or len(times) == 0:
            continue
        ring_radius = float(case.get("ring_radius", 5.0))
        v_tang = float(case.get("initial_tangential_velocity", 0.0))
        angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        ring_pos = np.column_stack(
            [ring_radius * np.cos(angles), ring_radius * np.sin(angles)]
        )
        init_positions = np.vstack([[[0.0, 0.0]], ring_pos]).tolist()
        ring_vel = np.column_stack([-v_tang * np.sin(angles), v_tang * np.cos(angles)])
        init_velocities = np.vstack([[[0.0, 0.0]], ring_vel]).tolist()
        for t, obs in zip(times, obs_positions):
            pred = law(
                positions=init_positions, velocities=init_velocities, duration=float(t)
            )
            pred = np.asarray(pred)
            if pred.shape != obs.shape:
                return float("inf")
            diff = pred - obs
            total_sq += float(np.sum(diff * diff))
            count += diff.size // 2
    if count == 0:
        return float("inf")
    return total_sq / count


def _three_species_loss(law: Callable, training: list) -> float:
    total_sq = 0.0
    count = 0
    for sample in training:
        case = sample["input"]
        out = sample["output"]
        init_positions = case.get("init_positions")
        init_velocities = case.get("init_velocities")
        times = out.get("measurement_times", case.get("measurement_times", []))
        obs_positions = np.asarray(out.get("positions", []))
        if init_positions is None or init_velocities is None:
            continue
        if obs_positions.ndim != 3 or len(times) == 0:
            continue
        for j, t in enumerate(times):
            try:
                pred = np.asarray(
                    law(
                        positions=init_positions,
                        velocities=init_velocities,
                        duration=float(t),
                    )
                )
            except Exception:
                return float("inf")
            if pred.shape != obs_positions[j].shape:
                return float("inf")
            diff = pred - obs_positions[j]
            total_sq += float(np.sum(diff * diff))
            count += diff.size // 2
    if count == 0:
        return float("inf")
    return total_sq / count


_DARK_MATTER_PROBE_SLICE = slice(20, 25)


def _dark_matter_loss(law: Callable, training: list) -> float:
    total_sq = 0.0
    count = 0
    for sample in training:
        case = sample["input"]
        out = sample["output"]
        init_positions = case.get("init_positions")
        init_velocities = case.get("init_velocities")
        times = out.get("measurement_times", case.get("measurement_times", []))
        obs_positions = np.asarray(out.get("positions", []))
        if init_positions is None or init_velocities is None:
            continue
        if obs_positions.ndim != 3 or len(times) == 0:
            continue
        for j, t in enumerate(times):
            try:
                pred = np.asarray(
                    law(
                        positions=init_positions,
                        velocities=init_velocities,
                        duration=float(t),
                    )
                )
            except Exception:
                return float("inf")
            if pred.shape != obs_positions[j].shape:
                return float("inf")
            diff = (
                pred[_DARK_MATTER_PROBE_SLICE]
                - obs_positions[j, _DARK_MATTER_PROBE_SLICE]
            )
            total_sq += float(np.sum(diff * diff))
            count += diff.size // 2
    if count == 0:
        return float("inf")
    return total_sq / count


_ETHER_PROBE_SLICE = slice(21, 26)


def _ether_loss(law: Callable, training: list) -> float:
    total_sq = 0.0
    count = 0
    for sample in training:
        case = sample["input"]
        out = sample["output"]
        times = out.get("measurement_times", case.get("measurement_times", []))
        obs_positions = np.asarray(out.get("positions", []))
        if obs_positions.ndim != 3 or len(times) == 0:
            continue
        init_positions = case.get("init_positions")
        init_velocities = case.get("init_velocities")
        init_masses = case.get("init_masses")
        if init_positions is None or init_velocities is None or init_masses is None:
            bg_pos = out.get("background_initial_positions")
            bg_vel = out.get("background_initial_velocities")
            particle_masses = out.get("particle_masses")
            if bg_pos is None or bg_vel is None or particle_masses is None:
                continue
            probe_pos = case.get("probe_positions")
            probe_vel = case.get("probe_velocities")
            if probe_pos is None or probe_vel is None:
                continue
            bg_pos = np.asarray(bg_pos, dtype=float)
            bg_vel = np.asarray(bg_vel, dtype=float)
            init_positions = np.vstack(
                [bg_pos, np.asarray(probe_pos, dtype=float)]
            ).tolist()
            init_velocities = np.vstack(
                [bg_vel, np.asarray(probe_vel, dtype=float)]
            ).tolist()
            init_masses = list(particle_masses)
        for j, t in enumerate(times):
            try:
                pred = np.asarray(
                    law(
                        positions=init_positions,
                        velocities=init_velocities,
                        masses=init_masses,
                        duration=float(t),
                    )
                )
            except Exception:
                return float("inf")
            if pred.shape != obs_positions[j].shape:
                return float("inf")
            diff = pred[_ETHER_PROBE_SLICE] - obs_positions[j, _ETHER_PROBE_SLICE]
            total_sq += float(np.sum(diff * diff))
            count += diff.size // 2
    if count == 0:
        return float("inf")
    return total_sq / count


def _validate_fit_spec(spec) -> list:
    if not isinstance(spec, dict):
        raise ValueError("fit_parameters() must return a dict")
    if len(spec) > MAX_FIT_PARAMETERS:
        raise ValueError(
            f"fit_parameters() declares {len(spec)} parameters; max allowed is {MAX_FIT_PARAMETERS}"
        )
    out = []
    for name, entry in spec.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"fit_parameters()['{name}'] must be a dict with 'init' and 'bounds'"
            )
        if "init" not in entry or "bounds" not in entry:
            raise ValueError(
                f"fit_parameters()['{name}'] must provide both 'init' and 'bounds'"
            )
        bounds = entry["bounds"]
        if not (isinstance(bounds, (list, tuple)) and len(bounds) == 2):
            raise ValueError(
                f"fit_parameters()['{name}']['bounds'] must be a 2-element sequence"
            )
        lo, hi = (float(bounds[0]), float(bounds[1]))
        if not lo < hi:
            raise ValueError(
                f"fit_parameters()['{name}']: lower bound must be below upper bound"
            )
        init = float(entry["init"])
        if not lo <= init <= hi:
            init = min(max(init, lo), hi)
        out.append((name, init, (lo, hi)))
    return out


def _fit_law_parameters(
    discovered_law: Callable,
    fit_spec_list: list,
    training: list,
    loss_fn: Callable,
    maxiter: int = FIT_MAXITER,
    time_budget_s: float = FIT_TIME_BUDGET_S,
) -> dict:
    if not fit_spec_list:
        return {}
    import time
    from scipy.optimize import minimize

    names = [s[0] for s in fit_spec_list]
    x0 = [s[1] for s in fit_spec_list]
    bounds = [s[2] for s in fit_spec_list]
    state = {
        "best_x": list(x0),
        "best_loss": float("inf"),
        "deadline": time.monotonic() + time_budget_s,
    }

    def _objective(x):
        if time.monotonic() > state["deadline"]:
            raise _FitTimeBudgetExceeded()
        kwargs = dict(zip(names, x.tolist()))
        bound_law = functools.partial(discovered_law, **kwargs)
        try:
            loss = loss_fn(bound_law, training)
        except Exception:
            return 1000000000000.0
        if loss < state["best_loss"]:
            state["best_loss"] = loss
            state["best_x"] = list(x)
        return loss

    try:
        result = minimize(
            _objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter},
        )
        return dict(zip(names, result.x.tolist()))
    except _FitTimeBudgetExceeded:
        print(
            f"  [fit time budget {time_budget_s:.0f}s exceeded; using best-so-far params (loss={state['best_loss']:.4g})]"
        )
        return dict(zip(names, state["best_x"]))


def _maybe_fit(
    law_source: str,
    discovered_law: Callable,
    training_trajectories: Optional[list],
    loss_fn: Callable,
    verbose: bool,
) -> tuple:
    fit_fn = _compile_fit_parameters(law_source)
    if fit_fn is None:
        return (discovered_law, None)
    if not training_trajectories:
        if verbose:
            print("  [fit skipped: no training trajectories available]")
        return (discovered_law, {"error": "no_training_trajectories"})
    try:
        raw_spec = fit_fn()
        fit_spec_list = _validate_fit_spec(raw_spec)
    except Exception as e:
        if verbose:
            print(f"  [fit skipped: invalid fit_parameters() — {e}]")
        return (discovered_law, {"error": f"invalid_spec: {e}"})
    if not fit_spec_list:
        return (discovered_law, None)
    declared = {
        name: {"init": init, "bounds": list(bounds)}
        for name, init, bounds in fit_spec_list
    }
    training = _subsample_training(training_trajectories)
    if verbose and len(training) < len(training_trajectories):
        print(
            f"  [fit using {len(training)}/{len(training_trajectories)} training trajectories (cap: {FIT_MAX_TRAJECTORIES} traj × {FIT_MAX_TIMES_PER_TRAJ} times)]"
        )
    init_kwargs = {name: init for name, init, _ in fit_spec_list}
    init_law = functools.partial(discovered_law, **init_kwargs)
    try:
        loss_before = loss_fn(init_law, training)
    except Exception:
        loss_before = float("inf")
    fit_t0 = time.monotonic()
    try:
        fitted = _fit_law_parameters(discovered_law, fit_spec_list, training, loss_fn)
    except Exception as e:
        if verbose:
            print(f"  [fit failed: {e}; falling back to init values]")
        return (
            functools.partial(discovered_law, **init_kwargs),
            {
                "declared_params": declared,
                "fitted_params": init_kwargs,
                "loss_before": loss_before,
                "loss_after": loss_before,
                "error": f"optimizer_failure: {e}",
            },
        )
    fit_elapsed = time.monotonic() - fit_t0
    if verbose and fit_elapsed >= FIT_TIME_BUDGET_S - 1.0:
        print(
            f"  [fit hit {FIT_TIME_BUDGET_S:.0f}s wall-clock budget; using best-so-far parameters]"
        )
    bound_law = functools.partial(discovered_law, **fitted)
    try:
        loss_after = loss_fn(bound_law, training)
    except Exception:
        loss_after = float("inf")
    if verbose:
        pretty = ", ".join((f"{k}={v:.4g}" for k, v in fitted.items()))
        print(f"  Fitted parameters: {pretty}")
        print(f"  Training-set loss: {loss_before:.4g} → {loss_after:.4g}")
    return (
        bound_law,
        {
            "declared_params": declared,
            "fitted_params": fitted,
            "loss_before": float(loss_before),
            "loss_after": float(loss_after),
            "error": None,
        },
    )


_JUDGE_SYSTEM_PROMPT = "You are an expert physicist grading how well a student's prose description of a simulated physical system matches the ground-truth description. You are precise, fair, and reward semantic correctness over surface phrasing — paraphrases and equivalent formulations (e.g. 'inverse-square-like' ≈ '∇²φ' in 2D) should receive credit, but missing or wrong physical content should not."
_GENERIC_SCORING_GUIDE = "10 — captures every essential element correctly, with correct quantitative or relational claims where applicable.\n 7–9 — captures the operator and qualitative structure but misses or muddles a quantitative detail or one structural feature.\n 4–6 — partially correct: identifies the general physics regime but misses key structural features (e.g. fails to identify multiple species).\n 1–3 — incorrect operator or fundamentally wrong physical picture, with only superficial correctness.\n   0 — empty, irrelevant, or completely wrong."
_JUDGE_USER_TEMPLATE = 'Compare the student\'s description against the ground-truth description of the physical system.\n\n<ground_truth>\n{ground_truth}\n</ground_truth>\n\n<student>\n{student}\n</student>\n\nScore the student description on a 0–10 integer scale based on how well it captures:\n  1. The correct field equation / governing operator (e.g. Laplacian, fractional Laplacian, Helmholtz, diffusion, wave).\n  2. The temporal character (static vs. time-evolving; instantaneous vs. retarded).\n  3. The force law / coupling structure (how particles couple to the field, including p1/p2 roles).\n  4. Any structural features unique to this world: hidden species and their relative coupling strengths and signs, neutral probes, hidden/dark sources, screening lengths, etc.\n\nUse the world-specific rubric below to calibrate the bands. A 10/10 represents the best explanation achievable given the experimental capabilities — reward semantically-equivalent phrasings and numeric estimates within the tolerance specified by the rubric.\n\n<scoring_rubric>\n{rubric}\n</scoring_rubric>\n\nRespond with 1–3 sentences of justification, then your final integer score inside <score>...</score> tags. Example: "<score>7</score>".'


class ExplanationJudge:
    def __init__(self, judge_model: str = "claude-opus-4-6", max_tokens: int = 1024):
        self.judge_model = judge_model
        self.max_tokens = max_tokens

    def score(
        self,
        agent_explanation: Optional[str],
        optimal_explanation: str,
        rubric: Optional[str] = None,
        verbose: bool = True,
    ) -> dict:
        if not agent_explanation or not agent_explanation.strip():
            result = {
                "score": 0.0,
                "raw_score": 0,
                "reasoning": "No <explanation> tag was submitted by the agent.",
                "error": None,
            }
            if verbose:
                print(f"  Explanation score: 0.00  (no explanation submitted)")
            return result
        if not optimal_explanation:
            result = {
                "score": None,
                "raw_score": None,
                "reasoning": "No optimal_explanation defined for this world.",
                "error": "missing_ground_truth",
            }
            if verbose:
                print("  Explanation score: skipped (no ground truth defined)")
            return result
        prompt = _JUDGE_USER_TEMPLATE.format(
            ground_truth=optimal_explanation.strip(),
            student=agent_explanation.strip(),
            rubric=rubric.strip()
            if rubric and rubric.strip()
            else _GENERIC_SCORING_GUIDE,
        )
        try:
            from scienceagent import llm_client

            reply = llm_client.complete(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                system=_JUDGE_SYSTEM_PROMPT,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            result = {
                "score": None,
                "raw_score": None,
                "reasoning": "",
                "error": f"Judge call failed: {type(e).__name__}",
            }
            if verbose:
                print(f"  Explanation score: ERROR — {type(e).__name__}")
            return result
        raw_score = _parse_judge_score(reply)
        if raw_score is None:
            result = {
                "score": None,
                "raw_score": None,
                "reasoning": reply,
                "error": "Could not parse <score> tag from judge reply.",
            }
            if verbose:
                print("  Explanation score: ERROR — unparseable judge reply")
            return result
        score = max(0.0, min(1.0, raw_score / 10.0))
        result = {
            "score": float(score),
            "raw_score": int(raw_score),
            "reasoning": reply,
            "error": None,
        }
        if verbose:
            print(f"  Explanation score: {score:.2f}  (raw {raw_score}/10)")
            print(f"  Judge reasoning: {reply.strip()}")
        return result


def _parse_judge_score(reply: str) -> Optional[float]:
    if not reply:
        return None
    patterns = (
        "<score>\\s*(\\d+(?:\\.\\d+)?)\\s*</score>",
        "<score>\\s*(\\d+(?:\\.\\d+)?)\\s*/\\s*10\\s*</score>",
        "(?:final\\s+)?score\\s*:?\\s*\\*{0,2}\\s*(\\d+(?:\\.\\d+)?)(?:\\s*/\\s*10)?\\b",
    )
    for pat in patterns:
        match = re.search(pat, reply, re.IGNORECASE)
        if not match:
            continue
        try:
            val = float(match.group(1))
        except ValueError:
            continue
        if 0.0 <= val <= 10.0:
            return val
    return None
