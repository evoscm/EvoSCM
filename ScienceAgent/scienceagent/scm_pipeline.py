from __future__ import annotations
import copy
import hashlib
import inspect
import json
import math
import re
import textwrap
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional
import numpy as np
from scipy.special import k1 as bessel_k1

SCM_SCHEMA_VERSION = "1.24"
_MAX_TIME_FREQUENCY_ALIASES = 5
_FINAL_CLAIM_FIELDS = (
    "selected_candidate_id",
    "remaining_alternatives",
    "identified_mechanism",
    "operator_or_symmetry",
    "source_response_roles",
    "scalar_magnitude_law",
    "vector_law",
    "time_and_scale_regimes",
    "remaining_uncertainty",
    "evidence_summary",
    "noise_cancelled_scaling_check",
    "operator_correspondence_check",
    "evidence_consistency_check",
    "numerical_robustness_check",
)
_TAG_RE_TEMPLATE = "<{tag}>\\s*(.*?)\\s*</{tag}>"
_MISSING = object()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _extract_tag(text: str, tag: str) -> Optional[str]:
    match = re.search(
        _TAG_RE_TEMPLATE.format(tag=re.escape(tag)),
        text or "",
        flags=re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def _json_tag(text: str, tag: str) -> tuple[Any, Optional[str]]:
    raw = _extract_tag(text, tag)
    if raw is None:
        return (None, None)
    try:
        return (json.loads(raw), None)
    except json.JSONDecodeError as exc:
        return (None, f"<{tag}> is not valid JSON: {exc}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and (not math.isfinite(value)):
        return None
    return value


def _fit_compactified_image_sum(responses: list[dict]) -> Optional[dict]:
    by_radius: dict[float, list[float]] = {}
    for record in responses:
        try:
            p1 = float(record["p1"])
            p2 = float(record["p2"])
            radius = float(record["radius"])
            inward = abs(float(record["inward_acceleration"]))
        except (KeyError, TypeError, ValueError):
            continue
        if (
            not all((math.isfinite(value) for value in (p1, p2, radius, inward)))
            or abs(p1) <= 1e-12
            or abs(p2) <= 1e-12
            or (radius <= 0.0)
            or (inward <= 1e-12)
        ):
            continue
        normalized = inward * abs(p2 / p1)
        by_radius.setdefault(round(radius, 12), []).append(normalized)
    if len(by_radius) < 4:
        return None
    ordered = sorted(
        ((radius, float(np.median(values))) for radius, values in by_radius.items())
    )
    radii = np.asarray([item[0] for item in ordered], dtype=float)
    magnitudes = np.asarray([item[1] for item in ordered], dtype=float)
    if radii[-1] / radii[0] < 4.0:
        return None
    image_indices = np.arange(-128, 129, dtype=float)[None, None, :]
    radii_3d = radii[None, :, None]

    def evaluate_grid(radius_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        circumference = (2.0 * math.pi * radius_grid)[:, None, None]
        denominators = (radii_3d**2 + (image_indices * circumference) ** 2) ** 1.5
        unit_kernel = (
            circumference[:, :, 0]
            / (4.0 * math.pi)
            * np.sum(radii_3d / denominators, axis=2)
        )
        unit_kernel = np.maximum(unit_kernel, 1e-300)
        log_scale = np.mean(np.log(magnitudes)[None, :] - np.log(unit_kernel), axis=1)
        residual = (
            np.log(unit_kernel) + log_scale[:, None] - np.log(magnitudes)[None, :]
        )
        return (log_scale, np.sqrt(np.mean(np.square(residual), axis=1)))

    lower = max(float(radii[0]) / 16.0, 1e-05)
    upper = max(float(radii[-1]) * 4.0, lower * 32.0)
    coarse = np.geomspace(lower, upper, 120)
    coarse_scale, coarse_rmse = evaluate_grid(coarse)
    best_index = int(np.argmin(coarse_rmse))
    lo_index = max(0, best_index - 1)
    hi_index = min(len(coarse) - 1, best_index + 1)
    refined = np.geomspace(coarse[lo_index], coarse[hi_index], 120)
    refined_scale, refined_rmse = evaluate_grid(refined)
    refined_index = int(np.argmin(refined_rmse))
    compact_radius = float(refined[refined_index])
    field_scale = float(math.exp(float(refined_scale[refined_index])))
    log_rmse = float(refined_rmse[refined_index])
    if not all(
        (math.isfinite(value) for value in (compact_radius, field_scale, log_rmse))
    ):
        return None
    return {
        "model": "one_compact_dimension_image_sum",
        "compactification_radius_R": compact_radius,
        "circumference_L": 2.0 * math.pi * compact_radius,
        "field_scale_G": field_scale,
        "relative_log_rmse": log_rmse,
        "radius_count": len(radii),
        "radius_range": [float(radii[0]), float(radii[-1])],
        "normalization": "fit to |inward_acceleration|*|p2/p1| from paired causal responses",
    }


def _fit_screened_helmholtz(responses: list[dict]) -> Optional[dict]:
    by_radius: dict[float, list[float]] = {}
    for record in responses:
        try:
            p1 = float(record["p1"])
            p2 = float(record["p2"])
            radius = float(record["radius"])
            inward = abs(float(record["inward_acceleration"]))
        except (KeyError, TypeError, ValueError):
            continue
        if (
            not all((math.isfinite(value) for value in (p1, p2, radius, inward)))
            or abs(p1) <= 1e-12
            or abs(p2) <= 1e-12
            or (radius <= 0.0)
            or (inward <= 1e-12)
        ):
            continue
        by_radius.setdefault(round(radius, 12), []).append(inward * abs(p2 / p1))
    if len(by_radius) < 4:
        return None
    ordered = sorted(
        ((radius, float(np.median(values))) for radius, values in by_radius.items())
    )
    radii = np.asarray([item[0] for item in ordered], dtype=float)
    magnitudes = np.asarray([item[1] for item in ordered], dtype=float)
    if radii[-1] / radii[0] < 4.0:
        return None

    def evaluate_grid(length_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lengths = length_grid[:, None]
        unit_kernel = bessel_k1(radii[None, :] / lengths) / (2.0 * math.pi * lengths)
        unit_kernel = np.maximum(unit_kernel, 1e-300)
        log_scale = np.mean(np.log(magnitudes)[None, :] - np.log(unit_kernel), axis=1)
        residual = (
            np.log(unit_kernel) + log_scale[:, None] - np.log(magnitudes)[None, :]
        )
        return (log_scale, np.sqrt(np.mean(np.square(residual), axis=1)))

    lower = max(float(radii[0]) / 16.0, 1e-05)
    upper = max(float(radii[-1]) * 4.0, lower * 32.0)
    coarse = np.geomspace(lower, upper, 160)
    _, coarse_rmse = evaluate_grid(coarse)
    best_index = int(np.argmin(coarse_rmse))
    lo_index = max(0, best_index - 1)
    hi_index = min(len(coarse) - 1, best_index + 1)
    refined = np.geomspace(coarse[lo_index], coarse[hi_index], 160)
    refined_scale, refined_rmse = evaluate_grid(refined)
    refined_index = int(np.argmin(refined_rmse))
    screening_length = float(refined[refined_index])
    field_scale = float(math.exp(float(refined_scale[refined_index])))
    log_rmse = float(refined_rmse[refined_index])
    if not all(
        (math.isfinite(value) for value in (screening_length, field_scale, log_rmse))
    ):
        return None
    return {
        "model": "two_dimensional_screened_helmholtz_K1",
        "screening_length_lambda": screening_length,
        "field_scale_G": field_scale,
        "relative_log_rmse": log_rmse,
        "radius_count": len(radii),
        "radius_range": [float(radii[0]), float(radii[-1])],
        "normalization": "fit to |inward_acceleration|*|p2/p1| from paired causal responses",
    }


def _fit_absolute_time_sinusoid(time_series: list[list[dict]]) -> Optional[dict]:
    candidates = []
    for series in time_series:
        if not isinstance(series, list):
            continue
        by_time: dict[float, list[float]] = {}
        for item in series:
            if not isinstance(item, dict):
                continue
            try:
                time_value = float(item["start_time"])
                response = float(item["inward_acceleration"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(time_value) and math.isfinite(response):
                by_time.setdefault(time_value, []).append(response)
        if len(by_time) < 3:
            continue
        ordered = sorted(
            (
                (time_value, float(np.median(values)))
                for time_value, values in by_time.items()
            )
        )
        times = np.asarray([item[0] for item in ordered], dtype=float)
        values = np.asarray([item[1] for item in ordered], dtype=float)
        if len(by_time) == 3:
            gaps = np.diff(times)
            span = float(times[-1] - times[0])
            endpoint_scale = max(abs(float(values[0])), abs(float(values[-1])), 1e-12)
            equally_spaced = bool(
                len(gaps) == 2
                and min(gaps) > 1e-12
                and (
                    abs(float(gaps[0] - gaps[1]))
                    <= 0.05 * max(float(gaps[0]), float(gaps[1]))
                )
            )
            opposite_endpoints = bool(
                float(values[0] * values[-1]) < 0.0
                and abs(abs(float(values[0])) - abs(float(values[-1])))
                <= 0.25 * endpoint_scale
            )
            midpoint_near_zero = bool(abs(float(values[1])) <= 0.25 * endpoint_scale)
            if not (
                equally_spaced
                and opposite_endpoints
                and midpoint_near_zero
                and (span > 0.0)
            ):
                continue
            offset = float((values[0] + values[-1]) / 2.0)
            amplitude = float(abs(values[0] - values[-1]) / 2.0)
            frequency = float(math.pi / span)

            def alias_phase(alias_frequency: float) -> float:
                raw_phase = (
                    -alias_frequency * float(times[0])
                    if values[0] >= offset
                    else math.pi - alias_frequency * float(times[0])
                )
                return float((raw_phase + math.pi) % (2.0 * math.pi) - math.pi)

            phase = alias_phase(frequency)
            frequency_aliases = [
                {
                    "alias_index": alias_index,
                    "angular_frequency_omega": float((2 * alias_index + 1) * frequency),
                    "phase": alias_phase(float((2 * alias_index + 1) * frequency)),
                    "period": float(
                        2.0 * math.pi / ((2 * alias_index + 1) * frequency)
                    ),
                }
                for alias_index in range(_MAX_TIME_FREQUENCY_ALIASES)
            ]
            prediction = offset + amplitude * np.cos(frequency * times + phase)
            rmse = float(np.sqrt(np.mean(np.square(prediction - values))))
            relative_rmse = rmse / max(amplitude, 1e-12)
            candidates.append(
                {
                    "model": "absolute_time_sinusoidal_coupling",
                    "fit_mode": "three_phase_antisymmetric",
                    "offset": offset,
                    "amplitude": amplitude,
                    "angular_frequency_omega": frequency,
                    "phase": phase,
                    "period": 2.0 * math.pi / frequency,
                    "rmse": rmse,
                    "relative_rmse": relative_rmse,
                    "time_count": len(times),
                    "time_range": [float(times[0]), float(times[-1])],
                    "frequency_aliases": frequency_aliases,
                    "frequency_identifiability": "three equally spaced antisymmetric phases identify an odd-harmonic alias family, not a unique period",
                }
            )
            continue
        gaps = np.diff(times)
        positive_gaps = gaps[gaps > 1e-12]
        span = float(times[-1] - times[0])
        if not len(positive_gaps) or span <= 0.0:
            continue
        min_gap = float(np.min(positive_gaps))
        lower = max(2.0 * math.pi / (8.0 * span), 0.0001)
        upper = math.pi / min_gap

        def evaluate_grid(
            frequency_grid: np.ndarray,
        ) -> tuple[list[np.ndarray], np.ndarray]:
            coefficients = []
            errors = []
            for frequency in frequency_grid:
                matrix = np.column_stack(
                    (
                        np.ones_like(times),
                        np.cos(frequency * times),
                        np.sin(frequency * times),
                    )
                )
                coefficient, _, _, _ = np.linalg.lstsq(matrix, values, rcond=None)
                residual = matrix @ coefficient - values
                coefficients.append(coefficient)
                errors.append(float(np.sqrt(np.mean(np.square(residual)))))
            return (coefficients, np.asarray(errors, dtype=float))

        coarse = np.linspace(lower, upper, 400)
        _, coarse_errors = evaluate_grid(coarse)
        best_index = int(np.argmin(coarse_errors))
        lo_index = max(0, best_index - 1)
        hi_index = min(len(coarse) - 1, best_index + 1)
        refined = np.linspace(coarse[lo_index], coarse[hi_index], 400)
        refined_coefficients, refined_errors = evaluate_grid(refined)
        refined_index = int(np.argmin(refined_errors))
        offset, cosine_coefficient, sine_coefficient = (
            float(value) for value in refined_coefficients[refined_index]
        )
        frequency = float(refined[refined_index])
        amplitude = float(math.hypot(cosine_coefficient, sine_coefficient))
        phase = float(math.atan2(-sine_coefficient, cosine_coefficient))
        rmse = float(refined_errors[refined_index])
        relative_rmse = rmse / max(amplitude, 1e-12)
        if amplitude <= 1e-10 or not all(
            (
                math.isfinite(value)
                for value in (offset, amplitude, frequency, phase, rmse, relative_rmse)
            )
        ):
            continue
        candidates.append(
            {
                "model": "absolute_time_sinusoidal_coupling",
                "offset": offset,
                "amplitude": amplitude,
                "angular_frequency_omega": frequency,
                "phase": phase,
                "period": 2.0 * math.pi / frequency,
                "rmse": rmse,
                "relative_rmse": relative_rmse,
                "time_count": len(times),
                "time_range": [float(times[0]), float(times[-1])],
            }
        )
    if not candidates:
        return None
    return min(candidates, key=lambda item: item["relative_rmse"])


def _as_list(value: Any) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _safe_confidence(value: Any, default: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not math.isfinite(number):
        number = default
    return min(max(number, 1e-09), 1.0)


def resolve_json_pointer(document: Any, pointer: str) -> Any:
    if pointer == "":
        return document
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("JSON pointer must be empty or start with '/'")
    current = document
    for raw_token in pointer.split("/")[1:]:
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            try:
                index = int(token)
            except ValueError as exc:
                if (
                    len(current) == 1
                    and isinstance(current[0], dict)
                    and (token in current[0])
                ):
                    current = current[0][token]
                else:
                    raise KeyError(f"{token!r} is not a list index") from exc
            else:
                current = current[index]
        elif isinstance(current, dict):
            if token not in current:
                raise KeyError(f"missing key {token!r}")
            current = current[token]
        else:
            raise KeyError(f"cannot traverse {token!r} through a scalar")
    return current


def set_json_pointer(document: Any, pointer: str, value: Any) -> None:
    if not pointer or not isinstance(pointer, str) or (not pointer.startswith("/")):
        raise ValueError("intervention pointer must start with '/'")
    tokens = [
        token.replace("~1", "/").replace("~0", "~") for token in pointer.split("/")[1:]
    ]
    current = document
    for token in tokens[:-1]:
        if isinstance(current, list):
            current = current[int(token)]
        elif isinstance(current, dict):
            if token not in current:
                raise KeyError(f"intervention path has no key {token!r}")
            current = current[token]
        else:
            raise KeyError(f"cannot traverse intervention path through {token!r}")
    final = tokens[-1]
    if isinstance(current, list):
        current[int(final)] = copy.deepcopy(value)
    elif isinstance(current, dict):
        if final not in current:
            raise KeyError(f"intervention path has no key {final!r}")
        current[final] = copy.deepcopy(value)
    else:
        raise KeyError(f"cannot set intervention path at {final!r}")


def _rng_snapshot(executor: Any) -> dict[str, Any]:
    snapshot = {}
    for name in ("_noise_rng", "_vel_noise_rng"):
        rng = getattr(executor, name, None)
        if rng is not None and hasattr(rng, "bit_generator"):
            snapshot[name] = copy.deepcopy(rng.bit_generator.state)
    return snapshot


def _rng_restore(executor: Any, snapshot: dict[str, Any]) -> None:
    for name, state in snapshot.items():
        rng = getattr(executor, name, None)
        if rng is not None and hasattr(rng, "bit_generator"):
            rng.bit_generator.state = copy.deepcopy(state)


def _numeric_difference(counterfactual: Any, factual: Any) -> Any:
    if (
        isinstance(counterfactual, (int, float, np.number))
        and (not isinstance(counterfactual, bool))
        and isinstance(factual, (int, float, np.number))
        and (not isinstance(factual, bool))
    ):
        left = float(counterfactual)
        right = float(factual)
        if math.isfinite(left) and math.isfinite(right):
            return left - right
        return None
    if isinstance(counterfactual, list) and isinstance(factual, list):
        if len(counterfactual) != len(factual):
            return None
        return [
            _numeric_difference(left, right)
            for left, right in zip(counterfactual, factual)
        ]
    if isinstance(counterfactual, dict) and isinstance(factual, dict):
        return {
            key: _numeric_difference(counterfactual[key], factual[key])
            for key in counterfactual.keys() & factual.keys()
        }
    return None


@dataclass
class SelectedDesign:
    design_id: str
    kind: str
    score: float
    payload: dict
    predictions: list[dict]
    proposal_scores: list[dict]

    def to_dict(self) -> dict:
        return {
            "design_id": self.design_id,
            "kind": self.kind,
            "information_score": float(self.score),
            "proposal_scores": copy.deepcopy(self.proposal_scores),
        }


class SCMPipeline:
    def __init__(
        self,
        world: str,
        mission: str = "",
        strict: bool = False,
        max_candidates: int = 8,
        complexity_penalty: float = 0.25,
        evidence_temperature: float = 0.7,
        min_disagreement_z: float = 0.5,
        observation_noise_std: float = 0.0,
        public_experiment_format: str = "",
    ):
        self.world = str(world)
        self.mission = mission or ""
        self.strict = bool(strict)
        self.max_candidates = max(2, int(max_candidates))
        self.complexity_penalty = max(0.0, float(complexity_penalty))
        self.evidence_temperature = min(max(float(evidence_temperature), 0.0), 1.0)
        self.min_disagreement_z = max(0.0, float(min_disagreement_z))
        self.observation_noise_std = max(0.0, float(observation_noise_std))
        self.public_experiment_format = str(public_experiment_format or "")
        self.episode_id = uuid.uuid4().hex
        self.reset()

    def reset(self) -> None:
        self.round = 0
        self.candidates: dict[str, dict] = {}
        self.log_weights: dict[str, float] = {}
        self.version_history: list[dict] = []
        self.abductions: list[dict] = []
        self.inductions: list[dict] = []
        self.deductions: list[dict] = []
        self.patches: list[dict] = []
        self.evidence: list[dict] = []
        self.experiments: list[dict] = []
        self.protocol_errors: list[dict] = []
        self.open_questions: list[Any] = []
        self.fit_attempts: list[dict] = []
        self.capacity_events: list[dict] = []
        self.final_claim: Any = None
        self.final_law: Optional[str] = None
        self.final_explanation: Optional[str] = None
        self.time_alias_selection: Optional[dict] = None
        self._last_contract_selection_errors: list[str] = []
        self._pending_selection: Optional[SelectedDesign] = None
        self._pending_loose_predictions: list[dict] = []
        self._last_update: Optional[dict] = None

    @property
    def pending_selection(self) -> Optional[SelectedDesign]:
        return self._pending_selection

    def prompt_block(self) -> str:
        root = Path(__file__).resolve().parents[2]
        path = root / "PhysicsSchool" / "prompts" / "scm_protocol.md"
        try:
            protocol = path.read_text()
        except OSError:
            protocol = "Maintain multiple explicit SCM hypotheses, preregister numerical predictions before every experiment, and revise them only after comparing those predictions with simulator evidence."
        return protocol

    def round_context(self, round_num: int) -> str:
        self.round = int(round_num)
        summary = self.state_summary()
        if not self.candidates:
            target = "3-6"
            requirement = f"Initialize {target} diverse candidate SCMs from weak scientific priors before proposing the first active experiment."
        else:
            priorities = []
            if self.experiments and (not self.version_history):
                priorities.append(
                    "make an explicit evidence-based model revision or parameter patch"
                )
            if self.experiments and (not self.inductions):
                priorities.append("state the rule induced across observations")
            if self.experiments and (
                not any(
                    (
                        record.get("kind") == "counterfactual"
                        for record in self.experiments
                    )
                )
            ):
                priorities.append(
                    "include paired-counterfactual alternatives that isolate one cause"
                )
            priorities.extend(self._causal_audit_priorities())
            priority_text = (
                " Highest-priority missing loop elements: "
                + "; ".join(priorities)
                + "."
                if priorities
                else ""
            )
            requirement = (
                "Use the evidence and posterior below to abduct causes, revise rules, deduce a discriminating consequence, and propose the next active experiment."
                + priority_text
            )
        return (
            "<scm_runtime_state>\n"
            + json.dumps(summary, separators=(",", ":"), ensure_ascii=False)
            + "\n</scm_runtime_state>\n"
            + requirement
        )

    def state_summary(self) -> dict:
        posterior = self.posterior()
        ranked = sorted(
            self.candidates.values(),
            key=lambda c: posterior.get(c["id"], 0.0),
            reverse=True,
        )
        candidate_summaries = []
        for candidate in ranked[: self.max_candidates]:
            candidate_summaries.append(
                {
                    "id": candidate["id"],
                    "name": candidate.get("name"),
                    "description": candidate.get("description"),
                    "revision": candidate.get("revision", 1),
                    "status": candidate.get("status", "active"),
                    "posterior": posterior.get(candidate["id"], 0.0),
                    "complexity": self._candidate_complexity(candidate),
                    "mechanisms": candidate.get("mechanisms", []),
                    "parameters": candidate.get("parameters", {}),
                    "falsifiers": candidate.get("falsifiers", []),
                }
            )
        summary = {
            "schema_version": SCM_SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "round": self.round,
            "candidates": candidate_summaries,
            "recent_evidence": self.evidence[-8:],
            "latest_abduction": self.abductions[-1] if self.abductions else None,
            "latest_induction": self.inductions[-3:],
            "latest_deductions": self.deductions[-3:],
            "open_questions": self.open_questions[-8:],
            "intervention_coverage": self._intervention_coverage(),
            "noise_cancelled_causal_audit": self._noise_cancelled_causal_audit(),
            "mandatory_audit_blockers": self.finalization_blockers(),
            "latest_mse_fit": copy.deepcopy(self.fit_attempts[-1])
            if self.fit_attempts
            else None,
            "time_alias_selection": copy.deepcopy(self.time_alias_selection),
            "metrics": self.metrics(),
        }
        return summary

    def _executed_input_cases(self) -> list[dict]:
        cases: list[dict] = []
        for record in self.experiments:
            payload = record.get("input")
            if isinstance(payload, list):
                cases.extend(
                    (copy.deepcopy(case) for case in payload if isinstance(case, dict))
                )
                continue
            if not isinstance(payload, dict):
                continue
            factual = payload.get("factual")
            if not isinstance(factual, dict):
                continue
            cases.append(copy.deepcopy(factual))
            for intervention in _as_list(payload.get("interventions")):
                assignments = (
                    intervention.get("set") if isinstance(intervention, dict) else None
                )
                if not isinstance(assignments, dict):
                    continue
                branch = copy.deepcopy(factual)
                try:
                    for pointer, value in assignments.items():
                        set_json_pointer(branch, pointer, value)
                except (KeyError, TypeError, ValueError):
                    continue
                cases.append(branch)
        return cases

    def _intervention_coverage(self) -> dict:
        cases = self._executed_input_cases()
        if not cases:
            return {"executed_cases": 0}
        scalar_values: dict[str, list[float]] = {}
        radii: list[float] = []
        speed_norms: list[float] = []
        for case in cases:
            for name, value in case.items():
                if (
                    isinstance(value, (int, float))
                    and (not isinstance(value, bool))
                    and math.isfinite(float(value))
                ):
                    scalar_values.setdefault(str(name), []).append(float(value))
            pos2 = case.get("pos2")
            if (
                isinstance(pos2, (list, tuple))
                and len(pos2) == 2
                and all((isinstance(value, (int, float)) for value in pos2))
            ):
                radii.append(float(np.linalg.norm(np.asarray(pos2, dtype=float))))
            velocity2 = case.get("velocity2")
            if (
                isinstance(velocity2, (list, tuple))
                and len(velocity2) == 2
                and all((isinstance(value, (int, float)) for value in velocity2))
            ):
                speed_norms.append(
                    float(np.linalg.norm(np.asarray(velocity2, dtype=float)))
                )

        def _summary(values: list[float]) -> dict:
            unique = sorted(set((round(value, 12) for value in values)))
            result = {
                "min": min(values),
                "max": max(values),
                "n_unique": len(unique),
                "has_negative": any((value < 0 for value in values)),
                "has_zero": any((value == 0 for value in values)),
                "has_positive": any((value > 0 for value in values)),
            }
            if len(unique) <= 8:
                result["values"] = unique
            return result

        coverage = {
            "executed_cases": len(cases),
            "scalar_controls": {
                name: _summary(values) for name, values in sorted(scalar_values.items())
            },
        }
        if radii:
            coverage["pos2_radius"] = _summary(radii)
        if speed_norms:
            coverage["velocity2_norm"] = _summary(speed_norms)
        return coverage

    @staticmethod
    def _two_particle_controls(case: Any) -> Optional[dict]:
        if not isinstance(case, dict):
            return None
        try:
            p1 = float(case["p1"])
            p2 = float(case["p2"])
            pos2 = np.asarray(case["pos2"], dtype=float)
            velocity2 = np.asarray(case["velocity2"], dtype=float)
            start_time = float(case.get("start_time", 0.0))
        except (KeyError, TypeError, ValueError):
            return None
        if (
            pos2.shape != (2,)
            or velocity2.shape != (2,)
            or (not np.isfinite(pos2).all())
            or (not np.isfinite(velocity2).all())
            or (not all((math.isfinite(value) for value in (p1, p2, start_time))))
        ):
            return None

        def clean(value: float) -> float:
            rounded = round(float(value), 12)
            return 0.0 if rounded == 0.0 else rounded

        return {
            "p1": clean(p1),
            "p2": clean(p2),
            "pos2": [clean(pos2[0]), clean(pos2[1])],
            "velocity2": [clean(velocity2[0]), clean(velocity2[1])],
            "start_time": clean(start_time),
        }

    @classmethod
    def _local_counterfactual_edges(cls, execution: Any) -> list[dict]:
        if not isinstance(execution, dict) or execution.get("kind") != "counterfactual":
            return []
        outcome = execution.get("outcome")
        if not isinstance(outcome, dict):
            return []
        factual = outcome.get("factual")
        if not isinstance(factual, dict):
            return []
        factual_controls = cls._two_particle_controls(factual.get("input"))
        factual_outputs = factual.get("output")
        if factual_controls is None or not isinstance(factual_outputs, list):
            return []
        if not factual_outputs or not isinstance(factual_outputs[0], dict):
            return []
        factual_output = factual_outputs[0]
        times = factual_output.get(
            "measurement_times",
            (factual.get("input") or {}).get("measurement_times", []),
        )
        if not isinstance(times, list):
            return []
        observation_index = next(
            (
                index
                for index, value in enumerate(times)
                if isinstance(value, (int, float))
                and (not isinstance(value, bool))
                and math.isfinite(float(value))
                and (float(value) > 0.0)
            ),
            None,
        )
        if observation_index is None:
            return []
        elapsed = float(times[observation_index])
        effects = {
            str(effect.get("id")): effect
            for effect in _as_list(outcome.get("paired_effects"))
            if isinstance(effect, dict)
        }
        edges: list[dict] = []
        for branch in _as_list(outcome.get("counterfactuals")):
            if not isinstance(branch, dict):
                continue
            branch_id = str(branch.get("id") or "")
            branch_controls = cls._two_particle_controls(branch.get("input"))
            effect = effects.get(branch_id)
            deltas = effect.get("output_delta") if isinstance(effect, dict) else None
            if (
                branch_controls is None
                or not isinstance(deltas, list)
                or (not deltas)
                or (not isinstance(deltas[0], dict))
            ):
                continue
            velocity_delta = deltas[0].get("velocity2")
            try:
                delta = np.asarray(velocity_delta[observation_index], dtype=float)
                initial_delta = np.asarray(
                    branch_controls["velocity2"], dtype=float
                ) - np.asarray(factual_controls["velocity2"], dtype=float)
            except (IndexError, TypeError, ValueError):
                continue
            if (
                delta.shape != (2,)
                or initial_delta.shape != (2,)
                or (not np.isfinite(delta).all())
                or (not np.isfinite(initial_delta).all())
            ):
                continue
            dynamic_delta = delta - initial_delta
            edges.append(
                {
                    "branch_id": branch_id,
                    "factual": factual_controls,
                    "counterfactual": branch_controls,
                    "elapsed_time": elapsed,
                    "initial_velocity_delta": initial_delta.tolist(),
                    "delta_acceleration": (dynamic_delta / elapsed).tolist(),
                    "noise_control": "common_random_numbers",
                }
            )
        return _jsonable(edges)

    @classmethod
    def _direct_intervention_edges(
        cls, experiment: Any, *, allow_prior_null_anchor: bool = False
    ) -> list[dict]:
        if not isinstance(experiment, dict) or experiment.get("kind") != "intervention":
            return []
        inputs = experiment.get("input")
        outputs = experiment.get("output")
        if (
            not isinstance(inputs, list)
            or not isinstance(outputs, list)
            or len(inputs) != len(outputs)
        ):
            return []
        observations = []
        for case, output in zip(inputs, outputs):
            controls = cls._two_particle_controls(case)
            if controls is None or not isinstance(output, dict):
                continue
            times = output.get("measurement_times", case.get("measurement_times", []))
            velocities = output.get("velocity2")
            if (
                not isinstance(times, list)
                or len(times) != 1
                or (not isinstance(velocities, list))
                or (len(velocities) != 1)
            ):
                continue
            try:
                elapsed = float(times[0])
                final_velocity = np.asarray(velocities[0], dtype=float)
                initial_velocity = np.asarray(controls["velocity2"], dtype=float)
            except (TypeError, ValueError):
                continue
            if (
                not math.isfinite(elapsed)
                or elapsed <= 0.0
                or final_velocity.shape != (2,)
                or (initial_velocity.shape != (2,))
                or (not np.isfinite(final_velocity).all())
                or (not np.isfinite(initial_velocity).all())
            ):
                continue
            observations.append(
                {
                    "controls": controls,
                    "elapsed_time": elapsed,
                    "acceleration": (final_velocity - initial_velocity) / elapsed,
                }
            )
        null_rows = [
            row for row in observations if abs(float(row["controls"]["p1"])) <= 1e-12
        ]
        active_rows = [
            row for row in observations if abs(float(row["controls"]["p1"])) > 1e-12
        ]
        if not active_rows:
            return []
        if null_rows:
            null_acceleration = np.median(
                np.asarray([row["acceleration"] for row in null_rows], dtype=float),
                axis=0,
            )
            shared_null_controls = copy.deepcopy(null_rows[0]["controls"])
            noise_control = "single-measurement velocity response minus observed p1=0 source-null reference"
        elif allow_prior_null_anchor and all(
            (
                float(
                    np.linalg.norm(
                        np.asarray(row["controls"]["velocity2"], dtype=float)
                    )
                )
                <= 1e-12
                for row in active_rows
            )
        ):
            null_acceleration = np.zeros(2, dtype=float)
            shared_null_controls = None
            noise_control = "exact single-measurement zero-velocity response connected to a previously observed paired p1=0 source-null anchor"
        else:
            return []
        edges = []
        for index, row in enumerate(active_rows):
            delta = np.asarray(row["acceleration"], dtype=float) - null_acceleration
            if delta.shape != (2,) or not np.isfinite(delta).all():
                continue
            null_controls = (
                copy.deepcopy(shared_null_controls)
                if shared_null_controls is not None
                else {**copy.deepcopy(row["controls"]), "p1": 0.0}
            )
            edges.append(
                {
                    "branch_id": f"direct_single_cadence_{index}",
                    "factual": copy.deepcopy(null_controls),
                    "counterfactual": copy.deepcopy(row["controls"]),
                    "elapsed_time": float(row["elapsed_time"]),
                    "delta_acceleration": delta.tolist(),
                    "noise_control": noise_control,
                }
            )
        return _jsonable(edges)

    @staticmethod
    def _control_key(controls: dict) -> tuple:
        return (
            float(controls["p1"]),
            float(controls["p2"]),
            float(controls["pos2"][0]),
            float(controls["pos2"][1]),
            float(controls["velocity2"][0]),
            float(controls["velocity2"][1]),
            float(controls.get("start_time", 0.0)),
        )

    def _noise_cancelled_causal_audit(self) -> Optional[dict]:
        paired_edges = [
            edge
            for experiment in self.experiments
            for edge in _as_list(experiment.get("local_causal_edges"))
            if isinstance(edge, dict)
        ]
        has_paired_null_anchor = any(
            (
                isinstance(edge.get("factual"), dict)
                and abs(float(edge["factual"].get("p1", math.inf))) <= 1e-12
                or (
                    isinstance(edge.get("counterfactual"), dict)
                    and abs(float(edge["counterfactual"].get("p1", math.inf))) <= 1e-12
                )
                for edge in paired_edges
            )
        )
        direct_edges = [
            edge
            for experiment in self.experiments
            for edge in self._direct_intervention_edges(
                experiment, allow_prior_null_anchor=has_paired_null_anchor
            )
            if isinstance(edge, dict)
        ]
        raw_edges = [*paired_edges, *direct_edges]
        if not raw_edges:
            return None
        nodes: dict[tuple, dict] = {}
        edges: list[tuple[tuple, tuple, np.ndarray]] = []
        adjacency: dict[tuple, set[tuple]] = {}
        for edge in raw_edges:
            factual = edge.get("factual")
            counterfactual = edge.get("counterfactual")
            try:
                left = self._control_key(factual)
                right = self._control_key(counterfactual)
                delta = np.asarray(edge["delta_acceleration"], dtype=float)
            except (KeyError, TypeError, ValueError):
                continue
            if delta.shape != (2,) or not np.isfinite(delta).all():
                continue
            nodes[left] = copy.deepcopy(factual)
            nodes[right] = copy.deepcopy(counterfactual)
            edges.append((left, right, delta))
            adjacency.setdefault(left, set()).add(right)
            adjacency.setdefault(right, set()).add(left)
        anchors = {
            key
            for key, controls in nodes.items()
            if abs(float(controls["p1"])) <= 1e-12
        }
        if not edges or not anchors:
            return {
                "method": "paired first-velocity contrasts were recorded, but no connected p1=0 null-source anchor is available yet",
                "edge_count": len(edges),
                "anchored_response_count": 0,
                "responses": [],
                "controlled_scaling": {},
                "candidate_correspondences": [],
            }
        connected = set(anchors)
        frontier = list(anchors)
        while frontier:
            current = frontier.pop()
            for neighbour in adjacency.get(current, set()):
                if neighbour not in connected:
                    connected.add(neighbour)
                    frontier.append(neighbour)
        ordered = sorted(connected)
        column = {key: index for index, key in enumerate(ordered)}
        matrix_rows: list[np.ndarray] = []
        rhs_rows: list[np.ndarray] = []
        for left, right, delta in edges:
            if left not in connected or right not in connected:
                continue
            row = np.zeros(len(ordered), dtype=float)
            row[column[left]] = -1.0
            row[column[right]] = 1.0
            matrix_rows.append(row)
            rhs_rows.append(delta)
        for key in sorted(anchors):
            row = np.zeros(len(ordered), dtype=float)
            row[column[key]] = 10.0
            matrix_rows.append(row)
            rhs_rows.append(np.zeros(2, dtype=float))
        matrix = np.asarray(matrix_rows, dtype=float)
        rhs = np.asarray(rhs_rows, dtype=float)
        solution, _, _, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
        residual = matrix @ solution - rhs
        residual_rmse = float(np.sqrt(np.mean(np.square(residual))))
        responses: list[dict] = []
        response_by_key: dict[tuple, dict] = {}
        for key in ordered:
            controls = nodes[key]
            acceleration = solution[column[key]]
            position = np.asarray(controls["pos2"], dtype=float)
            radius = float(np.linalg.norm(position))
            if radius > 0.0:
                radial_hat = position / radius
                outward = float(np.dot(acceleration, radial_hat))
                inward = -outward
                tangential = float(
                    radial_hat[0] * acceleration[1] - radial_hat[1] * acceleration[0]
                )
            else:
                inward = 0.0
                tangential = float(np.linalg.norm(acceleration))
            record = {
                **copy.deepcopy(controls),
                "radius": radius,
                "acceleration": acceleration.tolist(),
                "inward_acceleration": inward,
                "tangential_acceleration": tangential,
            }
            responses.append(record)
            response_by_key[key] = record

        def same_except(left: tuple, right: tuple, indices: set[int]) -> bool:
            return all(
                (
                    index in indices or left[index] == right[index]
                    for index in range(len(left))
                )
            )

        def power_estimates(control_index: int) -> list[dict]:
            estimates = []
            active_keys = [
                key
                for key in ordered
                if abs(float(nodes[key]["p1"])) > 1e-12
                and abs(float(response_by_key[key]["inward_acceleration"])) > 1e-10
            ]
            for left_index, left in enumerate(active_keys):
                for right in active_keys[left_index + 1 :]:
                    if not same_except(left, right, {control_index}):
                        continue
                    left_value = float(left[control_index])
                    right_value = float(right[control_index])
                    if (
                        left_value == 0.0
                        or right_value == 0.0
                        or math.copysign(1.0, left_value)
                        != math.copysign(1.0, right_value)
                        or (abs(abs(left_value) - abs(right_value)) <= 1e-12)
                    ):
                        continue
                    left_response = abs(
                        float(response_by_key[left]["inward_acceleration"])
                    )
                    right_response = abs(
                        float(response_by_key[right]["inward_acceleration"])
                    )
                    if min(left_response, right_response) <= 1e-10:
                        continue
                    exponent = math.log(right_response / left_response) / math.log(
                        abs(right_value) / abs(left_value)
                    )
                    if math.isfinite(exponent):
                        estimates.append(
                            {
                                "value_pair": [left_value, right_value],
                                "response_pair": [left_response, right_response],
                                "exponent": exponent,
                            }
                        )
            return estimates

        radius_groups: dict[tuple, list[tuple[float, float]]] = {}
        for key, record in response_by_key.items():
            if abs(float(record["p1"])) <= 1e-12:
                continue
            position = np.asarray(record["pos2"], dtype=float)
            radius = float(record["radius"])
            if radius <= 0.0:
                continue
            direction = tuple(np.round(position / radius, 6))
            group_key = (key[0], key[1], key[4], key[5], key[6], direction)
            radius_groups.setdefault(group_key, []).append(
                (radius, abs(float(record["inward_acceleration"])))
            )
        radius_decay: list[dict] = []
        for values in radius_groups.values():
            by_radius: dict[float, list[float]] = {}
            for radius, response in values:
                if response > 1e-10:
                    by_radius.setdefault(radius, []).append(response)
            ordered_values = sorted(
                (
                    (radius, float(np.median(items)))
                    for radius, items in by_radius.items()
                )
            )
            for (near_radius, near_response), (far_radius, far_response) in zip(
                ordered_values, ordered_values[1:]
            ):
                exponent = -math.log(far_response / near_response) / math.log(
                    far_radius / near_radius
                )
                if math.isfinite(exponent):
                    radius_decay.append(
                        {
                            "radius_pair": [near_radius, far_radius],
                            "inward_response_pair": [near_response, far_response],
                            "decay_exponent_q": exponent,
                            "geometric_midpoint": math.sqrt(near_radius * far_radius),
                        }
                    )
        decay_by_interval: dict[tuple[float, float], list[dict]] = {}
        for record in radius_decay:
            pair = record.get("radius_pair")
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            interval = (round(float(pair[0]), 12), round(float(pair[1]), 12))
            decay_by_interval.setdefault(interval, []).append(record)
        pooled_radius_decay: list[dict] = []
        for (near_radius, far_radius), records in decay_by_interval.items():
            exponents = [
                float(record["decay_exponent_q"])
                for record in records
                if isinstance(record.get("decay_exponent_q"), (int, float))
                and math.isfinite(float(record["decay_exponent_q"]))
            ]
            response_pairs = [
                record["inward_response_pair"]
                for record in records
                if isinstance(record.get("inward_response_pair"), list)
                and len(record["inward_response_pair"]) == 2
            ]
            if not exponents:
                continue
            pooled_radius_decay.append(
                {
                    "radius_pair": [near_radius, far_radius],
                    "inward_response_pair": [
                        float(np.median([float(pair[0]) for pair in response_pairs])),
                        float(np.median([float(pair[1]) for pair in response_pairs])),
                    ]
                    if response_pairs
                    else [],
                    "decay_exponent_q": float(np.median(exponents)),
                    "geometric_midpoint": math.sqrt(near_radius * far_radius),
                    "replicate_count": len(records),
                }
            )
        radius_decay = sorted(
            pooled_radius_decay, key=lambda item: item["geometric_midpoint"]
        )

        def sign_relations(control_index: int) -> list[dict]:
            relations = []
            active_keys = [
                key for key in ordered if abs(float(nodes[key]["p1"])) > 1e-12
            ]
            for left_index, left in enumerate(active_keys):
                for right in active_keys[left_index + 1 :]:
                    if not same_except(left, right, {control_index}):
                        continue
                    left_value = float(left[control_index])
                    right_value = float(right[control_index])
                    if (
                        left_value * right_value >= 0.0
                        or abs(abs(left_value) - abs(right_value)) > 1e-12
                    ):
                        continue
                    left_response = float(response_by_key[left]["inward_acceleration"])
                    right_response = float(
                        response_by_key[right]["inward_acceleration"]
                    )
                    if abs(left_response) <= 1e-10:
                        continue
                    relations.append(
                        {
                            "value_pair": [left_value, right_value],
                            "signed_response_pair": [left_response, right_response],
                            "signed_response_ratio": right_response / left_response,
                        }
                    )
            return relations

        time_groups: dict[tuple, list[dict]] = {}
        for key, record in response_by_key.items():
            if abs(float(record["p1"])) <= 1e-12:
                continue
            group_key = key[:6]
            time_groups.setdefault(group_key, []).append(
                {
                    "start_time": float(record["start_time"]),
                    "inward_acceleration": float(record["inward_acceleration"]),
                    "p1": float(record["p1"]),
                    "p2": float(record["p2"]),
                    "radius": float(record["radius"]),
                }
            )
        absolute_time_response = [
            sorted(values, key=lambda item: item["start_time"])
            for values in time_groups.values()
            if len({item["start_time"] for item in values}) >= 2
        ]
        absolute_time_response.sort(
            key=lambda values: (
                -len(values),
                values[0]["start_time"] if values else 0.0,
            )
        )
        p1_power = power_estimates(0)
        p2_power = power_estimates(1)
        p1_sign_evidence = sign_relations(0)
        p2_sign_evidence = sign_relations(1)

        def median_field(records: list[dict], name: str) -> Optional[float]:
            values = [
                float(record[name])
                for record in records
                if isinstance(record.get(name), (int, float))
                and math.isfinite(float(record[name]))
            ]
            return float(np.median(values)) if values else None

        p1_exponent = median_field(p1_power, "exponent")
        p2_exponent = median_field(p2_power, "exponent")
        radius_exponent = median_field(radius_decay, "decay_exponent_q")
        radial_power_coefficients: list[float] = []
        if radius_exponent is not None and math.isfinite(float(radius_exponent)):
            for record in responses:
                try:
                    p1_value = abs(float(record["p1"]))
                    p2_value = abs(float(record["p2"]))
                    radius_value = float(record["radius"])
                    response_value = abs(float(record["inward_acceleration"]))
                except (KeyError, TypeError, ValueError):
                    continue
                if min(
                    p1_value, p2_value, radius_value, response_value
                ) <= 1e-12 or not all(
                    (
                        math.isfinite(value)
                        for value in (p1_value, p2_value, radius_value, response_value)
                    )
                ):
                    continue
                if p2_exponent is not None and p2_exponent > 0.4:
                    control_factor = p1_value * p2_value
                elif p2_exponent is not None and p2_exponent < -0.4:
                    control_factor = p1_value / p2_value
                else:
                    control_factor = p1_value
                coefficient = (
                    response_value
                    * radius_value ** float(radius_exponent)
                    / control_factor
                )
                if math.isfinite(coefficient) and coefficient > 0.0:
                    radial_power_coefficients.append(coefficient)
        radial_power_coefficient = (
            float(np.median(radial_power_coefficients))
            if radial_power_coefficients
            else None
        )
        sorted_slopes = sorted(
            radius_decay, key=lambda item: item["geometric_midpoint"]
        )
        slope_window = max(1, len(sorted_slopes) // 2)
        near_q = (
            float(
                np.median(
                    [item["decay_exponent_q"] for item in sorted_slopes[:slope_window]]
                )
            )
            if sorted_slopes
            else None
        )
        far_q = (
            float(
                np.median(
                    [item["decay_exponent_q"] for item in sorted_slopes[-slope_window:]]
                )
            )
            if sorted_slopes
            else None
        )
        multiplicative_pair_pattern = bool(
            p2_exponent is not None and p2_exponent > 0.4
        )
        normalized_absolute_time_response: list[list[dict]] = []
        for series in absolute_time_response:
            normalized_series = []
            for item in series:
                try:
                    p1_value = float(item["p1"])
                    p2_value = float(item["p2"])
                    radius_value = float(item["radius"])
                    response_value = float(item["inward_acceleration"])
                except (KeyError, TypeError, ValueError):
                    continue
                if multiplicative_pair_pattern:
                    control_factor = abs(p1_value * p2_value)
                elif p2_exponent is not None and p2_exponent < -0.4:
                    control_factor = (
                        p1_value / p2_value if abs(p2_value) > 1e-12 else 0.0
                    )
                else:
                    control_factor = p1_value
                if abs(control_factor) <= 1e-12:
                    continue
                radial_factor = (
                    radius_value ** float(radius_exponent)
                    if radius_exponent is not None
                    and math.isfinite(float(radius_exponent))
                    and (radius_value > 0.0)
                    else 1.0
                )
                normalized_value = response_value * radial_factor / control_factor
                if not math.isfinite(normalized_value):
                    continue
                normalized_series.append(
                    {
                        **copy.deepcopy(item),
                        "raw_inward_acceleration": response_value,
                        "inward_acceleration": normalized_value,
                    }
                )
            if len(normalized_series) >= 2:
                normalized_absolute_time_response.append(normalized_series)
        time_sign_change = any(
            (
                min((item["inward_acceleration"] for item in series)) < -1e-06
                and max((item["inward_acceleration"] for item in series)) > 1e-06
                for series in normalized_absolute_time_response
            )
        )
        absolute_time_fit = (
            _fit_absolute_time_sinusoid(normalized_absolute_time_response)
            if time_sign_change
            else None
        )
        correspondences: list[str] = []
        ordered_radius_change_point = None
        if len(sorted_slopes) >= 3:
            slope_values = [float(item["decay_exponent_q"]) for item in sorted_slopes]
            monotone_fraction = float(
                np.mean(
                    [
                        right <= left + 0.15
                        for left, right in zip(slope_values, slope_values[1:])
                    ]
                )
            )
            change_candidates = []
            for split in range(1, len(slope_values) - 1):
                prefix = float(np.median(slope_values[:split]))
                suffix = float(np.median(slope_values[split:]))
                change_candidates.append(
                    {
                        "split_after_interval": int(split),
                        "near_decay_q": prefix,
                        "far_decay_q": suffix,
                        "decay_drop": prefix - suffix,
                        "transition_radius": float(
                            sorted_slopes[split - 1]["radius_pair"][1]
                        ),
                        "ordered_nonincrease_fraction": monotone_fraction,
                    }
                )
            if change_candidates:
                best_change = max(
                    change_candidates, key=lambda item: item["decay_drop"]
                )
                if (
                    best_change["near_decay_q"] >= 1.45
                    and best_change["far_decay_q"] <= 1.35
                    and (best_change["decay_drop"] > 0.35)
                    and (monotone_fraction >= 2.0 / 3.0)
                ):
                    ordered_radius_change_point = best_change
        screened_pattern = bool(
            near_q is not None
            and far_q is not None
            and (far_q - near_q > 0.5)
            and (not multiplicative_pair_pattern)
        )
        sparse_dimensional_crossover = bool(
            len(sorted_slopes) >= 2
            and near_q is not None
            and (far_q is not None)
            and (near_q >= 1.45)
            and (far_q <= 1.35)
            and (near_q - far_q > 0.35)
        )
        ordered_dimensional_crossover = bool(ordered_radius_change_point is not None)
        crossover_pattern = bool(
            near_q is not None
            and far_q is not None
            and (
                near_q - far_q > 0.5
                or sparse_dimensional_crossover
                or ordered_dimensional_crossover
            )
            and (not multiplicative_pair_pattern)
        )
        compact_image_fit = (
            _fit_compactified_image_sum(responses) if crossover_pattern else None
        )
        screened_helmholtz_fit = (
            _fit_screened_helmholtz(responses) if screened_pattern else None
        )
        if screened_pattern:
            screening_scale = float(
                max(sorted_slopes, key=lambda item: item["decay_exponent_q"])[
                    "geometric_midpoint"
                ]
            )
            if (
                screened_helmholtz_fit is not None
                and screened_helmholtz_fit["relative_log_rmse"] < 0.12
            ):
                fitted_length = screened_helmholtz_fit["screening_length_lambda"]
                fitted_scale = screened_helmholtz_fit["field_scale_G"]
                fitted_rmse = screened_helmholtz_fit["relative_log_rmse"]
                correspondences.append(
                    f"the controlled radial response quantitatively selects a static two-dimensional screened Helmholtz/Yukawa field: the canonical K1 Green-function gradient fits with G≈{fitted_scale:.4g}, screening length lambda≈{fitted_length:.4g}, and relative log-RMSE {fitted_rmse:.3g}. Its scalar magnitude is |a|=G*|p1/p2|*K1(r/lambda)/(2*pi*lambda), with vector direction -sign(p1/p2)*r_hat and exponential far suppression. The farthest or steepest sampled slope midpoint ({screening_scale:.3g}) is not the screening length"
                )
            else:
                correspondences.append(
                    "the local decay exponent steepens with radius; compare a screened/Helmholtz (Yukawa-like) tail against a single power law. In two dimensions the canonical scalar magnitude is |a|=G*|p1/p2|*K1(r/lambda)/(2*pi*lambda), with vector direction -sign(p1/p2)*r_hat and exponential far suppression. Fit lambda to all controlled radii rather than equating it to one slope midpoint"
                )
        if crossover_pattern:
            crossover_scale = (
                float(ordered_radius_change_point["transition_radius"])
                if ordered_radius_change_point is not None
                else float(
                    min(
                        sorted_slopes,
                        key=lambda item: abs(item["decay_exponent_q"] - 1.5),
                    )["geometric_midpoint"]
                )
            )
            if (
                compact_image_fit is not None
                and compact_image_fit["relative_log_rmse"] < 0.12
            ):
                fitted_radius = compact_image_fit["compactification_radius_R"]
                fitted_scale = compact_image_fit["field_scale_G"]
                fitted_rmse = compact_image_fit["relative_log_rmse"]
                correspondences.append(
                    f"the controlled radial response quantitatively selects a single extra spatial dimension compactified on a circle: the canonical Kaluza-Klein image sum fits the noise-cancelled responses with G≈{fitted_scale:.4g}, compactification radius R≈{fitted_radius:.4g}, and relative log-RMSE {fitted_rmse:.3g}. Its scalar magnitude is (G*L/(4*pi))*|p1/p2|*sum_n r/(r^2+(n*L)^2)^(3/2), L=2*pi*R, with signed vector direction -sign(p1/p2)*r_hat. It has the dimensional crossover |a| proportional to 1/r^2 at short range and 1/r at long range. This image-sum geometry is the governing structure supported by the data, rather than a single noninteger power or a generic two-power interpolation. The slope-transition midpoint ({crossover_scale:.3g}) is not the compactification radius"
                )
            else:
                correspondences.append(
                    "the force is about one inverse power steeper at short range than at long range; compare a single extra spatial dimension compactified on a circle against other crossover mechanisms. The canonical image-sum magnitude is (G*L/(4*pi))*|p1/p2|*sum_n r/(r^2+(n*L)^2)^(3/2), L=2*pi*R. Its defining dimensional crossover is explicitly |a| proportional to 1/r^2 at short range and 1/r at long range. A two-power interpolation can describe these asymptotes but its crossover coefficient must not be reported as the compactification radius"
                )
        if (
            not screened_pattern
            and (not crossover_pattern)
            and (radius_exponent is not None)
            and (0.7 <= radius_exponent <= 1.3)
        ):
            positive_control_responses = [
                float(record["inward_acceleration"])
                for record in responses
                if float(record["p1"]) > 0.0
                and float(record["p2"]) > 0.0
                and isinstance(record.get("inward_acceleration"), (int, float))
            ]
            direction = ""
            if positive_control_responses:
                typical_inward = float(np.median(positive_control_responses))
                direction = (
                    " The positive-control response is inward, so the force is attractive."
                    if typical_inward > 1e-08
                    else " The positive-control response is outward, so the force is repulsive."
                    if typical_inward < -1e-08
                    else ""
                )
            coefficient_statement = ""
            if radial_power_coefficient is not None:
                if time_sign_change:
                    coefficient_statement = f" The median absolute global coupling scale is about {radial_power_coefficient:.4g}. Do not multiply this again by the fitted time sinusoid: the later sinusoid's offset and amplitude are already the global 1/r coefficients after control/radius normalization."
                else:
                    coefficient_statement = f" After dividing out the identified source/response controls and multiplying the local response by r^q, the global radial coefficient is C≈{radial_power_coefficient:.4g}; this C, not the acceleration measured at one radius, is the normalization to encode and bracket in fit_parameters."
            correspondences.append(
                "q≈1 in two visible dimensions corresponds to the logarithmic Green function of an ordinary 2D Poisson/Laplacian field; its central-force magnitude falls explicitly as 1/r, with the source pinned at the origin in this control schema."
                + coefficient_statement
                + direction
            )
        if (
            not screened_pattern
            and (not crossover_pattern)
            and (radius_exponent is not None)
            and (1.3 < radius_exponent < 2.7)
        ):
            alpha = (3.0 - radius_exponent) / 2.0
            if p2_exponent is not None and p2_exponent < -0.4:
                correspondences.append(
                    f"the controlled radial response with p2 as an inertial denominator identifies the 2D Riesz/fractional-Laplacian Green-function family with alpha≈{alpha:.3g}. This is a non-local spatial operator, not an ordinary local central-force mechanism: its physical signature is an interaction that decays more slowly than the standard 2D Laplacian reference, with enhanced long-range coupling and non-local propagation"
                    + (
                        f". The control-normalized global radial coefficient is C≈{radial_power_coefficient:.4g}; do not confuse it with the smaller local acceleration C/r^q at a sampled radius"
                        if radial_power_coefficient is not None
                        else ""
                    )
                )
            if p2_exponent is not None and p2_exponent > 0.4:
                sign_ratios = [
                    float(record["signed_response_ratio"])
                    for record in p1_sign_evidence + p2_sign_evidence
                    if isinstance(record.get("signed_response_ratio"), (int, float))
                    and math.isfinite(float(record["signed_response_ratio"]))
                ]
                sign_invariant = bool(
                    sign_ratios and float(np.median(sign_ratios)) > 0.5
                )
                polarity_statement = (
                    " Sign flips of the exposed property values leave the attractive direction invariant, implying latent charges of fixed opposite polarity while p1 and p2 encode their magnitudes; the interaction is therefore always attractive."
                    if sign_invariant
                    else " The observed sign tests determine whether the signed product gives attraction or repulsion."
                )
                correspondences.append(
                    f"both scalar properties enter multiplicatively rather than as response inertia. With q≈{radius_exponent:.3g}, the controlled law is the two-charge Coulomb-like central force |a| proportional to |p1*p2|/r^2, with particle 1 pinned at the origin."
                    + (
                        f" Dividing out |p1*p2| and multiplying the local response by r^q gives the global coupling C≈{radial_power_coefficient:.4g}; use that normalized coefficient in executable code and make its fit bounds straddle it, rather than using the acceleration observed at one non-unit radius."
                        if radial_power_coefficient is not None
                        else ""
                    )
                    + polarity_statement
                )
            if p2_exponent is None:
                correspondences.append(
                    f"q≈{radius_exponent:.3g} has the 2D Riesz/fractional-Laplacian correspondence alpha=(3-q)/2≈{alpha:.3g}, but a p2 intervention is still needed to distinguish an inertial field response from a multiplicative two-charge central force; if the budget is exhausted, report both interpretations and this caveat"
                )
        if time_sign_change:
            if (
                absolute_time_fit is not None
                and absolute_time_fit["relative_rmse"] < 0.15
            ):
                aliases = _as_list(absolute_time_fit.get("frequency_aliases"))
                alias_selection = (
                    self.time_alias_selection
                    if isinstance(self.time_alias_selection, dict)
                    else None
                )
                if aliases and alias_selection:
                    correspondences.append(
                        f"the same controlled state changes force sign with absolute start_time. Three equally spaced phases initially left an odd-harmonic frequency alias family; causal-compatible replay on the already collected full trajectories selected f(t)=offset+A*cos(omega*t+phase), with offset≈{absolute_time_fit['offset']:.4g}, A≈{absolute_time_fit['amplitude']:.4g}, omega≈{float(alias_selection['angular_frequency_omega']):.4g}, phase≈{float(alias_selection['phase']):.4g}, period≈{float(alias_selection['period']):.4g}, and training loss≈{float(alias_selection['training_loss']):.4g}. This alias selection used no new simulator episode. Preserve the selected absolute-time sinusoid in the executable integration and use start_time+t, not elapsed t alone"
                    )
                elif aliases:
                    alias_text = ", ".join(
                        (
                            f"{float(item['angular_frequency_omega']):.4g}"
                            for item in aliases
                            if isinstance(item, dict)
                            and isinstance(
                                item.get("angular_frequency_omega"), (int, float)
                            )
                        )
                    )
                    correspondences.append(
                        f"the same controlled state changes force sign with absolute start_time. After dividing out the identified scalar controls and radial power, the three-phase response identifies a sinusoidal coupling but not a unique period: the odd-harmonic omega aliases are [{alias_text}]. Preserve these causal alternatives until a non-commensurate fourth phase or replay on the already collected full trajectories distinguishes them; do not report the lowest-frequency alias as a uniquely identified mechanism"
                    )
                else:
                    correspondences.append(
                        f"the same controlled state changes force sign with absolute start_time. After dividing out the identified scalar controls and radial power, the global sinusoidal coupling is quantitatively identified: f(t)=offset+A*cos(omega*t+phase), with offset≈{absolute_time_fit['offset']:.4g}, A≈{absolute_time_fit['amplitude']:.4g}, omega≈{absolute_time_fit['angular_frequency_omega']:.4g}, phase≈{absolute_time_fit['phase']:.4g}, period≈{absolute_time_fit['period']:.4g}, and relative RMSE {absolute_time_fit['relative_rmse']:.3g}. Preserve this absolute-time sinusoid in the executable integration; use start_time+t, not elapsed t alone"
                    )
            else:
                correspondences.append(
                    "the same controlled state changes force sign with absolute start_time; retain an explicitly phase-modulated coupling and collect at least four phases to fit its period instead of averaging it into a static power law"
                )
        controlled_scaling = {
            "p1_power_exponent_median": p1_exponent,
            "p1_power_estimates": p1_power[-8:],
            "p2_power_exponent_median": p2_exponent,
            "p2_power_estimates": p2_power[-8:],
            "radius_decay_exponent_median": radius_exponent,
            "radial_power_coefficient_median": radial_power_coefficient,
            "radius_local_decay": sorted_slopes[-12:],
            "near_range_decay_q": near_q,
            "far_range_decay_q": far_q,
            "p1_sign_relations": p1_sign_evidence[-6:],
            "p2_sign_relations": p2_sign_evidence[-6:],
            "absolute_time_response": normalized_absolute_time_response[:4],
            "raw_absolute_time_response": absolute_time_response[:4],
        }
        active_responses = [
            record for record in responses if abs(float(record["p1"])) > 1e-12
        ]
        return {
            "method": "common-noise paired velocity contrasts at the earliest positive time, plus exact single-cadence velocity responses when present, divided by elapsed time; least-squares response graph anchored by observed p1=0 source-null interventions",
            "paired_edge_count": len(paired_edges),
            "direct_single_cadence_edge_count": len(direct_edges),
            "interpretation_guardrail": "Use these noise-cancelled ratios for structural identification. A raw trajectory MSE fit is only nuisance calibration and must not override controlled input exponents, local slopes, sign tests, or absolute-time responses.",
            "edge_count": len(edges),
            "anchored_response_count": len(active_responses),
            "graph_residual_rmse": residual_rmse,
            "responses": active_responses[-24:],
            "controlled_scaling": _jsonable(controlled_scaling),
            "mechanism_fits": {
                "compactified_image_sum": _jsonable(compact_image_fit),
                "screened_helmholtz": _jsonable(screened_helmholtz_fit),
                "absolute_time_sinusoid": _jsonable(absolute_time_fit),
                "ordered_radius_change_point": _jsonable(ordered_radius_change_point),
            },
            "candidate_correspondences": correspondences,
        }

    def executable_causal_check(
        self,
        law_source: str,
        fitted_params: Optional[dict] = None,
        *,
        probe_duration: float = 0.02,
    ) -> dict:
        audit = self._noise_cancelled_causal_audit() or {}
        responses = [
            record
            for record in _as_list(audit.get("responses"))
            if isinstance(record, dict)
            and self._two_particle_controls(record) is not None
            and isinstance(record.get("acceleration"), (list, tuple))
        ]
        observed_norms = []
        for record in responses:
            try:
                acceleration = np.asarray(record["acceleration"], dtype=float)
            except (TypeError, ValueError):
                continue
            if (
                acceleration.shape == (2,)
                and np.isfinite(acceleration).all()
                and (float(np.linalg.norm(acceleration)) > 1e-09)
            ):
                observed_norms.append(float(np.linalg.norm(acceleration)))
        if not observed_norms:
            return {
                "available": False,
                "passed": True,
                "reason": "no reliable null-anchored two-particle responses",
                "case_count": 0,
            }
        typical_response = float(np.median(observed_norms))
        residual = audit.get("graph_residual_rmse")
        residual_ratio = (
            float(residual) / max(typical_response, 1e-12)
            if isinstance(residual, (int, float)) and math.isfinite(float(residual))
            else None
        )
        if residual_ratio is not None and residual_ratio > 0.25:
            return {
                "available": False,
                "passed": True,
                "reason": "paired response graph is too internally inconsistent for an executable normalization check",
                "graph_residual_relative_to_response": residual_ratio,
                "case_count": 0,
            }
        try:
            from scienceagent.evaluator import _compile_law

            law = _compile_law(law_source)
            signature = inspect.signature(law)
        except Exception as exc:
            return {
                "available": True,
                "passed": False,
                "reason": f"submitted law did not compile: {exc}",
                "case_count": 0,
            }
        parameters = {
            str(name): float(value)
            for name, value in (fitted_params or {}).items()
            if isinstance(value, (int, float))
            and (not isinstance(value, bool))
            and math.isfinite(float(value))
        }
        accepts_kwargs = any(
            (
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
        )
        if not accepts_kwargs:
            parameters = {
                name: value
                for name, value in parameters.items()
                if name in signature.parameters
            }
        duration = max(float(probe_duration), 0.001)
        case_rows = []
        for record in responses:
            try:
                observed = np.asarray(record["acceleration"], dtype=float)
                velocity = np.asarray(record["velocity2"], dtype=float)
                kwargs = dict(parameters)
                if "start_time" in signature.parameters or accepts_kwargs:
                    kwargs.setdefault(
                        "start_time", float(record.get("start_time", 0.0))
                    )
                output = law(
                    [0.0, 0.0],
                    list(record["pos2"]),
                    float(record["p1"]),
                    float(record["p2"]),
                    velocity.tolist(),
                    duration,
                    **kwargs,
                )
                predicted_velocity = np.asarray(output[1], dtype=float)
                predicted = (predicted_velocity - velocity) / duration
            except Exception as exc:
                case_rows.append(
                    {
                        "controls": {
                            key: copy.deepcopy(record.get(key))
                            for key in ("p1", "p2", "pos2", "start_time")
                        },
                        "error": str(exc),
                    }
                )
                continue
            observed_norm = float(np.linalg.norm(observed))
            if (
                observed.shape != (2,)
                or predicted.shape != (2,)
                or (not np.isfinite(observed).all())
                or (not np.isfinite(predicted).all())
                or (observed_norm <= 1e-09)
            ):
                continue
            relative_error = float(np.linalg.norm(predicted - observed) / observed_norm)
            case_rows.append(
                {
                    "controls": {
                        key: copy.deepcopy(record.get(key))
                        for key in ("p1", "p2", "pos2", "start_time")
                    },
                    "observed_acceleration": observed.tolist(),
                    "predicted_acceleration": predicted.tolist(),
                    "relative_vector_error": relative_error,
                    "magnitude_ratio": float(np.linalg.norm(predicted) / observed_norm),
                }
            )
        valid_errors = [
            float(row["relative_vector_error"])
            for row in case_rows
            if isinstance(row.get("relative_vector_error"), (int, float))
            and math.isfinite(float(row["relative_vector_error"]))
        ]
        if not valid_errors:
            return {
                "available": True,
                "passed": False,
                "reason": "submitted law produced no valid short-time checks",
                "case_count": len(case_rows),
                "cases": case_rows[:6],
            }
        median_error = float(np.median(valid_errors))
        p80_error = float(np.percentile(valid_errors, 80))
        errors_by_radius: dict[float, list[float]] = {}
        for row in case_rows:
            controls = row.get("controls") or {}
            pos2 = controls.get("pos2")
            error = row.get("relative_vector_error")
            if not (
                isinstance(pos2, (list, tuple))
                and len(pos2) == 2
                and isinstance(error, (int, float))
                and math.isfinite(float(error))
            ):
                continue
            try:
                radius = float(np.linalg.norm(np.asarray(pos2, dtype=float)))
            except (TypeError, ValueError):
                continue
            if math.isfinite(radius) and radius > 0.0:
                errors_by_radius.setdefault(round(radius, 12), []).append(float(error))
        per_radius_error = {
            str(radius): float(np.median(values))
            for radius, values in sorted(errors_by_radius.items())
        }
        max_radius_median_error = (
            max(per_radius_error.values()) if per_radius_error else None
        )
        mechanism_fits = audit.get("mechanism_fits") or {}
        compact_fit = mechanism_fits.get("compactified_image_sum")
        screened_fit = mechanism_fits.get("screened_helmholtz")
        strongly_identified_mechanism = bool(
            isinstance(compact_fit, dict)
            and isinstance(compact_fit.get("relative_log_rmse"), (int, float))
            and (float(compact_fit["relative_log_rmse"]) < 0.12)
            or (
                isinstance(screened_fit, dict)
                and isinstance(screened_fit.get("relative_log_rmse"), (int, float))
                and (float(screened_fit["relative_log_rmse"]) < 0.12)
            )
        )
        mechanism_consistent = bool(
            not strongly_identified_mechanism
            or max_radius_median_error is None
            or max_radius_median_error <= 0.3
        )
        passed = bool(
            median_error <= 0.35 and p80_error <= 0.75 and mechanism_consistent
        )
        worst = sorted(
            (
                row
                for row in case_rows
                if isinstance(row.get("relative_vector_error"), (int, float))
            ),
            key=lambda row: float(row["relative_vector_error"]),
            reverse=True,
        )[:6]
        return {
            "available": True,
            "passed": passed,
            "reason": "short-time executable acceleration agrees with collected null-anchored causal responses"
            if passed
            else "submitted code has the wrong sign, control scaling, radial normalization, time response, vector conversion, or misses an identified mechanism at one causal scale",
            "probe_duration": duration,
            "fitted_params_used": parameters,
            "graph_residual_relative_to_response": residual_ratio,
            "case_count": len(valid_errors),
            "median_relative_vector_error": median_error,
            "p80_relative_vector_error": p80_error,
            "per_radius_median_relative_vector_error": per_radius_error,
            "max_radius_median_relative_vector_error": max_radius_median_error,
            "strongly_identified_mechanism": strongly_identified_mechanism,
            "worst_cases": _jsonable(worst),
        }

    def executable_semigroup_check(
        self,
        law_source: str,
        fitted_params: Optional[dict] = None,
        *,
        total_duration: float = 0.5,
        relative_tolerance: float = 0.001,
    ) -> dict:
        public_cases: list[dict] = []
        seen_radii: set[float] = set()
        for raw_case in self._executed_input_cases():
            controls = self._two_particle_controls(raw_case)
            if controls is None or abs(float(controls["p2"])) <= 1e-12:
                continue
            position = np.asarray(controls["pos2"], dtype=float)
            radius = float(np.linalg.norm(position))
            if not math.isfinite(radius) or radius <= 0.25:
                continue
            radius_key = round(radius, 8)
            if radius_key in seen_radii:
                continue
            seen_radii.add(radius_key)
            public_cases.append(controls)
        public_cases.sort(
            key=lambda case: float(
                np.linalg.norm(np.asarray(case["pos2"], dtype=float))
            )
        )
        public_cases = public_cases[:3]
        if not public_cases:
            return {
                "available": False,
                "passed": True,
                "reason": "no public two-particle states for a semigroup audit",
                "case_count": 0,
                "simulator_episodes_spent": 0,
            }
        try:
            from scienceagent.evaluator import _compile_law

            law = _compile_law(law_source)
            signature = inspect.signature(law)
        except Exception as exc:
            return {
                "available": True,
                "passed": False,
                "reason": f"submitted law did not compile: {exc}",
                "case_count": 0,
                "simulator_episodes_spent": 0,
            }
        parameters = {
            str(name): float(value)
            for name, value in (fitted_params or {}).items()
            if isinstance(value, (int, float))
            and (not isinstance(value, bool))
            and math.isfinite(float(value))
        }
        accepts_kwargs = any(
            (
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
        )
        if not accepts_kwargs:
            parameters = {
                name: value
                for name, value in parameters.items()
                if name in signature.parameters
            }
        duration = max(float(total_duration), 0.05)
        tolerance = max(float(relative_tolerance), 0.0)
        rows: list[dict] = []

        def output_state(output: Any) -> np.ndarray:
            if not isinstance(output, (list, tuple)) or len(output) != 2:
                raise ValueError("law output must contain position and velocity")
            position = np.asarray(output[0], dtype=float)
            velocity = np.asarray(output[1], dtype=float)
            if (
                position.shape != (2,)
                or velocity.shape != (2,)
                or (not np.isfinite(position).all())
                or (not np.isfinite(velocity).all())
            ):
                raise ValueError("law output state is non-finite or malformed")
            return np.concatenate([position, velocity])

        for controls in public_cases:
            position = np.asarray(controls["pos2"], dtype=float)
            radius = float(np.linalg.norm(position))
            tangent = np.asarray([-position[1], position[0]], dtype=float) / radius
            observed_velocity = np.asarray(controls["velocity2"], dtype=float)
            tangential_speed = max(
                0.5, float(np.linalg.norm(observed_velocity)), 0.5 * radius
            )
            audit_velocity = observed_velocity + tangential_speed * tangent
            start_time = float(controls.get("start_time", 0.0))

            def call(pos: np.ndarray, vel: np.ndarray, span: float, clock: float):
                kwargs = dict(parameters)
                if "start_time" in signature.parameters or accepts_kwargs:
                    kwargs["start_time"] = clock
                return law(
                    [0.0, 0.0],
                    pos.tolist(),
                    float(controls["p1"]),
                    float(controls["p2"]),
                    vel.tolist(),
                    span,
                    **kwargs,
                )

            try:
                full = output_state(
                    call(position, audit_velocity, duration, start_time)
                )
                first_half = output_state(
                    call(position, audit_velocity, 0.5 * duration, start_time)
                )
                second_half = output_state(
                    call(
                        first_half[:2],
                        first_half[2:],
                        0.5 * duration,
                        start_time + 0.5 * duration,
                    )
                )
                initial = np.concatenate([position, audit_velocity])
                absolute_gap = float(np.linalg.norm(full - second_half))
                transition_scale = max(float(np.linalg.norm(full - initial)), 0.1)
                relative_gap = absolute_gap / transition_scale
                rows.append(
                    {
                        "radius": radius,
                        "total_duration": duration,
                        "synthetic_tangential_speed": tangential_speed,
                        "absolute_state_gap": absolute_gap,
                        "relative_state_gap": relative_gap,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "radius": radius,
                        "total_duration": duration,
                        "synthetic_tangential_speed": tangential_speed,
                        "error": str(exc),
                    }
                )
        valid_gaps = [
            float(row["relative_state_gap"])
            for row in rows
            if isinstance(row.get("relative_state_gap"), (int, float))
            and math.isfinite(float(row["relative_state_gap"]))
        ]
        passed = bool(
            len(valid_gaps) == len(public_cases)
            and valid_gaps
            and (max(valid_gaps) <= tolerance)
        )
        return {
            "available": True,
            "passed": passed,
            "reason": "full-step and composed half-step transitions agree"
            if passed
            else "state-transition composition failed; the executable may freeze initial state, mishandle absolute time, or integrate too coarsely",
            "case_count": len(public_cases),
            "relative_tolerance": tolerance,
            "max_relative_state_gap": max(valid_gaps) if valid_gaps else None,
            "cases": _jsonable(rows),
            "public_data_only": True,
            "simulator_episodes_spent": 0,
        }

    def deductive_compactified_law(self) -> Optional[dict]:
        audit = self._noise_cancelled_causal_audit() or {}
        scaling = audit.get("controlled_scaling") or {}
        fit = (audit.get("mechanism_fits") or {}).get("compactified_image_sum")
        p2_exponent = scaling.get("p2_power_exponent_median")
        if not (
            isinstance(fit, dict)
            and isinstance(fit.get("relative_log_rmse"), (int, float))
            and (float(fit["relative_log_rmse"]) < 0.12)
            and isinstance(p2_exponent, (int, float))
            and math.isfinite(float(p2_exponent))
            and (float(p2_exponent) < -0.4)
        ):
            return None
        field_scale = float(fit["field_scale_G"])
        compact_radius = float(fit["compactification_radius_R"])
        if not (
            field_scale > 1e-10
            and compact_radius > 1e-10
            and math.isfinite(field_scale)
            and math.isfinite(compact_radius)
        ):
            return None
        scale_bounds = [0.5 * field_scale, 1.5 * field_scale]
        radius_bounds = [0.5 * compact_radius, 1.5 * compact_radius]
        source = textwrap.dedent(
            f'\n            import numpy as np\n            from scipy.integrate import solve_ivp\n\n            def discovered_law(\n                pos1, pos2, p1, p2, velocity2, duration,\n                G={field_scale!r}, R={compact_radius!r}, **params\n            ):\n                x0, y0 = float(pos2[0]), float(pos2[1])\n                vx0, vy0 = float(velocity2[0]), float(velocity2[1])\n                source_x, source_y = float(pos1[0]), float(pos1[1])\n                p2_eff = (\n                    float(p2)\n                    if abs(float(p2)) > 1e-12\n                    else (1e-12 if float(p2) >= 0.0 else -1e-12)\n                )\n                circumference = 2.0 * np.pi * float(R)\n                images = np.arange(-128, 129, dtype=float)\n                prefactor = (\n                    float(G) * circumference / (4.0 * np.pi)\n                    * (float(p1) / p2_eff)\n                )\n\n                def rhs(t, state):\n                    x, y, vx, vy = state\n                    rx = x - source_x\n                    ry = y - source_y\n                    radius_sq = max(rx * rx + ry * ry, 1e-12)\n                    denominators = (\n                        radius_sq + (images * circumference) ** 2\n                    ) ** 1.5\n                    factor = -prefactor * float(\n                        np.sum(1.0 / denominators)\n                    )\n                    return [vx, vy, factor * rx, factor * ry]\n\n                if float(duration) <= 0.0:\n                    return [x0, y0], [vx0, vy0]\n                solution = solve_ivp(\n                    rhs,\n                    (0.0, float(duration)),\n                    [x0, y0, vx0, vy0],\n                    method="RK45",\n                    rtol=1e-7,\n                    atol=1e-9,\n                )\n                if not solution.success:\n                    raise RuntimeError("integration_failed")\n                xf, yf, vxf, vyf = solution.y[:, -1]\n                return [float(xf), float(yf)], [float(vxf), float(vyf)]\n\n            def fit_parameters():\n                return {{\n                    "G": {{\n                        "init": {field_scale!r},\n                        "bounds": {scale_bounds!r}\n                    }},\n                    "R": {{\n                        "init": {compact_radius!r},\n                        "bounds": {radius_bounds!r}\n                    }}\n                }}\n            '
        ).strip()
        return {
            "source": source,
            "deduction": f"canonical Kaluza-Klein image-sum central law with one compact dimension: G={field_scale:.4g}, R={compact_radius:.4g}, short-range r^-2 and long-range r^-1",
            "audit_fit": copy.deepcopy(fit),
        }

    def deductive_screened_helmholtz_law(self) -> Optional[dict]:
        audit = self._noise_cancelled_causal_audit() or {}
        scaling = audit.get("controlled_scaling") or {}
        fit = (audit.get("mechanism_fits") or {}).get("screened_helmholtz")
        p2_exponent = scaling.get("p2_power_exponent_median")
        if not (
            isinstance(fit, dict)
            and isinstance(fit.get("relative_log_rmse"), (int, float))
            and (float(fit["relative_log_rmse"]) < 0.12)
            and isinstance(p2_exponent, (int, float))
            and math.isfinite(float(p2_exponent))
            and (float(p2_exponent) < -0.4)
        ):
            return None
        field_scale = float(fit["field_scale_G"])
        screening_length = float(fit["screening_length_lambda"])
        if not (
            field_scale > 1e-10
            and screening_length > 1e-10
            and math.isfinite(field_scale)
            and math.isfinite(screening_length)
        ):
            return None
        scale_bounds = [0.5 * field_scale, 1.5 * field_scale]
        length_bounds = [0.5 * screening_length, 1.5 * screening_length]
        source = textwrap.dedent(
            f'\n            import numpy as np\n            from scipy.integrate import solve_ivp\n            from scipy.special import k1\n\n            def discovered_law(\n                pos1, pos2, p1, p2, velocity2, duration,\n                G={field_scale!r}, screening_length={screening_length!r}, **params\n            ):\n                x0, y0 = float(pos2[0]), float(pos2[1])\n                vx0, vy0 = float(velocity2[0]), float(velocity2[1])\n                source_x, source_y = float(pos1[0]), float(pos1[1])\n                p2_eff = (\n                    float(p2)\n                    if abs(float(p2)) > 1e-12\n                    else (1e-12 if float(p2) >= 0.0 else -1e-12)\n                )\n\n                def rhs(t, state):\n                    x, y, vx, vy = state\n                    rx = x - source_x\n                    ry = y - source_y\n                    radius = max(float(np.hypot(rx, ry)), 1e-8)\n                    length = max(float(screening_length), 1e-8)\n                    magnitude = (\n                        float(G) * (float(p1) / p2_eff)\n                        * float(k1(radius / length))\n                        / (2.0 * np.pi * length)\n                    )\n                    factor = -magnitude / radius\n                    return [vx, vy, factor * rx, factor * ry]\n\n                if float(duration) <= 0.0:\n                    return [x0, y0], [vx0, vy0]\n                solution = solve_ivp(\n                    rhs,\n                    (0.0, float(duration)),\n                    [x0, y0, vx0, vy0],\n                    method="RK45",\n                    rtol=1e-7,\n                    atol=1e-9,\n                )\n                if not solution.success:\n                    raise RuntimeError("integration_failed")\n                xf, yf, vxf, vyf = solution.y[:, -1]\n                return [float(xf), float(yf)], [float(vxf), float(vyf)]\n\n            def fit_parameters():\n                return {{\n                    "G": {{\n                        "init": {field_scale!r},\n                        "bounds": {scale_bounds!r}\n                    }},\n                    "screening_length": {{\n                        "init": {screening_length!r},\n                        "bounds": {length_bounds!r}\n                    }}\n                }}\n            '
        ).strip()
        return {
            "source": source,
            "deduction": f"canonical two-dimensional screened Helmholtz K1 gradient: G={field_scale:.4g}, screening_length={screening_length:.4g}",
            "audit_fit": copy.deepcopy(fit),
        }

    def deductive_central_power_law(self) -> Optional[dict]:
        audit = self._noise_cancelled_causal_audit() or {}
        scaling = audit.get("controlled_scaling") or {}
        p1_exponent = scaling.get("p1_power_exponent_median")
        p2_exponent = scaling.get("p2_power_exponent_median")
        exponent = scaling.get("radius_decay_exponent_median")
        coefficient = scaling.get("radial_power_coefficient_median")
        if not all(
            (
                isinstance(value, (int, float))
                and (not isinstance(value, bool))
                and math.isfinite(float(value))
                for value in (p1_exponent, p2_exponent, exponent, coefficient)
            )
        ):
            return None
        p1_exponent = float(p1_exponent)
        p2_exponent = float(p2_exponent)
        exponent = float(exponent)
        coefficient = float(coefficient)
        if not (
            0.5 <= p1_exponent <= 1.5
            and 0.25 <= exponent <= 4.0
            and (coefficient > 1e-10)
            and (p2_exponent < -0.4 or p2_exponent > 0.4)
        ):
            return None
        sign_rows = [
            row
            for key in ("p1_sign_relations", "p2_sign_relations")
            for row in scaling.get(key) or []
            if isinstance(row, dict)
            and isinstance(row.get("signed_response_ratio"), (int, float))
            and math.isfinite(float(row["signed_response_ratio"]))
        ]
        sign_invariant = bool(
            sign_rows
            and float(
                np.median([float(row["signed_response_ratio"]) for row in sign_rows])
            )
            > 0.5
        )
        if p2_exponent < -0.4:
            control_expression = "float(p1) / p2_eff"
            control_role = "signed source p1 divided by response inertia p2"
        elif sign_invariant:
            control_expression = "abs(float(p1) * float(p2))"
            control_role = "sign-invariant product of two exposed magnitudes"
        else:
            control_expression = "float(p1) * float(p2)"
            control_role = "signed multiplicative pair control"
        coefficient_bounds = [0.5 * coefficient, 1.5 * coefficient]
        exponent_bounds = [max(0.1, exponent - 0.3), min(4.5, exponent + 0.3)]
        if p2_exponent > 0.4:
            source = textwrap.dedent(
                f'\n                import numpy as np\n                from scipy.integrate import solve_ivp\n\n                def discovered_law(\n                    pos1, pos2, p1, p2, velocity2, duration,\n                    C={coefficient!r}, q={exponent!r}, **params\n                ):\n                    initial = np.asarray(\n                        [pos2[0], pos2[1], velocity2[0], velocity2[1]],\n                        dtype=float,\n                    )\n                    source_position = np.asarray(pos1, dtype=float)\n                    p2_eff = (\n                        float(p2)\n                        if abs(float(p2)) > 1e-12\n                        else (1e-12 if float(p2) >= 0.0 else -1e-12)\n                    )\n                    control = {control_expression}\n                    span = float(duration)\n                    if span <= 0.0:\n                        return initial[:2].tolist(), initial[2:].tolist()\n                    softening = 0.05\n\n                    def rhs(t, state):\n                        displacement = np.asarray(state[:2]) - source_position\n                        radius_sq = float(np.dot(displacement, displacement))\n                        effective_radius = float(\n                            np.sqrt(radius_sq + softening * softening)\n                        )\n                        acceleration = (\n                            -float(C) * control\n                            * displacement\n                            / effective_radius ** (float(q) + 1.0)\n                        )\n                        return [\n                            state[2], state[3],\n                            acceleration[0], acceleration[1],\n                        ]\n\n                    solution = solve_ivp(\n                        rhs,\n                        (0.0, span),\n                        initial,\n                        method="RK45",\n                        rtol=1e-8,\n                        atol=1e-10,\n                        max_step=0.01,\n                    )\n                    if not solution.success:\n                        raise RuntimeError("integration_failed")\n                    final = solution.y[:, -1]\n                    return final[:2].tolist(), final[2:].tolist()\n\n                def fit_parameters():\n                    return {{\n                        "C": {{\n                            "init": {coefficient!r},\n                            "bounds": {coefficient_bounds!r}\n                        }},\n                        "q": {{\n                            "init": {exponent!r},\n                            "bounds": {exponent_bounds!r}\n                        }}\n                    }}\n                '
            ).strip()
        else:
            source = textwrap.dedent(
                f'\n            import numpy as np\n            from scipy.integrate import solve_ivp\n\n            def discovered_law(\n                pos1, pos2, p1, p2, velocity2, duration,\n                C={coefficient!r}, q={exponent!r}, **params\n            ):\n                x0, y0 = float(pos2[0]), float(pos2[1])\n                vx0, vy0 = float(velocity2[0]), float(velocity2[1])\n                source_x, source_y = float(pos1[0]), float(pos1[1])\n                p2_eff = (\n                    float(p2)\n                    if abs(float(p2)) > 1e-12\n                    else (1e-12 if float(p2) >= 0.0 else -1e-12)\n                )\n                control = {control_expression}\n\n                def rhs(t, state):\n                    x, y, vx, vy = state\n                    rx = x - source_x\n                    ry = y - source_y\n                    radius = max(float(np.hypot(rx, ry)), 1e-8)\n                    factor = -float(C) * control / radius ** (float(q) + 1.0)\n                    return [vx, vy, factor * rx, factor * ry]\n\n                if float(duration) <= 0.0:\n                    return [x0, y0], [vx0, vy0]\n                solution = solve_ivp(\n                    rhs,\n                    (0.0, float(duration)),\n                    [x0, y0, vx0, vy0],\n                    method="RK45",\n                    rtol=1e-7,\n                    atol=1e-9,\n                )\n                if not solution.success:\n                    raise RuntimeError("integration_failed")\n                xf, yf, vxf, vyf = solution.y[:, -1]\n                return [float(xf), float(yf)], [float(vxf), float(vyf)]\n\n            def fit_parameters():\n                return {{\n                    "C": {{\n                        "init": {coefficient!r},\n                        "bounds": {coefficient_bounds!r}\n                    }},\n                    "q": {{\n                        "init": {exponent!r},\n                        "bounds": {exponent_bounds!r}\n                    }}\n                }}\n            '
            ).strip()
        return {
            "source": source,
            "deduction": f"central scalar magnitude C/r^q with C={coefficient:.4g}, q={exponent:.4g}, and {control_role}",
            "control_role": control_role,
            "audit_scaling": copy.deepcopy(scaling),
        }

    def deductive_causal_basis_laws(self) -> list[dict]:
        candidates: list[dict] = []

        def append(compiler: str, deduction: Optional[dict], priority: int) -> None:
            if not (
                isinstance(deduction, dict) and isinstance(deduction.get("source"), str)
            ):
                return
            candidates.append(
                {
                    **copy.deepcopy(deduction),
                    "compiler": compiler,
                    "structural_priority": int(priority),
                }
            )

        append("deductive_compactified_law", self.deductive_compactified_law(), 3)
        append(
            "deductive_screened_helmholtz_law",
            self.deductive_screened_helmholtz_law(),
            3,
        )
        for deduction in self.deductive_time_modulated_laws():
            append("deductive_time_modulated_laws", deduction, 3)
        append("deductive_central_power_law", self.deductive_central_power_law(), 1)
        return candidates

    def _deductive_time_modulated_law(
        self,
        *,
        omega_override: Optional[float] = None,
        phase_override: Optional[float] = None,
        alias_index: Optional[int] = None,
    ) -> Optional[dict]:
        audit = self._noise_cancelled_causal_audit() or {}
        scaling = audit.get("controlled_scaling") or {}
        fits = audit.get("mechanism_fits") or {}
        time_fit = fits.get("absolute_time_sinusoid")
        q = scaling.get("radius_decay_exponent_median")
        p2_exponent = scaling.get("p2_power_exponent_median")
        if not (
            isinstance(time_fit, dict)
            and isinstance(q, (int, float))
            and math.isfinite(float(q))
            and (0.2 <= float(q) <= 4.0)
            and isinstance(p2_exponent, (int, float))
            and math.isfinite(float(p2_exponent))
            and isinstance(time_fit.get("relative_rmse"), (int, float))
            and (float(time_fit["relative_rmse"]) < 0.2)
        ):
            return None
        offset = float(time_fit["offset"])
        amplitude = float(time_fit["amplitude"])
        omega = float(
            omega_override
            if omega_override is not None
            else time_fit["angular_frequency_omega"]
        )
        phase = float(
            phase_override if phase_override is not None else time_fit["phase"]
        )
        if not (
            amplitude > 1e-10
            and omega > 1e-10
            and all(
                (math.isfinite(value) for value in (offset, amplitude, omega, phase))
            )
        ):
            return None
        if float(p2_exponent) < -0.4:
            control_expression = "float(p1) / p2_eff"
            control_role = "signed p1/p2 source-over-inertia response"
        elif float(p2_exponent) > 0.4:
            sign_ratios = [
                float(record["signed_response_ratio"])
                for record in _as_list(scaling.get("p1_sign_relations"))
                + _as_list(scaling.get("p2_sign_relations"))
                if isinstance(record, dict)
                and isinstance(record.get("signed_response_ratio"), (int, float))
                and math.isfinite(float(record["signed_response_ratio"]))
            ]
            sign_invariant = bool(sign_ratios and float(np.median(sign_ratios)) > 0.5)
            control_expression = (
                "abs(float(p1) * float(p2))"
                if sign_invariant
                else "float(p1) * float(p2)"
            )
            control_role = (
                "magnitude-only p1*p2 pair response"
                if sign_invariant
                else "signed p1*p2 pair response"
            )
        else:
            return None
        q = float(q)
        scale = max(abs(offset), amplitude, 1e-06)
        offset_margin = max(0.1 * amplitude, 0.05 * scale)
        offset_bounds = [offset - offset_margin, offset + offset_margin]
        amplitude_bounds = [max(0.9 * amplitude, 1e-06), 1.1 * amplitude]
        source = textwrap.dedent(
            f'\n            import numpy as np\n            from scipy.integrate import solve_ivp\n\n            def discovered_law(\n                pos1, pos2, p1, p2, velocity2, duration,\n                start_time=0.0, q={q!r}, offset={offset!r},\n                amplitude={amplitude!r}, omega={omega!r},\n                phase={phase!r}, **params\n            ):\n                x0, y0 = float(pos2[0]), float(pos2[1])\n                vx0, vy0 = float(velocity2[0]), float(velocity2[1])\n                p2_eff = (\n                    float(p2)\n                    if abs(float(p2)) > 1e-12\n                    else (1e-12 if float(p2) >= 0.0 else -1e-12)\n                )\n                control = {control_expression}\n\n                def rhs(t, state):\n                    x, y, vx, vy = state\n                    rx = x - float(pos1[0])\n                    ry = y - float(pos1[1])\n                    radius = max(float(np.hypot(rx, ry)), 1e-8)\n                    coupling = offset + amplitude * np.cos(\n                        omega * (float(start_time) + t) + phase\n                    )\n                    factor = -control * coupling / radius ** (q + 1.0)\n                    return [vx, vy, factor * rx, factor * ry]\n\n                if float(duration) <= 0.0:\n                    return [x0, y0], [vx0, vy0]\n                solution = solve_ivp(\n                    rhs,\n                    (0.0, float(duration)),\n                    [x0, y0, vx0, vy0],\n                    method="RK45",\n                    rtol=1e-7,\n                    atol=1e-9,\n                )\n                if not solution.success:\n                    raise RuntimeError("integration_failed")\n                xf, yf, vxf, vyf = solution.y[:, -1]\n                return [float(xf), float(yf)], [float(vxf), float(vyf)]\n\n            def fit_parameters():\n                return {{\n                    "offset": {{\n                        "init": {offset!r},\n                        "bounds": {offset_bounds!r}\n                    }},\n                    "amplitude": {{\n                        "init": {amplitude!r},\n                        "bounds": {amplitude_bounds!r}\n                    }}\n                }}\n            '
        ).strip()
        return {
            "source": source,
            "deduction": f"central magnitude r^-{q:.4g}, {control_role}, global time coupling offset+A*cos(omega*t+phase) with offset={offset:.4g}, A={amplitude:.4g}, omega={omega:.4g}, phase={phase:.4g}",
            "audit_fit": copy.deepcopy(time_fit),
            "time_alias": {
                "alias_index": alias_index,
                "angular_frequency_omega": omega,
                "phase": phase,
                "period": 2.0 * math.pi / omega,
            },
        }

    def deductive_time_modulated_law(self) -> Optional[dict]:
        return self._deductive_time_modulated_law()

    def deductive_time_modulated_laws(self) -> list[dict]:
        audit = self._noise_cancelled_causal_audit() or {}
        time_fit = (audit.get("mechanism_fits") or {}).get("absolute_time_sinusoid")
        if not isinstance(time_fit, dict):
            return []
        aliases = [
            alias
            for alias in _as_list(time_fit.get("frequency_aliases"))
            if isinstance(alias, dict)
            and isinstance(alias.get("angular_frequency_omega"), (int, float))
            and isinstance(alias.get("phase"), (int, float))
        ]
        if not aliases:
            compiled = self._deductive_time_modulated_law()
            return [compiled] if compiled is not None else []
        compiled_aliases = []
        for alias in aliases:
            compiled = self._deductive_time_modulated_law(
                omega_override=float(alias["angular_frequency_omega"]),
                phase_override=float(alias["phase"]),
                alias_index=int(alias["alias_index"])
                if isinstance(alias.get("alias_index"), int)
                else None,
            )
            if compiled is not None:
                compiled_aliases.append(compiled)
        return compiled_aliases

    def _causal_audit_priorities(self) -> list[str]:
        if not self.experiments:
            return []
        active = [
            candidate
            for candidate in self.candidates.values()
            if candidate.get("status") == "active"
        ]
        candidate_text = json.dumps(
            [
                {
                    "name": candidate.get("name"),
                    "description": candidate.get("description"),
                    "mechanisms": candidate.get("mechanisms"),
                    "assumptions": candidate.get("assumptions"),
                    "falsifiers": candidate.get("falsifiers"),
                }
                for candidate in active
            ],
            ensure_ascii=False,
        ).lower()
        coverage = self._intervention_coverage()
        scalar_controls = coverage.get("scalar_controls") or {}
        priorities: list[str] = []
        missing_sign_controls = self._missing_sign_audits()
        if missing_sign_controls:
            priorities.append(
                "complete the mandatory matched sign-invariance audit before coding signed versus magnitude-only coupling: hold the complete state and absolute magnitudes fixed, flip the sign of "
                + " and ".join(missing_sign_controls)
                + ", and compare the null-anchored noise-cancelled response. A response ratio near -1 means the exposed sign is causal; a ratio near +1 means that control supplies magnitude only. Put both p1 and p2 flips in the same counterfactual design when both remain unresolved, and combine them with the null, magnitude, clock, and radius branches when the budget is tight"
            )
        scale_terms = (
            "screen",
            "crossover",
            "multi-scale",
            "multiscale",
            "regime",
            "compact",
            "soften",
            "break",
        )
        radial = coverage.get("pos2_radius")
        if not radial and any(
            (
                term in candidate_text
                for term in (
                    "central",
                    "radial",
                    "radius",
                    "inverse-distance",
                    "inverse-square",
                    "screen",
                )
            )
        ):
            priorities.append(
                "no accepted cross-scale response exists yet: the first short-time matched radius scan must be bidirectional around its factual radius, with at least one smaller and one larger radius (at least three distinct radii spanning 4x). This prevents an outward-only scan from falsely establishing a single far-field power law. Combine it with the null-source and scalar-control branches when the episode budget is tight"
            )
        elif radial and self._needs_radius_scale_audit():
            current_min = float(radial["min"])
            current_max = float(radial["max"])
            priorities.append(
                f"complete the generic cross-scale causal audit before accepting any familiar single-power law: use a short-time matched counterfactual with at least two radii and extend the accumulated anchored response table to at least four distinct radii spanning 8x or more. Include a radius below the current minimum ({current_min:.4g}) or above the current maximum ({current_max:.4g}) as needed, and compare ordered local log-slopes. Combine this radius scan with null-source, p2, and clock branches when the episode budget is tight"
            )
        elif radial and any((term in candidate_text for term in scale_terms)):
            current_min = float(radial["min"])
            current_max = float(radial["max"])
            dynamic_range = current_max / max(current_min, 1e-12)
            causal_audit = self._noise_cancelled_causal_audit() or {}
            local_slopes = (causal_audit.get("controlled_scaling") or {}).get(
                "radius_local_decay"
            ) or []
            slope_values = [
                float(item["decay_exponent_q"])
                for item in local_slopes
                if isinstance(item, dict)
                and isinstance(item.get("decay_exponent_q"), (int, float))
                and math.isfinite(float(item["decay_exponent_q"]))
            ]
            stable_local_power = (
                len(slope_values) >= 2
                and max(slope_values) - min(slope_values) < 0.25
                and (dynamic_range >= 4.0)
            )
            if current_min > 0 and dynamic_range < 32 and stable_local_power:
                priorities.append(
                    f"the sampled local slopes are stable only over the current radius interval; before accepting that familiar asymptote as a global law, use a short-time matched design below the current minimum radius ({current_min:.4g}, target about {current_min / 2:.4g} if legal). A farther-only proposal does not advance the scale frontier"
                )
            elif current_min > 0 and dynamic_range < 32 and (not stable_local_power):
                priorities.append(
                    f"a scale-dependent candidate remains active: use a short-time matched design below the current minimum radius ({current_min:.4g}, target about {current_min / 2:.4g} if legal) and compare its local log-slope with the middle/far regime; stop expanding once the tested radius range spans about 32x"
                )
        if self._needs_start_time_audit():
            causal_audit = self._noise_cancelled_causal_audit() or {}
            time_series = (causal_audit.get("controlled_scaling") or {}).get(
                "absolute_time_response"
            ) or []
            observed_sign_change = any(
                (
                    isinstance(series, list)
                    and series
                    and (
                        min(
                            (
                                float(item["inward_acceleration"])
                                for item in series
                                if isinstance(item, dict)
                                and isinstance(
                                    item.get("inward_acceleration"), (int, float)
                                )
                            )
                        )
                        < -1e-06
                    )
                    and (
                        max(
                            (
                                float(item["inward_acceleration"])
                                for item in series
                                if isinstance(item, dict)
                                and isinstance(
                                    item.get("inward_acceleration"), (int, float)
                                )
                            )
                        )
                        > 1e-06
                    )
                    for series in time_series
                    if any(
                        (
                            isinstance(item, dict)
                            and isinstance(
                                item.get("inward_acceleration"), (int, float)
                            )
                            for item in series
                        )
                    )
                )
            )
            if observed_sign_change:
                priorities.append(
                    "the clock audit has revealed a sign change: collect at least four matched absolute start_time phases and fit a sinusoidal offset, amplitude, angular frequency, phase, and period before finalizing; the executable force must use start_time+t inside integration"
                )
            else:
                priorities.append(
                    "complete the generic absolute-time invariance audit: compare an identical initial state at at least three start_time values spanning a possible sign change before declaring the law static; combine these branches with other pending matched audits when the simulator budget is tight"
                )
        if self._needs_p2_magnitude_audit():
            priorities.append(
                "the probe-property mechanism is not identified: include a matched counterfactual that changes |p2| by at least a factor of two while holding p1, position, velocity, and clock fixed; use the noise-cancelled response ratio to distinguish multiplicative coupling (+1 exponent), inertia (-1), and irrelevance (0)"
            )
        p1_coverage = scalar_controls.get("p1")
        source_terms = ("p1", "source", "central", "radial")
        if (
            p1_coverage
            and (not p1_coverage["has_zero"])
            and any((term in candidate_text for term in source_terms))
        ):
            priorities.append(
                "identify the absolute force direction with a matched null-source branch (set p1=0 if legal): the null-minus-active paired velocity effect cancels observation noise and fixes the sign that the final vector code must use"
            )
        has_fittable_structure = any(
            (bool(candidate.get("parameters")) for candidate in active)
        )
        successful_fit = any(
            (
                attempt.get("error") in (None, "")
                and isinstance(attempt.get("loss_after"), (int, float))
                and math.isfinite(float(attempt["loss_after"]))
                for attempt in self.fit_attempts
            )
        )
        if self.round >= 3 and has_fittable_structure and (not successful_fit):
            if self.fit_attempts:
                priorities.append(
                    "the previous MSE-fit attempt failed or was non-finite: repair the executable candidate and rerun <run_mse_fit> before finalizing"
                )
            else:
                priorities.append(
                    "continuous parameters remain uncalibrated: include <run_mse_fit> for the current executable candidate in this round and inspect loss, fitted values, and numerical failures before submitting the final law"
                )
        return priorities

    def _needs_null_source_audit(self) -> bool:
        if not self.experiments:
            return False
        coverage = self._intervention_coverage()
        p1_coverage = (coverage.get("scalar_controls") or {}).get("p1")
        if not p1_coverage or p1_coverage.get("has_zero"):
            return False
        active = [
            candidate
            for candidate in self.candidates.values()
            if candidate.get("status") == "active"
        ]
        candidate_text = json.dumps(
            [
                {
                    "description": candidate.get("description"),
                    "mechanisms": candidate.get("mechanisms"),
                    "falsifiers": candidate.get("falsifiers"),
                }
                for candidate in active
            ],
            ensure_ascii=False,
        ).lower()
        return any(
            (term in candidate_text for term in ("p1", "source", "central", "radial"))
        )

    def _needs_start_time_audit(self) -> bool:
        if not self.experiments:
            return False
        executed_cases = self._executed_input_cases()
        audit = (
            self._noise_cancelled_causal_audit() or {}
            if self.observation_noise_std > 0.0
            else {}
        )
        has_two_particle_case = any(
            (self._two_particle_controls(case) is not None for case in executed_cases)
        ) or any(
            (
                isinstance(record, dict)
                and all(
                    (field in record for field in ("p1", "p2", "pos2", "velocity2"))
                )
                for record in _as_list(audit.get("responses"))
            )
        )
        if not has_two_particle_case:
            return False
        if self.observation_noise_std > 0.0:
            series_list = (audit.get("controlled_scaling") or {}).get(
                "absolute_time_response"
            ) or []
            time_fit = (audit.get("mechanism_fits") or {}).get("absolute_time_sinusoid")
            for series in series_list:
                if not isinstance(series, list):
                    continue
                valid = [
                    item
                    for item in series
                    if isinstance(item, dict)
                    and isinstance(item.get("start_time"), (int, float))
                    and isinstance(item.get("inward_acceleration"), (int, float))
                ]
                if len({float(item["start_time"]) for item in valid}) < 3:
                    continue
                values = [float(item["inward_acceleration"]) for item in valid]
                sign_change = min(values) < -1e-06 and max(values) > 1e-06
                if not sign_change:
                    return False
                if (
                    isinstance(time_fit, dict)
                    and int(time_fit.get("time_count") or 0) >= 4
                    and isinstance(time_fit.get("relative_rmse"), (int, float))
                    and (float(time_fit["relative_rmse"]) < 0.2)
                ):
                    return False
                if (
                    isinstance(time_fit, dict)
                    and time_fit.get("fit_mode") == "three_phase_antisymmetric"
                    and isinstance(self.time_alias_selection, dict)
                ):
                    return False
            return True
        start_values = {
            float(case.get("start_time", 0.0))
            for case in executed_cases
            if isinstance(case.get("start_time", 0.0), (int, float))
            and (not isinstance(case.get("start_time", 0.0), bool))
        }
        if len(start_values) >= 3:
            return False
        if has_two_particle_case:
            return True
        active = [
            candidate
            for candidate in self.candidates.values()
            if candidate.get("status") == "active"
        ]
        candidate_text = json.dumps(
            [
                {
                    "description": candidate.get("description"),
                    "mechanisms": candidate.get("mechanisms"),
                    "assumptions": candidate.get("assumptions"),
                    "falsifiers": candidate.get("falsifiers"),
                }
                for candidate in active
            ],
            ensure_ascii=False,
        ).lower()
        return any(
            (
                term in candidate_text
                for term in (
                    "time-vary",
                    "time varying",
                    "oscillat",
                    "phase",
                    "nonstation",
                    "absolute time",
                )
            )
        )

    def _needs_p2_magnitude_audit(self) -> bool:
        cases = self._executed_input_cases()
        two_particle_cases = [
            case
            for case in cases
            if isinstance(case.get("p1"), (int, float))
            and (not isinstance(case.get("p1"), bool))
            and isinstance(case.get("p2"), (int, float))
            and (not isinstance(case.get("p2"), bool))
            and isinstance(case.get("pos2"), (list, tuple))
            and isinstance(case.get("velocity2"), (list, tuple))
        ]
        if not two_particle_cases:
            return False
        if self.observation_noise_std > 0.0:
            audit = self._noise_cancelled_causal_audit() or {}
            exponent = (audit.get("controlled_scaling") or {}).get(
                "p2_power_exponent_median"
            )
            return not (
                isinstance(exponent, (int, float)) and math.isfinite(float(exponent))
            )
        magnitudes = {
            round(abs(float(case["p2"])), 12)
            for case in two_particle_cases
            if math.isfinite(float(case["p2"])) and abs(float(case["p2"])) > 1e-12
        }
        return len(magnitudes) < 2

    def _missing_sign_audits(self) -> list[str]:
        cases = self._executed_input_cases()
        if not any((self._two_particle_controls(case) is not None for case in cases)):
            return []
        audit = self._noise_cancelled_causal_audit() or {}
        scaling = audit.get("controlled_scaling") or {}
        missing = []
        for control in ("p1", "p2"):
            relations = scaling.get(f"{control}_sign_relations") or []
            valid = any(
                (
                    isinstance(record, dict)
                    and isinstance(record.get("signed_response_ratio"), (int, float))
                    and (not isinstance(record.get("signed_response_ratio"), bool))
                    and math.isfinite(float(record["signed_response_ratio"]))
                    for record in relations
                )
            )
            if not valid:
                missing.append(control)
        return missing

    def _needs_sign_audit(self) -> bool:
        return bool(self._missing_sign_audits())

    def _needs_radius_scale_audit(self) -> bool:
        audit = self._noise_cancelled_causal_audit() or {}
        responses = [
            record
            for record in _as_list(audit.get("responses"))
            if isinstance(record, dict)
            and isinstance(record.get("radius"), (int, float))
            and math.isfinite(float(record["radius"]))
            and (float(record["radius"]) > 0.0)
        ]
        cases = self._executed_input_cases()
        has_two_particle_case = any(
            (self._two_particle_controls(case) is not None for case in cases)
        ) or any(
            (
                all((field in record for field in ("p1", "p2", "pos2", "velocity2")))
                for record in responses
            )
        )
        if not has_two_particle_case:
            return False
        radii = sorted(
            {
                round(float(record["radius"]), 12)
                for record in responses
                if abs(float(record.get("p1", 0.0))) > 1e-12
            }
        )
        local_slopes = (audit.get("controlled_scaling") or {}).get(
            "radius_local_decay"
        ) or []
        return not (
            len(radii) >= 4
            and radii[-1] / max(radii[0], 1e-12) >= 8.0
            and (len(local_slopes) >= 3)
        )

    def finalization_blockers(self) -> list[str]:
        if not self.experiments:
            return []
        blockers = []
        if self._needs_null_source_audit():
            blockers.append("null-source sign anchor")
        if self._needs_p2_magnitude_audit():
            blockers.append("controlled |p2| response exponent")
        missing_sign_controls = self._missing_sign_audits()
        if missing_sign_controls:
            blockers.append(
                "matched sign invariance for " + " and ".join(missing_sign_controls)
            )
        if self._needs_start_time_audit():
            blockers.append("three-phase absolute-time response")
        if self._needs_radius_scale_audit():
            blockers.append(
                "four-radius noise-cancelled scale scan spanning at least 8x"
            )
        return blockers

    @staticmethod
    def _design_input_cases(design: dict) -> list[dict]:
        if design.get("kind") == "intervention":
            return [
                copy.deepcopy(case)
                for case in _as_list(design.get("experiments"))
                if isinstance(case, dict)
            ]
        factual = design.get("factual")
        if not isinstance(factual, dict):
            return []
        cases = [copy.deepcopy(factual)]
        for intervention in _as_list(design.get("interventions")):
            assignments = (
                intervention.get("set") if isinstance(intervention, dict) else None
            )
            if not isinstance(assignments, dict):
                continue
            branch = copy.deepcopy(factual)
            try:
                for pointer, value in assignments.items():
                    set_json_pointer(branch, pointer, value)
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            cases.append(branch)
        return cases

    @classmethod
    def _design_has_null_source(cls, design: dict) -> bool:
        return any(
            (
                isinstance(case.get("p1"), (int, float))
                and (not isinstance(case.get("p1"), bool))
                and (float(case["p1"]) == 0.0)
                for case in cls._design_input_cases(design)
            )
        )

    @classmethod
    def _design_has_nonzero_start_time(cls, design: dict) -> bool:
        start_times = {
            round(float(case.get("start_time", 0.0)), 12)
            for case in cls._design_input_cases(design)
            if isinstance(case.get("start_time", 0.0), (int, float))
            and (not isinstance(case.get("start_time", 0.0), bool))
            and math.isfinite(float(case.get("start_time", 0.0)))
        }
        return len(start_times) >= 3

    @classmethod
    def _design_has_p2_magnitude_variation(cls, design: dict) -> bool:
        magnitudes = {
            round(abs(float(case["p2"])), 12)
            for case in cls._design_input_cases(design)
            if isinstance(case.get("p2"), (int, float))
            and (not isinstance(case.get("p2"), bool))
            and math.isfinite(float(case["p2"]))
            and (abs(float(case["p2"])) > 1e-12)
        }
        return len(magnitudes) >= 2

    @classmethod
    def _design_sign_variations(cls, design: dict) -> set[str]:
        cases = [
            controls
            for case in cls._design_input_cases(design)
            if (controls := cls._two_particle_controls(case)) is not None
        ]
        varied: set[str] = set()
        for control in ("p1", "p2"):
            for left_index, left in enumerate(cases):
                for right in cases[left_index + 1 :]:
                    left_value = float(left[control])
                    right_value = float(right[control])
                    if left_value * right_value >= 0.0 or not math.isclose(
                        abs(left_value), abs(right_value), rel_tol=0.0, abs_tol=1e-12
                    ):
                        continue
                    if all(
                        (
                            left[field] == right[field]
                            for field in ("p1", "p2", "pos2", "velocity2", "start_time")
                            if field != control
                        )
                    ):
                        varied.add(control)
                        break
                if control in varied:
                    break
        return varied

    @classmethod
    def _design_has_radius_variation(cls, design: dict) -> bool:
        radii = {
            round(float(np.linalg.norm(np.asarray(case["pos2"], dtype=float))), 12)
            for case in cls._design_input_cases(design)
            if isinstance(case.get("pos2"), (list, tuple))
            and len(case["pos2"]) == 2
            and all(
                (
                    isinstance(value, (int, float))
                    and (not isinstance(value, bool))
                    and math.isfinite(float(value))
                    for value in case["pos2"]
                )
            )
            and (float(np.linalg.norm(np.asarray(case["pos2"], dtype=float))) > 0.0)
        }
        return bool(len(radii) >= 2 and max(radii) / max(min(radii), 1e-12) >= 1.8)

    def _design_advances_radius_scale_audit(self, design: dict) -> bool:
        proposed_controls = [
            controls
            for case in self._design_input_cases(design)
            if (controls := self._two_particle_controls(case)) is not None
            and abs(float(controls["p1"])) > 1e-12
        ]
        proposed_radii = sorted(
            {
                round(
                    float(np.linalg.norm(np.asarray(controls["pos2"], dtype=float))), 12
                )
                for controls in proposed_controls
                if float(np.linalg.norm(np.asarray(controls["pos2"], dtype=float)))
                > 0.0
            }
        )
        if len(proposed_radii) < 2:
            return False
        audit = self._noise_cancelled_causal_audit() or {}
        anchored_radii = sorted(
            {
                round(float(record["radius"]), 12)
                for record in _as_list(audit.get("responses"))
                if isinstance(record, dict)
                and isinstance(record.get("radius"), (int, float))
                and math.isfinite(float(record["radius"]))
                and (float(record["radius"]) > 0.0)
                and (abs(float(record.get("p1", 0.0))) > 1e-12)
            }
        )
        if anchored_radii:
            tolerance = 1e-09
            local_slopes = (audit.get("controlled_scaling") or {}).get(
                "radius_local_decay"
            ) or []
            slope_values = [
                float(item["decay_exponent_q"])
                for item in local_slopes
                if isinstance(item, dict)
                and isinstance(item.get("decay_exponent_q"), (int, float))
                and math.isfinite(float(item["decay_exponent_q"]))
            ]
            stable_sampled_power = bool(
                len(slope_values) >= 2 and max(slope_values) - min(slope_values) < 0.25
            )
            if stable_sampled_power:
                return proposed_radii[0] < anchored_radii[0] - tolerance
            return bool(
                proposed_radii[0] < anchored_radii[0] - tolerance
                or proposed_radii[-1] > anchored_radii[-1] + tolerance
            )
        factual = design.get("factual")
        factual_controls = self._two_particle_controls(factual)
        if factual_controls is None:
            reference = proposed_radii[len(proposed_radii) // 2]
        else:
            reference = float(
                np.linalg.norm(np.asarray(factual_controls["pos2"], dtype=float))
            )
        return bool(
            len(proposed_radii) >= 3
            and proposed_radii[-1] / max(proposed_radii[0], 1e-12) >= 4.0
            and (proposed_radii[0] <= reference / 1.5)
            and (proposed_radii[-1] >= reference * 1.5)
        )

    def record_mse_fit(self, round_num: int, result: Any) -> None:
        payload = result if isinstance(result, dict) else {"error": str(result)}
        self.fit_attempts.append(
            {
                "round": int(round_num),
                "loss_before": _jsonable(payload.get("loss_before")),
                "loss_after": _jsonable(payload.get("loss_after")),
                "fitted_params": _jsonable(payload.get("fitted_params") or {}),
                "declared_params": _jsonable(payload.get("declared_params") or {}),
                "n_training": _jsonable(payload.get("n_training")),
                "training_mode": _jsonable(payload.get("training_mode")),
                "error": _jsonable(payload.get("error")),
            }
        )

    def ingest_reply(
        self, reply: str, round_num: int, max_episode_cost: Optional[int] = None
    ) -> dict:
        self.round = int(round_num)
        if max_episode_cost is not None:
            max_episode_cost = max(0, int(max_episode_cost))
        self._pending_selection = None
        self._pending_loose_predictions = []
        errors: list[str] = []
        candidates_before = len(self.candidates)
        progress_before = {
            "abductions": len(self.abductions),
            "revisions": len(self.version_history),
            "inductions": len(self.inductions),
            "deductions": len(self.deductions),
        }
        update, error = _json_tag(reply, "scm_update")
        if error:
            errors.append(error)
        if update is not None:
            errors.extend(self._apply_update(update, round_num))
            self._last_update = _jsonable(update)
            if self.strict:
                if candidates_before == 0 and len(self.candidates) < 2:
                    errors.append(
                        "initial SCM update must create at least two competing candidates"
                    )
                for required_field in (
                    "abduction",
                    "inductions",
                    "deductions",
                    "open_questions",
                ):
                    if required_field not in update:
                        errors.append(
                            f"strict SCM update is missing {required_field!r}"
                        )
        elif self.strict and _extract_tag(reply, "final_law") is None:
            errors.append("missing required <scm_update> block")
        final_claim, error = _json_tag(reply, "scm_final")
        if error:
            errors.append(error)
        if final_claim is not None:
            self.final_claim = _jsonable(final_claim)
            if self.strict:
                errors.extend(self._validate_final_claim(final_claim))
        elif self.strict and _extract_tag(reply, "final_law") is not None:
            errors.append("final submission is missing required <scm_final> block")
        designs_payload, error = _json_tag(reply, "design_experiments")
        if error:
            errors.append(error)
        selection = None
        if designs_payload is not None:
            designs, design_errors = self._normalise_designs(designs_payload)
            errors.extend(design_errors)
            proposed_two_particle_system = any(
                (
                    (controls := self._two_particle_controls(case)) is not None
                    and abs(float(controls["p1"])) > 1e-12
                    for design in designs
                    for case in self._design_input_cases(design)
                )
            )
            requires_radius_progress = self._needs_radius_scale_audit()
            if not self.experiments and proposed_two_particle_system:
                requires_radius_progress = True
            if self.strict and len(designs) < 2:
                errors.append(
                    "strict active design requires at least two legal alternatives"
                )
            if self.strict:
                for design in designs:
                    predicted_ids = {
                        prediction["hypothesis_id"]
                        for prediction in design.get("predictions", [])
                    }
                    if len(predicted_ids) < 2:
                        errors.append(
                            f"design {design['id']!r} must commit predictions from at least two candidate SCMs"
                        )
                    if design["kind"] == "counterfactual":
                        paired_predicted_ids = {
                            prediction["hypothesis_id"]
                            for prediction in design.get("predictions", [])
                            if prediction["target"].startswith("/paired_effects/")
                        }
                        missing_paired = predicted_ids - paired_predicted_ids
                        if missing_paired:
                            errors.append(
                                f"counterfactual design {design['id']!r} must include a /paired_effects/... noise-cancelled prediction for every competing candidate; missing "
                                + ", ".join(sorted(missing_paired))
                            )
                if self._needs_null_source_audit() and (
                    not any(
                        (self._design_has_null_source(design) for design in designs)
                    )
                ):
                    errors.append(
                        "mandatory causal audit is missing a design with a matched p1=0 null-source branch"
                    )
                if self._needs_start_time_audit() and (
                    not any(
                        (
                            self._design_has_nonzero_start_time(design)
                            for design in designs
                        )
                    )
                ):
                    errors.append(
                        "mandatory causal audit is missing a design with a matched three-phase absolute start_time intervention"
                    )
                if self._needs_p2_magnitude_audit() and (
                    not any(
                        (
                            self._design_has_p2_magnitude_variation(design)
                            for design in designs
                        )
                    )
                ):
                    errors.append(
                        "mandatory causal audit is missing a matched design that changes |p2| by at least a factor of two"
                    )
                missing_sign_controls = set(self._missing_sign_audits())
                if missing_sign_controls and (
                    not any(
                        (
                            missing_sign_controls
                            <= self._design_sign_variations(design)
                            for design in designs
                        )
                    )
                ):
                    errors.append(
                        "mandatory causal audit is missing one matched design with same-magnitude sign flips for "
                        + " and ".join(sorted(missing_sign_controls))
                    )
                if requires_radius_progress and (
                    not any(
                        (
                            self._design_advances_radius_scale_audit(design)
                            for design in designs
                        )
                    )
                ):
                    errors.append(
                        "mandatory causal audit is missing a matched design that genuinely extends the anchored radius range; the first accepted scan must straddle its factual radius with at least three radii spanning 4x"
                    )
            if designs:
                selection = self._select_design(
                    designs, max_episode_cost=max_episode_cost
                )
                errors.extend(self._last_contract_selection_errors)
                if (
                    selection is None
                    and max_episode_cost is not None
                    and (not self._last_contract_selection_errors)
                ):
                    errors.append(
                        f"no proposed design fits the remaining simulator episode budget ({max_episode_cost})"
                    )
                self._pending_selection = selection
        loose_predictions, error = _json_tag(reply, "prediction_commitment")
        if error:
            errors.append(error)
        self._pending_loose_predictions = self._normalise_predictions(
            loose_predictions, errors
        )
        for message in errors:
            self.protocol_errors.append({"round": round_num, "error": message})
        return {
            "update": _jsonable(update),
            "selected_design": selection.to_dict() if selection else None,
            "protocol_errors": errors,
        }

    def _validate_final_claim(self, claim: Any) -> list[str]:
        if not isinstance(claim, dict):
            return ["<scm_final> must contain a JSON object"]
        errors = []
        for field in _FINAL_CLAIM_FIELDS:
            if field not in claim:
                errors.append(f"strict <scm_final> is missing {field!r}")
                continue
            value = claim[field]
            if field == "remaining_alternatives":
                if not isinstance(value, list):
                    errors.append(
                        "strict <scm_final> field 'remaining_alternatives' must be a JSON list"
                    )
            elif not isinstance(value, str) or not value.strip():
                errors.append(
                    f"strict <scm_final> field {field!r} must be a non-empty string"
                )
        selected = str(claim.get("selected_candidate_id") or "").strip()
        if selected and selected not in self.candidates:
            errors.append(f"strict <scm_final> selects unknown candidate {selected!r}")
        return errors

    def _apply_update(self, update: Any, round_num: int) -> list[str]:
        errors = []
        if not isinstance(update, dict):
            return ["<scm_update> must contain a JSON object"]
        candidates = _as_list(update.get("candidates"))
        initializing_population = not self.candidates and bool(candidates)
        for raw_candidate in candidates:
            try:
                self._upsert_candidate(
                    raw_candidate, round_num, uniform_prior=initializing_population
                )
            except (TypeError, ValueError) as exc:
                errors.append(f"invalid candidate: {exc}")
        for patch in _as_list(update.get("patches")):
            try:
                self._apply_patch(patch, round_num)
            except (KeyError, TypeError, ValueError) as exc:
                record = {
                    "round": round_num,
                    "patch": _jsonable(patch),
                    "applied": False,
                    "error": str(exc),
                }
                self.patches.append(record)
                errors.append(f"invalid SCM patch: {exc}")
        if update.get("merges"):
            errors.append("unsupported update: use typed SCM patches")
        abduction = update.get("abduction")
        if abduction is not None:
            self.abductions.append(
                {"round": round_num, "content": _jsonable(abduction)}
            )
        for induction in _as_list(update.get("inductions")):
            self.inductions.append(
                {"round": round_num, "content": _jsonable(induction)}
            )
        for deduction in _as_list(update.get("deductions")):
            self.deductions.append(
                {"round": round_num, "content": _jsonable(deduction)}
            )
        if "open_questions" in update:
            self.open_questions = _jsonable(_as_list(update.get("open_questions")))
        errors.extend(self._enforce_candidate_capacity(round_num))
        self._renormalise_weights()
        return errors

    def _upsert_candidate(
        self, raw: Any, round_num: int, uniform_prior: bool = False
    ) -> None:
        if not isinstance(raw, dict):
            raise TypeError("candidate must be an object")
        candidate_id = str(raw.get("id", "")).strip()
        if not candidate_id:
            raise ValueError("candidate.id is required")
        confidence = _safe_confidence(raw.get("confidence"), default=1.0)
        normalised = {
            "id": candidate_id,
            "name": str(raw.get("name") or candidate_id),
            "description": str(raw.get("description") or ""),
            "variables": _jsonable(_as_list(raw.get("variables"))),
            "edges": _jsonable(_as_list(raw.get("edges"))),
            "mechanisms": _jsonable(_as_list(raw.get("mechanisms"))),
            "parameters": _jsonable(raw.get("parameters") or {}),
            "assumptions": _jsonable(_as_list(raw.get("assumptions"))),
            "falsifiers": _jsonable(_as_list(raw.get("falsifiers"))),
            "reported_confidence": confidence,
            "status": str(raw.get("status") or "active"),
            "created_round": round_num,
            "last_updated_round": round_num,
            "revision": 1,
        }
        if raw.get("parent_ids") is not None:
            normalised["parent_ids"] = _jsonable(_as_list(raw.get("parent_ids")))
        if raw.get("merge_reason") is not None:
            normalised["merge_reason"] = str(raw.get("merge_reason") or "")
        previous = self.candidates.get(candidate_id)
        if previous is not None:
            self.version_history.append(
                {
                    "round": round_num,
                    "candidate_id": candidate_id,
                    "revision": previous.get("revision", 1),
                    "reason": "candidate_upsert",
                    "snapshot": copy.deepcopy(previous),
                }
            )
            normalised["created_round"] = previous.get("created_round", round_num)
            normalised["revision"] = int(previous.get("revision", 1)) + 1
        else:
            if uniform_prior:
                entry_weight = 0.0
            else:
                entry_weight = math.log(confidence)
            self.log_weights[candidate_id] = entry_weight
        self.candidates[candidate_id] = normalised

    def _enforce_candidate_capacity(self, round_num: int) -> list[str]:
        active = [
            candidate
            for candidate in self.candidates.values()
            if candidate.get("status") == "active"
        ]
        if len(active) <= self.max_candidates:
            return []
        ranked = sorted(
            active,
            key=lambda candidate: self.log_weights.get(candidate["id"], -math.inf),
            reverse=True,
        )
        archived = ranked[self.max_candidates :]
        for candidate in archived:
            candidate["status"] = "retired_capacity"
        return [
            f"candidate limit {self.max_candidates} exceeded; lowest-weight candidates were retired"
        ]

    def _apply_patch(self, patch: Any, round_num: int) -> None:
        if not isinstance(patch, dict):
            raise TypeError("patch must be an object")
        candidate_id = str(patch.get("candidate_id", "")).strip()
        if candidate_id not in self.candidates:
            raise KeyError(f"unknown candidate {candidate_id!r}")
        operation = str(patch.get("operation", "")).strip()
        payload = patch.get("payload") or {}
        candidate = self.candidates[candidate_id]
        if operation == "retire_candidate" and (
            not self._retirement_is_supported(candidate_id)
        ):
            raise ValueError(
                "retire_candidate requires counterexamples in at least two distinct designs including a paired-effect falsification and negative relative evidence against competitors, or repeated negative relative evidence from at least three distinct designs; tension, unpaired noise, or one batch is not enough"
            )
        self.version_history.append(
            {
                "round": round_num,
                "candidate_id": candidate_id,
                "revision": candidate.get("revision", 1),
                "reason": operation,
                "snapshot": copy.deepcopy(candidate),
            }
        )
        if operation == "add_edge":
            edge = _normalise_edge(payload)
            if edge not in candidate["edges"]:
                candidate["edges"].append(edge)
        elif operation == "remove_edge":
            edge = _normalise_edge(payload)
            candidate["edges"] = [
                old for old in candidate["edges"] if _normalise_edge(old) != edge
            ]
        elif operation == "replace_mechanism":
            target = str(payload.get("target", "")).strip()
            mechanism = payload.get("mechanism")
            if not target or not isinstance(mechanism, dict):
                raise ValueError(
                    "replace_mechanism requires payload.target and payload.mechanism"
                )
            candidate["mechanisms"] = [
                old
                for old in candidate["mechanisms"]
                if not isinstance(old, dict) or str(old.get("target")) != target
            ]
            new_mechanism = _jsonable(mechanism)
            new_mechanism.setdefault("target", target)
            candidate["mechanisms"].append(new_mechanism)
        elif operation == "add_latent":
            variable = payload.get("variable")
            if not isinstance(variable, dict) or not variable.get("name"):
                raise ValueError("add_latent requires payload.variable.name")
            new_variable = _jsonable(variable)
            new_variable["observed"] = False
            new_variable.setdefault("role", "latent")
            candidate["variables"].append(new_variable)
        elif operation == "update_parameter":
            name = str(payload.get("name", "")).strip()
            if not name:
                raise ValueError("update_parameter requires payload.name")
            candidate["parameters"][name] = _jsonable(
                {
                    key: value
                    for key, value in payload.items()
                    if key in ("estimate", "lower", "upper", "unit", "uncertainty")
                }
            )
        elif operation == "retire_candidate":
            candidate["status"] = "retired"
        else:
            raise ValueError(
                "operation must be one of add_edge, remove_edge, replace_mechanism, add_latent, update_parameter, retire_candidate"
            )
        candidate["revision"] = int(candidate.get("revision", 1)) + 1
        candidate["last_updated_round"] = round_num
        self.patches.append(
            {
                "round": round_num,
                "patch": _jsonable(patch),
                "applied": True,
                "resulting_revision": candidate["revision"],
            }
        )

    def _retirement_is_supported(self, candidate_id: str) -> bool:
        records = [
            record
            for record in self.evidence
            if record.get("hypothesis_id") == candidate_id
            and record.get("relative_log_evidence") is not None
        ]
        counterexamples = [
            record
            for record in records
            if record.get("status") == "counterexample"
            and float(record.get("relative_log_evidence") or 0.0) < 0.0
        ]
        counterexample_designs = {record.get("design_id") for record in counterexamples}
        paired_counterexample = any(
            (
                str(record.get("target") or "").startswith("/paired_effects/")
                for record in counterexamples
            )
        )
        if len(counterexample_designs) >= 2 and paired_counterexample:
            return True
        negative = [
            record
            for record in records
            if float(record.get("relative_log_evidence") or 0.0) < 0.0
        ]
        distinct_designs = {record.get("design_id") for record in negative}
        cumulative = sum(
            (float(record.get("relative_log_evidence") or 0.0) for record in negative)
        )
        return len(distinct_designs) >= 3 and cumulative <= -3.0

    def _normalise_designs(self, payload: Any) -> tuple[list[dict], list[str]]:
        errors = []
        raw_designs = payload.get("designs") if isinstance(payload, dict) else payload
        if not isinstance(raw_designs, list):
            return ([], ["<design_experiments> requires a designs array"])
        designs = []
        for index, raw in enumerate(raw_designs):
            if not isinstance(raw, dict):
                errors.append(f"design {index} is not an object")
                continue
            kind = str(raw.get("kind") or "intervention")
            if kind not in ("intervention", "counterfactual"):
                errors.append(f"design {index} has unsupported kind {kind!r}")
                continue
            design_id = str(raw.get("id") or f"round{self.round}_design{index}")
            predictions = self._normalise_predictions(raw.get("predictions"), errors)
            design = {
                "id": design_id,
                "kind": kind,
                "predictions": predictions,
                "rationale": str(raw.get("rationale") or ""),
            }
            if kind == "intervention":
                experiments = raw.get("experiments")
                if experiments is None and raw.get("experiment") is not None:
                    experiments = [raw["experiment"]]
                if not isinstance(experiments, list) or not experiments:
                    errors.append(f"design {design_id!r} has no experiments")
                    continue
                design["experiments"] = _jsonable(experiments)
            else:
                factual = raw.get("factual")
                interventions = raw.get("interventions")
                if not isinstance(factual, dict):
                    errors.append(
                        f"counterfactual design {design_id!r} requires factual object"
                    )
                    continue
                if not isinstance(interventions, list) or not interventions:
                    errors.append(
                        f"counterfactual design {design_id!r} requires interventions"
                    )
                    continue
                design["factual"] = _jsonable(factual)
                design["interventions"] = _jsonable(interventions)
            designs.append(design)
        return (designs, errors)

    @staticmethod
    def _repair_counterfactual_target(prediction: dict, branch_count: int) -> None:
        target = str(prediction.get("target") or "")
        match = re.match(
            "^/(paired_effects|counterfactuals)/0/(output_delta|output)/([1-9]\\d*)(/.*)$",
            target,
        )
        if match is None:
            return
        collection, output_field, raw_branch, suffix = match.groups()
        if (collection, output_field) not in (
            ("paired_effects", "output_delta"),
            ("counterfactuals", "output"),
        ):
            return
        branch_index = int(raw_branch)
        if not 0 <= branch_index < branch_count:
            return
        prediction["original_target"] = target
        prediction["target"] = f"/{collection}/{branch_index}/{output_field}/0{suffix}"
        prediction[
            "target_repair"
        ] = "swapped_counterfactual_branch_and_singleton_batch_index"

    def _normalise_predictions(
        self, payload: Any, errors: Optional[list[str]] = None
    ) -> list[dict]:
        if errors is None:
            errors = []
        if payload is None:
            return []
        raw_predictions = (
            payload.get("predictions") if isinstance(payload, dict) else payload
        )
        if not isinstance(raw_predictions, list):
            errors.append("predictions must be an array")
            return []
        predictions = []
        for index, raw in enumerate(raw_predictions):
            if not isinstance(raw, dict):
                errors.append(f"prediction {index} is not an object")
                continue
            hypothesis_id = str(raw.get("hypothesis_id", "")).strip()
            target = str(raw.get("target", "")).strip()
            if not hypothesis_id or not target.startswith("/"):
                errors.append(
                    f"prediction {index} requires hypothesis_id and /JSON/pointer target"
                )
                continue
            try:
                mean = np.asarray(raw.get("mean"), dtype=float)
                sigma = float(raw.get("sigma"))
            except (TypeError, ValueError):
                errors.append(f"prediction {index} mean/sigma must be numeric")
                continue
            if mean.size == 0 or not np.all(np.isfinite(mean)):
                errors.append(f"prediction {index} mean must be finite")
                continue
            if not math.isfinite(sigma) or sigma <= 0:
                errors.append(f"prediction {index} sigma must be finite and > 0")
                continue
            prediction = {
                "hypothesis_id": hypothesis_id,
                "target": target,
                "mean": _jsonable(raw.get("mean")),
                "sigma": sigma,
                "reasoning": str(raw.get("reasoning") or ""),
            }
            candidate = self.candidates.get(hypothesis_id)
            if candidate is not None:
                prediction["candidate_revision"] = int(candidate.get("revision") or 1)
            predictions.append(prediction)
        return predictions

    @staticmethod
    def _design_episode_cost(design: dict) -> int:
        if design["kind"] == "intervention":
            return max(len(design.get("experiments", [])), 1)
        return 1 + len(design.get("interventions", []))

    def _select_design(
        self, designs: list[dict], max_episode_cost: Optional[int] = None
    ) -> Optional[SelectedDesign]:
        self._last_contract_selection_errors = []
        needs_null_source = self._needs_null_source_audit()
        if not self.experiments:
            needs_null_source = any(
                (
                    self._two_particle_controls(case) is not None
                    for design in designs
                    for case in self._design_input_cases(design)
                )
            )
        needs_p2_magnitude = self._needs_p2_magnitude_audit()
        missing_sign_controls = set(self._missing_sign_audits())
        needs_start_time = self._needs_start_time_audit()
        needs_radius_scale = self._needs_radius_scale_audit()
        if not self.experiments:
            proposed_two_particle_system = any(
                (
                    (controls := self._two_particle_controls(case)) is not None
                    and abs(float(controls["p1"])) > 1e-12
                    for design in designs
                    for case in self._design_input_cases(design)
                )
            )
            needs_radius_scale = bool(proposed_two_particle_system)
        mutation_weights = {}
        scores = []
        for design in designs:
            score, coverage = self._design_information_score(design)
            episode_cost = self._design_episode_cost(design)
            null_audit = self._design_has_null_source(design)
            start_time_audit = self._design_has_nonzero_start_time(design)
            p2_audit = self._design_has_p2_magnitude_variation(design)
            sign_audit_controls = self._design_sign_variations(design)
            sign_audit_gain = len(missing_sign_controls & sign_audit_controls)
            sign_audit = bool(
                missing_sign_controls and missing_sign_controls <= sign_audit_controls
            )
            radius_audit = self._design_advances_radius_scale_audit(design)
            predicted_ids = {
                prediction["hypothesis_id"]
                for prediction in design.get("predictions", [])
                if prediction.get("hypothesis_id") in self.candidates
                and self.candidates[prediction["hypothesis_id"]].get("status")
                == "active"
            }
            schema_valid, schema_total = self._design_prediction_schema_validity(design)
            schema_valid_rate = (
                float(schema_valid / schema_total) if schema_total else 0.0
            )
            audit_gain = (
                sum(
                    (
                        needs_null_source and null_audit,
                        needs_p2_magnitude and p2_audit,
                        needs_start_time and start_time_audit,
                        needs_radius_scale and radius_audit,
                    )
                )
                + sign_audit_gain
            )
            max_disagreement = self._design_max_disagreement(design)
            paired_crn_value = float(
                design["kind"] == "counterfactual" and self.observation_noise_std > 0.0
            )
            scores.append(
                {
                    "design_id": design["id"],
                    "information_score": score,
                    "posterior_coverage": coverage,
                    "kind": design["kind"],
                    "simulator_episode_cost": episode_cost,
                    "budget_feasible": max_episode_cost is None
                    or episode_cost <= max_episode_cost,
                    "satisfies_null_source_audit": null_audit,
                    "satisfies_start_time_audit": start_time_audit,
                    "satisfies_p2_magnitude_audit": p2_audit,
                    "satisfies_sign_audit": sign_audit,
                    "sign_audit_controls": sorted(sign_audit_controls),
                    "sign_audit_gain": int(sign_audit_gain),
                    "satisfies_radius_scale_audit": radius_audit,
                    "prediction_schema_valid": schema_valid,
                    "prediction_schema_total": schema_total,
                    "prediction_schema_valid_rate": schema_valid_rate,
                    "causal_audit_gain": int(audit_gain),
                    "max_standardized_disagreement": max_disagreement,
                    "paired_crn_value": paired_crn_value,
                    "cost_efficiency": 1.0 / max(episode_cost, 1),
                }
            )
        feasible_indices = [
            index for index, score in enumerate(scores) if score["budget_feasible"]
        ]
        if not feasible_indices:
            return None
        if needs_null_source:
            null_source_indices = [
                index
                for index in feasible_indices
                if scores[index]["satisfies_null_source_audit"]
            ]
            if null_source_indices:
                feasible_indices = null_source_indices
        if missing_sign_controls:
            best_sign_gain = max(
                (int(scores[index]["sign_audit_gain"]) for index in feasible_indices),
                default=0,
            )
            if best_sign_gain > 0:
                feasible_indices = [
                    index
                    for index in feasible_indices
                    if scores[index]["sign_audit_gain"] == best_sign_gain
                ]
        if needs_p2_magnitude:
            p2_indices = [
                index
                for index in feasible_indices
                if scores[index]["satisfies_p2_magnitude_audit"]
            ]
            if p2_indices:
                feasible_indices = p2_indices
        if needs_radius_scale:
            radius_indices = [
                index
                for index in feasible_indices
                if scores[index]["satisfies_radius_scale_audit"]
            ]
            if radius_indices:
                feasible_indices = radius_indices
            else:
                anchored_radius_count = len(
                    {
                        round(float(record["radius"]), 12)
                        for record in _as_list(
                            (self._noise_cancelled_causal_audit() or {}).get(
                                "responses"
                            )
                        )
                        if isinstance(record, dict)
                        and isinstance(record.get("radius"), (int, float))
                        and math.isfinite(float(record["radius"]))
                        and (float(record["radius"]) > 0.0)
                        and (abs(float(record.get("p1", 0.0))) > 1e-12)
                    }
                )
                if self.experiments and anchored_radius_count < 2:
                    anchored_radius_count = 0
                if self.experiments and anchored_radius_count == 0:
                    pass
                else:
                    return None
        if needs_start_time:
            start_time_indices = [
                index
                for index in feasible_indices
                if scores[index]["satisfies_start_time_audit"]
            ]
            if start_time_indices:
                feasible_indices = start_time_indices
        selected_index = max(
            feasible_indices, key=lambda idx: (scores[idx]["information_score"], -idx)
        )
        selected = designs[selected_index]
        return SelectedDesign(
            design_id=selected["id"],
            kind=selected["kind"],
            score=float(scores[selected_index]["information_score"]),
            payload=copy.deepcopy(selected),
            predictions=copy.deepcopy(selected["predictions"]),
            proposal_scores=scores,
        )

    def pending_episode_cost(self) -> int:
        if self._pending_selection is None:
            return 0
        return self._design_episode_cost(self._pending_selection.payload)

    def _design_information_score(self, design: dict) -> tuple[float, float]:
        posterior = self.posterior()
        predictions = design.get("predictions", [])
        by_target: dict[str, list[dict]] = {}
        covered = set()
        for prediction in predictions:
            hypothesis_id = prediction["hypothesis_id"]
            if hypothesis_id not in self.candidates:
                continue
            if self.candidates[hypothesis_id].get("status") != "active":
                continue
            by_target.setdefault(prediction["target"], []).append(prediction)
            covered.add(hypothesis_id)
        score = 0.0
        for target, target_predictions in by_target.items():
            causal_weight = (
                1.0
                if target.startswith("/paired_effects/")
                or self.observation_noise_std == 0.0
                else 0.5
            )
            for left_index, left in enumerate(target_predictions):
                for right in target_predictions[left_index + 1 :]:
                    if left["hypothesis_id"] == right["hypothesis_id"]:
                        continue
                    left_mean = np.asarray(left["mean"], dtype=float).reshape(-1)
                    right_mean = np.asarray(right["mean"], dtype=float).reshape(-1)
                    if left_mean.shape != right_mean.shape:
                        continue
                    sigma = math.sqrt(left["sigma"] ** 2 + right["sigma"] ** 2)
                    disagreement = float(
                        np.linalg.norm(left_mean - right_mean)
                        / (math.sqrt(max(left_mean.size, 1)) * sigma)
                    )
                    pair_weight = posterior.get(
                        left["hypothesis_id"], 0.0
                    ) * posterior.get(right["hypothesis_id"], 0.0)
                    score += causal_weight * pair_weight * disagreement
        cost = self._design_episode_cost(design)
        score /= math.sqrt(cost)
        coverage = sum((posterior.get(candidate_id, 0.0) for candidate_id in covered))
        return (float(score), float(coverage))

    @staticmethod
    def _pointer_index(token: str, length: int) -> bool:
        try:
            index = int(token)
        except (TypeError, ValueError):
            return False
        return -length <= index < length

    def _prediction_target_is_plausible(self, design: dict, target: str) -> bool:
        tokens = [token for token in str(target).split("/") if token != ""]
        if not tokens:
            return False
        if design["kind"] == "intervention":
            return self._pointer_index(tokens[0], len(design.get("experiments") or []))
        branch_count = len(design.get("interventions") or [])
        if tokens[0] == "paired_effects":
            return bool(
                len(tokens) >= 4
                and self._pointer_index(tokens[1], branch_count)
                and (tokens[2] == "output_delta")
                and self._pointer_index(tokens[3], 1)
            )
        if tokens[0] == "counterfactuals":
            return bool(
                len(tokens) >= 4
                and self._pointer_index(tokens[1], branch_count)
                and (tokens[2] == "output")
                and self._pointer_index(tokens[3], 1)
            )
        if tokens[0] == "factual":
            return bool(
                len(tokens) >= 3
                and tokens[1] == "output"
                and self._pointer_index(tokens[2], 1)
            )
        return False

    def _design_prediction_schema_validity(self, design: dict) -> tuple[int, int]:
        predictions = design.get("predictions") or []
        return (
            sum(
                (
                    self._prediction_target_is_plausible(
                        design, prediction.get("target", "")
                    )
                    for prediction in predictions
                )
            ),
            len(predictions),
        )

    @staticmethod
    def _design_max_disagreement(design: dict) -> float:
        by_target: dict[str, list[dict]] = {}
        for prediction in design.get("predictions") or []:
            by_target.setdefault(prediction["target"], []).append(prediction)
        maximum = 0.0
        for predictions in by_target.values():
            for left_index, left in enumerate(predictions):
                for right in predictions[left_index + 1 :]:
                    if left["hypothesis_id"] == right["hypothesis_id"]:
                        continue
                    left_mean = np.asarray(left["mean"], dtype=float).reshape(-1)
                    right_mean = np.asarray(right["mean"], dtype=float).reshape(-1)
                    if left_mean.shape != right_mean.shape:
                        continue
                    pooled_sigma = math.sqrt(
                        float(left["sigma"]) ** 2 + float(right["sigma"]) ** 2
                    )
                    disagreement = float(
                        np.linalg.norm(left_mean - right_mean)
                        / (math.sqrt(max(left_mean.size, 1)) * max(pooled_sigma, 1e-12))
                    )
                    maximum = max(maximum, disagreement)
        return maximum

    def execute_pending(self, executor: Any) -> dict:
        if self._pending_selection is None:
            raise RuntimeError("there is no selected SCM experiment to execute")
        selection = self._pending_selection
        if selection.kind == "intervention":
            experiments = copy.deepcopy(selection.payload["experiments"])
            outputs = executor.run(experiments)
            return {
                "kind": "intervention",
                "outcome": outputs,
                "display_tag": "experiment_output",
                "experiment_input": experiments,
                "experiment_output": outputs,
                "log_pairs": list(zip(experiments, outputs)),
            }
        return self._execute_counterfactual(executor, selection)

    def _execute_counterfactual(self, executor: Any, selection: SelectedDesign) -> dict:
        factual_input = copy.deepcopy(selection.payload["factual"])
        before = _rng_snapshot(executor)
        factual_output = executor.run([factual_input])
        after_factual = _rng_snapshot(executor)
        branches = []
        paired_effects = []
        log_pairs = list(zip([factual_input], factual_output))
        try:
            for branch_index, intervention in enumerate(
                selection.payload["interventions"]
            ):
                if not isinstance(intervention, dict):
                    raise TypeError("counterfactual intervention must be an object")
                assignments = intervention.get("set")
                if not isinstance(assignments, dict) or not assignments:
                    raise ValueError(
                        "counterfactual intervention requires non-empty set mapping"
                    )
                branch_input = copy.deepcopy(factual_input)
                for pointer, value in assignments.items():
                    set_json_pointer(branch_input, pointer, value)
                _rng_restore(executor, before)
                branch_output = executor.run([branch_input])
                branch_id = str(
                    intervention.get("id") or f"counterfactual_{branch_index}"
                )
                branches.append(
                    {
                        "id": branch_id,
                        "intervention": _jsonable(assignments),
                        "input": branch_input,
                        "output": branch_output,
                    }
                )
                paired_effects.append(
                    {
                        "id": branch_id,
                        "definition": "counterfactual_minus_factual",
                        "output_delta": _numeric_difference(
                            branch_output, factual_output
                        ),
                    }
                )
                log_pairs.extend(zip([branch_input], branch_output))
        finally:
            _rng_restore(executor, after_factual)
        outcome = {
            "kind": "paired_counterfactual",
            "factual": {"input": factual_input, "output": factual_output},
            "counterfactuals": branches,
            "paired_effects": paired_effects,
            "shared_context": {
                "hidden_world": "held_fixed",
                "observation_noise": "common_random_numbers"
                if before
                else "deterministic_or_unavailable",
            },
        }
        return {
            "kind": "counterfactual",
            "outcome": outcome,
            "display_tag": "counterfactual_output",
            "experiment_input": {
                "factual": factual_input,
                "interventions": selection.payload["interventions"],
            },
            "experiment_output": outcome,
            "log_pairs": log_pairs,
        }

    def validate_pending(self, execution: dict) -> dict:
        if self._pending_selection is None:
            raise RuntimeError("there is no selected experiment to validate")
        selection = self._pending_selection
        validation = self._validate_predictions(
            predictions=selection.predictions,
            outcome=execution["outcome"],
            design_id=selection.design_id,
        )
        experiment_record = {
            "round": self.round,
            **selection.to_dict(),
            "input": _jsonable(execution["experiment_input"]),
            "output": _jsonable(execution["experiment_output"]),
            "output_kind": execution["kind"],
            "prediction_count": len(selection.predictions),
            "validation": copy.deepcopy(validation),
            "local_causal_edges": self._local_counterfactual_edges(execution),
        }
        self.experiments.append(experiment_record)
        self._pending_selection = None
        return validation

    def observe_external_intervention(
        self, experiments: list[dict], outputs: list[dict]
    ) -> dict:
        validation = self._validate_predictions(
            self._pending_loose_predictions,
            outputs,
            design_id=f"round{self.round}_legacy_intervention",
        )
        self.experiments.append(
            {
                "round": self.round,
                "design_id": f"round{self.round}_legacy_intervention",
                "kind": "intervention",
                "information_score": None,
                "proposal_scores": [],
                "input": _jsonable(experiments),
                "output": _jsonable(outputs),
                "output_kind": "intervention",
                "prediction_count": len(self._pending_loose_predictions),
                "validation": copy.deepcopy(validation),
            }
        )
        self._pending_loose_predictions = []
        return validation

    def record_execution_error(self, error: Exception | str) -> dict:
        selection = self._pending_selection
        record = {
            "round": self.round,
            "design_id": selection.design_id if selection else None,
            "kind": selection.kind if selection else "unknown",
            "status": "execution_failure",
            "error": str(error),
        }
        self.experiments.append(record)
        self._pending_selection = None
        return record

    def _validate_predictions(
        self, predictions: list[dict], outcome: Any, design_id: str
    ) -> dict:
        records = []
        for prediction in predictions:
            hypothesis_id = prediction["hypothesis_id"]
            record = {
                "round": self.round,
                "design_id": design_id,
                **copy.deepcopy(prediction),
            }
            try:
                actual_raw = resolve_json_pointer(outcome, prediction["target"])
                actual = np.asarray(actual_raw, dtype=float)
                mean = np.asarray(prediction["mean"], dtype=float)
                if actual.shape != mean.shape:
                    raise ValueError(
                        f"shape mismatch: predicted {mean.shape}, observed {actual.shape}"
                    )
                if not np.all(np.isfinite(actual)):
                    raise ValueError("observed target contains non-finite values")
                rmse = float(np.sqrt(np.mean(np.square(actual - mean))))
                z_score = rmse / prediction["sigma"]
                if z_score <= 1.0:
                    status = "success"
                elif z_score <= 3.0:
                    status = "tension"
                else:
                    status = "counterexample"
                log_likelihood = float(
                    -0.5 * z_score**2 - math.log(prediction["sigma"])
                )
                log_likelihood = min(max(log_likelihood, -50.0), 20.0)
                record.update(
                    {
                        "actual": _jsonable(actual_raw),
                        "rmse": rmse,
                        "standardized_error": z_score,
                        "log_likelihood": log_likelihood,
                        "status": status,
                    }
                )
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                record.update(
                    {
                        "actual": None,
                        "rmse": None,
                        "standardized_error": None,
                        "log_likelihood": None,
                        "status": "invalid_prediction",
                        "error": str(exc),
                    }
                )
            self.evidence.append(record)
            records.append(record)
        by_target: dict[str, list[dict]] = {}
        for record in records:
            hypothesis_id = record.get("hypothesis_id")
            if (
                record.get("log_likelihood") is None
                or hypothesis_id not in self.candidates
                or self.candidates[hypothesis_id].get("status") != "active"
            ):
                continue
            by_target.setdefault(record["target"], []).append(record)
        log_updates: dict[str, list[float]] = {}
        for target_records in by_target.values():
            unique_ids = {record["hypothesis_id"] for record in target_records}
            if len(unique_ids) < 2:
                for record in target_records:
                    record["relative_log_evidence"] = 0.0
                continue
            parents = list(range(len(target_records)))

            def find(index: int) -> int:
                while parents[index] != index:
                    parents[index] = parents[parents[index]]
                    index = parents[index]
                return index

            def union(left: int, right: int) -> None:
                left_root = find(left)
                right_root = find(right)
                if left_root != right_root:
                    parents[right_root] = left_root

            for left_index, left in enumerate(target_records):
                for right_index in range(left_index + 1, len(target_records)):
                    right = target_records[right_index]
                    left_mean = np.asarray(left["mean"], dtype=float).reshape(-1)
                    right_mean = np.asarray(right["mean"], dtype=float).reshape(-1)
                    if left_mean.shape != right_mean.shape:
                        continue
                    pooled_sigma = math.sqrt(
                        float(left["sigma"]) ** 2 + float(right["sigma"]) ** 2
                    )
                    disagreement = float(
                        np.linalg.norm(left_mean - right_mean)
                        / (math.sqrt(max(left_mean.size, 1)) * max(pooled_sigma, 1e-12))
                    )
                    if disagreement < self.min_disagreement_z:
                        union(left_index, right_index)
            groups: dict[int, list[int]] = {}
            for index in range(len(target_records)):
                groups.setdefault(find(index), []).append(index)
            group_scores = {
                root: max(
                    (
                        float(target_records[index]["log_likelihood"])
                        for index in indices
                    )
                )
                for root, indices in groups.items()
            }
            best = max(group_scores.values())
            causal_weight = (
                1.0
                if target_records[0]["target"].startswith("/paired_effects/")
                or self.observation_noise_std == 0.0
                else 0.25
            )
            for group_number, (root, indices) in enumerate(groups.items()):
                relative = (group_scores[root] - best) * causal_weight
                for index in indices:
                    record = target_records[index]
                    record["prediction_equivalence_group"] = group_number
                    record["causal_evidence_weight"] = causal_weight
                    record["relative_log_evidence"] = relative
                    hypothesis_id = record["hypothesis_id"]
                    log_updates.setdefault(hypothesis_id, []).append(relative)
        for hypothesis_id, relative_values in log_updates.items():
            relative_log_evidence = (
                float(np.mean(relative_values))
                * math.sqrt(len(relative_values))
                * self.evidence_temperature
            )
            for record in records:
                if (
                    record.get("hypothesis_id") == hypothesis_id
                    and record.get("relative_log_evidence") is not None
                ):
                    record["tempered_design_log_evidence"] = relative_log_evidence
            self.log_weights[hypothesis_id] = (
                self.log_weights.get(hypothesis_id, 0.0) + relative_log_evidence
            )
        self._renormalise_weights()
        counts = {
            status: sum((1 for record in records if record["status"] == status))
            for status in ("success", "tension", "counterexample", "invalid_prediction")
        }
        return {
            "design_id": design_id,
            "records": records,
            "counts": counts,
            "posterior": self.posterior(),
            "uncommitted": len(predictions) == 0,
        }

    def validation_feedback(self, validation: dict) -> str:
        compact = {
            "design_id": validation.get("design_id"),
            "counts": validation.get("counts"),
            "posterior": validation.get("posterior"),
            "evidence": [
                {
                    "hypothesis_id": record.get("hypothesis_id"),
                    "target": record.get("target"),
                    "original_target": record.get("original_target"),
                    "target_repair": record.get("target_repair"),
                    "status": record.get("status"),
                    "standardized_error": record.get("standardized_error"),
                    "actual": record.get("actual"),
                }
                for record in validation.get("records", [])
            ],
            "instruction": "Treat counterexamples as evidence against the committed prediction, but distinguish nuisance miscalibration and unpaired observation noise from structural falsification. On the next round, abduct likely causes, apply an explicit SCM patch or plan a replicated paired falsification, derive a new risky consequence, and then design the next experiment. Retire a mechanism only after the runtime's evidence threshold is met.",
        }
        return (
            "<scm_validation>\n"
            + json.dumps(_jsonable(compact), separators=(",", ":"), ensure_ascii=False)
            + "\n</scm_validation>"
        )

    def posterior(self) -> dict[str, float]:
        active_ids = [
            candidate_id
            for candidate_id, candidate in self.candidates.items()
            if candidate.get("status") == "active"
        ]
        if not active_ids:
            return {}
        values = np.asarray(
            [
                self.log_weights.get(candidate_id, 0.0)
                - self.complexity_penalty
                * self._candidate_complexity(self.candidates[candidate_id])
                for candidate_id in active_ids
            ],
            dtype=float,
        )
        values -= np.max(values)
        weights = np.exp(np.clip(values, -700.0, 0.0))
        total = float(weights.sum())
        if total <= 0 or not math.isfinite(total):
            weights = np.ones_like(weights) / len(weights)
        else:
            weights /= total
        return {
            candidate_id: float(weight)
            for candidate_id, weight in zip(active_ids, weights)
        }

    def _renormalise_weights(self) -> None:
        active_ids = [
            candidate_id
            for candidate_id, candidate in self.candidates.items()
            if candidate.get("status") == "active"
        ]
        if not active_ids:
            return
        values = np.asarray(
            [self.log_weights.get(candidate_id, 0.0) for candidate_id in active_ids],
            dtype=float,
        )
        finite = np.isfinite(values)
        if not finite.any():
            values = np.zeros_like(values)
        else:
            floor = float(np.min(values[finite])) - 700.0
            values = np.where(finite, values, floor)
        maximum = float(np.max(values))
        log_normalizer = maximum + math.log(
            float(np.exp(np.clip(values - maximum, -700.0, 0.0)).sum())
        )
        for candidate_id, value in zip(active_ids, values):
            self.log_weights[candidate_id] = float(value - log_normalizer)

    @staticmethod
    def _candidate_complexity(candidate: dict) -> float:
        parameters = len(candidate.get("parameters") or {})
        variables = candidate.get("variables") or []
        latent = sum(
            (
                isinstance(variable, dict)
                and (
                    variable.get("observed") is False
                    or "latent" in str(variable.get("role", "")).lower()
                )
                and (str(variable.get("role", "")).lower() != "parameter")
                for variable in variables
            )
        )
        mechanisms = max(len(candidate.get("mechanisms") or []) - 1, 0)
        edges = max(len(candidate.get("edges") or []) - 3, 0) / 3.0
        return float(parameters + latent + mechanisms + edges)

    def metrics(self) -> dict:
        valid = [
            record
            for record in self.evidence
            if record.get("standardized_error") is not None
        ]
        statuses = [record.get("status") for record in self.evidence]
        design_scores = [
            record.get("information_score")
            for record in self.experiments
            if isinstance(record.get("information_score"), (int, float))
        ]
        z_scores = [float(record["standardized_error"]) for record in valid]
        simulator_episodes = 0
        counterfactual_branches = 0
        for record in self.experiments:
            inputs = record.get("input")
            if record.get("kind") == "counterfactual" and isinstance(inputs, dict):
                branches = len(inputs.get("interventions") or [])
                simulator_episodes += 1 + branches
                counterfactual_branches += branches
            elif isinstance(inputs, list):
                simulator_episodes += len(inputs)
        metrics = {
            "candidate_count": len(self.candidates),
            "active_candidate_count": sum(
                (
                    candidate.get("status") == "active"
                    for candidate in self.candidates.values()
                )
            ),
            "candidate_revisions": len(self.version_history),
            "patches_applied": sum(
                (record.get("applied") is True for record in self.patches)
            ),
            "experiments_executed": len(self.experiments),
            "simulator_episodes": simulator_episodes,
            "counterfactual_experiments": sum(
                (record.get("kind") == "counterfactual" for record in self.experiments)
            ),
            "counterfactual_branches": counterfactual_branches,
            "prediction_count": len(self.evidence),
            "prediction_successes": statuses.count("success"),
            "prediction_tensions": statuses.count("tension"),
            "counterexamples": statuses.count("counterexample"),
            "invalid_predictions": statuses.count("invalid_prediction"),
            "repaired_prediction_targets": sum(
                (bool(record.get("target_repair")) for record in self.evidence)
            ),
            "valid_prediction_rate": float(len(valid) / len(self.evidence))
            if self.evidence
            else 0.0,
            "within_1sigma_rate": float(np.mean(np.asarray(z_scores) <= 1.0))
            if z_scores
            else None,
            "within_2sigma_rate": float(np.mean(np.asarray(z_scores) <= 2.0))
            if z_scores
            else None,
            "mean_standardized_error": float(np.mean(z_scores)) if z_scores else None,
            "mean_information_score": float(np.mean(design_scores))
            if design_scores
            else None,
            "protocol_error_count": len(self.protocol_errors),
            "mse_fit_attempts": len(self.fit_attempts),
            "successful_mse_fits": sum(
                (
                    attempt.get("error") in (None, "")
                    and isinstance(attempt.get("loss_after"), (int, float))
                    and math.isfinite(float(attempt["loss_after"]))
                    for attempt in self.fit_attempts
                )
            ),
            "noise_cancelled_causal_edges": sum(
                (
                    len(_as_list(experiment.get("local_causal_edges")))
                    for experiment in self.experiments
                )
            ),
            "direct_single_cadence_causal_edges": sum(
                (
                    len(self._direct_intervention_edges(experiment))
                    for experiment in self.experiments
                )
            ),
        }
        causal_audit = self._noise_cancelled_causal_audit()
        metrics["anchored_local_responses"] = (
            causal_audit.get("anchored_response_count", 0)
            if isinstance(causal_audit, dict)
            else 0
        )
        coverage = self.loop_coverage()
        metrics["loop_coverage"] = coverage
        metrics["loop_coverage_rate"] = float(np.mean(list(coverage.values())))
        return metrics

    def loop_coverage(self) -> dict[str, bool]:
        statuses = [record.get("status") for record in self.evidence]
        return {
            "commonsense_candidate_initialization": len(self.candidates) >= 2,
            "observation": bool(self.experiments),
            "abduction": bool(self.abductions),
            "active_intervention_planning": any(
                (
                    len(record.get("proposal_scores", [])) >= 2
                    for record in self.experiments
                )
            ),
            "paired_counterfactual": any(
                (record.get("kind") == "counterfactual" for record in self.experiments)
            ),
            "preregistered_prediction": bool(self.evidence),
            "simulator_validation": any(
                (
                    record.get("status") != "invalid_prediction"
                    for record in self.evidence
                )
            ),
            "success_evidence": "success" in statuses,
            "failure_or_counterexample_evidence": "counterexample" in statuses
            or any(
                (
                    record.get("status") == "execution_failure"
                    for record in self.experiments
                )
            ),
            "rule_revision": bool(self.version_history),
            "induction": bool(self.inductions),
            "deduction": bool(self.deductions),
            "final_scm": self.final_claim is not None,
            "final_executable_law": self.final_law is not None,
        }

    def finalize(
        self,
        law_source: Optional[str],
        explanation: Optional[str],
        reply: Optional[str] = None,
    ) -> None:
        self.final_law = law_source
        self.final_explanation = explanation
        if reply and self.final_claim is None:
            final_claim, error = _json_tag(reply, "scm_final")
            if final_claim is not None:
                self.final_claim = _jsonable(final_claim)
            elif error:
                self.protocol_errors.append({"round": self.round, "error": error})

    def to_dict(self) -> dict:
        return {
            "schema_version": SCM_SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "world": self.world,
            "strict": self.strict,
            "observation_noise_std": self.observation_noise_std,
            "max_candidates": self.max_candidates,
            "complexity_penalty": self.complexity_penalty,
            "round": self.round,
            "posterior": self.posterior(),
            "candidates": copy.deepcopy(list(self.candidates.values())),
            "version_history": copy.deepcopy(self.version_history),
            "abductions": copy.deepcopy(self.abductions),
            "inductions": copy.deepcopy(self.inductions),
            "deductions": copy.deepcopy(self.deductions),
            "patches": copy.deepcopy(self.patches),
            "evidence": copy.deepcopy(self.evidence),
            "experiments": copy.deepcopy(self.experiments),
            "open_questions": copy.deepcopy(self.open_questions),
            "fit_attempts": copy.deepcopy(self.fit_attempts),
            "capacity_events": copy.deepcopy(self.capacity_events),
            "protocol_errors": copy.deepcopy(self.protocol_errors),
            "final_claim": copy.deepcopy(self.final_claim),
            "final_law": self.final_law,
            "final_explanation": self.final_explanation,
            "time_alias_selection": copy.deepcopy(self.time_alias_selection),
            "metrics": self.metrics(),
            "loop_coverage": self.loop_coverage(),
        }

    def write_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(_jsonable(self.to_dict()), indent=2, ensure_ascii=False) + "\n"
        )


def _normalise_edge(edge: Any) -> dict:
    if isinstance(edge, (list, tuple)) and len(edge) == 2:
        source, target = edge
    elif isinstance(edge, dict):
        source, target = (edge.get("source"), edge.get("target"))
    else:
        raise ValueError("edge must be [source, target] or {source, target}")
    source = str(source or "").strip()
    target = str(target or "").strip()
    if not source or not target:
        raise ValueError("edge source and target are required")
    return {"source": source, "target": target}
