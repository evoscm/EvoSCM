from __future__ import annotations
import csv
import os
import re
import datetime as _dt
from pathlib import Path
from typing import Optional, Sequence
import numpy as np

COLUMNS: tuple[str, ...] = (
    "run_id",
    "experiment_id",
    "round",
    "source",
    "time",
    "particle_id",
    "x",
    "y",
    "vx",
    "vy",
    "p1",
    "p2",
    "start_time",
    "ring_radius",
    "initial_tangential_velocity",
    "mass",
    "charge",
)


def make_run_id(model: str, when: Optional[_dt.datetime] = None) -> str:
    when = when or _dt.datetime.now()
    stamp = when.strftime("%Y%m%dT%H%M%S")
    slug = re.sub("[^A-Za-z0-9._-]+", "-", model).strip("-") or "model"
    return f"{stamp}_{slug}"


class TrajectoryLogger:
    def __init__(self, world: str, executor, csv_path: os.PathLike | str, run_id: str):
        self.world = world
        self.executor = executor
        self.csv_path = Path(csv_path)
        self.run_id = run_id
        self._row_builder = _RowBuilders.get(world)
        if self._row_builder is None:
            raise ValueError(
                f"TrajectoryLogger has no row builder for world '{world}'. Known worlds: {sorted(_RowBuilders)}"
            )
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._needs_header = (
            not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        )

    def log_experiment(
        self,
        round_num: int,
        source: str,
        exp_input: dict,
        exp_output: dict,
        exp_idx_in_round: int = 0,
    ) -> None:
        if not isinstance(exp_input, dict) or not isinstance(exp_output, dict):
            return
        experiment_id = f"{self.run_id}__r{round_num}__e{exp_idx_in_round}"
        rows = self._row_builder(self.executor, exp_input, exp_output)
        if not rows:
            return
        for row in rows:
            row["run_id"] = self.run_id
            row["experiment_id"] = experiment_id
            row["round"] = round_num
            row["source"] = source
        self._write(rows)

    def _write(self, rows: Sequence[dict]) -> None:
        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            if self._needs_header:
                writer.writeheader()
                self._needs_header = False
            writer.writerows(rows)


def _scalar_params(
    p1=None, p2=None, start_time=None, ring_radius=None, v_tang=None
) -> dict:
    return {
        "p1": p1,
        "p2": p2,
        "start_time": start_time,
        "ring_radius": ring_radius,
        "initial_tangential_velocity": v_tang,
    }


def _row(time, particle_id, pos, vel, params, mass=None, charge=None) -> dict:
    row = {
        "time": float(time),
        "particle_id": int(particle_id),
        "x": float(pos[0]),
        "y": float(pos[1]),
        "vx": float(vel[0]),
        "vy": float(vel[1]),
        **params,
    }
    if mass is not None:
        row["mass"] = float(mass)
    if charge is not None:
        row["charge"] = float(charge)
    return row


def _rows_two_particle(executor, exp_input, exp_output) -> list[dict]:
    p1 = float(exp_input["p1"])
    p2 = float(exp_input["p2"])
    pos2_init = list(exp_input["pos2"])
    vel2_init = list(exp_input["velocity2"])
    times = exp_output.get("measurement_times", [])
    pos1 = exp_output.get("pos1", [])
    pos2 = exp_output.get("pos2", [])
    vel1 = exp_output.get("velocity1", [])
    vel2 = exp_output.get("velocity2", [])
    params = _scalar_params(
        p1=p1,
        p2=p2,
        start_time=float(exp_input["start_time"])
        if "start_time" in exp_input
        else None,
    )
    rows: list[dict] = []
    rows.append(_row(0.0, 0, [0.0, 0.0], [0.0, 0.0], params))
    rows.append(_row(0.0, 1, pos2_init, vel2_init, params))
    for i, t in enumerate(times):
        if i < len(pos1) and i < len(vel1):
            rows.append(_row(t, 0, pos1[i], vel1[i], params))
        if i < len(pos2) and i < len(vel2):
            rows.append(_row(t, 1, pos2[i], vel2[i], params))
    return rows


def _rows_circle(executor, exp_input, exp_output) -> list[dict]:
    ring_radius = float(exp_input.get("ring_radius", 5.0))
    v_tang = float(exp_input.get("initial_tangential_velocity", 0.0))
    times = exp_output.get("measurement_times", [])
    positions = exp_output.get("positions", [])
    velocities = exp_output.get("velocities", [])
    params = _scalar_params(ring_radius=ring_radius, v_tang=v_tang)
    n_ring = getattr(executor, "N_RING", 10)
    n_total = getattr(executor, "N_TOTAL", 11)
    angles = np.linspace(0.0, 2.0 * np.pi, n_ring, endpoint=False)
    ring_pos = np.column_stack(
        [ring_radius * np.cos(angles), ring_radius * np.sin(angles)]
    )
    ring_vel = np.column_stack([-v_tang * np.sin(angles), v_tang * np.cos(angles)])
    init_positions = np.vstack([[[0.0, 0.0]], ring_pos])
    init_velocities = np.vstack([[[0.0, 0.0]], ring_vel])
    rows: list[dict] = []
    for pid in range(n_total):
        rows.append(_row(0.0, pid, init_positions[pid], init_velocities[pid], params))
    for i, t in enumerate(times):
        if i >= len(positions) or i >= len(velocities):
            break
        snap_p = positions[i]
        snap_v = velocities[i]
        for pid in range(min(n_total, len(snap_p), len(snap_v))):
            rows.append(_row(t, pid, snap_p[pid], snap_v[pid], params))
    return rows


def _rows_three_species(executor, exp_input, exp_output) -> list[dict]:
    bg_positions = np.asarray(
        exp_output.get("background_initial_positions", []), dtype=float
    )
    if bg_positions.size == 0:
        bg_positions = np.asarray(executor._bg_positions_rel, dtype=float)
    n_bg = bg_positions.shape[0]
    bg_velocities = np.zeros_like(bg_positions)
    probe_pos = np.asarray(exp_input["probe_positions"], dtype=float)
    probe_vel = np.asarray(exp_input["probe_velocities"], dtype=float)
    init_positions = np.vstack([bg_positions, probe_pos])
    init_velocities = np.vstack([bg_velocities, probe_vel])
    n_total = init_positions.shape[0]
    times = exp_output.get("measurement_times", [])
    positions = exp_output.get("positions", [])
    velocities = exp_output.get("velocities", [])
    params = _scalar_params()
    rows: list[dict] = []
    for pid in range(n_total):
        rows.append(_row(0.0, pid, init_positions[pid], init_velocities[pid], params))
    for i, t in enumerate(times):
        if i >= len(positions) or i >= len(velocities):
            break
        snap_p = positions[i]
        snap_v = velocities[i]
        for pid in range(min(n_total, len(snap_p), len(snap_v))):
            rows.append(_row(t, pid, snap_p[pid], snap_v[pid], params))
    return rows


def _rows_dark_matter(executor, exp_input, exp_output) -> list[dict]:
    visible_positions = np.asarray(
        exp_output.get("background_initial_positions", []), dtype=float
    )
    if visible_positions.size == 0:
        visible_positions = np.asarray(executor._visible_positions_rel, dtype=float)
    visible_velocities = np.asarray(
        getattr(executor, "_visible_velocities", np.zeros_like(visible_positions)),
        dtype=float,
    )
    if visible_velocities.shape != visible_positions.shape:
        visible_velocities = np.zeros_like(visible_positions)
    sign = float(exp_input.get("visible_velocity_sign", 1.0))
    visible_velocities = sign * visible_velocities
    probe_pos = np.asarray(exp_input["probe_positions"], dtype=float)
    probe_vel = np.asarray(exp_input["probe_velocities"], dtype=float)
    init_positions = np.vstack([visible_positions, probe_pos])
    init_velocities = np.vstack([visible_velocities, probe_vel])
    n_total = init_positions.shape[0]
    times = exp_output.get("measurement_times", [])
    positions = exp_output.get("positions", [])
    velocities = exp_output.get("velocities", [])
    params = _scalar_params()
    rows: list[dict] = []
    for pid in range(n_total):
        rows.append(_row(0.0, pid, init_positions[pid], init_velocities[pid], params))
    for i, t in enumerate(times):
        if i >= len(positions) or i >= len(velocities):
            break
        snap_p = positions[i]
        snap_v = velocities[i]
        for pid in range(min(n_total, len(snap_p), len(snap_v))):
            rows.append(_row(t, pid, snap_p[pid], snap_v[pid], params))
    return rows


def _rows_ether(executor, exp_input, exp_output) -> list[dict]:
    bg_positions = np.asarray(
        exp_output.get("background_initial_positions", []), dtype=float
    )
    if bg_positions.size == 0:
        bg_positions = np.asarray(executor._bg_positions_rel, dtype=float)
    bg_velocities = np.asarray(
        exp_output.get("background_initial_velocities", []), dtype=float
    )
    if bg_velocities.shape != bg_positions.shape:
        bg_velocities = np.asarray(executor._bg_velocities, dtype=float)
    probe_pos = np.asarray(exp_input["probe_positions"], dtype=float)
    probe_vel = np.asarray(exp_input["probe_velocities"], dtype=float)
    init_positions = np.vstack([bg_positions, probe_pos])
    init_velocities = np.vstack([bg_velocities, probe_vel])
    masses = np.asarray(exp_output.get("particle_masses", []), dtype=float)
    n_total = init_positions.shape[0]
    if masses.shape != (n_total,):
        n_bg = executor._bg_masses.shape[0]
        n_probes = executor.N_PROBES
        probe_masses = np.asarray(
            exp_input.get("probe_masses", [executor.DEFAULT_PROBE_MASS] * n_probes),
            dtype=float,
        )
        masses = np.concatenate([executor._bg_masses, probe_masses])
    times = exp_output.get("measurement_times", [])
    positions = exp_output.get("positions", [])
    velocities = exp_output.get("velocities", [])
    params = _scalar_params()
    rows: list[dict] = []
    for pid in range(n_total):
        rows.append(
            _row(
                0.0,
                pid,
                init_positions[pid],
                init_velocities[pid],
                params,
                mass=float(masses[pid]),
            )
        )
    for i, t in enumerate(times):
        if i >= len(positions) or i >= len(velocities):
            break
        snap_p = positions[i]
        snap_v = velocities[i]
        for pid in range(min(n_total, len(snap_p), len(snap_v))):
            rows.append(
                _row(t, pid, snap_p[pid], snap_v[pid], params, mass=float(masses[pid]))
            )
    return rows


_rows_hubble = _rows_ether


def _rows_coulomb_easy(executor, exp_input, exp_output) -> list[dict]:
    p1 = float(exp_input["p1"])
    p2 = float(exp_input["p2"])
    pos2_init = list(exp_input["pos2"])
    vel2_init = list(exp_input["velocity2"])
    times = exp_output.get("measurement_times", [])
    pos1 = exp_output.get("pos1", [])
    pos2 = exp_output.get("pos2", [])
    vel1 = exp_output.get("velocity1", [])
    vel2 = exp_output.get("velocity2", [])
    params = _scalar_params(p1=p1, p2=p2)
    charges = exp_output.get("particle_charges", [+abs(p1), -abs(p2)])
    rows: list[dict] = []
    rows.append(_row(0.0, 0, [0.0, 0.0], [0.0, 0.0], params, charge=charges[0]))
    rows.append(_row(0.0, 1, pos2_init, vel2_init, params, charge=charges[1]))
    for i, t in enumerate(times):
        if i < len(pos1) and i < len(vel1):
            rows.append(_row(t, 0, pos1[i], vel1[i], params, charge=charges[0]))
        if i < len(pos2) and i < len(vel2):
            rows.append(_row(t, 1, pos2[i], vel2[i], params, charge=charges[1]))
    return rows


_RowBuilders = {
    "gravity": _rows_two_particle,
    "yukawa": _rows_two_particle,
    "fractional": _rows_two_particle,
    "oscillator": _rows_two_particle,
    "circle": _rows_circle,
    "three_species": _rows_three_species,
    "dark_matter": _rows_dark_matter,
    "ether": _rows_ether,
    "hubble": _rows_hubble,
    "coulomb_easy": _rows_coulomb_easy,
    "extra_dimensions": _rows_two_particle,
}
