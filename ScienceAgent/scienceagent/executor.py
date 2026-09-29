import json
import numpy as np
import sys
import os
from contextlib import contextmanager
import jax.numpy as jnp

_repo_root = os.path.join(os.path.dirname(__file__), "..", "..", "PhysicsSchool")
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from physchool.worlds.field_sampler import FieldSampler
from physchool.worlds.nbody_sampler import NBodySampler
from physchool.worlds.force_laws import (
    poisson_2d_force,
    poisson_2d_potential,
    yukawa_2d_force,
    yukawa_2d_potential,
    riesz_2d_force,
    riesz_2d_potential,
    extra_dimensions_2d_force,
    extra_dimensions_2d_potential,
    coulomb_force,
    coulomb_potential,
)


def _operator_to_pairwise(operators):
    if not isinstance(operators, list) or len(operators) != 1:
        raise ValueError(
            f"engine='nbody' only supports a single operator at a time; got {operators!r}"
        )
    op = operators[0]
    op_type = op["type"]
    params = op.get("params", {})
    strength = float(params.get("strength", 1.0))
    if op_type == "laplacian":
        force_law = lambda r, qi, qj, mi, mj: poisson_2d_force(
            r, qi, qj, mi, mj, G=strength
        )
        pot_law = lambda r, qi, qj, mi, mj: poisson_2d_potential(
            r, qi, qj, mi, mj, G=strength
        )
        return (force_law, pot_law)
    if op_type == "screening":
        lam = float(params["screening_length"])
        force_law = lambda r, qi, qj, mi, mj: yukawa_2d_force(
            r, qi, qj, mi, mj, G=strength, lam=lam
        )
        pot_law = lambda r, qi, qj, mi, mj: yukawa_2d_potential(
            r, qi, qj, mi, mj, G=strength, lam=lam
        )
        return (force_law, pot_law)
    if op_type == "helmholtz":
        m2 = float(params["mass_squared"])
        lam = 1.0 / np.sqrt(m2)
        force_law = lambda r, qi, qj, mi, mj: yukawa_2d_force(
            r, qi, qj, mi, mj, G=strength, lam=lam
        )
        pot_law = lambda r, qi, qj, mi, mj: yukawa_2d_potential(
            r, qi, qj, mi, mj, G=strength, lam=lam
        )
        return (force_law, pot_law)
    if op_type == "fractional_laplacian":
        alpha = float(params["alpha"])
        force_law = lambda r, qi, qj, mi, mj: riesz_2d_force(
            r, qi, qj, mi, mj, G=strength, alpha=alpha
        )
        pot_law = lambda r, qi, qj, mi, mj: riesz_2d_potential(
            r, qi, qj, mi, mj, G=strength, alpha=alpha
        )
        return (force_law, pot_law)
    if op_type == "coulomb":
        force_law = lambda r, qi, qj, mi, mj: coulomb_force(
            r, qi, qj, mi, mj, k=strength
        )
        pot_law = lambda r, qi, qj, mi, mj: coulomb_potential(
            r, qi, qj, mi, mj, k=strength
        )
        return (force_law, pot_law)
    raise ValueError(
        f"engine='nbody' does not support operator type {op_type!r}; diffusion / wave / arbitrary linear operators must use engine='field'."
    )


def _record_at_times(sim: NBodySampler, dt: float, duration: float, measurement_times):
    n_steps_total = max(int(round(duration / dt)), 1)
    traj = sim.run(n_steps=n_steps_total, record_every=1)
    positions = np.asarray(traj["positions"])
    velocities = np.asarray(traj["velocities"])
    indices = []
    for mt in measurement_times:
        idx = int(round(float(mt) / dt))
        idx = max(0, min(n_steps_total, idx))
        indices.append(idx)
    return (
        np.array([positions[i] for i in indices]),
        np.array([velocities[i] for i in indices]),
    )


def _check_nbody_supports(temporal_order):
    if temporal_order != 0:
        raise ValueError(
            "engine='nbody' supports only temporal_order=0 (instantaneous central forces); the diffusion / wave worlds need engine='field'."
        )


class _NoisyExecutorMixin:
    def _init_noise(self, noise_std: float = 0.0, noise_seed: int = None):
        self.noise_std = float(noise_std or 0.0)
        self.noise_seed = noise_seed
        self._noise_rng = np.random.default_rng(noise_seed)
        if noise_seed is None:
            vel_seed = None
        else:
            vel_seed = np.random.SeedSequence(noise_seed).spawn(1)[0]
        self._vel_noise_rng = np.random.default_rng(vel_seed)

    def _noisy_positions(self, positions):
        if self.noise_std <= 0.0:
            return positions
        arr = np.asarray(positions, dtype=np.float64)
        return arr + self._noise_rng.normal(0.0, self.noise_std, size=arr.shape)

    def _noisy_velocities(self, velocities, measurement_times):
        arr = np.asarray(velocities, dtype=np.float64)
        if self.noise_std <= 0.0:
            return arr.tolist()
        times = np.asarray(sorted(measurement_times), dtype=np.float64)
        if times.size < 2:
            return arr.tolist()
        dt = float(np.median(np.diff(times)))
        if not np.isfinite(dt) or dt <= 0.0:
            return arr.tolist()
        sigma_v = self.noise_std / dt
        noised = arr + self._vel_noise_rng.normal(0.0, sigma_v, size=arr.shape)
        return noised.tolist()

    @contextmanager
    def noise_disabled(self):
        saved = self.noise_std
        self.noise_std = 0.0
        try:
            yield
        finally:
            self.noise_std = saved


class SimulationExecutor(_NoisyExecutorMixin):
    def __init__(
        self,
        operators=None,
        temporal_order=0,
        grid_size=(64, 64),
        domain_size=20.0,
        dt=0.005,
        noise_std=0.0,
        noise_seed=None,
    ):
        self.operators = operators or [
            {"type": "laplacian", "params": {"strength": 1.0}}
        ]
        self.temporal_order = temporal_order
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.dt = dt
        self._init_noise(noise_std, noise_seed)

    def run(self, experiments: list[dict]) -> list[dict]:
        return [self._run_one(exp) for exp in experiments]

    def run_json(self, json_str: str) -> str:
        experiments = json.loads(json_str)
        results = self.run(experiments)
        return json.dumps(results, indent=2)

    def _run_one(self, exp: dict) -> dict:
        p1 = float(exp["p1"])
        p2 = float(exp["p2"])
        pos2 = list(exp["pos2"])
        velocity2 = list(exp["velocity2"])
        measurement_times = sorted(exp["measurement_times"])
        duration = float(exp.get("duration", max(measurement_times)))
        duration = max(duration, 5.0)
        centre = self.domain_size / 2.0
        init_positions = np.array(
            [[centre, centre], [centre + pos2[0], centre + pos2[1]]], dtype=np.float64
        )
        init_velocities = np.array([[0.0, 0.0], velocity2], dtype=np.float64)
        sim = FieldSampler(
            particle_inertia=np.array([1, p2]),
            particle_source=np.array([p1, 1.0]),
            particle_force=np.array([0.0, 1.0]),
            initial_positions=init_positions,
            initial_velocities=init_velocities,
            n_particles=2,
            spatial_dimensions=2,
            temporal_order=self.temporal_order,
            grid_size=self.grid_size,
            domain_size=self.domain_size,
            operators=self.operators,
            dt=self.dt,
            source_coupling=np.array([p1, 1.0]),
            force_coupling=1.0,
            periodic_boundaries=True,
        )
        pos1_traj, pos2_traj = ([], [])
        vel1_traj, vel2_traj = ([], [])
        recorded = set()
        n_steps = int(round(duration / self.dt))
        for i in range(n_steps + 1):
            t = round(i * self.dt, 10)
            for mt in measurement_times:
                if mt not in recorded and t >= mt:
                    p1_pos = sim.positions[0] - centre
                    p2_pos = sim.positions[1] - centre
                    pos1_traj.append(self._noisy_positions(p1_pos).tolist())
                    pos2_traj.append(self._noisy_positions(p2_pos).tolist())
                    vel1_traj.append(sim.velocities[0].tolist())
                    vel2_traj.append(sim.velocities[1].tolist())
                    recorded.add(mt)
            if len(recorded) == len(measurement_times):
                break
            if i < n_steps:
                sim.step()
        return {
            "measurement_times": measurement_times,
            "pos1": pos1_traj,
            "pos2": pos2_traj,
            "velocity1": self._noisy_velocities(vel1_traj, measurement_times),
            "velocity2": self._noisy_velocities(vel2_traj, measurement_times),
        }


class CircleExecutor(_NoisyExecutorMixin):
    N_RING = 10
    N_TOTAL = 11
    ALPHA = 0.75

    def __init__(
        self,
        operators=None,
        temporal_order=0,
        grid_size=(128, 128),
        domain_size=50.0,
        dt=0.005,
        noise_std=0.0,
        noise_seed=None,
    ):
        self.operators = operators or [
            {
                "type": "fractional_laplacian",
                "params": {"strength": 1.0, "alpha": self.ALPHA},
            }
        ]
        self.temporal_order = temporal_order
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.dt = dt
        self._init_noise(noise_std, noise_seed)

    def run(self, experiments: list[dict]) -> list[dict]:
        return [self._run_one(exp) for exp in experiments]

    def run_json(self, json_str: str) -> str:
        experiments = json.loads(json_str)
        results = self.run(experiments)
        return json.dumps(results, indent=2)

    def _run_one(self, exp: dict) -> dict:
        ring_radius = float(exp.get("ring_radius", 5.0))
        v_tang = float(exp.get("initial_tangential_velocity", 0.0))
        measurement_times = sorted(exp["measurement_times"])
        duration = float(exp.get("duration", max(measurement_times)))
        duration = max(duration, 5.0)
        centre = self.domain_size / 2.0
        angles = np.linspace(0, 2 * np.pi, self.N_RING, endpoint=False)
        ring_pos = np.column_stack(
            [
                centre + ring_radius * np.cos(angles),
                centre + ring_radius * np.sin(angles),
            ]
        )
        positions = np.vstack([[[centre, centre]], ring_pos])
        ring_vel = np.column_stack([-v_tang * np.sin(angles), v_tang * np.cos(angles)])
        velocities = np.vstack([[[0.0, 0.0]], ring_vel])
        masses = np.ones(self.N_TOTAL)
        sim = FieldSampler(
            particle_inertia=masses,
            particle_source=masses,
            particle_force=masses,
            initial_positions=positions,
            initial_velocities=velocities,
            n_particles=self.N_TOTAL,
            spatial_dimensions=2,
            temporal_order=self.temporal_order,
            grid_size=self.grid_size,
            domain_size=self.domain_size,
            operators=self.operators,
            dt=self.dt,
            source_coupling=masses,
            force_coupling=1.0,
            periodic_boundaries=False,
        )
        pos_traj, vel_traj = ([], [])
        recorded = set()
        n_steps = int(round(duration / self.dt))
        for i in range(n_steps + 1):
            t = round(i * self.dt, 10)
            for mt in measurement_times:
                if mt not in recorded and t >= mt:
                    pos_traj.append(
                        self._noisy_positions(sim.positions - centre).tolist()
                    )
                    vel_traj.append(sim.velocities.tolist())
                    recorded.add(mt)
            if len(recorded) == len(measurement_times):
                break
            if i < n_steps:
                sim.step()
        return {
            "measurement_times": measurement_times,
            "positions": pos_traj,
            "velocities": self._noisy_velocities(vel_traj, measurement_times),
        }


class ThreeSpeciesExecutor(_NoisyExecutorMixin):
    N_BACKGROUND = 30
    N_PROBES = 5
    N_TOTAL = 35
    SPECIES_A = list(range(0, 10))
    SPECIES_B = list(range(10, 20))
    SPECIES_C = list(range(20, 30))
    PROBES = list(range(30, 35))
    SOURCE_A = 1.0
    SOURCE_B = 3.0
    SOURCE_C = -2.0

    def __init__(
        self,
        operators=None,
        temporal_order=0,
        grid_size=(128, 128),
        domain_size=50.0,
        dt=0.005,
        noise_std=0.0,
        noise_seed=None,
    ):
        self.operators = operators or [
            {"type": "laplacian", "params": {"strength": 1.0}}
        ]
        self.temporal_order = temporal_order
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.dt = dt
        self._init_noise(noise_std, noise_seed)
        rng = np.random.RandomState(42)
        self._bg_positions_rel = rng.uniform(-10, 10, (self.N_BACKGROUND, 2))
        self._bg_velocities = np.zeros((self.N_BACKGROUND, 2))

    def run(self, experiments: list[dict]) -> list[dict]:
        return [self._run_one(exp) for exp in experiments]

    def run_json(self, json_str: str) -> str:
        experiments = json.loads(json_str)
        results = self.run(experiments)
        return json.dumps(results, indent=2)

    def _run_one(self, exp: dict) -> dict:
        probe_pos_rel = np.array(exp["probe_positions"], dtype=np.float64)
        probe_vel = np.array(exp["probe_velocities"], dtype=np.float64)
        measurement_times = sorted(exp["measurement_times"])
        duration = float(exp.get("duration", max(measurement_times)))
        duration = max(duration, 5.0)
        assert probe_pos_rel.shape == (
            self.N_PROBES,
            2,
        ), f"Expected {self.N_PROBES} probe positions, got {probe_pos_rel.shape[0]}"
        assert probe_vel.shape == (
            self.N_PROBES,
            2,
        ), f"Expected {self.N_PROBES} probe velocities, got {probe_vel.shape[0]}"
        centre = self.domain_size / 2.0
        positions = np.vstack([self._bg_positions_rel + centre, probe_pos_rel + centre])
        velocities = np.vstack([self._bg_velocities, probe_vel])
        masses = np.ones(self.N_TOTAL)
        source_coupling = np.zeros(self.N_TOTAL)
        source_coupling[self.SPECIES_A] = self.SOURCE_A
        source_coupling[self.SPECIES_B] = self.SOURCE_B
        source_coupling[self.SPECIES_C] = self.SOURCE_C
        sim = FieldSampler(
            particle_inertia=masses,
            particle_source=masses,
            particle_force=masses,
            initial_positions=positions,
            initial_velocities=velocities,
            n_particles=self.N_TOTAL,
            spatial_dimensions=2,
            temporal_order=self.temporal_order,
            grid_size=self.grid_size,
            domain_size=self.domain_size,
            operators=self.operators,
            dt=self.dt,
            source_coupling=source_coupling,
            force_coupling=1.0,
            periodic_boundaries=False,
        )
        pos_traj, vel_traj = ([], [])
        recorded = set()
        n_steps = int(round(duration / self.dt))
        for i in range(n_steps + 1):
            t = round(i * self.dt, 10)
            for mt in measurement_times:
                if mt not in recorded and t >= mt:
                    pos_traj.append(
                        self._noisy_positions(sim.positions - centre).tolist()
                    )
                    vel_traj.append(sim.velocities.tolist())
                    recorded.add(mt)
            if len(recorded) == len(measurement_times):
                break
            if i < n_steps:
                sim.step()
        return {
            "measurement_times": measurement_times,
            "positions": pos_traj,
            "velocities": self._noisy_velocities(vel_traj, measurement_times),
            "background_initial_positions": self._bg_positions_rel.tolist(),
        }


class DarkMatterExecutor(_NoisyExecutorMixin):
    N_VISIBLE = 20
    N_DARK = 10
    N_PROBES = 5
    N_TOTAL = 35
    N_AGENT = 25
    VISIBLE = list(range(0, 20))
    DARK = list(range(20, 30))
    PROBES = list(range(30, 35))
    SOURCE_VISIBLE = 1.0
    SOURCE_DARK = 5.0

    def __init__(
        self,
        operators=None,
        temporal_order=0,
        grid_size=(128, 128),
        domain_size=50.0,
        dt=0.005,
        noise_std=0.0,
        noise_seed=None,
    ):
        self.operators = operators or [
            {"type": "laplacian", "params": {"strength": 1.0}}
        ]
        self.temporal_order = temporal_order
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.dt = dt
        self._init_noise(noise_std, noise_seed)
        rng = np.random.RandomState(123)
        vis_angles = rng.uniform(0, 2 * np.pi, self.N_VISIBLE)
        vis_radii = rng.uniform(8, 15, self.N_VISIBLE)
        self._visible_positions_rel = np.column_stack(
            [vis_radii * np.cos(vis_angles), vis_radii * np.sin(vis_angles)]
        )
        self._dark_positions_rel = rng.normal(0, 1.0, (self.N_DARK, 2))
        vis_r = np.linalg.norm(self._visible_positions_rel, axis=1)
        dark_r = np.linalg.norm(self._dark_positions_rel, axis=1)
        M_enclosed = np.zeros(self.N_VISIBLE)
        for i in range(self.N_VISIBLE):
            ri = vis_r[i]
            M_enclosed[i] = (
                np.sum(dark_r < ri) * self.SOURCE_DARK
                + np.sum(vis_r < ri) * self.SOURCE_VISIBLE
                - self.SOURCE_VISIBLE
            )
        v_circ = np.sqrt(np.maximum(M_enclosed, 0.0) / (2 * np.pi))
        r_safe = np.maximum(vis_r, 1e-06)
        tangent = (
            np.column_stack(
                [-self._visible_positions_rel[:, 1], self._visible_positions_rel[:, 0]]
            )
            / r_safe[:, None]
        )
        self._visible_velocities = v_circ[:, None] * tangent
        self._dark_velocities = np.zeros((self.N_DARK, 2))
        self._agent_indices = self.VISIBLE + self.PROBES

    def run(self, experiments: list[dict]) -> list[dict]:
        return [self._run_one(exp) for exp in experiments]

    def run_json(self, json_str: str) -> str:
        experiments = json.loads(json_str)
        results = self.run(experiments)
        return json.dumps(results, indent=2)

    def _run_one(self, exp: dict) -> dict:
        probe_pos_rel = np.array(exp["probe_positions"], dtype=np.float64)
        probe_vel = np.array(exp["probe_velocities"], dtype=np.float64)
        measurement_times = sorted(exp["measurement_times"])
        duration = float(exp.get("duration", max(measurement_times)))
        duration = max(duration, 10.0)
        vis_vel_sign = float(exp.get("visible_velocity_sign", 1.0))
        assert probe_pos_rel.shape == (
            self.N_PROBES,
            2,
        ), f"Expected {self.N_PROBES} probe positions, got {probe_pos_rel.shape[0]}"
        assert probe_vel.shape == (
            self.N_PROBES,
            2,
        ), f"Expected {self.N_PROBES} probe velocities, got {probe_vel.shape[0]}"
        centre = self.domain_size / 2.0
        positions = np.vstack(
            [
                self._visible_positions_rel + centre,
                self._dark_positions_rel + centre,
                probe_pos_rel + centre,
            ]
        )
        velocities = np.vstack(
            [vis_vel_sign * self._visible_velocities, self._dark_velocities, probe_vel]
        )
        masses = np.ones(self.N_TOTAL)
        source_coupling = np.zeros(self.N_TOTAL)
        source_coupling[self.VISIBLE] = self.SOURCE_VISIBLE
        source_coupling[self.DARK] = self.SOURCE_DARK
        sim = FieldSampler(
            particle_inertia=masses,
            particle_source=masses,
            particle_force=masses,
            initial_positions=positions,
            initial_velocities=velocities,
            n_particles=self.N_TOTAL,
            spatial_dimensions=2,
            temporal_order=self.temporal_order,
            grid_size=self.grid_size,
            domain_size=self.domain_size,
            operators=self.operators,
            dt=self.dt,
            source_coupling=source_coupling,
            force_coupling=1.0,
            periodic_boundaries=False,
        )
        pos_traj, vel_traj = ([], [])
        recorded = set()
        n_steps = int(round(duration / self.dt))
        for i in range(n_steps + 1):
            t = round(i * self.dt, 10)
            for mt in measurement_times:
                if mt not in recorded and t >= mt:
                    all_pos = sim.positions - centre
                    all_vel = sim.velocities
                    agent_pos = all_pos[self._agent_indices]
                    pos_traj.append(self._noisy_positions(agent_pos).tolist())
                    vel_traj.append(all_vel[self._agent_indices].tolist())
                    recorded.add(mt)
            if len(recorded) == len(measurement_times):
                break
            if i < n_steps:
                sim.step()
        return {
            "measurement_times": measurement_times,
            "positions": pos_traj,
            "velocities": self._noisy_velocities(vel_traj, measurement_times),
            "background_initial_positions": self._visible_positions_rel.tolist(),
        }

    def run_full(self, experiments: list[dict]) -> list[dict]:
        return [self._run_one_full(exp) for exp in experiments]

    def _run_one_full(self, exp: dict) -> dict:
        probe_pos_rel = np.array(exp["probe_positions"], dtype=np.float64)
        probe_vel = np.array(exp["probe_velocities"], dtype=np.float64)
        measurement_times = sorted(exp["measurement_times"])
        duration = float(exp.get("duration", max(measurement_times)))
        duration = max(duration, 10.0)
        vis_vel_sign = float(exp.get("visible_velocity_sign", 1.0))
        centre = self.domain_size / 2.0
        positions = np.vstack(
            [
                self._visible_positions_rel + centre,
                self._dark_positions_rel + centre,
                probe_pos_rel + centre,
            ]
        )
        velocities = np.vstack(
            [vis_vel_sign * self._visible_velocities, self._dark_velocities, probe_vel]
        )
        masses = np.ones(self.N_TOTAL)
        source_coupling = np.zeros(self.N_TOTAL)
        source_coupling[self.VISIBLE] = self.SOURCE_VISIBLE
        source_coupling[self.DARK] = self.SOURCE_DARK
        sim = FieldSampler(
            particle_inertia=masses,
            particle_source=masses,
            particle_force=masses,
            initial_positions=positions,
            initial_velocities=velocities,
            n_particles=self.N_TOTAL,
            spatial_dimensions=2,
            temporal_order=self.temporal_order,
            grid_size=self.grid_size,
            domain_size=self.domain_size,
            operators=self.operators,
            dt=self.dt,
            source_coupling=source_coupling,
            force_coupling=1.0,
            periodic_boundaries=False,
        )
        pos_traj, vel_traj = ([], [])
        field_snapshots = []
        recorded = set()
        n_steps = int(round(duration / self.dt))
        for i in range(n_steps + 1):
            t = round(i * self.dt, 10)
            for mt in measurement_times:
                if mt not in recorded and t >= mt:
                    pos_traj.append((sim.positions - centre).tolist())
                    vel_traj.append(sim.velocities.tolist())
                    field_snapshots.append(np.asarray(sim.field).tolist())
                    recorded.add(mt)
            if len(recorded) == len(measurement_times):
                break
            if i < n_steps:
                sim.step()
        return {
            "measurement_times": measurement_times,
            "positions": pos_traj,
            "velocities": self._noisy_velocities(vel_traj, measurement_times),
            "field_snapshots": field_snapshots,
            "dark_initial_positions": self._dark_positions_rel.tolist(),
            "background_initial_positions": self._visible_positions_rel.tolist(),
        }


_NBODY_INTEGRATOR_DEFAULT = "yoshida4"
_NBODY_SOFTENING_DEFAULT = 0.05


class NBodySimulationExecutor(_NoisyExecutorMixin):
    def __init__(
        self,
        operators=None,
        temporal_order=0,
        grid_size=None,
        domain_size=20.0,
        dt=0.005,
        noise_std=0.0,
        noise_seed=None,
        integrator=_NBODY_INTEGRATOR_DEFAULT,
        softening=_NBODY_SOFTENING_DEFAULT,
    ):
        self.operators = operators or [
            {"type": "laplacian", "params": {"strength": 1.0}}
        ]
        self.temporal_order = temporal_order
        self.grid_size = grid_size
        self.domain_size = float(domain_size)
        self.dt = float(dt)
        self.integrator = integrator
        self.softening = float(softening)
        self._init_noise(noise_std, noise_seed)
        _check_nbody_supports(temporal_order)
        self._force_law, self._potential_law = _operator_to_pairwise(self.operators)

    def run(self, experiments: list[dict]) -> list[dict]:
        return [self._run_one(exp) for exp in experiments]

    def run_json(self, json_str: str) -> str:
        return json.dumps(self.run(json.loads(json_str)), indent=2)

    def _run_one(self, exp: dict) -> dict:
        p1 = float(exp["p1"])
        p2 = float(exp["p2"])
        pos2 = list(exp["pos2"])
        velocity2 = list(exp["velocity2"])
        measurement_times = sorted(exp["measurement_times"])
        duration = float(exp.get("duration", max(measurement_times)))
        duration = max(duration, 5.0)
        centre = self.domain_size / 2.0
        init_positions = np.array(
            [[centre, centre], [centre + pos2[0], centre + pos2[1]]], dtype=np.float64
        )
        init_velocities = np.array([[0.0, 0.0], velocity2], dtype=np.float64)
        masses = np.array([1000000000000000.0, p2], dtype=np.float64)
        source_charges = np.array([p1, 1.0], dtype=np.float64)
        force_charges = np.array([0.0, 1.0], dtype=np.float64)
        sim = NBodySampler(
            masses=masses,
            source_charges=source_charges,
            force_charges=force_charges,
            initial_positions=init_positions,
            initial_velocities=init_velocities,
            force_law=self._force_law,
            potential_law=self._potential_law,
            integrator=self.integrator,
            dt=self.dt,
            softening=self.softening,
            spatial_dimensions=2,
        )
        positions, velocities = _record_at_times(
            sim, self.dt, duration, measurement_times
        )
        pos1 = self._noisy_positions(positions[:, 0, :] - centre)
        pos2_arr = self._noisy_positions(positions[:, 1, :] - centre)
        return {
            "measurement_times": measurement_times,
            "pos1": pos1.tolist(),
            "pos2": pos2_arr.tolist(),
            "velocity1": self._noisy_velocities(velocities[:, 0, :], measurement_times),
            "velocity2": self._noisy_velocities(velocities[:, 1, :], measurement_times),
        }


class NBodyCircleExecutor(_NoisyExecutorMixin):
    N_RING = 10
    N_TOTAL = 11
    ALPHA = 0.75

    def __init__(
        self,
        operators=None,
        temporal_order=0,
        grid_size=None,
        domain_size=50.0,
        dt=0.005,
        noise_std=0.0,
        noise_seed=None,
        integrator=_NBODY_INTEGRATOR_DEFAULT,
        softening=_NBODY_SOFTENING_DEFAULT,
    ):
        self.operators = operators or [
            {
                "type": "fractional_laplacian",
                "params": {"strength": 1.0, "alpha": self.ALPHA},
            }
        ]
        self.temporal_order = temporal_order
        self.grid_size = grid_size
        self.domain_size = float(domain_size)
        self.dt = float(dt)
        self.integrator = integrator
        self.softening = float(softening)
        self._init_noise(noise_std, noise_seed)
        _check_nbody_supports(temporal_order)
        self._force_law, self._potential_law = _operator_to_pairwise(self.operators)

    def run(self, experiments):
        return [self._run_one(e) for e in experiments]

    def run_json(self, s):
        return json.dumps(self.run(json.loads(s)), indent=2)

    def _run_one(self, exp):
        ring_radius = float(exp.get("ring_radius", 5.0))
        v_tang = float(exp.get("initial_tangential_velocity", 0.0))
        measurement_times = sorted(exp["measurement_times"])
        duration = max(float(exp.get("duration", max(measurement_times))), 5.0)
        centre = self.domain_size / 2.0
        angles = np.linspace(0, 2 * np.pi, self.N_RING, endpoint=False)
        ring_pos = np.column_stack(
            [
                centre + ring_radius * np.cos(angles),
                centre + ring_radius * np.sin(angles),
            ]
        )
        positions = np.vstack([[[centre, centre]], ring_pos])
        ring_vel = np.column_stack([-v_tang * np.sin(angles), v_tang * np.cos(angles)])
        velocities = np.vstack([[[0.0, 0.0]], ring_vel])
        masses = np.ones(self.N_TOTAL)
        source_charges = np.ones(self.N_TOTAL)
        force_charges = np.ones(self.N_TOTAL)
        sim = NBodySampler(
            masses=masses,
            source_charges=source_charges,
            force_charges=force_charges,
            initial_positions=positions,
            initial_velocities=velocities,
            force_law=self._force_law,
            potential_law=self._potential_law,
            integrator=self.integrator,
            dt=self.dt,
            softening=self.softening,
            spatial_dimensions=2,
        )
        positions_rec, velocities_rec = _record_at_times(
            sim, self.dt, duration, measurement_times
        )
        pos_traj = [self._noisy_positions(p - centre).tolist() for p in positions_rec]
        vel_traj = [v.tolist() for v in velocities_rec]
        return {
            "measurement_times": measurement_times,
            "positions": pos_traj,
            "velocities": self._noisy_velocities(vel_traj, measurement_times),
        }


class NBodyThreeSpeciesExecutor(_NoisyExecutorMixin):
    N_BACKGROUND = 30
    N_PROBES = 5
    N_TOTAL = 35
    SPECIES_A = list(range(0, 10))
    SPECIES_B = list(range(10, 20))
    SPECIES_C = list(range(20, 30))
    PROBES = list(range(30, 35))
    SOURCE_A = 1.0
    SOURCE_B = 3.0
    SOURCE_C = -2.0

    def __init__(
        self,
        operators=None,
        temporal_order=0,
        grid_size=None,
        domain_size=50.0,
        dt=0.005,
        noise_std=0.0,
        noise_seed=None,
        integrator=_NBODY_INTEGRATOR_DEFAULT,
        softening=_NBODY_SOFTENING_DEFAULT,
    ):
        self.operators = operators or [
            {"type": "laplacian", "params": {"strength": 1.0}}
        ]
        self.temporal_order = temporal_order
        self.grid_size = grid_size
        self.domain_size = float(domain_size)
        self.dt = float(dt)
        self.integrator = integrator
        self.softening = float(softening)
        self._init_noise(noise_std, noise_seed)
        _check_nbody_supports(temporal_order)
        self._force_law, self._potential_law = _operator_to_pairwise(self.operators)
        rng = np.random.RandomState(42)
        self._bg_positions_rel = rng.uniform(-10, 10, (self.N_BACKGROUND, 2))
        self._bg_velocities = np.zeros((self.N_BACKGROUND, 2))

    def run(self, experiments):
        return [self._run_one(e) for e in experiments]

    def run_json(self, s):
        return json.dumps(self.run(json.loads(s)), indent=2)

    def _run_one(self, exp):
        probe_pos_rel = np.array(exp["probe_positions"], dtype=np.float64)
        probe_vel = np.array(exp["probe_velocities"], dtype=np.float64)
        measurement_times = sorted(exp["measurement_times"])
        duration = max(float(exp.get("duration", max(measurement_times))), 5.0)
        assert probe_pos_rel.shape == (self.N_PROBES, 2)
        assert probe_vel.shape == (self.N_PROBES, 2)
        centre = self.domain_size / 2.0
        positions = np.vstack([self._bg_positions_rel + centre, probe_pos_rel + centre])
        velocities = np.vstack([self._bg_velocities, probe_vel])
        masses = np.ones(self.N_TOTAL)
        source_charges = np.zeros(self.N_TOTAL)
        source_charges[self.SPECIES_A] = self.SOURCE_A
        source_charges[self.SPECIES_B] = self.SOURCE_B
        source_charges[self.SPECIES_C] = self.SOURCE_C
        force_charges = np.ones(self.N_TOTAL)
        sim = NBodySampler(
            masses=masses,
            source_charges=source_charges,
            force_charges=force_charges,
            initial_positions=positions,
            initial_velocities=velocities,
            force_law=self._force_law,
            potential_law=self._potential_law,
            integrator=self.integrator,
            dt=self.dt,
            softening=self.softening,
            spatial_dimensions=2,
        )
        positions_rec, velocities_rec = _record_at_times(
            sim, self.dt, duration, measurement_times
        )
        return {
            "measurement_times": measurement_times,
            "positions": [
                self._noisy_positions(p - centre).tolist() for p in positions_rec
            ],
            "velocities": self._noisy_velocities(velocities_rec, measurement_times),
            "background_initial_positions": self._bg_positions_rel.tolist(),
        }


class NBodyDarkMatterExecutor(_NoisyExecutorMixin):
    N_VISIBLE = 20
    N_DARK = 10
    N_PROBES = 5
    N_TOTAL = 35
    N_AGENT = 25
    VISIBLE = list(range(0, 20))
    DARK = list(range(20, 30))
    PROBES = list(range(30, 35))
    SOURCE_VISIBLE = 1.0
    SOURCE_DARK = 5.0

    def __init__(
        self,
        operators=None,
        temporal_order=0,
        grid_size=None,
        domain_size=50.0,
        dt=0.005,
        noise_std=0.0,
        noise_seed=None,
        integrator=_NBODY_INTEGRATOR_DEFAULT,
        softening=_NBODY_SOFTENING_DEFAULT,
    ):
        self.operators = operators or [
            {"type": "laplacian", "params": {"strength": 1.0}}
        ]
        self.temporal_order = temporal_order
        self.grid_size = grid_size
        self.domain_size = float(domain_size)
        self.dt = float(dt)
        self.integrator = integrator
        self.softening = float(softening)
        self._init_noise(noise_std, noise_seed)
        _check_nbody_supports(temporal_order)
        self._force_law, self._potential_law = _operator_to_pairwise(self.operators)
        rng = np.random.RandomState(123)
        vis_angles = rng.uniform(0, 2 * np.pi, self.N_VISIBLE)
        vis_radii = rng.uniform(8, 15, self.N_VISIBLE)
        self._visible_positions_rel = np.column_stack(
            [vis_radii * np.cos(vis_angles), vis_radii * np.sin(vis_angles)]
        )
        self._dark_positions_rel = rng.normal(0, 1.0, (self.N_DARK, 2))
        vis_r = np.linalg.norm(self._visible_positions_rel, axis=1)
        dark_r = np.linalg.norm(self._dark_positions_rel, axis=1)
        Q_enc = np.zeros(self.N_VISIBLE)
        for i in range(self.N_VISIBLE):
            ri = vis_r[i]
            Q_enc[i] = (
                np.sum(dark_r < ri) * self.SOURCE_DARK
                + np.sum(vis_r < ri) * self.SOURCE_VISIBLE
                - self.SOURCE_VISIBLE
            )
        v_circ = np.sqrt(np.maximum(Q_enc, 0.0) / (2 * np.pi))
        r_safe = np.maximum(vis_r, 1e-06)
        tangent = (
            np.column_stack(
                [-self._visible_positions_rel[:, 1], self._visible_positions_rel[:, 0]]
            )
            / r_safe[:, None]
        )
        self._visible_velocities = v_circ[:, None] * tangent
        self._dark_velocities = np.zeros((self.N_DARK, 2))
        self._agent_indices = self.VISIBLE + self.PROBES

    def run(self, experiments):
        return [self._run_one(e) for e in experiments]

    def run_json(self, s):
        return json.dumps(self.run(json.loads(s)), indent=2)

    def _build_sim(self, probe_pos_rel, probe_vel, vis_vel_sign):
        centre = self.domain_size / 2.0
        positions = np.vstack(
            [
                self._visible_positions_rel + centre,
                self._dark_positions_rel + centre,
                probe_pos_rel + centre,
            ]
        )
        velocities = np.vstack(
            [vis_vel_sign * self._visible_velocities, self._dark_velocities, probe_vel]
        )
        masses = np.ones(self.N_TOTAL)
        source_charges = np.zeros(self.N_TOTAL)
        source_charges[self.VISIBLE] = self.SOURCE_VISIBLE
        source_charges[self.DARK] = self.SOURCE_DARK
        force_charges = np.ones(self.N_TOTAL)
        sim = NBodySampler(
            masses=masses,
            source_charges=source_charges,
            force_charges=force_charges,
            initial_positions=positions,
            initial_velocities=velocities,
            force_law=self._force_law,
            potential_law=self._potential_law,
            integrator=self.integrator,
            dt=self.dt,
            softening=self.softening,
            spatial_dimensions=2,
        )
        return (sim, centre)

    def _run_one(self, exp):
        probe_pos_rel = np.array(exp["probe_positions"], dtype=np.float64)
        probe_vel = np.array(exp["probe_velocities"], dtype=np.float64)
        measurement_times = sorted(exp["measurement_times"])
        duration = max(float(exp.get("duration", max(measurement_times))), 10.0)
        vis_vel_sign = float(exp.get("visible_velocity_sign", 1.0))
        assert probe_pos_rel.shape == (self.N_PROBES, 2)
        assert probe_vel.shape == (self.N_PROBES, 2)
        sim, centre = self._build_sim(probe_pos_rel, probe_vel, vis_vel_sign)
        positions_rec, velocities_rec = _record_at_times(
            sim, self.dt, duration, measurement_times
        )
        return {
            "measurement_times": measurement_times,
            "positions": [
                self._noisy_positions(p[self._agent_indices] - centre).tolist()
                for p in positions_rec
            ],
            "velocities": self._noisy_velocities(
                [v[self._agent_indices] for v in velocities_rec], measurement_times
            ),
            "background_initial_positions": self._visible_positions_rel.tolist(),
        }

    def run_full(self, experiments: list[dict]) -> list[dict]:
        return [self._run_one_full(e) for e in experiments]

    def _run_one_full(self, exp: dict) -> dict:
        probe_pos_rel = np.array(exp["probe_positions"], dtype=np.float64)
        probe_vel = np.array(exp["probe_velocities"], dtype=np.float64)
        measurement_times = sorted(exp["measurement_times"])
        duration = max(float(exp.get("duration", max(measurement_times))), 10.0)
        vis_vel_sign = float(exp.get("visible_velocity_sign", 1.0))
        assert probe_pos_rel.shape == (self.N_PROBES, 2)
        assert probe_vel.shape == (self.N_PROBES, 2)
        sim, centre = self._build_sim(probe_pos_rel, probe_vel, vis_vel_sign)
        positions_rec, velocities_rec = _record_at_times(
            sim, self.dt, duration, measurement_times
        )
        return {
            "measurement_times": measurement_times,
            "positions": [(p - centre).tolist() for p in positions_rec],
            "velocities": self._noisy_velocities(velocities_rec, measurement_times),
            "field_snapshots": [],
            "dark_initial_positions": self._dark_positions_rel.tolist(),
            "background_initial_positions": self._visible_positions_rel.tolist(),
        }


class NBodyEtherExecutor(_NoisyExecutorMixin):
    N_BACKGROUND = 21
    N_RING = 20
    N_PROBES = 5
    N_TOTAL = 26
    ANCHOR_INDEX = 0
    RING_INDICES = list(range(1, 21))
    PROBE_INDICES = list(range(21, 26))
    ANCHOR_MASS = 1000000000000000.0
    ANCHOR_SOURCE = 50.0
    RING_RADIUS = 5.0
    MASS_PATTERN = (1.0, 2.0, 4.0)
    DEFAULT_PROBE_MASS = 1.0
    ETHER_ALPHA = 0.05

    def __init__(
        self,
        operators=None,
        temporal_order=0,
        grid_size=None,
        domain_size=50.0,
        dt=0.005,
        noise_std=0.0,
        noise_seed=None,
        integrator=_NBODY_INTEGRATOR_DEFAULT,
        softening=_NBODY_SOFTENING_DEFAULT,
    ):
        self.operators = operators or [
            {"type": "laplacian", "params": {"strength": 1.0}}
        ]
        self.temporal_order = temporal_order
        self.grid_size = grid_size
        self.domain_size = float(domain_size)
        self.dt = float(dt)
        self.integrator = integrator
        self.softening = float(softening)
        self._init_noise(noise_std, noise_seed)
        _check_nbody_supports(temporal_order)
        self._force_law, self._potential_law = _operator_to_pairwise(self.operators)
        angles = np.linspace(0, 2 * np.pi, self.N_RING, endpoint=False)
        self._ring_positions_rel = np.column_stack(
            [self.RING_RADIUS * np.cos(angles), self.RING_RADIUS * np.sin(angles)]
        )
        self._ring_masses = np.array(
            [self.MASS_PATTERN[i % len(self.MASS_PATTERN)] for i in range(self.N_RING)],
            dtype=np.float64,
        )
        v_circ = np.sqrt(self.ANCHOR_SOURCE / (2 * np.pi))
        self._ring_velocities = v_circ * np.column_stack(
            [-np.sin(angles), np.cos(angles)]
        )
        self._bg_positions_rel = np.vstack(
            [np.array([[0.0, 0.0]]), self._ring_positions_rel]
        )
        self._bg_velocities = np.vstack([np.array([[0.0, 0.0]]), self._ring_velocities])
        self._bg_masses = np.concatenate(
            [np.array([self.ANCHOR_MASS]), self._ring_masses]
        )

    def run(self, experiments):
        return [self._run_one(e) for e in experiments]

    def run_json(self, s):
        return json.dumps(self.run(json.loads(s)), indent=2)

    def _run_one(self, exp):
        probe_pos_rel = np.array(exp["probe_positions"], dtype=np.float64)
        probe_vel = np.array(exp["probe_velocities"], dtype=np.float64)
        measurement_times = sorted(exp["measurement_times"])
        duration = max(float(exp.get("duration", max(measurement_times))), 5.0)
        assert probe_pos_rel.shape == (self.N_PROBES, 2)
        assert probe_vel.shape == (self.N_PROBES, 2)
        if "probe_masses" in exp and exp["probe_masses"] is not None:
            probe_masses = np.array(exp["probe_masses"], dtype=np.float64)
            assert probe_masses.shape == (self.N_PROBES,)
        else:
            probe_masses = np.full(
                self.N_PROBES, self.DEFAULT_PROBE_MASS, dtype=np.float64
            )
        centre = self.domain_size / 2.0
        positions = np.vstack([self._bg_positions_rel + centre, probe_pos_rel + centre])
        velocities = np.vstack([self._bg_velocities, probe_vel])
        masses = np.concatenate([self._bg_masses, probe_masses])
        source_charges = np.zeros(self.N_TOTAL)
        source_charges[self.ANCHOR_INDEX] = self.ANCHOR_SOURCE
        force_charges = np.zeros(self.N_TOTAL)
        force_charges[self.RING_INDICES] = self._ring_masses
        force_charges[self.PROBE_INDICES] = probe_masses
        sim = NBodySampler(
            masses=masses,
            source_charges=source_charges,
            force_charges=force_charges,
            initial_positions=positions,
            initial_velocities=velocities,
            force_law=self._force_law,
            potential_law=self._potential_law,
            integrator=self.integrator,
            dt=self.dt,
            softening=self.softening,
            spatial_dimensions=2,
            external_acceleration=np.array([0.0, self.ETHER_ALPHA]),
        )
        positions_rec, velocities_rec = _record_at_times(
            sim, self.dt, duration, measurement_times
        )
        return {
            "measurement_times": measurement_times,
            "positions": [
                self._noisy_positions(p - centre).tolist() for p in positions_rec
            ],
            "velocities": self._noisy_velocities(velocities_rec, measurement_times),
            "particle_masses": masses.tolist(),
            "background_initial_positions": self._bg_positions_rel.tolist(),
            "background_initial_velocities": self._bg_velocities.tolist(),
        }


class NBodyHubbleExecutor(_NoisyExecutorMixin):
    N_BACKGROUND = 21
    N_RING = 20
    N_PROBES = 5
    N_TOTAL = 26
    ANCHOR_INDEX = 0
    RING_INDICES = list(range(1, 21))
    PROBE_INDICES = list(range(21, 26))
    ANCHOR_MASS = 1000000000000000.0
    ANCHOR_SOURCE = 50.0
    RING_RADIUS = 5.0
    MASS_PATTERN = (1.0, 2.0, 4.0)
    DEFAULT_PROBE_MASS = 1.0
    HUBBLE_H = 0.05

    def __init__(
        self,
        operators=None,
        temporal_order=0,
        grid_size=None,
        domain_size=50.0,
        dt=0.005,
        noise_std=0.0,
        noise_seed=None,
        integrator=_NBODY_INTEGRATOR_DEFAULT,
        softening=_NBODY_SOFTENING_DEFAULT,
    ):
        self.operators = operators or [
            {"type": "laplacian", "params": {"strength": 1.0}}
        ]
        self.temporal_order = temporal_order
        self.grid_size = grid_size
        self.domain_size = float(domain_size)
        self.dt = float(dt)
        self.integrator = integrator
        self.softening = float(softening)
        self._init_noise(noise_std, noise_seed)
        _check_nbody_supports(temporal_order)
        self._force_law, self._potential_law = _operator_to_pairwise(self.operators)
        angles = np.linspace(0, 2 * np.pi, self.N_RING, endpoint=False)
        self._ring_positions_rel = np.column_stack(
            [self.RING_RADIUS * np.cos(angles), self.RING_RADIUS * np.sin(angles)]
        )
        self._ring_masses = np.array(
            [self.MASS_PATTERN[i % len(self.MASS_PATTERN)] for i in range(self.N_RING)],
            dtype=np.float64,
        )
        v_circ_sq = (
            self.ANCHOR_SOURCE / (2 * np.pi) - self.HUBBLE_H * self.RING_RADIUS**2
        )
        if v_circ_sq <= 0:
            raise ValueError(
                f"Hubble outward force exceeds central gravity at r={self.RING_RADIUS}; reduce HUBBLE_H or increase ANCHOR_SOURCE."
            )
        v_circ = float(np.sqrt(v_circ_sq))
        self._ring_velocities = v_circ * np.column_stack(
            [-np.sin(angles), np.cos(angles)]
        )
        self._bg_positions_rel = np.vstack(
            [np.array([[0.0, 0.0]]), self._ring_positions_rel]
        )
        self._bg_velocities = np.vstack([np.array([[0.0, 0.0]]), self._ring_velocities])
        self._bg_masses = np.concatenate(
            [np.array([self.ANCHOR_MASS]), self._ring_masses]
        )

    def run(self, experiments):
        return [self._run_one(e) for e in experiments]

    def run_json(self, s):
        return json.dumps(self.run(json.loads(s)), indent=2)

    def _run_one(self, exp):
        probe_pos_rel = np.array(exp["probe_positions"], dtype=np.float64)
        probe_vel = np.array(exp["probe_velocities"], dtype=np.float64)
        measurement_times = sorted(exp["measurement_times"])
        duration = max(float(exp.get("duration", max(measurement_times))), 5.0)
        assert probe_pos_rel.shape == (self.N_PROBES, 2)
        assert probe_vel.shape == (self.N_PROBES, 2)
        if "probe_masses" in exp and exp["probe_masses"] is not None:
            probe_masses = np.array(exp["probe_masses"], dtype=np.float64)
            assert probe_masses.shape == (self.N_PROBES,)
        else:
            probe_masses = np.full(
                self.N_PROBES, self.DEFAULT_PROBE_MASS, dtype=np.float64
            )
        centre = self.domain_size / 2.0
        positions = np.vstack([self._bg_positions_rel + centre, probe_pos_rel + centre])
        velocities = np.vstack([self._bg_velocities, probe_vel])
        masses = np.concatenate([self._bg_masses, probe_masses])
        source_charges = np.zeros(self.N_TOTAL)
        source_charges[self.ANCHOR_INDEX] = self.ANCHOR_SOURCE
        force_charges = np.zeros(self.N_TOTAL)
        force_charges[self.RING_INDICES] = self._ring_masses
        force_charges[self.PROBE_INDICES] = probe_masses
        H = self.HUBBLE_H
        centre_vec = jnp.array([centre, centre], dtype=jnp.float64)

        def hubble_flow(pos):
            return H * (pos - centre_vec[None, :])

        sim = NBodySampler(
            masses=masses,
            source_charges=source_charges,
            force_charges=force_charges,
            initial_positions=positions,
            initial_velocities=velocities,
            force_law=self._force_law,
            potential_law=self._potential_law,
            integrator=self.integrator,
            dt=self.dt,
            softening=self.softening,
            spatial_dimensions=2,
            external_acceleration=hubble_flow,
        )
        positions_rec, velocities_rec = _record_at_times(
            sim, self.dt, duration, measurement_times
        )
        return {
            "measurement_times": measurement_times,
            "positions": [
                self._noisy_positions(p - centre).tolist() for p in positions_rec
            ],
            "velocities": self._noisy_velocities(velocities_rec, measurement_times),
            "particle_masses": masses.tolist(),
            "background_initial_positions": self._bg_positions_rel.tolist(),
            "background_initial_velocities": self._bg_velocities.tolist(),
        }


class NBodyOscillatorExecutor(_NoisyExecutorMixin):
    G_0 = 5.0
    OMEGA = float(np.pi / 2.0)
    PHI = 0.0

    def __init__(
        self,
        operators=None,
        temporal_order=0,
        grid_size=None,
        domain_size=20.0,
        dt=0.005,
        noise_std=0.0,
        noise_seed=None,
        integrator=_NBODY_INTEGRATOR_DEFAULT,
        softening=_NBODY_SOFTENING_DEFAULT,
    ):
        self.operators = operators
        self.temporal_order = temporal_order
        self.grid_size = grid_size
        self.domain_size = float(domain_size)
        self.dt = float(dt)
        self.integrator = integrator
        self.softening = float(softening)
        self._init_noise(noise_std, noise_seed)
        self._force_law = lambda r, qi, qj, mi, mj: poisson_2d_force(
            r, qi, qj, mi, mj, G=1.0
        )
        self._potential_law = lambda r, qi, qj, mi, mj: poisson_2d_potential(
            r, qi, qj, mi, mj, G=1.0
        )

    @classmethod
    def coupling(cls, t):
        return cls.G_0 * np.cos(cls.OMEGA * np.asarray(t, dtype=np.float64) + cls.PHI)

    def run(self, experiments: list[dict]) -> list[dict]:
        return [self._run_one(e) for e in experiments]

    def run_json(self, json_str: str) -> str:
        return json.dumps(self.run(json.loads(json_str)), indent=2)

    def _run_one(self, exp: dict) -> dict:
        p1 = float(exp["p1"])
        p2 = float(exp["p2"])
        pos2 = list(exp["pos2"])
        velocity2 = list(exp["velocity2"])
        measurement_times = sorted(exp["measurement_times"])
        duration = float(exp.get("duration", max(measurement_times)))
        duration = max(duration, 5.0)
        t0 = float(exp.get("start_time", 0.0))
        centre = self.domain_size / 2.0
        init_positions = np.array(
            [[centre, centre], [centre + pos2[0], centre + pos2[1]]], dtype=np.float64
        )
        init_velocities = np.array([[0.0, 0.0], velocity2], dtype=np.float64)
        masses = np.array([1000000000000000.0, p2], dtype=np.float64)
        source_charges = np.array([p1, 1.0], dtype=np.float64)
        force_charges = np.array([0.0, 1.0], dtype=np.float64)
        G_0 = self.G_0
        omega = self.OMEGA
        phi = self.PHI

        def coupling_fn(t):
            return G_0 * jnp.cos(omega * t + phi)

        sim = NBodySampler(
            masses=masses,
            source_charges=source_charges,
            force_charges=force_charges,
            initial_positions=init_positions,
            initial_velocities=init_velocities,
            force_law=self._force_law,
            potential_law=self._potential_law,
            integrator=self.integrator,
            dt=self.dt,
            softening=self.softening,
            spatial_dimensions=2,
            force_modulation=coupling_fn,
            initial_time=t0,
        )
        positions, velocities = _record_at_times(
            sim, self.dt, duration, measurement_times
        )
        pos1 = self._noisy_positions(positions[:, 0, :] - centre)
        pos2_arr = self._noisy_positions(positions[:, 1, :] - centre)
        return {
            "measurement_times": measurement_times,
            "pos1": pos1.tolist(),
            "pos2": pos2_arr.tolist(),
            "velocity1": self._noisy_velocities(velocities[:, 0, :], measurement_times),
            "velocity2": self._noisy_velocities(velocities[:, 1, :], measurement_times),
        }


class NBodyExtraDimensionsExecutor(_NoisyExecutorMixin):
    G = 1.0
    R_COMPACT = 0.5
    N_IMAGES = 20

    def __init__(
        self,
        operators=None,
        temporal_order=0,
        grid_size=None,
        domain_size=20.0,
        dt=0.005,
        noise_std=0.0,
        noise_seed=None,
        integrator=_NBODY_INTEGRATOR_DEFAULT,
        softening=_NBODY_SOFTENING_DEFAULT,
    ):
        self.operators = operators
        self.temporal_order = temporal_order
        self.grid_size = grid_size
        self.domain_size = float(domain_size)
        self.dt = float(dt)
        self.integrator = integrator
        self.softening = float(softening)
        self._init_noise(noise_std, noise_seed)
        G = self.G
        R = self.R_COMPACT
        N = self.N_IMAGES
        self._force_law = lambda r, qi, qj, mi, mj: extra_dimensions_2d_force(
            r, qi, qj, mi, mj, G=G, R_compact=R, n_images=N
        )
        self._potential_law = lambda r, qi, qj, mi, mj: extra_dimensions_2d_potential(
            r, qi, qj, mi, mj, G=G, R_compact=R, n_images=N
        )

    @classmethod
    def force_magnitude(cls, r, q_i=1.0, q_j=1.0):
        L = 2.0 * np.pi * cls.R_COMPACT
        n_arr = np.arange(-cls.N_IMAGES, cls.N_IMAGES + 1, dtype=np.float64)
        y_n = n_arr * L
        r_arr = np.asarray(r, dtype=np.float64)
        denom = (r_arr[..., None] ** 2 + y_n**2) ** 1.5
        F_geom = np.sum(r_arr[..., None] / denom, axis=-1)
        return cls.G * L * q_i * q_j * F_geom / (4.0 * np.pi)

    def run(self, experiments: list[dict]) -> list[dict]:
        return [self._run_one(e) for e in experiments]

    def run_json(self, json_str: str) -> str:
        return json.dumps(self.run(json.loads(json_str)), indent=2)

    def _run_one(self, exp: dict) -> dict:
        p1 = float(exp["p1"])
        p2 = float(exp["p2"])
        pos2 = list(exp["pos2"])
        velocity2 = list(exp["velocity2"])
        measurement_times = sorted(exp["measurement_times"])
        duration = float(exp.get("duration", max(measurement_times)))
        duration = max(duration, 5.0)
        centre = self.domain_size / 2.0
        init_positions = np.array(
            [[centre, centre], [centre + pos2[0], centre + pos2[1]]], dtype=np.float64
        )
        init_velocities = np.array([[0.0, 0.0], velocity2], dtype=np.float64)
        masses = np.array([1000000000000000.0, p2], dtype=np.float64)
        source_charges = np.array([p1, 1.0], dtype=np.float64)
        force_charges = np.array([0.0, 1.0], dtype=np.float64)
        sim = NBodySampler(
            masses=masses,
            source_charges=source_charges,
            force_charges=force_charges,
            initial_positions=init_positions,
            initial_velocities=init_velocities,
            force_law=self._force_law,
            potential_law=self._potential_law,
            integrator=self.integrator,
            dt=self.dt,
            softening=self.softening,
            spatial_dimensions=2,
        )
        positions, velocities = _record_at_times(
            sim, self.dt, duration, measurement_times
        )
        pos1 = self._noisy_positions(positions[:, 0, :] - centre)
        pos2_arr = self._noisy_positions(positions[:, 1, :] - centre)
        return {
            "measurement_times": measurement_times,
            "pos1": pos1.tolist(),
            "pos2": pos2_arr.tolist(),
            "velocity1": self._noisy_velocities(velocities[:, 0, :], measurement_times),
            "velocity2": self._noisy_velocities(velocities[:, 1, :], measurement_times),
        }


class NBodyCoulombEasyExecutor(_NoisyExecutorMixin):
    def __init__(
        self,
        operators=None,
        temporal_order=0,
        grid_size=None,
        domain_size=20.0,
        dt=0.005,
        noise_std=0.0,
        noise_seed=None,
        integrator=_NBODY_INTEGRATOR_DEFAULT,
        softening=_NBODY_SOFTENING_DEFAULT,
    ):
        self.operators = operators or [{"type": "coulomb", "params": {"strength": 1.0}}]
        self.temporal_order = temporal_order
        self.grid_size = grid_size
        self.domain_size = float(domain_size)
        self.dt = float(dt)
        self.integrator = integrator
        self.softening = float(softening)
        self._init_noise(noise_std, noise_seed)
        _check_nbody_supports(temporal_order)
        self._force_law, self._potential_law = _operator_to_pairwise(self.operators)

    def run(self, experiments: list[dict]) -> list[dict]:
        return [self._run_one(exp) for exp in experiments]

    def run_json(self, json_str: str) -> str:
        return json.dumps(self.run(json.loads(json_str)), indent=2)

    def _run_one(self, exp: dict) -> dict:
        p1 = float(exp["p1"])
        p2 = float(exp["p2"])
        pos2 = list(exp["pos2"])
        velocity2 = list(exp["velocity2"])
        measurement_times = sorted(exp["measurement_times"])
        duration = float(exp.get("duration", max(measurement_times)))
        duration = max(duration, 5.0)
        centre = self.domain_size / 2.0
        init_positions = np.array(
            [[centre, centre], [centre + pos2[0], centre + pos2[1]]], dtype=np.float64
        )
        init_velocities = np.array([[0.0, 0.0], velocity2], dtype=np.float64)
        masses = np.array([1000000000000000.0, 1.0], dtype=np.float64)
        source_charges = np.array([+abs(p1), -abs(p2)], dtype=np.float64)
        force_charges = np.array([0.0, -abs(p2)], dtype=np.float64)
        sim = NBodySampler(
            masses=masses,
            source_charges=source_charges,
            force_charges=force_charges,
            initial_positions=init_positions,
            initial_velocities=init_velocities,
            force_law=self._force_law,
            potential_law=self._potential_law,
            integrator=self.integrator,
            dt=self.dt,
            softening=self.softening,
            spatial_dimensions=2,
        )
        positions, velocities = _record_at_times(
            sim, self.dt, duration, measurement_times
        )
        pos1 = self._noisy_positions(positions[:, 0, :] - centre)
        pos2_arr = self._noisy_positions(positions[:, 1, :] - centre)
        return {
            "measurement_times": measurement_times,
            "pos1": pos1.tolist(),
            "pos2": pos2_arr.tolist(),
            "velocity1": self._noisy_velocities(velocities[:, 0, :], measurement_times),
            "velocity2": self._noisy_velocities(velocities[:, 1, :], measurement_times),
            "particle_charges": source_charges.tolist(),
        }
