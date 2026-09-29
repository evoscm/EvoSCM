from __future__ import annotations
import inspect
from typing import Callable, Optional
import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint

jax.config.update("jax_enable_x64", True)


def _wrap_for_time(fn: Callable, role: str) -> Callable:
    try:
        n_params = len(
            [
                p
                for p in inspect.signature(fn).parameters.values()
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
        )
    except (ValueError, TypeError):
        n_params = 2
    if n_params == 1:
        return lambda pos, t, _f=fn: _f(pos)
    if n_params == 2:
        return fn
    raise ValueError(
        f"{role} callable must accept either (positions,) or (positions, time); got {n_params}-arg signature."
    )


def _pairwise_displacements(positions, softening):
    diff = positions[None, :, :] - positions[:, None, :]
    r2 = jnp.sum(diff * diff, axis=-1)
    r_mag = jnp.sqrt(r2)
    r_eff = jnp.sqrt(r2 + softening**2)
    return (diff, r_mag, r_eff)


def make_acceleration_fn(force_law: Callable, softening: float) -> Callable:
    soft = float(softening)

    def accelerations(positions, source_charges, force_charges, masses):
        n = positions.shape[0]
        diff, r_mag, r_eff = _pairwise_displacements(positions, soft)
        eye = jnp.eye(n, dtype=bool)
        r_eff_safe = jnp.where(eye, 1.0, r_eff)
        r_mag_safe = jnp.where(eye, 1.0, r_mag)
        Q_recv = force_charges[:, None]
        Q_send = source_charges[None, :]
        M_i, M_j = (masses[:, None], masses[None, :])
        F_mag = force_law(r_eff_safe, Q_recv, Q_send, M_i, M_j)
        F_mag = jnp.where(eye, 0.0, F_mag)
        F_vec = F_mag[..., None] * diff / r_mag_safe[..., None]
        F_total = jnp.sum(F_vec, axis=1)
        return F_total / masses[:, None]

    return accelerations


def make_potential_fn(potential_law: Callable, softening: float) -> Callable:
    soft = float(softening)

    def potential_energy(positions, source_charges, masses):
        n = positions.shape[0]
        _, _, r_eff = _pairwise_displacements(positions, soft)
        eye = jnp.eye(n, dtype=bool)
        r_eff_safe = jnp.where(eye, 1.0, r_eff)
        Q_i, Q_j = (source_charges[:, None], source_charges[None, :])
        M_i, M_j = (masses[:, None], masses[None, :])
        V_pair = potential_law(r_eff_safe, Q_i, Q_j, M_i, M_j)
        upper = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
        return jnp.sum(jnp.where(upper, V_pair, 0.0))

    return potential_energy


_W4 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_YOSHIDA4_C = (0.5 * _W4, 0.5 * (1.0 - _W4), 0.5 * (1.0 - _W4), 0.5 * _W4)
_YOSHIDA4_D = (_W4, 1.0 - 2.0 * _W4, _W4)
_Y6_W = (-1.17767998417887, 0.235573213359357, 0.78451361047756)


def _yoshida6_coefs():
    w1, w2, w3 = _Y6_W
    w0 = 1.0 - 2.0 * (w1 + w2 + w3)
    d = (w3, w2, w1, w0, w1, w2, w3)
    c = []
    c.append(0.5 * d[0])
    for i in range(1, len(d)):
        c.append(0.5 * (d[i - 1] + d[i]))
    c.append(0.5 * d[-1])
    return (tuple(c), d)


_YOSHIDA6_C, _YOSHIDA6_D = _yoshida6_coefs()


def _make_step(integrator: str, accel_fn: Callable, dt: float):
    def euler_step(state):
        pos, vel, t = state
        a = accel_fn(pos, t)
        vel = vel + dt * a
        pos = pos + dt * vel
        return (pos, vel, t + dt)

    def leapfrog_step(state):
        pos, vel, t = state
        a = accel_fn(pos, t)
        vel_half = vel + 0.5 * dt * a
        pos = pos + dt * vel_half
        t_new = t + dt
        a_new = accel_fn(pos, t_new)
        vel = vel_half + 0.5 * dt * a_new
        return (pos, vel, t_new)

    def make_yoshida(c_seq, d_seq):
        c_arr = jnp.asarray(c_seq)
        d_arr = jnp.asarray(d_seq)
        n_drift = len(c_seq)
        n_kick = len(d_seq)
        assert n_drift == n_kick + 1

        def yoshida_step(state):
            pos, vel, t = state
            for i in range(n_kick):
                pos = pos + c_arr[i] * dt * vel
                t = t + c_arr[i] * dt
                vel = vel + d_arr[i] * dt * accel_fn(pos, t)
            pos = pos + c_arr[-1] * dt * vel
            t = t + c_arr[-1] * dt
            return (pos, vel, t)

        return yoshida_step

    def rk4_step(state):
        pos, vel, t = state
        k1_p, k1_v = (vel, accel_fn(pos, t))
        k2_p = vel + 0.5 * dt * k1_v
        k2_v = accel_fn(pos + 0.5 * dt * k1_p, t + 0.5 * dt)
        k3_p = vel + 0.5 * dt * k2_v
        k3_v = accel_fn(pos + 0.5 * dt * k2_p, t + 0.5 * dt)
        k4_p = vel + dt * k3_v
        k4_v = accel_fn(pos + dt * k3_p, t + dt)
        pos = pos + dt / 6.0 * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)
        vel = vel + dt / 6.0 * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        return (pos, vel, t + dt)

    if integrator == "euler":
        step_fn = euler_step
    elif integrator == "leapfrog":
        step_fn = leapfrog_step
    elif integrator == "yoshida4":
        step_fn = make_yoshida(_YOSHIDA4_C, _YOSHIDA4_D)
    elif integrator == "yoshida6":
        step_fn = make_yoshida(_YOSHIDA6_C, _YOSHIDA6_D)
    elif integrator == "rk4":
        step_fn = rk4_step
    elif integrator == "dopri5":
        return None
    else:
        raise ValueError(f"Unknown integrator: {integrator!r}")
    return jax.jit(step_fn)


class NBodySampler:
    SUPPORTED_INTEGRATORS = (
        "euler",
        "leapfrog",
        "yoshida4",
        "yoshida6",
        "rk4",
        "dopri5",
    )

    def __init__(
        self,
        masses,
        initial_positions,
        initial_velocities,
        force_law: Callable,
        potential_law: Optional[Callable] = None,
        charges=None,
        source_charges=None,
        force_charges=None,
        integrator: str = "leapfrog",
        dt: float = 0.01,
        softening: float = 0.0,
        spatial_dimensions: Optional[int] = None,
        external_acceleration=None,
        force_modulation: Optional[Callable] = None,
        force_position_modifier: Optional[Callable] = None,
        force_anisotropy=None,
        initial_time: float = 0.0,
    ):
        if integrator not in self.SUPPORTED_INTEGRATORS:
            raise ValueError(
                f"Integrator {integrator!r} not in {self.SUPPORTED_INTEGRATORS}"
            )
        self.integrator = integrator
        self.dt = float(dt)
        self.softening = float(softening)
        self.force_law = force_law
        self.potential_law = potential_law
        self.masses = jnp.asarray(masses, dtype=jnp.float64)
        if source_charges is not None or force_charges is not None:
            if source_charges is None or force_charges is None:
                raise ValueError(
                    "Pass both `source_charges` and `force_charges`, or neither."
                )
            if charges is not None:
                raise ValueError(
                    "Cannot pass `charges` together with `source_charges` / `force_charges`; pick one calling convention."
                )
            self.source_charges = jnp.asarray(source_charges, dtype=jnp.float64)
            self.force_charges = jnp.asarray(force_charges, dtype=jnp.float64)
        elif charges is not None:
            arr = jnp.asarray(charges, dtype=jnp.float64)
            self.source_charges = arr
            self.force_charges = arr
        else:
            self.source_charges = self.masses
            self.force_charges = self.masses
        if (
            self.source_charges.shape != self.masses.shape
            or self.force_charges.shape != self.masses.shape
        ):
            raise ValueError(
                "source_charges, force_charges, and masses must all share the same shape"
            )
        self.charges = self.source_charges
        self.positions = jnp.asarray(initial_positions, dtype=jnp.float64)
        self.velocities = jnp.asarray(initial_velocities, dtype=jnp.float64)
        if self.positions.shape != self.velocities.shape:
            raise ValueError("positions and velocities must have the same shape")
        if self.positions.shape[0] != self.masses.shape[0]:
            raise ValueError("masses, positions, velocities must agree on N")
        self.n_particles = self.positions.shape[0]
        self.spatial_dimensions = (
            spatial_dimensions
            if spatial_dimensions is not None
            else self.positions.shape[1]
        )
        self.time = float(initial_time)
        self._accel_fn = jax.jit(make_acceleration_fn(force_law, softening))
        if potential_law is not None:
            self._potential_fn = jax.jit(make_potential_fn(potential_law, softening))
        else:
            self._potential_fn = None
        self._external_accel_fn: Optional[Callable] = None
        self.external_acceleration = None
        if external_acceleration is not None:
            if callable(external_acceleration):
                self.external_acceleration = external_acceleration
                self._external_accel_fn = _wrap_for_time(
                    external_acceleration, role="external_acceleration"
                )
            else:
                ext = jnp.asarray(external_acceleration, dtype=jnp.float64)
                if ext.shape != (self.spatial_dimensions,):
                    raise ValueError(
                        f"external_acceleration must have shape ({self.spatial_dimensions},), got {ext.shape}"
                    )
                self.external_acceleration = ext
                self._external_accel_fn = lambda pos, _t, _ext=ext: jnp.broadcast_to(
                    _ext[None, :], pos.shape
                )
        self.force_modulation = force_modulation
        self._force_modulation_fn = force_modulation
        self.force_position_modifier = force_position_modifier
        self._force_position_modifier_fn = force_position_modifier
        if force_anisotropy is not None:
            self.force_anisotropy = jnp.asarray(force_anisotropy, dtype=jnp.float64)
            if self.force_anisotropy.shape != (self.spatial_dimensions,):
                raise ValueError(
                    f"force_anisotropy must have shape ({self.spatial_dimensions},), got {self.force_anisotropy.shape}"
                )
        else:
            self.force_anisotropy = None
        src_fixed = self.source_charges
        frc_fixed = self.force_charges
        masses_fixed = self.masses
        ext_fn = self._external_accel_fn
        mod_fn = self._force_modulation_fn
        pos_mod_fn = self._force_position_modifier_fn
        aniso = self.force_anisotropy

        def _accel_pos_t(pos, t):
            a = self._accel_fn(pos, src_fixed, frc_fixed, masses_fixed)
            if aniso is not None:
                a = a * aniso[None, :]
            if mod_fn is not None:
                a = mod_fn(t) * a
            if pos_mod_fn is not None:
                a = pos_mod_fn(pos, t)[:, None] * a
            if ext_fn is not None:
                a = a + ext_fn(pos, t)
            return a

        accel_pos_only = jax.jit(_accel_pos_t)
        if integrator != "dopri5":
            self._step_fn = _make_step(integrator, accel_pos_only, self.dt)
        else:
            self._step_fn = None

    def kinetic_energy(self, velocities=None):
        v = self.velocities if velocities is None else velocities
        return 0.5 * jnp.sum(self.masses * jnp.sum(v * v, axis=-1))

    def potential_energy(self, positions=None):
        if self._potential_fn is None:
            raise ValueError("potential_law was not provided")
        p = self.positions if positions is None else positions
        return self._potential_fn(p, self.source_charges, self.masses)

    def total_energy(self, positions=None, velocities=None):
        return self.kinetic_energy(velocities) + self.potential_energy(positions)

    def linear_momentum(self, velocities=None):
        v = self.velocities if velocities is None else velocities
        return jnp.sum(self.masses[:, None] * v, axis=0)

    def angular_momentum(self, positions=None, velocities=None):
        p = self.positions if positions is None else positions
        v = self.velocities if velocities is None else velocities
        if p.shape[1] == 2:
            return jnp.sum(self.masses * (p[:, 0] * v[:, 1] - p[:, 1] * v[:, 0]))
        elif p.shape[1] == 3:
            r_cross_v = jnp.cross(p, v)
            return jnp.sum(self.masses[:, None] * r_cross_v, axis=0)
        else:
            raise ValueError("angular momentum only defined for 2D or 3D")

    def center_of_mass(self, positions=None):
        p = self.positions if positions is None else positions
        return jnp.sum(self.masses[:, None] * p, axis=0) / jnp.sum(self.masses)

    def step(self):
        if self.integrator == "dopri5":
            raise RuntimeError(
                "Single-step interface not available for adaptive 'dopri5'; use ``run`` instead."
            )
        t_in = jnp.asarray(self.time, dtype=jnp.float64)
        new_pos, new_vel, new_t = self._step_fn((self.positions, self.velocities, t_in))
        self.positions = new_pos
        self.velocities = new_vel
        self.time = float(new_t)

    def run(self, n_steps: int, record_every: int = 1, t_eval=None):
        if self.integrator == "dopri5":
            return self._run_dopri5(n_steps, t_eval=t_eval)
        step_fn = self._step_fn
        rec_n = int(record_every)

        def chunk(state, _):
            state = jax.lax.fori_loop(0, rec_n, lambda _i, s: step_fn(s), state)
            return (state, state)

        n_records = n_steps // record_every
        t0 = jnp.asarray(self.time, dtype=jnp.float64)
        state0 = (self.positions, self.velocities, t0)
        final_state, recorded = jax.lax.scan(chunk, state0, jnp.arange(n_records))
        positions = jnp.concatenate([self.positions[None], recorded[0]], axis=0)
        velocities = jnp.concatenate([self.velocities[None], recorded[1]], axis=0)
        times = jnp.concatenate([t0[None], recorded[2]], axis=0)
        self.positions = final_state[0]
        self.velocities = final_state[1]
        self.time = float(final_state[2])
        return {"times": times, "positions": positions, "velocities": velocities}

    def _run_dopri5(
        self, n_steps: int, t_eval=None, rtol: float = 1e-10, atol: float = 1e-12
    ):
        if t_eval is None:
            t_eval = self.time + jnp.arange(n_steps + 1) * self.dt
        else:
            t_eval = jnp.asarray(t_eval, dtype=jnp.float64)
        n = self.n_particles
        d = self.spatial_dimensions
        accel = self._accel_fn
        src = self.source_charges
        frc = self.force_charges
        masses = self.masses
        ext_fn = self._external_accel_fn
        mod_fn = self._force_modulation_fn
        pos_mod_fn = self._force_position_modifier_fn
        aniso = self.force_anisotropy

        @jax.jit
        def rhs(y, t):
            pos = y[: n * d].reshape((n, d))
            vel = y[n * d :].reshape((n, d))
            a = accel(pos, src, frc, masses)
            if aniso is not None:
                a = a * aniso[None, :]
            if mod_fn is not None:
                a = mod_fn(t) * a
            if pos_mod_fn is not None:
                a = pos_mod_fn(pos, t)[:, None] * a
            if ext_fn is not None:
                a = a + ext_fn(pos, t)
            return jnp.concatenate([vel.reshape(-1), a.reshape(-1)])

        y0 = jnp.concatenate([self.positions.reshape(-1), self.velocities.reshape(-1)])
        ys = odeint(rhs, y0, t_eval, rtol=rtol, atol=atol, mxstep=50000)
        positions = ys[:, : n * d].reshape((-1, n, d))
        velocities = ys[:, n * d :].reshape((-1, n, d))
        self.positions = positions[-1]
        self.velocities = velocities[-1]
        self.time = float(t_eval[-1])
        return {"times": t_eval, "positions": positions, "velocities": velocities}
