from __future__ import annotations
from typing import Callable
import jax
import jax.numpy as jnp


def solve_kepler(mean_anomaly, eccentricity, max_iter: int = 64):
    M = jnp.asarray(mean_anomaly)
    e = eccentricity
    E = M + e * jnp.sin(M)

    def body(_, E):
        f = E - e * jnp.sin(E) - M
        fp = 1.0 - e * jnp.cos(E)
        return E - f / fp

    return jax.lax.fori_loop(0, max_iter, body, E)


def kepler_two_body_solution(
    times,
    M_central,
    m_orbiter,
    semi_major,
    eccentricity,
    G: float = 1.0,
    t_periapsis: float = 0.0,
):
    a = semi_major
    e = eccentricity
    mu = G * (M_central + m_orbiter)
    n = jnp.sqrt(mu / a**3)
    period = 2 * jnp.pi / n
    M = n * (jnp.asarray(times) - t_periapsis)
    E = solve_kepler(M, e)
    cosE = jnp.cos(E)
    sinE = jnp.sin(E)
    sqrt1me2 = jnp.sqrt(1.0 - e**2)
    x = a * (cosE - e)
    y = a * sqrt1me2 * sinE
    dEdt = n / (1.0 - e * cosE)
    vx = -a * sinE * dEdt
    vy = a * sqrt1me2 * cosE * dEdt
    pos = jnp.stack([x, y], axis=-1)
    vel = jnp.stack([vx, vy], axis=-1)
    return (pos, vel, period)


def circular_orbit_velocity(
    force_law: Callable, M_central, m_orbiter, radius, q_central=None, q_orbiter=None
):
    if q_central is None:
        q_central = M_central
    if q_orbiter is None:
        q_orbiter = m_orbiter
    F = float(
        force_law(
            jnp.asarray(radius),
            jnp.asarray(q_central),
            jnp.asarray(q_orbiter),
            jnp.asarray(M_central),
            jnp.asarray(m_orbiter),
        )
    )
    return float(jnp.sqrt(F * radius / m_orbiter))
