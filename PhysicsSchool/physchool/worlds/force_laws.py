from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
import scipy.special


def gravity_force(r_mag, q_i, q_j, m_i, m_j, G: float = 1.0):
    return G * q_i * q_j / r_mag**2


def gravity_potential(r_mag, q_i, q_j, m_i, m_j, G: float = 1.0):
    return -G * q_i * q_j / r_mag


def screened_gravity_force(r_mag, q_i, q_j, m_i, m_j, G: float = 1.0, lam: float = 5.0):
    return G * q_i * q_j / r_mag**2 * jnp.exp(-r_mag / lam)


def coulomb_force(r_mag, q_i, q_j, m_i, m_j, k: float = 1.0):
    return -k * q_i * q_j / r_mag**2


def coulomb_potential(r_mag, q_i, q_j, m_i, m_j, k: float = 1.0):
    return k * q_i * q_j / r_mag


_INV_TWO_PI = 1.0 / (2.0 * jnp.pi)


def poisson_2d_force(r_mag, q_i, q_j, m_i, m_j, G: float = 1.0):
    return G * q_i * q_j * _INV_TWO_PI / r_mag


def poisson_2d_potential(r_mag, q_i, q_j, m_i, m_j, G: float = 1.0):
    return G * q_i * q_j * _INV_TWO_PI * jnp.log(r_mag)


def _k0_jax(x):
    shape_dtype = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return jax.pure_callback(
        lambda xx: np.asarray(scipy.special.k0(xx), dtype=xx.dtype), shape_dtype, x
    )


def _k1_jax(x):
    shape_dtype = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return jax.pure_callback(
        lambda xx: np.asarray(scipy.special.k1(xx), dtype=xx.dtype), shape_dtype, x
    )


def yukawa_2d_force(r_mag, q_i, q_j, m_i, m_j, G: float = 1.0, lam: float = 2.0):
    return G * q_i * q_j * _INV_TWO_PI * _k1_jax(r_mag / lam) / lam


def yukawa_2d_potential(r_mag, q_i, q_j, m_i, m_j, G: float = 1.0, lam: float = 2.0):
    return -G * q_i * q_j * _INV_TWO_PI * _k0_jax(r_mag / lam)


def _riesz_2d_prefactor(alpha: float) -> float:
    return float(
        scipy.special.gamma(1.0 - alpha)
        / (2.0 ** (2.0 * alpha) * np.pi * scipy.special.gamma(alpha))
    )


def riesz_2d_force(r_mag, q_i, q_j, m_i, m_j, G: float = 1.0, alpha: float = 0.5):
    c = _riesz_2d_prefactor(alpha)
    return G * c * (2.0 - 2.0 * alpha) * q_i * q_j / r_mag ** (3.0 - 2.0 * alpha)


def riesz_2d_potential(r_mag, q_i, q_j, m_i, m_j, G: float = 1.0, alpha: float = 0.5):
    c = _riesz_2d_prefactor(alpha)
    return -G * c * q_i * q_j / r_mag ** (2.0 - 2.0 * alpha)


def extra_dimensions_2d_force(
    r_mag,
    q_i,
    q_j,
    m_i,
    m_j,
    G: float = 1.0,
    R_compact: float = 0.5,
    n_images: int = 20,
):
    L = 2.0 * jnp.pi * R_compact
    n_arr = jnp.arange(-n_images, n_images + 1)
    y_n = n_arr * L
    r_expanded = r_mag[..., None]
    denom = (r_expanded * r_expanded + y_n * y_n) ** 1.5
    F_geom = jnp.sum(r_expanded / denom, axis=-1)
    return G * L * q_i * q_j * F_geom / (4.0 * jnp.pi)


def extra_dimensions_2d_potential(
    r_mag,
    q_i,
    q_j,
    m_i,
    m_j,
    G: float = 1.0,
    R_compact: float = 0.5,
    n_images: int = 20,
):
    L = 2.0 * jnp.pi * R_compact
    n_arr = jnp.arange(-n_images, n_images + 1)
    y_n = n_arr * L
    r_expanded = r_mag[..., None]
    V_geom = -jnp.sum(1.0 / jnp.sqrt(r_expanded * r_expanded + y_n * y_n), axis=-1)
    return G * L * q_i * q_j * V_geom / (4.0 * jnp.pi)
