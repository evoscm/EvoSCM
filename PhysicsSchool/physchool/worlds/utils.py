import jax.numpy as jnp
import numpy as np
import jax
import jax.lax as lax

jax.config.update("jax_enable_x64", True)


def cic_paint_2d(mesh, positions, weight, periodic=True):
    positions = positions.reshape([-1, 2])
    positions = jnp.expand_dims(positions, 1)
    floor = jnp.floor(positions)
    connection = jnp.array([[0, 0], [1.0, 0], [0.0, 1], [1.0, 1]])
    neighboor_coords = floor + connection
    kernel = 1.0 - jnp.abs(positions - neighboor_coords)
    kernel = kernel[..., 0] * kernel[..., 1]
    if weight is not None:
        kernel = kernel * weight.reshape(*positions.shape[:-1])
    neighboor_coords = neighboor_coords.reshape([-1, 4, 2]).astype("int32")
    if periodic:
        neighboor_coords = jnp.mod(neighboor_coords, jnp.array(mesh.shape))
    else:
        mesh_shape = jnp.array(mesh.shape)
        valid = (neighboor_coords >= 0) & (neighboor_coords < mesh_shape)
        valid = jnp.all(valid, axis=-1)
        kernel = kernel.reshape([-1, 4]) * valid
        neighboor_coords = jnp.clip(neighboor_coords, 0, mesh_shape - 1)
    dnums = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0, 1),
        scatter_dims_to_operand_dims=(0, 1),
    )
    mesh = lax.scatter_add(mesh, neighboor_coords, kernel.reshape([-1, 4]), dnums)
    return mesh


def cic_paint_3d(mesh, positions, weight, periodic=True):
    positions = positions.reshape([-1, 3])
    positions = jnp.expand_dims(positions, 1)
    floor = jnp.floor(positions)
    connection = jnp.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=jnp.float32,
    )
    neighboor_coords = floor + connection
    kernel = 1.0 - jnp.abs(positions - neighboor_coords)
    kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]
    if weight is not None:
        kernel = kernel * weight.reshape([-1, 1])
    neighboor_coords = neighboor_coords.reshape([-1, 8, 3]).astype("int32")
    if periodic:
        neighboor_coords = jnp.mod(neighboor_coords, jnp.array(mesh.shape))
    else:
        valid = jnp.all(
            (neighboor_coords >= 0) & (neighboor_coords < jnp.array(mesh.shape)),
            axis=-1,
        )
        kernel = kernel * valid
        neighboor_coords = jnp.clip(neighboor_coords, 0, jnp.array(mesh.shape) - 1)
    dnums = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0, 1, 2),
        scatter_dims_to_operand_dims=(0, 1, 2),
    )
    mesh = lax.scatter_add(mesh, neighboor_coords, kernel.reshape([-1, 8]), dnums)
    return mesh


def cic_paint(positions, values, grid_shape, box_size, periodic=True):
    n_dims = positions.shape[1]
    if jnp.isscalar(box_size):
        box_size = jnp.array([box_size] * n_dims)
    else:
        box_size = jnp.array(box_size)
    grid_dims = jnp.array(grid_shape[:n_dims])
    dx = box_size / grid_dims
    grid_coords = positions / dx
    mesh = jnp.zeros(grid_shape)
    if n_dims == 2:
        mesh = cic_paint_2d(mesh, grid_coords, values, periodic=periodic)
    elif n_dims == 3:
        mesh = cic_paint_3d(mesh, grid_coords, values, periodic=periodic)
    else:
        raise ValueError(f"Only 2D and 3D painting supported, got {n_dims}D")
    return mesh / jnp.prod(box_size / grid_dims)


def cic_read(field, positions, grid_shape, box_size, periodic=True):
    spatial_dimensions = positions.shape[1]
    grid_size = jnp.array(grid_shape)
    dx = box_size / grid_shape[0]
    grid_coords = positions / dx
    grid_coords = jnp.expand_dims(grid_coords, 1)
    if spatial_dimensions == 2:
        corners = jnp.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=jnp.float32)
    else:
        corners = jnp.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=jnp.float32,
        )
    floor = jnp.floor(grid_coords)
    corner_positions = floor + corners
    kernel = 1.0 - jnp.abs(grid_coords - corner_positions)
    kernel = jnp.prod(kernel, axis=2)
    indices = corner_positions.reshape(-1, spatial_dimensions).astype(jnp.int32)
    if periodic:
        indices = indices % grid_size
    else:
        indices = jnp.clip(indices, 0, grid_size - 1)
    if spatial_dimensions == 2:
        field_values = field[indices[:, 0], indices[:, 1]]
    else:
        field_values = field[indices[:, 0], indices[:, 1], indices[:, 2]]
    field_values = field_values.reshape(len(positions), -1)
    values = jnp.sum(field_values * kernel, axis=1)
    return values
