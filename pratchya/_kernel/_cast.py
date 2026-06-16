import jax.experimental.pallas as pl
import jax.numpy as jnp
from jax.typing import ArrayLike, DTypeLike
import jax
import os


# Set to False when running on TPU for Pallas kernel performance.
# Set to True for local debugging (uses pure JAX fallback to avoid
# a bug in JAX 0.10.1's pallas_call interpret mode).
cpu = os.environ.get('CPU', '1')
INTERPRET = True if cpu == '1' else False


def block_quantize_kernel(x_ref, out_ref, scale_ref):
    x_block = x_ref[...]
    abs_max = jnp.max(jnp.abs(x_block))

    fp8_min = jnp.finfo(jnp.float8_e4m3fn).smallest_subnormal.astype(abs_max.dtype)
    scale = jnp.maximum(abs_max, fp8_min) / 448
    
    x_fp8 = jnp.clip(x_block / scale, -448, 448).astype(jnp.float8_e4m3fn)

    out_ref[...] = x_fp8
    scale_ref[...] = scale.reshape(1, 1).astype(jnp.bfloat16)


def block_quantize(x: ArrayLike, block_size = 128):

    orig_shape = x.shape
    C = orig_shape[-1]

    assert C % block_size == 0, \
        "last dimension of x can't be divided by block_size"

    x = x.reshape(-1, C)
    N = x.shape[0]
    n_blocks = C // block_size

    if INTERPRET:
        # Pure JAX fallback — avoids a bug in JAX 0.10.1's
        # pallas_call(interpret=True) where intermediate grid cells
        # fail to write outputs, producing NaN.
        x_blocks = x.reshape(N, n_blocks, block_size)

        abs_max = jnp.max(jnp.abs(x_blocks), axis=-1, keepdims=True)
        fp8_min = jnp.finfo(jnp.float8_e4m3fn).smallest_subnormal.astype(abs_max.dtype)
        scale = jnp.maximum(abs_max, fp8_min) / 448

        out_fp8 = jnp.clip(x_blocks / scale, -448, 448).astype(jnp.float8_e4m3fn)

        out_fp8 = out_fp8.reshape(*orig_shape)
        scales = scale.squeeze(-1).astype(jnp.bfloat16).reshape(*orig_shape[:-1], n_blocks)
    else:
        grid = (N, n_blocks)

        in_spec = pl.BlockSpec(
            memory_space=pl.MemorySpace.ANY,
            index_map=lambda i, j: (i, j * block_size),
            block_shape=(1, block_size)
        )

        out_spec = pl.BlockSpec(
            memory_space=pl.MemorySpace.ANY,
            index_map=lambda i, j: (i, j * block_size),
            block_shape=(1, block_size)
        )

        scale_spec = pl.BlockSpec(
            memory_space=pl.MemorySpace.ANY,
            index_map=lambda i, j: (i, j),
            block_shape=(1, 1)
        )

        out_fp8, scales = pl.pallas_call(
            block_quantize_kernel,
            out_shape=(
                jax.ShapeDtypeStruct((N, C), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((N, n_blocks), jnp.bfloat16)
            ),
            grid=grid,
            in_specs=[in_spec],
            out_specs=[out_spec, scale_spec],
        )(x)

        out_fp8 = out_fp8.reshape(*orig_shape)
        scales = scales.reshape(*orig_shape[:-1], n_blocks)

    return out_fp8, scales