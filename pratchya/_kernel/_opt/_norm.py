import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.typing import ArrayLike
from ..._utils import dequantize_cast
from .._cast import block_quantize

def rmsnorm_quantize_kernel(x_ref, x_scale, gamma_ref, out_fp8_ref, scale_ref, *, block_size, eps=1e-5):
    x_row = x_ref[...]
    x_scale = x_scale[...]

    x_block_size = x_row.shape[-1] // x_scale.shape[-1]

    x_fp32 = x_row.astype(jnp.float32).reshape(-1, x_block_size) * x_scale.reshape(-1, 1)
    x_fp32 = x_fp32.reshape(1, -1)
    
    rms = jnp.sqrt(jnp.mean(jnp.square(x_fp32), axis=-1, keepdims=True) + eps)
    
    gamma = gamma_ref[...]
    x_norm = (x_fp32 / rms).astype(jnp.bfloat16) * gamma.astype(jnp.bfloat16)
    
    x_blocked = x_norm.reshape(-1, block_size)
    abs_max = jnp.max(jnp.abs(x_blocked), axis=-1, keepdims=True)
    
    scale = jnp.maximum(abs_max, 1e-4) / 448.0
    
    x_fp8 = (x_blocked / scale).astype(jnp.float8_e4m3fn)
    
    out_fp8_ref[...] = x_fp8.reshape(1, -1)
    scale_ref[...] = scale.reshape(1, -1).astype(jnp.bfloat16)

def rmsnorm_quantize(x: ArrayLike, x_scale: ArrayLike, gamma: ArrayLike, eps: float=1e-5):
    
    orig_shape = x.shape
    C = orig_shape[-1]
    block_size = C // x_scale.shape[-1]
    
    assert C % block_size == 0, \
        "last dimension can't be divided by block_size"
    
    x_flat = x.reshape(-1, C)
    N = x_flat.shape[0]
    x_scale = x_scale.reshape(N, -1)
    
    grid = (N,)
    
    in_spec = pl.BlockSpec(
        memory_space=pl.MemorySpace.ANY,
        index_map=lambda i: (i, 0),
        block_shape=(1, C)
    )

    x_scale_spec = pl.BlockSpec(
        memory_space=pl.MemorySpace.ANY,
        index_map=lambda i: (i, 0),
        block_shape=(1, 1)
    )
    
    gamma_spec = pl.BlockSpec(
        memory_space=pl.MemorySpace.ANY,
        index_map=lambda i: (0,),
        block_shape=(C,)
    )
    
    out_spec = pl.BlockSpec(
        memory_space=pl.MemorySpace.ANY,
        index_map=lambda i: (i, 0),
        block_shape=(1, C)
    )
    
    scale_spec = pl.BlockSpec(
        memory_space=pl.MemorySpace.ANY,
        index_map=lambda i: (i, 0),
        block_shape=(1, C // block_size)
    )
    
    out_fp8_flat, scales_flat = pl.pallas_call(
        lambda x_r, x_s, g_r, o_r, s_r: rmsnorm_quantize_kernel(x_r, x_s, g_r, o_r, s_r, block_size=block_size, eps=eps),
        out_shape=(
            jax.ShapeDtypeStruct((N, C), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((N, C // block_size), jnp.bfloat16)
        ),
        grid=grid,
        in_specs=[in_spec, x_scale_spec, gamma_spec],
        out_specs=[out_spec, scale_spec],
        interpret=True
    )(x_flat, x_scale, gamma)
    
    out_fp8 = out_fp8_flat.reshape(*orig_shape)
    scales = scales_flat.reshape(*orig_shape[:-1], C // block_size)
    
    return out_fp8, scales

# WHY ...
@jax.custom_gradient
def rmsnorm(x: ArrayLike, x_sc: ArrayLike, gamma: ArrayLike, eps: float=1e-5):

    y = rmsnorm_quantize(x, x_sc, gamma, eps)

    def bwd_fn(g, g_sc):

        def rmsnorm_math(x: ArrayLike, x_sc: ArrayLike, gamma: ArrayLike):
            block_size = x.shape[-1] // x_sc.shape[-1]

            x = dequantize_cast(x, x_sc, dtype=gamma.dtype)
            inv_rms = jax.lax.rsqrt(jnp.var(x, axis=-1, dtype=gamma.dtype, keepdims=True) + eps)
            x = x * inv_rms * gamma

            x, x_sc = block_quantize(x, block_size)

            return x, x_sc

        y, pullback = jax.vjp(rmsnorm_math, x, x_sc, gamma)
        dx, dx_sc, d_gamma = pullback(g, g_sc)

        return dx, dx_sc, d_gamma, None

    return y, bwd_fn 


# WHY DID I DO THIS...
@jax.custom_gradient
def rmsnorm(x: ArrayLike, x_sc: ArrayLike, gamma: ArrayLike, eps: float=1e-5):

    def rmsnorm_math(x: ArrayLike, x_sc: ArrayLike, gamma: ArrayLike, eps: float):
        block_size = x.shape[-1] // x_sc.shape[-1]

        x = dequantize_cast(x, x_sc, dtype=gamma.dtype)
        inv_rms = jax.lax.rsqrt(jnp.var(x, axis=-1, dtype=gamma.dtype, keepdims=True) + eps)
        x = x * inv_rms * gamma

        x, x_sc = block_quantize(x, block_size)

        return x, x_sc


    y, pullback = jax.vjp(rmsnorm_math, x, x_sc, gamma)

    def bwd_fn(g, g_sc):
        dx, dx_sc, d_gamma = pullback(g, g_sc)

        return dx, dx_sc, d_gamma, None

    return y, bwd_fn 