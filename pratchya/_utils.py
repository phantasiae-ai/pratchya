
from jax.typing import ArrayLike, DTypeLike
import jax.numpy as jnp

def dequantize_cast(x: ArrayLike, x_s: ArrayLike, *, dtype: DTypeLike):
    orig_shape = x.shape
    block_size = orig_shape[-1] // x_s.shape[-1]

    x = x.reshape(-1, block_size) 
    x_s = x_s.reshape(-1, 1) 

    x = x.astype(jnp.bfloat16) * x_s
    return x.astype(dtype).reshape(*orig_shape)


# def linear_block_quantize(x: ArrayLike, x_s: ArrayLike, params: ArrayLike):
