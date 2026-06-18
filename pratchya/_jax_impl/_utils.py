
from jax.typing import ArrayLike
import jax.numpy as jnp

def lora_ffn(x: ArrayLike, params):
    x =  x @ params['lin1']['w']
    x = x @ params['lin2']['w']
    return x

def linear(x: ArrayLike, params):
    x =  x @ params['w']
    if 'b' in params:
        x = x + params['b']

    return x

def lerp(a: ArrayLike, b: ArrayLike, w: ArrayLike):
    x = a + (b - a) * w
    return x

def normalized(x: ArrayLike, *, axis: int, eps: float = 1e-12) -> ArrayLike:
    x_norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=axis, keepdims=True) + eps**2)
    x = x / x_norm
    return x

def group_norm(x, n_groups, params, eps=1e-5):
    shape = x.shape
    N = shape[0]
    C = shape[-1]
    G = n_groups
    
    spatial_shape = shape[1:-1]
    x_reshaped = x.reshape(N, *spatial_shape, G, C // G)
    
    ndim = x_reshaped.ndim
    reduction_axes = tuple(range(1, ndim - 2)) + (ndim - 1,)
    
    mean = jnp.mean(x_reshaped, axis=reduction_axes, keepdims=True)
    var = jnp.var(x_reshaped, axis=reduction_axes, keepdims=True)
    
    x_norm = (x_reshaped - mean) / jnp.sqrt(var + eps)
    x_out = x_norm.reshape(*shape)
    
    return x_out * params['w'] + params['b']