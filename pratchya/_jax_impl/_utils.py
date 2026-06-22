
from jax.typing import ArrayLike
import jax.numpy as jnp, jax
from typing import Union, Dict
from .._qualia._qarr import QArrayImpl

def lora_ffn(x: QArrayImpl, params: Dict):
    x = x @ params['lin1']['w']
    x = x.apply(lambda x: jax.nn.silu(x))
    x = x @ params['lin2']['w']
    return x

def linear(x: QArrayImpl, params: Dict):
    x =  x @ params['w']
    return x

def lerp(a: ArrayLike, b: ArrayLike, w: ArrayLike):
    x = a + (b - a) * w
    return x

def normalized(x: ArrayLike, *, axis: int, eps: float = 1e-12) -> ArrayLike:
    if hasattr(x, 'apply'):
        return x.apply(lambda v: v / jnp.sqrt(jnp.sum(jnp.square(v), axis=axis, keepdims=True) + eps**2))
    x_norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=axis, keepdims=True) + eps**2)
    x = x / x_norm
    return x

def group_norm(x, n_groups, params, eps=1e-5):
    if hasattr(x, 'apply'):
        return x.apply(lambda v: group_norm(v, n_groups, params, eps))
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

def apply_rope(x: ArrayLike, t: Union[int, ArrayLike], inv_freq: ArrayLike):
    t_array = jnp.atleast_1d(t).astype(jnp.float32)
    freqs = jnp.outer(t_array, inv_freq) # Shape: (Batch or 1, head_dim // 2)
    
    emb = jnp.concatenate([freqs, freqs], axis=-1) # Shape: (Batch or 1, head_dim)
    
    emb = emb[:, None, :] 
    cos = jnp.cos(emb)
    sin = jnp.sin(emb)
    
    return (x * cos) + (rotate_half(x) * sin)

def rotate_half(x: ArrayLike):
    d = x.shape[-1]
    x1 = x[..., :d//2]
    x2 = x[..., d//2:]
    if hasattr(x, 'concat'):
        return x.__class__.concat([-x2, x1], axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)