
from jax.typing import ArrayLike, DTypeLike
import jax.numpy as jnp, jax
import optax

FP8MIN = jnp.finfo(jnp.float8_e4m3fn).smallest_subnormal

def block_quantize(x: ArrayLike, block_size: int):
    orig_shape = x.shape
    x = x.reshape(-1, block_size)

    # abs_max = jnp.max(jnp.abs(x), axis=-1)
    
    # x_sc = jnp.maximum(abs_max, FP8MIN.astype(abs_max.dtype)) / 448
    # x_sc = x_sc.reshape(-1, 1)
    # x = jnp.clip(x / x_sc, -448, 448).astype(jnp.float8_e4m3fn)

    
    N = x.shape[0]
    x_sc = jnp.ones((N, 1))

    x = x.reshape(orig_shape)
    x_sc = x_sc.reshape(*orig_shape[:-1], orig_shape[-1] // block_size).astype(jnp.bfloat16)
    
    return x, x_sc

def dequantize_cast(x: ArrayLike, x_sc: ArrayLike, *, dtype: DTypeLike):
    orig_shape = x.shape
    block_size = orig_shape[-1] // x_sc.shape[-1]

    x = x.reshape(-1, block_size) 
    x_sc = x_sc.reshape(-1, 1) 

    x = x.astype(jnp.bfloat16) * x_sc
    return x.astype(dtype).reshape(*orig_shape)


def compute_loss(logits: ArrayLike, labels: ArrayLike):
    one_hot_labels = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
    smoothed_labels = optax.smooth_labels(one_hot_labels, alpha=0.1)
    
    loss = optax.softmax_cross_entropy(logits=logits, labels=smoothed_labels)

    return jnp.average(loss, axis=[0, 1])
