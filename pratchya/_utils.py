
from jax.typing import ArrayLike, DTypeLike
import jax.numpy as jnp, jax
import optax

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
