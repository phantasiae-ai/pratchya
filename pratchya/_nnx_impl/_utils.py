
from jax.typing import ArrayLike, DTypeLike
import jax.numpy as jnp, jax
import optax
from .._qualia import QArrayImpl

def compute_loss(logits: QArrayImpl, labels: ArrayLike):
    one_hot_labels = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
    smoothed_labels = optax.smooth_labels(one_hot_labels, alpha=0.1)
    
    loss = optax.softmax_cross_entropy(logits=logits.astype(jnp.float32), labels=smoothed_labels)

    return jnp.average(loss, axis=[0, 1])
