
from typing import NamedTuple, Any
from jax.typing import ArrayLike
from jax.tree_util import register_pytree_node_class, register_dataclass
from dataclasses import dataclass
import functools
import jax, jax.numpy as jnp
import optax
from .._qualia._qarr import QArrayImpl

class MiulionState(NamedTuple):
    count: ArrayLike
    momentum: QArrayImpl

@dataclass
class MiulionHyperParams:
    lion_lr: float = 3e-4
    muon_lr: float = 0.02
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    total_steps: int = 100000
    beta1: float = 0.9
    beta2: float = 0.99
    blksize: int = 128
    ns_steps: int = 5


@functools.partial(register_dataclass(meta_fields=['muon_schedule', 'lion_schedule']))
@dataclass
class MiulionScheduler:
    muon_schedule: Any
    lion_schedule: Any
    weight_decay: float

    def get_hyperparams(self, step: jax.Array):
        current_lion_lr = self.lion_schedule(step)
        current_muon_lr = self.muon_schedule(step)
        
        return current_lion_lr, current_muon_lr, self.weight_decay
    

def newton_schulz(G, steps=5):
    m, n = G.shape
    transpose = m > n
    if transpose:
        G = G.T

    X = G / (jnp.linalg.norm(G, ord='fro') + 1e-7)

    def loop_fn(i, X):
        A = X @ X.T
        B = A @ X
        return 1.5 * X - 0.5 * B
    
    X = jax.lax.fori_loop(0, steps, loop_fn, X)
    
    if transpose:
        X = X.T

    return X

def miulion_optimizer(hyperparams: MiulionHyperParams, scheduler: MiulionScheduler):
    beta1 = hyperparams.beta1
    beta2 = hyperparams.beta2
    blksize = hyperparams.blksize
    ns_steps = hyperparams.ns_steps

    def init_fn(params):
        
        def make_leaf(params):
            shape = params.shape
            m = jnp.zeros_like(shape)
            if shape[-2] < blksize:
                return QArrayImpl(m, (1, blksize))
            
            return QArrayImpl(m, (blksize, blksize))
        
        momentum = jax.tree_util.tree_map(make_leaf, params)

        return MiulionState(
            count=jnp.zeros([], jnp.int32),
            momentum=momentum,
        )
    

    def update_fn(updates, state: MiulionState, params=None):

        if params is None:
            raise ValueError("Miulion Optimizer requires 'params' to perform Weight Decay and precision mapping.")
        
        step = state.count
        lion_lr, muon_lr, weight_decay = scheduler.get_hyperparams(step)

        def should_use_muon(path, param):
            if param.ndim < 2:
                return False
            
            path_str = "".join(str(p) for p in path).lower()
            
            forbidden_keywords = ['embed_tokens', 'lm_head']
            
            if any(keyword in path_str for keyword in forbidden_keywords):
                return False
                
            return True

        def update_leaf(path, g: ArrayLike, m: ArrayLike, param):
            orig_shape = g.shape
            orig_size = g.size
            flat_g = g.ravel()

            blksize = m.shape[-1] // m_sc.shape[-1]

            pad_size = m.size - orig_size
            padded_flat_g = jnp.pad(flat_g, (0, pad_size)) if pad_size > 0 else flat_g
            g_blocked = padded_flat_g.reshape(m.shape)

            m = dequantize_cast(m, m_sc, dtype=g.dtype)

            c_blocked = beta1 * m + (1.0 - beta1) * g_blocked
            new_m_blocked = beta2 * m + (1.0 - beta2) * g_blocked

            new_m, new_m_sc = block_quantize(new_m_blocked, blksize)
            c = c_blocked.ravel()[:orig_size].reshape(orig_shape)

            if should_use_muon(path, g):
                u = newton_schulz(c, steps=ns_steps)
                scale_factor = jnp.sqrt(jnp.maximum(orig_shape[0], orig_shape[1]))
                update = -(muon_lr * (u * scale_factor) + muon_lr * weight_decay * param)
            else:
                u = jnp.sign(c)
                update = -(lion_lr * u + lion_lr * weight_decay * param)
                
            return update, new_m
        
        results = jax.tree_util.tree_map_with_path(
            update_leaf, updates, state.momentum, params
        )

        new_updates = jax.tree_util.tree_map(lambda x: x[0], results, is_leaf=lambda x: isinstance(x, tuple))
        new_m = jax.tree_util.tree_map(lambda x: x[1], results, is_leaf=lambda x: isinstance(x, tuple))
        new_m_sc = jax.tree_util.tree_map(lambda x: x[2], results, is_leaf=lambda x: isinstance(x, tuple))

        new_state = MiulionState(
            count=state.count + 1,
            momentum=new_m,
            momentum_sc=new_m_sc
        )

        return new_updates, new_state
    
    return optax.GradientTransformation(init_fn, update_fn)
