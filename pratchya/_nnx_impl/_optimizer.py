import jax.numpy as jnp
import jax
import optax
from flax import nnx
from flax.struct import dataclass
from typing import NamedTuple, Any
from jax.typing import ArrayLike, DTypeLike

from .._qualia import QArrayImpl


def newton_schulz(G, steps=5):
    m, n = G.shape
    transpose = m > n
    if transpose:
        G = G.T

    X = G / (jnp.linalg.norm(G, ord='fro') + 1e-7)

    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = 1.5 * X - 0.5 * B
    
    if transpose:
        X = X.T

    return X


class MiulionState(NamedTuple):
    count: ArrayLike
    momentum: QArrayImpl | jax.Array

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
    dtype: DTypeLike = jnp.bfloat16

@dataclass
class MiulionScheduler:
    muon_schedule: Any
    lion_schedule: Any
    weight_decay: float

    def get_hyperparams(self, step: jax.Array):
        current_lion_lr = self.lion_schedule(step)
        current_muon_lr = self.muon_schedule(step)
        
        return current_lion_lr, current_muon_lr, self.weight_decay
    

def miulion_optimizer(hyperparams: MiulionHyperParams, scheduler: MiulionScheduler):
    beta1 = hyperparams.beta1
    beta2 = hyperparams.beta2
    ns_steps = hyperparams.ns_steps

    def init_fn(params):
        def make_momentum_leaf(p):
            if getattr(p, 'dtype', None) == jnp.float8_e8m0fnu:
                from pratchya._qualia._qarr import FP8E8M0_MIN
                return jnp.full_like(p, FP8E8M0_MIN, dtype=p.dtype)
            if jnp.issubdtype(p.dtype, jnp.floating) and p.dtype not in (jnp.float8_e4m3fn, jnp.float8_e8m0fnu):
                return jnp.zeros_like(p, dtype=jnp.float32)
            return jnp.zeros_like(p)
        
        momentum = jax.tree_util.tree_map(make_momentum_leaf, params)

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
            if param.ndim != 2:
                return False
            
            path_str = "".join(str(p) for p in path).lower()
            
            forbidden_keywords = ['embed_tokens', 'lm_head']
            
            if any(keyword in path_str for keyword in forbidden_keywords):
                return False
                
            return True

        def update_leaf(path, g, m, param):
            if isinstance(param, QArrayImpl) and param.get_value().ndim == 3:
                tgrid = param.tgrid
                if getattr(m, '__class__', None).__name__ == 'State' and not isinstance(m, QArrayImpl):
                    m = QArrayImpl._tree_unflatten((True, tgrid), (m[0], m[1], m[2]))
                
                def map_fn(carry, args):
                    g_val, g_sc8, g_sc32, m_val, m_sc8, m_sc32, p_val, p_sc8, p_sc32 = args
                    g_layer = QArrayImpl._tree_unflatten((True, tgrid), (g_val, g_sc8, g_sc32)).astype(jnp.bfloat16)
                    m_layer = QArrayImpl._tree_unflatten((True, tgrid), (m_val, m_sc8, m_sc32)).astype(jnp.bfloat16)
                    p_layer = QArrayImpl._tree_unflatten((True, tgrid), (p_val, p_sc8, p_sc32)).astype(jnp.bfloat16)
                    
                    orig_shape = g_layer.shape
                    orig_size = g_layer.size
                    
                    if m_layer.shape == orig_shape:
                        g_blocked = g_layer
                        c_blocked = beta1 * m_layer + (1.0 - beta1) * g_blocked
                        new_m_blocked = beta2 * m_layer + (1.0 - beta2) * g_blocked
                        c = c_blocked
                    else:
                        flat_g = g_layer.ravel()
                        pad_size = m_layer.size - orig_size
                        padded_flat_g = jnp.pad(flat_g, (0, pad_size)) if pad_size > 0 else flat_g
                        g_blocked = padded_flat_g.reshape(m_layer.shape)
                        c_blocked = beta1 * m_layer + (1.0 - beta1) * g_blocked
                        new_m_blocked = beta2 * m_layer + (1.0 - beta2) * g_blocked
                        c = c_blocked.ravel()[:orig_size].reshape(orig_shape)
                        
                    new_m_q = QArrayImpl(new_m_blocked, tgrid)._tree_flatten()[0]
                    
                    if should_use_muon(path, g_layer):
                        u = newton_schulz(c, steps=ns_steps)
                        scale_factor = jnp.sqrt(jnp.maximum(orig_shape[0], orig_shape[1]))
                        update = -(muon_lr * (u * scale_factor) + muon_lr * weight_decay * p_layer)
                    else:
                        u = jnp.sign(c)
                        update = -(lion_lr * u + lion_lr * weight_decay * p_layer)
                        
                    return carry, (update.astype(jnp.bfloat16), new_m_q[0], new_m_q[1], new_m_q[2])

                _, (update, nm_val, nm_sc8, nm_sc32) = jax.lax.scan(
                    map_fn,
                    None,
                    (g.get_value(), g._QArrayImpl__sc_fp8, g._QArrayImpl__sc_fp32,
                     m.get_value(), m._QArrayImpl__sc_fp8, m._QArrayImpl__sc_fp32,
                     param.get_value(), param._QArrayImpl__sc_fp8, param._QArrayImpl__sc_fp32)
                )
                new_m = QArrayImpl._tree_unflatten((True, tgrid), (nm_val, nm_sc8, nm_sc32))
                return update, new_m

            tgrid = None
            if isinstance(param, QArrayImpl):
                tgrid = param.tgrid
                if getattr(m, '__class__', None).__name__ == 'State' and not isinstance(m, QArrayImpl):
                    m = QArrayImpl._tree_unflatten((True, tgrid), (m[0], m[1], m[2]))
                m = m.astype(jnp.float32)
                g = g.astype(jnp.float32)
            elif isinstance(m, QArrayImpl):
                tgrid = m.tgrid
                m = m.astype(jnp.bfloat16)
                g = g.astype(jnp.bfloat16)

            orig_shape = g.shape
            orig_size = g.size

            if m.shape == orig_shape:
                g_blocked = g
                c_blocked = beta1 * m + (1.0 - beta1) * g_blocked
                new_m_blocked = beta2 * m + (1.0 - beta2) * g_blocked
                c = c_blocked
            else:
                flat_g = g.ravel()
                pad_size = m.size - orig_size
                padded_flat_g = jnp.pad(flat_g, (0, pad_size)) if pad_size > 0 else flat_g
                g_blocked = padded_flat_g.reshape(m.shape)

                c_blocked = beta1 * m + (1.0 - beta1) * g_blocked
                new_m_blocked = beta2 * m + (1.0 - beta2) * g_blocked

                c = c_blocked.ravel()[:orig_size].reshape(orig_shape)

            new_m = new_m_blocked
            if isinstance(param, QArrayImpl):
                new_m = QArrayImpl(new_m_blocked, tgrid)

            if should_use_muon(path, g):
                u = newton_schulz(c, steps=ns_steps)
                scale_factor = jnp.sqrt(jnp.maximum(orig_shape[0], orig_shape[1]))
                p_f32 = param.astype(jnp.float32) if isinstance(param, QArrayImpl) else param
                update = -(muon_lr * (u * scale_factor) + muon_lr * weight_decay * p_f32)
            else:
                u = jnp.sign(c)
                p_f32 = param.astype(jnp.float32) if isinstance(param, QArrayImpl) else param
                update = -(lion_lr * u + lion_lr * weight_decay * p_f32)
                
            return update, new_m
        
        results = jax.tree_util.tree_map_with_path(
            update_leaf, updates, state.momentum, params,
            is_leaf=lambda p: isinstance(p, QArrayImpl)
        )

        new_updates = jax.tree_util.tree_map(lambda x: x[0], results, is_leaf=lambda x: isinstance(x, tuple) and len(x) == 2)
        new_m = jax.tree_util.tree_map(lambda x: x[1], results, is_leaf=lambda x: isinstance(x, tuple) and len(x) == 2)
        
        new_state = MiulionState(
            count=state.count + 1,
            momentum=new_m
        )

        return new_updates, new_state
    
    return optax.GradientTransformation(init_fn, update_fn)


def miulion_optimizer_nq(hyperparams: MiulionHyperParams, scheduler: MiulionScheduler):
    beta1 = hyperparams.beta1
    beta2 = hyperparams.beta2
    ns_steps = hyperparams.ns_steps
    dtype = hyperparams.dtype

    def init_fn(params):
        def make_momentum_leaf(p):
            return jnp.zeros_like(p, dtype=dtype)
        
        momentum = jax.tree_util.tree_map(make_momentum_leaf, params)

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
            if param.ndim != 2:
                return False
            
            path_str = "".join(str(p) for p in path).lower()
            
            forbidden_keywords = ['embed_tokens', 'lm_head']
            
            if any(keyword in path_str for keyword in forbidden_keywords):
                return False
                
            return True

        def update_leaf(path, g, m, param):
            if isinstance(param, QArrayImpl) and param.get_value().ndim == 3:
                tgrid = param.tgrid
                if getattr(m, '__class__', None).__name__ == 'State' and not isinstance(m, QArrayImpl):
                    m = QArrayImpl._tree_unflatten((True, tgrid), (m[0], m[1], m[2]))
                
                def map_fn(args):
                    g_val, g_sc8, g_sc32, m_val, m_sc8, m_sc32, p_val, p_sc8, p_sc32 = args
                    g_layer = QArrayImpl._tree_unflatten((True, tgrid), (g_val, g_sc8, g_sc32)).astype(jnp.float32)
                    m_layer = QArrayImpl._tree_unflatten((True, tgrid), (m_val, m_sc8, m_sc32)).astype(jnp.float32)
                    p_layer = QArrayImpl._tree_unflatten((True, tgrid), (p_val, p_sc8, p_sc32)).astype(jnp.float32)
                    
                    orig_shape = g_layer.shape
                    orig_size = g_layer.size
                    
                    if m_layer.shape == orig_shape:
                        g_blocked = g_layer
                        c_blocked = beta1 * m_layer + (1.0 - beta1) * g_blocked
                        new_m_blocked = beta2 * m_layer + (1.0 - beta2) * g_blocked
                        c = c_blocked
                    else:
                        flat_g = g_layer.ravel()
                        pad_size = m_layer.size - orig_size
                        padded_flat_g = jnp.pad(flat_g, (0, pad_size)) if pad_size > 0 else flat_g
                        g_blocked = padded_flat_g.reshape(m_layer.shape)
                        c_blocked = beta1 * m_layer + (1.0 - beta1) * g_blocked
                        new_m_blocked = beta2 * m_layer + (1.0 - beta2) * g_blocked
                        c = c_blocked.ravel()[:orig_size].reshape(orig_shape)
                        
                    new_m_q = QArrayImpl(new_m_blocked, tgrid)._tree_flatten()[0]
                    
                    if should_use_muon(path, g_layer):
                        u = newton_schulz(c, steps=ns_steps)
                        scale_factor = jnp.sqrt(jnp.maximum(orig_shape[0], orig_shape[1]))
                        update = -(muon_lr * (u * scale_factor) + muon_lr * weight_decay * p_layer)
                    else:
                        u = jnp.sign(c)
                        update = -(lion_lr * u + lion_lr * weight_decay * p_layer)
                        
                    return update, new_m_q[0], new_m_q[1], new_m_q[2]

                update, nm_val, nm_sc8, nm_sc32 = jax.lax.map(
                    map_fn,
                    (g.get_value(), g._QArrayImpl__sc_fp8, g._QArrayImpl__sc_fp32,
                     m.get_value(), m._QArrayImpl__sc_fp8, m._QArrayImpl__sc_fp32,
                     param.get_value(), param._QArrayImpl__sc_fp8, param._QArrayImpl__sc_fp32)
                )
                new_m = QArrayImpl._tree_unflatten((True, tgrid), (nm_val, nm_sc8, nm_sc32))
                return update, new_m

            tgrid = None
            if isinstance(param, QArrayImpl):
                tgrid = param.tgrid
                if getattr(m, '__class__', None).__name__ == 'State' and not isinstance(m, QArrayImpl):
                    m = QArrayImpl._tree_unflatten((True, tgrid), (m[0], m[1], m[2]))
                m = m.astype(jnp.float32)
                g = g.astype(jnp.float32)

            orig_shape = g.shape
            orig_size = g.size

            if m.shape == orig_shape:
                g_blocked = g
                c_blocked = beta1 * m + (1.0 - beta1) * g_blocked
                new_m_blocked = beta2 * m + (1.0 - beta2) * g_blocked
                c = c_blocked
            else:
                flat_g = g.ravel()
                pad_size = m.size - orig_size
                padded_flat_g = jnp.pad(flat_g, (0, pad_size)) if pad_size > 0 else flat_g
                g_blocked = padded_flat_g.reshape(m.shape)

                c_blocked = beta1 * m + (1.0 - beta1) * g_blocked
                new_m_blocked = beta2 * m + (1.0 - beta2) * g_blocked

                c = c_blocked.ravel()[:orig_size].reshape(orig_shape)

            new_m = new_m_blocked
            if isinstance(param, QArrayImpl):
                new_m = QArrayImpl(new_m_blocked, tgrid)

            if should_use_muon(path, g):
                u = newton_schulz(c, steps=ns_steps)
                scale_factor = jnp.sqrt(jnp.maximum(orig_shape[0], orig_shape[1]))
                p_f32 = param.astype(jnp.float32) if isinstance(param, QArrayImpl) else param
                update = -(muon_lr * (u * scale_factor) + muon_lr * weight_decay * p_f32)
            else:
                u = jnp.sign(c)
                p_f32 = param.astype(jnp.float32) if isinstance(param, QArrayImpl) else param
                update = -(lion_lr * u + lion_lr * weight_decay * p_f32)
                
            return update, new_m
        
        results = jax.tree_util.tree_map_with_path(
            update_leaf, updates, state.momentum, params,
            is_leaf=lambda p: isinstance(p, QArrayImpl)
        )

        new_updates = jax.tree_util.tree_map(lambda x: x[0], results, is_leaf=lambda x: isinstance(x, tuple) and len(x) == 2)
        new_m = jax.tree_util.tree_map(lambda x: x[1], results, is_leaf=lambda x: isinstance(x, tuple) and len(x) == 2)
        
        new_state = MiulionState(
            count=state.count + 1,
            momentum=new_m
        )

        return new_updates, new_state
    
    return optax.GradientTransformation(init_fn, update_fn)
