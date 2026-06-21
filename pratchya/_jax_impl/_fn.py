
from jax.typing import DTypeLike, ArrayLike
import jax, jax.numpy as jnp

from .._config import PratchyaConfig

from ._utils import lerp, linear, lora_ffn, normalized, group_norm, apply_rope

def embed(input_ids: ArrayLike, params: ArrayLike):
    scale = params['w'].shape[-1]
    x = params['w'][input_ids] * jnp.sqrt(scale)
    return x

def rmsnorm(x: ArrayLike, params: ArrayLike, *, eps: float = 1e-6):
    inv_rms = jax.lax.rsqrt(jnp.average(jnp.square(x), axis=-1, keepdims=True) + eps)
    x = x * inv_rms * params['w']
    return x


def rwkv_block(x: ArrayLike, params, *, v0, state, config: PratchyaConfig):
    x_rmsnorm = x.apply(lambda v: rmsnorm(v, params['pre_tm_rmsnorm'])) if hasattr(x, 'apply') else rmsnorm(x, params['pre_tm_rmsnorm'])
    dx, v0, state = tm_fwd(x_rmsnorm, params['tm'], vprime_0=v0, state=state, config=config)
    x = x + dx
    x_rmsnorm2 = x.apply(lambda v: rmsnorm(v, params['pre_cm_rmsnorm'])) if hasattr(x, 'apply') else rmsnorm(x, params['pre_cm_rmsnorm'])
    dx, state = cm_fwd(x_rmsnorm2, params['cm'], state=state)
    x = x + dx

    state = {
        **state,
        'layer_idx': state['layer_idx'] + 1
    }
    return x, v0, state

def rwkv_init_block(x: ArrayLike, params, *, state, config: PratchyaConfig):
    x_rmsnorm = x.apply(lambda v: rmsnorm(v, params['pre_tm_rmsnorm'])) if hasattr(x, 'apply') else rmsnorm(x, params['pre_tm_rmsnorm'])
    dx, v0, state = tm_init_fwd(x_rmsnorm, params['tm'], state=state, config=config)
    x = x + dx
    x_rmsnorm2 = x.apply(lambda v: rmsnorm(v, params['pre_cm_rmsnorm'])) if hasattr(x, 'apply') else rmsnorm(x, params['pre_cm_rmsnorm'])
    dx, state = cm_fwd(x_rmsnorm2, params['cm'], state=state)
    x = x + dx

    state = {
        **state,
        'layer_idx': state['layer_idx'] + 1
    }
    return x, v0, state

def tm_prepare(layer_idx, x: ArrayLike, params, *, state, config: PratchyaConfig):
    B, T, C = x.shape
    H = config.head_dim
    
    tm_state = state['tm_state'][layer_idx]
    if hasattr(tm_state, 'concat'):
        x_fp8 = tm_state.__class__(x, tm_state.block_grid)
        x_prev_fp8 = tm_state.__class__(x[:, :-1, :], tm_state.block_grid)
        x_shift = tm_state.__class__.concat([tm_state, x_prev_fp8], axis=1)
        x_to_lerp = x_fp8
    else:
        x_shift = jnp.concat([tm_state, x[:, :-1, :]], axis=1)
        x_to_lerp = x
        
    tm_state = x[:, -1:, :]

    mu_r = params['mu']['w'][0]
    mu_d = params['mu']['w'][1]
    mu_k = params['mu']['w'][2]
    mu_v = params['mu']['w'][3]
    mu_a = params['mu']['w'][4]
    mu_g = params['mu']['w'][5]

    x_receptance    = lerp(x_to_lerp, x_shift, mu_r)
    x_decay         = lerp(x_to_lerp, x_shift, mu_d)
    x_key           = lerp(x_to_lerp, x_shift, mu_k)
    x_value         = lerp(x_to_lerp, x_shift, mu_v)
    x_iclr          = lerp(x_to_lerp, x_shift, mu_a)
    x_gate          = lerp(x_to_lerp, x_shift, mu_g)

    r       = linear(x_receptance, params['w_receptance'])
    d       = lora_ffn(x_decay, params['decay_lora'])
    k       = linear(x_key, params['w_key'])
    vprime  = linear(x_value, params['w_value'])
    gate    = lora_ffn(x_gate, params['gate_lora'])
    
    iclr    = lora_ffn(x_iclr, params['iclr_lora'])
    iclr    = iclr.apply(jax.nn.sigmoid) if hasattr(iclr, 'apply') else jax.nn.sigmoid(iclr)

    # rope
    k = k.reshape(B, T, -1, H)
    r = r.reshape(B, T, -1, H)

    t_positions = state['step'] + jnp.arange(T, dtype=jnp.int32)

    k = apply_rope(k, t_positions, state['inv_freq'])
    r = apply_rope(r, t_positions, state['inv_freq'])

    k = k.reshape(B, T, -1)
    r = r.reshape(B, T, -1)
    
    return r, d, k, vprime, gate, iclr, x_value, tm_state

def tm_final(x: ArrayLike, params, r, d, k, v, gate, iclr, wkv_state, *, config: PratchyaConfig):
    B, T, C = x.shape
    H = config.head_dim
    N = C // H

    decay = jnp.exp(-jnp.exp(-0.5) * jax.nn.sigmoid(d.astype(jnp.float32)))
    removal_k = k * params['removal_key_multiplier']['w']
    removal_k = normalized(removal_k.reshape(B, T, N, -1), axis=-1).reshape(B, T, -1)
    replacement_k = lerp(k, k * iclr, params['iclr_mix_amt']['w'])

    def recurent_scan_fn(carry, xs):
        wkv_state, t = carry

        decay_t         = decay[:, t].reshape(B, N, H, 1)
        iclr_t          = iclr[:, t].reshape(B, N, H, 1)
        removal_k_t     = removal_k[:, t].reshape(B, N, H, 1)
        replacement_k_t = replacement_k[:, t].reshape(B, N, H, 1)
        v_t             = v[:, t].reshape(B, N, H, 1)
        r_t             = r[:, t].reshape(B, N, H, 1)

        wkv_state = wkv_state * decay_t.mT - wkv_state @ removal_k_t @ (iclr_t * removal_k_t).mT
        wkv_state = wkv_state + v_t @ replacement_k_t.mT
        wkv_state = wkv_state.astype(jnp.float32) if hasattr(wkv_state, 'astype') else wkv_state
        y = wkv_state @ r_t

        return (wkv_state, t + 1), y.reshape(B, C)
    
    (wkv_state, _), out = jax.lax.scan(recurent_scan_fn, (wkv_state, 0), None, length=T)
    out = out.transpose(1, 0, 2)
    out = group_norm(out.reshape(B * T, C), N, params['group_norm']).reshape(B, T, C)

    bonus_base = r * k * params['bonus_multiplier']['w']
    bonus = bonus_base.sum(axis=-1, keepdims=True) if hasattr(bonus_base, 'sum') else jnp.sum(bonus_base, axis=-1, keepdims=True)
    bonus = bonus * v
    bonus = bonus.reshape(B, T, C)
    out = (out + bonus) * gate
    out = linear(out, params['w_output'])

    return out, wkv_state


def tm_init_fwd(x: ArrayLike, params, *, state, config: PratchyaConfig):
    layer_idx = state['layer_idx']
    r, d, k, vprime, gate, iclr, _, tm_state = tm_prepare(layer_idx, x, params, state=state, config=config)
    v = vprime
    out, wkv_state = tm_final(x, params, r, d, k, v, gate, iclr, state['wkv_state'][layer_idx], config=config)

    wkv_state = state['wkv_state'].at[layer_idx].set(wkv_state)
    tm_state = state['tm_state'].at[layer_idx].set(tm_state)
    state = {
        **state,
        'wkv_state': wkv_state,
        'tm_state': tm_state
    }

    return out, v, state


def tm_fwd(x: ArrayLike, params, *, vprime_0, state, config: PratchyaConfig):
    layer_idx = state['layer_idx']
    r, d, k, vprime, gate, iclr, x_value, tm_state = tm_prepare(layer_idx, x, params, state=state, config=config)
    v_residual_gate = lora_ffn(x_value, params['nu_lora'])
    v_residual_gate = v_residual_gate.apply(jax.nn.sigmoid) if hasattr(v_residual_gate, 'apply') else jax.nn.sigmoid(v_residual_gate)
    v = lerp(vprime, vprime_0, v_residual_gate)
    out, wkv_state = tm_final(x, params, r, d, k, v, gate, iclr, state['wkv_state'][layer_idx], config=config)
    
    wkv_state = state['wkv_state'].at[layer_idx].set(wkv_state)
    tm_state = state['tm_state'].at[layer_idx].set(tm_state)
    state = {
        **state,
        'wkv_state': wkv_state,
        'tm_state': tm_state
    }

    return out, v, state


def cm_fwd(x: ArrayLike, params, *, state):
    layer_idx = state['layer_idx']
    cm_state = state['cm_state'][layer_idx]
    if hasattr(cm_state, 'concat'):
        x_fp8 = cm_state.__class__(x, cm_state.block_grid)
        x_prev_fp8 = cm_state.__class__(x[:, :-1, :], cm_state.block_grid)
        x_shift = cm_state.__class__.concat([cm_state, x_prev_fp8], axis=1)
        x_to_lerp = x_fp8
    else:
        x_shift = jnp.concat([cm_state, x[:, :-1, :]], axis=1)
        x_to_lerp = x
        
    cm_state = x[:, -1:, :]
    x_k = lerp(x_to_lerp, x_shift, params['mu_x']['w'])
    k = linear(x_k, params['w_k'])
    k = k.apply(lambda v: jnp.pow(jax.nn.relu(v), 2)) if hasattr(k, 'apply') else jnp.pow(jax.nn.relu(k), 2)
    v = linear(k, params['w_v'])

    cm_state = state['cm_state'].at[layer_idx].set(cm_state)
    state = {
        **state,
        'cm_state': cm_state
    }

    return v, state


def fwd_fn(input_ids: ArrayLike, params: dict, *, state, config: PratchyaConfig):
    x = embed(input_ids, params['embed_tokens'])
    x = x.apply(lambda v: rmsnorm(v, params['pre_rmsnorm'], eps=config.rmsnorm_epsilon)) if hasattr(x, 'apply') else rmsnorm(x, params['pre_rmsnorm'], eps=config.rmsnorm_epsilon)
    x, v0, state = rwkv_init_block(x, params['rwkv_init_block'], state=state, config=config)

    def rwkv_block_scan(carry, params):
        x, v0, state = carry
        x, v0, state = rwkv_block(x, params, v0=v0, state=state, config=config)
        return (x, v0, state), None

    (x, _, state), _ = jax.lax.scan(rwkv_block_scan, (x, v0, state), params['rwkv_block'])
    x = x.apply(lambda v: rmsnorm(v, params['final_rmsnorm'], eps=config.rmsnorm_epsilon)) if hasattr(x, 'apply') else rmsnorm(x, params['final_rmsnorm'], eps=config.rmsnorm_epsilon)
    logits = linear(x, params['lm_head'])

    state = {
        **state,
        'layer_idx': 0
    }
    return logits
    




