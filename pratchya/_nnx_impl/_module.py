from flax import nnx
import jax.numpy as jnp
import jax
from jax.typing import DTypeLike
from typing import Optional, Union

from .._config import PratchyaConfig, PratchyaOutput, PratchyaState
from ._utils import compute_loss
from .._qualia import QArrayImpl


def lerp(i: jax.Array, j: jax.Array, w: jax.Array):
    return i + (j - i) * w

def normalized(x: jax.Array, *, axis: int, ord: int = 2, eps: float = 1e-12) -> jax.Array:
    x_norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=axis, keepdims=True) + eps)
    x = x / x_norm
    return x


class EmbeddingScaled(nnx.Module):
    def __init__(self, vocab_size: int, hidden_size: int, tgrid: tuple[int, int], *, rngs: nnx.Rngs, dtype: DTypeLike):
        embed_init = nnx.initializers.variance_scaling(
            1.0, 'fan_in', 'normal', out_axis=0
        )
        embed = embed_init(rngs.params(), (vocab_size, hidden_size), dtype)
        embed = QArrayImpl(embed, tgrid)
        self.embedding = nnx.Param(embed)
        self.embed_scale = jnp.sqrt(hidden_size)
        self.tgrid = tgrid

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        blksize = self.tgrid[-1]
        x = self.embedding[input_ids]
        x = x.astype(jnp.float32)
        x = x * self.embed_scale
        return x

class Linear(nnx.Module):
    def __init__(self, fan_in: int, fan_out: int, tgrid: tuple, *, rngs: nnx.Rngs, dtype: DTypeLike):
        kernel_init = nnx.initializers.lecun_normal()
        kernel = kernel_init(rngs.params(), (fan_in, fan_out), dtype)
        kernel = QArrayImpl(kernel, tgrid)
        self.kernel = nnx.Param(kernel)

    def __call__(self, x: jax.Array):
        a, b = self.kernel.tgrid
        tgrid_x = (1, a)
        x_q = QArrayImpl(x, tgrid_x)
        return (x_q @ self.kernel).astype(x.dtype)
    

class Param(nnx.Param):
    def __init__(self, shape: tuple, tgrid: tuple, *, rngs: nnx.Rngs, dtype: DTypeLike):
        kernel_init = nnx.initializers.lecun_normal()
        kernel = kernel_init(rngs.params(), shape, dtype)
        kernel = QArrayImpl(kernel, tgrid)
        super().__init__(kernel)


class RMSNorm(nnx.Module):
    def __init__(self, hidden_size: int, tgrid: tuple, *, epsilon: float = 1e-6):
        self.epsilon = epsilon
        scale = jnp.ones((1, hidden_size), dtype=jnp.float32)
        scale = QArrayImpl(scale, tgrid)
        self.scale = nnx.Param(scale)

    def __call__(self, x: jax.Array):
        x_f32 = x.astype(jnp.float32)
        inv_rms = jax.lax.rsqrt(jnp.average(jnp.square(x_f32), axis=-1, keepdims=True) + self.epsilon)
        out = x_f32 * inv_rms * self.scale.astype(jnp.float32)
        return out.astype(x.dtype)

class LowRankFFN(nnx.Module):
    def __init__(
        self, hidden_size: int, 
        lora_rank: int, 
        tgrid: tuple, *, 
        rngs: nnx.Rngs, 
        dtype: DTypeLike
    ):
        a, b = tgrid
        self.down = Linear(hidden_size, lora_rank, (a, 1), rngs=rngs, dtype=dtype)
        self.up = Linear(lora_rank, hidden_size, (b, 1), rngs=rngs, dtype=dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.down(x)
        x = jax.nn.tanh(x)
        return self.up(x)


class TimeMix(nnx.Module):
    def __init__(self, config: PratchyaConfig, *, rngs: nnx.Rngs, layer_idx: Optional[int] = None):
        self.head_dim = config.head_dim
        self.n_head = config.hidden_size // config.head_dim

        self.mu = Param((6, config.hidden_size), (1, config.blksize), rngs=rngs, dtype=config.mu_dtype)

        tgrid = (config.blksize, config.blksize)
        self.w_key = Linear(config.hidden_size, config.hidden_size, tgrid, rngs=rngs, dtype=config.dtype)
        self.w_value = Linear(config.hidden_size, config.hidden_size, tgrid, rngs=rngs, dtype=config.dtype)
        self.w_output = Linear(config.hidden_size, config.hidden_size, tgrid, rngs=rngs, dtype=config.dtype)
        self.w_receptance = Linear(config.hidden_size, config.hidden_size, tgrid, rngs=rngs, dtype=config.dtype)

        self.gate_lora = LowRankFFN(config.hidden_size, config.lora_rank, tgrid, rngs=rngs, dtype=config.lora_dtype)
        self.decay_lora = LowRankFFN(config.hidden_size, config.lora_rank, tgrid, rngs=rngs, dtype=config.lora_dtype)
        self.iclr_lora = LowRankFFN(config.hidden_size, config.lora_rank, tgrid, rngs=rngs, dtype=config.lora_dtype)
        
        if layer_idx != 0:
            self.nu_lora = LowRankFFN(config.hidden_size, config.lora_rank, tgrid, rngs=rngs, dtype=config.lora_dtype)

        self.iclr_mix_amt = Param((1, config.hidden_size), (1, config.blksize), rngs=rngs, dtype=config.mu_dtype)
        self.removal_key_multiplier = Param((1, config.hidden_size), (1, config.blksize), rngs=rngs, dtype=config.mu_dtype)
        self.bonus_multiplier = Param((1, config.hidden_size), (1, config.blksize), rngs=rngs, dtype=config.mu_dtype)

        self.group_norm = nnx.GroupNorm(
            config.hidden_size, self.n_head, 
            epsilon=self.n_head * 1e-5, rngs=rngs, 
            dtype=config.norm_dtype, 
            param_dtype=config.norm_dtype
        )

        self.rotary_emb = RotaryEmbedding(config)
        self.is_first_layer = layer_idx == 0

    def __call__(
        self, x: jax.Array, 
        v_prime_0: jax.Array, 
        t_positions: jax.Array, 
        state: PratchyaState
    ):
        B, T, C, = x.shape
        N, H = self.n_head, self.head_dim
        layer_idx = state.layer_idx

        tm_state = state.tm_state[layer_idx]
        x_shifted = jnp.concatenate([tm_state, x[:, :-1, :]], axis=1)
        tm_state = x[:, -1:, :]

        _x = x.astype(jnp.float32)
        _x_shift = x_shifted.astype(jnp.float32)
        mu = self.mu.astype(jnp.float32)

        x_receptance    = lerp(_x, _x_shift, mu[0])
        x_decay         = lerp(_x, _x_shift, mu[1])
        x_key           = lerp(_x, _x_shift, mu[2])
        x_value         = lerp(_x, _x_shift, mu[3])
        x_iclr          = lerp(_x, _x_shift, mu[4])
        x_gate          = lerp(_x, _x_shift, mu[5])

        r =         self.w_receptance(x_receptance)
        d =         self.decay_lora(x_decay)
        k =         self.w_key(x_key)
        v_prime =   self.w_value(x_value)
        gate =      self.gate_lora(x_gate)
        iclr =      jax.nn.sigmoid(self.iclr_lora(x_iclr))

        k = k.astype(jnp.float32)
        r = r.astype(jnp.float32)

        k = k.reshape(B, T, -1, H)
        r = r.reshape(B, T, -1, H)

        k = self.rotary_emb(k, t_positions)
        r = self.rotary_emb(r, t_positions)

        k = k.reshape(B, T, -1)
        r = r.reshape(B, T, -1)

        if self.is_first_layer:
            v = v_prime_0 = v_prime.astype(jnp.float32)
        else:
            value_residual_gate = self.nu_lora(x_value)
            value_residual_gate = jax.nn.sigmoid(value_residual_gate)
            value_residual_gate = value_residual_gate.astype(jnp.float32)
            v_prime = v_prime.astype(jnp.float32)
            v_prime_0 = v_prime_0.astype(jnp.float32)
            v = lerp(v_prime, v_prime_0, value_residual_gate)

        decay = jnp.exp(-jnp.exp(d))
        removal_k = k * self.removal_key_multiplier.astype(jnp.float32)
        removal_k = normalized(removal_k.reshape(B, T, N, -1), axis=-1).reshape(B, T, C)
        iclr = iclr.astype(jnp.float32)
        iclr_mix_amt = self.iclr_mix_amt.astype(jnp.float32)
        replacement_k = lerp(k, k * iclr, iclr_mix_amt)

        @nnx.scan(in_axes=(nnx.Carry, 1, 1, 1, 1, 1, 1), out_axes=(nnx.Carry, 0))
        def wkv_state_scan(wkv_state, decay_t, iclr_t, removal_k_t, replacement_k_t, v_t, r_t):

            decay_t =           decay_t.reshape(B, N, H, 1)
            iclr_t =            iclr_t.reshape(B, N, H, 1)
            removal_k_t =       removal_k_t.reshape(B, N, H, 1)
            replacement_k_t =   replacement_k_t.reshape(B, N, H, 1)
            v_t =               v_t.reshape(B, N, H, 1)
            r_t =               r_t.reshape(B, N, H, 1)

            wkv_state = wkv_state * decay_t.mT - wkv_state @ removal_k_t @ (iclr_t * removal_k_t).mT
            wkv_state = wkv_state + v_t @ replacement_k_t.mT
            y = wkv_state @ r_t

            return wkv_state, y.reshape(B, C)

        wkv_state, out = wkv_state_scan(
            state.wkv_state[layer_idx],
            decay, iclr, removal_k,
            replacement_k, v, r
        )

        # [T, B, C]
        out = out.transpose(1, 0, 2)
        out = self.group_norm(out)

        bonus = jnp.sum(r * k * self.bonus_multiplier.astype(k.dtype), axis=-1, keepdims=True) * v
        bonus = bonus.reshape(B, T, C)
        out = (out + bonus) * gate.astype(jnp.float32)
        out = self.w_output(out)

        wkv_state = state.wkv_state.at[layer_idx].set(wkv_state)
        tm_state = state.tm_state.at[layer_idx].set(tm_state)
        state = state.replace(
            tm_state = tm_state,
            wkv_state = wkv_state
        )

        return out, v, state


class ChannelMix(nnx.Module):

    def __init__(self, config: PratchyaConfig, *, rngs: nnx.Rngs, layer_idx: Optional[int] = None):
        tgrid = (config.blksize, config.blksize)
        self.w_k = Linear(config.hidden_size, config.intermediate_size, tgrid, rngs=rngs, dtype=config.dtype)
        self.w_v = Linear(config.intermediate_size, config.hidden_size, tgrid, rngs=rngs, dtype=config.dtype)
        self.mu_x = Param((1, config.hidden_size), (1, config.blksize), rngs=rngs, dtype=config.mu_dtype)

    def __call__(self, x: jax.Array, state: PratchyaState):
        tgrid = self.w_k.kernel.tgrid
        layer_idx = state.layer_idx

        cm_state = state.cm_state[layer_idx]
        x_shifted = jnp.concatenate([cm_state, x[:, :-1, :]], axis=1)
        cm_state = x[:, -1:, :]

        _x = x.astype(jnp.float32)
        _x_shifted = x_shifted.astype(jnp.float32)

        x_k = lerp(_x, _x_shifted, self.mu_x.astype(_x.dtype))
        k = self.w_k(x_k).astype(jnp.float32)

        k = jnp.pow(jax.nn.relu(k), 2)
        v = self.w_v(k)

        cm_state = state.cm_state.at[layer_idx].set(cm_state)
        state = state.replace(
            cm_state = cm_state,
        )

        return v, state


class RotaryEmbedding(nnx.Module):
    def __init__(self, config: PratchyaConfig):
        head_dim = config.head_dim
        base = config.rope_theta
        self.inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))

    def rotate_half(self, x: jax.Array):
        d = x.shape[-1]
        x1 = x[..., :d//2]
        x2 = x[..., d//2:]
        return jnp.concatenate([-x2, x1], axis=-1)

    def __call__(self, x: jax.Array, t: Union[int, jax.Array]):
        t_array = jnp.atleast_1d(t).astype(jnp.float32)
        freqs = jnp.outer(t_array, self.inv_freq) # Shape: (Batch or 1, head_dim // 2)
        
        emb = jnp.concatenate([freqs, freqs], axis=-1) # Shape: (Batch or 1, head_dim)
        
        emb = emb[:, None, :] 
        cos = jnp.cos(emb)
        sin = jnp.sin(emb)
        
        return (x * cos) + (self.rotate_half(x) * sin)


class RWKVBlock(nnx.Module):
    def __init__(self, config: PratchyaConfig, *, rngs: nnx.Rngs, layer_idx: Optional[int] = None):

        self.tm = TimeMix(config, rngs=rngs, layer_idx=layer_idx)
        self.cm = ChannelMix(config, rngs=rngs, layer_idx=layer_idx)
        self.pre_rmsnorm = RMSNorm(
            config.hidden_size, (1, config.blksize),
            epsilon=config.rmsnorm_epsilon
        )
        self.post_rmsnorm = RMSNorm(
            config.hidden_size, (1, config.blksize),
            epsilon=config.rmsnorm_epsilon, 
        )

    def __call__(
        self, x: jax.Array, 
        v_0: jax.Array, 
        t_positions: jax.Array,
        state: PratchyaState
    ):

        x = self.pre_rmsnorm(x)
        dx, v_0, state = self.tm(x, v_0, t_positions, state)
        x = x + dx
        
        x = self.post_rmsnorm(x)
        dx, state = self.cm(x, state)
        x = x + dx

        state = state.replace(layer_idx=state.layer_idx + 1)
        
        return x, v_0, state



class RWKVState(nnx.Module):
    def __init__(self, config: PratchyaConfig, *, rngs: nnx.Rngs):
        self.w_tm = Linear(config.hidden_size, config.hidden_size, (config.blksize, config.blksize), rngs=rngs, dtype=config.act_dtype)
        self.w_cm = Linear(config.hidden_size, config.hidden_size, (config.blksize, config.blksize), rngs=rngs, dtype=config.act_dtype)
        self.w_wkv = nnx.Linear(config.head_dim, config.head_dim * config.head_dim, rngs=rngs, dtype=config.wkv_dtype)
        self.gate = Linear(config.hidden_size, 1, (config.blksize, 1), rngs=rngs, dtype=config.dtype)
        
        self.spl_tm = nnx.Param(jnp.ones((config.n_layers, 1, 1, 1), config.act_dtype))
        self.spl_cm = nnx.Param(jnp.ones((config.n_layers, 1, 1, 1), config.act_dtype))
        self.spl_wkv = nnx.Param(jnp.ones((config.n_layers, 1, 1, 1, 1), config.wkv_dtype))

        self.head_dim = config.head_dim

    def __call__(self, x: jax.Array) -> PratchyaState:
        B, T, C = x.shape
        H = self.head_dim
        N = C // H

        score = self.gate(x)
        weights = jax.nn.softmax(score, axis=-2)
        gstate = jnp.sum(x * weights, axis=-2, keepdims=True)
        
        tm_state_i = self.spl_tm * self.w_tm(gstate)[None, ...]
        cm_state_i = self.spl_cm * self.w_cm(gstate)[None, ...]
        wkv_state_i = self.spl_wkv * self.w_wkv(gstate.reshape(B, N, H)).reshape(B, N, H, H)[None, ...]

        return PratchyaState(
            tm_state=tm_state_i,
            cm_state=cm_state_i,
            wkv_state=wkv_state_i,
            step=0
        )



class NQEmbeddingScaled(nnx.Embed):
    def __init__(self, vocab_size: int, hidden_size: int, *, rngs: nnx.Rngs, dtype: DTypeLike):
        super().__init__(vocab_size, hidden_size, param_dtype=dtype, rngs=rngs)
        self.embed_scale = jnp.sqrt(hidden_size)

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        return super().__call__(input_ids) * self.embed_scale

class NQLinear(nnx.Module):
    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs, dtype: DTypeLike):
        kernel_init = nnx.initializers.lecun_normal()
        kernel = kernel_init(rngs.params(), (in_features, out_features), dtype)
        self.kernel = nnx.Param(kernel)

        def get_blksize(f: int) -> int:
            if f <= 0: return 1
            return min(128, f & -f)
            
        self.blksize_in = get_blksize(in_features)
        self.blksize_out = get_blksize(out_features)

    def __call__(self, x: jax.Array):
        dtype = x.dtype
        # tgrid_w = (self.blksize_in, self.blksize_out)
        # tgrid_x = (1, self.blksize_in)

        # x_q = QArrayImpl(x, tgrid_x)
        # w_q = QArrayImpl(self.kernel, tgrid_w)

        # return (x_q @ w_q).astype(dtype)
        return (x @ self.kernel).astype(dtype)
    

class NQParam(nnx.Param):
    def __init__(self, shape: tuple, *, rngs: nnx.Rngs, dtype: DTypeLike):
        kernel_init = nnx.initializers.lecun_normal()
        kernel = kernel_init(rngs.params(), shape, dtype)
        super().__init__(kernel)


class NQRMSNorm(nnx.Module):
    def __init__(self, hidden_size: int, *, epsilon: float = 1e-6):
        self.epsilon = epsilon
        scale = jnp.ones((1, hidden_size), dtype=jnp.float32)
        self.scale = nnx.Param(scale)

    def __call__(self, x: jax.Array):
        x_f32 = x.astype(jnp.float32)
        inv_rms = jax.lax.rsqrt(jnp.average(jnp.square(x_f32), axis=-1, keepdims=True) + self.epsilon)
        out = x_f32 * inv_rms * self.scale.astype(jnp.float32)
        return out.astype(x.dtype)

class NQLowRankFFN(nnx.Module):
    def __init__(
        self, hidden_size: int, 
        lora_rank: int, *, 
        rngs: nnx.Rngs, 
        dtype: DTypeLike
    ):
        self.down = NQLinear(hidden_size, lora_rank, rngs=rngs, dtype=dtype)
        self.up = NQLinear(lora_rank, hidden_size, rngs=rngs, dtype=dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.down(x)
        x = jax.nn.tanh(x)
        return self.up(x)


class NQTimeMix(nnx.Module):
    def __init__(self, config: PratchyaConfig, *, rngs: nnx.Rngs, layer_idx: Optional[int] = None):
        self.head_dim = config.head_dim
        self.n_head = config.hidden_size // config.head_dim

        self.mu = NQParam((6, config.hidden_size), rngs=rngs, dtype=config.mu_dtype)

        self.w_key = NQLinear(config.hidden_size, config.hidden_size, rngs=rngs, dtype=config.dtype)
        self.w_value = NQLinear(config.hidden_size, config.hidden_size, rngs=rngs, dtype=config.dtype)
        self.w_output = NQLinear(config.hidden_size, config.hidden_size, rngs=rngs, dtype=config.dtype)
        self.w_receptance = NQLinear(config.hidden_size, config.hidden_size, rngs=rngs, dtype=config.dtype)

        self.gate_lora = NQLowRankFFN(config.hidden_size, config.lora_rank, rngs=rngs, dtype=config.lora_dtype)
        self.decay_lora = NQLowRankFFN(config.hidden_size, config.lora_rank, rngs=rngs, dtype=config.lora_dtype)
        self.iclr_lora = NQLowRankFFN(config.hidden_size, config.lora_rank, rngs=rngs, dtype=config.lora_dtype)
        
        if layer_idx != 0:
            self.nu_lora = NQLowRankFFN(config.hidden_size, config.lora_rank, rngs=rngs, dtype=config.lora_dtype)

        self.iclr_mix_amt = NQParam((1, config.hidden_size), rngs=rngs, dtype=config.mu_dtype)
        self.removal_key_multiplier = NQParam((1, config.hidden_size), rngs=rngs, dtype=config.mu_dtype)
        self.bonus_multiplier = NQParam((1, config.hidden_size), rngs=rngs, dtype=config.mu_dtype)

        self.group_norm = nnx.GroupNorm(
            config.hidden_size, self.n_head, 
            epsilon=self.n_head * 1e-5, rngs=rngs, 
            dtype=config.norm_dtype, 
            param_dtype=config.norm_dtype
        )

        self.rotary_emb = RotaryEmbedding(config)
        self.is_first_layer = layer_idx == 0

    def __call__(
        self, x: jax.Array, 
        v_prime_0: jax.Array, 
        t_positions: jax.Array, 
        state: PratchyaState
    ):
        B, T, C, = x.shape
        N, H = self.n_head, self.head_dim
        layer_idx = state.layer_idx

        tm_state = state.tm_state[layer_idx]
        x_shifted = jnp.concatenate([tm_state, x[:, :-1, :]], axis=1)
        tm_state = x[:, -1:, :]

        x_receptance    = lerp(x, x_shifted, self.mu[0])
        x_decay         = lerp(x, x_shifted, self.mu[1])
        x_key           = lerp(x, x_shifted, self.mu[2])
        x_value         = lerp(x, x_shifted, self.mu[3])
        x_iclr          = lerp(x, x_shifted, self.mu[4])
        x_gate          = lerp(x, x_shifted, self.mu[5])

        r =         self.w_receptance(x_receptance)
        d =         self.decay_lora(x_decay)
        k =         self.w_key(x_key)
        v_prime =   self.w_value(x_value)
        gate =      self.gate_lora(x_gate)
        iclr =      jax.nn.sigmoid(self.iclr_lora(x_iclr))

        k = k.reshape(B, T, -1, H)
        r = r.reshape(B, T, -1, H)

        k = self.rotary_emb(k, t_positions)
        r = self.rotary_emb(r, t_positions)

        k = k.reshape(B, T, -1)
        r = r.reshape(B, T, -1)

        if self.is_first_layer:
            v = v_prime_0 = v_prime
        else:
            value_residual_gate = jax.nn.sigmoid(self.nu_lora(x_value))
            v = lerp(v_prime, v_prime_0, value_residual_gate)

        decay = jnp.exp(-jnp.exp(d.astype(jnp.float32)))
        removal_k = k * self.removal_key_multiplier
        removal_k = normalized(removal_k.reshape(B, T, N, -1), axis=-1).reshape(B, T, C)
        replacement_k = lerp(k, k * iclr, self.iclr_mix_amt)

        @nnx.scan(in_axes=(nnx.Carry, 1, 1, 1, 1, 1, 1), out_axes=(nnx.Carry, 0))
        def wkv_state_scan(wkv_state, decay_t, iclr_t, removal_k_t, replacement_k_t, v_t, r_t):

            decay_t =           decay_t.reshape(B, N, H, 1)
            iclr_t =            iclr_t.reshape(B, N, H, 1)
            removal_k_t =       removal_k_t.reshape(B, N, H, 1)
            replacement_k_t =   replacement_k_t.reshape(B, N, H, 1)
            v_t =               v_t.reshape(B, N, H, 1)
            r_t =               r_t.reshape(B, N, H, 1)

            wkv_state = wkv_state * decay_t.mT - wkv_state @ removal_k_t @ (iclr_t * removal_k_t).mT
            wkv_state = wkv_state + v_t @ replacement_k_t.mT
            y = wkv_state @ r_t

            return wkv_state, y.reshape(B, C)

        wkv_state, out = wkv_state_scan(
            state.wkv_state[layer_idx],
            decay, iclr, removal_k,
            replacement_k, v, r
        )

        # [T, B, C]
        out = out.transpose(1, 0, 2)
        out = self.group_norm(out)

        bonus = jnp.sum(r * k * self.bonus_multiplier, axis=-1, keepdims=True) * v
        bonus = bonus.reshape(B, T, C)
        out = (out + bonus) * gate
        out = self.w_output(out)

        wkv_state = state.wkv_state.at[layer_idx].set(wkv_state)
        tm_state = state.tm_state.at[layer_idx].set(tm_state)
        state = state.replace(
            tm_state = tm_state,
            wkv_state = wkv_state
        )

        return out, v, state


class NQChannelMix(nnx.Module):

    def __init__(self, config: PratchyaConfig, *, rngs: nnx.Rngs, layer_idx: Optional[int] = None):
        self.w_k = NQLinear(config.hidden_size, config.intermediate_size, rngs=rngs, dtype=config.dtype)
        self.w_v = NQLinear(config.intermediate_size, config.hidden_size, rngs=rngs, dtype=config.dtype)
        self.mu_x = NQParam((1, config.hidden_size), rngs=rngs, dtype=config.mu_dtype)

    def __call__(self, x: jax.Array, state: PratchyaState):
        layer_idx = state.layer_idx

        cm_state = state.cm_state[layer_idx]
        x_shifted = jnp.concatenate([cm_state, x[:, :-1, :]], axis=1)
        cm_state = x[:, -1:, :]

        x_k = lerp(x, x_shifted, self.mu_x)
        k = self.w_k(x_k)

        k = jnp.pow(jax.nn.relu(k), 2)
        v = self.w_v(k)

        cm_state = state.cm_state.at[layer_idx].set(cm_state)
        state = state.replace(
            cm_state = cm_state,
        )

        return v, state


class NQRWKVBlock(nnx.Module):
    def __init__(self, config: PratchyaConfig, *, rngs: nnx.Rngs, layer_idx: Optional[int] = None):

        self.tm = NQTimeMix(config, rngs=rngs, layer_idx=layer_idx)
        self.cm = NQChannelMix(config, rngs=rngs, layer_idx=layer_idx)
        self.pre_rmsnorm = NQRMSNorm(
            config.hidden_size,
            epsilon=config.rmsnorm_epsilon
        )
        self.post_rmsnorm = NQRMSNorm(
            config.hidden_size,
            epsilon=config.rmsnorm_epsilon, 
        )

    def __call__(
        self, x: jax.Array, 
        v_0: jax.Array, 
        t_positions: jax.Array,
        state: PratchyaState
    ):

        x = self.pre_rmsnorm(x)
        dx, v_0, state = self.tm(x, v_0, t_positions, state)
        x = x + dx
        
        x = self.post_rmsnorm(x)
        dx, state = self.cm(x, state)
        x = x + dx

        state = state.replace(layer_idx=state.layer_idx + 1)
        
        return x, v_0, state


class NQRWKVState(nnx.Module):
    def __init__(self, config: PratchyaConfig, *, rngs: nnx.Rngs):
        self.w_tm = NQLinear(config.hidden_size, config.hidden_size, rngs=rngs, dtype=config.act_dtype)
        self.w_cm = NQLinear(config.hidden_size, config.hidden_size, rngs=rngs, dtype=config.act_dtype)
        self.w_wkv = NQLinear(config.head_dim, config.head_dim * config.head_dim, rngs=rngs, dtype=config.wkv_dtype)
        self.gate = NQLinear(config.hidden_size, 1, rngs=rngs, dtype=config.dtype)
        
        self.spl_tm = nnx.Param(jnp.ones((config.n_layers), config.act_dtype))
        self.spl_cm = nnx.Param(jnp.ones((config.n_layers), config.act_dtype))
        self.spl_wkv = nnx.Param(jnp.ones((config.n_layers), config.wkv_dtype))

        self.head_dim = config.head_dim

    def __call__(self, x: jax.Array) -> PratchyaState:
        B, T, C = x.shape
        H = self.head_dim
        N = C // H

        score = self.gate(x)
        weights = jax.nn.softmax(score, axis=-2)
        gstate = jnp.sum(x * weights, axis=-2, keepdims=True)
        
        tm_state_i = self.spl_tm[:, None, None, None] * self.w_tm(gstate)[None, ...]
        cm_state_i = self.spl_cm[:, None, None, None] * self.w_cm(gstate)[None, ...]
        wkv_state_i = self.spl_wkv[:, None, None, None, None] * self.w_wkv(gstate.reshape(B, N, H)).reshape(B, N, H, H)[None, ...]

        return PratchyaState(
            tm_state=tm_state_i,
            cm_state=cm_state_i,
            wkv_state=wkv_state_i,
            step=0
        )