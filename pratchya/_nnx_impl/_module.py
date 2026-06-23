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
    x_norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=axis, keepdims=True) + eps**2)
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

    def __call__(self, input_ids: jax.Array) -> QArrayImpl:
        blksize = self.tgrid[-1]
        x = self.embedding[input_ids]
        x = x.astype(jnp.float32)
        x = x * self.embed_scale
        return QArrayImpl(x, (1, blksize))

class Linear(nnx.Module):
    def __init__(self, fan_in: int, fan_out: int, tgrid: tuple, *, rngs: nnx.Rngs, dtype: DTypeLike):
        kernel_init = nnx.initializers.lecun_normal()
        kernel = kernel_init(rngs.params(), (fan_in, fan_out), dtype)
        kernel = QArrayImpl(kernel, tgrid)
        self.kernel = nnx.Param(kernel)

    def __call__(self, x: QArrayImpl):
        return x @ self.kernel
    

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

    def __call__(self, x: QArrayImpl):
        tgrid = x.tgrid
        x = x.astype(jnp.float32)
        inv_rms = jax.lax.rsqrt(jnp.average(jnp.square(x), axis=-1, keepdims=True) + self.epsilon)
        x = x * inv_rms * self.scale.astype(jnp.float32)

        return QArrayImpl(x, tgrid)

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

    def __call__(self, x: QArrayImpl) -> QArrayImpl:
        x = self.down(x)
        x = x.apply(lambda _x: jax.nn.tanh(_x))
        return self.up(x)


class TimeMix(nnx.Module):
    def __init__(self, config: PratchyaConfig, *, rngs: nnx.Rngs, layer_idx: Optional[int] = None):
        self.layer_idx = layer_idx
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

    def __call__(
        self, x: QArrayImpl, 
        v_prime_0: jax.Array, 
        t_positions: jax.Array, 
        state: PratchyaState
    ):
        B, T, C, = x.shape
        N, H = self.n_head, self.head_dim
        layer_idx = self.layer_idx

        tm_state = state.tm_state[layer_idx]
        x_shifted = QArrayImpl.concat([tm_state, x[:, :-1, :]], axis=1)
        tm_state = x[:, -1:, :]

        _x = x.astype(jnp.float32)
        _x_shift = x_shifted.astype(jnp.float32)

        mu = self.mu.astype(jnp.float32)
        x_receptance    = QArrayImpl(lerp(_x, _x_shift, mu[0]), x.tgrid)
        x_decay         = QArrayImpl(lerp(_x, _x_shift, mu[1]), x.tgrid)
        x_key           = QArrayImpl(lerp(_x, _x_shift, mu[2]), x.tgrid)
        x_value         = QArrayImpl(lerp(_x, _x_shift, mu[3]), x.tgrid)
        x_iclr          = QArrayImpl(lerp(_x, _x_shift, mu[4]), x.tgrid)
        x_gate          = QArrayImpl(lerp(_x, _x_shift, mu[5]), x.tgrid)

        r =         self.w_receptance(x_receptance)
        d =         self.decay_lora(x_decay)
        k =         self.w_key(x_key)
        v_prime =   self.w_value(x_value)
        gate =      self.gate_lora(x_gate)
        iclr =      self.iclr_lora(x_iclr).apply(lambda _x_iclr: jax.nn.sigmoid(_x_iclr))

        k = k.astype(jnp.float32)
        r = r.astype(jnp.float32)

        k = k.reshape(B, T, -1, H)
        r = r.reshape(B, T, -1, H)

        k = self.rotary_emb(k, t_positions)
        r = self.rotary_emb(r, t_positions)

        k = k.reshape(B, T, -1)
        r = r.reshape(B, T, -1)

        if layer_idx == 0:
            v = v_prime_0 = v_prime.astype(jnp.float32)
        else:
            value_residual_gate = self.nu_lora(x_value)
            value_residual_gate = value_residual_gate.apply(lambda _x_value: jax.nn.sigmoid(_x_value))
            value_residual_gate = value_residual_gate.astype(jnp.float32)
            v_prime = v_prime.astype(jnp.float32)
            v_prime_0 = v_prime_0.astype(jnp.float32)
            v = lerp(v_prime, v_prime_0, value_residual_gate)

        decay = jnp.exp(-jnp.exp(-0.5) * jax.nn.sigmoid(d.astype(jnp.float32)))
        removal_k = k * jnp.astype(self.removal_key_multiplier.astype(jnp.float32), k.dtype)
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

        out = QArrayImpl(out, x.tgrid)
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

        self.layer_idx = layer_idx   

    def __call__(self, x: QArrayImpl, state: PratchyaState):
        tgrid = x.tgrid
        layer_idx = self.layer_idx

        cm_state = state.cm_state[layer_idx]
        x_shifted = QArrayImpl.concat([cm_state, x[:, :-1, :]], axis=1)
        cm_state = x[:, -1:, :]

        _x = x.astype(jnp.float32)
        _x_shifted = x_shifted.astype(jnp.float32)

        x_k = lerp(_x, _x_shifted, self.mu_x.astype(_x.dtype))
        x_k = QArrayImpl(x_k, tgrid)
        k = self.w_k(x_k).astype(jnp.float32)

        k = jnp.pow(jax.nn.relu(k), 2)
        k = QArrayImpl(k, tgrid)
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
        self, x: QArrayImpl, 
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
        
        return x, v_0, state


