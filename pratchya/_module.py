from flax import nnx
import jax.numpy as jnp
import jax
from jax.typing import ArrayLike, DTypeLike
from typing import Optional, Union
from ._config import PratchyaConfig, PratchyaOutput, PratchyaState
from ._kernel import block_quantize
from ._utils import dequantize_cast


def lerp(i: ArrayLike, j: ArrayLike, w: ArrayLike):
    return i + (j - i) * w

def normalized(x: ArrayLike, *, axis: int, ord: int = 2, eps: float = 1e-12) -> ArrayLike:
    x_norm = jnp.linalg.norm(x, ord=ord, axis=axis, keepdims=True)
    x = x / jnp.maximum(x_norm, eps)
    return x

class ScaledWordEmbedding(nnx.Embed):
    def __init__(
        self, vocab_size: int, 
        hidden_size: int, *, 
        rngs: nnx.Rngs, 
        dtype: DTypeLike | None = None, 
        param_dtype: DTypeLike = jnp.float32
    ):
        super().__init__(
            vocab_size, hidden_size, 
            rngs=rngs, dtype=dtype, 
            param_dtype=param_dtype
        )
        self.embed_scaled = jnp.sqrt(hidden_size)

    def __call__(self, inputs: ArrayLike):
        return super().__call__(inputs) * self.embed_scaled

from ._kernel._opt import rmsnorm_quantize

class RMSNorm(nnx.Module):
    def __init__(
        self, hidden_size, *, 
        epsilon = 0.000001
    ):
        self.epsilon = epsilon
        self.scale = nnx.Param(jnp.ones(hidden_size, jnp.float32))

    def __call__(self, x: ArrayLike, x_s: ArrayLike):
        x, x_s = rmsnorm_quantize(
            x, x_s, 
            self.scale.get_value(), 
            eps=self.epsilon
        )
        return x, x_s

class LowRankFFN(nnx.Module):
    def __init__(
        self, hidden_size: int, 
        lora_rank: int, *, 
        rngs: nnx.Rngs, 
        dtype: DTypeLike | None = None, 
        param_dtype: DTypeLike = jnp.float32,
        preferred_element_type: DTypeLike | None = None
    ):
        self.lin1 = nnx.Linear(
            hidden_size, lora_rank, 
            use_bias=False, rngs=rngs, 
            dtype=dtype, param_dtype=param_dtype, 
            preferred_element_type=preferred_element_type
        )
        self.lin2 = nnx.Linear(
            lora_rank, hidden_size, 
            use_bias=False, rngs=rngs, 
            dtype=dtype, param_dtype=param_dtype, 
            preferred_element_type=preferred_element_type
        )
    
    def __call__(self, x: ArrayLike):
        return self.lin2(jax.nn.tanh(self.lin1(x)))


class TimeMix(nnx.Module):
    def __init__(self, config: PratchyaConfig, *, rngs: nnx.Rngs, first_layer: Optional[bool] = None):

        if config.head_dim:
            assert config.hidden_size % config.head_dim == 0, \
                "hidden_size can't be devided by head_dim"
            self.head_dim = config.head_dim
            self.num_head = config.hidden_size // self.head_dim
        elif config.n_head:
            assert config.hidden_size % config.n_head == 0, \
                "hidden_size can't be devided by n_head"
            self.num_head = config.n_head
            self.head_dim = config.hidden_size // self.num_head
        else:
            assert config.head_dim or config.n_head, \
                "neither head_dim nor n_head specified, can't define multihead time mixing module"

        self.mu = nnx.Param(jnp.ones((6, config.hidden_size), dtype=config.mu_dtype) * 0.5)

        self.w_key = nnx.Linear(
            config.hidden_size, config.hidden_size, 
            use_bias=False, rngs=rngs, 
            dtype=config.gemm_dtype, 
            param_dtype=config.param_dtype,
            preferred_element_type=jnp.bfloat16
        )
        self.w_value = nnx.Linear(
            config.hidden_size, config.hidden_size, 
            use_bias=False, rngs=rngs, 
            dtype=config.gemm_dtype, 
            param_dtype=config.param_dtype,
            preferred_element_type=jnp.bfloat16
        )
        self.w_output = nnx.Linear(
            config.hidden_size, config.hidden_size, 
            use_bias=False, rngs=rngs, 
            dtype=config.gemm_dtype, 
            param_dtype=config.param_dtype
        )
        self.w_receptance = nnx.Linear(
            config.hidden_size, config.hidden_size, 
            use_bias=False, rngs=rngs, 
            dtype=config.gemm_dtype, 
            param_dtype=config.param_dtype,
            preferred_element_type=jnp.bfloat16
        )

        self.gate_lora = LowRankFFN(
            config.hidden_size, config.lora_rank, 
            rngs=rngs, dtype=config.gemm_dtype, 
            param_dtype=config.lora_dtype,
            preferred_element_type=jnp.bfloat16
        )

        self.first_layer = first_layer
        if not first_layer:
            self.nu_lora = LowRankFFN(
                config.hidden_size, config.lora_rank, 
                rngs=rngs, dtype=config.gemm_dtype, 
                param_dtype=config.lora_dtype,
                preferred_element_type=jnp.bfloat16
            )
    
        self.decay_lora = LowRankFFN(
            config.hidden_size, config.lora_rank, 
            rngs=rngs, dtype=config.gemm_dtype, 
            param_dtype=config.lora_dtype,
            preferred_element_type=jnp.bfloat16
        )
        self.iclr_lora = LowRankFFN(
            config.hidden_size, config.lora_rank, 
            rngs=rngs, dtype=config.gemm_dtype, 
            param_dtype=config.lora_dtype,
            preferred_element_type=jnp.bfloat16
        )
        self.iclr_mix_amt = nnx.Param(jnp.ones(config.hidden_size, dtype=config.lora_dtype) * 0.5)
        self.removal_key_multiplier = nnx.Param(jnp.zeros(config.hidden_size, dtype=config.lora_dtype))
        self.bonus_multiplier = nnx.Param(jnp.ones(config.hidden_size, dtype=config.lora_dtype))

        self.group_norm = nnx.GroupNorm(
            config.hidden_size, self.num_head, 
            epsilon=self.num_head*1e-5, rngs=rngs, 
            dtype=config.norm_dtype, 
            param_dtype=config.norm_dtype
        )

        self.rotary_emb = RotaryEmbedding(config)
        self.use_rope = config.use_rope

    def __call__(self, x: ArrayLike, x_sc: ArrayLike, v_prime_0: ArrayLike, t_positions: ArrayLike, state: PratchyaState):
        B, T, C, = x.shape
        N, H = self.num_head, self.head_dim
        layer_idx = state.layer_idx

        tm_state = state.tm_state[layer_idx]
        tm_state_sc = state.tm_state_sc[layer_idx]

        x_shifted = jnp.concat([tm_state, x[:, :-1, :]], axis=1)
        x_shifted_sc = jnp.concat([tm_state_sc, x_sc[:, :-1, :]], axis=1)
        
        tm_state = x[:, -1:, :]
        tm_state_sc = x_sc[:, -1:, :]

        _x = x.astype(jnp.bfloat16) * x_sc
        _x_shifted = x_shifted.astype(jnp.bfloat16) * x_shifted_sc

        x_mu = _x + jnp.einsum('mC,BTC->mBTC', self.mu.get_value().astype(_x.dtype), _x_shifted - _x, preferred_element_type=_x.dtype)

        r =         self.w_receptance(x_mu[0])
        d =         self.decay_lora(x_mu[1])
        k =         self.w_key(x_mu[2])
        v_prime =   self.w_value(x_mu[3])
        gate =      self.gate_lora(x_mu[5])
        iclr =      jax.nn.sigmoid(self.iclr_lora(x_mu[4]))

        if self.use_rope:
            k = k.reshape(B, T, -1, H)
            r = r.reshape(B, T, -1, H)

            k = self.rotary_emb(k, t_positions)
            r = self.rotary_emb(r, t_positions)

            k = k.reshape(B, T, -1)
            r = r.reshape(B, T, -1)

        if self.first_layer:
            v = v_prime_0 = v_prime
        else:
            value_residual_gate = jax.nn.sigmoid(self.nu_lora(x_mu[3]))
            v = lerp(v_prime, v_prime_0, value_residual_gate)

        decay = jnp.exp(-jnp.exp(-0.5) * jax.nn.sigmoid(d.astype(jnp.float32)))
        removal_k = k * jnp.astype(self.removal_key_multiplier, k.dtype)
        removal_k = normalized(removal_k.reshape(B, T, N, -1), axis=-1).reshape(B, T, C)
        replacement_k = lerp(k, k * iclr, self.iclr_mix_amt)

        general_scan_shape = (B, N, H, 1)
        @nnx.scan(in_axes=(nnx.Carry, 1, 1, 1, 1, 1, 1), out_axes=(nnx.Carry, 0))
        def wkv_state_scan(wkv_state, decay_t, iclr_t, removal_k_t, replacement_k_t, v_t, r_t):

            decay_t =           decay_t.reshape(*general_scan_shape)
            iclr_t =            iclr_t.reshape(*general_scan_shape)
            removal_k_t =       removal_k_t.reshape(*general_scan_shape)
            replacement_k_t =   replacement_k_t.reshape(*general_scan_shape)
            v_t =               v_t.reshape(*general_scan_shape)
            r_t =               r_t.reshape(*general_scan_shape)

            wkv_state = wkv_state * decay_t.mT - wkv_state @ removal_k_t @ (iclr_t * removal_k_t).mT
            wkv_state = wkv_state + v_t @ replacement_k_t.mT
            y = wkv_state @ r_t

            return wkv_state, y.reshape(B, C)

        wkv_state, out = wkv_state_scan(
            state.wkv_state[layer_idx],
            decay, iclr, removal_k,
            replacement_k, v, r
        )
        
        out = out.transpose(1, 0, 2)
        out = self.group_norm(out.reshape(B * T, -1)).reshape(B, T, -1)

        bonus = jnp.sum(r * k * self.bonus_multiplier.get_value().astype(k.dtype), axis=-1, keepdims=True) * v
        bonus = bonus.reshape(B, T, C)
        out = (out + bonus) * gate

        out, out_sc = block_quantize(out)
        b = out.shape[-1] // out_sc.shape[-1]
        out = out.reshape(B, T, -1, b)

        w_o, w_o_sc = block_quantize(self.w_output.kernel.get_value().T)
        d = w_o.shape[0]
        w_o = w_o.T.reshape(-1, b, d)

        out = jnp.einsum('atnb,nbd->atnd', out, w_o, preferred_element_type=jnp.bfloat16)
        out = out * out_sc[..., None] * w_o_sc.T[None, None, ...]
        out = jnp.sum(out, axis=2)

        wkv_state = state.wkv_state.at[layer_idx].set(wkv_state)
        tm_state = state.tm_state.at[layer_idx].set(tm_state)
        tm_state_sc = state.tm_state_sc.at[layer_idx].set(tm_state_sc)
        state = state.replace(
            tm_state = tm_state,
            tm_state_sc = tm_state_sc,
            wkv_state = wkv_state
        )

        return out, v, state


class ChannelMix(nnx.Module):

    def __init__(self, config: PratchyaConfig, *, rngs: nnx.Rngs):
        self.w_k = nnx.Linear(
            config.hidden_size, config.intermediate_size, 
            use_bias=False, rngs=rngs, 
            dtype=config.gemm_dtype, 
            param_dtype=config.param_dtype,
            preferred_element_type=jnp.bfloat16
        )
        self.w_v = nnx.Linear(
            config.intermediate_size, config.hidden_size, 
            use_bias=False, rngs=rngs, 
            dtype=config.gemm_dtype, 
            param_dtype=config.param_dtype,
            preferred_element_type=jnp.bfloat16
        )
        self.mu_x = nnx.Param(jnp.ones(config.hidden_size, dtype=config.mu_dtype) * 0.5)

        self.hidden_size = config.hidden_size
        self.inter_size = config.intermediate_size        

    def __call__(self, x: ArrayLike, x_sc: ArrayLike, state: PratchyaState):
        
        B, T, C = x.shape
        I = self.inter_size
        b = C // x_sc.shape[-1]
        n = C // b
        N = I // b
        layer_idx = state.layer_idx

        cm_state = state.cm_state[layer_idx]
        cm_state_sc = state.cm_state_sc[layer_idx]

        x_shifted = jnp.concat([cm_state, x[:, :-1, :]], axis=1)
        x_shifted_sc = jnp.concat([cm_state_sc, x_sc[:, :-1, :]], axis=1)

        cm_state = x[:, -1:, :]
        cm_state_sc = x_sc[:, -1:, :]

        _x = dequantize_cast(x, x_sc, dtype=jnp.bfloat16)
        _x_shifted = dequantize_cast(x_shifted, x_shifted_sc, dtype=jnp.bfloat16)

        x_k = lerp(_x, _x_shifted, self.mu_x.get_value().astype(_x.dtype))
        x_k, x_k_sc = block_quantize(x_k)
        x_k = x_k.reshape(B, T, n, b)

        w_k, w_k_sc = block_quantize(self.w_k.kernel.get_value().T)
        w_k = w_k.T.reshape(n, b, I)

        k = jnp.einsum('atnb,nbi->atni', x_k, w_k, preferred_element_type=jnp.bfloat16)
        k = k * x_k_sc[..., None] * w_k_sc.T[None, None, ...]
        k = jnp.sum(k, axis=2)
        k = jnp.pow(jax.nn.relu(k), 2)
        k, k_sc = block_quantize(k)
        k = k.reshape(B, T, N, b)

        w_v, w_v_sc = block_quantize(self.w_v.kernel.get_value().T)
        w_v = w_v.T.reshape(N, b, C)

        v = jnp.einsum('atnb,nbc->atnc', k, w_v, preferred_element_type=jnp.bfloat16)
        v = v * k_sc[..., None] * w_v_sc.T[None, None, ...]
        v = jnp.sum(v, axis=2)

        cm_state = state.cm_state.at[layer_idx].set(cm_state)
        cm_state_sc = state.cm_state_sc.at[layer_idx].set(cm_state_sc)

        state = state.replace(
            cm_state = cm_state,
            cm_state_sc = cm_state_sc
        )

        return v, state


class RotaryEmbedding(nnx.Module):
    def __init__(self, config: PratchyaConfig):
        self.head_dim = config.head_dim
        base = config.rope_theta
        self.inv_freq = 1.0 / (base ** (jnp.arange(0, self.head_dim, 2, dtype=jnp.float32) / self.head_dim))

    def rotate_half(self, x: ArrayLike):
        d = x.shape[-1]
        x1 = x[..., :d//2]
        x2 = x[..., d//2:]
        return jnp.concatenate([-x2, x1], axis=-1)

    def __call__(self, x: ArrayLike, t: Union[int, ArrayLike]):
        t_array = jnp.atleast_1d(t).astype(jnp.float32)
        freqs = jnp.outer(t_array, self.inv_freq) # Shape: (Batch or 1, head_dim // 2)
        
        emb = jnp.concatenate([freqs, freqs], axis=-1) # Shape: (Batch or 1, head_dim)
        
        emb = emb[:, None, :] 
        cos = jnp.cos(emb)
        sin = jnp.sin(emb)
        
        return (x * cos) + (self.rotate_half(x) * sin)


class RWKVBlock(nnx.Module):
    def __init__(self, config: PratchyaConfig, *, rngs: nnx.Rngs, first_layer: Optional[bool] = None):

        first_layer = first_layer if first_layer else False

        self.timemix = TimeMix(config, rngs=rngs, first_layer=first_layer)
        self.channelmix = ChannelMix(config, rngs=rngs)
        self.pre_timemix_rmsnorm = RMSNorm(
            config.hidden_size, 
            epsilon=config.rmsnorm_epsilon
        )
        self.pre_channelmix_rmsnorm = RMSNorm(
            config.hidden_size, 
            epsilon=config.rmsnorm_epsilon, 
        )

    def __call__(
        self, x: ArrayLike, 
        x_sc: ArrayLike, 
        v_0: ArrayLike, 
        t_positions: ArrayLike,
        state: PratchyaState
    ):

        x, x_sc = self.pre_timemix_rmsnorm(x, x_sc)
        dx, v_0, state = self.timemix(x, x_sc, v_0, t_positions, state)
        x = dequantize_cast(x, x_sc, dtype=dx.dtype) + dx
        x, x_sc = block_quantize(x)
        
        x, x_sc = self.pre_channelmix_rmsnorm(x, x_sc)
        dx, state = self.channelmix(x, x_sc, state)
        x = dequantize_cast(x, x_sc, dtype=dx.dtype) + dx
        x, x_sc = block_quantize(x)

        state = state.replace(layer_idx=state.layer_idx + 1)
        
        return x, x_sc, v_0, state


class PratchyaModel(nnx.Module):

    """Implemented modified: `RWKV-7 "Goose" with Expressive Dynamic State Evolution` paper"""
    
    def __init__(self, config: PratchyaConfig, *, rngs: Optional[nnx.Rngs] = None):
        
        self.embed_tokens = ScaledWordEmbedding(
            config.vocab_size, 
            config.hidden_size, rngs=rngs, 
            dtype=config.language_dtype, 
            param_dtype=config.language_dtype
        )

        assert config.n_layers > 1, \
            "n_layer can't be less than 1"
        
        self.init_layer = RWKVBlock(config, rngs=rngs, first_layer=True)
        self.blocks = nnx.data(None)
        if config.n_layers - 1 > 0:
            @nnx.split_rngs(splits=config.n_layers - 1)
            @nnx.vmap(in_axes=(0,), out_axes=0)
            def create_block(rngs: nnx.Rngs):
                return RWKVBlock(config, rngs=rngs)
            
            self.blocks = create_block(rngs)

        self.pre_rmsnorm = RMSNorm(
            config.hidden_size, 
            epsilon=config.rmsnorm_epsilon,
        )
        self.final_rmsnorm = RMSNorm(
            config.hidden_size, 
            epsilon=config.rmsnorm_epsilon,
        )

        self.n_layers = config.n_layers
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.n_head = self.hidden_size // self.head_dim
        self.use_rope = config.use_rope

    def __call__(self, input_ids: ArrayLike, state: PratchyaState):

        B, T = input_ids.shape
        C = self.hidden_size

        if state is None:
            tm_state = jnp.zeros((self.n_layers, B, 1, C))
            tm_state, tm_state_sc = block_quantize(tm_state)
            cm_state = jnp.zeros((self.n_layers, B, 1, C))
            cm_state, cm_state_sc = block_quantize(cm_state)
            wkv_state = jnp.zeros((self.n_layers, B, self.n_head, self.head_dim, self.head_dim), dtype=jnp.float32)
            step = jnp.array(0, dtype=jnp.int32)

            state = PratchyaState(
                tm_state=tm_state, 
                tm_state_sc=tm_state_sc,
                cm_state=cm_state, 
                cm_state_sc=cm_state_sc,
                wkv_state=wkv_state,
                step=step
            )

        if self.use_rope:
            t_positions = state.step + jnp.arange(T, dtype=jnp.int32)
        else:
            t_positions = None

        x = self.embed_tokens(input_ids)
        x, x_sc = block_quantize(x)
        x, x_sc = self.pre_rmsnorm(x, x_sc)
        
        x, x_sc, v, state = self.init_layer(x, x_sc, None, t_positions, state)

        @nnx.scan(in_axes=(nnx.Carry, None, 0), out_axes=nnx.Carry)
        def forward(carry, t_positions, layer):
            x, x_sc, v, state = carry
            x, x_sc, v, state = layer(x, x_sc, v, t_positions, state)
             
            return (x, x_sc, v, state)

        x, x_sc, _, state = forward((x, x_sc, v, state), t_positions, self.blocks)
        x, x_sc = self.final_rmsnorm(x, x_sc)

        state = state.replace(layer_idx=0, step=state.step + T)

        return x, x_sc, state

class PratchyaCausalLM(nnx.Module):
    def __init__(self, config: PratchyaConfig, *, rngs: Optional[nnx.Rngs] = None):
        
        if rngs is None:
            rngs = nnx.Rngs(42)

        self.model = PratchyaModel(config, rngs=rngs)
        self.lm_head = nnx.Linear(
            config.hidden_size, config.vocab_size, 
            use_bias=False, rngs=rngs, 
            dtype=config.language_dtype, 
            param_dtype=config.language_dtype
        )

    def __call__(
        self, input_ids: ArrayLike, 
        label: Optional[ArrayLike] = None,
        *, state: Optional[PratchyaState] = None
    ):
        
        x, x_sc, state = self.model(input_ids, state)

        x = dequantize_cast(x, x_sc, dtype=self.lm_head.dtype)
        logits = self.lm_head(x)

        return PratchyaOutput(
            logits=logits,
            loss=None,
            state=state
        )