

from jax.typing import ArrayLike
import jax, jax.numpy as jnp

from .._config import PratchyaConfig


def init_linear(key, in_features, out_features, dtype, bias=False, initializer_fn=None):
    if initializer_fn is None:
        initializer_fn = jax.nn.initializers.lecun_uniform()

    w = initializer_fn(key, (in_features, out_features), dtype=dtype)
    if bias:
        b = initializer_fn(key, (1, out_features), dtype=dtype)
        return {
            'w': w,
            'b': b
        }

    return {
        'w': w,
    }


def init_fn(key, config: PratchyaConfig):

    def lora_ffn(k, hidden_size, lora_rank, dtype):
        k1, k2 = jax.random.split(k, 2)
        return dict(
            lin1=init_linear(k1, hidden_size, lora_rank, dtype),
            lin2=init_linear(k2, lora_rank, hidden_size, dtype),
        )
    
    def tm(k, config: PratchyaConfig):
        k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13 = jax.random.split(k, 13)
        layer = dict(
            mu=init_linear(k1, 6, config.hidden_size, jnp.bfloat16),
            w_key=init_linear(k2, config.hidden_size, config.hidden_size, jnp.bfloat16),
            w_value=init_linear(k3, config.hidden_size, config.hidden_size, jnp.bfloat16),
            w_output=init_linear(k4, config.hidden_size, config.hidden_size, jnp.bfloat16),
            w_receptance=init_linear(k5, config.hidden_size, config.hidden_size, jnp.bfloat16),

            gate_lora=lora_ffn(k6, config.hidden_size, config.lora_rank, jnp.bfloat16),
            nu_lora = lora_ffn(k7, config.hidden_size, config.lora_rank, jnp.bfloat16),
            decay_lora=lora_ffn(k8, config.hidden_size, config.lora_rank, jnp.bfloat16),
            iclr_lora=lora_ffn(k9, config.hidden_size, config.lora_rank, jnp.bfloat16),

            iclr_mix_amt=init_linear(k10, 1, config.hidden_size, jnp.bfloat16),
            removal_key_multiplier=init_linear(k11, 1, config.hidden_size, jnp.bfloat16),
            bonus_multiplier=init_linear(k12, 1, config.hidden_size, jnp.bfloat16),

            group_norm=init_linear(k13, 1, config.hidden_size, jnp.bfloat16, bias=True),
        )

        return layer
    
    def cm(k, config: PratchyaConfig):
        k1, k2, k3 = jax.random.split(k, 3)
        return dict(
            w_k=init_linear(k1, config.hidden_size, config.intermediate_size, jnp.bfloat16),
            w_v=init_linear(k2, config.intermediate_size, config.hidden_size, jnp.bfloat16),
            mu_x=init_linear(k3, 1, config.hidden_size, jnp.bfloat16),
        )

    def rwkv_block(k, xs):
        k1, k2, k3, k4, k = jax.random.split(k, 5)
        layer = dict(
            pre_tm_rmsnorm=init_linear(k1, 1, config.hidden_size, jnp.float32),
            tm=tm(k2, config),
            pre_cm_rmsnorm=init_linear(k3, 1, config.hidden_size, jnp.float32),
            cm=cm(k4, config),
        )
        return k, layer
    
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

    params = dict(
        embed_tokens=init_linear(k1, config.vocab_size, config.hidden_size, config.language_dtype),
        pre_rmsnorm=init_linear(k2, 1, config.hidden_size, jnp.float32),
        rwkv_init_block=dict(
            pre_tm_rmsnorm=init_linear(k1, 1, config.hidden_size, jnp.float32),
            tm=tm(k2, config),
            pre_cm_rmsnorm=init_linear(k3, 1, config.hidden_size, jnp.float32),
            cm=cm(k4, config),
        ),
        rwkv_block=jax.lax.scan(rwkv_block, k4, None, config.n_layers - 1)[-1],
        final_rmsnorm=init_linear(k5, 1, config.hidden_size, jnp.float32),
        lm_head=init_linear(k6, config.hidden_size, config.vocab_size, config.language_dtype),
    )

    return params


def init_state(x: ArrayLike, config: PratchyaConfig):
    B, T = x.shape
    C = config.hidden_size
    N, H = C // config.head_dim, config.head_dim
    L = config.n_layers
    tm_state = jnp.zeros((L, B, 1, C), dtype=jnp.bfloat16)
    cm_state = jnp.zeros((L, B, 1, C), dtype=jnp.bfloat16)
    wkv_state = jnp.zeros((L, B, N, H, H), dtype=jnp.float32)
    return dict(
        tm_state=tm_state,
        cm_state=cm_state,
        wkv_state=wkv_state,
        layer_idx=0
    )