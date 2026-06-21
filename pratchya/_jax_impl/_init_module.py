

from jax.typing import ArrayLike
import jax, jax.numpy as jnp

from .._config import PratchyaConfig
from .._kernel._fp8 import ArrayFP8
from ._key import Key


def init_linear(key, in_features, out_features, dtype, bias=False, initializer_fn=None, quantize=True, block_grid=(4, 4)):
    if initializer_fn is None:
        initializer_fn = jax.nn.initializers.lecun_uniform()

    w = initializer_fn(key, (in_features, out_features), dtype=dtype)
    if quantize:
        w = ArrayFP8(w, block_grid)

    if bias:
        b = initializer_fn(key, (1, out_features), dtype=dtype)
        if quantize:
            b = ArrayFP8(b, (1, block_grid[-1]))

        return {
            'w': w,
            'b': b
        }

    return {
        'w': w,
    }


def init_fn(key: int, config: PratchyaConfig):

    def lora_ffn(hidden_size, lora_rank, dtype):
        return dict(
            lin1=init_linear(key(), hidden_size, lora_rank, dtype, block_grid=(1, config.block_size)),
            lin2=init_linear(key(), lora_rank, hidden_size, dtype, block_grid=(1, config.block_size)),
        )
    
    def tm(init=False):
        layer = dict(
            mu=init_linear(key(), 6, config.hidden_size, jnp.float32, quantize=False),
            w_key=init_linear(key(), config.hidden_size, config.hidden_size, jnp.float32, block_grid=(config.block_size, config.block_size)),
            w_value=init_linear(key(), config.hidden_size, config.hidden_size, jnp.float32, block_grid=(config.block_size, config.block_size)),
            w_output=init_linear(key(), config.hidden_size, config.hidden_size, jnp.float32, block_grid=(config.block_size, config.block_size)),
            w_receptance=init_linear(key(), config.hidden_size, config.hidden_size, jnp.float32, block_grid=(config.block_size, config.block_size)),

            gate_lora=lora_ffn(config.hidden_size, config.lora_rank, jnp.float32),
            decay_lora=lora_ffn(config.hidden_size, config.lora_rank, jnp.float32),
            iclr_lora=lora_ffn(config.hidden_size, config.lora_rank, jnp.float32),

            iclr_mix_amt=init_linear(key(), 1, config.hidden_size, jnp.float32, quantize=False),
            removal_key_multiplier=init_linear(key(), 1, config.hidden_size, jnp.float32, quantize=False),
            bonus_multiplier=init_linear(key(), 1, config.hidden_size, jnp.float32, quantize=False),

            group_norm=init_linear(key(), 1, config.hidden_size, jnp.float32, bias=True, quantize=False),
        )

        if not init:
            layer['nu_lora'] = lora_ffn(config.hidden_size, config.lora_rank, jnp.float32)

        return layer
    
    def cm():
        return dict(
            w_k=init_linear(key(), config.hidden_size, config.intermediate_size, jnp.float32, block_grid=(config.block_size, config.block_size)),
            w_v=init_linear(key(), config.intermediate_size, config.hidden_size, jnp.float32, block_grid=(config.block_size, config.block_size)),
            mu_x=init_linear(key(), 1, config.hidden_size, jnp.float32, quantize=False),
        )

    def rwkv_block(xs):
        layer = dict(
            pre_tm_rmsnorm=init_linear(key(), 1, config.hidden_size, jnp.float32, quantize=False),
            tm=tm(),
            pre_cm_rmsnorm=init_linear(key(), 1, config.hidden_size, jnp.float32, quantize=False),
            cm=cm(),
        )
        return layer
    
    key = Key(key)

    params = dict(
        embed_tokens=init_linear(key(), config.vocab_size, config.hidden_size, config.language_dtype, quantize=False),
        pre_rmsnorm=init_linear(key(), 1, config.hidden_size, jnp.float32, quantize=False),
        rwkv_init_block=dict(
            pre_tm_rmsnorm=init_linear(key(), 1, config.hidden_size, jnp.float32, quantize=False),
            tm=tm(init=True),
            pre_cm_rmsnorm=init_linear(key(), 1, config.hidden_size, jnp.float32, quantize=False),
            cm=cm(),
        ),
        rwkv_block=jax.vmap(rwkv_block)(jnp.arange(config.n_layers - 1)),
        final_rmsnorm=init_linear(key(), 1, config.hidden_size, jnp.float32, quantize=False),
        lm_head=init_linear(key(), config.hidden_size, config.vocab_size, config.language_dtype, quantize=False),
    )

    return params


def init_state(x: ArrayLike, config: PratchyaConfig, quantize=True):
    B, T = x.shape
    C = config.hidden_size
    N, H = C // config.head_dim, config.head_dim
    L = config.n_layers
    tm_state = jnp.zeros((L, B, 1, C), dtype=jnp.float32)
    cm_state = jnp.zeros((L, B, 1, C), dtype=jnp.float32)
    wkv_state = jnp.zeros((L, B, N, H, H), dtype=jnp.float32)
    if quantize:
        tm_state = ArrayFP8(tm_state, block_grid=(1, config.block_size))
        cm_state = ArrayFP8(cm_state, block_grid=(1, config.block_size))

    # rope
    inv_freq = 1.0 / (config.rope_theta ** (jnp.arange(0, H, 2, dtype=jnp.float32) / H))

    return dict(
        tm_state=tm_state,
        cm_state=cm_state,
        wkv_state=wkv_state,
        layer_idx=0,
        inv_freq=inv_freq,
        step=0
    )


def _count_params_and_memory(params: dict):
    count = 0
    memory = 0
    for k, v in params.items():
        if isinstance(v, dict):
            c, m = _count_params_and_memory(v)
            count = count + c
            memory = memory + m
            
        else:
            count = count + v.size
            memory = memory + v.size*v.dtype.itemsize

    return count, memory

def _display_params_suffix(n: int):
    if n > 1e+12:
        return f'{n/1e12:.2f}T'
    
    if n > 1e9:
        return f'{n/1e9:.2f}B'
    
    if n > 1e6:
        return f'{n/1e6:.2f}M'
    
    return f'{n}'

def _display_memory_suffix(n: int):
    if n > 1e+12:
        return f'{n/1e12:.2f}TB'
    
    if n > 1e9:
        return f'{n/1e9:.2f}GB'
    
    if n > 1e6:
        return f'{n/1e6:.2f}MB'
    
    if n > 1e3:
        return f'{n/1e3:.2f}kB'
    
    return f'{n}B'

def pnm_usage(params: dict):
    c, m = _count_params_and_memory(params)
    print(f'count params: {_display_params_suffix(c)}')
    print(f'memory usage: {_display_memory_suffix(m)}')