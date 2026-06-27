from flax import nnx
import jax, jax.numpy as jnp
from typing import Optional

from .._config import PratchyaConfig, PratchyaState, PratchyaOutput, init_state
from ._module import EmbeddingScaled, RWKVBlock, RMSNorm, Linear, NQEmbeddingScaled, NQRWKVBlock, NQRMSNorm, NQLinear, NQRWKVState, RWKVState
from ._utils import compute_loss


class PratchyaModel(nnx.Module):

    """Implemented modified: `RWKV-7 "Goose" with Expressive Dynamic State Evolution` paper"""
    
    def __init__(self, config: PratchyaConfig, *, rngs: Optional[nnx.Rngs] = None):
        self.embed_tokens = EmbeddingScaled(
            config.vocab_size, config.hidden_size, 
            (1, config.blksize), rngs=rngs, 
            dtype=config.language_dtype
        )

        self.init_block = RWKVBlock(config, rngs=rngs, layer_idx=0)
        @nnx.split_rngs(splits=config.n_layers - 1)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def vmap_block(rngs: nnx.Rngs):
            return RWKVBlock(config, rngs=rngs)
        
        self.blocks = vmap_block(rngs)

        self.pre_rmsnorm = RMSNorm(
            config.hidden_size, (1, config.blksize),
            epsilon=config.rmsnorm_epsilon,
        )
        self.final_rmsnorm = RMSNorm(
            config.hidden_size, (1, config.blksize),
            epsilon=config.rmsnorm_epsilon,
        )

        self.init_state = RWKVState(config, rngs=rngs)

        self.config = config

    def __call__(self, input_ids: jax.Array, state: PratchyaState):

        B, T = input_ids.shape

        if state is None:
            state = init_state(self.config, B)

        t_positions = state.step + jnp.arange(T, dtype=jnp.int32)

        x = nnx.remat(self.embed_tokens)(input_ids)
        x = nnx.remat(self.pre_rmsnorm)(x)

        x, v, state = nnx.remat(self.init_block)(x, None, t_positions, state)
        
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def scan_block(carry, layer):
            x, v, state = carry
            x, v, state = nnx.remat(layer)(x, v, t_positions, state)
            return (x, v, state)
        
        x, _, state = scan_block((x, v, state), self.blocks)

        x = nnx.remat(self.final_rmsnorm)(x)

        state = state.replace(step=state.step + T)

        return x, state

class PratchyaCausalLM(nnx.Module):
    def __init__(self, config: PratchyaConfig, *, rngs: Optional[nnx.Rngs] = None):
        
        if rngs is None:
            rngs = nnx.Rngs(42)

        self.model = PratchyaModel(config, rngs=rngs)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, (config.blksize, 1), rngs=rngs, dtype=config.language_dtype)

    def __call__(
        self, input_ids: jax.Array, 
        label: Optional[jax.Array] = None,
        *, state: Optional[PratchyaState] = None
    ):
        
        x, state = self.model(input_ids, state)
        logits = nnx.remat(self.lm_head)(x)

        loss = jnp.array(0.0)
        if label is not None:
            loss = compute_loss(logits[:, :-1, :], label[:, 1:])

        return PratchyaOutput(
            logits=logits,
            loss=loss,
            state=state
        )
    

class NQPratchyaModel(nnx.Module):

    """Implemented modified: `RWKV-7 "Goose" with Expressive Dynamic State Evolution` paper"""
    
    def __init__(self, config: PratchyaConfig, *, rngs: Optional[nnx.Rngs] = None):
        self.embed_tokens = NQEmbeddingScaled(
            config.vocab_size, config.hidden_size, 
            rngs=rngs, dtype=config.language_dtype
        )

        self.init_block = NQRWKVBlock(config, rngs=rngs, layer_idx=0)
        
        # CRITICAL MEMORY FIX: Do NOT use nnx.vmap!
        # vmap stacks all parameters into a single massive 14GB array. 
        # XLA's partitioner tries to hoist the AllGather of this array, causing a 54GB OOM!
        # Using a list of independent blocks forces XLA to AllGather layer-by-layer (450MB peak)!
        self.blocks = nnx.List([NQRWKVBlock(config, rngs=rngs) for _ in range(config.n_layers - 1)])

        self.pre_rmsnorm = NQRMSNorm(
            config.hidden_size,
            epsilon=config.rmsnorm_epsilon,
        )
        self.final_rmsnorm = NQRMSNorm(
            config.hidden_size,
            epsilon=config.rmsnorm_epsilon,
        )
        self.init_state = NQRWKVState(config, rngs=rngs)

        self.config = config

    def __call__(self, input_ids: jax.Array, state: PratchyaState):

        B, T = input_ids.shape

        x = nnx.remat(self.embed_tokens)(input_ids)

        if state is None:
            state = self.init_state(x)

        t_positions = state.step + jnp.arange(T, dtype=jnp.int32)

        x = nnx.remat(self.pre_rmsnorm)(x)

        # Set layer_idx to 0 for init block
        state = state.replace(layer_idx=0)
        x, v, state = nnx.remat(self.init_block)(x, None, t_positions, state)
        
        # Unroll the loop to enforce layer-by-layer rematerialization
        for i, block in enumerate(self.blocks):
            state = state.replace(layer_idx=i + 1)
            x, v, state = nnx.remat(block)(x, v, t_positions, state)

        x = nnx.remat(self.final_rmsnorm)(x)

        state = state.replace(step=state.step + T)

        return x, state

class NQPratchyaCausalLM(nnx.Module):
    def __init__(self, config: PratchyaConfig, *, rngs: Optional[nnx.Rngs] = None):
        
        if rngs is None:
            rngs = nnx.Rngs(42)

        self.model = NQPratchyaModel(config, rngs=rngs)
        self.lm_head = NQLinear(config.hidden_size, config.vocab_size, rngs=rngs, dtype=config.language_dtype)

    def __call__(
        self, input_ids: jax.Array, 
        label: Optional[jax.Array] = None,
        *, state: Optional[PratchyaState] = None
    ):
        
        x, state = self.model(input_ids, state)

        loss = None
        logits = None
        if label is not None:
            loss = jnp.array(0., dtype=jnp.bfloat16)
            # B = x.shape[0]
            # for i in range(B):
            #     logit_i = self.lm_head(x[i:i+1, :, :])
            #     loss = loss + compute_loss(logit_i[:, :-1, :], label[:, 1:])

            # loss = loss / B
        
        else:
            logits = self.lm_head(x)

        state = state.replace(layer_idx=0)

        return PratchyaOutput(
            logits=logits,
            loss=loss,
            state=state
        )