from flax import nnx
import jax, jax.numpy as jnp
from typing import Optional

from .._config import PratchyaConfig, PratchyaState, PratchyaOutput, init_state
from ._module import EmbeddingScaled, RWKVBlock, RMSNorm, Linear
from ._utils import compute_loss


class PratchyaModel(nnx.Module):

    """Implemented modified: `RWKV-7 "Goose" with Expressive Dynamic State Evolution` paper"""
    
    def __init__(self, config: PratchyaConfig, *, rngs: Optional[nnx.Rngs] = None):
        self.embed_tokens = EmbeddingScaled(
            config.vocab_size, config.hidden_size, 
            (1, config.blksize), rngs=rngs, 
            dtype=config.language_dtype
        )

        self.blocks = nnx.List([RWKVBlock(config, rngs=rngs, layer_idx=i) for i in range(config.n_layers)])

        self.pre_rmsnorm = RMSNorm(
            config.hidden_size, (1, config.blksize),
            epsilon=config.rmsnorm_epsilon,
        )
        self.final_rmsnorm = RMSNorm(
            config.hidden_size, (1, config.blksize),
            epsilon=config.rmsnorm_epsilon,
        )

        self.config = config

    def __call__(self, input_ids: jax.Array, state: PratchyaState):

        B, T = input_ids.shape

        if state is None:
            state = init_state(self.config, B)

        t_positions = state.step + jnp.arange(T, dtype=jnp.int32)

        x = self.embed_tokens(input_ids)
        x = self.pre_rmsnorm(x)

        v = jnp.zeros_like(x, jnp.float32)
        for i, layer in enumerate(self.blocks):
            x, v, state = layer(x, v, t_positions, state)

        x = self.final_rmsnorm(x)

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
        logits = self.lm_head(x)

        loss = jnp.array(0.0)
        if label is not None:
            loss = compute_loss(logits[:, :-1, :], label[:, 1:])

        return PratchyaOutput(
            logits=logits,
            loss=loss,
            state=state
        )