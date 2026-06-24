from jax.typing import DTypeLike, ArrayLike
from flax.struct import dataclass
from ._qualia._qarr import QArrayImpl
import jax.numpy as jnp
from ._qualia import QArrayImpl


@dataclass
class PratchyaConfig:
    vocab_size: int = 82369
    hidden_size: int = 128
    intermediate_size: int = 128*4
    lora_rank: int = 32
    head_dim: int = 32
    rmsnorm_epsilon: float = 1e-6
    n_layers: int = 2
    language_dtype: str = "float32"
    mu_dtype: str = "bfloat16"
    norm_dtype: str = "float32"
    lora_dtype: str = "float32"
    dtype: str = "bfloat16"
    act_dtype: str = "bfloat16"
    wkv_dtype: str = "float32"
    blksize: int = 128

    rope_theta: float = 1e+4

@dataclass
class PratchyaState:
    tm_state: ArrayLike
    cm_state: ArrayLike
    wkv_state: ArrayLike
    step: ArrayLike | None
    layer_idx: int = 0

@dataclass
class PratchyaOutput:
    logits: ArrayLike | None
    loss: float | None
    state: PratchyaState | None


def init_state(config: PratchyaConfig, B=1):
    tm_state = jnp.zeros((config.n_layers, B, 1, config.hidden_size), jnp.float32)
    cm_state = jnp.zeros((config.n_layers, B, 1, config.hidden_size), jnp.float32)
    wkv_state = jnp.zeros((config.n_layers, B, config.hidden_size // config.head_dim, config.head_dim, config.head_dim), jnp.float32)   
    tgrid = (1, config.blksize)
    return PratchyaState(
        tm_state=tm_state,
        cm_state=cm_state,
        wkv_state=wkv_state,
        step=0
    )