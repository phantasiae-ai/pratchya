from flax.struct import dataclass, field
from flax import struct
from jax.typing import DTypeLike, ArrayLike
import jax.numpy as jnp


@dataclass
class PratchyaConfig:
    vocab_size: int = 80_300
    hidden_size: int = 128
    intermediate_size: int = 128*4
    lora_rank: int = 32
    n_head: int = 4
    head_dim: int = 32
    rmsnorm_epsilon: float = 1e-6
    n_layers: int = 2
    language_dtype: str = "bfloat16"
    param_dtype: str = "bfloat16"
    mu_dtype: str = "float32"
    norm_dtype: str = "float32"
    lora_dtype: str = "float32"
    gemm_dtype: str = "float8_e4m3fn"
    dtype: str = "float8_e4m3fn"

    # RoPE
    use_rope: bool = True
    rope_theta: float = 1e+4

@dataclass
class PratchyaState:
    tm_state: ArrayLike
    tm_state_sc: ArrayLike
    cm_state: ArrayLike
    cm_state_sc: ArrayLike
    wkv_state: ArrayLike
    step: ArrayLike | None
    layer_idx: int = 0

@dataclass
class PratchyaOutput:
    logits: DTypeLike | None
    loss: float | None
    state: PratchyaState | None


