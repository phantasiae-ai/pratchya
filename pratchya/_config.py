from jax.typing import DTypeLike, ArrayLike
from dataclasses import dataclass
from ._kernel._fp8 import ArrayFP8

@dataclass(frozen=True)
class PratchyaConfig:
    vocab_size: int = 82369
    hidden_size: int = 128
    intermediate_size: int = 128*4
    lora_rank: int = 32
    n_head: int = 4
    head_dim: int = 32
    rmsnorm_epsilon: float = 1e-6
    n_layers: int = 2
    language_dtype: str = "float32"
    param_dtype: str = "float32"
    mu_dtype: str = "float32"
    norm_dtype: str = "float32"
    lora_dtype: str = "float32"
    gemm_dtype: str = "float32"
    dtype: str = "float32"
    block_size: int = 128

    # RoPE
    use_rope: bool = True # I think will drop this because will be used anyway ;P
    rope_theta: float = 1e+4

@dataclass(frozen=True)
class PratchyaState:
    tm_state: ArrayFP8
    cm_state: ArrayFP8
    wkv_state: ArrayLike
    step: ArrayLike | None
    layer_idx: int = 0

@dataclass(frozen=True)
class PratchyaOutput:
    logits: ArrayLike | None
    loss: float | None
    state: PratchyaState | None


