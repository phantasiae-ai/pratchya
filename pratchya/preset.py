

from ._config import PratchyaConfig

Pratchya500M = PratchyaConfig(
    hidden_size = 1280,
    intermediate_size = 5120,
    lora_rank = 128,
    head_dim = 64,
    n_layers = 16,
)

Pratchya1B = PratchyaConfig(
    hidden_size = 2048,
    intermediate_size = 8192,
    lora_rank = 128,
    head_dim = 128,
    n_layers = 16
)

Pratchya3B = PratchyaConfig(
    hidden_size = 3072,
    intermediate_size = 11392,
    lora_rank = 256,
    head_dim = 128,
    n_layers = 22,
)

Pratchya7B = PratchyaConfig(
    hidden_size = 4096,
    intermediate_size = 15104,
    lora_rank = 256,
    head_dim = 256,
    n_layers = 32,
)
