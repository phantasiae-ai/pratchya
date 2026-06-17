

from ._config import PratchyaConfig

PratchyaDummyConfig = PratchyaConfig( # dummy config for test
    vocab_size = 120,
    hidden_size = 32,
    intermediate_size = 64,
    lora_rank = 8,
    head_dim = 8,
    n_layers = 4,
    block_size = 4
)

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
