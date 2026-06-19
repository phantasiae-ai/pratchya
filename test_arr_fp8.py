

from pratchya._kernel._fp8 import ArrayFP8

import jax, jax.numpy as jnp

rand = jax.random.normal(jax.random.key(42), (128, 128), dtype=jnp.float32)


grid = (4, 4) # n1/n_grid^2

arr = ArrayFP8(rand, grid)

print((arr @ arr).shape)
