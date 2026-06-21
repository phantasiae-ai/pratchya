

from pratchya._kernel._fp8 import ArrayFP8

import jax, jax.numpy as jnp

rand_1 = jax.random.normal(jax.random.key(42), (128, 128, 128), dtype=jnp.float32)


grid = (1, 8) # n1/n_grid^2

arr_1 = ArrayFP8(rand_1, grid)

print(type(arr_1[10:12, :20]))
print(type(arr_1))
print(type(rand_1))

print(arr_1[:, -1:, :])
print(arr_1)
print(rand_1)