

from pratchya._kernel._fp8 import ArrayFP8

import jax, jax.numpy as jnp

rand_1 = jax.random.normal(jax.random.key(42), (128, 128), dtype=jnp.float32)
rand_2 = jax.random.normal(jax.random.key(59), (128, 128), dtype=jnp.float32)


grid = (4, 4) # n1/n_grid^2

arr_1 = ArrayFP8(rand_1, (8, 8))
arr_2 = ArrayFP8(rand_2, (8, 8))

# print(arr_1)
# print(arr_2)

a = arr_1 @ arr_2 @ arr_1 @ arr_2 @ arr_1
# print(a)

# print(arr_1.x_fp8 @ arr_2.x_fp8)

r = rand_1 @ rand_2 @ rand_1 @ rand_2 @ rand_1
 
# print(r.shape)
# print(r)

a_deq = a.dequantize(jnp.float32)

print(jnp.sum(jnp.abs(a_deq - r) / r))

print(a_deq[0, :20])
print(r[0, :20])

