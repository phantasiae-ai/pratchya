from pratchya._jax_impl._fn import fwd_fn
from pratchya._jax_impl._init_module import init_fn, init_state
import jax, jax.numpy as jnp

import jax.profiler

from pratchya.preset import PratchyaDummyConfig, Pratchya500M


config = Pratchya500M

params = init_fn(jax.random.key(42), config)

inp = jnp.arange(5).reshape(1, -1)
state = init_state(inp, config)

def forward(inp, params):
    return fwd_fn(inp, params, state=state, config=config)

jit_forward = jax.jit(forward)

print(jit_forward(inp, params))

jax.profiler.save_device_memory_profile("memory.prof")
