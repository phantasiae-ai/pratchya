

from pratchya.preset import PratchyaDummyConfig, Pratchya500M
from pratchya._nnx_impl._model import NQPratchyaCausalLM, PratchyaCausalLM

import jax, jax.numpy as jnp
from flax import nnx
import optax

model = NQPratchyaCausalLM(PratchyaDummyConfig, rngs=nnx.Rngs(0))
tx = optax.adamw(1e-3, mu_dtype=jnp.bfloat16)
optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

@nnx.jit(donate_argnums=(0, 1))
def train_step(model: nnx.Module, optimizer: nnx.Optimizer, batch):
    def loss_fn(model):
        output = model(batch['input_ids'], batch['input_ids'])
        return output.loss
        
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    
    return loss

inp = jnp.arange(10).reshape(1, -1)
inp = {'input_ids': inp}

for _ in range(10):
    loss = train_step(model, optimizer, inp)
    print(f"loss: {loss}")