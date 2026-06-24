

from pratchya.preset import PratchyaDummyConfig, Pratchya500M
from pratchya._nnx_impl._model import NQPratchyaCausalLM, PratchyaCausalLM
from pratchya._nnx_impl._optimizer import miulion_optimizer_nq, MiulionHyperParams, MiulionScheduler

import jax, jax.numpy as jnp
from flax import nnx
import optax


hyperparams = MiulionHyperParams(
    lion_lr=4e-3,  
    muon_lr=5e-2,
)

cosine_muon = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=hyperparams.muon_lr,
    warmup_steps=100,
    decay_steps=hyperparams.total_steps,
    end_value=0.0
)

cosine_lion = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=hyperparams.lion_lr,
    warmup_steps=100,
    decay_steps=hyperparams.total_steps,
    end_value=0.0
)

schedule = MiulionScheduler(
    muon_schedule=cosine_muon,
    lion_schedule=cosine_lion,
    weight_decay=hyperparams.weight_decay
)

tx = miulion_optimizer_nq(hyperparams, schedule)

model = NQPratchyaCausalLM(PratchyaDummyConfig)
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