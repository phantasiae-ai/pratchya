

from pratchya.optimizer import miulion_optimizer, MiulionHyperParams, MiulionScheduler
from pratchya.preset import PratchyaDummyConfig
from pratchya import PratchyaCausalLM

import jax, jax.numpy as jnp
from flax import nnx
import optax

hyperparams = MiulionHyperParams()

cosine_muon = optax.cosine_decay_schedule(
    init_value=hyperparams.muon_lr,
    decay_steps=hyperparams.total_steps,
    alpha=0.0
)

cosine_lion = optax.cosine_decay_schedule(
    init_value=hyperparams.lion_lr,
    decay_steps=hyperparams.total_steps,
    alpha=0.0
)
## NOT USED WARMUP STEP YET!
schedule = MiulionScheduler(
    muon_schedule=cosine_muon,
    lion_schedule=cosine_lion,
    weight_decay=hyperparams.weight_decay
)

tx = miulion_optimizer(hyperparams, schedule)
model = PratchyaCausalLM(PratchyaDummyConfig)
optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

@nnx.jit
def train_step(model, optimizer, batch):
    def loss_fn(model):
        output = model(batch['input_ids'], batch['input_ids'])
        return output.loss

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    
    return loss

inp = jnp.arange(30).reshape(1, -1)
inp = {'input_ids': inp}

# WHAT A TRAINING SCRIPT 🗣🗣
for _ in range(20):

    loss = train_step(model, optimizer, inp)

    print(f"loss: {loss}")