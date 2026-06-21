

import jax, jax.numpy as jnp
from flax import nnx
from pratchya._nnx_impl._optimizer import miulion_optimizer, MiulionHyperParams, MiulionScheduler
import optax

from pratchya.preset import PratchyaDummyConfig
from pratchya import PratchyaCausalLM

import jax.profiler

# orig_model = PratchyaCausalLM(PratchyaDummyConfig)
# orig_param = nnx.state(orig_model, nnx.Param).flat_state()

model = PratchyaCausalLM(PratchyaDummyConfig)

# -------

hyperparams = MiulionHyperParams(weight_decay=0.0)

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
optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

@nnx.jit(donate_argnums=(0, 1))
def train_step(model, optimizer, batch):
    def loss_fn(model):
        output = model(batch['input_ids'], batch['input_ids'])
        return output.loss

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    
    return loss

inp = jnp.arange(30).reshape(1, -1)
inp = {'input_ids': inp}


for _ in range(10):

    loss = train_step(model, optimizer, inp)

    print(f"loss: {loss}")

# --------

# param = nnx.state(model, nnx.Param).flat_state()

# n_params = len(param)
# changes = 0
# unchanges = []

# print(f"\nunchange params (disable wd): ")
# for i in range(n_params):
#     orig = orig_param[i][-1].get_value()
#     p = param[i][-1].get_value()
#     if (orig == p).all():
#         print(param[i][0])
#         continue

#     changes += 1

# print(f"\nchanges: {changes}/{n_params}")