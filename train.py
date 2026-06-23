

from pratchya._nnx_impl._optimizer import miulion_optimizer, MiulionHyperParams, MiulionScheduler
from pratchya.preset import PratchyaDummyConfig, Pratchya500M
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
model = PratchyaCausalLM(PratchyaDummyConfig, rngs=nnx.Rngs(0))
param_arrays = nnx.state(model, nnx.Param)
opt_state = tx.init(param_arrays)

@nnx.jit
def train_step(model: nnx.Module, opt_state, batch):
    def loss_fn(model):
        output = model(batch['input_ids'], batch['input_ids'])
        return output.loss
        
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    
    param_arrays = nnx.state(model, nnx.Param)
    grad_arrays = nnx.state(grads, nnx.Param)
    
    updates, new_opt_state = tx.update(grad_arrays, opt_state, param_arrays)

    from pratchya._qualia._qarr import QArrayImpl
    new_params = jax.tree_util.tree_map(
        lambda p, u: p + u,
        param_arrays, updates,
        is_leaf=lambda x: isinstance(x, QArrayImpl)
    )
    nnx.update(model, new_params)
    
    return loss, new_opt_state

inp = jnp.arange(10).reshape(1, -1)
inp = {'input_ids': inp}

for _ in range(10):
    loss, opt_state = train_step(model, opt_state, inp)
    print(f"loss: {loss}")