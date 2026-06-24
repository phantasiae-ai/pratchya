

from pratchya._nnx_impl._optimizer import miulion_optimizer, MiulionHyperParams, MiulionScheduler
from pratchya.preset import PratchyaDummyConfig, Pratchya500M
from pratchya._nnx_impl._model import NQPratchyaCausalLM, PratchyaCausalLM

import jax, jax.numpy as jnp
from flax import nnx
import optax

hyperparams = MiulionHyperParams(
    lion_lr=1e-3,  
    muon_lr=1e-2,
    total_steps=1000
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

tx = miulion_optimizer(hyperparams, schedule)

model = PratchyaCausalLM(PratchyaDummyConfig)
param_arrays = nnx.state(model, nnx.Param)
opt_state = tx.init(param_arrays)

@nnx.jit(donate_argnums=(0, 1))
def train_step(model: nnx.Module, opt_state, batch):
    def loss_fn(model):
        output = model(batch['input_ids'], batch['input_ids'])
        return output.loss
        
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    
    param_arrays = nnx.state(model, nnx.Param)
    grad_arrays = nnx.state(grads, nnx.Param)
    
    updates, new_opt_state = tx.update(grad_arrays, opt_state, param_arrays)

    from pratchya._qualia._qarr import QArrayImpl

    def add_param(p, u):
        if isinstance(p, QArrayImpl):
            if p.get_value().ndim == 3:
                def add_fn(args):
                    p_v, p_s8, p_s32, u_v = args
                    p_layer = QArrayImpl._tree_unflatten((True, p.tgrid), (p_v, p_s8, p_s32))
                    new_p = p_layer.astype(jnp.float32) + u_v
                    return QArrayImpl(new_p, p.tgrid)._tree_flatten()[0]
                nv, ns8, ns32 = jax.lax.map(add_fn, (p.get_value(), p._QArrayImpl__sc_fp8, p._QArrayImpl__sc_fp32, u))
                return QArrayImpl._tree_unflatten((True, p.tgrid), (nv, ns8, ns32))
            return QArrayImpl(p.astype(jnp.float32) + u, p.tgrid)
        return p + u

    step_num = opt_state.count

    new_params = jax.tree_util.tree_map(
        add_param,
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