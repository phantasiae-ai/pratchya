

from pratchya.preset import PratchyaDummyConfig, Pratchya500M
from pratchya._nnx_impl._model import NQPratchyaCausalLM

import jax, jax.numpy as jnp
from flax import nnx
import optax


@nnx.jit(donate_argnums=(0, 1))
def train_step_fn(model: NQPratchyaCausalLM, optimizer: nnx.Optimizer, batch):
    def loss_fn(model):
        output = model(batch['input_ids'], batch['input_ids'])
        return output.loss
        
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    
    return loss

def prepare_for_train(max_steps = 10, learning_rate = 3e-4, warmup_steps = 5):
    init_lr, max_lr, end_lr = 1e-6, learning_rate, 1e-6
    if isinstance(learning_rate, tuple):
        init_lr, max_lr, end_lr = learning_rate

    schedule = optax.warmup_cosine_decay_schedule(init_lr, max_lr, warmup_steps, max_steps, end_lr)
    tx = optax.adamw(schedule, mu_dtype=jnp.bfloat16)
    model = NQPratchyaCausalLM(PratchyaDummyConfig)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    return model, optimizer, schedule

max_steps = 100
warmup_steps = 20
lr = (1e-6, 5e-3, 1e-4)
logging_steps = 10

print("=== prepare model and optimizer ===")
model, optimizer, schedule = prepare_for_train(max_steps, lr, warmup_steps)

print("=== prepare dataset for training ===")
inp = jnp.arange(10).reshape(1, -1)
inp = {'input_ids': inp}

def display_hyperparam():
    print("\n=== Display Hyper Parameters ===")
    print(f"    max_steps:      {max_steps}")
    if isinstance(lr, tuple):
        i, j, k = lr
        print(f"    init_lr:        {i}")
        print(f"    max_lr:         {j}")
        print(f"    end_lr:         {k}")
    else:
        print(f"    init_lr:        {lr}")
    
    print(f"    logging_steps:  {logging_steps}")
    print(f"    warmup_steps:   {warmup_steps}")
    print()


def train():
    history = []
    print(f"=== init training ===")
    display_hyperparam()

    for i in range(max_steps):
        loss = train_step_fn(model, optimizer, inp)
        if (i + 1) % logging_steps == 0 or i == 0:
            curr_lr = schedule(i + 1)
            print(f"step {i + 1:<4} loss: {loss:.4f} | lr: {curr_lr:.4f}")
            history.append((i + 1, loss))

    print("=== training completed ===")
    return history


history = train()

import matplotlib.pyplot as plt

x, y = zip(*history)

filename = "plot.png"

plt.plot(x, y)
plt.xlabel("step")
plt.ylabel("loss")
plt.title("Training Loss")
plt.savefig(filename)
print(f"=== Save Training Loss Picture as {filename} ===")