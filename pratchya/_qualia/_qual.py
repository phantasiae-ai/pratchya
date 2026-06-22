import jax, jax.numpy as jnp
from ._qarr import QArrayImpl


# The magic module-level __getattr__! 
# This intercepts ANY function call you make on this module (e.g. qual.silu, qual.sum)
def __getattr__(name):
    # 1. Find the target function in JAX
    if hasattr(jnp, name):
        jax_fn = getattr(jnp, name)
    elif hasattr(jax.nn, name):
        jax_fn = getattr(jax.nn, name)
    elif hasattr(jax.lax, name):
        jax_fn = getattr(jax.lax, name)
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
        
    # 2. Create the wrapper that unboxes and re-boxes QArrayImpl
    def wrapper(*args, **kwargs):
        
        # Helper to unwrap QArrayImpl arguments safely
        def unwrap(v):
            if hasattr(v, '__jax_array__'):
                return v.__jax_array__()
            # If QArrayImpl doesn't have __jax_array__ yet, use its internal value
            if isinstance(v, QArrayImpl):
                return v.astype(jnp.float32) # Or whatever method you prefer to unpack
            return v
            
        unwrapped_args = [unwrap(a) for a in args]
        unwrapped_kwargs = {k: unwrap(v) for k, v in kwargs.items()}
        
        # Run the actual JAX math
        result = jax_fn(*unwrapped_args, **unwrapped_kwargs)
        
        # Re-wrap the result back into a QArrayImpl (not quantized by default for activations)
        if isinstance(result, tuple):
            return tuple(QArrayImpl(r, quant=False) for r in result)
        return QArrayImpl(result, quant=False)
        
    return wrapper
