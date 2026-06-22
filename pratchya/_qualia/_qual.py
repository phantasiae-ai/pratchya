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
        
        # Find the first quantized input to steal its grid logic and shape
        quant_arg = next((a for a in args if isinstance(a, QArrayImpl) and hasattr(a, '_QArrayImpl__tgrid')), None)
        
        if quant_arg:
            if isinstance(result, tuple):
                # Only requantize elements that preserved their shape
                return tuple(
                    quant_arg._requantize(r) if r.shape == quant_arg.shape else QArrayImpl(r, quant=False)
                    for r in result
                )
            
            # If shape is exactly the same, requantize. Otherwise leave it as FP32.
            if result.shape == quant_arg.shape:
                return quant_arg._requantize(result)
            return QArrayImpl(result, quant=False)
            
        # Re-wrap the result back into an unquantized QArrayImpl if no input was quantized
        if isinstance(result, tuple):
            return tuple(QArrayImpl(r, quant=False) for r in result)
        return QArrayImpl(result, quant=False)
        
    return wrapper
