
from jax.typing import ArrayLike, DTypeLike

import jax.numpy as jnp, jax
import math
from typing import Tuple
import functools
from jax.tree_util import register_pytree_node
from jax.experimental import pallas as pl

FP8E4M3_MIN = jnp.finfo(jnp.float8_e4m3fn).smallest_subnormal
FP8E8M0_MIN = jnp.finfo(jnp.float8_e8m0fnu).smallest_subnormal
import numpy as np
FP32_MIN = np.float32(1e-12)

IS_CPU = jax.default_backend() == "cpu"
IS_TPU = jax.default_backend() == "tpu"
INTERPRET = IS_CPU

if IS_TPU:
    from jax.experimental.pallas import tpu as pltpu
    MEM_SPACE = pltpu.VMEM
else:
    MEM_SPACE = None


class ArrayFP8:

    def __init__(self, x, block_grid: Tuple[int, int] = (1, 16), _sc_fp8=None, _sc_fp32=None):
        if _sc_fp8 is not None and _sc_fp32 is not None:
            self.__value = x
            self.__sc_fp8 = _sc_fp8
            self.__sc_fp32 = _sc_fp32
            self.__block_grid = block_grid
        else:
            self.__block_grid = block_grid
            if isinstance(x, ArrayFP8):
                if x.block_grid == block_grid:
                    self.__value = x._ArrayFP8__value
                    self.__sc_fp8 = x._ArrayFP8__sc_fp8
                    self.__sc_fp32 = x._ArrayFP8__sc_fp32
                else:
                    self.__value, self.__sc_fp8, self.__sc_fp32 = self.__quantize(x.astype(jnp.float32))
            else:
                self.__value, self.__sc_fp8, self.__sc_fp32 = self.__quantize(x)

    def get_value(self):
        return self.__value

    def __quantize(self, x: ArrayLike):
        x, sc_fp8, sc_fp32 = quantize_impl(x, self.__block_grid)
        return x, sc_fp8, sc_fp32

    def dequantize(self, dtype):
        x = dequantize_impl(self.__value, self.__sc_fp8, self.__sc_fp32, dtype=dtype, block_grid=self.__block_grid)
        return x
    
    @property
    def shape(self):
        return self.__value.shape
    
    @property
    def size(self):
        return self.__value.size
    
    @property
    def dtype(self):
        return self.__value.dtype
    
    @property
    def grid_shape(self):
        return self.__block_grid
    
    def _promote(self, other, grid_idx=0):
        if isinstance(other, ArrayFP8):
            return other
            
        shape = other.shape if hasattr(other, 'shape') else jnp.array(other).shape
        if len(shape) < 2:
            H, W = 1, shape[0] if len(shape) > 0 else 1
        else:
            H, W = shape[-2:]
            
        # Target grid based on grid_idx (0 for LHS matching, 1 for RHS matching)
        target_a = self.__block_grid[0] if grid_idx == 0 else self.__block_grid[1]
        target_b = self.__block_grid[1]
        
        grid_a = math.gcd(target_a, H)
        grid_b = math.gcd(target_b, W)
        
        return ArrayFP8(other, block_grid=(grid_a, grid_b))

    def __matmul__(self, other):
        other = self._promote(other, 1)
        x_out = matmul_impl(self.__value, self.__sc_fp8, self.__sc_fp32, other.__value, other.__sc_fp8, other.__sc_fp32, self.__block_grid, other.__block_grid)
        return ArrayFP8(x_out, (self.__block_grid[0], other.__block_grid[1]))
    
    def __rmatmul__(self, other):
        other = self._promote(other, 0)
        x_out = matmul_impl(other.__value, other.__sc_fp8, other.__sc_fp32, self.__value, self.__sc_fp8, self.__sc_fp32, other.__block_grid, self.__block_grid)
        return ArrayFP8(x_out, (other.__block_grid[0], self.__block_grid[1]))
    
    def __add__(self, other):
        other = self._promote(other)
        if self.__block_grid != other.block_grid:
            return ArrayFP8(self.astype(jnp.float32) + other.astype(jnp.float32), self.__block_grid)
        x_out = add_impl(self.__value, self.__sc_fp8, self.__sc_fp32, other.__value, other.__sc_fp8, other.__sc_fp32, self.__block_grid)
        return ArrayFP8(x_out, self.__block_grid)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = self._promote(other)
        if self.__block_grid != other.block_grid:
            return ArrayFP8(self.astype(jnp.float32) - other.astype(jnp.float32), self.__block_grid)
        x_out = sub_impl(self.__value, self.__sc_fp8, self.__sc_fp32, other.__value, other.__sc_fp8, other.__sc_fp32, self.__block_grid)
        return ArrayFP8(x_out, self.__block_grid)

    def __rsub__(self, other):
        other = self._promote(other)
        if self.__block_grid != other.block_grid:
            return ArrayFP8(other.astype(jnp.float32) - self.astype(jnp.float32), self.__block_grid)
        x_out = sub_impl(other.__value, other.__sc_fp8, other.__sc_fp32, self.__value, self.__sc_fp8, self.__sc_fp32, self.__block_grid)
        return ArrayFP8(x_out, self.__block_grid)

    def __mul__(self, other):
        other = self._promote(other)
        if self.__block_grid != other.block_grid:
            return ArrayFP8(self.astype(jnp.float32) * other.astype(jnp.float32), self.__block_grid)
        x_out = mul_impl(self.__value, self.__sc_fp8, self.__sc_fp32, other.__value, other.__sc_fp8, other.__sc_fp32, self.__block_grid)
        return ArrayFP8(x_out, self.__block_grid)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = self._promote(other)
        if self.__block_grid != other.block_grid:
            return ArrayFP8(self.astype(jnp.float32) / other.astype(jnp.float32), self.__block_grid)
        x_out = truediv_impl(self.__value, self.__sc_fp8, self.__sc_fp32, other.__value, other.__sc_fp8, other.__sc_fp32, self.__block_grid)
        return ArrayFP8(x_out, self.__block_grid)

    def __rtruediv__(self, other):
        other = self._promote(other)
        if self.__block_grid != other.block_grid:
            return ArrayFP8(other.astype(jnp.float32) / self.astype(jnp.float32), self.__block_grid)
        x_out = truediv_impl(other.__value, other.__sc_fp8, other.__sc_fp32, self.__value, self.__sc_fp8, self.__sc_fp32, self.__block_grid)
        return ArrayFP8(x_out, self.__block_grid)

    def __floordiv__(self, other):
        other = self._promote(other)
        if self.__block_grid != other.block_grid:
            return ArrayFP8(self.astype(jnp.float32) // other.astype(jnp.float32), self.__block_grid)
        x_out = floordiv_impl(self.__value, self.__sc_fp8, self.__sc_fp32, other.__value, other.__sc_fp8, other.__sc_fp32, self.__block_grid)
        return ArrayFP8(x_out, self.__block_grid)

    def __rfloordiv__(self, other):
        other = self._promote(other)
        if self.__block_grid != other.block_grid:
            return ArrayFP8(other.astype(jnp.float32) // self.astype(jnp.float32), self.__block_grid)
        x_out = floordiv_impl(other.__value, other.__sc_fp8, other.__sc_fp32, self.__value, self.__sc_fp8, self.__sc_fp32, self.__block_grid)
        return ArrayFP8(x_out, self.__block_grid)

    def __getitem__(self, idx):
        # We use pure JAX here instead of self.dequantize() (which calls a Pallas kernel).
        # Pallas kernels are opaque to XLA, so it would force a full materialization of the array.
        # By using pure JAX, XLA will push the `[idx]` slice ALL the way down through the reshapes
        # and only dequantize the exact bytes you requested!
        x_fp8 = self.__value
        orig_shape = x_fp8.shape
        if x_fp8.ndim == 1:
            x_fp8 = x_fp8.reshape(1, orig_shape[0])
        elif x_fp8.ndim == 0:
            x_fp8 = x_fp8.reshape(1, 1)
            
        shape = x_fp8.shape
        M, K = self.__sc_fp8.shape
        a, b = self.__block_grid
        
        x_reshaped = x_fp8.reshape(*shape[:-2], shape[-2] // a, a, shape[-1] // b, b)
        dims = list(range(x_reshaped.ndim))
        x_reshaped = x_reshaped.transpose(*dims[:-4], dims[-4], dims[-2], dims[-3], dims[-1])
        x_reshaped = x_reshaped.reshape(M, K, a, b)
        
        sc_fp8_reshaped = self.__sc_fp8.reshape(M, K, 1, 1).astype(jnp.float32)
        sc_fp32_reshaped = self.__sc_fp32.reshape(M, 1, 1, 1)
        
        x_sc = sc_fp8_reshaped * sc_fp32_reshaped
        x_out = x_reshaped.astype(jnp.float32) * x_sc / 256.0
        
        x_out = x_out.reshape(*shape[:-2], shape[-2] // a, shape[-1] // b, a, b)
        x_out = x_out.transpose(*dims[:-4], dims[-4], dims[-2], dims[-3], dims[-1])
        x_fp32 = x_out.reshape(*orig_shape)
        
        sliced_x = x_fp32[idx]
        
        old_a, old_b = self.__block_grid
        
        if sliced_x.ndim == 0:
            H, W = 1, 1
        elif sliced_x.ndim == 1:
            H, W = 1, sliced_x.shape[0]
        else:
            H, W = sliced_x.shape[-2:]
            
        # Adapt block_grid to the new spatial dimensions to avoid non-divisible crashes!
        new_grid = (math.gcd(old_a, H), math.gcd(old_b, W))
        
        return ArrayFP8(sliced_x, new_grid)

    def __neg__(self):
        new_obj = ArrayFP8.__new__(ArrayFP8)
        new_obj._ArrayFP8__block_grid = self.__block_grid
        new_obj._ArrayFP8__value = self.__value
        new_obj._ArrayFP8__sc_fp8 = self.__sc_fp8
        new_obj._ArrayFP8__sc_fp32 = -self.__sc_fp32
        return new_obj

    def apply(self, func):
        fp32_val = self.dequantize(jnp.float32)
        out_val = func(fp32_val)
        
        old_a, old_b = self.__block_grid
        if out_val.ndim == 0:
            H, W = 1, 1
        elif out_val.ndim == 1:
            H, W = 1, out_val.shape[0]
        else:
            H, W = out_val.shape[-2:]
            
        new_grid = (math.gcd(old_a, H), math.gcd(old_b, W))
        return ArrayFP8(out_val, new_grid)

    def __abs__(self):
        new_obj = ArrayFP8.__new__(ArrayFP8)
        new_obj._ArrayFP8__block_grid = self.__block_grid
        new_obj._ArrayFP8__value = jnp.abs(self.__value)
        new_obj._ArrayFP8__sc_fp8 = self.__sc_fp8
        new_obj._ArrayFP8__sc_fp32 = jnp.abs(self.__sc_fp32)
        return new_obj

    def __repr__(self):
        return f'{self.__value}'

    @property
    def shape(self):
        return self.__value.shape

    @property
    def at(self):
        class _FP8IndexUpdateHelper:
            def __init__(self, fp8_array):
                self.fp8_array = fp8_array
                
            def __getitem__(self, idx):
                class _Setter:
                    def __init__(self, fp8_array, idx):
                        self.fp8_array = fp8_array
                        self.idx = idx
                        
                    def set(self, val):
                        val_fp32 = val.astype(jnp.float32) if isinstance(val, ArrayFP8) else val
                        out_fp32 = self.fp8_array.dequantize(jnp.float32).at[self.idx].set(val_fp32)
                        return ArrayFP8(out_fp32, self.fp8_array.block_grid)
                return _Setter(self.fp8_array, idx)
        return _FP8IndexUpdateHelper(self)

    @property
    def block_grid(self):
        return self.__block_grid

    def astype(self, dtype):
        return self.dequantize(dtype)

    @property
    def mT(self):
        grid = (self.__block_grid[1], self.__block_grid[0])
        return ArrayFP8(self.astype(jnp.float32).mT, grid)

    def reshape(self, *args, **kwargs):
        x_fp32 = self.dequantize(jnp.float32)
        reshaped_x = x_fp32.reshape(*args, **kwargs)
        return self._requantize(reshaped_x)

    def transpose(self, *axes):
        axes = axes[0] if len(axes) == 1 and isinstance(axes[0], (tuple, list)) else axes
        x_fp32 = self.dequantize(jnp.float32)
        transposed_x = x_fp32.transpose(*axes)
        return self._requantize(transposed_x)

    def sum(self, *args, **kwargs):
        x_fp32 = self.dequantize(jnp.float32)
        summed_x = x_fp32.sum(*args, **kwargs)
        return self._requantize(summed_x)

    def _requantize(self, out_val):
        old_a, old_b = self.__block_grid
        if out_val.ndim == 0:
            H, W = 1, 1
        elif out_val.ndim == 1:
            H, W = 1, out_val.shape[0]
        else:
            H, W = out_val.shape[-2:]
        new_grid = (math.gcd(old_a, H), math.gcd(old_b, W))
        return ArrayFP8(out_val, new_grid)


    @classmethod
    def concat(cls, arrays, axis=0):
        fp32_arrays = [a.astype(jnp.float32) if isinstance(a, ArrayFP8) else a for a in arrays]
        out_fp32 = jnp.concat(fp32_arrays, axis=axis)
        
        # Use the block_grid of the first ArrayFP8 in the list
        first_fp8 = next((a for a in arrays if isinstance(a, ArrayFP8)), None)
        grid = first_fp8.block_grid if first_fp8 else (1, 128)
        
        # Re-adapt the grid just in case
        old_a, old_b = grid
        if out_fp32.ndim == 0:
            H, W = 1, 1
        elif out_fp32.ndim == 1:
            H, W = 1, out_fp32.shape[0]
        else:
            H, W = out_fp32.shape[-2:]
            
        new_grid = (math.gcd(old_a, H), math.gcd(old_b, W))
        return cls(out_fp32, new_grid)

    def _tree_flatten(self):
        children = (self.__value, self.__sc_fp8, self.__sc_fp32)
        aux_data = self.__block_grid
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.__block_grid = aux_data
        obj.__value, obj.__sc_fp8, obj.__sc_fp32 = children
        return obj


register_pytree_node(ArrayFP8, ArrayFP8._tree_flatten, ArrayFP8._tree_unflatten)

def quantize_kernel(x_ref, x_out_ref, sc_fp8_ref, sc_fp32_ref):
    x = x_ref[...]
    a = x.shape[-2]
    b = x.shape[-1]
    
    m_i = jnp.max(jnp.abs(x), axis=(-2, -1), keepdims=True)
    m_i = jnp.maximum(m_i, FP8E8M0_MIN.astype(m_i.dtype))
    
    m_i_tmp = m_i.reshape(1, sc_fp8_ref.shape[-2], sc_fp8_ref.shape[-1])
    
    M_j = jnp.max(m_i_tmp, axis=(-2, -1), keepdims=True)
    M_j = jnp.maximum(M_j, FP32_MIN.astype(M_j.dtype))
    
    S_1i_f32 = m_i_tmp / M_j
    S_1i_f32 = jnp.maximum(S_1i_f32, FP32_MIN.astype(M_j.dtype))
    S_1i = S_1i_f32.astype(jnp.float8_e8m0fnu)
    
    S_eff = (S_1i.astype(M_j.dtype) * M_j).reshape(-1, 1, 1)
    S_eff = jnp.maximum(S_eff, 1e-7)
    
    x_out = (x / S_eff * 256.0).astype(jnp.float8_e4m3fn)
    
    x_out_ref[...] = x_out
    sc_fp8_ref[...] = S_1i.reshape(1, 1, sc_fp8_ref.shape[-2], sc_fp8_ref.shape[-1])
    sc_fp32_ref[...] = M_j.reshape(1, 1, 1, 1)


def quantize_impl(x: ArrayLike, block_grid: Tuple = (128, 128)):
    orig_shape = x.shape
    if x.ndim == 1:
        x = x.reshape(1, orig_shape[0])
    elif x.ndim == 0:
        x = x.reshape(1, 1)
        
    shape = x.shape
    a, b = block_grid
    
    # Valid spatial memory layout
    x_reshaped = x.reshape(*shape[:-2], shape[-2] // a, a, shape[-1] // b, b)
    dims = list(range(x_reshaped.ndim))
    x_reshaped = x_reshaped.transpose(*dims[:-4], dims[-4], dims[-2], dims[-3], dims[-1])
    
    total_elements = math.prod(shape)
    M_blocks = total_elements // (a * b)
    
    target_K = max(1, (a * b) // 16)
    K = math.gcd(target_K, M_blocks)
    
    M = M_blocks // K
    
    x_reshaped = x_reshaped.reshape(M, K, a, b)
    
    x_fp8, sc_fp8, sc_fp32 = pl.pallas_call(
        quantize_kernel,
        out_shape=(
            jax.ShapeDtypeStruct(x_reshaped.shape, dtype=jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((M, 1, K, 1), dtype=jnp.float8_e8m0fnu),
            jax.ShapeDtypeStruct((M, 1, 1, 1), dtype=jnp.float32)
        ),
        in_specs=[
            pl.BlockSpec((1, K, a, b), lambda i: (i, 0, 0, 0), memory_space=MEM_SPACE)
        ],
        out_specs=[
            pl.BlockSpec((1, K, a, b), lambda i: (i, 0, 0, 0), memory_space=MEM_SPACE),
            pl.BlockSpec((1, 1, K, 1), lambda i: (i, 0, 0, 0), memory_space=MEM_SPACE),
            pl.BlockSpec((1, 1, 1, 1), lambda i: (i, 0, 0, 0), memory_space=MEM_SPACE),
        ],
        interpret=INTERPRET,
        grid=(M,)
    )(x_reshaped)

    x_fp8 = x_fp8.reshape(*shape[:-2], shape[-2] // a, shape[-1] // b, a, b)
    x_fp8 = x_fp8.transpose(*dims[:-4], dims[-4], dims[-2], dims[-3], dims[-1])

    return x_fp8.reshape(*orig_shape), sc_fp8.reshape(M, K), sc_fp32.reshape(M, 1, 1)

quantize_impl = jax.custom_vjp(quantize_impl, nondiff_argnums=(1,))

def quantize_impl_fwd(x, block_grid):
    out = quantize_impl(x, block_grid)
    return out, (block_grid, out)

def quantize_impl_bwd(block_grid, res, g_out):
    _, out = res
    x_fp8, sc_fp8, sc_fp32 = out
    g_fp8, g_sc8, g_sc32 = g_out
    # We dequantize the gradient g_fp8 using the PRIMAL scales!
    g_f32 = dequantize_impl(g_fp8, sc_fp8, sc_fp32, jnp.float32, block_grid)
    return g_f32,

quantize_impl.defvjp(quantize_impl_fwd, quantize_impl_bwd)


def dequantize_kernel(x_ref, sc_fp8_ref, sc_fp32_ref, o_ref):
    x = x_ref[...]
    sc_fp8 = sc_fp8_ref[...]
    sc_fp32 = sc_fp32_ref[...]
    
    x_sc = sc_fp8.astype(jnp.float32)
    x_sc = (x_sc * sc_fp32).reshape(-1, 1, 1)
    
    o_ref[...] = x.astype(jnp.float32) * x_sc / 256.0


def dequantize_impl(x_fp8: ArrayLike, sc_fp8: ArrayLike, sc_fp32: ArrayLike, dtype: DTypeLike, block_grid: Tuple):
    orig_shape = x_fp8.shape
    if x_fp8.ndim == 1:
        x_fp8 = x_fp8.reshape(1, orig_shape[0])
    elif x_fp8.ndim == 0:
        x_fp8 = x_fp8.reshape(1, 1)
        
    shape = x_fp8.shape
    M, K = sc_fp8.shape[-2:]
    batch_size = math.prod(sc_fp8.shape[:-2]) if sc_fp8.ndim > 2 else 1
    M_total = batch_size * M
    a, b = block_grid
    
    # Valid spatial memory layout
    x_reshaped = x_fp8.reshape(*shape[:-2], shape[-2] // a, a, shape[-1] // b, b)
    dims = list(range(x_reshaped.ndim))
    x_reshaped = x_reshaped.transpose(*dims[:-4], dims[-4], dims[-2], dims[-3], dims[-1])
    
    x_reshaped = x_reshaped.reshape(M_total, K, a, b)
    sc_fp8_reshaped = sc_fp8.reshape(M_total, 1, K, 1)
    sc_fp32_reshaped = sc_fp32.reshape(M_total, 1, 1, 1)
    
    x_out = pl.pallas_call(
        dequantize_kernel,
        out_shape=jax.ShapeDtypeStruct(x_reshaped.shape, dtype=jnp.float32),
        in_specs=[
            pl.BlockSpec((1, K, a, b), lambda i: (i, 0, 0, 0), memory_space=MEM_SPACE),
            pl.BlockSpec((1, 1, K, 1), lambda i: (i, 0, 0, 0), memory_space=MEM_SPACE),
            pl.BlockSpec((1, 1, 1, 1), lambda i: (i, 0, 0, 0), memory_space=MEM_SPACE),
        ],
        out_specs=pl.BlockSpec((1, K, a, b), lambda i: (i, 0, 0, 0), memory_space=MEM_SPACE),
        interpret=INTERPRET,
        grid=(M_total,)
    )(x_reshaped, sc_fp8_reshaped, sc_fp32_reshaped)

    x_out = x_out.reshape(*shape[:-2], shape[-2] // a, shape[-1] // b, a, b)
    x_out = x_out.transpose(*dims[:-4], dims[-4], dims[-2], dims[-3], dims[-1])

    return x_out.reshape(*orig_shape).astype(dtype)

dequantize_impl = jax.custom_vjp(dequantize_impl, nondiff_argnums=(3, 4))

def dequantize_impl_fwd(x_fp8, sc_fp8, sc_fp32, dtype, block_grid):
    out = dequantize_impl(x_fp8, sc_fp8, sc_fp32, dtype, block_grid)
    return out, (sc_fp8,)

def dequantize_impl_bwd(dtype, block_grid, res, g_out):
    sc_fp8, = res
    M, K = sc_fp8.shape
    g_x_fp8, g_sc_fp8, g_sc_fp32 = quantize_impl(g_out, block_grid=block_grid)
    return g_x_fp8, g_sc_fp8.reshape(M, K), g_sc_fp32.reshape(M, 1, 1)

dequantize_impl.defvjp(dequantize_impl_fwd, dequantize_impl_bwd)

def matmul_kernel(x_ref, y_ref, sc_x8_ref, sc_x32_ref, sc_y8_ref, sc_y32_ref, o_ref, *, b):
    acc = jnp.zeros((x_ref.shape[0], y_ref.shape[1]), dtype=jnp.float32)
    def body(k, acc):
        x_block = jax.lax.dynamic_slice(x_ref[...], (0, k*b), (x_ref.shape[0], b))
        y_block = jax.lax.dynamic_slice(y_ref[...], (k*b, 0), (b, y_ref.shape[1]))
        sc_x8 = jax.lax.dynamic_slice(sc_x8_ref[...], (0, k), (1, 1))[0, 0]
        sc_x32 = jax.lax.dynamic_slice(sc_x32_ref[...], (0, k), (1, 1))[0, 0]
        sc_y8 = jax.lax.dynamic_slice(sc_y8_ref[...], (k, 0), (1, 1))[0, 0]
        sc_y32 = jax.lax.dynamic_slice(sc_y32_ref[...], (k, 0), (1, 1))[0, 0]
        scale_x = sc_x8.astype(jnp.float32) * sc_x32
        scale_y = sc_y8.astype(jnp.float32) * sc_y32
        prod = jnp.matmul(x_block, y_block, preferred_element_type=jnp.float32)
        prod = prod * (scale_x * scale_y) / 65536.0
        return acc + prod
    acc = jax.lax.fori_loop(0, x_ref.shape[1] // b, body, acc)
    o_ref[...] = acc

def matmul_impl_2d(x_fp8, sc_x8, sc_x32, y_fp8, sc_y8, sc_y32, a, b, c):
    H_a, W_a = x_fp8.shape
    H_b, W_b = y_fp8.shape
    x_out = pl.pallas_call(
        functools.partial(matmul_kernel, b=b),
        out_shape=jax.ShapeDtypeStruct((H_a, W_b), dtype=jnp.float32),
        in_specs=[
            pl.BlockSpec((a, W_a), lambda i, j: (i, 0), memory_space=MEM_SPACE),
            pl.BlockSpec((H_b, c), lambda i, j: (0, j), memory_space=MEM_SPACE),
            pl.BlockSpec((1, W_a // b), lambda i, j: (i, 0), memory_space=MEM_SPACE),
            pl.BlockSpec((1, W_a // b), lambda i, j: (i, 0), memory_space=MEM_SPACE),
            pl.BlockSpec((H_b // b, 1), lambda i, j: (0, j), memory_space=MEM_SPACE),
            pl.BlockSpec((H_b // b, 1), lambda i, j: (0, j), memory_space=MEM_SPACE),
        ],
        out_specs=pl.BlockSpec((a, c), lambda i, j: (i, j), memory_space=MEM_SPACE),
        grid=(H_a // a, W_b // c),
        interpret=INTERPRET
    )(x_fp8, y_fp8, sc_x8, sc_x32, sc_y8, sc_y32)
    return x_out

def matmul_impl(x_fp8, x_sc8, x_sc32, y_fp8, y_sc8, y_sc32, block_grid_x: Tuple, block_grid_y: Tuple):
    H_a, W_a = x_fp8.shape[-2:]
    H_b, W_b = y_fp8.shape[-2:]
    M_x, K_x = x_sc8.shape
    M_y, K_y = y_sc8.shape
    a, b = block_grid_x
    b2, c = block_grid_y
    
    if b != b2 or H_a % a != 0 or W_a % b != 0 or H_b % b2 != 0 or W_b % c != 0:
        # Graceful fallback to pure JAX if grids are totally misaligned
        x_f32 = dequantize_impl(x_fp8, x_sc8, x_sc32, jnp.float32, block_grid_x)
        y_f32 = dequantize_impl(y_fp8, y_sc8, y_sc32, jnp.float32, block_grid_y)
        return jnp.matmul(x_f32, y_f32)
        
    orig_batch_x = x_fp8.shape[:-2]
    orig_batch_y = y_fp8.shape[:-2]
    batch_shape = jnp.broadcast_shapes(orig_batch_x, orig_batch_y)
    x_fp8 = jnp.broadcast_to(x_fp8, (*batch_shape, H_a, W_a))
    y_fp8 = jnp.broadcast_to(y_fp8, (*batch_shape, H_b, W_b))
    sc_x8 = x_sc8.reshape(*orig_batch_x, H_a // a, W_a // b)
    sc_x8 = jnp.broadcast_to(sc_x8, (*batch_shape, H_a // a, W_a // b))
    sc_x32 = jnp.repeat(x_sc32.reshape(-1), K_x).reshape(*orig_batch_x, H_a // a, W_a // b)
    sc_x32 = jnp.broadcast_to(sc_x32, (*batch_shape, H_a // a, W_a // b))
    sc_y8 = y_sc8.reshape(*orig_batch_y, H_b // b, W_b // c)
    sc_y8 = jnp.broadcast_to(sc_y8, (*batch_shape, H_b // b, W_b // c))
    sc_y32 = jnp.repeat(y_sc32.reshape(-1), K_y).reshape(*orig_batch_y, H_b // b, W_b // c)
    sc_y32 = jnp.broadcast_to(sc_y32, (*batch_shape, H_b // b, W_b // c))
    B = math.prod(batch_shape) if len(batch_shape) > 0 else 1
    x_fp8 = x_fp8.reshape(B, H_a, W_a)
    y_fp8 = y_fp8.reshape(B, H_b, W_b)
    sc_x8 = sc_x8.reshape(B, H_a // a, W_a // b)
    sc_x32 = sc_x32.reshape(B, H_a // a, W_a // b)
    sc_y8 = sc_y8.reshape(B, H_b // b, W_b // c)
    sc_y32 = sc_y32.reshape(B, H_b // b, W_b // c)
    
    mapped_fn = jax.vmap(functools.partial(matmul_impl_2d, a=a, b=b, c=c))
    out = mapped_fn(x_fp8, sc_x8, sc_x32, y_fp8, sc_y8, sc_y32)
    return out.reshape(*batch_shape, H_a, W_b)

matmul_impl = jax.custom_vjp(matmul_impl, nondiff_argnums=(6, 7))

def matmul_impl_fwd(x_fp8, x_sc8, x_sc32, y_fp8, y_sc8, y_sc32, block_grid_x, block_grid_y):
    out = matmul_impl(x_fp8, x_sc8, x_sc32, y_fp8, y_sc8, y_sc32, block_grid_x, block_grid_y)
    return out, (x_fp8, x_sc8, x_sc32, y_fp8, y_sc8, y_sc32)

def matmul_impl_bwd(block_grid_x, block_grid_y, res, g_out):
    x_fp8, x_sc8, x_sc32, y_fp8, y_sc8, y_sc32 = res
    x_f32 = dequantize_impl(x_fp8, x_sc8, x_sc32, jnp.float32, block_grid_x)
    y_f32 = dequantize_impl(y_fp8, y_sc8, y_sc32, jnp.float32, block_grid_y)
    
    g_x_f32 = g_out @ jnp.swapaxes(y_f32, -1, -2)
    g_y_f32 = jnp.swapaxes(x_f32, -1, -2) @ g_out
    
    g_x_fp8, g_x_sc8, g_x_sc32 = quantize_impl(g_x_f32, block_grid=block_grid_x)
    g_x_sc8 = g_x_sc8.reshape(*x_sc8.shape)
    g_x_sc32 = g_x_sc32.reshape(*x_sc32.shape)
    
    g_y_fp8, g_y_sc8, g_y_sc32 = quantize_impl(g_y_f32, block_grid=block_grid_y)
    g_y_sc8 = g_y_sc8.reshape(*y_sc8.shape)
    g_y_sc32 = g_y_sc32.reshape(*y_sc32.shape)
    
    return g_x_fp8, g_x_sc8, g_x_sc32, g_y_fp8, g_y_sc8, g_y_sc32

matmul_impl.defvjp(matmul_impl_fwd, matmul_impl_bwd)


def elementwise_kernel(x_ref, y_ref, sc_x8_ref, sc_x32_ref, sc_y8_ref, sc_y32_ref, o_ref, *, op):
    x_block = x_ref[...]
    y_block = y_ref[...]
    sc_x8 = sc_x8_ref[...]
    sc_x32 = sc_x32_ref[...]
    sc_y8 = sc_y8_ref[...]
    sc_y32 = sc_y32_ref[...]
    
    scale_x = sc_x8.astype(jnp.float32) * sc_x32 / 256.0
    scale_y = sc_y8.astype(jnp.float32) * sc_y32 / 256.0
    
    x_val = x_block.astype(jnp.float32) * scale_x
    y_val = y_block.astype(jnp.float32) * scale_y
    o_ref[...] = op(x_val, y_val)

def elementwise_impl_2d(x_fp8, sc_x8, sc_x32, y_fp8, sc_y8, sc_y32, *, a, b, op):
    H, W = x_fp8.shape
    x_out = pl.pallas_call(
        functools.partial(elementwise_kernel, op=op),
        out_shape=jax.ShapeDtypeStruct((H, W), dtype=jnp.float32),
        in_specs=[
            pl.BlockSpec((a, b), lambda i, j: (i, j), memory_space=MEM_SPACE),
            pl.BlockSpec((a, b), lambda i, j: (i, j), memory_space=MEM_SPACE),
            pl.BlockSpec((1, 1), lambda i, j: (i, j), memory_space=MEM_SPACE),
            pl.BlockSpec((1, 1), lambda i, j: (i, j), memory_space=MEM_SPACE),
            pl.BlockSpec((1, 1), lambda i, j: (i, j), memory_space=MEM_SPACE),
            pl.BlockSpec((1, 1), lambda i, j: (i, j), memory_space=MEM_SPACE),
        ],
        out_specs=pl.BlockSpec((a, b), lambda i, j: (i, j), memory_space=MEM_SPACE),
        grid=(H // a, W // b),
        interpret=INTERPRET
    )(x_fp8, y_fp8, sc_x8, sc_x32, sc_y8, sc_y32)
    return x_out


def make_elementwise_impl(op):
    def impl(x_fp8, x_sc8, x_sc32, y_fp8, y_sc8, y_sc32, block_grid: Tuple):
        M_x, K_x = x_sc8.shape
        M_y, K_y = y_sc8.shape
        a, b = block_grid
        
        orig_batch_x = x_fp8.shape[:-2]
        orig_batch_y = y_fp8.shape[:-2]
        batch_shape = jnp.broadcast_shapes(orig_batch_x, orig_batch_y)
        
        orig_H_x = x_fp8.shape[-2] if x_fp8.ndim >= 2 else 1
        orig_W_x = x_fp8.shape[-1] if x_fp8.ndim >= 1 else 1
        orig_H_y = y_fp8.shape[-2] if y_fp8.ndim >= 2 else 1
        orig_W_y = y_fp8.shape[-1] if y_fp8.ndim >= 1 else 1
        
        H, W = x_fp8.shape[-2:] if x_fp8.ndim >= 2 else (1, x_fp8.shape[-1] if x_fp8.ndim == 1 else 1)
        x_fp8 = jnp.broadcast_to(x_fp8, (*batch_shape, H, W)).reshape(-1, H, W)
        y_fp8 = jnp.broadcast_to(y_fp8, (*batch_shape, H, W)).reshape(-1, H, W)
        
        sc_x8 = x_sc8.reshape(*orig_batch_x, orig_H_x // a, orig_W_x // b)
        sc_x8 = jnp.broadcast_to(sc_x8, (*batch_shape, H // a, W // b)).reshape(-1, H // a, W // b)
        
        sc_x32 = jnp.repeat(x_sc32.reshape(-1), K_x).reshape(*orig_batch_x, orig_H_x // a, orig_W_x // b)
        sc_x32 = jnp.broadcast_to(sc_x32, (*batch_shape, H // a, W // b)).reshape(-1, H // a, W // b)
        
        sc_y8 = y_sc8.reshape(*orig_batch_y, orig_H_y // a, orig_W_y // b)
        sc_y8 = jnp.broadcast_to(sc_y8, (*batch_shape, H // a, W // b)).reshape(-1, H // a, W // b)
        
        sc_y32 = jnp.repeat(y_sc32.reshape(-1), K_y).reshape(*orig_batch_y, orig_H_y // a, orig_W_y // b)
        sc_y32 = jnp.broadcast_to(sc_y32, (*batch_shape, H // a, W // b)).reshape(-1, H // a, W // b)
        
        mapped_fn = jax.vmap(functools.partial(elementwise_impl_2d, a=a, b=b, op=op))
        out = mapped_fn(x_fp8, sc_x8, sc_x32, y_fp8, sc_y8, sc_y32)
        return out.reshape(*batch_shape, H, W)
    
    impl = jax.custom_vjp(impl, nondiff_argnums=(6,))
    
    def impl_fwd(x_fp8, x_sc8, x_sc32, y_fp8, y_sc8, y_sc32, block_grid):
        out = impl(x_fp8, x_sc8, x_sc32, y_fp8, y_sc8, y_sc32, block_grid)
        return out, (x_fp8, x_sc8, x_sc32, y_fp8, y_sc8, y_sc32)
        
    def impl_bwd(block_grid, res, g_out):
        x_fp8, x_sc8, x_sc32, y_fp8, y_sc8, y_sc32 = res
        x_f32 = dequantize_impl(x_fp8, x_sc8, x_sc32, jnp.float32, block_grid)
        y_f32 = dequantize_impl(y_fp8, y_sc8, y_sc32, jnp.float32, block_grid)
        
        _, vjp_fn = jax.vjp(op, x_f32, y_f32)
        g_x_f32, g_y_f32 = vjp_fn(g_out)
        
        g_x_fp8, g_x_sc8, g_x_sc32 = quantize_impl(g_x_f32, block_grid=block_grid)
        g_x_sc8 = g_x_sc8.reshape(*x_sc8.shape)
        g_x_sc32 = g_x_sc32.reshape(*x_sc32.shape)
        
        g_y_fp8, g_y_sc8, g_y_sc32 = quantize_impl(g_y_f32, block_grid=block_grid)
        g_y_sc8 = g_y_sc8.reshape(*y_sc8.shape)
        g_y_sc32 = g_y_sc32.reshape(*y_sc32.shape)
        
        return g_x_fp8, g_x_sc8, g_x_sc32, g_y_fp8, g_y_sc8, g_y_sc32

    impl.defvjp(impl_fwd, impl_bwd)
    return impl

add_impl = make_elementwise_impl(jnp.add)
mul_impl = make_elementwise_impl(jnp.multiply)
sub_impl = make_elementwise_impl(jnp.subtract)
truediv_impl = make_elementwise_impl(jnp.divide)
floordiv_impl = make_elementwise_impl(jnp.floor_divide)


def fp8_matmul(x_f32, y_f32, block_grid=(1, 16)):
    """End-to-End differentiable FP8 Matrix Multiplication"""
    x_fp8 = ArrayFP8(x_f32, block_grid)
    y_fp8 = ArrayFP8(y_f32, block_grid)
    out_fp8 = x_fp8 @ y_fp8
    return out_fp8.dequantize(jnp.float32)

fp8_matmul = jax.custom_vjp(fp8_matmul, nondiff_argnums=(2,))

def fp8_matmul_fwd(x_f32, y_f32, block_grid):
    out = fp8_matmul(x_f32, y_f32, block_grid)
    # Save the float32 inputs for the backward pass
    return out, (x_f32, y_f32)

def fp8_matmul_bwd(block_grid, res, g_out):
    x_f32, y_f32 = res
    
    # Transpose the raw float32 inputs
    y_f32_T = jnp.swapaxes(y_f32, -1, -2)
    x_f32_T = jnp.swapaxes(x_f32, -1, -2)
    
    # Calculate gradients using the same accelerated FP8 matmul!
    g_x = fp8_matmul(g_out, y_f32_T, block_grid)
    g_y = fp8_matmul(x_f32_T, g_out, block_grid)
    
    return g_x, g_y

fp8_matmul.defvjp(fp8_matmul_fwd, fp8_matmul_bwd)





