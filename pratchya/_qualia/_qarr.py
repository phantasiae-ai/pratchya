
from jax.typing import ArrayLike, DTypeLike
import jax.numpy as jnp, jax
import math
from typing import Tuple, Optional, Union
import functools
from jax.tree_util import register_pytree_node
import numpy as np

FP8E4M3_MIN = jnp.finfo(jnp.float8_e4m3fn).smallest_subnormal
FP8E8M0_MIN = jnp.finfo(jnp.float8_e8m0fnu).smallest_subnormal
FP32_MIN = np.float32(1e-12)


class QArrayImpl:
    def __init__(self, value: Union['QArrayImpl', jax.Array], tgrid: Optional[Tuple[int, int]]=None, quant: bool=True):
        if not quant:
            self.__value = value

        elif isinstance(value, QArrayImpl):
            self.__tgrid = value._QArrayImpl__tgrid
            self.__value = value._QArrayImpl__value
            self.__sc_fp8 = value._QArrayImpl__sc_fp8
            self.__sc_fp32 = value._QArrayImpl__sc_fp32

        else:
            assert tgrid is not None
            self.__tgrid = tgrid
            self.__value, self.__sc_fp8, self.__sc_fp32 = self.__quantize(value.astype(jnp.float32))

    def get_value(self):
        return self.__value

    def __quantize(self, value):
        from ._qarr import quantize_impl
        
        # Prevent OOM during optimizer steps by sequentially quantizing layers
        if value.ndim == 3:
            def q_fn(x):
                return quantize_impl(x, self.__tgrid)
            # jax.lax.map returns a tuple of stacked arrays
            return jax.lax.map(q_fn, value)
            
        return quantize_impl(value, self.__tgrid)

    def dequantize(self, dtype):
        if self.__value.dtype != jnp.float8_e4m3fn:
            return self.__value.astype(dtype)
        
        # Prevent OOM during optimizer steps by sequentially dequantizing layers
        if self.__value.ndim == 3:
            def dq_fn(args):
                v, sc8, sc32 = args
                return dequantize_impl(v, sc8, sc32, dtype=dtype, tgrid=self.__tgrid)
            return jax.lax.map(dq_fn, (self.__value, self.__sc_fp8, self.__sc_fp32))
            
        x = dequantize_impl(self.__value, self.__sc_fp8, self.__sc_fp32, dtype=dtype, tgrid=self.__tgrid)
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
    def tgrid(self):
        return self.__tgrid
    
    def _promote(self, other, grid_idx=0):
        if isinstance(other, QArrayImpl):
            return other
            
        shape = other.shape if hasattr(other, 'shape') else jnp.array(other).shape
        if len(shape) < 2:
            H, W = 1, shape[0] if len(shape) > 0 else 1
        else:
            H, W = shape[-2:]
            
        # Target grid based on grid_idx (0 for LHS matching, 1 for RHS matching)
        target_a = self.__tgrid[0] if grid_idx == 0 else self.__tgrid[1]
        target_b = self.__tgrid[1]
        
        grid_a = math.gcd(target_a, H)
        grid_b = math.gcd(target_b, W)
        
        return QArrayImpl(other, tgrid=(grid_a, grid_b))

    def __matmul__(self, other):
        other = self._promote(other, 1)
        x_out = matmul_impl(self.__value, self.__sc_fp8, self.__sc_fp32, other.__value, other.__sc_fp8, other.__sc_fp32, self.__tgrid, other.__tgrid)
        return QArrayImpl(x_out, (self.__tgrid[0], other.__tgrid[1]))
    
    def __rmatmul__(self, other):
        other = self._promote(other, 0)
        x_out = matmul_impl(other.__value, other.__sc_fp8, other.__sc_fp32, self.__value, self.__sc_fp8, self.__sc_fp32, other.__tgrid, self.__tgrid)
        return QArrayImpl(x_out, (other.__tgrid[0], self.__tgrid[1]))
    
    def __add__(self, other):
        other = self._promote(other)
        if self.__tgrid != other.tgrid:
            return QArrayImpl(self.astype(jnp.float32) + other.astype(jnp.float32), self.__tgrid)
        x_out = add_impl(self.__value, self.__sc_fp8, self.__sc_fp32, other.__value, other.__sc_fp8, other.__sc_fp32, self.__tgrid)
        return QArrayImpl(x_out, self.__tgrid)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = self._promote(other)
        if self.__tgrid != other.tgrid:
            return QArrayImpl(self.astype(jnp.float32) - other.astype(jnp.float32), self.__tgrid)
        x_out = sub_impl(self.__value, self.__sc_fp8, self.__sc_fp32, other.__value, other.__sc_fp8, other.__sc_fp32, self.__tgrid)
        return QArrayImpl(x_out, self.__tgrid)

    def __rsub__(self, other):
        other = self._promote(other)
        if self.__tgrid != other.tgrid:
            return QArrayImpl(other.astype(jnp.float32) - self.astype(jnp.float32), self.__tgrid)
        x_out = sub_impl(other.__value, other.__sc_fp8, other.__sc_fp32, self.__value, self.__sc_fp8, self.__sc_fp32, self.__tgrid)
        return QArrayImpl(x_out, self.__tgrid)

    def __mul__(self, other):
        other = self._promote(other)
        if self.__tgrid != other.tgrid:
            return QArrayImpl(self.astype(jnp.float32) * other.astype(jnp.float32), self.__tgrid)
        x_out = mul_impl(self.__value, self.__sc_fp8, self.__sc_fp32, other.__value, other.__sc_fp8, other.__sc_fp32, self.__tgrid)
        return QArrayImpl(x_out, self.__tgrid)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = self._promote(other)
        if self.__tgrid != other.tgrid:
            return QArrayImpl(self.astype(jnp.float32) / other.astype(jnp.float32), self.__tgrid)
        x_out = truediv_impl(self.__value, self.__sc_fp8, self.__sc_fp32, other.__value, other.__sc_fp8, other.__sc_fp32, self.__tgrid)
        return QArrayImpl(x_out, self.__tgrid)

    def __rtruediv__(self, other):
        other = self._promote(other)
        if self.__tgrid != other.tgrid:
            return QArrayImpl(other.astype(jnp.float32) / self.astype(jnp.float32), self.__tgrid)
        x_out = truediv_impl(other.__value, other.__sc_fp8, other.__sc_fp32, self.__value, self.__sc_fp8, self.__sc_fp32, self.__tgrid)
        return QArrayImpl(x_out, self.__tgrid)

    def __floordiv__(self, other):
        other = self._promote(other)
        if self.__tgrid != other.tgrid:
            return QArrayImpl(self.astype(jnp.float32) // other.astype(jnp.float32), self.__tgrid)
        x_out = floordiv_impl(self.__value, self.__sc_fp8, self.__sc_fp32, other.__value, other.__sc_fp8, other.__sc_fp32, self.__tgrid)
        return QArrayImpl(x_out, self.__tgrid)

    def __rfloordiv__(self, other):
        other = self._promote(other)
        if self.__tgrid != other.tgrid:
            return QArrayImpl(other.astype(jnp.float32) // self.astype(jnp.float32), self.__tgrid)
        x_out = floordiv_impl(other.__value, other.__sc_fp8, other.__sc_fp32, self.__value, self.__sc_fp8, self.__sc_fp32, self.__tgrid)
        return QArrayImpl(x_out, self.__tgrid)

    def __getitem__(self, idx):
        # Fast path for unquantized arrays (like the output of qx.silu)
        if not hasattr(self, '_QArrayImpl__tgrid'):
            return QArrayImpl(self.get_value()[idx], quant=False)
            
        # Dequantize using the proper custom_vjp path, then slice
        x_fp32 = self.dequantize(jnp.float32)
        sliced_x = x_fp32[idx]
        
        old_a, old_b = self.__tgrid
        
        if sliced_x.ndim == 0:
            H, W = 1, 1
        elif sliced_x.ndim == 1:
            H, W = 1, sliced_x.shape[0]
        else:
            H, W = sliced_x.shape[-2:]
            
        # Adapt tgrid to the new spatial dimensions to avoid non-divisible crashes!
        new_grid = (math.gcd(old_a, H), math.gcd(old_b, W))
        
        return QArrayImpl(sliced_x, new_grid)

    def __neg__(self):
        new_obj = QArrayImpl.__new__(QArrayImpl)
        new_obj._QArrayImpl__tgrid = self.__tgrid
        new_obj._QArrayImpl__value = self.__value
        new_obj._QArrayImpl__sc_fp8 = self.__sc_fp8
        new_obj._QArrayImpl__sc_fp32 = -self.__sc_fp32
        return new_obj

    def apply(self, func):
        fp32_val = self.dequantize(jnp.float32)
        out_val = func(fp32_val)
        
        old_a, old_b = self.__tgrid
        if out_val.ndim == 0:
            H, W = 1, 1
        elif out_val.ndim == 1:
            H, W = 1, out_val.shape[0]
        else:
            H, W = out_val.shape[-2:]
            
        new_grid = (math.gcd(old_a, H), math.gcd(old_b, W))
        return QArrayImpl(out_val, new_grid)

    def __abs__(self):
        new_obj = QArrayImpl.__new__(QArrayImpl)
        new_obj._QArrayImpl__tgrid = self.__tgrid
        new_obj._QArrayImpl__value = jnp.abs(self.__value)
        new_obj._QArrayImpl__sc_fp8 = self.__sc_fp8
        new_obj._QArrayImpl__sc_fp32 = jnp.abs(self.__sc_fp32)
        return new_obj

    def __repr__(self):
        return f'{self.__value}'

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
                        val_fp32 = val.astype(jnp.float32) if isinstance(val, QArrayImpl) else val
                        out_fp32 = self.fp8_array.dequantize(jnp.float32).at[self.idx].set(val_fp32)
                        return QArrayImpl(out_fp32, self.fp8_array.tgrid)
                return _Setter(self.fp8_array, idx)
        return _FP8IndexUpdateHelper(self)

    def astype(self, dtype):
        if not hasattr(self, '_QArrayImpl__tgrid'):
            return self.get_value().astype(dtype)
        return self.dequantize(dtype)

    @property
    def mT(self):
        grid = (self.__tgrid[1], self.__tgrid[0])
        return QArrayImpl(self.astype(jnp.float32).mT, grid)

    @property
    def T(self):
        grid = (self.__tgrid[1], self.__tgrid[0])
        return QArrayImpl(self.astype(jnp.float32).T, grid)

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
        old_a, old_b = self.__tgrid
        if out_val.ndim == 0:
            H, W = 1, 1
        elif out_val.ndim == 1:
            H, W = 1, out_val.shape[0]
        else:
            H, W = out_val.shape[-2:]
        new_grid = (math.gcd(old_a, H), math.gcd(old_b, W))
        return QArrayImpl(out_val, new_grid)


    @classmethod
    def concat(cls, arrays, axis=0):
        fp32_arrays = [a.astype(jnp.float32) if isinstance(a, QArrayImpl) else a for a in arrays]
        out_fp32 = jnp.concat(fp32_arrays, axis=axis)
        
        # Use the tgrid of the first QArrayImpl in the list
        first_fp8 = next((a for a in arrays if isinstance(a, QArrayImpl)), None)
        grid = first_fp8.tgrid if first_fp8 else (1, 128)
        
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
        if hasattr(self, '_QArrayImpl__tgrid'):
            children = (self.__value, self.__sc_fp8, self.__sc_fp32)
            aux_data = (True, self.__tgrid)
        else:
            children = (self.__value,)
            aux_data = (False, None)
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        is_quant, tgrid = aux_data
        obj = cls.__new__(cls)
        if is_quant:
            obj._QArrayImpl__tgrid = tgrid
            obj._QArrayImpl__value, obj._QArrayImpl__sc_fp8, obj._QArrayImpl__sc_fp32 = children
        else:
            obj._QArrayImpl__value = children[0]
        return obj


register_pytree_node(QArrayImpl, QArrayImpl._tree_flatten, QArrayImpl._tree_unflatten)

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
    S_1i_u32 = jax.lax.bitcast_convert_type(S_1i_f32, jnp.uint32)
    S_1i = jax.lax.bitcast_convert_type((S_1i_u32 >> 23).astype(jnp.uint8), jnp.float8_e8m0fnu)
    
    S_1i_f32_q = jax.lax.bitcast_convert_type(jax.lax.bitcast_convert_type(S_1i, jnp.uint8).astype(jnp.uint32) << 23, jnp.float32)
    S_eff = (S_1i_f32_q * M_j).reshape(-1, 1, 1)
    S_eff = jnp.maximum(S_eff, 1e-7)
    
    x_out = (x / S_eff * 256.0).astype(jnp.bfloat16).astype(jnp.float8_e4m3fn)
    
    x_out_ref[...] = x_out
    sc_fp8_ref[...] = S_1i.reshape(1, 1, sc_fp8_ref.shape[-2], sc_fp8_ref.shape[-1])
    sc_fp32_ref[...] = M_j.reshape(1, 1, 1, 1)


import functools

@functools.partial(jax.jit, static_argnames=['tgrid'])
def quantize_impl(x: ArrayLike, tgrid: Tuple = (128, 128)):
    orig_shape = x.shape
    if x.ndim == 1:
        x = x.reshape(1, orig_shape[0])
    elif x.ndim == 0:
        x = x.reshape(1, 1)
        
    shape = x.shape
    a, b = tgrid
    
    # Valid spatial memory layout
    x_reshaped = x.reshape(*shape[:-2], shape[-2] // a, a, shape[-1] // b, b)
    dims = list(range(x_reshaped.ndim))
    x_reshaped = x_reshaped.transpose(*dims[:-4], dims[-4], dims[-2], dims[-3], dims[-1])
    
    spatial_elements = shape[-2] * shape[-1]
    M_blocks = spatial_elements // (a * b)
    
    target_K = max(1, (a * b) // 16)
    K = math.gcd(target_K, M_blocks)
    
    M = M_blocks // K
    
    x_reshaped = x_reshaped.reshape(*shape[:-2], M, K, a, b)
    
    m_i_f32 = jnp.max(jnp.abs(x_reshaped), axis=(-2, -1), keepdims=True).astype(jnp.float32)
    M_j_f32 = jnp.max(m_i_f32, axis=(-3, -2, -1), keepdims=True)
    
    # Avoid division by zero when the entire array is zero
    safe_M_j = jnp.where(M_j_f32 == 0.0, 1.0, M_j_f32)
    S_1i_f32 = m_i_f32 / safe_M_j
    
    # Bitcast trick to avoid TPU f32->f8E8M0 compiler errors
    S_1i_u32 = jax.lax.bitcast_convert_type(S_1i_f32, jnp.uint32)
    S_1i = jax.lax.bitcast_convert_type((S_1i_u32 >> 23).astype(jnp.uint8), jnp.float8_e8m0fnu)
    
    S_1i_f32_q = jax.lax.bitcast_convert_type(jax.lax.bitcast_convert_type(S_1i, jnp.uint8).astype(jnp.uint32) << 23, jnp.float32)
    S_eff = (S_1i_f32_q * M_j_f32).reshape(*shape[:-2], M, K, 1, 1)
    S_eff = jnp.maximum(S_eff, 1e-7)
    
    x_fp8 = (x_reshaped / S_eff * 128.0).astype(jnp.bfloat16).astype(jnp.float8_e4m3fn)
    sc_fp8 = S_1i.reshape(*shape[:-2], M, 1, K, 1)
    sc_fp32 = M_j_f32.reshape(*shape[:-2], M, 1, 1, 1)

    x_fp8 = x_fp8.reshape(*shape[:-2], shape[-2] // a, shape[-1] // b, a, b)
    x_fp8 = x_fp8.transpose(*dims[:-4], dims[-4], dims[-2], dims[-3], dims[-1])

    return x_fp8.reshape(*orig_shape), sc_fp8.reshape(*shape[:-2], M, K), sc_fp32.reshape(*shape[:-2], M, 1, 1)

quantize_impl = jax.custom_vjp(quantize_impl, nondiff_argnums=(1,))

def quantize_impl_fwd(x, tgrid):
    out = quantize_impl(x, tgrid)
    return out, (tgrid, out)

def quantize_impl_bwd(tgrid, res, g_out):
    _, out = res
    x_fp8, sc_fp8, sc_fp32 = out
    g_fp8, g_sc8, g_sc32 = g_out
    
    if g_fp8.dtype != jnp.float8_e4m3fn:
        return g_fp8,
        
    valid_sc8 = jnp.array(1.0, dtype=jnp.float8_e8m0fnu)
    g_sc8 = jnp.where(jnp.isnan(g_sc8), valid_sc8, g_sc8)
    
    g_f32 = dequantize_impl(g_fp8, g_sc8, g_sc32, jnp.float32, tgrid)
    return g_f32,

quantize_impl.defvjp(quantize_impl_fwd, quantize_impl_bwd)


def dequantize_kernel(x_ref, sc_fp8_ref, sc_fp32_ref, o_ref):
    x = x_ref[...]
    sc_fp8 = sc_fp8_ref[...]
    sc_fp32 = sc_fp32_ref[...]
    
    sc_fp8_f32 = jax.lax.bitcast_convert_type(jax.lax.bitcast_convert_type(sc_fp8, jnp.uint8).astype(jnp.uint32) << 23, jnp.float32)
    x_sc = sc_fp8_f32 * sc_fp32
    o_ref[...] = x.astype(jnp.bfloat16).astype(jnp.float32) * x_sc / 256.0


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def dequantize_impl(x_fp8: ArrayLike, sc_fp8: ArrayLike, sc_fp32: ArrayLike, dtype: DTypeLike, tgrid: Tuple):
    orig_shape = x_fp8.shape
    if x_fp8.ndim == 1:
        x_fp8 = x_fp8.reshape(1, orig_shape[0])
    elif x_fp8.ndim == 0:
        x_fp8 = x_fp8.reshape(1, 1)
        
    shape = x_fp8.shape
    M, K = sc_fp8.shape[-2:]
    a, b = tgrid
    
    # Valid spatial memory layout
    x_reshaped = x_fp8.reshape(*shape[:-2], shape[-2] // a, a, shape[-1] // b, b)
    dims = list(range(x_reshaped.ndim))
    x_reshaped = x_reshaped.transpose(*dims[:-4], dims[-4], dims[-2], dims[-3], dims[-1])
    
    x_reshaped = x_reshaped.reshape(*shape[:-2], M, K, a, b)
    sc_fp8_reshaped = sc_fp8.reshape(*shape[:-2], M, K, 1, 1)
    sc_fp32_reshaped = sc_fp32.reshape(*shape[:-2], M, 1, 1, 1)
    
    sc_fp8_f32 = jax.lax.bitcast_convert_type(jax.lax.bitcast_convert_type(sc_fp8_reshaped, jnp.uint8).astype(jnp.uint32) << 23, jnp.float32)
    x_sc = sc_fp8_f32 * sc_fp32_reshaped
    x_out = x_reshaped.astype(jnp.bfloat16).astype(jnp.float32) * x_sc / 128.0

    x_out = x_out.reshape(*shape[:-2], shape[-2] // a, shape[-1] // b, a, b)
    x_out = x_out.transpose(*dims[:-4], dims[-4], dims[-2], dims[-3], dims[-1])

    return x_out.reshape(*orig_shape).astype(dtype)

def dequantize_impl_fwd(x_fp8, sc_fp8, sc_fp32, dtype, tgrid):
    out = dequantize_impl(x_fp8, sc_fp8, sc_fp32, dtype, tgrid)
    return out, (sc_fp8, sc_fp32)

def dequantize_impl_bwd(dtype, tgrid, res, g_out):
    sc_fp8, sc_fp32 = res
    M, K = sc_fp8.shape[-2:]
    a, b = tgrid
    shape = g_out.shape
    
    g_reshaped = g_out.astype(jnp.float32)
    if g_reshaped.ndim == 1:
        g_reshaped = g_reshaped.reshape(1, shape[0])
    elif g_reshaped.ndim == 0:
        g_reshaped = g_reshaped.reshape(1, 1)
        
    g_reshaped = g_reshaped.reshape(*shape[:-2], shape[-2] // a, a, shape[-1] // b, b)
    dims = list(range(g_reshaped.ndim))
    g_reshaped = g_reshaped.transpose(*dims[:-4], dims[-4], dims[-2], dims[-3], dims[-1])
    g_reshaped = g_reshaped.reshape(*shape[:-2], M, K, a, b)
    
    sc_fp8_reshaped = sc_fp8.reshape(*shape[:-2], M, K, 1, 1)
    sc_fp32_reshaped = sc_fp32.reshape(*shape[:-2], M, 1, 1, 1)
    sc_fp8_f32 = jax.lax.bitcast_convert_type(jax.lax.bitcast_convert_type(sc_fp8_reshaped, jnp.uint8).astype(jnp.uint32) << 23, jnp.float32)
    x_sc = sc_fp8_f32 * sc_fp32_reshaped
    
    g_x = (g_reshaped * x_sc / 128.0).reshape(*shape[:-2], shape[-2] // a, shape[-1] // b, a, b)
    g_x = g_x.transpose(*dims[:-4], dims[-4], dims[-2], dims[-3], dims[-1])
    g_x = g_x.reshape(*shape)
    
    return g_x, None, None

dequantize_impl.defvjp(dequantize_impl_fwd, dequantize_impl_bwd)

def matmul_impl(x_fp8, x_sc8, x_sc32, y_fp8, y_sc8, y_sc32, block_grid_x: Tuple, block_grid_y: Tuple):
    x_f32 = dequantize_impl(x_fp8, x_sc8, x_sc32, jnp.float32, block_grid_x)
    y_f32 = dequantize_impl(y_fp8, y_sc8, y_sc32, jnp.float32, block_grid_y)
    return jnp.matmul(x_f32, y_f32)

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
    
    def reduce_broadcast(g, target_shape):
        if g.shape == target_shape:
            return g
        axes = list(range(g.ndim - len(target_shape)))
        for i, (gs, ts) in enumerate(zip(g.shape[len(axes):], target_shape)):
            if ts == 1 and gs > 1:
                axes.append(len(axes) + i)
        g = jnp.sum(g, axis=tuple(axes))
        return jnp.reshape(g, target_shape)

    g_x_f32 = reduce_broadcast(g_x_f32, x_fp8.shape)
    g_y_f32 = reduce_broadcast(g_y_f32, y_fp8.shape)
    
    g_x_fp8, g_x_sc8, g_x_sc32 = quantize_impl(g_x_f32, block_grid_x)
    g_y_fp8, g_y_sc8, g_y_sc32 = quantize_impl(g_y_f32, block_grid_y)
    
    return g_x_fp8, g_x_sc8, g_x_sc32, g_y_fp8, g_y_sc8, g_y_sc32

matmul_impl.defvjp(matmul_impl_fwd, matmul_impl_bwd)


def make_elementwise_impl(op):
    def impl(x_fp8, x_sc8, x_sc32, y_fp8, y_sc8, y_sc32, tgrid):
        x_f32 = dequantize_impl(x_fp8, x_sc8, x_sc32, jnp.float32, tgrid)
        y_f32 = dequantize_impl(y_fp8, y_sc8, y_sc32, jnp.float32, tgrid)
        out_f32 = op(x_f32, y_f32)
        return out_f32
    
    impl = jax.custom_vjp(impl, nondiff_argnums=(6,))
    
    def impl_fwd(x_fp8, x_sc8, x_sc32, y_fp8, y_sc8, y_sc32, tgrid):
        out = impl(x_fp8, x_sc8, x_sc32, y_fp8, y_sc8, y_sc32, tgrid)
        return out, (x_fp8, x_sc8, x_sc32, y_fp8, y_sc8, y_sc32)
        
    def impl_bwd(tgrid, res, g_out):
        x_fp8, x_sc8, x_sc32, y_fp8, y_sc8, y_sc32 = res
        x_f32 = dequantize_impl(x_fp8, x_sc8, x_sc32, jnp.float32, tgrid)
        y_f32 = dequantize_impl(y_fp8, y_sc8, y_sc32, jnp.float32, tgrid)
        
        _, vjp_fn = jax.vjp(op, x_f32, y_f32)
        g_x_f32, g_y_f32 = vjp_fn(g_out)
        
        def reduce_broadcast(g, target_shape):
            if g.shape == target_shape:
                return g
            axes = list(range(g.ndim - len(target_shape)))
            for i, (gs, ts) in enumerate(zip(g.shape[len(axes):], target_shape)):
                if ts == 1 and gs > 1:
                    axes.append(len(axes) + i)
            g = jnp.sum(g, axis=tuple(axes))
            return jnp.reshape(g, target_shape)
            
        g_x_f32 = reduce_broadcast(g_x_f32, x_fp8.shape)
        g_y_f32 = reduce_broadcast(g_y_f32, y_fp8.shape)
        
        g_x_fp8, g_x_sc8, g_x_sc32 = quantize_impl(g_x_f32, tgrid)
        g_y_fp8, g_y_sc8, g_y_sc32 = quantize_impl(g_y_f32, tgrid)
        
        return g_x_fp8, g_x_sc8, g_x_sc32, g_y_fp8, g_y_sc8, g_y_sc32

    impl.defvjp(impl_fwd, impl_bwd)
    return impl

add_impl = make_elementwise_impl(jnp.add)
mul_impl = make_elementwise_impl(jnp.multiply)
sub_impl = make_elementwise_impl(jnp.subtract)
truediv_impl = make_elementwise_impl(jnp.divide)
floordiv_impl = make_elementwise_impl(jnp.floor_divide)


def fp8_matmul(x_f32, y_f32, tgrid=(1, 16)):
    """End-to-End differentiable FP8 Matrix Multiplication"""
    x_fp8 = QArrayImpl(x_f32, tgrid)
    y_fp8 = QArrayImpl(y_f32, tgrid)
    out_fp8 = x_fp8 @ y_fp8
    return out_fp8.dequantize(jnp.float32)

fp8_matmul = jax.custom_vjp(fp8_matmul, nondiff_argnums=(2,))

def fp8_matmul_fwd(x_f32, y_f32, tgrid):
    out = fp8_matmul(x_f32, y_f32, tgrid)
    # Save the float32 inputs for the backward pass
    return out, (x_f32, y_f32)

def fp8_matmul_bwd(tgrid, res, g_out):
    x_f32, y_f32 = res
    
    # Transpose the raw float32 inputs
    y_f32_T = jnp.swapaxes(y_f32, -1, -2)
    x_f32_T = jnp.swapaxes(x_f32, -1, -2)
    
    # Calculate gradients using the same accelerated FP8 matmul!
    g_x = fp8_matmul(g_out, y_f32_T, tgrid)
    g_y = fp8_matmul(x_f32_T, g_out, tgrid)
    
    return g_x, g_y

fp8_matmul.defvjp(fp8_matmul_fwd, fp8_matmul_bwd)





