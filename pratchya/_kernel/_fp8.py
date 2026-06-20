
from jax.typing import ArrayLike, DTypeLike

import jax.numpy as jnp, jax
import math
from typing import Tuple


FP8E4M3_MIN = jnp.finfo(jnp.float8_e4m3fn).smallest_subnormal
FP8E8M0_MIN = jnp.finfo(jnp.float8_e8m0fnu).smallest_subnormal
FP32_MIN = jnp.finfo(jnp.float32).smallest_subnormal

IS_CPU = jax.default_backend() == "cpu"
IS_TPU = jax.default_backend() == "tpu"
INTERPRET = IS_CPU

if IS_TPU:
    from jax.experimental.pallas import tpu as pltpu
    MEM_SPACE = pltpu.VMEM
else:
    MEM_SPACE = None


class ArrayFP8:

    def __init__(self, x: ArrayLike, block_grid=(1, 16)):
        assert len(block_grid) == 2
        assert jnp.size(x) % math.prod(block_grid)**2 == 0
        self.__block_grid = block_grid
        self.__value, self.__sc_fp8, self.__sc_fp32 = self.__quantize(x)

    def get_value(self):
        return self.__value

    def __quantize(self, x: ArrayLike):
        x, sc_fp8, sc_fp32 = quantize_impl(x, self.__block_grid)
        return x, sc_fp8, sc_fp32

    def dequantize(self, dtype):
        x = dequantize_impl(self.__value, self.__sc_fp8, self.__sc_fp32, dtype=dtype)
        return x
    
    @property
    def shape(self):
        return self.__value.shape
    
    @property
    def size(self):
        return self.__value.size
    
    def __matmul__(self, other: 'ArrayFP8'):
        assert type(other) == ArrayFP8
        x_out, block_grid = matmul_impl(self.__value, self.__sc_fp8, self.__sc_fp32, other.__value, other.__sc_fp8, other.__sc_fp32)
        return ArrayFP8(x_out, block_grid)
    
    def __imatmul__(self, other):
        pass

    def __add__(self, other):
        pass

    def __iadd__(self, other):
        pass
    
    def __repr__(self):
        return f'{self.__value}'

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


from jax.tree_util import register_pytree_node
register_pytree_node(ArrayFP8, ArrayFP8._tree_flatten, ArrayFP8._tree_unflatten)

from jax.experimental import pallas as pl


def quantize_kernel(x_ref, x_out_ref, sc_fp8_ref, sc_fp32_ref):
    x = x_ref[...]
    a = x.shape[-2]
    b = x.shape[-1]
    
    m_i = jnp.max(jnp.abs(x), axis=(-2, -1), keepdims=True)
    m_i = jnp.maximum(m_i, FP8E8M0_MIN.astype(m_i.dtype))
    
    m_i_tmp = m_i.reshape(1, a, b)
    
    M_j = jnp.max(m_i_tmp, axis=(-2, -1), keepdims=True)
    M_j = jnp.maximum(M_j, FP32_MIN.astype(M_j.dtype))
    
    S_1i = (m_i_tmp / M_j).astype(jnp.float8_e8m0fnu)
    S_eff = (S_1i.astype(M_j.dtype) * M_j).reshape(-1, 1, 1)
    
    x_out = (x / S_eff * 256.0).astype(jnp.float8_e4m3fn)
    
    x_out_ref[...] = x_out
    sc_fp8_ref[...] = jax.lax.bitcast_convert_type(S_1i, jnp.uint8).reshape(1, 1, a, b)
    sc_fp32_ref[...] = M_j.reshape(1, 1, 1, 1)


def quantize_impl(x: ArrayLike, block_grid: Tuple = (128, 128)):
    assert jnp.size(x) % math.prod(block_grid)**2 == 0, f"Array size {jnp.size(x)} is not divisible by super-block size {math.prod(block_grid)**2}"
    shape = x.shape
    a, b = block_grid
    
    # Valid spatial memory layout
    x_reshaped = x.reshape(*shape[:-2], shape[-2] // a, a, shape[-1] // b, b)
    dims = list(range(x_reshaped.ndim))
    x_reshaped = x_reshaped.transpose(*dims[:-4], dims[-4], dims[-2], dims[-3], dims[-1])
    
    total_elements = math.prod(shape)
    M = total_elements // (a * a * b * b)
    
    x_reshaped = x_reshaped.reshape(M, a*b, a, b)
    
    x_fp8, sc_fp8, sc_fp32 = pl.pallas_call(
        quantize_kernel,
        out_shape=[
            jax.ShapeDtypeStruct(x_reshaped.shape, dtype=jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((M, 1, a, b), dtype=jnp.uint8),
            jax.ShapeDtypeStruct((M, 1, 1, 1), dtype=jnp.float32),
        ],
        in_specs=[
            pl.BlockSpec((1, a*b, a, b), lambda i: (i, 0, 0, 0), memory_space=MEM_SPACE)
        ],
        out_specs=[
            pl.BlockSpec((1, a*b, a, b), lambda i: (i, 0, 0, 0), memory_space=MEM_SPACE),
            pl.BlockSpec((1, 1, a, b), lambda i: (i, 0, 0, 0), memory_space=MEM_SPACE),
            pl.BlockSpec((1, 1, 1, 1), lambda i: (i, 0, 0, 0), memory_space=MEM_SPACE),
        ],
        interpret=INTERPRET,
        grid=(M,)
    )(x_reshaped)

    x_fp8 = x_fp8.reshape(*shape[:-2], shape[-2] // a, shape[-1] // b, a, b)
    x_fp8 = x_fp8.transpose(*dims[:-4], dims[-4], dims[-2], dims[-3], dims[-1])

    return x_fp8.reshape(*shape), sc_fp8.reshape(M, a, b), sc_fp32.reshape(M, 1, 1)


def dequantize_kernel(x_ref, sc_fp8_ref, sc_fp32_ref, o_ref):
    x = x_ref[...]
    sc_fp8 = sc_fp8_ref[...]
    sc_fp32 = sc_fp32_ref[...]
    
    bias = 127.0
    x_sc = jnp.exp2(sc_fp8.astype(jnp.float32) - bias)
    x_sc = (x_sc * sc_fp32).reshape(-1, 1, 1)
    
    o_ref[...] = x.astype(jnp.float32) * x_sc / 256.0


def dequantize_impl(x_fp8: ArrayLike, sc_fp8: ArrayLike, sc_fp32: ArrayLike, dtype: DTypeLike):
    shape = x_fp8.shape
    M, a, b = sc_fp8.shape
    
    assert jnp.size(x_fp8) % (a * a * b * b) == 0, f"Array size {jnp.size(x_fp8)} is not divisible by super-block size {a * a * b * b}"
    
    # Valid spatial memory layout
    x_reshaped = x_fp8.reshape(*shape[:-2], shape[-2] // a, a, shape[-1] // b, b)
    dims = list(range(x_reshaped.ndim))
    x_reshaped = x_reshaped.transpose(*dims[:-4], dims[-4], dims[-2], dims[-3], dims[-1])
    
    x_reshaped = x_reshaped.reshape(M, a*b, a, b)
    sc_fp8_reshaped = sc_fp8.reshape(M, 1, a, b)
    sc_fp32_reshaped = sc_fp32.reshape(M, 1, 1, 1)
    
    x_out = pl.pallas_call(
        dequantize_kernel,
        out_shape=jax.ShapeDtypeStruct(x_reshaped.shape, dtype=jnp.float32),
        in_specs=[
            pl.BlockSpec((1, a*b, a, b), lambda i: (i, 0, 0, 0), memory_space=MEM_SPACE),
            pl.BlockSpec((1, 1, a, b), lambda i: (i, 0, 0, 0), memory_space=MEM_SPACE),
            pl.BlockSpec((1, 1, 1, 1), lambda i: (i, 0, 0, 0), memory_space=MEM_SPACE),
        ],
        out_specs=pl.BlockSpec((1, a*b, a, b), lambda i: (i, 0, 0, 0), memory_space=MEM_SPACE),
        interpret=INTERPRET,
        grid=(M,)
    )(x_reshaped, sc_fp8_reshaped, sc_fp32_reshaped)

    x_out = x_out.reshape(*shape[:-2], shape[-2] // a, shape[-1] // b, a, b)
    x_out = x_out.transpose(*dims[:-4], dims[-4], dims[-2], dims[-3], dims[-1])

    return x_out.reshape(*shape).astype(dtype)

import functools

def matmul_kernel(x_ref, y_ref, sc_x8_ref, sc_x32_ref, sc_y8_ref, sc_y32_ref, o_ref, *, b):
    acc = jnp.zeros((x_ref.shape[0], y_ref.shape[1]), dtype=jnp.float32)
    def body(k, acc):
        x_block = jax.lax.dynamic_slice(x_ref[...], (0, k*b), (x_ref.shape[0], b))
        y_block = jax.lax.dynamic_slice(y_ref[...], (k*b, 0), (b, y_ref.shape[1]))
        sc_x8 = jax.lax.dynamic_slice(sc_x8_ref[...], (0, k), (1, 1))[0, 0]
        sc_x32 = jax.lax.dynamic_slice(sc_x32_ref[...], (0, k), (1, 1))[0, 0]
        sc_y8 = jax.lax.dynamic_slice(sc_y8_ref[...], (k, 0), (1, 1))[0, 0]
        sc_y32 = jax.lax.dynamic_slice(sc_y32_ref[...], (k, 0), (1, 1))[0, 0]
        bias = 127.0
        scale_x = jnp.exp2(sc_x8.astype(jnp.float32) - bias) * sc_x32
        scale_y = jnp.exp2(sc_y8.astype(jnp.float32) - bias) * sc_y32
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

def matmul_impl(x_fp8, x_sc8, x_sc32, y_fp8, y_sc8, y_sc32):
    H_a, W_a = x_fp8.shape[-2:]
    H_b, W_b = y_fp8.shape[-2:]
    M_x, a, b = x_sc8.shape
    M_y, b2, c = y_sc8.shape
    assert b == b2
    orig_batch_x = x_fp8.shape[:-2]
    orig_batch_y = y_fp8.shape[:-2]
    batch_shape = jnp.broadcast_shapes(orig_batch_x, orig_batch_y)
    x_fp8 = jnp.broadcast_to(x_fp8, (*batch_shape, H_a, W_a))
    y_fp8 = jnp.broadcast_to(y_fp8, (*batch_shape, H_b, W_b))
    sc_x8 = x_sc8.reshape(*orig_batch_x, H_a // a, W_a // b)
    sc_x8 = jnp.broadcast_to(sc_x8, (*batch_shape, H_a // a, W_a // b))
    sc_x32 = jnp.repeat(x_sc32.reshape(-1), a * b).reshape(*orig_batch_x, H_a // a, W_a // b)
    sc_x32 = jnp.broadcast_to(sc_x32, (*batch_shape, H_a // a, W_a // b))
    sc_y8 = y_sc8.reshape(*orig_batch_y, H_b // b, W_b // c)
    sc_y8 = jnp.broadcast_to(sc_y8, (*batch_shape, H_b // b, W_b // c))
    sc_y32 = jnp.repeat(y_sc32.reshape(-1), b * c).reshape(*orig_batch_y, H_b // b, W_b // c)
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
    return out.reshape(*batch_shape, H_a, W_b), (a, c)
