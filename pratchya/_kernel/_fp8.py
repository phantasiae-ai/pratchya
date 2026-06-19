
from jax.typing import ArrayLike, DTypeLike

import jax.numpy as jnp
import math

FP8E4M3_MIN = jnp.finfo(jnp.float8_e4m3fn).smallest_subnormal
FP8E8M0_MIN = jnp.finfo(jnp.float8_e8m0fnu).smallest_subnormal
FP32_MIN = jnp.finfo(jnp.float32).smallest_subnormal

class ArrayFP8:

    def __init__(self, x: ArrayLike, block_grid=(1, 16)):
        assert len(block_grid) == 2
        assert jnp.size(x) % math.prod(block_grid)**2 == 0
        self.x_fp8, self.sc_fp8e8, self.sc_fp32 = self.quantize(x, block_grid)

    def quantize(self, x: ArrayLike, block_grid):
        a, b = block_grid
        shape = x.shape
        x = x.reshape(-1, a, b)

        m_i = jnp.max(jnp.abs(x), axis=(-2, -1), keepdims=True)
        m_i = jnp.maximum(m_i, FP8E8M0_MIN.astype(m_i.dtype))

        m_i_tmp = m_i.reshape(-1, a, b)

        M_j = jnp.max(jnp.abs(m_i_tmp), axis=(-2, -1), keepdims=True)
        M_j = jnp.maximum(M_j, FP32_MIN.astype(M_j.dtype))

        S_1i = (m_i_tmp / M_j).astype(jnp.float8_e8m0fnu)
        S_eff = (S_1i.astype(M_j.dtype) * M_j).reshape(-1, 1, 1)

        x = (x / S_eff).astype(jnp.float8_e4m3fn)

        return x.reshape(*shape), S_1i, M_j


    def dequantize(self, dtype):
        C, a, b = self.sc_fp8e8.shape
        
        shape = self.x_fp8.shape
        x = self.x_fp8.reshape(-1, C, a, b)

        x = x.astype(jnp.float32) * self.sc_fp8e8.astype(jnp.float32).reshape(-1, C, 1, 1)

        x = x.reshape(-1, C, a, b) * self.sc_fp32.reshape(-1, C, 1, 1)

        return x.reshape(*shape).astype(dtype)
    
    def __matmul__(self, other: 'ArrayFP8'):
        assert type(other) == ArrayFP8
        assert self.sc_fp8e8.shape[-2:] == other.sc_fp8e8.shape[-2:]
        assert self.x_fp8.shape[-1] == other.x_fp8.shape[-1]
        
        self_shape = self.x_fp8.shape
        other_shape = other.x_fp8.shape
        C, a, b = self.sc_fp8e8.shape
        x, y = self.x_fp8.shape[0], other.x_fp8.shape[0]
        nx, ny = self_shape[-2] // a, self_shape[-1] // b

        x = self.x_fp8.reshape(-1, ny, a, nx, b)
        x_other = other.x_fp8.reshape(-1, ny, a, nx, b)

        print(self.x_fp8.shape)
        print(self.sc_fp8e8.shape)
        print(self.sc_fp32.shape)
        print(a, b, C)    

        x = jnp.einsum('lyaxb, syaxb -> lyxs', x, x_other, preferred_element_type=jnp.float32)

        # dequantize fp8e8m0
        bias = 127.0

        # self.sc_fp8e8: [L1, a, b]
        # self.sc_fp32: [L1, 1, 1] 
        x_sc = jnp.exp2(self.sc_fp8e8.astype(jnp.float32) - bias)
        x_sc_other = jnp.exp2(other.sc_fp8e8.astype(jnp.float32) - bias) ###########################OIPHHPOIHPOHIOHOPHOIHDOHKHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH

        x_sc = x_sc * self.sc_fp32
        x_sc_other = x_sc_other * other.sc_fp32

        # x = x * x_sc * x_sc_other

        # x = jnp.sum(x, axis=-1).reshape(*self_shape[:-1], other_shape[-1])


        return x
    
    def __repr__(self):
        return f'{self.x_fp8}'
    