import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def silu_kernel(x_ref, o_ref):
    x = x_ref[...]  # equal to load the entire [64, 8192] tile
    o_ref[...] = x / (1 + jnp.exp(-x))

def silu_pallas(x, block_m=64):
    m, n = x.shape
    assert m % block_m == 0, "rows must divide block_m"

    return pl.pallas_call(
        silu_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(m // block_m,),
        in_specs=[
            pl.BlockSpec((block_m, n), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((block_m, n), lambda i: (i, 0)),
    )(x)
    
