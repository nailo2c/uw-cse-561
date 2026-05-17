"""Fully Sharded Data Parallel (FSDP) for JAX.

Each parameter is sharded along its first dimension across the data-parallel
axis. Before computing a transformer block, all parameters in that block are
all-gathered (unshard) so the forward runs on full replicated weights. After
the block, the gathered copies(i.e. temporary buffers) go dead while the original 
parameter tree remains sharded. During the backward pass XLA re-gathers parameters 
as needed for gradient computation.
"""

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from model import (
    block_forward,
    compute_segment_mask,
    embedding_forward,
    linear_forward,
    rmsnorm_forward,
)
from utils import DP_AXIS_NAME


def _fsdp_spec(ndim):
    """Shard first dimension across FSDP devices, replicate the rest."""
    if ndim == 0:
        return P()
    return P(DP_AXIS_NAME, *((None,) * (ndim - 1)))


def _replicated_spec(ndim):
    return P(*((None,) * ndim))


@jax.custom_vjp
def _ordered_identity(after, value):
    """Return `value`, but keep it ordered after `after` in compiled code."""
    _, value = jax.lax.optimization_barrier((after, value))
    return value


def _ordered_identity_fwd(after, value):
    after = jax.lax.stop_gradient(after)
    _, value = jax.lax.optimization_barrier((after, value))
    return value, None


def _ordered_identity_bwd(_, g):
    return (None, g)


_ordered_identity.defvjp(_ordered_identity_fwd, _ordered_identity_bwd)


def shard_params(params, mesh):
    """Initial FSDP sharding: place each parameter sharded along dim-0."""
    def _shard(x):
        if x is None:
            return None
        return jax.device_put(x, NamedSharding(mesh, _fsdp_spec(x.ndim)))
    return jax.tree_util.tree_map(_shard, params)


def unshard(params, mesh, after=None):
    """All-gather: replicate sharded params across all FSDP devices.

    When `after` is provided, the all-gather is ordered after that value. This
    prevents XLA from hoisting all block gathers to the start of the program.
    """
    if after is not None:
        params = _ordered_identity(after, params)

    def _replicate(x):
        if x is None:
            return None
        return jax.lax.with_sharding_constraint(
            x, NamedSharding(mesh, _replicated_spec(x.ndim))
        )
    return jax.tree_util.tree_map(_replicate, params)


def make_fsdp_forward(mesh):
    """Return a forward function (same signature as model.forward) that
    unshards each transformer block's parameters before its computation.

    Granularity: one transformer block = one unshard/compute/free cycle.
    Embedding and lm_head are also unsharded for their respective ops.
    But the compiler determines the actual calling order.
    """

    @jax.checkpoint
    def _checkpointed_block(block, x, mask, freqs, i):
        with jax.named_scope(f"fsdp_block_{i}"):
            with jax.named_scope("unshard"):
                full_block = unshard(block, mesh, after=x)
            with jax.named_scope("compute"):
                x = block_forward(full_block, x, mask, freqs)
        return x

    def fsdp_forward(params, x, segment_ids, freqs):
        if segment_ids is not None:
            with jax.named_scope("compute_mask"):
                mask = compute_segment_mask(segment_ids)
        else:
            mask = None

        with jax.named_scope("embedding"):
            full_embed = unshard(params.embed, mesh, after=x)
            x = embedding_forward(full_embed, x)

        for i, block in enumerate(params.blocks):
            x = _checkpointed_block(block, x, mask, freqs, i)

        with jax.named_scope("norm"):
            x = rmsnorm_forward(x)

        with jax.named_scope("unembed"):
            full_lm_head = unshard(params.lm_head, mesh, after=x)
            logits = linear_forward(full_lm_head, x)

        with jax.named_scope("logit_soft_capping"):
            logits = logits.astype(jnp.float32)
            logits = 15.0 * jnp.tanh(logits / 15.0)

        return logits

    return fsdp_forward
