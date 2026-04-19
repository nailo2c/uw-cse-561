import math
import dataclasses

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from utils import layer_repr
from utils import ParamSpec, ParamInitializer
from utils import jax_pytree_struct
from pallas.flash_attention import flash_attention as pallas_flash_attention

ATTN_IMPL = "flash_attn"
ACTIVATION_CHECKPOINTING = True


def linear_init(fan_in, fan_out):
    std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
    return jax.nn.initializers.normal(stddev=std)


def embed_init(std=1.0):
    return jax.nn.initializers.normal(stddev=std)


@jax_pytree_struct
class Linear(ParamInitializer):
    in_features: int = dataclasses.field(metadata=dict(static=True))
    out_features: int = dataclasses.field(metadata=dict(static=True))
    weight: jax.Array | ParamSpec
    bias: jax.Array | ParamSpec
    use_bias: bool = dataclasses.field(default=False, metadata=dict(static=True))

    @classmethod
    def param_specs(cls, cfg):
        weight = ParamSpec(
            shape=(cfg.in_features, cfg.out_features),
            dtype=cfg.dtype,
            logical_axes=cfg.weight_logical_axes,
            initializer=cfg.weight_initializer
            or linear_init(cfg.in_features, cfg.out_features),
        )
        if cfg.use_bias:
            bias = ParamSpec(
                shape=(cfg.out_features,),
                dtype=cfg.dtype,
                logical_axes=cfg.bias_logical_axes,
                initializer=cfg.bias_initializer or jax.nn.initializers.zeros,
            )
        else:
            bias = None
        return Linear(
            weight=weight,
            bias=bias,
            in_features=cfg.in_features,
            out_features=cfg.out_features,
            use_bias=cfg.use_bias,
        )

    @classmethod
    def init(cls, key, mesh, cfg):
        return cls._init_fn(key, mesh, cfg)

    def __repr__(self):
        return layer_repr(self)


@jax_pytree_struct
class GroupedQueryAttention(ParamInitializer):
    wq: jax.Array | ParamSpec
    wk: jax.Array | ParamSpec
    wv: jax.Array | ParamSpec
    wo: jax.Array | ParamSpec
    d_emb: int = dataclasses.field(metadata=dict(static=True))
    q_heads: int = dataclasses.field(metadata=dict(static=True))
    kv_heads: int = dataclasses.field(metadata=dict(static=True))
    head_dim: int = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def param_specs(cls, cfg):
        wq = ParamSpec(
            shape=(cfg.d_emb, cfg.q_heads, cfg.head_dim),
            dtype=cfg.dtype,
            logical_axes=cfg.wq_logical_axes,
            initializer=cfg.wq_initializer or linear_init(cfg.d_emb, cfg.head_dim),
        )
        wk = ParamSpec(
            shape=(cfg.d_emb, cfg.kv_heads, cfg.head_dim),
            dtype=cfg.dtype,
            logical_axes=cfg.wk_logical_axes,
            initializer=cfg.wk_initializer or linear_init(cfg.d_emb, cfg.head_dim),
        )
        wv = ParamSpec(
            shape=(cfg.d_emb, cfg.kv_heads, cfg.head_dim),
            dtype=cfg.dtype,
            logical_axes=cfg.wv_logical_axes,
            initializer=cfg.wv_initializer or linear_init(cfg.d_emb, cfg.head_dim),
        )
        wo = ParamSpec(
            shape=(cfg.q_heads, cfg.head_dim, cfg.d_emb),
            dtype=cfg.dtype,
            logical_axes=cfg.wo_logical_axes,
            initializer=cfg.wo_initializer or linear_init(cfg.head_dim, cfg.d_emb),
        )

        return GroupedQueryAttention(
            d_emb=cfg.d_emb,
            q_heads=cfg.q_heads,
            kv_heads=cfg.kv_heads,
            head_dim=cfg.head_dim,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
        )

    @classmethod
    def init(cls, key, mesh, cfg):
        return cls._init_fn(key, mesh, cfg)

    def __repr__(self):
        return layer_repr(self)


@jax_pytree_struct
class Embedding(ParamInitializer):
    weight: jax.Array | ParamSpec
    vocab_size: int = dataclasses.field(metadata=dict(static=True))
    d_emb: int = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def param_specs(cls, cfg):
        weight = ParamSpec(
            shape=(cfg.vocab_size, cfg.d_emb),
            dtype=cfg.dtype,
            logical_axes=cfg.weight_logical_axes,
            initializer=cfg.weight_initializer or embed_init,
        )
        return Embedding(
            vocab_size=cfg.vocab_size,
            d_emb=cfg.d_emb,
            weight=weight,
        )

    @classmethod
    def init(cls, key, mesh, cfg):
        return cls._init_fn(key, mesh, cfg)

    def __repr__(self):
        return layer_repr(self)


def set_attn_impl(impl: str):
    global ATTN_IMPL
    assert impl in ("flash_attn", "xla"), (
        f"attn_impl must be 'flash_attn' or 'xla', got '{impl}'"
    )
    ATTN_IMPL = impl


def set_activation_checkpointing(enabled: bool):
    global ACTIVATION_CHECKPOINTING
    ACTIVATION_CHECKPOINTING = bool(enabled)


@jax_pytree_struct
class MLP(ParamInitializer):
    fc1: Linear
    fc2: Linear

    @classmethod
    def param_specs(cls, cfg):
        fc1 = Linear.param_specs(cfg.fc1)
        fc2 = Linear.param_specs(cfg.fc2)
        return MLP(fc1=fc1, fc2=fc2)

    def __repr__(self):
        return layer_repr(self)


@jax_pytree_struct
class TransformerBlock(ParamInitializer):
    attn: GroupedQueryAttention
    mlp: MLP

    @classmethod
    def param_specs(cls, cfg):
        attn = GroupedQueryAttention.param_specs(cfg.attn)
        mlp = MLP.param_specs(cfg.mlp)
        return TransformerBlock(attn=attn, mlp=mlp)

    def __repr__(self):
        return layer_repr(self)


@jax_pytree_struct
class GPT(ParamInitializer):
    embed: Embedding
    blocks: list[TransformerBlock]
    lm_head: Linear

    @classmethod
    def param_specs(cls, cfg):
        embed = Embedding.param_specs(cfg.embed)
        blocks = [TransformerBlock.param_specs(cfg) for _ in range(cfg.num_layers)]
        lm_head = Linear.param_specs(cfg.lm_head)
        return GPT(embed=embed, blocks=blocks, lm_head=lm_head)

    @classmethod
    def init(cls, key, cfg):
        return cls._init_fn(key, cfg.mesh, cfg.model)

    def __repr__(self):
        return layer_repr(self)


def count_params(model):
    """Count the parameters in an Equinox model"""
    return sum(x.size for x in jax.tree_util.tree_leaves(model))


def precompute_frequencies(
    positions: jax.Array, features: int, theta=10000.0, dtype=None
):
    """Generate Sin/Cos for Rotary Embeddings."""
    fraction = jnp.arange(0, features, 2, dtype=jnp.float32) / features
    timescale = theta**fraction
    rotational_frequency = 1.0 / timescale
    sinusoid_inp = jnp.einsum(
        "BT,k->BTk",
        positions,
        rotational_frequency,
        precision=jax.lax.Precision.HIGHEST,
        out_sharding=P(None, None, None),
    )
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)
    if dtype is not None:
        sin = sin.astype(dtype)
        cos = cos.astype(dtype)
    return sin, cos


def calculate_rope(x: jax.Array, sin: jax.Array, cos: jax.Array, heads_first=False) -> jax.Array:
    """Apply rotary position embeddings.

    Args:
        x: (B, T, H, D) if heads_first=False, (B, H, T, D) if heads_first=True
        sin, cos: (B, T, D/2)
        heads_first: if True, x has layout (B, H, T, D) and sin/cos broadcast accordingly
    """
    assert x.ndim == 4 and sin.ndim == 3 and cos.ndim == 3
    orig_dtype = x.dtype
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    if heads_first:
        sin, cos = sin[:, None, :, :], cos[:, None, :, :]  # (B, 1, T, D/2)
    else:
        sin, cos = sin[:, :, None, :], cos[:, :, None, :]  # (B, T, 1, D/2)
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(
        orig_dtype
    )


def embedding_forward(params, x):
    return params.weight.at[x, :].get()


def rmsnorm_forward(x, eps=1e-5):
    orig_dtype = x.dtype
    x = x.astype(jnp.float32)
    scale = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    return (x / scale).astype(orig_dtype)


def linear_forward(params, x):
    out = jnp.einsum("...d, dv-> ...v", x, params.weight)
    if params.bias is not None:
        return out + params.bias
    else:
        return out


def mlp_forward(params, x):
    x = linear_forward(params.fc1, x)
    x = jnp.square(jax.nn.relu(x))
    x = linear_forward(params.fc2, x)
    return x


def attn_forward(params, x, mask, freqs):
    orig_dtype = x.dtype
    sin, cos = freqs
    use_flash = ATTN_IMPL == "flash_attn"

    # flash_attn (Pallas) expects (B, H, T, D); XLA expects (B, T, H, D).
    # Produce the right layout directly from the einsum to avoid transposes.
    qkv_einsum = "btd, dhq -> bhtq" if use_flash else "btd, dhq -> bthq"

    with jax.named_scope("qkv_matmul"):
        q = jnp.einsum(qkv_einsum, x, params.wq)
        k = jnp.einsum(qkv_einsum, x, params.wk)
        v = jnp.einsum(qkv_einsum, x, params.wv)

    with jax.named_scope("qk_norm"):
        q = rmsnorm_forward(q)
        k = rmsnorm_forward(k)

    with jax.named_scope("rope"):
        q = calculate_rope(q, sin, cos, heads_first=use_flash)
        k = calculate_rope(k, sin, cos, heads_first=use_flash)

    with jax.named_scope("attention"):
        scale = 1.0 / math.sqrt(q.shape[-1])
        if use_flash:
            # Pallas doesn't support GQA natively; expand KV heads to match Q heads.
            q_heads = q.shape[1]
            kv_heads = k.shape[1]
            if kv_heads != q_heads:
                repeats = q_heads // kv_heads
                # Runtime overhead because pallas flash_attn doesn't support GQA natively.
                k = jnp.repeat(k, repeats, axis=1)
                v = jnp.repeat(v, repeats, axis=1)

            attn = pallas_flash_attention(
                q, k, v, 
                causal=True, sm_scale=scale
            ).astype(orig_dtype)
        else:
            # Materialize the mask tensor: https://docs.jax.dev/en/latest/_autosummary/jax.nn.dot_product_attention.html.
            if mask is not None:
                attn = jax.nn.dot_product_attention(
                    q, k, v, mask=mask, scale=scale,
                    is_causal=True, implementation="xla",
                ).astype(orig_dtype)
            else:
                attn = jax.nn.dot_product_attention(
                    q, k, v, scale=scale,
                    is_causal=True, implementation="xla",
                ).astype(orig_dtype)

    with jax.named_scope("projection"):
        proj_einsum = "bhtq, hqd -> btd" if use_flash else "bthq, hqd -> btd"
        out = jnp.einsum(proj_einsum, attn, params.wo)
    return out


def block_forward(params, x, mask, freqs):
    with jax.named_scope("pre_attn_norm"):
        attn_in = rmsnorm_forward(x)

    attn_out = attn_forward(params.attn, attn_in, mask, freqs)

    with jax.named_scope("residual"):
        x = x + attn_out

    with jax.named_scope("post_attn_norm"):
        ffn_in = rmsnorm_forward(x)

    with jax.named_scope("ffn"):
        ffn_out = mlp_forward(params.mlp, ffn_in)

    with jax.named_scope("residual"):
        x = x + ffn_out
    return x


@jax.checkpoint
def _checkpointed_block_forward(block, x, mask, freqs):
    return block_forward(block, x, mask, freqs)


def compute_segment_mask(segment_ids):
    """Compute once, reuse across all layers. Returns (B, 1, T, S) bias or None."""
    if segment_ids is None:
        return None

    # (B, T) valid token positions (segment_id != 0)
    # We discard padding tokens as valid tokens. 
    valid_segment_ids = jnp.where(segment_ids != 0, 1, 0)

    # (B, T, T) same segment
    same_segment = jnp.equal(segment_ids[:, :, None], segment_ids[:, None, :])

    # (B, T, T) valid on both query and key axes
    valid = valid_segment_ids[:, :, None] & valid_segment_ids[:, None, :]

    mask = (same_segment & valid).astype(jnp.bool)
    return mask[:, None, :, :]  # (B, 1, T, T)


def forward(params, x, segment_ids, freqs):
    if segment_ids is not None:
        with jax.named_scope("compute_mask"):
            mask = compute_segment_mask(segment_ids)
    else:
        mask = None

    with jax.named_scope("embedding"):
        x = embedding_forward(params.embed, x)

    for block in params.blocks:
        if ACTIVATION_CHECKPOINTING:
            x = _checkpointed_block_forward(block, x, mask, freqs)
        else:
            x = block_forward(block, x, mask, freqs)

    with jax.named_scope("norm"):
        x = rmsnorm_forward(x)

    with jax.named_scope("unembed"):
        logits = linear_forward(params.lm_head, x)

    with jax.named_scope("logit_soft_capping"):
        logits = logits.astype(jnp.float32)
        logits = 15.0 * jnp.tanh(logits / 15.0)
    return logits
