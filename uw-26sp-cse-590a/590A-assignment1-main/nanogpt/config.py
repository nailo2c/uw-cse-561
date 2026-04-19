import jax
import jax.numpy as jnp
import yaml
import dataclasses
from pathlib import Path
from typing import Callable, Optional, Tuple
from jax.sharding import Mesh
from utils import jax_pytree_struct


def init_uniform(scale=1.0):
    def kernel_init(key, shape, dtype):
        return jax.random.uniform(key, shape, dtype, minval=-scale, maxval=scale)

    return kernel_init


@dataclasses.dataclass
class EmbeddingConfig:
    dtype: jnp.dtype = jnp.bfloat16
    vocab_size: int = 50304
    d_emb: int = 768
    weight_initializer: Callable = dataclasses.field(init=False)
    weight_logical_axes: Tuple[str, str] = ("embed_in", "embed_out")

    def __post_init__(self):
        self.weight_initializer = jax.nn.initializers.normal(stddev=1.0)


@dataclasses.dataclass
class GroupedQueryAttentionConfig:
    dtype: jnp.dtype = jnp.bfloat16
    d_emb: int = 768
    q_heads: int = 8
    kv_heads: int = 4
    head_dim: int = dataclasses.field(init=False)

    wq_initializer: Callable = dataclasses.field(init=False)
    wk_initializer: Callable = dataclasses.field(init=False)
    wv_initializer: Callable = dataclasses.field(init=False)
    wo_initializer: Callable = dataclasses.field(init=False)

    wq_logical_axes: Tuple[str, str, str] = ("attn_wqkv_in", "attn_q_heads", "attn_head_dim")
    wk_logical_axes: Tuple[str, str, str] = ("attn_wqkv_in", "attn_kv_heads", "attn_head_dim")
    wv_logical_axes: Tuple[str, str, str] = ("attn_wqkv_in", "attn_kv_heads", "attn_head_dim")
    wo_logical_axes: Tuple[str, str, str] = ("attn_wo_in", "attn_head_dim", "attn_wo_out")

    def __post_init__(self):
        self.head_dim = self.d_emb // self.q_heads
        self.wq_initializer = init_uniform(scale=3**0.5 * self.d_emb**-0.5)
        self.wk_initializer = init_uniform(scale=3**0.5 * self.d_emb**-0.5)
        self.wv_initializer = init_uniform(scale=3**0.5 * self.d_emb**-0.5)
        self.wo_initializer = jax.nn.initializers.zeros


@dataclasses.dataclass
class LinearConfig:
    dtype: jnp.dtype = jnp.bfloat16
    in_features: int = 768
    out_features: int = 50304
    use_bias: bool = False
    weight_initializer: Callable = None
    weight_logical_axes: Tuple[str, str] = ("linear_in", "linear_out")


@dataclasses.dataclass
class MLPConfig:
    d_emb: int = 768
    mlp_hidden_dim: Optional[int] = None
    dtype: jnp.dtype = jnp.bfloat16
    fc1: LinearConfig = dataclasses.field(init=False)
    fc2: LinearConfig = dataclasses.field(init=False)

    def __post_init__(self):
        hidden_dim = self.mlp_hidden_dim or (self.d_emb * 4)
        if hidden_dim <= 0:
            raise ValueError(f"mlp_hidden_dim must be positive, got {hidden_dim}")
        self.mlp_hidden_dim = hidden_dim
        self.fc1 = LinearConfig(
            dtype=self.dtype,
            in_features=self.d_emb,
            out_features=self.mlp_hidden_dim,
            weight_initializer=init_uniform(scale=3**0.5 * self.d_emb**-0.5),
            weight_logical_axes=("mlp_fc1_in", "mlp_fc1_out"),
        )
        self.fc2 = LinearConfig(
            dtype=self.dtype,
            in_features=self.mlp_hidden_dim,
            out_features=self.d_emb,
            weight_initializer=jax.nn.initializers.zeros,
            weight_logical_axes=("mlp_fc2_in", "mlp_fc2_out"),
        )


@dataclasses.dataclass
class ModelConfig:
    seqlen: int = 2048
    vocab_size: int = 50304
    d_emb: int = 768
    mlp_hidden_dim: Optional[int] = None
    num_layers: int = 16
    q_heads: int = 8
    kv_heads: int = 4
    attn_impl: str = "flash_attn"
    activation_checkpointing: bool = True
    dtype: jnp.dtype = jnp.bfloat16

    embed: EmbeddingConfig = dataclasses.field(init=False)
    mlp: MLPConfig = dataclasses.field(init=False)
    lm_head: LinearConfig = dataclasses.field(init=False)
    attn: GroupedQueryAttentionConfig = dataclasses.field(init=False)

    def __post_init__(self):
        hidden_dim = self.mlp_hidden_dim or (self.d_emb * 4)
        if hidden_dim <= 0:
            raise ValueError(f"mlp_hidden_dim must be positive, got {hidden_dim}")
        self.mlp_hidden_dim = hidden_dim
        self.embed = EmbeddingConfig(
            dtype=self.dtype,
            vocab_size=self.vocab_size,
            d_emb=self.d_emb,
        )
        self.attn = GroupedQueryAttentionConfig(
            dtype=self.dtype,
            d_emb=self.d_emb,
            q_heads=self.q_heads,
            kv_heads=self.kv_heads,
        )
        self.mlp = MLPConfig(
            dtype=self.dtype,
            d_emb=self.d_emb,
            mlp_hidden_dim=self.mlp_hidden_dim,
        )
        self.lm_head = LinearConfig(
            dtype=self.dtype,
            in_features=self.d_emb,
            out_features=self.vocab_size,
            weight_initializer=jax.nn.initializers.normal(stddev=0.001),
        )


@dataclasses.dataclass
class ProfileConfig:
    enabled: bool = False
    profile_dir: str = "profiles/"
    start_step: int = 5
    end_step: int = 10


@dataclasses.dataclass
class CheckpointConfig:
    # Checkpoint related
    max_checkpoints_to_keep: int = 5
    checkpoint_save_steps: int = 100
    last_checkpoint_step: int = 0
    # Directory where checkpoints will be saved
    save_ckpt_dir: Path | str = ""
    # Path to params subdirectory within a checkpoint from which weights will be loaded
    load_params_ckpt_path: Path | str = ""


@dataclasses.dataclass
class HyperParams:
    # Batch size related
    micro_batch_size: int = 32
    global_batch_size: int = 256

    # Optimizer related
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    embedding_lr: float = 0.2
    unembedding_lr: float = 0.004
    b1: float = 0.8
    b2: float = 0.95
    weight_decay: float = 0.0
    cautious_weight_decay: float = 0.01
    clip_grad_norm: float = 1.0
    total_train_steps: int = 10000
    warmup_steps: int = int(min(300, 0.01 * total_train_steps))  # ~10% of total steps


DTYPE_MAP = {
    "float32": jnp.float32,
    "float16": jnp.float16,
    "bfloat16": jnp.bfloat16,
}


def _dtype_from_str(s: str) -> jnp.dtype:
    if s not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{s}'. Choose from {list(DTYPE_MAP)}")
    return DTYPE_MAP[s]


def _build_model_config(d: dict) -> "ModelConfig":
    d = dict(d)
    if "dtype" in d:
        d["dtype"] = _dtype_from_str(d["dtype"])
    return ModelConfig(**d)


def _build_hparams(d: dict) -> "HyperParams":
    d = dict(d)
    return HyperParams(**d)


def _build_checkpoint_config(d: dict) -> "CheckpointConfig":
    d = dict(d)
    return CheckpointConfig(**d)


def _build_profile_config(d: dict) -> "ProfileConfig":
    d = dict(d)
    return ProfileConfig(**d)


def load_config_from_yaml(
    path: str | Path,
    *,
    mesh: Mesh = None,
    seed: jax.Array = None,
) -> "Config":
    """Build a Config from a YAML file.

    The YAML file is expected to have top-level keys matching the Config
    sub-configs: ``model``, ``hparams``, ``checkpoint``, and optionally
    ``data_dir``. Runtime objects (mesh, seed) are passed as
    keyword arguments since they cannot be serialized.
    """
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    model_cfg = _build_model_config(raw.get("model", {}))
    hparams = _build_hparams(raw.get("hparams", {}))
    ckpt_cfg = _build_checkpoint_config(raw.get("checkpoint", {}))
    profile_cfg = _build_profile_config(raw.get("profile", {}))
    data_dir = raw.get("data_dir", "")

    return Config(
        seed=seed,
        mesh=mesh,
        model=model_cfg,
        hparams=hparams,
        ckpt_cfg=ckpt_cfg,
        profile_cfg=profile_cfg,
        data_dir=data_dir,
    )


@jax_pytree_struct
class Config:
    seed: jax.Array = None
    mesh: Mesh = None
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    hparams: HyperParams = dataclasses.field(default_factory=HyperParams)
    ckpt_cfg: CheckpointConfig = dataclasses.field(default_factory=CheckpointConfig)
    profile_cfg: ProfileConfig = dataclasses.field(default_factory=ProfileConfig)
    data_dir: Path | str = ""
