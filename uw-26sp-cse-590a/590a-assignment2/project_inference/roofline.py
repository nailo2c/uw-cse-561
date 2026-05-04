"""
Model configs and GEMM shape definitions shared by all bench scripts.

Operational intensity computations and roofline plots have been intentionally
removed from this file — they are part of the student assignment (Section 1).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


# ── Model Configs ───────────────────────────────────────────────────────────
MODELS = {
    "small": {
        "d_emb": 768,
        "mlp_hidden_dim": 3072,
        "num_layers": 12,
        "q_heads": 8,
        "kv_heads": 8,
        "head_dim": 768 // 8,  # 96
    },
    "default": {
        "d_emb": 1536,
        "mlp_hidden_dim": 6144,
        "num_layers": 24,
        "q_heads": 12,
        "kv_heads": 12,
        "head_dim": 1536 // 12,  # 128
    },
}


def get_gemm_shapes(model_cfg):
    """Return the GEMM shapes (K, N) for each layer type in the model.

    For attention:
      QKV projection:  (M, d_emb) @ (d_emb, head_dim * num_heads)
      Out projection:  (M, head_dim * num_heads) @ (head_dim * num_heads, d_emb)
    For MLP:
      fc1 (up):   (M, d_emb) @ (d_emb, mlp_hidden_dim)
      fc2 (down): (M, mlp_hidden_dim) @ (mlp_hidden_dim, d_emb)
    """
    d = model_cfg["d_emb"]
    h = model_cfg["mlp_hidden_dim"]
    q_h = model_cfg["q_heads"]
    kv_h = model_cfg["kv_heads"]
    hd = model_cfg["head_dim"]

    return {
        "QKV proj (Q)":    (d, q_h * hd),
        "QKV proj (KV)":   (d, kv_h * hd),
        "Out proj":        (q_h * hd, d),
        "MLP up (fc1)":    (d, h),
        "MLP down (fc2)":  (h, d),
    }
