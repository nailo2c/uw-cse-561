import jax
import optax
import jax.numpy as jnp
from jax.tree_util import GetAttrKey, SequenceKey, DictKey


def build_optimizer(
    params,
    *,
    d_model: int,
    peak_lr: float,
    min_lr: float,
    total_train_steps: int,
    warmup_steps: int = 30,
    b1: float = 0.9,
    b2: float = 0.95,
    embedding_lr: float = 0.2,
    unembedding_lr: float = 0.004,
    weight_decay: float = 0.0,
    cautious_weight_decay: float = 0.01,
):
    dmodel_lr_scale = (d_model / 768.0) ** -0.5
    emb_lr = embedding_lr * dmodel_lr_scale
    unemb_lr = unembedding_lr * dmodel_lr_scale

    schedules = {
        "embed": optax.constant_schedule(emb_lr),
        "lm_head": optax.constant_schedule(unemb_lr),
        "other": optax.warmup_cosine_decay_schedule(
            init_value=min_lr,
            peak_value=peak_lr,
            warmup_steps=warmup_steps,
            decay_steps=max(1, total_train_steps - warmup_steps),
            end_value=min_lr,
        ),
    }

    def _path_names(path):
        out = []
        for k in path:
            if isinstance(k, GetAttrKey):
                out.append(k.name)
            elif isinstance(k, SequenceKey):
                out.append(str(k.idx))
            elif isinstance(k, DictKey):
                out.append(str(k.key))
            else:
                out.append(str(k))
        return out

    def label_fn(path, leaf):
        names = _path_names(path)
        top = names[0] if names else ""
        if top == "embed":
            return "embed"
        if top == "lm_head":
            return "lm_head"
        return "other"

    def cautious_decay(schedule, wd):
        def init_fn(params):
            return {"count": jnp.array(0, dtype=jnp.int32)}

        def update_fn(updates, state, params=None):
            step = state["count"]
            scale = schedule(step)
            if params is None:
                return updates, {"count": step + 1}

            def apply_updates(update, param):
                if param is None:
                    return update
                s = getattr(param, "shape", None)
                if s is None or len(s) < 2:
                    return update
                decay = jnp.where(
                    (update * param) >= 0,
                    param.astype(update.dtype),
                    jnp.zeros_like(update),
                )
                return update - (scale * wd) * decay

            new_updates = jax.tree_util.tree_map(apply_updates, updates, params)
            return new_updates, {"count": step + 1}

        return optax.GradientTransformation(init_fn, update_fn)

    param_labels = jax.tree_util.tree_map_with_path(label_fn, params)

    def make_adamw(lr_schedule, weight_decay=0.0):
        return optax.adamw(
            learning_rate=lr_schedule,
            b1=b1,
            b2=b2,
            eps=1e-10,
            weight_decay=weight_decay,
            mu_dtype=jnp.float32,
        )

    tx = optax.multi_transform(
        {
            "embed": make_adamw(schedules["embed"]),
            "lm_head": make_adamw(schedules["lm_head"]),
            "other": optax.chain(
                make_adamw(schedules["other"], weight_decay=weight_decay),
                cautious_decay(schedules["other"], cautious_weight_decay),
            ),
        },
        param_labels,
    )

    return tx
