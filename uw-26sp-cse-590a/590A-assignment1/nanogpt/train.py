import os
import warnings
import logging
import time
from pathlib import Path
from functools import partial

import jax
import optax
import grain
import numpy as np
import jax.numpy as jnp
import orbax.checkpoint as ocp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.sharding import set_mesh


from model import count_params
from model import precompute_frequencies
from model import GPT, forward, set_activation_checkpointing, set_attn_impl
from utils import DP_AXIS_NAME, print_param_info
from optim import build_optimizer
from config import load_config_from_yaml
from fineweb_dataloader import make_grain_shard_loader, BOSFinder


logging.getLogger("absl").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, message=".*CheckpointManager.*")


def compute_loss(params, x_batch, y_batch, segment_ids, freqs, loss_mask):
    logits = forward(params, x_batch, segment_ids, freqs)
    if loss_mask is not None:
        per_token_loss = optax.losses.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=y_batch,
            where=loss_mask,
        )
        return jnp.sum(per_token_loss) / jnp.maximum(jnp.sum(loss_mask), 1.0)
    else:
        return jnp.mean(
            optax.losses.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=y_batch
            )
        )


@partial(
    jax.jit,
    static_argnames=("optim", "grad_accum_steps"),
    donate_argnums=(0, 5),
)
def train_step_accum(
    params, x_batch, y_batch, segment_ids, freqs, optim_state, optim, grad_accum_steps
):
    def body(carry, xy):
        param, opt_state, lsum = carry
        xb, yb = xy
        loss, grad = jax.value_and_grad(compute_loss)(
            param, xb, yb, segment_ids, freqs, None
        )

        # MultiSteps accumulates grad internally and returns a zero-tree update on
        # every micro-step except the last, where it emits the real update.
        updates, new_opt_state = optim.update(grad, opt_state, param)
        new_param = optax.apply_updates(param, updates)
        return (new_param, new_opt_state, lsum + loss), None

    carry0 = (params, optim_state, jnp.array(0.0, dtype=jnp.result_type(0.0)))
    (params, optim_state, lsum), _ = jax.lax.scan(
        body, carry0, (x_batch, y_batch), length=grad_accum_steps
    )
    loss = lsum / grad_accum_steps
    return params, loss, optim_state


@partial(jax.jit, static_argnames=("optim",), donate_argnums=(0, 1, 3, 4, 5))
def train_step(params, x_batch, y_batch, segment_ids, freqs, optim_state, optim):
    loss, grads = jax.value_and_grad(compute_loss)(
        params, x_batch, y_batch, segment_ids, freqs
    )
    updates, optim_state = optim.update(grads, optim_state, params)
    updated_params = optax.apply_updates(params, updates)
    return updated_params, loss, optim_state


@jax.jit
def val_step(params, x_batch, y_batch, segment_ids, freqs):
    loss = compute_loss(params, x_batch, y_batch, segment_ids, freqs, None)
    return loss


def line(label, value, comma=False, label_w=30, colon_w=2, value_w=20):
    fmt = f">{value_w}," if comma else f">{value_w}"
    return f"{label:<{label_w}}{':':<{colon_w}}{value:{fmt}}"


def get_next_batch(
    starts,
    ends,
    bsz,
    seqlen,
    tokens,
    data_sharding,
    buf_u16,
    transfer_to_device=False,
    create_new_buf=False,
):
    """Gathers batches of input-labels pairs.

    Given the `starts` and `ends` of sequences provided by the
    BOSFinder, this method generates batches of inputs-labels
    efficiently.
    """
    if buf_u16 is None and create_new_buf:
        buf_u16 = np.empty((bsz, seqlen + 1), dtype=np.uint16)

    ptr = 0
    for i, j in zip(starts, ends):
        n = j - i
        row = ptr // (seqlen + 1)
        col = ptr % (seqlen + 1)
        buf_u16[row, col : col + n] = tokens[i:j]
        ptr += n

    # If no new array was created
    if not create_new_buf:
        return None
    else:
        if transfer_to_device:
            x = jax.device_put(buf_u16[:, :-1], data_sharding)
            y = jax.device_put(buf_u16[:, 1:], data_sharding)
        else:
            x = buf_u16[:, :-1]
            y = buf_u16[:, 1:]
        return x, y


def build_optim(model, cfg, grad_accum_steps):
    optim = optax.chain(
        optax.clip_by_global_norm(cfg.hparams.clip_grad_norm),
        build_optimizer(
            model,
            d_model=cfg.model.d_emb,
            peak_lr=cfg.hparams.max_lr,
            min_lr=cfg.hparams.min_lr,
            total_train_steps=cfg.hparams.total_train_steps,
            warmup_steps=cfg.hparams.warmup_steps,
            b1=cfg.hparams.b1,
            b2=cfg.hparams.b2,
            embedding_lr=cfg.hparams.embedding_lr,
            unembedding_lr=cfg.hparams.unembedding_lr,
            weight_decay=cfg.hparams.weight_decay,
            cautious_weight_decay=cfg.hparams.cautious_weight_decay,
        ),
    )
    if grad_accum_steps > 1:
        print("Using `MultiSteps` in optax for gradient accumulation...")
        optim = optax.MultiSteps(optim, every_k_schedule=grad_accum_steps)
    optim_state = optim.init(model)
    return optim, optim_state


def build_checkpoint_manager(cfg):
    ckpt_path = Path(cfg.ckpt_cfg.save_ckpt_dir).resolve()
    options = ocp.CheckpointManagerOptions(
        max_to_keep=cfg.ckpt_cfg.max_checkpoints_to_keep,
        save_interval_steps=cfg.ckpt_cfg.checkpoint_save_steps,
        enable_async_checkpointing=True,
        enable_background_delete=False,
    )
    handlers = {
        "params": ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
        "optim_state": ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
        "ds": ocp.Checkpointer(grain.checkpoint.CheckpointHandler()),
    }
    mngr = ocp.CheckpointManager(ckpt_path, handlers, options=options)
    return ckpt_path, mngr


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train a GPT model")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to a YAML configuration file (e.g. configs/default.yaml)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    assert jax.default_backend() == "tpu", (f"Expected TPU backend, got '{jax.default_backend()}'")
    devices = np.array(jax.devices()[:1])
    mesh = Mesh(devices, (DP_AXIS_NAME,))
    cfg = load_config_from_yaml(args.config, mesh=mesh)
    set_attn_impl(cfg.model.attn_impl)
    set_activation_checkpointing(cfg.model.activation_checkpointing)

    # Prepare the data loaders.
    train_files = list(Path(cfg.data_dir).glob("*train*.bin"))
    val_files = list(Path(cfg.data_dir).glob("*val*.bin"))
    num_train_files = len(train_files)
    num_val_files = len(val_files)
    train_dl = make_grain_shard_loader(train_files)
    val_dl = make_grain_shard_loader(val_files)
    train_iter = iter(train_dl)
    print("[Data Loader]: Number of train files found: ", num_train_files)
    print("[Data Loader]: Number of validation files found: ", num_val_files, "\n")

    micro_batch_size = cfg.hparams.micro_batch_size
    step_batch_size = micro_batch_size * mesh.shape[DP_AXIS_NAME]
    global_batch_size = cfg.hparams.global_batch_size
    assert global_batch_size % step_batch_size == 0, "Global batch size must be divisible by step batch size"
    grad_accum_steps = global_batch_size // step_batch_size

    # Shard the data across the data-parallel mesh axis.
    data_sharding = NamedSharding(cfg.mesh, P(DP_AXIS_NAME))
    data_accum_sharding = NamedSharding(mesh, P(None, DP_AXIS_NAME, None))

    seqlen = cfg.model.seqlen
    total_train_steps = cfg.hparams.total_train_steps
    checkpoint_save_steps = cfg.ckpt_cfg.checkpoint_save_steps

    # Model, optimizer, and checkpointing
    print("Building GPT model based on the config...")
    model = GPT.init(jax.random.PRNGKey(0), cfg)
    print("Model built successfully!", model)
    print_param_info(model, mesh)

    optim, optim_state = build_optim(model, cfg, grad_accum_steps)
    ckpt_path, mngr = build_checkpoint_manager(cfg)

    # Resume from checkpoint if available
    resume_from_step = cfg.ckpt_cfg.last_checkpoint_step
    if resume_from_step > 0:
        resume_ckpt_path = os.path.join(
            str(ckpt_path), str(resume_from_step)
        )
        if os.path.exists(resume_ckpt_path):
            from checkpoint_utils import load_checkpoint

            model, optim_state, train_iter = load_checkpoint(
                mngr, resume_from_step, model, optim_state, mesh, train_iter
            )
        else:
            resume_from_step = 0
            print(
                f"Checkpoint path {resume_ckpt_path} not found! Resuming training without restoring checkpoint..."
            )

    # Print the configuration summary
    print("\n" + "-" * 75 + "\n")
    print(line("Number of trainable params", count_params(model), comma=True))
    print(line("Attention implementation", cfg.model.attn_impl))
    print(line("Activation checkpointing", cfg.model.activation_checkpointing))
    print(line("Sequence length per sample", cfg.model.seqlen))
    print(line("Micro batch size", cfg.hparams.micro_batch_size))
    print(line("Global batch size", cfg.hparams.global_batch_size))
    print(line("Grad accumulation steps", grad_accum_steps), "\n")
    print(line("LR (min, max)", str((cfg.hparams.min_lr, cfg.hparams.max_lr))))
    print(line("Warmup steps", cfg.hparams.warmup_steps))
    print(line("Weight decay", cfg.hparams.weight_decay))
    if cfg.profile_cfg.enabled:
        print(f"\n{'[Profiler]':<30}{'enabled':>20}")
        print(line("Profile dir", cfg.profile_cfg.profile_dir))
        print(line("Profile steps", f"[{cfg.profile_cfg.start_step}, {cfg.profile_cfg.end_step})"))
    print("\n" + "-" * 75 + "\n")

    # Compute the frequencies
    positions = jnp.arange(seqlen)[None, :]
    with set_mesh(cfg.mesh):
        head_dim = cfg.model.attn.head_dim
        freqs = precompute_frequencies(positions=positions, features=head_dim)

    segment_ids = None
    best_val_loss = float("inf")
    last_val_loss = float("inf")
    num_shards_used = 0
    total_tokens_consumed = 0

    # Reusable data buffers
    grad_accum_batch = np.zeros((grad_accum_steps, step_batch_size, seqlen + 1), dtype=np.uint16)
    val_data_buf = np.zeros((step_batch_size, seqlen + 1), dtype=np.uint16)

    step = resume_from_step
    profiling_active = False
    profile_dir = str(Path(cfg.profile_cfg.profile_dir).resolve()) if cfg.profile_cfg.enabled else ""
    print("Starting training (the first step will take some time for compilation...)\n")

    training_complete = False
    train_start_time = time.time()

    # Training loop: iterate over data shards
    for shard in train_iter:
        if step >= total_train_steps or training_complete:
            mngr.wait_until_finished()
            print("Finished checkpointing! Cleaned.")
            break

        tokens = shard["tokens"]
        shard_name = Path(shard["path"]).name

        bf = BOSFinder(tokens)
        bf.bos_idx = shard["bos_idx"]
        bf.size = shard["size"]

        num_batches_in_shard = bf.build(step_batch_size, seqlen)
        steps_in_shard = num_batches_in_shard // grad_accum_steps
        print(f"\n=== Processing Shard: {num_shards_used} ({shard_name})", end=" | ")  # fmt: off
        print(f"{num_batches_in_shard} batches, {steps_in_shard} steps ===")

        for _ in range(steps_in_shard):
            if cfg.profile_cfg.enabled and step == cfg.profile_cfg.start_step and not profiling_active:
                jax.profiler.start_trace(profile_dir)
                profiling_active = True
                print(f"[Profiler] Started tracing at step {step}")

            start = time.time()
            for micro_step in range(grad_accum_steps):
                starts, ends = bf.next_batch(step_batch_size, seqlen)
                get_next_batch(
                    starts, ends, step_batch_size, seqlen,
                    tokens, data_accum_sharding,
                    grad_accum_batch[micro_step],
                    transfer_to_device=False,
                )
            stacked_batch = jnp.asarray(
                grad_accum_batch, dtype=jnp.int32, device=data_accum_sharding
            )
            stacked_x = stacked_batch[:, :, :-1]
            stacked_y = stacked_batch[:, :, 1:]
            model, loss, optim_state = train_step_accum(
                model, stacked_x, stacked_y, segment_ids, freqs,
                optim_state, optim, grad_accum_steps,
            )

            jax.block_until_ready(loss)

            if profiling_active and step + 1 >= cfg.profile_cfg.end_step:
                jax.profiler.stop_trace()
                profiling_active = False
                print(f"[Profiler] Stopped tracing at step {step}. Trace saved to {profile_dir}")

            end = time.time()
            dt = end - start
            train_time_elapsed = (end - train_start_time) / 60
            tokens_processed = step_batch_size * seqlen * grad_accum_steps
            total_tokens_consumed += tokens_processed
            tokens_per_sec = int(tokens_processed / dt)

            # fmt: off
            print(f"Step: [{str(step).zfill(len(str(total_train_steps)))}/{total_train_steps}] | loss: {loss:8.4f} | Step time: {dt:5.2f} s | Train time: {train_time_elapsed:6.2f} min | Tokens processed/s: {tokens_per_sec:>9,}")
            # fmt: on

            step += 1

            if (step % checkpoint_save_steps) == 0:
                mngr.save(
                    step,
                    args=ocp.args.Composite(
                        params=ocp.args.PyTreeSave(model),
                        optim_state=ocp.args.PyTreeSave(optim_state),
                        ds=grain.checkpoint.CheckpointSave(train_iter),
                    ),
                )

            if step >= total_train_steps:
                print(f"\nReached maximum training steps  : {total_train_steps}")
                print(f"Total number of shards consumed : {num_shards_used}")
                mngr.wait_until_finished()
                print("Finished checkpointing! Cleaned.")
                training_complete = True
                break

        del tokens

        if training_complete:
            continue

        # Shard exhausted — run validation
        num_shards_used += 1
        print("Shard exhausted")
        print(f"Total shards consumed: {num_shards_used:<5}")
        print(f"Total Tokens consumed: {total_tokens_consumed:>9,}")
        print("-" * 75)

        print("\nScoring model performance on validation data...\n")
        val_loss = 0.0
        val_steps_count = 0
        for val_shard in iter(val_dl):
            val_tokens = val_shard["tokens"]
            val_bf = BOSFinder(val_tokens)
            val_bf.bos_idx = val_shard["bos_idx"]
            val_bf.size = val_shard["size"]

            num_val_batches = val_bf.build(step_batch_size, seqlen)
            if num_val_batches <= 0:
                del val_tokens
                continue

            for _ in range(num_val_batches):
                starts, ends = val_bf.next_batch(step_batch_size, seqlen)
                get_next_batch(
                    starts, ends, step_batch_size, seqlen,
                    val_tokens, data_sharding, val_data_buf,
                )
                curr_val_data = jnp.asarray(
                    val_data_buf, dtype=jnp.int32, device=data_sharding
                )
                x = curr_val_data[:, :-1]
                y = curr_val_data[:, 1:]
                loss = val_step(model, x, y, segment_ids, freqs)
                val_loss += loss.item()
                val_steps_count += 1
            del val_tokens

        avg_val_loss = val_loss / val_steps_count
        avg_val_loss = jax.block_until_ready(avg_val_loss)
        best_val_loss = min(best_val_loss, avg_val_loss)

        print(f"last_val_loss : {last_val_loss:.4f}")
        print(f"curr_val_loss : {avg_val_loss:.4f}")
        print(f"best_val_loss : {best_val_loss:.4f}\n")
        last_val_loss = avg_val_loss

    train_end_time = time.time()
    print(
        f"\nTotal time taken to train the model: {(train_end_time - train_start_time) / 60:.2f} minutes"
    )
