"""
Shared plotting utilities for Project 3: Inference Profiling.

Provides consistent styling, roofline plotting, and common chart types.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for servers
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ── Plot style ──────────────────────────────────────────────────────────────
COLORS = {
    "xla": "#1f77b4",
    "pallas": "#ff7f0e",
    "flash_attn": "#ff7f0e",
    "small": "#2ca02c",
    "default": "#d62728",
    "prefill": "#9467bd",
    "decode": "#8c564b",
    "roofline_compute": "#e377c2",
    "roofline_memory": "#7f7f7f",
}

MARKERS = {
    "xla": "o",
    "pallas": "s",
    "flash_attn": "s",
    "small": "^",
    "default": "v",
}


def setup_plot_style():
    """Apply consistent plot style."""
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 2,
        "lines.markersize": 7,
        "grid.alpha": 0.3,
        "axes.grid": True,
    })


def save_plot(fig, name, output_dir="project_inference/plots"):
    """Save figure to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  Plot saved to: {path}")
    plt.close(fig)


def plot_roofline(ax, peak_tflops, peak_bw_gbs, title="TPU v5e Roofline"):
    """Draw a roofline model on the given axes.

    Args:
        ax: matplotlib Axes object.
        peak_tflops: Peak compute throughput in TFLOPS.
        peak_bw_gbs: Peak memory bandwidth in GB/s.
        title: Plot title.
    """
    # Crossover point: operational intensity where compute = bandwidth limit
    # peak_tflops * 1e12 = peak_bw_gbs * 1e9 * OI_cross
    # OI_cross = peak_tflops * 1e3 / peak_bw_gbs  (in FLOP/byte)
    oi_cross = peak_tflops * 1e3 / peak_bw_gbs

    # Plot range
    oi_range = np.logspace(-2, 4, 500)  # FLOP/byte

    # Memory-bound region: performance = BW * OI (in TFLOPS)
    mem_bound = peak_bw_gbs * oi_range / 1e3  # Convert GB/s * FLOP/byte to TFLOPS

    # Compute-bound region: performance = peak_tflops
    perf = np.minimum(mem_bound, peak_tflops)

    ax.loglog(oi_range, perf, 'k-', linewidth=2.5, label="Roofline")
    ax.axhline(y=peak_tflops, color=COLORS["roofline_compute"], linestyle='--',
               alpha=0.7, label=f"Compute ceiling ({peak_tflops} TFLOPS)")
    ax.axvline(x=oi_cross, color=COLORS["roofline_memory"], linestyle=':',
               alpha=0.7, label=f"Ridge point (OI={oi_cross:.1f})")

    ax.set_xlabel("Operational Intensity (FLOP/byte)")
    ax.set_ylabel("Performance (TFLOPS)")
    ax.set_title(title)
    ax.set_xlim(1e-1, 1e4)
    ax.set_ylim(1e-2, peak_tflops * 2)
    ax.legend(loc="lower right")
    ax.grid(True, which="both", alpha=0.3)


def plot_tflops_vs_param(
    ax, param_values, tflops_dict, xlabel, title,
    log_x=True, peak_tflops=None
):
    """Plot TFLOPS vs. a parameter for multiple implementations.

    Args:
        ax: matplotlib Axes.
        param_values: List of x-axis values.
        tflops_dict: dict mapping label -> list of TFLOPS values.
        xlabel: X-axis label.
        title: Plot title.
        log_x: Whether to use log scale on x-axis.
        peak_tflops: If provided, draw a horizontal line for peak.
    """
    for label, values in tflops_dict.items():
        color = COLORS.get(label, None)
        marker = MARKERS.get(label, "o")
        ax.plot(param_values, values, marker=marker, color=color,
                label=label, linewidth=2, markersize=6)

    if peak_tflops:
        ax.axhline(y=peak_tflops, color='red', linestyle='--', alpha=0.5,
                   label=f"Peak ({peak_tflops} TFLOPS)")

    if log_x:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel(xlabel)
    ax.set_ylabel("TFLOPS")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_bandwidth_vs_param(
    ax, param_values, bw_dict, xlabel, title,
    log_x=True, peak_bw_gbs=None
):
    """Plot memory bandwidth utilization vs. a parameter.

    Args:
        ax: matplotlib Axes.
        param_values: List of x-axis values.
        bw_dict: dict mapping label -> list of GB/s values.
        xlabel: X-axis label.
        title: Plot title.
        log_x: Whether to use log scale on x-axis.
        peak_bw_gbs: If provided, draw a horizontal line for peak.
    """
    for label, values in bw_dict.items():
        color = COLORS.get(label, None)
        marker = MARKERS.get(label, "o")
        ax.plot(param_values, values, marker=marker, color=color,
                label=label, linewidth=2, markersize=6)

    if peak_bw_gbs:
        ax.axhline(y=peak_bw_gbs, color='red', linestyle='--', alpha=0.5,
                   label=f"Peak ({peak_bw_gbs} GB/s)")

    if log_x:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Memory Bandwidth (GB/s)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_stacked_bar(ax, categories, breakdown_dict, title, ylabel="Time (ms)"):
    """Plot a stacked bar chart for latency breakdown.

    Args:
        ax: matplotlib Axes.
        categories: List of category labels (x-axis).
        breakdown_dict: OrderedDict mapping component_name -> list of values.
        title: Plot title.
        ylabel: Y-axis label.
    """
    x = np.arange(len(categories))
    width = 0.6
    bottom = np.zeros(len(categories))

    cmap = plt.cm.Set2
    for i, (component, values) in enumerate(breakdown_dict.items()):
        color = cmap(i / max(len(breakdown_dict) - 1, 1))
        ax.bar(x, values, width, bottom=bottom, label=component, color=color)
        bottom += np.array(values)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
