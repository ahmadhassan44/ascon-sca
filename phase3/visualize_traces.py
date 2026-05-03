"""
Phase 3 — Trace Visualisation
Loads fixed_key_traces.h5 and saves three plots to phase3/plots/.
"""

import os
import sys
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt

H5_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixed_key_traces.h5")
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


def load_data():
    with h5py.File(H5_PATH, "r") as f:
        traces = f["profiling/traces"][:]
        labels = f["profiling/labels"][:]
    return traces, labels


def plot_trace_overlay(traces, labels):
    """10 profiling traces overlaid, coloured by HW label."""
    fig, ax = plt.subplots(figsize=(12, 4))
    cmap = plt.cm.get_cmap("tab10", 9)

    shown = set()
    idx = 0
    while len(shown) < min(10, len(traces)):
        lbl = int(labels[idx])
        color = cmap(lbl)
        label_str = f"HW={lbl}" if lbl not in shown else None
        ax.plot(traces[idx], color=color, alpha=0.7, linewidth=0.8, label=label_str)
        shown.add(lbl)
        idx += 1
        if idx >= len(traces):
            break

    ax.set_title("Power Trace Overlay (10 profiling traces, coloured by HW label)")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Power (a.u.)")
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles, lbls, loc="upper right", fontsize=8, ncol=3)
    path = os.path.join(PLOT_DIR, "trace_overlay.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_mean_trace(traces):
    """Mean trace with ±1 std band."""
    mean = traces.mean(axis=0)
    std  = traces.std(axis=0)
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(mean))
    ax.plot(x, mean, color="steelblue", linewidth=1.2, label="Mean")
    ax.fill_between(x, mean - std, mean + std, alpha=0.3, color="steelblue", label="±1 std")
    ax.set_title(f"Mean Power Trace ± Std Dev  (N={len(traces):,} profiling traces)")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Power (a.u.)")
    ax.legend()
    path = os.path.join(PLOT_DIR, "mean_trace.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_label_dist(labels):
    """Bar chart of HW label distribution (classes 0-8)."""
    classes = np.arange(9)
    counts  = np.array([(labels == c).sum() for c in classes])

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(classes, counts, color="steelblue", edgecolor="white", linewidth=0.5)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                str(cnt), ha="center", va="bottom", fontsize=9)
    ax.set_title("Hamming Weight Label Distribution (profiling set)")
    ax.set_xlabel("HW class (0 – 8)")
    ax.set_ylabel("Count")
    ax.set_xticks(classes)
    path = os.path.join(PLOT_DIR, "label_dist.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    if not os.path.exists(H5_PATH):
        print(f"ERROR: {H5_PATH} not found. Run generate_traces.py first.")
        sys.exit(1)

    os.makedirs(PLOT_DIR, exist_ok=True)

    print(f"Loading {H5_PATH} ...")
    traces, labels = load_data()
    print(f"  traces shape : {traces.shape}")
    print(f"  labels shape : {labels.shape}")

    plot_trace_overlay(traces, labels)
    plot_mean_trace(traces)
    plot_label_dist(labels)

    print("\nAll plots saved.")
