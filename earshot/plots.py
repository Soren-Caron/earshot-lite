import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np


def training_curves(history: list[dict], path) -> None:
    ep = [h["epoch"] for h in history]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.5))
    a.plot(ep, [h["train_loss"] for h in history])
    a.set(xlabel="epoch", ylabel="train loss")
    b.plot(ep, [h["train_acc"] for h in history], label="train (augmented)")
    b.plot(ep, [h["val_acc"] for h in history], label="val")
    b.set(xlabel="epoch", ylabel="accuracy")
    b.legend()
    for ax in (a, b):
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# Reference categorical slots 1-3 (validated all-pairs for CVD); color follows the model.
MODEL_COLORS = {"dscnn": "#2a78d6", "gru": "#eb6834", "kwt": "#1baf7a"}
INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def model_comparison(rows: list[dict], path) -> None:
    """Accuracy vs size (left) and vs MACs/clip (right), shared accuracy axis, direct-labeled points."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, facecolor=SURFACE)
    float_rows = [r for r in rows if r["precision"] == "float32"]
    int8_rows = [r for r in rows if r["precision"] == "int8"]

    def point(ax, x, r, hollow=False):
        c = MODEL_COLORS.get(r["model"], INK_2)
        ax.scatter(
            [x], [100 * r["test_acc"]], s=90, zorder=3, linewidths=2,
            facecolors=SURFACE if hollow else c, edgecolors=c,
        )

    def label(ax, x, r, text, dy=8):
        ax.annotate(text, (x, 100 * r["test_acc"]), xytext=(0, dy), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)

    size_ax, mac_ax = axes
    for r in float_rows:
        point(size_ax, r["size_kb"], r)
        label(size_ax, r["size_kb"], r, f"{r['name']}\n{r['test_acc']:.1%}")
        point(mac_ax, r["macs"] / 1e6, r)
        label(mac_ax, r["macs"] / 1e6, r, f"{r['name']}\n{r['test_acc']:.1%}")
    for r in int8_rows:
        src = next(f for f in float_rows if f["model"] == r["model"])
        size_ax.annotate("", (r["size_kb"], 100 * r["test_acc"]), (src["size_kb"], 100 * src["test_acc"]),
                         arrowprops=dict(arrowstyle="->", color=INK_2, lw=1, shrinkA=7, shrinkB=7))
        point(size_ax, r["size_kb"], r, hollow=True)
        label(size_ax, r["size_kb"], r, f"{r['name']} (C)\n{r['test_acc']:.1%}", dy=-26)

    size_ax.set(xscale="log", xlabel="parameter size (KB, log scale)", ylabel="test accuracy (%), 12-class")
    mac_ax.set(xscale="log", xlabel="multiply-accumulates per clip (millions, log scale)")
    size_ax.set_title("Accuracy vs model size", loc="left", fontsize=11, color=INK)
    mac_ax.set_title("Accuracy vs compute", loc="left", fontsize=11, color=INK)
    accs = [100 * r["test_acc"] for r in rows]
    size_ax.set_ylim(min(accs) - 0.8, max(accs) + 0.8)
    for ax, ticks in ((size_ax, [10, 20, 50, 100, 200, 500, 1000]), (mac_ax, [0.5, 1, 2, 5, 10, 20, 50])):
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo / 1.6, hi * 1.6)  # room for direct labels
        lo, hi = ax.get_xlim()
        ax.set_xticks([t for t in ticks if lo <= t <= hi], [f"{t:g}" for t in ticks if lo <= t <= hi])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9)
        ax.xaxis.label.set_color(INK_2)
        ax.yaxis.label.set_color(INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


MODEL_NAMES = {"dscnn": "DS-CNN", "gru": "GRU", "kwt": "KWT"}


def scaling(result: dict, path) -> None:
    """Speedup vs device count, weak (left) and strong (right) scaling, with the ideal line."""
    rows = result["rows"]
    counts = sorted({r["devices"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, facecolor=SURFACE)
    titles = {
        "weak": f"Weak scaling ({result['per_device_batch']} clips per device)",
        "strong": f"Strong scaling (global batch {result['global_batch']})",
    }
    for ax, regime in zip(axes, ("weak", "strong")):
        ax.plot(counts, counts, color=INK_2, lw=1, ls="--", zorder=1)
        ax.annotate("ideal", (counts[-1], counts[-1]), xytext=(4, -2), textcoords="offset points", fontsize=9, color=INK_2)
        for model in dict.fromkeys(r["model"] for r in rows):
            pts = sorted((r["devices"], r["speedup"]) for r in rows if r["model"] == model and r["regime"] == regime)
            xs, ys = zip(*pts)
            c = MODEL_COLORS.get(model, INK_2)
            ax.plot(xs, ys, color=c, lw=2, marker="o", ms=7, mec=SURFACE, mew=1.5, zorder=3)
            ax.annotate(f"{MODEL_NAMES.get(model, model)} {ys[-1]:.1f}x", (xs[-1], ys[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK)
        ax.set_title(titles[regime], loc="left", fontsize=11, color=INK)
        ax.set(xscale="log", yscale="log", xlabel="devices")
        ax.set_xticks(counts, [str(c) for c in counts])
        ax.set_yticks([0.5, 1, 2, 4, 8], ["0.5x", "1x", "2x", "4x", "8x"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlim(counts[0] / 1.3, counts[-1] * 2.2)
        ys_all = [r["speedup"] for r in rows] + counts
        ax.set_ylim(min(0.8, min(ys_all) / 1.3), max(ys_all) * 1.3)
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9)
        ax.xaxis.label.set_color(INK_2)
    axes[0].set_ylabel("training throughput speedup vs 1 device", color=INK_2)
    fig.suptitle(f"{result['n_devices']} x {result['device_kind']} ({result['platform']})", x=0.01, ha="left",
                 fontsize=9, color=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# Replay buffer size is ordered -> one-hue ordinal ramp (light = small buffer); naive = categorical orange;
# joint = neutral dashed reference. Distinct markers + direct labels as secondary encoding.
REPLAY_RAMP = ["#6da7ec", "#2a78d6", "#104281"]
REPLAY_MARKERS = ["o", "s", "D"]


def _continual_style(rows):
    replay = sorted({r["buffer_per_class"] for r in rows if r["buffer_per_class"]})
    styles = {}
    for r in rows:
        m = r["method"]
        if m == "naive":
            styles[m] = dict(color="#eb6834", marker="v", ls="-", label="naive fine-tune")
        elif m == "joint":
            styles[m] = dict(color=INK_2, marker="^", ls="--", label="joint (upper bound)")
        else:
            i = replay.index(r["buffer_per_class"])
            styles[m] = dict(color=REPLAY_RAMP[min(i, 2)], marker=REPLAY_MARKERS[min(i, 2)], ls="-",
                             label=f"replay {r['buffer_per_class']}/class")
    return styles


def continual(rows: list[dict], path, task_names: list[str]) -> None:
    """Left: accuracy on the first task's words over stages (forgetting). Right: accuracy over all seen words."""
    styles = _continual_style(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, facecolor=SURFACE)
    n = len(task_names)
    words = np.cumsum([len(t.split(", ")) for t in task_names])
    xticks = [f"after T{i}\n({words[i]} words)" for i in range(n)]
    panels = [
        (axes[0], lambda h: h["acc_per_task"][0], f"Accuracy on the first {words[0]} words (T0)"),
        (axes[1], lambda h: h["acc_seen"], "Accuracy on all words seen so far"),
    ]
    for ax, metric, title in panels:
        ends = []
        for r in rows:
            st = styles[r["method"]]
            ys = [100 * metric(h) for h in r["history"]]
            ax.plot(range(n), ys, color=st["color"], ls=st["ls"], lw=2, marker=st["marker"], ms=7,
                    mec=SURFACE, mew=1.2, zorder=3)
            ends.append([ys[-1], st["label"]])
        # direct labels at the line ends, nudged apart so they don't collide
        ends.sort(key=lambda e: e[0])
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 5.5)
        for (y, label), r in zip(ends, sorted(rows, key=lambda r: 100 * metric(r["history"][-1]))):
            ax.annotate(label, (n - 1, 100 * metric(r["history"][-1])), xytext=(n - 1 + 0.12, y),
                        textcoords="data", va="center", fontsize=9, color=INK,
                        arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8, shrinkA=0, shrinkB=4))
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xticks(range(n), xticks)
        ax.set_xlim(-0.2, n - 1 + 1.15)
        ax.set_ylim(0, 105)
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9)
    axes[0].set_ylabel("test accuracy (%), single head", color=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


AL_STYLES = {
    "entropy": dict(color="#2a78d6", marker="o", label="entropy"),
    "margin": dict(color="#eb6834", marker="s", label="margin"),
    "random": dict(color=INK_2, marker="^", label="random (baseline)"),
}


def active_learning(rows: list[dict], full: dict, path) -> None:
    """Test accuracy vs fraction of labels per strategy, with the full-data reference and the full-1pt band."""
    fig, ax = plt.subplots(figsize=(7.5, 4.4), facecolor=SURFACE)
    full_acc = 100 * full["test_acc"]
    ax.axhspan(full_acc - 1, full_acc, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.axhline(full_acc, color=INK_2, ls=":", lw=1.2, zorder=1)
    ends = []
    for r in rows:
        st = AL_STYLES[r["strategy"]]
        xs = [100 * x["frac"] for x in r["rounds"]]
        ys = [100 * x["test_acc"] for x in r["rounds"]]
        ax.plot(xs, ys, color=st["color"], lw=2, marker=st["marker"], ms=7, mec=SURFACE, mew=1.2, zorder=3)
        ends.append([ys[-1], ys[-1], st["label"]])
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):
        ends[i][1] = max(ends[i][1], ends[i - 1][1] + 1.1)
    x_end = 100 * rows[0]["rounds"][-1]["frac"]
    for y, y_text, label in ends:
        ax.annotate(label, (x_end, y), xytext=(x_end + 1.2, y_text), textcoords="data", va="center", fontsize=9,
                    color=INK, arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8, shrinkA=0, shrinkB=4))
    ax.annotate(f"100% labels: {full_acc:.1f}%  (shaded: within 1 pt)", (1, full_acc), xytext=(0, 4),
                textcoords="offset points", fontsize=9, color=INK_2)
    lo = min(100 * x["test_acc"] for r in rows for x in r["rounds"])
    ax.set_ylim(lo - 1.5, full_acc + 2)
    ax.set_xlim(0, x_end + 9)
    ax.set_xticks([2, 5, 10, 15, 20, 25, 30], ["2%", "5%", "10%", "15%", "20%", "25%", "30%"])
    ax.set(xlabel="fraction of training clips labelled", ylabel="test accuracy (%), 12-class")
    ax.set_title(f"Active learning ({full['model'].upper()}, retrained from scratch each round)", loc="left",
                 fontsize=11, color=INK)
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def confusion_matrix(cm: np.ndarray, classes: list[str], path, title: str = "") -> None:
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set(xlabel="predicted", ylabel="true", title=title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="white" if norm[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
