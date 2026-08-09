"""Render the blog's data plots to white-background PNGs.

The Rust port's copy of the C project's generator. Plots about the MODEL or the
CHECKPOINT keep their numbers verbatim from the reference capture under
kimi-k3-c/blog_artifacts_kimi_k3/: shard sizes, expert reuse, the memory ladder,
KV layout, cache traces and quantization error are properties of Kimi K3 and of
the released weights, not of the implementation language. Plots about THIS
implementation (binary_sizes, rust_vs_c_kernels, fma_instruction_mix, port_loc)
were measured on the Rust port on one machine. README.md in this directory lists
the split. Style: white background, one consistent categorical palette (the same
hues as the hand-drawn diagram strokes), light grid, no chartjunk.

Chart type is chosen per dataset rather than defaulting to bars:
  waterfall   a chain of reductions          fit_cascade
  treemap     part-to-whole with dominance   bytes_census
  donut       a two-way split                expert_reuse
  nested area a single large ratio           kv_layout, mxfp4_savings
  step        a value that holds then breaks gb_read_shelf, shard_sizes
  lollipop    a ranking of scalars           storage_regimes, port_loc, ...
  dumbbell    paired before/after            duplicate_config, preset_ladder, ...
  interval    mean with a spread             quant_error, replication_noise
  line        a trend over a real axis       ladders, sweeps, traces
  scatter     two measured quantities        rss_fidelity, bytes_paradox
  grouped bar the same shape twice           rust_vs_c_kernels, fma_instruction_mix
  stacked bar a genuine part-to-whole        token_time_split, whats_missing
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent

BLUE, AMBER, GREEN = "#1e40af", "#b45309", "#15803d"
PURPLE, RED, TEAL = "#6d28d9", "#b91c1c", "#0f766e"
ORANGE, GRAY = "#c2410c", "#4b5563"
INK, MUTE = "#1f2937", "#6b7280"


def style(ax, title, xlabel=None, ylabel=None, grid="y"):
    ax.set_title(title, fontsize=13, color=INK, pad=12, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, color=MUTE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, color=MUTE)
    if grid:
        ax.grid(True, axis=grid, color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d1d5db")
    ax.tick_params(colors=MUTE, labelsize=10)


def bare(ax):
    """No axes at all, for treemaps and area diagrams."""
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def lollipop(ax, labels, values, colors, fmt="{:,.0f}", xmax_pad=1.25, unit=""):
    ys = np.arange(len(labels))[::-1]
    ax.hlines(ys, 0, values, color=colors, linewidth=2.2, zorder=2)
    ax.scatter(values, ys, s=110, color=colors, zorder=3)
    for y, v, c in zip(ys, values, colors):
        ax.text(v * 1.03, y, (fmt.format(v)) + unit, va="center", fontsize=10.5,
                color=INK, fontweight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, max(values) * xmax_pad)


def dumbbell(ax, labels, left, right, lcolor, rcolor, llab, rlab, fmt="{:,.1f}"):
    ys = np.arange(len(labels))[::-1]
    for y, a, b in zip(ys, left, right):
        ax.plot([a, b], [y, y], color="#d1d5db", linewidth=2.4, zorder=1)
    ax.scatter(left, ys, s=120, color=lcolor, zorder=3, label=llab)
    ax.scatter(right, ys, s=120, color=rcolor, zorder=3, label=rlab)
    for y, a, b in zip(ys, left, right):
        lo, hi = (a, b) if a <= b else (b, a)
        ax.text(lo, y + 0.26, fmt.format(lo), ha="center", fontsize=9, color=MUTE)
        ax.text(hi, y + 0.26, fmt.format(hi), ha="center", fontsize=9, color=INK,
                fontweight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=10)


def save(fig, name):
    path = OUT / f"{name}.png"
    fig.savefig(path, facecolor="white", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  rendered {path.name}")


# ============================================================ THE FIT LEDGER

def fit_cascade():
    """WATERFALL, readable at three levels: exact byte counts for the specialist,
    the mechanism for the developer, and a familiar machine for everyone else.

    Derived: 2.78T x 2 B = 5560 GB; shards.txt 1.56 TB; k3.h 113.49 GB;
    memory_ladder.tsv peak_rss_gb 8.24 at the 8 GB rung."""
    gb = [5560.0, 1560.0, 113.49, 8.24]
    num = ["5,560 GB", "1,560 GB", "113 GB", "8.24 GB"]
    what = ["every weight\nat full precision", "the file you\nactually download",
            "what must stay\nin memory", "what we\nactually measured"]
    machine = ["a multi-GPU\nserver cluster", "a 2 TB\nhard drive",
               "a high-end\nworkstation", "an ordinary\nlaptop"]
    why = ["the experts already\narrive at 4 bits",
           "only 16 of 896 run, so\nthe rest stay on disk",
           "read the rest from disk,\none layer at a time"]
    colors = [RED, ORANGE, AMBER, GREEN]

    fig, ax = plt.subplots(figsize=(10.0, 5.9))
    fig.subplots_adjust(bottom=0.30)
    xs = np.arange(4)

    for i, (x, v, c) in enumerate(zip(xs, gb, colors)):
        bottom = gb[i + 1] if i < 3 else 2.2
        ax.add_patch(Rectangle((x - 0.30, bottom), 0.60, v - bottom,
                               facecolor=c, edgecolor="white", linewidth=1.5))
        ax.text(x, v * 1.30, num[i], ha="center", fontsize=13.5,
                color=INK, fontweight="bold")

    for i in range(3):
        ax.plot([xs[i] + 0.30, xs[i + 1] - 0.30], [gb[i + 1], gb[i + 1]],
                color=MUTE, linewidth=1.1, linestyle="--", zorder=1)
        top = gb[i] * 2.4
        ax.text(xs[i] + 0.5, top * 1.20, why[i], ha="center", va="bottom",
                fontsize=9.5, color=MUTE)
        ax.annotate("", xy=(xs[i] + 0.5, gb[i + 1] * 1.9),
                    xytext=(xs[i] + 0.5, top),
                    arrowprops=dict(arrowstyle="->", color=MUTE, linewidth=1.3))

    ax.text(1.5, 120000, "675x smaller, and it answers with the same words",
            ha="center", fontsize=12.5, color=INK, fontweight="bold")

    ax.set_yscale("log")
    ax.set_ylim(2.2, 900000)
    ax.set_xlim(-0.60, 3.60)
    ax.set_xticks(xs)
    ax.set_xticklabels(what, fontsize=10.5)
    style(ax, "How a model far too big for any desk ends up running on one",
          None, "memory needed (GB)")

    # say plainly that the axis is not linear, in the clear space under the first bar
    ax.text(-0.50, 5.0, "each gridline is 10x the one below,\notherwise the last bar "
            "would be invisible", fontsize=9, color=MUTE, va="center", ha="left")

    # the row anyone can read: which machine actually holds each amount
    for x, m, c in zip(xs, machine, colors):
        ax.text(x, -0.17, m, transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=10.5, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.45", facecolor=c, edgecolor="none"))
    save(fig, "fit_cascade")


def bytes_census():
    """PROPORTIONAL BAR + MAGNIFIED INSET ROW. 02_model/shards.txt:
    routed experts 1.447 TB, trunk 108.81 GB, embed+lm_head 4.70 GB, vision 0.89 GB.
    One slice is 93%, so the remainder is re-expanded to full width underneath."""
    experts, trunk, embed, vision = 1447.0, 108.81, 4.70, 0.89
    total = experts + trunk + embed + vision
    rest = total - experts

    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    # --- row 1: the whole checkpoint, to scale
    fe = experts / total
    ax.add_patch(Rectangle((0, 0.62), fe, 0.26, facecolor=GRAY,
                           edgecolor="white", linewidth=2))
    ax.add_patch(Rectangle((fe, 0.62), 1 - fe, 0.26, facecolor="#cbd5e1",
                           edgecolor="white", linewidth=2))
    ax.text(fe / 2, 0.75, f"routed experts, MXFP4, streamed\n{experts:,.0f} GB   "
            f"{100 * fe:.1f}% of the checkpoint",
            ha="center", va="center", fontsize=11, color="white", fontweight="bold")
    ax.text(1.005, 0.75, f"everything else\n{rest:.1f} GB", va="center",
            fontsize=9.5, color=MUTE)
    ax.text(0, 0.94, "the whole 1.56 TB checkpoint, to scale", fontsize=10,
            color=INK, fontweight="bold")

    # --- row 2: that remainder, re-expanded to the full width
    segs = [("dense trunk, bf16, streamed", trunk, BLUE, None),
            ("embed + lm_head, resident", embed, GREEN, (-0.10, 0.05, "right")),
            ("vision, never used", vision, TEAL, (0.03, -0.05, "left"))]
    x = 0.0
    for name, v, c, off in segs:
        w = v / rest
        ax.add_patch(Rectangle((x, 0.16), w, 0.26, facecolor=c,
                               edgecolor="white", linewidth=2))
        if off is None:
            ax.text(x + w / 2, 0.29, f"{name}\n{v:,.2f} GB", ha="center",
                    va="center", fontsize=10.5, color="white", fontweight="bold")
        else:
            dx, dy, ha = off
            ax.annotate(f"{name}, {v:.2f} GB", xy=(x + w / 2, 0.16),
                        xytext=(x + w / 2 + dx, dy), ha=ha, va="center",
                        fontsize=9.5, color=c,
                        arrowprops=dict(arrowstyle="-", color=c, linewidth=1.1))
        x += w
    ax.text(0, 0.48, f"the remaining {rest:.1f} GB, magnified to the same width",
            fontsize=10, color=INK, fontweight="bold")

    ax.set_xlim(-0.02, 1.18); ax.set_ylim(-0.14, 1.02)
    bare(ax)
    ax.set_title("Where the 1.56 TB lives: 93% of it is experts that never load",
                 fontsize=13, color=INK, pad=12, fontweight="bold")
    save(fig, "bytes_census")


def preset_ladder():
    """DUMBBELL. k3_run.c K3_PRESETS + docs/TUNING.md measured peak RSS."""
    names = ["laptop", "desktop", "workstation", "server", "max"]
    budget = [4.0, 26.0, 90.0, 123.0, 219.0]      # trunk + cache requested
    rss = [8.2, 31.9, 95.5, 127.9, 223.8]         # measured peak
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    dumbbell(ax, names, budget, rss, MUTE, GREEN,
             "trunk + cache requested", "measured peak RSS")
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.set_xlim(0, 250)
    style(ax, "Every preset lands just above what it asked for", "gigabytes", grid="x")
    save(fig, "preset_ladder")


# ============================================================ BUILD AND FORMAT

def binary_sizes():
    """LOLLIPOP. cargo build --release on this machine, then ls -l on what it
    produced: the engine first, then the nine test binaries that gate it."""
    names = ["k3  (the engine)", "real_layer", "model_oracle", "ops", "expert",
             "cache", "scale", "tok", "st", "cfg"]
    kb = [v / 1024.0 for v in
          [1159344, 1367104, 1301408, 1297648, 1172064, 1151712, 1121360,
           1083248, 1064416, 988464]]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    lollipop(ax, names, kb, [GREEN] + [GRAY] * 9, fmt="{:.0f}", unit=" KB")
    style(ax, "A 1.16 MB binary that runs a 1.56 TB model", "kilobytes", grid="x")
    ax.text(0, -0.16, "Every test binary statically links the whole engine plus "
            "the harness and the fixtures it needs,\nso they all land inside the "
            "same megabyte as the engine itself.", transform=ax.transAxes,
            fontsize=9.5, color=MUTE, va="top")
    save(fig, "binary_sizes")


def shard_sizes():
    """STEP. 02_model/shards.txt + scripts/shard_sizes.txt."""
    sizes = [2.34] + [16.99] * 91 + [16.57, 4.70, 0.09, 0.80]
    fig, ax = plt.subplots(figsize=(9.4, 4.0))
    xs = np.arange(1, 97)
    ax.step(xs, sizes, where="mid", color=BLUE, linewidth=1.8)
    ax.fill_between(xs, sizes, step="mid", color=BLUE, alpha=0.10)
    ax.axvspan(94.5, 96.5, color="#e6fbf8", zorder=0)
    ax.annotate("the whole vision tower\n0.89 GB, 0.057% of the checkpoint",
                xy=(95.5, 1.2), xytext=(70, 8.0), fontsize=9.5, color=TEAL,
                arrowprops=dict(arrowstyle="->", color=TEAL))
    ax.annotate("shard 1 carries\nthe embedding table", xy=(1, 2.4), xytext=(5, 6.6),
                fontsize=9.5, color=MUTE,
                arrowprops=dict(arrowstyle="->", color=MUTE))
    ax.set_ylim(0, 19)
    style(ax, "96 shards, 1,560,936,091,448 bytes, verified one file at a time",
          "shard number", "gigabytes")
    save(fig, "shard_sizes")


def roundtrip_sizes():
    """SCATTER with a fitted ratio line. tokenizer_parity.txt roundtrips."""
    files = ["k3.h", "k3_ops.c", "modeling_kimi_k3.py", "REPORT.md"]
    by = np.array([24499, 48353, 53444, 201775], float)
    ids = np.array([6862, 14797, 12145, 52671], float)
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    r = by.sum() / ids.sum()
    xr = np.linspace(0, 60000, 10)
    ax.plot(xr, xr * r, color="#d1d5db", linestyle="--", linewidth=1.4,
            label=f"{r:.1f} bytes per token")
    ax.scatter(ids, by, s=130, color=BLUE, zorder=3)
    for f, i, b in zip(files, ids, by):
        ax.annotate(f, xy=(i, b), xytext=(i + 1400, b - 6000), fontsize=9.5, color=MUTE)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    style(ax, "Four files in, byte-identical files back out, 45 of 45 cases matching",
          "token ids produced", "input bytes")
    save(fig, "roundtrip_sizes")


def mxfp4_savings():
    """NESTED AREA. 2.723T expert params at bf16 vs MXFP4."""
    a, b = 5445.0, 1447.0
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    sa = 1.0
    sb = (b / a) ** 0.5
    ax.add_patch(Rectangle((0, 0), sa, sa, facecolor=RED, edgecolor="white",
                           linewidth=2, alpha=0.92))
    ax.add_patch(Rectangle((0, 0), sb, sb, facecolor=GREEN, edgecolor="white",
                           linewidth=2))
    ax.text(sa * 0.62, sa * 0.88, f"experts at bf16\n{a:,.0f} GB",
            fontsize=11.5, color="white", fontweight="bold", ha="center")
    ax.text(sb / 2, sb / 2, f"as shipped\nMXFP4\n{b:,.0f} GB", ha="center",
            va="center", fontsize=11, color="white", fontweight="bold")
    ax.text(1.10, 0.52, "3.8x smaller\nbefore we write\na line of code",
            fontsize=11, color=INK, va="center", fontweight="bold")
    ax.text(1.10, 0.20, "and never expanded,\nwhich would cost\n194 GB per token",
            fontsize=9.5, color=MUTE, va="center")
    ax.set_xlim(-0.03, 1.62); ax.set_ylim(-0.03, 1.10)
    ax.set_aspect("equal")
    bare(ax)
    ax.set_title("Areas to scale: half a byte per weight saves four terabytes",
                 fontsize=13, color=INK, pad=12, fontweight="bold")
    save(fig, "mxfp4_savings")


def token_time_split():
    """STACKED BAR, a genuine part-to-whole. bench_kernels.c header."""
    parts, secs = ["trunk read", "expert read", "compute"], [36.0, 11.0, 10.0]
    colors = [BLUE, AMBER, GREEN]
    fig, ax = plt.subplots(figsize=(9.0, 2.5))
    left = 0.0
    for p, s, c in zip(parts, secs, colors):
        ax.barh([0], [s], left=left, color=c, edgecolor="white", height=0.5)
        ax.text(left + s / 2, 0, f"{p}\n{s:.0f} s", ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")
        left += s
    ax.set_yticks([]); ax.set_xlim(0, 58)
    style(ax, "Where one token goes on the floor configuration: 80% is waiting on disk",
          "seconds", grid=None)
    save(fig, "token_time_split")


# ============================================================ ATTENTION

def context_scaling():
    """LINE, log-log. k3.h KV ladder + gates.txt fixed state."""
    pos = np.array([1024, 4096, 8192, 16384, 32768, 131072, 1048576], float)
    kv_gb = pos * 2370000.0 / 1e9
    kda_gb = np.full_like(pos, 0.62625)
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.plot(pos, kv_gb, marker="o", color=RED, linewidth=2, label="MLA KV cache (24 layers)")
    ax.plot(pos, kda_gb, marker="s", color=GREEN, linewidth=2,
            label="KDA recurrent state (93 layers)")
    ax.axhspan(228, 9000, color="#fdeaea", zorder=0)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_ylim(0.3, 9000)
    ax.text(1150, 300, "above this band it no longer fits\non our 228 GB machine",
            fontsize=9.5, color=RED, va="bottom")
    for p, v, dx, dy in [(8192, 19.4, 0.42, 2.4), (131072, 310.0, 0.30, 2.6),
                         (1048576, 2485.0, 0.10, 1.9)]:
        ax.annotate(f"{v:,.0f} GB", xy=(p, v), xytext=(p * dx, v * dy),
                    fontsize=9.5, color=RED)
    ax.annotate("626 MB at every length", xy=(32768, 0.62), xytext=(1400, 0.36),
                fontsize=9.5, color=GREEN)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    style(ax, "Why 69 layers are KDA: its state does not grow with context",
          "context length (positions)", "gigabytes (log scale)")
    save(fig, "context_scaling")


def kv_layout():
    """NESTED AREA. verify_mla.py: 96 x (192+128) floats vs 512 + 64."""
    a, b = 96 * 320, 576
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    sa, sb = 1.0, (b / a) ** 0.5
    ax.add_patch(Rectangle((0, 0), sa, sa, facecolor=RED, edgecolor="white",
                           linewidth=2, alpha=0.92))
    ax.add_patch(Rectangle((0, 0), sb, sb, facecolor=GREEN, edgecolor="white",
                           linewidth=2))
    ax.text(sa * 0.60, sa * 0.90, f"expanded per-head K and V\n96 x 320 = {a:,} floats",
            fontsize=11, color="white", fontweight="bold", ha="center")
    ax.annotate(f"shared latent\n512 + 64 = {b} floats", xy=(sb, sb * 0.5),
                xytext=(0.42, 0.30), fontsize=10.5, color=INK, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=INK))
    ax.text(1.10, 0.55, "53x smaller\nfor identical math",
            fontsize=12, color=PURPLE, va="center", fontweight="bold")
    ax.set_xlim(-0.03, 1.62); ax.set_ylim(-0.03, 1.08)
    ax.set_aspect("equal")
    bare(ax)
    ax.set_title("Areas to scale: what MLA caches per position, per layer",
                 fontsize=13, color=INK, pad=12, fontweight="bold")
    save(fig, "kv_layout")


def activation_spikes():
    """LINE. 08_precision/ref_forward_93.log |h| max per layer, all 93 rows."""
    h = [0.145457, 0.187096, 0.463652, 1.284419, 2.116883, 3.499256, 5.183913,
         7.938455, 11.42631, 16.28194, 27.33167, 25.27419, 2.190458, 3.104772,
         4.512609, 6.328841, 8.926314, 12.06947, 16.44982, 21.33874, 27.19248,
         33.19071, 30.28815, 2.483391, 3.732213, 5.316144, 7.442885, 10.14285,
         13.81236, 18.29174, 23.44916, 29.68182, 36.71184, 34.19388, 2.611832,
         3.918445, 5.582716, 7.812139, 10.65127, 14.50293, 19.20583, 24.61962,
         31.16391, 38.54324, 47.71483, 62.18339, 76.28153, 2.902113, 4.353188,
         6.202875, 8.680446, 11.83474, 16.11548, 21.33982, 27.35513, 34.62653,
         42.82849, 68.34799, 136.70716, 1.784206, 2.676309, 3.812872, 5.336268,
         7.276244, 9.908325, 13.11913, 16.81654, 21.28461, 26.32548, 32.44112,
         39.87213, 1.998432, 2.997648, 4.271034, 5.977448, 8.149177, 11.09872,
         14.69413, 18.83519, 23.83987, 29.48372, 36.33547, 44.65112, 2.213765,
         3.320648, 4.731827, 6.622394, 9.028471, 12.29341, 16.27718, 20.86213,
         26.41128, 31.06418]
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    layers = np.arange(93)
    ax.plot(layers, h, color=BLUE, linewidth=1.6)
    ax.scatter(layers, h, s=9, color=BLUE, zorder=3)
    for b in (12, 24, 36, 48, 60, 72, 84):
        ax.axvline(b, color="#e5e7eb", linewidth=1.2, zorder=0)
    ax.set_yscale("log")
    ax.annotate("136.7 at L59, then 1.78 at L60", xy=(59, 136.7), xytext=(30, 210),
                fontsize=9.5, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
    ax.annotate("every block boundary resets it", xy=(12, 2.19), xytext=(2, 0.24),
                fontsize=9.5, color=MUTE, arrowprops=dict(arrowstyle="->", color=MUTE))
    style(ax, "Attention residuals drawn by the data: activations climb, then collapse every 12 layers",
          "layer", "max |h| (log scale)")
    save(fig, "activation_spikes")


def expert_reuse():
    """DONUT. cache_capacity_curve.txt: 90,086 of 100,096 requests are repeats."""
    repeats, first = 90086, 10010
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.pie([repeats, first], colors=[GREEN, RED], startangle=90,
           counterclock=False,
           wedgeprops=dict(width=0.38, edgecolor="white", linewidth=3))
    ax.text(0, 0.10, "100,096", ha="center", fontsize=17, color=INK, fontweight="bold")
    ax.text(0, -0.16, "expert requests\nin 68 tokens", ha="center", fontsize=9.5,
            color=MUTE)
    # leader-line labels, anchored well inside the widened axes
    ax.annotate("90,086 repeats, 90.0%\nwhat a cache could catch",
                xy=(0.92, -0.30), xytext=(1.28, -0.55), ha="left", va="center",
                fontsize=10.5, color=GREEN, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=GREEN, linewidth=1.2))
    ax.annotate("10,010 first touches\nthe compulsory floor",
                xy=(-0.26, 0.79), xytext=(-1.05, 1.16), ha="center", va="center",
                fontsize=10.5, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=RED, linewidth=1.2))
    ax.set_xlim(-1.55, 2.55); ax.set_ylim(-1.35, 1.45)
    ax.set_title("Nine of every ten expert reads is a repeat",
                 fontsize=13, color=INK, pad=6, fontweight="bold")
    save(fig, "expert_reuse")


# ============================================================ STREAMING

def pack_progress():
    """LINE + twin. 04_loading/pack_trunk.log progress lines."""
    secs = [25, 47, 74, 99, 127, 153, 181, 208, 236, 244]
    gb = [12.90, 24.31, 36.14, 47.55, 59.38, 70.79, 82.62, 94.02, 105.86, 108.81]
    mbs = [521, 520, 488, 480, 466, 462, 457, 451, 448, 447]
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.plot(secs, gb, marker="o", color=BLUE, linewidth=2)
    ax.fill_between(secs, gb, color=BLUE, alpha=0.08)
    ax.set_ylim(0, 125)
    style(ax, "Packing 93 contiguous layer runs into one 108.81 GB file",
          "seconds", "gigabytes written")
    ax2 = ax.twinx()
    ax2.plot(secs, mbs, marker="s", color=RED, linewidth=1.6, linestyle="--")
    ax2.set_ylabel("MB/s", color=RED, fontsize=11)
    ax2.tick_params(colors=RED, labelsize=10)
    ax2.set_ylim(400, 560)
    for s in ("top", "left"):
        ax2.spines[s].set_visible(False)
    ax.text(30, 100, "throughput drifts 521 to 447 MB/s", fontsize=9.5, color=RED)
    save(fig, "pack_progress")


def resident_vs_streamed():
    """DUMBBELL. run.log (resident) vs inc12.log (streamed)."""
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    dumbbell(ax, ["memory the run asks for (GB)", "seconds before the first token"],
             [315.34, 150.0], [11.33, 0.0], RED, GREEN,
             "trunk held resident", "trunk streamed")
    ax.legend(frameon=False, fontsize=9.5, loc="center right")
    ax.set_xlim(-18, 360)
    style(ax, "Streaming turns a 315 GB floor into an 11 GB dial", grid="x")
    save(fig, "resident_vs_streamed")


def storage_regimes():
    """LOLLIPOP. campaign_host.txt + pack_trunk.py + k3_st.h."""
    labels = ["network volume\n(sequential)", "expert reads\n(random, buffered)",
              "device cold\n(O_DIRECT)", "engine trunk\n(sustained)"]
    mbs = [772, 1247, 3200, 6064]
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    lollipop(ax, labels, mbs, [RED, ORANGE, AMBER, GREEN], unit=" MB/s")
    style(ax, "An eightfold spread in storage speed, and the engine is I/O bound",
          "megabytes per second", grid="x")
    save(fig, "storage_regimes")


def belady_vs_lru():
    """LINE. 05_expert_cache/cache_capacity_curve.txt."""
    cache = [8, 16, 32, 64, 128, 192]
    lru = [36.24, 36.24, 36.24, 36.24, 49.19, 90.00]
    belady = [39.42, 42.61, 48.99, 61.74, 84.59, 90.00]
    pinlru = [37.82, 39.42, 42.57, 48.66, 62.86, 90.00]
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.fill_between(cache, lru, belady, color="#f1ecfe", zorder=0)
    ax.plot(cache, lru, marker="o", color=BLUE, linewidth=2, label="LRU (what we run)")
    ax.plot(cache, pinlru, marker="^", color=AMBER, linewidth=1.8, label="pinned + LRU")
    ax.plot(cache, belady, marker="s", color=GREEN, linewidth=2, label="Belady (the ceiling)")
    ax.axhline(90.0, color=RED, linestyle="--", linewidth=1.2)
    ax.text(8.6, 91.5, "90.0% compulsory ceiling: every expert must be read once",
            fontsize=9.5, color=RED)
    ax.text(30, 46, "the policy gap,\nfree at the same RAM", fontsize=9.5, color=PURPLE)
    ax.set_xscale("log", base=2); ax.set_ylim(30, 100)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    style(ax, "The lever is the policy, not the size: LRU is flat where Belady climbs",
          "expert cache size (GB, log scale)", "simulated hit rate (%)")
    save(fig, "belady_vs_lru")


# ============================================================ VALIDATION

def gate_ladder():
    """FUNNEL. The four validation levels and what each covers."""
    levels = ["op fixtures\n22 kernels", "toy oracle\n13 layers",
              "real layers\n93 of 93", "real logits\n163,840 values"]
    covers = [1, 13, 93, 93]
    weights = ["no weights", "9 MB fixture", "1.56 TB", "1.56 TB"]
    colors = [GRAY, BLUE, AMBER, GREEN]
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    for i, (lv, c, w, col) in enumerate(zip(levels, covers, weights, colors)):
        half = 0.5 * (c / max(covers)) ** 0.5
        y = len(levels) - 1 - i
        ax.add_patch(Rectangle((0.5 - half, y - 0.34), half * 2, 0.68,
                               facecolor=col, edgecolor="white", linewidth=2))
        ax.text(0.5, y, lv, ha="center", va="center", fontsize=10,
                color="white", fontweight="bold")
        ax.text(1.02, y, w, va="center", fontsize=10, color=MUTE)
    ax.set_xlim(0, 1.30); ax.set_ylim(-0.6, 3.6)
    bare(ax)
    ax.set_title("Four levels of proof, and only the last two touch the real checkpoint",
                 fontsize=13, color=INK, pad=12, fontweight="bold")
    save(fig, "gate_ladder")


def layer_conformance():
    """STRIP SCATTER by layer type. conform_all_93.log per-layer seconds."""
    secs = [25, 45, 44, 42, 39, 41, 43, 40, 38, 44, 46, 42, 43, 47, 45, 41,
            39, 44, 48, 46, 43, 42, 45, 49, 47, 44, 41, 43, 46, 48, 45, 42,
            44, 47, 50, 46, 43, 45, 48, 52, 49, 46, 44, 47, 51, 55, 58, 54,
            50, 47, 45, 48, 53, 57, 62, 66, 71, 76, 80, 74, 68, 61, 55, 50,
            47, 45, 48, 52, 56, 60, 57, 53, 49, 46, 44, 47, 51, 54, 50, 47,
            45, 48, 52, 49, 46, 44, 47, 50, 48, 45, 43, 46, 49]
    mla = set(list(range(3, 92, 4)) + [92])
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    xs = np.arange(93)
    k = [i for i in xs if i not in mla]
    m = [i for i in xs if i in mla]
    ax.scatter(k, [secs[i] for i in k], s=34, color=BLUE, label="KDA (69 layers)", zorder=3)
    ax.scatter(m, [secs[i] for i in m], s=52, color=GREEN, marker="D",
               label="Gated MLA (24 layers)", zorder=3)
    ax.plot(xs, secs, color="#e5e7eb", linewidth=1.2, zorder=1)
    ax.annotate("L91 and L92 are both MLA", xy=(91.5, 46), xytext=(66, 74),
                fontsize=9.5, color=GREEN, arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.set_ylim(0, 92)
    style(ax, "All 93 layers checked against torch: 93 passed, 0 failed, 5639 seconds",
          "layer", "seconds to verify")
    save(fig, "layer_conformance")


def same_answer():
    """LOLLIPOP with an identical-output band. decoded_generations.txt."""
    caps = ["96G", "64G", "48G", "32G", "24G", "16G", "12G"]
    sec = [72.3430, 67.8921, 63.1822, 63.4072, 60.8162, 68.4805, 61.0702]
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    xs = np.arange(7)
    ax.vlines(xs, 0, sec, color=AMBER, linewidth=2.2, zorder=2)
    ax.scatter(xs, sec, s=120, color=AMBER, zorder=3)
    for x, v in zip(xs, sec):
        ax.text(x, v + 2.4, f"{v:.1f}", ha="center", fontsize=9.5, color=INK)
    ax.axhspan(0, 12, color="#eafcef", zorder=0)
    ax.text(3, 5.4, "every arm emitted  17374, 20829, 10   ( Paris.)",
            ha="center", fontsize=11, color=GREEN, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(caps, fontsize=10.5)
    ax.set_ylim(0, 86)
    style(ax, "Seven memory ceilings, seven different speeds, one identical answer",
          "hard memory cap", "seconds per token")
    save(fig, "same_answer")


# ============================================================ GENERATION

def step_trace():
    """LINE, four series. generations/*.log STEP tables."""
    hello = [53.04, 8.92, 8.92, 8.74, 8.85, 8.91, 8.79, 8.70, 8.66, 8.66, 8.61,
             8.90, 9.35, 8.92, 9.20, 8.93, 9.18, 8.94, 8.63, 8.90, 8.65, 8.52,
             8.42, 8.43]
    paris = [54.04, 9.21, 8.91, 8.97, 8.90, 9.12, 8.85, 8.84, 8.75, 8.74, 8.81,
             9.05, 8.99, 9.24, 8.98, 9.21]
    code = [53.17, 9.45, 9.03, 8.93, 8.95, 9.06, 8.88, 8.99, 8.97, 8.74, 9.19,
            9.72, 9.39, 9.44, 9.31, 9.40, 9.39, 9.23, 9.12, 9.29, 9.16, 8.98,
            8.92, 8.98, 8.88, 9.00, 8.92, 8.82]
    explain = [97.98, 10.07, 9.25, 9.16, 8.99, 8.86, 9.02, 9.02, 8.57, 8.56,
               8.57, 8.65, 8.21, 8.17, 9.22, 8.38, 8.18, 8.31, 8.34, 8.18,
               8.12, 8.12, 8.34, 8.29, 8.24, 8.03, 8.22, 8.14, 8.05, 8.05,
               8.06, 8.28]
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    for data, c, lab in ((hello, BLUE, "hello, 24 tokens"),
                         (paris, GREEN, "paris, 16 tokens"),
                         (code, AMBER, "code, 28 tokens"),
                         (explain, PURPLE, "explain, 32 tokens")):
        ax.plot(range(len(data)), data, marker="o", markersize=3.5,
                color=c, linewidth=1.6, label=lab)
    ax.axvspan(-0.4, 0.4, color="#fdeaea", zorder=0)
    ax.text(1.2, 82, "step 0 is the prefill wall", fontsize=10, color=RED)
    ax.text(12, 16, "every later step: 8 to 10 seconds, 25.83 GB read",
            fontsize=10, color=MUTE)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    ax.set_ylim(0, 108)
    style(ax, "One prefill wall, then a steady shelf, on all four prompts",
          "generation step", "seconds")
    save(fig, "step_trace")


def prefill_cost():
    """SCATTER + twin. Step 0 from the four generation logs."""
    prompts, names = [5, 5, 5, 17], ["paris", "hello", "code", "explain"]
    secs, gb = [54.04, 53.04, 53.17, 97.98], [99.70, 89.21, 93.25, 200.67]
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.scatter(prompts, secs, s=110, color=BLUE, zorder=3, label="step 0 seconds")
    for p, s, n in zip(prompts, secs, names):
        ax.annotate(n, xy=(p, s), xytext=(p + 0.4, s + 2), fontsize=9.5, color=MUTE)
    ax.set_ylim(0, 125)
    style(ax, "Prefill scales with the prompt, and nothing after it does",
          "prompt length (tokens)", "seconds")
    ax2 = ax.twinx()
    ax2.scatter(prompts, gb, s=90, marker="s", color=RED, zorder=3)
    ax2.set_ylabel("step 0 gigabytes read", color=RED, fontsize=11)
    ax2.tick_params(colors=RED, labelsize=10); ax2.set_ylim(0, 260)
    for s in ("top", "left"):
        ax2.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    save(fig, "prefill_cost")


# ============================================================ THE LADDER

LADDER_GB = [8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224]
LADDER_SEC = [32.69, 31.41, 32.21, 31.85, 31.44, 29.76, 28.60, 24.40, 29.40,
              26.31, 21.32, 19.21]
LADDER_READ = [25.83, 25.83, 25.83, 25.83, 25.83, 25.83, 25.83, 18.11, 17.51,
               17.28, 16.65, 14.53]
LADDER_RSS = [8.24, 10.53, 16.00, 23.95, 31.90, 47.80, 63.71, 95.51, 128.18,
              159.98, 191.83, 223.82]
LADDER_HITT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 29.89, 32.20, 33.08, 35.53, 43.75]
LADDER_IO = [57.3, 59.6, 55.8, 55.2, 53.6, 51.9, 53.6, 47.2, 60.6, 47.4, 49.9, 40.9]
LADDER_CACHE = [0.49, 2.79, 4.39, 7.58, 10.80, 17.19, 23.59, 36.39, 49.19,
                61.99, 77.00, 108.98]


def memory_ladder():
    """LINE with a noise band. 07_performance/memory_ladder.tsv."""
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    mean = float(np.mean(LADDER_SEC))
    ax.axhspan(mean * 0.835, mean * 1.165, color="#f3f4f6", zorder=0)
    ax.text(8.4, mean * 1.19, "the measured 33% noise band", fontsize=9.5, color=MUTE)
    ax.plot(LADDER_GB, LADDER_SEC, marker="o", color=BLUE, linewidth=2)
    for g, s in ((8, 32.69), (224, 19.21)):
        ax.annotate(f"{s:.2f} s/token", xy=(g, s), xytext=(g * 1.1, s + 2.4),
                    fontsize=10, color=INK, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks(LADDER_GB)
    ax.set_xticklabels([str(g) for g in LADDER_GB], fontsize=9)
    ax.set_ylim(15, 38)
    style(ax, "28x the memory buys 1.70x the speed, and most steps are inside the noise",
          "memory budget (GB, log scale)", "seconds per token")
    save(fig, "memory_ladder")


def gb_read_shelf():
    """STEP. The flat 25.83 GB/token shelf, then the break."""
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    xs = np.arange(12)
    ax.step(xs, LADDER_READ, where="mid", color=BLUE, linewidth=2.4)
    ax.fill_between(xs, LADDER_READ, step="mid", color=BLUE, alpha=0.10)
    ax.axvspan(-0.5, 6.5, color="#fdeaea", zorder=0)
    ax.scatter(xs[:7], LADDER_READ[:7], s=60, color=RED, zorder=3)
    ax.scatter(xs[7:], LADDER_READ[7:], s=60, color=GREEN, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(g) for g in LADDER_GB], fontsize=9)
    ax.set_ylim(0, 32)
    ax.annotate("seven budgets, a 48x bigger cache,\nand not one byte of difference",
                xy=(3, 25.83), xytext=(0.0, 8.5), fontsize=10, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.text(8.2, 20.6, "the cache finally engages", fontsize=10, color=GREEN)
    style(ax, "The expert cache does nothing at all until about 36 GB of arena",
          "memory budget (GB)", "gigabytes read per token")
    save(fig, "gb_read_shelf")


def rss_fidelity():
    """SCATTER against the identity line."""
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot([0, 240], [0, 240], color="#d1d5db", linestyle="--", linewidth=1.2)
    ax.scatter(LADDER_GB, LADDER_RSS, s=70, color=BLUE, zorder=3)
    ax.annotate("8 GB asked, 8.24 GB used", xy=(8, 8.24), xytext=(24, 30),
                fontsize=10, color=INK, arrowprops=dict(arrowstyle="->", color=MUTE))
    ax.annotate("224 GB asked, 223.82 GB used", xy=(224, 223.82), xytext=(96, 196),
                fontsize=10, color=INK, arrowprops=dict(arrowstyle="->", color=MUTE))
    ax.set_xlim(0, 245); ax.set_ylim(0, 245)
    style(ax, "The memory plan is not lying: every rung lands on its own budget",
          "budget requested (GB)", "peak RSS measured (GB)")
    save(fig, "rss_fidelity")


def sim_vs_measured():
    """LINE. Simulated LRU against measured retention."""
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.plot([8, 16, 32, 64, 128, 192], [36.24, 36.24, 36.24, 36.24, 49.19, 90.00],
            marker="s", color=AMBER, linewidth=2, linestyle="--",
            label="simulated LRU (an upper bound)")
    ax.plot(LADDER_CACHE, LADDER_HITT, marker="o", color=BLUE, linewidth=2,
            label="measured retention")
    ax.set_xscale("log", base=2); ax.set_ylim(-4, 100)
    ax.annotate("the simulated trace re-prefills\nand manufactures reuse", xy=(10, 20),
                xytext=(1.1, 62), fontsize=10, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    style(ax, "Replay predicts a floor of 36%, steady-state decode measures zero",
          "expert cache arena (GB, log scale)", "hit rate (%)")
    save(fig, "sim_vs_measured")


def cache_wakeup():
    """LINE. rung_8gb.log against rung_96gb.log per-step READ GB."""
    r8 = [99.70, 25.83, 25.83, 25.83, 25.83, 25.83, 25.83, 25.83]
    r96 = [99.70, 23.58, 19.90, 14.07, 12.34, 12.79, 15.14, 18.11]
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.plot(range(8), r8, marker="o", color=RED, linewidth=2, label="8 GB budget")
    ax.plot(range(8), r96, marker="s", color=GREEN, linewidth=2, label="96 GB budget")
    ax.fill_between(range(8), r96, r8, color="#eafcef", zorder=0)
    ax.set_ylim(0, 112)
    ax.text(2.4, 28, "flat: nothing is retained", fontsize=10, color=RED)
    ax.text(3.2, 8, "declining: the cache is finally holding experts",
            fontsize=10, color=GREEN)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    style(ax, "What a working expert cache actually looks like, step by step",
          "generation step", "gigabytes read")
    save(fig, "cache_wakeup")


def hit_metrics():
    """LINE, three definitions of one number."""
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.plot(LADDER_GB, [100.0] * 11 + [99.93], marker="^", color=RED,
            linewidth=2, label="counted at request time")
    ax.plot(LADDER_GB, [0.0] * 12, marker="v", color=ORANGE, linewidth=2,
            label="counted as resident before the step")
    ax.plot(LADDER_GB, LADDER_HITT, marker="o", color=BLUE, linewidth=2,
            label="derived: 1 - evictions/requests")
    ax.set_xscale("log", base=2); ax.set_ylim(-6, 110)
    ax.legend(frameon=False, fontsize=9.5, loc="center left")
    style(ax, "Three definitions of one number, and only the third agrees with the bytes",
          "memory budget (GB, log scale)", "reported hit rate (%)")
    save(fig, "hit_metrics")


def io_share():
    """LOLLIPOP with a 50% reference. memory_ladder.tsv io_share."""
    fig, ax = plt.subplots(figsize=(9.4, 4.0))
    xs = np.arange(12)
    ax.vlines(xs, 0, LADDER_IO, color=BLUE, linewidth=2.2, zorder=2)
    ax.scatter(xs, LADDER_IO, s=95, color=BLUE, zorder=3)
    ax.axhline(50, color=RED, linestyle="--", linewidth=1.3)
    ax.text(0.0, 64, "above this line, more than half the wall clock is waiting on disk",
            fontsize=9.5, color=RED)
    for x, v in zip(xs, LADDER_IO):
        ax.text(x, v + 2.2, f"{v:.0f}", ha="center", fontsize=8.5, color=MUTE)
    ax.set_xticks(xs); ax.set_xticklabels([str(g) for g in LADDER_GB], fontsize=9)
    ax.set_ylim(0, 72)
    style(ax, "This is an I/O problem at every budget, from 8 GB to 224 GB",
          "memory budget (GB)", "I/O share of wall clock (%)")
    save(fig, "io_share")


# ============================================================ ALLOCATION

def trunk_cache_split():
    """LINE, two series. deep/engine/trunk_cache_split.tsv."""
    t32 = [2.7, 6.8, 10.8, 16.2, 21.6, 25.6]
    s32 = [31.23, 30.58, 31.25, 30.88, 29.13, 28.06]
    t128 = [12.3, 30.8, 49.2, 73.8, 98.4, 110.0]
    s128 = [28.38, 25.20, 25.69, 18.37, 19.46, 16.80]
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.plot(t32, s32, marker="s", color=AMBER, linewidth=2, label="32 GB total")
    ax.plot(t128, s128, marker="o", color=BLUE, linewidth=2, label="128 GB total")
    ax.annotate("28.38", xy=(12.3, 28.38), xytext=(6, 31.4), fontsize=10, color=BLUE)
    ax.annotate("16.80, same total memory", xy=(110.0, 16.80), xytext=(64, 14.6),
                fontsize=10, color=GREEN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.legend(frameon=False, fontsize=9.5)
    ax.set_ylim(13, 34)
    style(ax, "At a fixed budget, giving the trunk more is 1.69x faster",
          "gigabytes given to the trunk", "seconds per token")
    save(fig, "trunk_cache_split")


def bytes_paradox():
    """SCATTER with a colour dimension. The six 128 GB rows."""
    read = [14.46, 14.74, 17.06, 17.51, 25.83, 25.83]
    sec = [28.38, 25.20, 25.69, 18.37, 19.46, 16.80]
    hit = [44.02, 42.93, 33.97, 32.20, 0.0, 0.0]
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    sc = ax.scatter(read, sec, s=170, c=hit, cmap="coolwarm_r", zorder=3,
                    edgecolor="white", linewidth=1.5)
    ax.annotate("slowest run\n44.0% cache hit\n14.46 GB read",
                xy=(14.46, 28.38), xytext=(15.6, 29.4), fontsize=9.5, color=RED)
    ax.annotate("fastest run\n0.0% cache hit\n25.83 GB read",
                xy=(25.83, 16.80), xytext=(19.2, 15.0), fontsize=9.5, color=GREEN,
                fontweight="bold", arrowprops=dict(arrowstyle="->", color=GREEN))
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("expert cache hit rate (%)", fontsize=10, color=MUTE)
    cb.ax.tick_params(colors=MUTE, labelsize=9)
    ax.set_ylim(14, 31)
    style(ax, "The winner reads 79% more expert bytes and still wins",
          "gigabytes read per token", "seconds per token")
    save(fig, "bytes_paradox")


# ============================================================ SPREAD

def replication_noise():
    """INTERVAL / dot plot with mean and band. replication.md."""
    sec = [14.78, 14.67, 20.14]
    mean, sd = 16.53, 3.13
    fig, ax = plt.subplots(figsize=(8.8, 3.6))
    ax.axhspan(mean - sd, mean + sd, color="#f3f4f6", zorder=0)
    ax.axhline(mean, color=MUTE, linestyle="--", linewidth=1.4)
    ax.scatter(range(3), sec, s=180, color=[BLUE, BLUE, RED], zorder=3)
    for x, v in zip(range(3), sec):
        ax.text(x, v + 0.55, f"{v:.2f}", ha="center", fontsize=11,
                color=INK, fontweight="bold")
    ax.annotate("", xy=(2.42, 14.67), xytext=(2.42, 20.14),
                arrowprops=dict(arrowstyle="<->", color=RED, linewidth=1.6))
    ax.text(2.50, 17.4, "spread\n33.1%", fontsize=10.5, color=RED, fontweight="bold",
            va="center")
    ax.text(-0.42, mean + 0.25, f"mean {mean}, sd {sd}", fontsize=10, color=MUTE)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["run 1", "run 2", "run 3"], fontsize=10.5)
    ax.set_xlim(-0.5, 2.9); ax.set_ylim(12.5, 22)
    style(ax, "Same binary, same prompt, same flags, same machine, same minute",
          None, "seconds per token")
    save(fig, "replication_noise")


def duplicate_config():
    """DUMBBELL. rung_128gb.log against p2_128_0.60.log."""
    fig, ax = plt.subplots(figsize=(9.4, 3.2))
    dumbbell(ax, ["gigabytes read", "trunk bandwidth (MB/s ÷ 100)", "seconds per token"],
             [374.99, 27.09, 29.40], [374.99, 58.74, 18.37], GRAY, GREEN,
             "run A", "run B", fmt="{:,.2f}")
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.set_xlim(-20, 430)
    style(ax, "One configuration measured twice: identical work, very different clocks",
          grid="x")
    save(fig, "duplicate_config")


# ============================================================ PRECISION

def quant_error():
    """INTERVAL plot: mean with the p99-to-max range. quant_study.json."""
    names = ["L1.v_proj", "L1.g_proj", "L5.v_proj", "L5.o_proj", "L9.k_proj",
             "L9.g_proj", "L9.o_proj", "L3.o_proj", "L3.g_proj", "L7.kv_b_proj",
             "L7.o_proj"]
    i8 = [0.00926, 0.00895, 0.00916, 0.01029, 0.00914, 0.00898, 0.01020,
          0.01090, 0.01092, 0.00741, 0.01099]
    i4m = [0.16811, 0.16233, 0.16614, 0.18670, 0.16568, 0.16286, 0.18512,
           0.19774, 0.19801, 0.13427, 0.19919]
    i4x = [0.22531, 0.21741, 0.24928, 0.45223, 0.26996, 0.26282, 0.55945,
           0.28295, 0.42202, 0.20659, 0.64514]
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ys = np.arange(len(names))[::-1]
    for y, m, x in zip(ys, i4m, i4x):
        ax.plot([m * 100, x * 100], [y, y], color="#fca5a5", linewidth=3, zorder=2)
    ax.scatter([v * 100 for v in i4x], ys, s=40, color=INK, zorder=4,
               label="int4 worst row")
    ax.scatter([v * 100 for v in i4m], ys, s=110, color=RED, zorder=4,
               label="int4 mean")
    ax.scatter([v * 100 for v in i8], ys, s=110, color=GREEN, zorder=4,
               label="int8 mean")
    ax.set_yticks(ys); ax.set_yticklabels(names, fontsize=9)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    ax.set_xlim(0, 74)
    style(ax, "int8 costs about 1%, int4 costs about 17%, and its worst rows cost 65%",
          "mean relative weight error (%)", grid="x")
    save(fig, "quant_error")


def quant_by_layer_type():
    """DOT PLOT. quant_sensitivity.txt means by layer type."""
    kinds = ["KDA layers\n(n=2)", "MLA layers\n(n=18)", "all sampled\n(n=31)"]
    i8, i4 = [1.046, 0.948, 0.961], [18.746, 17.154, 17.383]
    fig, ax = plt.subplots(figsize=(8.8, 3.4))
    dumbbell(ax, kinds, i8, i4, GREEN, RED, "int8", "int4", fmt="{:.2f}%")
    ax.legend(frameon=False, fontsize=9.5, loc="center right")
    ax.set_xlim(0, 23)
    style(ax, "No layer type tolerates 4 bits any better than the others",
          "mean relative weight error (%)", grid="x")
    save(fig, "quant_by_layer_type")


def whats_missing():
    """STACKED PROPORTION, where one part is meant to be invisible."""
    total, vis = 1560.94, 0.89
    fig, ax = plt.subplots(figsize=(9.0, 2.4))
    ax.barh([0], [total - vis], color=GREEN, edgecolor="white", height=0.42,
            label="implemented (text path)")
    ax.barh([0], [vis], left=total - vis, color=RED, edgecolor="white",
            height=0.42, hatch="//", label="absent (vision tower)")
    ax.annotate("0.89 GB, or 0.057% of the checkpoint,\nfully specified and entirely uncoded",
                xy=(total, 0.24), xytext=(880, 0.60), fontsize=9.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_yticks([]); ax.set_xlim(0, 1720)
    ax.legend(frameon=False, fontsize=9.5, loc="lower left")
    style(ax, "What this engine does not implement, drawn to scale", "gigabytes",
          grid=None)
    save(fig, "whats_missing")


# ============================================================ THE RUST PORT
#
# Everything from here down was measured on the Rust port itself, on one machine
# (Apple M5, 10 cores, macOS, rustc 1.97.1, cargo build --release with no
# target-cpu=native). Nothing above this line was.

def rust_vs_c_kernels():
    """GROUPED HORIZONTAL BARS. The kernel benchmark run at the same shapes in
    both languages on one machine. The C build has no OpenMP here because libomp
    is absent, so it is single-threaded; the Rust build uses rayon on 10 cores."""
    shapes = ["bf16 matmul\n12288 x 7168", "MXFP4 matmul\n3072 x 3584"]
    c_ms, c_gf = [18.29, 2.15], [9.6, 10.3]
    r_ms, r_gf = [2.46, 0.58], [71.6, 37.9]
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    ys = np.arange(len(shapes), dtype=float)[::-1]
    ax.barh(ys + 0.18, c_ms, height=0.30, color=GRAY, edgecolor="white",
            label="C, one thread (no libomp on this machine)")
    ax.barh(ys - 0.18, r_ms, height=0.30, color=GREEN, edgecolor="white",
            label="Rust, rayon across 10 cores")
    for y, v, g in zip(ys + 0.18, c_ms, c_gf):
        ax.text(v + 0.45, y, f"{v:.2f} ms      {g:.1f} GFLOP/s", va="center",
                fontsize=10, color=MUTE)
    for y, v, g in zip(ys - 0.18, r_ms, r_gf):
        ax.text(v + 0.45, y, f"{v:.2f} ms      {g:.1f} GFLOP/s", va="center",
                fontsize=10, color=INK, fontweight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels(shapes, fontsize=10)
    ax.set_ylim(-0.62, 1.62)
    ax.set_xlim(0, 27)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    style(ax, "The same two kernels in both languages, on the same machine",
          "milliseconds per call", grid="x")
    ax.text(0, -0.20, "The C build runs on one core because libomp is absent here, "
            "and the Rust build uses rayon on all 10.\nThis compares two builds, "
            "not two cores.", transform=ax.transAxes, fontsize=9.5, color=RED,
            va="top")
    ax.text(0, -0.42, "Projected over one token: bf16 11.78 s in C against 1.59 s "
            "in Rust, MXFP4 9.49 s against 2.56 s.", transform=ax.transAxes,
            fontsize=9.5, color=MUTE, va="top")
    save(fig, "rust_vs_c_kernels")


def fma_instruction_mix():
    """GROUPED BARS. Disassembly of the release numeric core on this arm64
    machine: fmla.2d in each kernel body, C against Rust."""
    kernels = ["f32 matmul\nmatmul  /  dot_f32",
               "bf16 matmul\nmatmul_bf16  /  dot_bf16",
               "MXFP4 matmul\nmatmul_mxfp4  /  dot_mxfp4"]
    c, r = [8, 8, 4], [8, 8, 4]
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    xs = np.arange(len(kernels), dtype=float)
    ax.bar(xs - 0.17, c, width=0.32, color=GRAY, edgecolor="white", label="C")
    ax.bar(xs + 0.17, r, width=0.32, color=GREEN, edgecolor="white", label="Rust")
    for x, v in list(zip(xs - 0.17, c)) + list(zip(xs + 0.17, r)):
        ax.text(x, v + 0.18, str(v), ha="center", fontsize=11.5, color=INK,
                fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(kernels, fontsize=9.5)
    ax.set_ylim(0, 11)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    style(ax, "Two languages, one instruction mix: 8, 8 and 4 vector FMAs either way",
          None, "fmla.2d in the kernel body")
    ax.text(-0.46, 9.6, "20 fmla.2d and 3 scalar fmadd across the whole numeric "
            "core, on both sides", fontsize=9.5, color=MUTE)
    ax.text(0, -0.24, "On x86-64 the same three Rust kernels emit 4, 4 and 2 "
            "vfmadd*pd, matching the C accumulator counts.",
            transform=ax.transAxes, fontsize=9.5, color=MUTE, va="top")
    save(fig, "fma_instruction_mix")


def port_loc():
    """LOLLIPOP. wc -l over src/ in the Rust port, largest module first."""
    mods = ["main.rs", "tok/unicode.rs", "ops/mod.rs", "bind.rs", "tok/mod.rs",
            "cache.rs", "trunk.rs", "st.rs", "ops/dispatch.rs", "cfg.rs",
            "tok/loader.rs", "load.rs", "io_util.rs", "lib.rs"]
    loc = [2224, 2121, 1805, 1290, 917, 764, 732, 632, 580, 417, 364, 304, 233, 77]
    colors = [AMBER if m == "tok/unicode.rs" else BLUE for m in mods]
    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    lollipop(ax, mods, loc, colors, xmax_pad=1.22, fmt="   {:,.0f}")
    ax.annotate("transcribed Unicode tables,\nnot logic", xy=(1700, 11.93),
                xytext=(1180, 8.4), fontsize=9.5, color=AMBER, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=AMBER))
    style(ax, "Where the port's lines went, and the second largest module is a table",
          "lines of Rust", grid="x")
    ax.text(0, -0.11, "Fourteen modules here, 12,673 lines across all of src/, and "
            "4,582 more in tests and benches.\nExactly four dependencies: libc, "
            "rayon, serde, serde_json.", transform=ax.transAxes, fontsize=9.5,
            color=MUTE, va="top")
    save(fig, "port_loc")


FNS = [fit_cascade, bytes_census, preset_ladder, binary_sizes, shard_sizes,
       roundtrip_sizes, mxfp4_savings, token_time_split, context_scaling,
       kv_layout, activation_spikes, expert_reuse, pack_progress,
       resident_vs_streamed, storage_regimes, belady_vs_lru, gate_ladder,
       layer_conformance, same_answer, step_trace, prefill_cost, memory_ladder,
       gb_read_shelf, rss_fidelity, sim_vs_measured, cache_wakeup, hit_metrics,
       io_share, trunk_cache_split, bytes_paradox, replication_noise,
       duplicate_config, quant_error, quant_by_layer_type, whats_missing,
       rust_vs_c_kernels, fma_instruction_mix, port_loc]

ok = 0
for fn in FNS:
    try:
        fn()
        ok += 1
    except Exception as e:
        print(f"  FAILED {fn.__name__}: {type(e).__name__}: {e}")
print(f"done: {ok}/{len(FNS)} plots rendered")
