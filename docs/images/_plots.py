"""Render this port's own figures to white-background PNGs.

Every number here was measured on one machine: an Apple M5 with 10 cores running macOS,
rustc 1.97.1, `cargo build --release`, no `target-cpu=native`. The C side of any comparison
was built on the same machine with `clang -O3 -mcpu=native -ffp-contract=off` and OpenMP
through libomp, and thread counts are pinned per run.

The style helpers below (`style`, `bare`, `lollipop`, `save`, the palette) are derived from
`docs/images/_plots.py` in the C reference project, which is Apache 2.0; see NOTICE. The
reference's own figures - the ones that explain Kimi K3, MXFP4, KDA, MLA, the trunk and the
cache - are NOT reproduced here. They live in the original:
https://github.com/FareedKhan-dev/kimi-k3-in-c

Usage:  python _plots.py
"""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def lollipop(ax, labels, values, colors, fmt="{:,.0f}", xmax_pad=1.25, unit=""):
    ys = np.arange(len(labels))[::-1]
    ax.hlines(ys, 0, values, color=colors, linewidth=2.2, zorder=2)
    ax.scatter(values, ys, s=110, color=colors, zorder=3)
    for y, v, c in zip(ys, values, colors):
        ax.text(
            v * 1.03,
            y,
            (fmt.format(v)) + unit,
            va="center",
            fontsize=10.5,
            color=INK,
            fontweight="bold",
        )
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
        ax.text(
            hi,
            y + 0.26,
            fmt.format(hi),
            ha="center",
            fontsize=9,
            color=INK,
            fontweight="bold",
        )
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=10)


def save(fig, name):
    path = OUT / f"{name}.png"
    fig.savefig(path, facecolor="white", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  rendered {path.name}")


# ============================================================ THE FIT LEDGER


SWEEP_THREADS = [1, 2, 4, 6, 8, 10]
SWEEP_MS = {
    ("C", "bf16"): [18.58, 10.18, 6.20, 4.93, 4.26, 4.33],
    ("Rust", "bf16"): [9.21, 4.93, 2.87, 2.16, 1.92, 1.81],
    ("C", "mxfp4"): [2.39, 1.20, 0.75, 0.72, 0.62, 0.56],
    ("Rust", "mxfp4"): [2.42, 1.32, 0.73, 0.71, 0.60, 0.57],
}
SWEEP_GF = {
    ("C", "bf16"): [9.5, 17.3, 28.4, 35.7, 41.3, 40.7],
    ("Rust", "bf16"): [19.1, 35.7, 61.4, 81.6, 91.6, 97.2],
    ("C", "mxfp4"): [9.2, 18.3, 29.2, 30.7, 35.4, 39.4],
    ("Rust", "mxfp4"): [9.1, 16.7, 30.1, 31.0, 36.4, 38.6],
}


# The whole engine, not two kernels. Read straight from the CSV that
# tools/bench_end_to_end.py wrote, rather than transcribed: deriving a speedup from
# already-rounded seconds is how 1.144x becomes 1.15x. Per-step seconds on a synthetic
# 13-layer model at hidden 2048 (9 KDA / 4 MLA / 1 dense / 12 MoE), prefill excluded,
# median of five runs. Both binaries read the same bytes and emitted identical token ids.
def _load_e2e():
    rows = {}
    with open(OUT.parent / "data" / "end-to-end.csv") as f:
        for r in csv.DictReader(f):
            rows[(r["lang"], int(r["threads"]))] = float(r["s_per_token"])
    threads = sorted({t for _, t in rows})
    return threads, {lang: [rows[(lang, t)] for t in threads] for lang in ("C", "Rust")}


E2E_THREADS, E2E_S = _load_e2e()


# The released checkpoint, layers 0-1, from docs/data/real-checkpoint.csv. This one
# streams MXFP4 experts off disk on a machine whose page cache cannot hold the shards,
# so the median carries I/O stalls and the MINIMUM is the compute comparison.
def _load_real():
    rows = {}
    with open(OUT.parent / "data" / "real-checkpoint.csv") as f:
        for r in csv.DictReader(f):
            rows[(r["lang"], int(r["threads"]))] = (
                float(r["s_per_token_min"]),
                float(r["s_per_token_median"]),
            )
    threads = sorted({t for _, t in rows})
    return threads, rows


REAL_THREADS, REAL_S = _load_real()


def binary_sizes():
    """LOLLIPOP. cargo build --release on this machine, then ls -l on what it
    produced: the engine first, then the nine test binaries that gate it."""
    names = [
        "k3  (the engine)",
        "real_layer",
        "model_oracle",
        "ops",
        "expert",
        "cache",
        "scale",
        "tok",
        "st",
        "cfg",
    ]
    kb = [
        v / 1024.0
        for v in [
            1159344,
            1367104,
            1301408,
            1297648,
            1172064,
            1151712,
            1121360,
            1083248,
            1064416,
            988464,
        ]
    ]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    lollipop(ax, names, kb, [GREEN] + [GRAY] * 9, fmt="{:.0f}", unit=" KB")
    style(ax, "A 1.16 MB binary that runs a 1.56 TB model", "kilobytes", grid="x")
    ax.text(
        0,
        -0.16,
        "Every test binary statically links the whole engine plus "
        "the harness and the fixtures it needs,\nso they all land inside the "
        "same megabyte as the engine itself.",
        transform=ax.transAxes,
        fontsize=9.5,
        color=MUTE,
        va="top",
    )
    save(fig, "binary_sizes")


def real_checkpoint():
    """GROUPED BARS + SPEEDUP, on the released weights. Minimum rather than median:
    this workload streams experts from disk on a box whose page cache cannot hold the
    shards, so medians measure the storage and the minimum measures the code."""
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(10.8, 4.1), gridspec_kw={"width_ratios": [1.55, 1]}
    )
    xs = np.arange(len(REAL_THREADS), dtype=float)
    c = np.array([REAL_S[("C", t)][0] for t in REAL_THREADS])
    r = np.array([REAL_S[("Rust", t)][0] for t in REAL_THREADS])
    cm = np.array([REAL_S[("C", t)][1] for t in REAL_THREADS])
    rm = np.array([REAL_S[("Rust", t)][1] for t in REAL_THREADS])

    ax.bar(xs - 0.19, c, width=0.34, color=GRAY, edgecolor="white", label="C")
    ax.bar(xs + 0.19, r, width=0.34, color=GREEN, edgecolor="white", label="Rust")
    # The median sits above each bar as a thin cap, so the I/O spread stays visible
    # rather than being quietly dropped.
    ax.plot(
        xs - 0.19,
        cm,
        "_",
        color=INK,
        markersize=13,
        markeredgewidth=1.6,
        label="median (carries I/O stalls)",
    )
    ax.plot(xs + 0.19, rm, "_", color=INK, markersize=13, markeredgewidth=1.6)
    # Label above whichever is higher, the bar or its median cap, so the two never collide.
    for x, v, m in zip(xs - 0.19, c, cm):
        ax.text(x, max(v, m) + 0.025, f"{v:.2f}", ha="center", fontsize=9.5, color=MUTE)
    for x, v, m in zip(xs + 0.19, r, rm):
        ax.text(
            x,
            max(v, m) + 0.025,
            f"{v:.2f}",
            ha="center",
            fontsize=9.5,
            color=INK,
            fontweight="bold",
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{t} thread" if t == 1 else f"{t} threads" for t in REAL_THREADS]
    )
    ax.set_ylim(0, max(cm.max(), rm.max()) * 1.12)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    style(ax, "Seconds per token, released weights", None, "seconds per token")

    ax2.axhline(1.0, color="#9ca3af", linewidth=1.2, linestyle="--", zorder=1)
    ax2.text(xs[-1] + 0.45, 1.02, "parity", fontsize=9, color=MUTE, ha="right")
    sp = c / r
    ax2.bar(xs, sp, width=0.5, color=GREEN, edgecolor="white", zorder=2)
    for x, v in zip(xs, sp):
        ax2.text(
            x,
            v + 0.015,
            f"{v:.2f}x",
            ha="center",
            fontsize=10,
            color=INK,
            fontweight="bold",
        )
    ax2.set_xticks(xs)
    ax2.set_xticklabels([str(t) for t in REAL_THREADS])
    ax2.set_ylim(0.9, max(sp) * 1.10)
    style(ax2, "Rust speedup over C", "threads")

    fig.suptitle(
        "The released Kimi K3 checkpoint: real bf16 trunk, real MXFP4 experts",
        fontsize=13,
        color=INK,
        fontweight="bold",
        y=1.04,
    )
    fig.text(
        0.5,
        -0.09,
        "Layers 0 and 1 of 93, 8 tokens, prefill excluded, 7 runs. Bars are the minimum "
        "and the caps the median:\nthe shards do not fit in page cache, so medians "
        "measure the disk. Identical token ids in every run.",
        ha="center",
        fontsize=9.5,
        color=MUTE,
    )
    save(fig, "real_checkpoint")


def end_to_end():
    """GROUPED BARS + SPEEDUP. The headline: the whole engine, not two kernels."""
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(10.8, 4.1), gridspec_kw={"width_ratios": [1.55, 1]}
    )
    xs = np.arange(len(E2E_THREADS), dtype=float)
    c, r = np.array(E2E_S["C"]), np.array(E2E_S["Rust"])
    ax.bar(xs - 0.19, c, width=0.34, color=GRAY, edgecolor="white", label="C")
    ax.bar(xs + 0.19, r, width=0.34, color=GREEN, edgecolor="white", label="Rust")
    for x, v in zip(xs - 0.19, c):
        ax.text(x, v + 0.0022, f"{v:.3f}", ha="center", fontsize=9.5, color=MUTE)
    for x, v in zip(xs + 0.19, r):
        ax.text(
            x,
            v + 0.0022,
            f"{v:.3f}",
            ha="center",
            fontsize=9.5,
            color=INK,
            fontweight="bold",
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{t} thread" if t == 1 else f"{t} threads" for t in E2E_THREADS]
    )
    ax.set_ylim(0, max(c) * 1.22)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    style(ax, "Seconds per token, whole engine", None, "seconds per token")

    # Same numbers as a ratio, so the gap is readable without eyeballing bar heights.
    ax2.axhline(1.0, color="#9ca3af", linewidth=1.2, linestyle="--", zorder=1)
    ax2.text(xs[-1] + 0.45, 1.015, "parity", fontsize=9, color=MUTE, ha="right")
    sp = c / r
    ax2.bar(xs, sp, width=0.5, color=GREEN, edgecolor="white", zorder=2)
    for x, v in zip(xs, sp):
        ax2.text(
            x,
            v + 0.012,
            f"{v:.2f}x",
            ha="center",
            fontsize=10,
            color=INK,
            fontweight="bold",
        )
    ax2.set_xticks(xs)
    ax2.set_xticklabels([str(t) for t in E2E_THREADS])
    ax2.set_ylim(0.9, max(sp) * 1.12)
    style(ax2, "Rust speedup over C", "threads")

    fig.suptitle(
        "The whole engine: 13 layers of KDA, MLA, dense and MoE, decoding token by token",
        fontsize=13,
        color=INK,
        fontweight="bold",
        y=1.04,
    )
    fig.text(
        0.5,
        -0.09,
        "Synthetic checkpoint at hidden 2048, prefill excluded, median of five runs. "
        "Both binaries read the same\nbytes and emitted identical token ids. "
        "Past four threads both hit the memory wall and the gap narrows.",
        ha="center",
        fontsize=9.5,
        color=MUTE,
    )
    save(fig, "end_to_end")


def rust_vs_c_kernels():
    """SPEEDUP BARS against a parity line. Absolute milliseconds put a 2.4 ms kernel
    next to an 18.6 ms one, where the smaller pair collapses into two stubs and the
    real finding - that MXFP4 lands ON parity, exactly as an identical instruction mix
    predicts - reads as nothing at all. A ratio puts both kernels on one scale."""
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    kernels = [
        ("bf16", "bf16 matmul\n12288 x 7168"),
        ("mxfp4", "MXFP4 matmul\n3072 x 3584"),
    ]
    ys = np.arange(len(kernels), dtype=float)[::-1]

    ax.axvline(1.0, color="#9ca3af", linewidth=1.3, linestyle="--", zorder=1)
    ax.text(1.04, -0.50, "parity", ha="left", fontsize=9.5, color=MUTE)
    for off, ti, col, lab in [
        (0.17, 0, TEAL, "one core"),
        (-0.17, 5, GREEN, "all 10 cores"),
    ]:
        sp = [SWEEP_MS[("C", k)][ti] / SWEEP_MS[("Rust", k)][ti] for k, _ in kernels]
        ax.barh(
            ys + off, sp, height=0.29, color=col, edgecolor="white", label=lab, zorder=2
        )
        for y, v, (k, _) in zip(ys + off, sp, kernels):
            cm, rm = SWEEP_MS[("C", k)][ti], SWEEP_MS[("Rust", k)][ti]
            ax.text(
                v + 0.045,
                y,
                f"{v:.2f}x    {cm:.2f} -> {rm:.2f} ms",
                va="center",
                fontsize=9.5,
                color=INK,
                fontweight="bold",
            )
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for _, lab in kernels], fontsize=10)
    ax.set_ylim(-0.62, 1.62)
    ax.set_xlim(0, 3.1)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    style(
        ax, "Rust speedup over C, per kernel", "times faster than the C build", grid="x"
    )
    fig.text(
        0.5,
        -0.10,
        "Median of five runs, same inputs, byte-identical outputs. MXFP4 sits on the "
        "parity line, which is the\nexpected result: both builds issue the same "
        "instructions for it. bf16 is the one that diverges, and\nthe cause is the "
        "unpack, not the arithmetic.",
        ha="center",
        fontsize=9.5,
        color=MUTE,
    )
    save(fig, "rust_vs_c_kernels")


def perf_kernel_scaling():
    """LINES. Throughput against thread count, four series, both languages."""
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    for key, col, ls, mk, lab in [
        (("Rust", "bf16"), GREEN, "-", "o", "Rust  bf16"),
        (("C", "bf16"), GRAY, "-", "s", "C     bf16"),
        (("Rust", "mxfp4"), TEAL, "--", "o", "Rust  MXFP4"),
        (("C", "mxfp4"), ORANGE, "--", "s", "C     MXFP4"),
    ]:
        ax.plot(
            SWEEP_THREADS,
            SWEEP_GF[key],
            color=col,
            linestyle=ls,
            marker=mk,
            linewidth=2.0,
            markersize=6,
            label=lab,
        )
    ax.text(
        1.15,
        71,
        "one core: 19.1 against 9.5 GFLOP/s.\nThe 2x is codegen, not threading.",
        fontsize=9.5,
        color=GREEN,
        fontweight="bold",
    )
    ax.annotate(
        "MXFP4 has no bf16 widening,\nand the two builds tie",
        xy=(8, 36.4),
        xytext=(4.6, 8),
        fontsize=9.5,
        color=MUTE,
        arrowprops=dict(arrowstyle="->", color=MUTE),
    )
    ax.set_xticks(SWEEP_THREADS)
    ax.set_ylim(0, 108)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    style(
        ax,
        "Throughput against thread count, both builds at their real settings",
        "threads (OMP_NUM_THREADS / RAYON_NUM_THREADS)",
        "GFLOP/s",
    )
    save(fig, "perf_kernel_scaling")


def perf_scaling_efficiency():
    """LINES against the ideal. Does either threading runtime scale better?"""
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    ts = np.array(SWEEP_THREADS, float)
    ax.plot(ts, ts, color="#d1d5db", linestyle=":", linewidth=1.6, label="ideal")
    for key, col, mk, lab in [
        (("Rust", "bf16"), GREEN, "o", "Rust  bf16  (rayon)"),
        (("C", "bf16"), GRAY, "s", "C     bf16  (OpenMP)"),
        (("Rust", "mxfp4"), TEAL, "o", "Rust  MXFP4 (rayon)"),
        (("C", "mxfp4"), ORANGE, "s", "C     MXFP4 (OpenMP)"),
    ]:
        ms = np.array(SWEEP_MS[key], float)
        ax.plot(
            ts, ms[0] / ms, color=col, marker=mk, linewidth=2.0, markersize=6, label=lab
        )
    ax.set_xticks(SWEEP_THREADS)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    style(
        ax,
        "Parallel speedup over each build's own single-thread time",
        "threads",
        "speedup",
    )
    ax.text(
        0,
        -0.20,
        "Both runtimes scale about the same way and both fall off the ideal "
        "past six cores, which is a memory-bandwidth\nwall rather than a scheduler "
        "difference. So the bf16 gap in the previous chart is not a threading effect.",
        transform=ax.transAxes,
        fontsize=9.5,
        color=MUTE,
        va="top",
    )
    save(fig, "perf_scaling_efficiency")


def perf_bf16_instructions():
    """STACKED BARS. Why bf16 differs: the two kernel bodies, disassembled."""
    groups = ["load", "bf16 -> f32 widen", "f32 -> f64 widen", "vector FMA"]
    # Opcode histogram of the two kernel bodies on this arm64 machine: the C
    # OpenMP-outlined k3_matmul_bf16 loop, and the Rust neon::dot_bf16 body.
    c_counts = [
        4 + 6 + 9 + 1,
        8 + 8,
        16,
        8,
    ]  # ld2/ldp/ldr/ldrh | shl.2s+mov.h | fcvtl | fmla.2d
    r_counts = [
        5 + 1 + 1,
        4,
        8 + 8,
        8,
    ]  # ldp/ldr/ldrh | shll.4s | fcvtl+fcvtl2 | fmla.2d
    cols = [BLUE, RED, AMBER, GREEN]
    rows = [("Rust  neon::dot_bf16", r_counts), ("C     k3_matmul_bf16", c_counts)]
    fig, ax = plt.subplots(figsize=(9.8, 3.4))
    for row, (_, counts) in enumerate(rows):
        left = 0
        for n, col in zip(counts, cols):
            ax.barh([row], [n], left=left, color=col, edgecolor="white", height=0.52)
            if n >= 4:
                ax.text(
                    left + n / 2,
                    row,
                    str(n),
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white",
                    fontweight="bold",
                )
            left += n
        ax.text(
            left + 1.0,
            row,
            f"{left} total",
            va="center",
            fontsize=10,
            color=INK,
            fontweight="bold",
        )
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in cols]
    ax.legend(
        handles,
        groups,
        frameon=False,
        fontsize=9,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.32),
    )
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.set_xlim(0, 66)
    style(ax, None, "instructions in the kernel body", grid="x")
    ax.text(
        0,
        -0.30,
        "The vector FMA count is identical, 8 either way, and so is the f32 "
        "to f64 widening at 16 lanes.\nThe difference is the bf16 unpack: clang chose "
        "2-lane shifts with per-halfword inserts off a deinterleaving\nload, where the "
        "hand-written NEON does 4-lane shifts off a plain paired load. That is the "
        "whole 2x.",
        transform=ax.transAxes,
        fontsize=9.5,
        color=MUTE,
        va="top",
    )
    save(fig, "perf_bf16_instructions")


def port_loc():
    """LOLLIPOP. wc -l over src/ in the Rust port, largest module first."""
    mods = [
        "main.rs",
        "tok/unicode.rs",
        "ops/mod.rs",
        "bind.rs",
        "tok/mod.rs",
        "cache.rs",
        "trunk.rs",
        "st.rs",
        "ops/dispatch.rs",
        "cfg.rs",
        "tok/loader.rs",
        "load.rs",
        "io_util.rs",
        "lib.rs",
    ]
    loc = [2224, 2121, 1805, 1290, 917, 764, 732, 632, 580, 417, 364, 304, 233, 77]
    colors = [AMBER if m == "tok/unicode.rs" else BLUE for m in mods]
    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    lollipop(ax, mods, loc, colors, xmax_pad=1.22, fmt="   {:,.0f}")
    ax.annotate(
        "transcribed Unicode tables,\nnot logic",
        xy=(1700, 11.93),
        xytext=(1180, 8.4),
        fontsize=9.5,
        color=AMBER,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=AMBER),
    )
    style(
        ax,
        "Where the port's lines went, and the second largest module is a table",
        "lines of Rust",
        grid="x",
    )
    ax.text(
        0,
        -0.11,
        "Fourteen modules here, 12,673 lines across all of src/, and "
        "4,582 more in tests and benches.\nExactly four dependencies: libc, "
        "rayon, serde, serde_json.",
        transform=ax.transAxes,
        fontsize=9.5,
        color=MUTE,
        va="top",
    )
    save(fig, "port_loc")


FNS = [
    binary_sizes,
    real_checkpoint,
    port_loc,
    end_to_end,
    rust_vs_c_kernels,
    perf_kernel_scaling,
    perf_scaling_efficiency,
    perf_bf16_instructions,
]

ok = 0
for fn in FNS:
    try:
        fn()
        ok += 1
    except Exception as e:
        print(f"  FAILED {fn.__name__}: {type(e).__name__}: {e}")
print(f"done: {ok}/{len(FNS)} plots rendered")
