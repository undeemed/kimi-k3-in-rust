# Figures

Eight figures, all of them about **this port**. Every number in them was measured on one
machine: an Apple M5 with 10 cores running macOS, rustc 1.97.1, `cargo build --release`,
no `target-cpu=native`. The C side of any comparison was built on the same machine with
`clang -O3 -mcpu=native -ffp-contract=off` and OpenMP through libomp, with thread counts
pinned per run via `OMP_NUM_THREADS` and `RAYON_NUM_THREADS`.

| figure | what it shows |
| --- | --- |
| `rust_vs_c_kernels.png` | the two kernel benchmarks in both languages, at one core and at ten |
| `perf_kernel_scaling.png` | throughput against thread count, four series |
| `perf_scaling_efficiency.png` | parallel speedup against each build's own single-thread time |
| `perf_bf16_instructions.png` | why bf16 differs: load and unpack against arithmetic, from the disassembly |
| `fma_instruction_mix.png` | aarch64 `fmla.2d` per kernel, 8/8/4 on both sides |
| `binary_sizes.png` | `target/release/k3` at 1,159,344 bytes, then the nine test binaries |
| `port_loc.png` | lines of Rust per module, largest first |
| `verification-ladder.png` | fixtures, the three oracle gates, then the byte-for-byte logits compare |

The four performance figures are a fair comparison: both builds threaded, both at their
real optimisation settings, thread count pinned, and the same input bytes producing the
same output bytes on both sides before any timing was compared. `rust_vs_c_kernels.png`
shows one core and ten side by side so nothing rests on a threading difference, and
`perf_bf16_instructions.png` traces the one gap that remains to a specific
instruction-selection choice in the bf16 unpack.

Raw timings: [`../data/perf-sweep.csv`](../data/perf-sweep.csv).

## Regenerating

Needs Python with matplotlib and numpy (rendered here with 3.9.4 and 2.0.2), plus Node for
the Mermaid CLI.

```sh
cd docs/images
python _plots.py
npx -y -p @mermaid-js/mermaid-cli mmdc -i verification-ladder.mmd \
    -o verification-ladder.png -b transparent -s 2
```

## What is not here

The C reference project has 115 figures explaining Kimi K3 itself: the MXFP4 encoding, the
fit ledger, the KDA and MLA algebra, the router, the shard census, the memory ladder, the
cache traces. Those describe the model and the released weights rather than this port, and
they were measured on the reference's hardware. They are not reproduced or re-hosted here.

Read them where they were made: <https://github.com/FareedKhan-dev/kimi-k3-in-c>

The style helpers in `_plots.py` (`style`, `bare`, `lollipop`, `save`, the palette) are
derived from that project's `docs/images/_plots.py`, which is Apache 2.0. See
[`NOTICE`](../../NOTICE).
