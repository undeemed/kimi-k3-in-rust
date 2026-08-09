# Figures

Nine figures, all of them about **this port**. Every number in them was measured on one
machine: an Apple M5 with 10 cores running macOS, rustc 1.97.1, `cargo build --release`,
no `target-cpu=native`. The C side of any comparison was built on the same machine with
`clang -O3 -mcpu=native -ffp-contract=off` and OpenMP through libomp, with thread counts
pinned per run via `OMP_NUM_THREADS` and `RAYON_NUM_THREADS`.

| figure | what it shows |
| --- | --- |
| `real_checkpoint.png` | seconds per token on the released checkpoint, C against Rust, minimum and median |
| `end_to_end.png` | seconds per token for the whole engine, C against Rust, at one two four and ten threads |
| `rust_vs_c_kernels.png` | per-kernel speedup against a parity line, at one core and at ten |
| `perf_kernel_scaling.png` | throughput against thread count, four series |
| `perf_scaling_efficiency.png` | parallel speedup against each build's own single-thread time |
| `perf_bf16_instructions.png` | why bf16 differs: load and unpack against arithmetic, from the disassembly |
| `binary_sizes.png` | `target/release/k3` at 1,159,344 bytes, then the nine test binaries |
| `port_loc.png` | lines of Rust per module, largest first |
| `verification-ladder.png` | fixtures, the three oracle gates, then the byte-for-byte logits compare |

The six performance figures are a fair comparison: both builds threaded, both at their
real optimisation settings, thread count pinned, and the same input bytes producing the
same output bytes on both sides before any timing was compared. `end_to_end.png` runs the
complete decode loop rather than two kernels, and asserts identical token ids from both
binaries before reporting a single number. `rust_vs_c_kernels.png` is drawn as a ratio
against parity because absolute milliseconds put a 2.4 ms kernel beside an 18.6 ms one,
where the smaller pair collapses into two stubs and the real finding - that MXFP4 lands
exactly on parity, as an identical instruction mix predicts - reads as nothing at all.
`perf_bf16_instructions.png` traces the one gap that remains to a specific
instruction-selection choice in the bf16 unpack.

Raw timings: [`../data/perf-sweep.csv`](../data/perf-sweep.csv) (kernels),
[`../data/end-to-end.csv`](../data/end-to-end.csv) (synthetic whole engine) and
[`../data/real-checkpoint.csv`](../data/real-checkpoint.csv) (released weights).

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
