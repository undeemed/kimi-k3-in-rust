# Figures

117 PNGs: 45 equations, 38 data plots, 33 diagrams, and one hero image.
Three generators produce 116 of them and the hero is copied from the reference project.
Each generator is a copy of the one in the C reference project (`kimi-k3-in-c/docs/images/`) with only the implementation-specific figures rewritten.

## Regenerating everything

Needs Python with matplotlib, numpy and pillow (rendered here with matplotlib 3.9.4 and numpy 2.0.2), plus Node for the Mermaid CLI.

```sh
cd docs/images

python _eq.py                      # 45 equation PNGs, straight from mathtext
python _plots.py                   # 38 data plots
python _gen.py                     # writes the 33 .mmd sources

for f in *.mmd; do                 # renders each .mmd to its PNG
  npx -y -p @mermaid-js/mermaid-cli mmdc -i "$f" -o "${f%.mmd}.png" -b transparent -s 2
done
```

`main_architecture.png` is not generated.
It is copied verbatim from the reference project:

```sh
cp ../../../kimi-k3-in-c/docs/images/main_architecture.png .
```

The reference project also ships `main_architecture_with_spongbob.png` and `patrick_pray.png`.
Both are jokes and neither is copied here.

## Measured on this port

These nine figures describe the Rust implementation, and every number in them was measured on one machine: an Apple M5 with 10 cores running macOS, rustc 1.97.1, `cargo build --release`, no `target-cpu=native`.
The C side of any comparison was built on the same machine with `clang -O3 -mcpu=native -ffp-contract=off` and no OpenMP, because libomp is absent there.

| Figure | What it shows | Status |
| --- | --- | --- |
| `binary_sizes.png` | `target/release/k3` at 1,159,344 bytes, then the nine test binaries | rewritten |
| `rust_vs_c_kernels.png` | the two kernel benchmarks at the same shapes in both languages, in ms and GFLOP/s | new |
| `fma_instruction_mix.png` | aarch64 `fmla.2d` per kernel, 8/8/4 on both sides | new |
| `port_loc.png` | lines of Rust per module, largest first | new |
| `build-flow.png` | 14 modules, `cargo build --release`, 9 test binaries, 43 passed and 0 failed | rewritten |
| `kernel-contract.png` | the portable, NEON and AVX2 paths reaching one identical output | rewritten |
| `verification-ladder.png` | fixtures, then the three oracle gates, then the byte-for-byte logits compare against the C build | new |
| `recap.png` | inherited except the engine node, now `./target/release/k3` at about 1.16 MB | one node rewritten |
| `eq_accum_order.png` | the equation is inherited, the caption now names all three code paths | caption widened |

`rust_vs_c_kernels.png` is a build-to-build comparison and says so on its face.
The C build runs single-threaded because libomp is unavailable on this machine, and the Rust build uses rayon across all 10 cores, so nothing in that figure is a per-core claim.

## Inherited from the reference implementation

The other 108 figures are unchanged, and their numbers were measured by the C project on its own hardware against the released checkpoint.
They describe Kimi K3 and the weights that ship with it, not the language the engine is written in, so re-deriving them for this port would have produced the same values or no values at all.
That covers:

- All 45 equations except the caption noted above: the MXFP4 encoding, the fit ledger, the KDA and MLA algebra, the router, the streaming and cache arithmetic, the tolerance budget.
- Every model and checkpoint plot: `fit_cascade`, `bytes_census`, `shard_sizes`, `mxfp4_savings`, `kv_layout`, `context_scaling`, `activation_spikes`, `expert_reuse`, `quant_error`, `quant_by_layer_type`, `whats_missing`.
- Every streaming, cache and performance plot: `preset_ladder`, `pack_progress`, `resident_vs_streamed`, `storage_regimes`, `belady_vs_lru`, `memory_ladder`, `gb_read_shelf`, `rss_fidelity`, `sim_vs_measured`, `cache_wakeup`, `hit_metrics`, `io_share`, `trunk_cache_split`, `bytes_paradox`, `replication_noise`, `duplicate_config`, `same_answer`, `step_trace`, `prefill_cost`, `layer_conformance`, `gate_ladder`.
- `token_time_split.png`, which splits one token into 36 s of trunk read, 11 s of expert read and 10 s of compute. That split is a property of the streaming design measured on the released checkpoint, and this port cannot re-measure it without the 1.56 TB weights.
- `roundtrip_sizes.png`, whose four points are the reference project's own files tokenized and decoded back. The byte and token counts are tokenizer properties, so they carry over unchanged.
- The remaining 29 diagrams, from `big-picture` through `hosts-confound`.
- `main_architecture.png`, the hero image.

## Where the two sets meet

`oracle-ladder.png` is inherited and describes the C project's validation against torch on the real checkpoint.
`verification-ladder.png` is new and describes what this port actually proves on a machine with no checkpoint: kernel fixtures inside the manifest tolerances, exact bit equality between the bf16 and f32 kernels and between all three dispatch paths, the three tiny-model oracle gates, and finally the 1024 bytes of final-position logits comparing equal to the C binary byte for byte.
