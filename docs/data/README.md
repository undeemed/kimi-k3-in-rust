# Measurement data

## `perf-sweep.csv`

The C-against-Rust kernel sweep behind `docs/images/rust_vs_c_kernels.png`,
`perf_kernel_scaling.png` and `perf_scaling_efficiency.png`.

One machine: Apple M5, 10 cores, macOS. Both builds at their real settings, C with
`clang -O3 -mcpu=native -ffp-contract=off` and OpenMP through libomp, Rust with
`opt-level = 3`, thin LTO and rayon. Thread count pinned per run with `OMP_NUM_THREADS`
and `RAYON_NUM_THREADS`. Each row is the median of five runs of the benchmark, and the
benchmark itself does one warm call plus five timed ones.

Both builds were verified to compute identical output bytes before any timing was
compared: hashing the output vectors with the reference's own offset basis gives
`6c14a49941ce86d9` (bf16) and `a231061237b5579d` (mxfp4) on both sides.

To reproduce:

```sh
# C, with OpenMP
cd ../../../kimi-k3-in-c && make bin/bench_kernels
OMP_NUM_THREADS=10 ./bin/bench_kernels

# Rust
cd ../kimi-k3-in-rust && cargo build --release --benches
RAYON_NUM_THREADS=10 ./target/release/deps/kernels-*
```

Everything else the README quotes about the released checkpoint (the memory ladder, the
per-token times, the cache hit rates) is inherited from the reference implementation and
was measured on its hardware, not here. See its
[`docs/data/`](https://github.com/FareedKhan-dev/kimi-k3-in-c/tree/main/docs/data).
