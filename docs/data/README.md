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

## `real-checkpoint.csv`

The released weights, behind `docs/images/real_checkpoint.png`. Layers 0 and 1 of 93, on
the real bf16 trunk and real MXFP4 experts, from 3 of the 96 shards (24 GB). 8 tokens,
prefill excluded, 7 runs per configuration after one discarded warm run. Both binaries
read the same shards and emitted identical token ids in every run.

**Two columns, and the minimum is the one to read.** The shard set is 24 GB on a 24 GB
machine, so the page cache cannot hold it and every run does real disk reads. That leaves
long stalls in the distribution: Rust's one-thread median is 1.24 s against a 0.52 s
floor, and an earlier median-of-medians briefly showed Rust losing at four threads purely
on one bad run. `s_per_token_min` is the run least disturbed by storage and is the code
comparison; `s_per_token_median` is kept so the spread is not hidden.

To reproduce, fetch the three shards named in the README, then:

```sh
A="$HOME/k3model --ids 1,2,3,4,5,6,7,8 --gen 8 --layers 2 --incremental --cache-gb 1"
OMP_NUM_THREADS=10   ../../../kimi-k3-in-c/bin/k3 $A --out /tmp/c.json
RAYON_NUM_THREADS=10 ./target/release/k3        $A --out /tmp/rust.json
```

The C build needs [`docs/patches/c-macos-pread-cap.patch`](../patches/) first on macOS,
or it cannot read a tensor over 2 GiB at all.

## `end-to-end.csv`

The whole-engine sweep behind `docs/images/end_to_end.png`: per-token seconds for the
complete decode loop, not just two kernels. Same machine and same build settings as above,
median of five runs, `s_min` and `s_max` giving the spread across those runs.

The released checkpoint is 1.56 TB and is not available here, so the model is synthetic:
13 layers at hidden 2048 with the real layer mix (9 KDA, 4 MLA, one dense, twelve MoE,
eight routed MXFP4 experts each) and random weights. What is being compared is arithmetic
throughput, not model quality.

**Both binaries read the same checkpoint bytes and emitted identical token ids**, which
`bench_end_to_end.py` asserts before it reports any timing; a speed comparison between two
engines that computed different things is worthless. Per-step seconds come from the
engine's own STEP table, so checkpoint load is excluded, and step 0 is dropped because
prefill scales with the prompt rather than with one token.

This model is fully resident in RAM. The real engine is not - the reference measures
40.9-60.6% of wall clock in storage waits - so this ratio is an arithmetic-half result and
does not carry over to the released checkpoint unscaled. See the README for the Amdahl
bound.

To reproduce:

```sh
cd ../../../kimi-k3-in-c && make bin/k3
cd ../kimi-k3-in-rust && cargo build --release --bin k3
tools/gen_bench_model.py /tmp/k3-bench2k 2048
tools/bench_end_to_end.py /tmp/k3-bench2k docs/data/end-to-end.csv
```

Everything else the README quotes about the released checkpoint (the memory ladder, the
per-token times, the cache hit rates) is inherited from the reference implementation and
was measured on its hardware, not here. See its
[`docs/data/`](https://github.com/FareedKhan-dev/kimi-k3-in-c/tree/main/docs/data).
