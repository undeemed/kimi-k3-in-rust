// SPDX-License-Identifier: Apache-2.0
//! bench_kernels.rs - where does the arithmetic actually go?
//!
//! Port of `benchmarks/bench_kernels.c`.
//!
//! The engine's decode time at the memory floor splits roughly 36 s trunk read, 11 s
//! expert read, 10 s compute. The read paths are now within 10-20% of what the device can
//! deliver, so the only lossless win left is the compute. Before optimising it, measure
//! which kernel owns it - guessing which loop is hot is how people spend a week making
//! something 2% faster.
//!
//! Dimensions here are the REAL ones, per token:
//!   bf16 trunk matmuls   the attention projections, latent down/up, shared experts and
//!                        the dense MLP. Sized from the actual layer shapes.
//!   MXFP4 expert matmuls 16 experts x 3 matrices x 92 layers, in latent space.
//!
//! Reports GFLOP/s and the projected per-token seconds for each, so the two can be
//! compared directly against the measured 10 s compute budget.

use std::time::Instant;

use k3::cfg::MXFP4_GROUP;
use k3::ops::{matmul_bf16, matmul_mxfp4};

/// Exact bits of a whole output vector.
///
/// Running this with and without the vector path and comparing these hashes is the ONLY
/// real proof that a vector path is bit-identical rather than merely close. A tolerance
/// check would happily pass a kernel that quietly reassociated the reduction, which is
/// exactly the mistake worth catching: this engine's claim is that its output equals the
/// reference, and "equals" has to mean equals.
///
/// It doubles as the consumer of every measured loop's result, so nothing measured here
/// can be optimised away.
fn fnv(label: &str, v: &[f32]) -> u64 {
    // Two hashes are printed. The first uses the real FNV-1a 64-bit offset basis,
    // 14695981039346656037. The second uses 1469598103934665603, which is the constant
    // bench_kernels.c actually carries: the FNV basis with its last digit dropped. That
    // makes it not FNV-1a, but it is what the reference prints, so it is the only value a
    // cross-build diff can be run against. Both are emitted because the point of the hash
    // is comparability, not hash pedigree.
    const FNV_BASIS: u64 = 14695981039346656037;
    const REF_BASIS: u64 = 1469598103934665603;
    let mut h = FNV_BASIS;
    let mut hr = REF_BASIS;
    for &f in v {
        let u = f.to_bits();
        for t in 0..4 {
            let byte = ((u >> (8 * t)) & 0xFF) as u64;
            h ^= byte;
            h = h.wrapping_mul(1099511628211);
            hr ^= byte;
            hr = hr.wrapping_mul(1099511628211);
        }
    }
    println!("             {label} OUTPUT FNV1a = {h:016x}");
    println!("             {label} same bytes, reference basis = {hr:016x}");
    h
}

/// xorshift32, the same generator and the same constants as the C benchmark, so both
/// builds see identical inputs.
struct Xs(u32);

impl Xs {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
}

fn fillf(p: &mut [f32], s: u32) {
    let mut r = Xs(s);
    for v in p.iter_mut() {
        let s = r.next();
        *v = ((s >> 8) as f32 / 8388608.0 - 1.0) * 0.05;
    }
}

fn fillb(p: &mut [u8], s: u32) {
    let mut r = Xs(s);
    for v in p.iter_mut() {
        let s = r.next();
        *v = (s >> 13) as u8;
    }
}

/// The C source fills the bf16 weight matrix as a raw byte array, so reproduce that byte
/// stream and assemble the u16 elements little-endian, which is what the C build reads
/// back on both supported architectures.
fn fillb16(p: &mut [u16], s: u32) {
    let mut r = Xs(s);
    for v in p.iter_mut() {
        let lo = (r.next() >> 13) as u8;
        let hi = (r.next() >> 13) as u8;
        *v = u16::from_le_bytes([lo, hi]);
    }
}

fn main() {
    println!("kernel benchmark at REAL Kimi K3 dimensions");
    // The C build decides this at compile time with `#ifdef __AVX2__`. This port selects
    // the vector path at RUNTIME (see `ops::dispatch::select`), so report what dispatch
    // itself would pick, using the same two feature checks.
    #[cfg(target_arch = "x86_64")]
    let avx2 =
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let avx2 = false;
    if avx2 {
        println!("built WITH AVX2\n");
    } else {
        println!("built WITHOUT AVX2 (scalar)\n");
    }

    let mut sum: u64 = 0;

    // ---------- bf16 trunk matmul: KDA q_proj, 7168 -> 12288 ----------
    {
        let (input, out) = (7168usize, 12288usize);
        let mut w = vec![0u16; input * out];
        let mut x = vec![0f32; input];
        let mut y = vec![0f32; out];
        fillb16(&mut w, 12345);
        fillf(&mut x, 999);

        matmul_bf16(&mut y, &x, &w, input); // warm
        let reps = 5;
        let t0 = Instant::now();
        for _ in 0..reps {
            matmul_bf16(&mut y, &x, &w, input);
        }
        let dt = t0.elapsed().as_secs_f64() / reps as f64;
        let gflop = 2.0 * input as f64 * out as f64 / 1e9;
        println!(
            "bf16 matmul  {:5} x {:<5}  {:7.2} ms  {:8.1} GFLOP/s",
            out,
            input,
            dt * 1e3,
            gflop / dt
        );

        // Per token the trunk is 56.74 G always-active params = 113.49 GFLOP of
        // multiply-add. Project from the rate just measured.
        sum ^= fnv("bf16 ", &y);
        println!(
            "             trunk is 56.74 G params/token -> {:.2} s/token at this rate",
            2.0 * 56.74e9 / 1e9 / (gflop / dt)
        );
    }

    // ---------- MXFP4 expert matmul: w1, latent 3584 -> inter 3072 ----------
    {
        let (input, rows, group) = (3584usize, 3072usize, MXFP4_GROUP);
        let (pcols, ngrp) = (input / 2, input / group);
        let mut pk = vec![0u8; rows * pcols];
        let sc = vec![127u8; rows * ngrp]; // scale 2^0, none skipped
        let mut x = vec![0f32; input];
        let mut y = vec![0f32; rows];
        fillb(&mut pk, 777);
        fillf(&mut x, 4242);

        matmul_mxfp4(&mut y, &x, &pk, &sc, input, group);
        let reps = 5;
        let t0 = Instant::now();
        for _ in 0..reps {
            matmul_mxfp4(&mut y, &x, &pk, &sc, input, group);
        }
        let dt = t0.elapsed().as_secs_f64() / reps as f64;
        let gflop = 2.0 * input as f64 * rows as f64 / 1e9;
        println!(
            "\nMXFP4 matmul {:5} x {:<5}  {:7.2} ms  {:8.1} GFLOP/s",
            rows,
            input,
            dt * 1e3,
            gflop / dt
        );

        // One expert is w1 + w3 (both 3072x3584) + w2 (3584x3072) = 3 of these.
        // 16 experts x 92 MoE layers per token.
        sum ^= fnv("mxfp4", &y);
        let per_tok = dt * 3.0 * 16.0 * 92.0;
        println!("             16 experts x 3 mats x 92 layers -> {per_tok:.2} s/token");
    }

    println!(
        "\nmeasured compute budget at the floor is about 10 s/token; whichever line\n\
         above dominates it is the one worth vectorising."
    );
    // Consume both measured outputs once more so no loop above can be elided.
    println!("checksum {sum:016x}");
}
