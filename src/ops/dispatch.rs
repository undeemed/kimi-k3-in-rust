// SPDX-License-Identifier: Apache-2.0
//! Per-row hot kernels, multiversioned.
//!
//! FLOATING-POINT CONTRACT. Ported from k3_ops.c, which builds with
//! `-ffp-contract=off` and explicit `fma()`. Rust never contracts, so the mapping is
//! mechanical: C `fma(a,b,c)` becomes `a.mul_add(b, c)` (the same IEEE
//! fusedMultiplyAdd), and C `a*b + c` stays a separate multiply and add.
//!
//! The lane partitions and reduction trees below are copied from the C source and are
//! part of the exactness contract, not an optimisation. `k3_matmul` keeps sixteen f64
//! accumulators partitioned by `i % 16` and reduces as
//! `((a0+a4)+(a8+a12)) + ((a1+a5)+(a9+a13)) ...`; the C AVX2 path reproduces exactly
//! that tree with four `__m256d`, which is why the two agree bit for bit.
//!
//! THIS PORT NEEDS NO INTRINSICS TO INHERIT THAT GUARANTEE. Each kernel body is one
//! `#[inline(always)]` generic function over the fixed lane partition, and the final
//! reduction is explicit scalar code on the accumulator array. Vectorising it can only
//! pack the sixteen INDEPENDENT dependency chains into registers; it cannot reassociate
//! within one, and it cannot touch the explicit tree. So the plain build, the AVX2 build
//! and the NEON build all produce the same bits as the C scalar path.
//!
//! On x86-64 the body is stamped out twice, once plain and once under
//! `#[target_feature(enable = "avx2,fma")]`, and selected once via a `OnceLock` of
//! function pointers. Dispatch is per output ROW, i.e. one indirect call per few
//! thousand multiply-adds, mirroring `k3_mmw`'s per-layer branch. On aarch64 FMA and
//! NEON are baseline, so the plain body already compiles to `fmla` and there is nothing
//! to select.

use std::sync::LazyLock;

// ------------------------------------------------------------------ MXFP4 tables ----

/// OCP MX E2M1: index by the 4-bit code; bit 3 is the sign. k3_ops.c:1027.
pub const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0,
    -6.0,
];

/// A whole byte to its two E2M1 values. Low nibble is the EVEN element; reversing it
/// yields right values in wrong places, which every statistic passes. k3_ops.c:1188.
pub const E2M1_PAIR: [[f32; 2]; 256] = {
    let mut t = [[0.0f32; 2]; 256];
    let mut b = 0usize;
    while b < 256 {
        t[b][0] = E2M1[b & 0x0F];
        t[b][1] = E2M1[b >> 4];
        b += 1;
    }
    t
};

/// E8M0 byte to its power of two, i.e. C's `ldexpf(1.0f, b - 127)`, built by bits so it
/// is a compile-time constant: exponent field `b` with a zero mantissa IS `2^(b-127)`
/// for every normal `b`. `b == 0` is `2^-127`, which is subnormal in f32
/// (`0x0040_0000`), and `b == 255` is NaN by the OCP MX spec and maps to zero so one
/// bad byte cannot poison a row. k3_ops.c:1208.
pub const E8M0: [f32; 256] = {
    let mut t = [0.0f32; 256];
    t[0] = f32::from_bits(0x0040_0000);
    let mut b = 1usize;
    while b < 255 {
        t[b] = f32::from_bits((b as u32) << 23);
        b += 1;
    }
    t[255] = 0.0;
    t
};

// -------------------------------------------------------------------- row kernels ----

/// `k3_matmul`'s row body: sixteen f64 accumulators partitioned by `i % 16`, reduced as
/// the C source's tree, then the scalar tail. k3_ops.c:248-267.
#[inline(always)]
fn dot_f32(row: &[f32], x: &[f32]) -> f64 {
    let n = row.len();
    let mut a = [0.0f64; 16];
    let mut i = 0usize;
    while i + 15 < n {
        for l in 0..16 {
            a[l] = (row[i + l] as f64).mul_add(x[i + l] as f64, a[l]);
        }
        i += 16;
    }
    let b0 = (a[0] + a[4]) + (a[8] + a[12]);
    let b1 = (a[1] + a[5]) + (a[9] + a[13]);
    let b2 = (a[2] + a[6]) + (a[10] + a[14]);
    let b3 = (a[3] + a[7]) + (a[11] + a[15]);
    let mut acc = (b0 + b1) + (b2 + b3);
    while i < n {
        acc = (row[i] as f64).mul_add(x[i] as f64, acc);
        i += 1;
    }
    acc
}

/// `k3_matmul_bf16`'s row body. bf16 -> f32 is a pure 16-bit left shift, no rounding,
/// so this computes with exactly the values an fp32 copy would have supplied; the
/// partition and tree match `dot_f32` so the two kernels agree to the bit.
/// k3_ops.c:1071-1126.
#[inline(always)]
fn dot_bf16(row: &[u16], x: &[f32]) -> f64 {
    let n = row.len();
    let mut a = [0.0f64; 16];
    let mut i = 0usize;
    while i + 15 < n {
        for l in 0..16 {
            a[l] = (bf16f(row[i + l]) as f64).mul_add(x[i + l] as f64, a[l]);
        }
        i += 16;
    }
    let b0 = (a[0] + a[4]) + (a[8] + a[12]);
    let b1 = (a[1] + a[5]) + (a[9] + a[13]);
    let b2 = (a[2] + a[6]) + (a[10] + a[14]);
    let b3 = (a[3] + a[7]) + (a[11] + a[15]);
    let mut acc = (b0 + b1) + (b2 + b3);
    while i < n {
        acc = (bf16f(row[i]) as f64).mul_add(x[i] as f64, acc);
        i += 1;
    }
    acc
}

/// `k3_matmul_q8`'s row body: f32 accumulation, four lanes by `i % 4`, reduced
/// `(a0+a1)+(a2+a3)`. Draft-model only, so this carries NO determinism contract; the
/// C AVX2 form uses its own natural reduction. k3_ops.c:1143-1181.
///
/// `row` is the int8 weights as raw bytes (the caller has already stripped the row's
/// leading f32 scale), cast per element rather than through a pointer transmute.
#[inline(always)]
fn dot_q8(row: &[u8], x: &[f32]) -> f32 {
    let n = row.len();
    let (mut a0, mut a1, mut a2, mut a3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let mut i = 0usize;
    while i + 3 < n {
        a0 += row[i] as i8 as f32 * x[i];
        a1 += row[i + 1] as i8 as f32 * x[i + 1];
        a2 += row[i + 2] as i8 as f32 * x[i + 2];
        a3 += row[i + 3] as i8 as f32 * x[i + 3];
        i += 4;
    }
    let mut acc = (a0 + a1) + (a2 + a3);
    while i < n {
        acc += row[i] as i8 as f32 * x[i];
        i += 1;
    }
    acc
}

/// `k3_matmul_mxfp4`'s row body: per 32-element group, expand the nibbles into `wf`,
/// dot with eight f64 lanes partitioned by `i % 8`, reduce `(s0+s4)+(s1+s5)...` the way
/// the C source does, then apply the group's E8M0 scale as a separate multiply and add.
///
/// PRECONDITIONS, unchecked in C and unchecked here: `group <= 64`, `in` even, `packed`
/// is `in/2` bytes per row, `scales` is `ceil(in/group)` bytes per row. k3_ops.c:1256.
#[inline(always)]
fn dot_mxfp4(packed: &[u8], scales: &[u8], x: &[f32], input: usize, group: usize) -> f64 {
    let ngrp = (input + group - 1) / group;
    let gbyte = group / 2;
    let mut acc = 0.0f64;

    for g in 0..ngrp {
        let sb = scales[g];
        if sb == 255 {
            continue; // NaN scale: contribute nothing
        }
        let pb = &packed[g * gbyte..];
        let xg = &x[g * group..];

        let n = core::cmp::min(input - g * group, group);

        // Expand the group to floats first, then a plain dot product: a table lookup in
        // the middle of the accumulation blocks vectorisation.
        let mut wf = [0.0f32; 64];
        let half = n >> 1;
        for j in 0..half {
            let pv = &E2M1_PAIR[pb[j] as usize];
            wf[2 * j] = pv[0];
            wf[2 * j + 1] = pv[1];
        }
        if n & 1 != 0 {
            wf[n - 1] = E2M1_PAIR[pb[half] as usize][0];
        }

        let mut s = [0.0f64; 8];
        let mut i = 0usize;
        while i + 7 < n {
            for l in 0..8 {
                s[l] = (wf[i + l] as f64).mul_add(xg[i + l] as f64, s[l]);
            }
            i += 8;
        }
        let b0 = s[0] + s[4];
        let b1 = s[1] + s[5];
        let b2 = s[2] + s[6];
        let b3 = s[3] + s[7];
        let mut sub = (b0 + b1) + (b2 + b3);
        while i < n {
            sub = (wf[i] as f64).mul_add(xg[i] as f64, sub);
            i += 1;
        }
        // Separate multiply and add: the C source is `acc += sub * (double)E8M0[sb]`.
        acc += sub * E8M0[sb as usize] as f64;
    }
    acc
}

/// bf16 -> f32. k3.h:274.
#[inline(always)]
pub fn bf16f(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

// ---------------------------------------------------------------------- dispatch ----

type DotF32 = fn(&[f32], &[f32]) -> f64;
type DotBf16 = fn(&[u16], &[f32]) -> f64;
type DotQ8 = fn(&[u8], &[f32]) -> f32;
type DotMxfp4 = fn(&[u8], &[u8], &[f32], usize, usize) -> f64;

pub(crate) struct RowKernels {
    pub f32: DotF32,
    pub bf16: DotBf16,
    pub q8: DotQ8,
    pub mxfp4: DotMxfp4,
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;

    #[target_feature(enable = "avx2,fma")]
    pub fn dot_f32(row: &[f32], x: &[f32]) -> f64 {
        super::dot_f32(row, x)
    }
    #[target_feature(enable = "avx2,fma")]
    pub fn dot_bf16(row: &[u16], x: &[f32]) -> f64 {
        super::dot_bf16(row, x)
    }
    #[target_feature(enable = "avx2,fma")]
    pub fn dot_q8(row: &[u8], x: &[f32]) -> f32 {
        super::dot_q8(row, x)
    }
    #[target_feature(enable = "avx2,fma")]
    pub fn dot_mxfp4(
        packed: &[u8],
        scales: &[u8],
        x: &[f32],
        input: usize,
        group: usize,
    ) -> f64 {
        super::dot_mxfp4(packed, scales, x, input, group)
    }
}

static KERNELS: LazyLock<RowKernels> = LazyLock::new(select);

#[inline]
pub(crate) fn kernels() -> &'static RowKernels {
    &KERNELS
}

fn select() -> RowKernels {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            return RowKernels {
                f32: avx2::dot_f32,
                bf16: avx2::dot_bf16,
                q8: avx2::dot_q8,
                mxfp4: avx2::dot_mxfp4,
            };
        }
    }
    plain()
}

/// The un-multiversioned bodies. Public to the crate so `tests/ops.rs` can assert the
/// selected kernels agree with these bit for bit, which is the Rust replacement for the
/// C build's scalar-versus-AVX2 gate.
pub(crate) fn plain() -> RowKernels {
    RowKernels {
        f32: dot_f32,
        bf16: dot_bf16,
        q8: dot_q8,
        mxfp4: dot_mxfp4,
    }
}
