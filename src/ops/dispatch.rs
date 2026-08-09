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
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
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
///
/// The 16-element step is expressed with `chunks_exact` rather than an index loop
/// because that is what lets this vectorise: with a proven length of 16 the per-element
/// bounds checks disappear, and without them LLVM packs the sixteen independent
/// accumulator chains into vector registers. The arithmetic is untouched, so this stays
/// bit-identical to the C scalar path and to the plain instantiation.
#[inline(always)]
fn dot_f32(row: &[f32], x: &[f32]) -> f64 {
    let n = row.len();
    let x = &x[..n];
    let mut a = [0.0f64; 16];
    for (r16, x16) in row.chunks_exact(16).zip(x.chunks_exact(16)) {
        for l in 0..16 {
            a[l] = (r16[l] as f64).mul_add(x16[l] as f64, a[l]);
        }
    }
    let b0 = (a[0] + a[4]) + (a[8] + a[12]);
    let b1 = (a[1] + a[5]) + (a[9] + a[13]);
    let b2 = (a[2] + a[6]) + (a[10] + a[14]);
    let b3 = (a[3] + a[7]) + (a[11] + a[15]);
    let mut acc = (b0 + b1) + (b2 + b3);
    let tail = n - n % 16;
    for i in tail..n {
        acc = (row[i] as f64).mul_add(x[i] as f64, acc);
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
    let x = &x[..n];
    let mut a = [0.0f64; 16];
    let rc = row.chunks_exact(16);
    let xc = x.chunks_exact(16);
    for (r16, x16) in rc.zip(xc) {
        for l in 0..16 {
            a[l] = (bf16f(r16[l]) as f64).mul_add(x16[l] as f64, a[l]);
        }
    }
    let b0 = (a[0] + a[4]) + (a[8] + a[12]);
    let b1 = (a[1] + a[5]) + (a[9] + a[13]);
    let b2 = (a[2] + a[6]) + (a[10] + a[14]);
    let b3 = (a[3] + a[7]) + (a[11] + a[15]);
    let mut acc = (b0 + b1) + (b2 + b3);
    let tail = n - n % 16;
    for i in tail..n {
        acc = (bf16f(row[i]) as f64).mul_add(x[i] as f64, acc);
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
    let ngrp = input.div_ceil(group);
    let gbyte = group / 2;
    let mut acc = 0.0f64;

    for g in 0..ngrp {
        let sb = scales[g];
        if sb == 255 {
            continue; // NaN scale: contribute nothing
        }
        let n = core::cmp::min(input - g * group, group);
        // Slice both operands to exactly the group once, so the expansion and the dot
        // product below carry no per-element bounds check.
        let pb = &packed[g * gbyte..g * gbyte + n.div_ceil(2)];
        let xg = &x[g * group..g * group + n];

        // Expand the group to floats first, then a plain dot product: a table lookup in
        // the middle of the accumulation blocks vectorisation.
        let wf = expand_group(pb, n);

        let mut s = [0.0f64; 8];
        let wn = &wf[..n];
        for (w8, x8) in wn.chunks_exact(8).zip(xg.chunks_exact(8)) {
            for l in 0..8 {
                s[l] = (w8[l] as f64).mul_add(x8[l] as f64, s[l]);
            }
        }
        let b0 = s[0] + s[4];
        let b1 = s[1] + s[5];
        let b2 = s[2] + s[6];
        let b3 = s[3] + s[7];
        let mut sub = (b0 + b1) + (b2 + b3);
        for i in (n - n % 8)..n {
            sub = (wf[i] as f64).mul_add(xg[i] as f64, sub);
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
/// Expand one MXFP4 group's packed nibbles into floats. Shared by every variant of the
/// kernel so the nibble convention lives in one place: the low nibble of each byte is
/// the EVEN element, and reversing it yields a matrix with exactly the right values in
/// the wrong places. k3_ops.c:1275-1280.
#[inline(always)]
fn expand_group(pb: &[u8], n: usize) -> [f32; 64] {
    let mut wf = [0.0f32; 64];
    let half = n >> 1;
    for (j, &byte) in pb.iter().take(half).enumerate() {
        let pv = &E2M1_PAIR[byte as usize];
        wf[2 * j] = pv[0];
        wf[2 * j + 1] = pv[1];
    }
    if n & 1 != 0 {
        wf[n - 1] = E2M1_PAIR[pb[half] as usize][0];
    }
    wf
}

// ---------------------------------------------------------------------- dispatch ----
//
// The pointers are `unsafe fn` because a `#[target_feature]` function can only be
// coerced to an unsafe pointer: calling one on a CPU without the feature is UB, so the
// type system refuses to hide that. `select` is the only place that installs them, and
// it installs a feature-gated variant only after `is_x86_feature_detected!` has said so.

type DotF32 = unsafe fn(&[f32], &[f32]) -> f64;
type DotBf16 = unsafe fn(&[u16], &[f32]) -> f64;
type DotQ8 = unsafe fn(&[u8], &[f32]) -> f32;
type DotMxfp4 = unsafe fn(&[u8], &[u8], &[f32], usize, usize) -> f64;

pub(crate) struct RowKernels {
    pub f32: DotF32,
    pub bf16: DotBf16,
    pub q8: DotQ8,
    pub mxfp4: DotMxfp4,
}

/// Explicit AVX2, mirroring the intrinsics in k3_ops.c one for one.
///
/// The lane mapping IS the exactness contract. A `__m256d` holds four f64, and loading
/// four consecutive elements per accumulator places element `i` in lane `i % 4` of
/// accumulator `(i / 4) % 4`, i.e. scalar accumulator `a[i % 16]`. Reducing
/// `(v0+v1)+(v2+v3)` lanewise therefore produces exactly the scalar `b0..b3`, and the
/// final cross-lane `(a0+a1)+(a2+a3)` is the scalar tree. `_mm256_fmadd_pd` per lane is
/// the same IEEE operation as a scalar `mul_add`. k3_ops.c:1077-1110.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use core::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma")]
    pub fn dot_f32(row: &[f32], x: &[f32]) -> f64 {
        let n = row.len();
        let x = &x[..n];
        let blocks = n / 16;
        let mut acc;
        unsafe {
            let (mut v0, mut v1) = (_mm256_setzero_pd(), _mm256_setzero_pd());
            let (mut v2, mut v3) = (_mm256_setzero_pd(), _mm256_setzero_pd());
            let (rp, xp) = (row.as_ptr(), x.as_ptr());
            for b in 0..blocks {
                let i = b * 16;
                v0 = _mm256_fmadd_pd(
                    _mm256_cvtps_pd(_mm_loadu_ps(rp.add(i))),
                    _mm256_cvtps_pd(_mm_loadu_ps(xp.add(i))),
                    v0,
                );
                v1 = _mm256_fmadd_pd(
                    _mm256_cvtps_pd(_mm_loadu_ps(rp.add(i + 4))),
                    _mm256_cvtps_pd(_mm_loadu_ps(xp.add(i + 4))),
                    v1,
                );
                v2 = _mm256_fmadd_pd(
                    _mm256_cvtps_pd(_mm_loadu_ps(rp.add(i + 8))),
                    _mm256_cvtps_pd(_mm_loadu_ps(xp.add(i + 8))),
                    v2,
                );
                v3 = _mm256_fmadd_pd(
                    _mm256_cvtps_pd(_mm_loadu_ps(rp.add(i + 12))),
                    _mm256_cvtps_pd(_mm_loadu_ps(xp.add(i + 12))),
                    v3,
                );
            }
            let vt = _mm256_add_pd(_mm256_add_pd(v0, v1), _mm256_add_pd(v2, v3));
            let mut a = [0.0f64; 4];
            _mm256_storeu_pd(a.as_mut_ptr(), vt);
            acc = (a[0] + a[1]) + (a[2] + a[3]);
        }
        for i in (blocks * 16)..n {
            acc = (row[i] as f64).mul_add(x[i] as f64, acc);
        }
        acc
    }

    #[target_feature(enable = "avx2,fma")]
    pub fn dot_bf16(row: &[u16], x: &[f32]) -> f64 {
        let n = row.len();
        let x = &x[..n];
        let blocks = n / 16;
        let mut acc;
        unsafe {
            // bf16 -> f32 is a 16-bit left shift, so widen u16 to u32, shift, and
            // reinterpret. No table, no rounding. k3_ops.c:1086.
            #[inline(always)]
            unsafe fn wide(p: *const u16) -> __m256d {
                let h = _mm_loadl_epi64(p as *const __m128i);
                _mm256_cvtps_pd(_mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(h), 16)))
            }
            let (mut v0, mut v1) = (_mm256_setzero_pd(), _mm256_setzero_pd());
            let (mut v2, mut v3) = (_mm256_setzero_pd(), _mm256_setzero_pd());
            let (rp, xp) = (row.as_ptr(), x.as_ptr());
            for b in 0..blocks {
                let i = b * 16;
                v0 = _mm256_fmadd_pd(
                    wide(rp.add(i)),
                    _mm256_cvtps_pd(_mm_loadu_ps(xp.add(i))),
                    v0,
                );
                v1 = _mm256_fmadd_pd(
                    wide(rp.add(i + 4)),
                    _mm256_cvtps_pd(_mm_loadu_ps(xp.add(i + 4))),
                    v1,
                );
                v2 = _mm256_fmadd_pd(
                    wide(rp.add(i + 8)),
                    _mm256_cvtps_pd(_mm_loadu_ps(xp.add(i + 8))),
                    v2,
                );
                v3 = _mm256_fmadd_pd(
                    wide(rp.add(i + 12)),
                    _mm256_cvtps_pd(_mm_loadu_ps(xp.add(i + 12))),
                    v3,
                );
            }
            let vt = _mm256_add_pd(_mm256_add_pd(v0, v1), _mm256_add_pd(v2, v3));
            let mut a = [0.0f64; 4];
            _mm256_storeu_pd(a.as_mut_ptr(), vt);
            acc = (a[0] + a[1]) + (a[2] + a[3]);
        }
        for i in (blocks * 16)..n {
            acc = (bf16f(row[i]) as f64).mul_add(x[i] as f64, acc);
        }
        acc
    }

    #[target_feature(enable = "avx2,fma")]
    pub fn dot_q8(row: &[u8], x: &[f32]) -> f32 {
        super::dot_q8(row, x)
    }

    /// Two f64 accumulators over the 32-element group, element `i` to accumulator
    /// `(i/4) % 2`, lanewise-summed then cross-lane paired. Element `i` lands in scalar
    /// slot `s[i % 8]`, and `v0 + v1` gives `s[j] + s[4+j]`, which is the scalar
    /// `b0..b3`. k3_ops.c:1300-1312.
    #[target_feature(enable = "avx2,fma")]
    pub fn dot_mxfp4(packed: &[u8], scales: &[u8], x: &[f32], input: usize, group: usize) -> f64 {
        let ngrp = input.div_ceil(group);
        let gbyte = group / 2;
        let mut acc = 0.0f64;
        for g in 0..ngrp {
            let sb = scales[g];
            if sb == 255 {
                continue;
            }
            let n = core::cmp::min(input - g * group, group);
            let pb = &packed[g * gbyte..g * gbyte + n.div_ceil(2)];
            let xg = &x[g * group..g * group + n];
            let wf = expand_group(pb, n);

            let mut sub;
            let blocks = n / 8;
            unsafe {
                let (mut v0, mut v1) = (_mm256_setzero_pd(), _mm256_setzero_pd());
                let (wp, xp) = (wf.as_ptr(), xg.as_ptr());
                for b in 0..blocks {
                    let i = b * 8;
                    v0 = _mm256_fmadd_pd(
                        _mm256_cvtps_pd(_mm_loadu_ps(wp.add(i))),
                        _mm256_cvtps_pd(_mm_loadu_ps(xp.add(i))),
                        v0,
                    );
                    v1 = _mm256_fmadd_pd(
                        _mm256_cvtps_pd(_mm_loadu_ps(wp.add(i + 4))),
                        _mm256_cvtps_pd(_mm_loadu_ps(xp.add(i + 4))),
                        v1,
                    );
                }
                let mut a = [0.0f64; 4];
                _mm256_storeu_pd(a.as_mut_ptr(), _mm256_add_pd(v0, v1));
                sub = (a[0] + a[1]) + (a[2] + a[3]);
            }
            for i in (blocks * 8)..n {
                sub = (wf[i] as f64).mul_add(xg[i] as f64, sub);
            }
            acc += sub * E8M0[sb as usize] as f64;
        }
        acc
    }
}

/// Explicit NEON. FMA and NEON are baseline on aarch64, so this needs no detection.
///
/// A `float64x2_t` holds two f64, so the sixteen scalar accumulators become eight
/// vectors with `u[m] = (a[2m], a[2m+1])`. The reduction then falls out exactly:
/// `(u0+u2)+(u4+u6)` lanewise is `(b0, b1)` and `(u1+u3)+(u5+u7)` is `(b2, b3)`,
/// associated in the same order as the scalar tree, and `vaddvq_f64` of each is
/// `b0+b1` and `b2+b3`. Clang reaches the same eight `fmla.2d` from the C source; this
/// spells it out because rustc's vectoriser does not.
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use core::arch::aarch64::*;

    pub fn dot_f32(row: &[f32], x: &[f32]) -> f64 {
        let n = row.len();
        let x = &x[..n];
        let blocks = n / 16;
        let mut acc;
        unsafe {
            let mut u = [vdupq_n_f64(0.0); 8];
            let (rp, xp) = (row.as_ptr(), x.as_ptr());
            for b in 0..blocks {
                let i = b * 16;
                for k in 0..4 {
                    let rf = vld1q_f32(rp.add(i + 4 * k));
                    let xf = vld1q_f32(xp.add(i + 4 * k));
                    u[2 * k] = vfmaq_f64(
                        u[2 * k],
                        vcvt_f64_f32(vget_low_f32(rf)),
                        vcvt_f64_f32(vget_low_f32(xf)),
                    );
                    u[2 * k + 1] =
                        vfmaq_f64(u[2 * k + 1], vcvt_high_f64_f32(rf), vcvt_high_f64_f32(xf));
                }
            }
            let v01 = vaddq_f64(vaddq_f64(u[0], u[2]), vaddq_f64(u[4], u[6]));
            let v23 = vaddq_f64(vaddq_f64(u[1], u[3]), vaddq_f64(u[5], u[7]));
            acc = vaddvq_f64(v01) + vaddvq_f64(v23);
        }
        for i in (blocks * 16)..n {
            acc = (row[i] as f64).mul_add(x[i] as f64, acc);
        }
        acc
    }

    pub fn dot_bf16(row: &[u16], x: &[f32]) -> f64 {
        let n = row.len();
        let x = &x[..n];
        let blocks = n / 16;
        let mut acc;
        unsafe {
            // bf16 -> f32 is a 16-bit left shift: widen u16 to u32, shift, reinterpret.
            #[inline(always)]
            unsafe fn wide4(p: *const u16) -> float32x4_t {
                vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_u16(vld1_u16(p))))
            }
            let mut u = [vdupq_n_f64(0.0); 8];
            let (rp, xp) = (row.as_ptr(), x.as_ptr());
            for b in 0..blocks {
                let i = b * 16;
                for k in 0..4 {
                    let rf = wide4(rp.add(i + 4 * k));
                    let xf = vld1q_f32(xp.add(i + 4 * k));
                    u[2 * k] = vfmaq_f64(
                        u[2 * k],
                        vcvt_f64_f32(vget_low_f32(rf)),
                        vcvt_f64_f32(vget_low_f32(xf)),
                    );
                    u[2 * k + 1] =
                        vfmaq_f64(u[2 * k + 1], vcvt_high_f64_f32(rf), vcvt_high_f64_f32(xf));
                }
            }
            let v01 = vaddq_f64(vaddq_f64(u[0], u[2]), vaddq_f64(u[4], u[6]));
            let v23 = vaddq_f64(vaddq_f64(u[1], u[3]), vaddq_f64(u[5], u[7]));
            acc = vaddvq_f64(v01) + vaddvq_f64(v23);
        }
        for i in (blocks * 16)..n {
            acc = (bf16f(row[i]) as f64).mul_add(x[i] as f64, acc);
        }
        acc
    }

    /// Eight scalar accumulators become four vectors, `t[m] = (s[2m], s[2m+1])`. Then
    /// `t0+t2` is `(b0, b1)` and `t1+t3` is `(b2, b3)`, matching the scalar tree.
    pub fn dot_mxfp4(packed: &[u8], scales: &[u8], x: &[f32], input: usize, group: usize) -> f64 {
        let ngrp = input.div_ceil(group);
        let gbyte = group / 2;
        let mut acc = 0.0f64;
        for g in 0..ngrp {
            let sb = scales[g];
            if sb == 255 {
                continue;
            }
            let n = core::cmp::min(input - g * group, group);
            let pb = &packed[g * gbyte..g * gbyte + n.div_ceil(2)];
            let xg = &x[g * group..g * group + n];
            let wf = expand_group(pb, n);

            let mut sub;
            let blocks = n / 8;
            unsafe {
                let mut t = [vdupq_n_f64(0.0); 4];
                let (wp, xp) = (wf.as_ptr(), xg.as_ptr());
                for b in 0..blocks {
                    let i = b * 8;
                    for k in 0..2 {
                        let wv = vld1q_f32(wp.add(i + 4 * k));
                        let xv = vld1q_f32(xp.add(i + 4 * k));
                        t[2 * k] = vfmaq_f64(
                            t[2 * k],
                            vcvt_f64_f32(vget_low_f32(wv)),
                            vcvt_f64_f32(vget_low_f32(xv)),
                        );
                        t[2 * k + 1] =
                            vfmaq_f64(t[2 * k + 1], vcvt_high_f64_f32(wv), vcvt_high_f64_f32(xv));
                    }
                }
                sub = vaddvq_f64(vaddq_f64(t[0], t[2])) + vaddvq_f64(vaddq_f64(t[1], t[3]));
            }
            for i in (blocks * 8)..n {
                sub = (wf[i] as f64).mul_add(xg[i] as f64, sub);
            }
            acc += sub * E8M0[sb as usize] as f64;
        }
        acc
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
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return RowKernels {
                f32: avx2::dot_f32,
                bf16: avx2::dot_bf16,
                q8: avx2::dot_q8,
                mxfp4: avx2::dot_mxfp4,
            };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return RowKernels {
            f32: neon::dot_f32,
            bf16: neon::dot_bf16,
            q8: dot_q8,
            mxfp4: neon::dot_mxfp4,
        };
    }
    #[allow(unreachable_code)]
    plain()
}

/// The portable bodies, with no feature gating. `tests/ops.rs` reproduces this same
/// partition and tree independently and asserts bit equality against whichever variant
/// `select` installed, which is the Rust replacement for the C build's
/// scalar-versus-AVX2 gate.
pub(crate) fn plain() -> RowKernels {
    RowKernels {
        f32: dot_f32,
        bf16: dot_bf16,
        q8: dot_q8,
        mxfp4: dot_mxfp4,
    }
}
