// SPDX-License-Identifier: Apache-2.0
//! The numeric core. Statement-for-statement port of `src/core/k3_ops.c`.
//!
//! Every routine here is gated on a JSON fixture under `fixtures/ops/`. The per-op
//! fixtures exist alongside the full-model oracle because the oracle is pass-or-fail: it
//! proves the stack is wrong without indicating which of ~40 kernels is responsible.
//!
//! FLOATING-POINT CONTRACT. See `dispatch.rs`. In short: `fma(a,b,c)` becomes
//! `a.mul_add(b, c)`, plain `a*b + c` stays two operations, accumulators are f64
//! throughout, and the hand-written lane partitions and reduction trees are copied
//! verbatim because they fix a summation ORDER, not because they are faster to read.
//!
//! WHERE C ABORTS, THIS PANICS. `k3_fatal_oom` and `k3_fatal_bound` exist because these
//! kernels return void: a failed allocation or an exceeded bound could only be handled by
//! returning early, which leaves the output buffer holding the previous layer's values,
//! and the caller folds that straight into the residual. The run then finishes and prints
//! a plausible token computed from stale memory. That is the one failure mode this engine
//! treats as unacceptable, so it fails loudly instead. Rust's allocator aborts on OOM by
//! itself; the bound check in `mla_cached` is an explicit panic.
//!
//! WHERE THIS PORT DEPARTS FROM THE C SIGNATURES, and why each is not a design change:
//!   - Output width comes from `y.len()` instead of a separate `out` argument. Callers
//!     slice exactly what the C code passed as `out`, so the borrow checker enforces the
//!     disjointness the C scratch-layout comments only asked for.
//!   - `wdt` tags become `WMat`, one enum per matrix rather than one int per struct. A
//!     superset of the C behaviour, and it removes the "memset the struct first" hazard.
//!   - Exactly-one-of `kda`/`mla` becomes the `Attn` enum, for the same reason.
//!   - The expert source is threaded as a separate `&mut dyn ExpertSrc` parameter rather
//!     than living inside `MoeW`, because it is mutated (a cache admits and evicts) while
//!     the weight struct is read-only.

pub mod dispatch;

use crate::cfg::{Cfg, MXFP4_GROUP};
pub use dispatch::bf16f;
use rayon::prelude::*;
use std::sync::atomic::{AtomicI64, Ordering};

// ------------------------------------------------------------------ weight views ----

/// A weight matrix and its storage format. The always-active weights ship as bf16 and
/// total 113.49 GB; held as fp32 they are ~227 GB. These kernels are bandwidth bound, so
/// storing bf16 and widening inside the matmul both halves the memory and runs faster.
/// Small vectors (norms, biases, A_log, dt_bias, the conv kernels) stay fp32: together
/// well under 0.1% of the bytes, and several are read ELEMENTWISE, where a silent type
/// change would be read as garbage. k3.h:253.
///
/// `I8R` is a per-row int8 matrix used ONLY by the hybrid draft model, whose output is
/// never emitted directly and which therefore carries no exactness contract. Each row is
/// `[f32 scale][i8 * in]`. Never used on the exact model. k3.h:265.
#[derive(Clone, Copy, Debug)]
pub enum WMat<'a> {
    F32(&'a [f32]),
    Bf16(&'a [u16]),
    I8R(&'a [u8]),
}

impl WMat<'_> {
    /// Byte stride of one row. bf16 and fp32 are per-element; int8 rows carry a scale.
    /// k3.h:300.
    pub fn row_bytes(&self, input: usize) -> usize {
        match self {
            WMat::F32(_) => input * 4,
            WMat::Bf16(_) => input * 2,
            WMat::I8R(_) => 4 + input,
        }
    }
}

/// Gated MLA weights. `g` is `None` when the output gate is disabled. k3.h:202.
pub struct MlaW<'a> {
    pub q_a: WMat<'a>,
    pub q_b: WMat<'a>,
    pub kv_a: WMat<'a>,
    pub kv_b: WMat<'a>,
    pub o: WMat<'a>,
    pub g: Option<WMat<'a>>,
    /// Elementwise in `rmsnorm`: stays fp32.
    pub q_a_norm: &'a [f32],
    pub kv_a_norm: &'a [f32],
}

/// Kimi Delta Attention weights. k3.h:447.
pub struct KdaW<'a> {
    pub q: WMat<'a>,
    pub k: WMat<'a>,
    pub v: WMat<'a>,
    /// `[H*D][conv_k]` depthwise. Read elementwise, so fp32.
    pub q_conv: &'a [f32],
    pub k_conv: &'a [f32],
    pub v_conv: &'a [f32],
    pub f_a: WMat<'a>,
    pub f_b: WMat<'a>,
    /// `[H]`, PER HEAD.
    pub a_log: &'a [f32],
    /// `[H*D]` per (head, channel).
    pub dt_bias: &'a [f32],
    pub b: WMat<'a>,
    pub g: WMat<'a>,
    /// `[D]` head-wise norm gain.
    pub o_norm: &'a [f32],
    pub o: WMat<'a>,
}

/// One routed expert as it sits in the cache: still MXFP4, never widened. A dequantised
/// expert is 132 MB against 17.55 MB packed, and a token touches 1,472 of them, so
/// widening on load would need 194 GB per token. k3.h:323.
#[derive(Clone, Copy, Debug)]
pub struct ExpertQ<'a> {
    /// w1 gate, packed nibbles and E8M0 scales.
    pub p1: &'a [u8],
    pub s1: &'a [u8],
    /// w3 up.
    pub p3: &'a [u8],
    pub s3: &'a [u8],
    /// w2 down.
    pub p2: &'a [u8],
    pub s2: &'a [u8],
}

/// A source of experts. `get` must leave the returned slices valid until the caller
/// finishes with them; a cache satisfies that by pinning what the current token needs
/// before evicting anything, and the borrow of `self` enforces it here. k3.h:332.
pub trait ExpertSrc {
    fn get(&mut self, layer: usize, expert: usize) -> Option<ExpertQ<'_>>;

    /// OPTIONAL batch prefetch: bring all the listed experts resident, issuing their
    /// reads concurrently. Returns the number brought resident.
    ///
    /// WHY IT EXISTS. `moe` walks the top-16 calling `get` one at a time, so each miss is
    /// a blocking 17.55 MB read and the drive sees one request in flight: on a 92-layer
    /// decode, 1,472 serial round trips. Handing the whole top-k over at once lets the
    /// reads overlap, which is the difference between a queue depth of 1 and one of 16 on
    /// hardware that needs depth to reach its rated bandwidth.
    ///
    /// A short return is NOT fatal: `get` will simply miss on the remainder and read it
    /// the slow way. The default is the always-correct no-op.
    fn get_many(&mut self, _layer: usize, _experts: &[i32]) -> usize {
        0
    }

    /// OPTIONAL: is the expert already resident, i.e. would `get` read no disk? The draft
    /// model's cache-only routing uses this to propose tokens with zero expert I/O.
    fn resident(&mut self, _layer: usize, _expert: usize) -> Option<ExpertQ<'_>> {
        None
    }

    /// `resident` without borrowing out a result, for the cache-only filter pass.
    fn is_resident(&mut self, layer: usize, expert: usize) -> bool {
        self.resident(layer, expert).is_some()
    }
}

/// Stable LatentMoE weights. k3.h:357.
pub struct MoeW<'a> {
    /// Router: `[n_experts][hidden]`. Stays fp32; `router` carries its own inline matmul.
    pub gate: &'a [f32],
    pub bias: Option<&'a [f32]>,
    /// Resident expert bank, fixtures only. The real model streams experts and multiplies
    /// them straight out of MXFP4.
    pub w1: Option<&'a [f32]>,
    pub w3: Option<&'a [f32]>,
    pub w2: Option<&'a [f32]>,
    /// `[latent]`, elementwise: stays fp32.
    pub latent_norm: Option<&'a [f32]>,
    pub down: WMat<'a>,
    pub up: WMat<'a>,
    /// Shared expert, full width.
    pub sh1: WMat<'a>,
    pub sh3: WMat<'a>,
    pub sh2: WMat<'a>,
    /// Identifies this layer to the expert source.
    pub layer: usize,
    /// Draft-only: route among ONLY the experts already resident in the cache and
    /// renormalise over them, so a draft token reads zero new expert bytes. Never set on
    /// the exact model, whose output is authoritative.
    pub cache_only: bool,
}

/// Exactly one attention flavour per layer, made structural. k3.h:501.
pub enum Attn<'a> {
    Kda(KdaW<'a>),
    Mla(MlaW<'a>),
}

/// One decoder layer's weights. k3.h:497.
pub struct LayerW<'a> {
    pub in_norm: &'a [f32],
    pub post_norm: &'a [f32],
    /// Folded at load time in the C engine; folded per call here, as C does.
    pub attn_res_norm: &'a [f32],
    pub attn_res_proj: &'a [f32],
    pub mlp_res_norm: &'a [f32],
    pub mlp_res_proj: &'a [f32],
    pub attn: Attn<'a>,
    /// `None` on the dense layer, which uses `dense_*` instead.
    pub moe: Option<MoeW<'a>>,
    pub dense_gate: Option<WMat<'a>>,
    pub dense_up: Option<WMat<'a>>,
    pub dense_down: Option<WMat<'a>>,
}

// ------------------------------------------------------------------ expert drops ----

/// Number of routed experts that failed to load and were dropped from a MoE sum.
/// Non-zero means some token was computed with part of its routed contribution missing,
/// which is silent numerical corruption: the run still finishes and still prints a
/// plausible token. Callers MUST check this and fail. k3.h:388.
static EXPERT_DROPS: AtomicI64 = AtomicI64::new(0);

pub fn expert_drops() -> i64 {
    EXPERT_DROPS.load(Ordering::Relaxed)
}
pub fn reset_expert_drops() {
    EXPERT_DROPS.store(0, Ordering::Relaxed);
}

// ---------------------------------------------------------------------- rmsnorm ----

/// `y = w * x / sqrt(mean(x^2) + eps)`. The f64 accumulator is load bearing: 7168 squared
/// terms in f32 loses real precision, and every downstream comparison against the
/// reference depends on it. eps is INSIDE the rsqrt. k3_ops.c:91.
pub fn rmsnorm(y: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let n = y.len();
    let mut ss = 0.0f64;
    for i in 0..n {
        ss += x[i] as f64 * x[i] as f64;
    }
    let inv = (1.0 / (ss / n as f64 + eps as f64).sqrt()) as f32;
    for i in 0..n {
        y[i] = w[i] * x[i] * inv;
    }
}

/// `rmsnorm` where the C source passes the same pointer for input and output. Identical
/// arithmetic: the scale is computed from the whole vector before any element is written.
pub fn rmsnorm_ip(v: &mut [f32], w: &[f32], eps: f32) {
    let n = v.len();
    let mut ss = 0.0f64;
    for i in 0..n {
        ss += v[i] as f64 * v[i] as f64;
    }
    let inv = (1.0 / (ss / n as f64 + eps as f64).sqrt()) as f32;
    for i in 0..n {
        v[i] = w[i] * v[i] * inv;
    }
}

#[inline(always)]
fn sigmoidf(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// -------------------------------------------------------------------- SiTU-GLU ----

/// SiTU-GLU over a `2*n` input laid out as `[gate | up]`.
/// ```text
/// a = b1 * tanh(gate / b1) * sigmoid(gate)   the sigmoid sees the UNCAPPED gate
/// u = b2 * tanh(up / b2)
/// y = a * u                                  |y| <= b1*b2
/// ```
/// Feeding the sigmoid the capped value instead still yields a bounded, plausible
/// function and is WRONG. modeling_kimi_linear.py:79, k3_ops.c:107.
pub fn situ_glu(y: &mut [f32], x: &[f32], b1: f32, b2: f32) {
    let n = y.len();
    let (gate, up) = (&x[..n], &x[n..2 * n]);
    for i in 0..n {
        let g = gate[i];
        let a = b1 * (g / b1).tanh() * sigmoidf(g);
        let u = b2 * (up[i] / b2).tanh();
        y[i] = a * u;
    }
}

// -------------------------------------------------------------------- ShortConv ----

/// Causal depthwise convolution with a fused SiLU, exactly as
/// `ShortConvolution(activation='silu')`.
///
/// `w` is `[channels][k]`; taps are ordered oldest..newest, so `w[k-1]` multiplies the
/// CURRENT input. `state[c*(k-1) + j]` holds the previous inputs for channel `c`, oldest
/// first, and is UPDATED IN PLACE so a decode loop can carry it. `None` treats history as
/// zero. This state is a second piece of per-sequence memory beyond the recurrent matrix,
/// and it is the piece a decode loop forgets. k3_ops.c:126.
pub fn shortconv_ip(
    v: &mut [f32],
    w: &[f32],
    state: Option<&mut [f32]>,
    channels: usize,
    k: usize,
    t_len: usize,
) {
    let hist = k - 1;
    // Guard on `hist`, not on the buffer: a legitimate k == 1 configuration must still
    // run the convolution rather than silently leave the output untouched.
    let mut buf = vec![0.0f32; hist];
    let mut state = state;

    for c in 0..channels {
        if hist > 0 {
            match state.as_deref() {
                Some(s) => buf.copy_from_slice(&s[c * hist..(c + 1) * hist]),
                None => buf.fill(0.0),
            }
        }

        for t in 0..t_len {
            let cur = v[t * channels + c];
            let mut acc = w[c * k + hist] * cur;
            for j in 0..hist {
                acc += w[c * k + j] * buf[j];
            }

            for j in 0..hist.saturating_sub(1) {
                buf[j] = buf[j + 1];
            }
            if hist > 0 {
                buf[hist - 1] = cur;
            }

            v[t * channels + c] = acc * sigmoidf(acc); // SiLU, fused
        }
        if hist > 0 {
            if let Some(s) = state.as_deref_mut() {
                s[c * hist..(c + 1) * hist].copy_from_slice(&buf);
            }
        }
    }
}

/// Out-of-place `shortconv_ip`. The C source reads `x[t]` before writing `y[t]` and never
/// revisits an earlier `t`, so copying first and convolving in place reads exactly the
/// same values.
pub fn shortconv(
    y: &mut [f32],
    x: &[f32],
    w: &[f32],
    state: Option<&mut [f32]>,
    channels: usize,
    k: usize,
    t_len: usize,
) {
    y[..t_len * channels].copy_from_slice(&x[..t_len * channels]);
    shortconv_ip(y, w, state, channels, k, t_len);
}

// ------------------------------------------------------------------- KDA decay ----

/// Decay chain, per head `h` and channel `d`:
/// ```text
/// z     = f_b(f_a(x))[h][d] + dt_bias[h*D + d]
/// g     = lb * sigmoid(exp(A_log[h]) * z)   in (lb, 0]
/// alpha = exp(g)                            in (e^lb, 1]
/// ```
/// A_log is indexed PER HEAD. The checkpoint stores `head_dim` floats but only the first
/// `H` are meaningful; indexing this per channel is a silent, fatal error. In f32 the
/// sigmoid underflows to 0 once its argument falls below about -87.3, so `alpha == 1.0`
/// exactly is legitimate saturation meaning perfect retention, not an error.
/// fla/ops/kda/gate.py:60-69, k3_ops.c:161.
pub fn kda_decay(
    g: &mut [f32],
    alpha: &mut [f32],
    z: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    h_count: usize,
    d: usize,
    lb: f32,
) {
    for h in 0..h_count {
        let a = a_log[h].exp();
        for j in 0..d {
            let i = h * d + j;
            let u = a * (z[i] + dt_bias[i]);
            let gi = lb * sigmoidf(u);
            g[i] = gi;
            alpha[i] = gi.exp();
        }
    }
}

/// `kda_decay` where the C source passes `z` as both input and output for `g`.
pub fn kda_decay_ip(
    g: &mut [f32],
    alpha: &mut [f32],
    a_log: &[f32],
    dt_bias: &[f32],
    h_count: usize,
    d: usize,
    lb: f32,
) {
    for h in 0..h_count {
        let a = a_log[h].exp();
        for j in 0..d {
            let i = h * d + j;
            let u = a * (g[i] + dt_bias[i]);
            let gi = lb * sigmoidf(u);
            g[i] = gi;
            alpha[i] = gi.exp();
        }
    }
}

// -------------------------------------------------------------- KDA recurrence ----

/// One KDA recurrence step for one head. `s` is `[dk][dv]`, row-major.
///
/// ORDER IS LOAD BEARING (fla/ops/kda/naive.py:59-63):
/// 1. decay  `S[i][:] *= alpha[i]`
/// 2. read   `u = S^T k`
/// 3. write  `S += k (beta*(v-u))^T`
/// 4. output `o = S^T q` from the ALREADY UPDATED state
///
/// `q` must arrive pre-scaled by `d_k^-0.5`. k3_ops.c:179.
pub fn kda_step(
    s: &mut [f32],
    o: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    alpha: &[f32],
    beta: f32,
    dk: usize,
    dv: usize,
) {
    // 1. channel-wise decay: scale ROW i of S by alpha[i]. The gate is per key channel,
    //    not a scalar, which is what "channel-wise forget gate" means.
    for i in 0..dk {
        let a = alpha[i];
        for j in 0..dv {
            s[i * dv + j] *= a;
        }
    }

    // 2. read the state along k: u = S^T k
    let mut u = vec![0.0f32; dv];
    for i in 0..dk {
        let ki = k[i];
        if ki == 0.0 {
            continue;
        }
        let row = &s[i * dv..i * dv + dv];
        for j in 0..dv {
            u[j] += ki * row[j];
        }
    }

    // 3. rank-one delta write. (v - u) is the prediction error: this is what makes it a
    //    DELTA rule rather than plain accumulation.
    for i in 0..dk {
        let ki = k[i];
        if ki == 0.0 {
            continue;
        }
        let row = &mut s[i * dv..i * dv + dv];
        for j in 0..dv {
            row[j] += ki * beta * (v[j] - u[j]);
        }
    }

    // 4. output from the ALREADY UPDATED state: o = S^T q
    for j in 0..dv {
        o[j] = 0.0;
    }
    for i in 0..dk {
        let qi = q[i];
        if qi == 0.0 {
            continue;
        }
        let row = &s[i * dv..i * dv + dv];
        for j in 0..dv {
            o[j] += qi * row[j];
        }
    }
}

// ----------------------------------------------------------------------- matmul ----

/// `y[out] = W[out][in] . x[in]`, row-major, no bias anywhere in this model.
///
/// The dominant cost in the engine. Two properties are load bearing:
///
/// 1. OUTPUT ROWS ARE INDEPENDENT. Parallelising the outer loop introduces no reduction
///    and no race, and changes no arithmetic at all: each row is summed by exactly one
///    worker in exactly the order `dot_f32` fixes. Results are therefore identical at any
///    thread count, which the fixtures rely on.
/// 2. SIXTEEN f64 ACCUMULATORS partitioned by `i % 16`, reduced by the explicit tree in
///    `dispatch::dot_f32`. Written out rather than left to the compiler because it fixes
///    a summation ORDER that every other matmul kernel here reproduces.
///
/// k3_ops.c:243. The `out > 64` serial threshold is the C `if` clause on the OpenMP
/// pragma, kept so small matmuls do not pay for a fork.
pub fn matmul(y: &mut [f32], x: &[f32], w: &[f32], input: usize) {
    let dot = dispatch::kernels().f32;
    // SAFETY: `dispatch::select` installs a feature-gated kernel only after runtime
    // detection confirmed the CPU has it; on every other target the pointer holds the
    // portable body. The same argument covers the three kernels below.
    let row = |o: usize| &w[o * input..o * input + input];
    if y.len() > 64 {
        y.par_iter_mut().enumerate().for_each(|(o, yo)| {
            *yo = unsafe { dot(row(o), x) } as f32;
        });
    } else {
        for (o, yo) in y.iter_mut().enumerate() {
            *yo = unsafe { dot(&w[o * input..o * input + input], x) } as f32;
        }
    }
}

/// `matmul` with `W` stored as bf16 and widened on read.
///
/// WHY IT LOSES NOTHING: bf16 is what the checkpoint already contains, and widening bf16
/// to f32 is a pure left shift by 16 bits with no rounding, so multiplying from bf16
/// storage computes with exactly the values an fp32 copy would have supplied. The only
/// difference from `matmul` is WHERE the widening happens. k3_ops.c:1066.
pub fn matmul_bf16(y: &mut [f32], x: &[f32], w: &[u16], input: usize) {
    let dot = dispatch::kernels().bf16;
    if y.len() > 64 {
        y.par_iter_mut().enumerate().for_each(|(o, yo)| {
            *yo = unsafe { dot(&w[o * input..o * input + input], x) } as f32;
        });
    } else {
        for (o, yo) in y.iter_mut().enumerate() {
            *yo = unsafe { dot(&w[o * input..o * input + input], x) } as f32;
        }
    }
}

/// Per-row int8 matmul for the draft model. `w` is `y.len()` rows of
/// `[f32 scale][i8 * in]`. No determinism contract: draft output is never emitted
/// directly, which is what lets this accumulate in f32. k3_ops.c:1136.
pub fn matmul_q8(y: &mut [f32], x: &[f32], w: &[u8], input: usize) {
    let dot = dispatch::kernels().q8;
    let rowb = 4 + input;
    let body = |o: usize, yo: &mut f32| {
        let row = &w[o * rowb..o * rowb + rowb];
        let scale = f32::from_ne_bytes([row[0], row[1], row[2], row[3]]);
        *yo = unsafe { dot(&row[4..], x) } * scale;
    };
    if y.len() > 64 {
        y.par_iter_mut().enumerate().for_each(|(o, yo)| body(o, yo));
    } else {
        for (o, yo) in y.iter_mut().enumerate() {
            body(o, yo);
        }
    }
}

/// The one call every trunk matmul goes through. Dispatch is a predictable branch on a
/// per-layer tag, outside the inner loops, so it costs nothing measurable. k3.h:289.
pub fn mmw(y: &mut [f32], x: &[f32], w: WMat, input: usize) {
    match w {
        WMat::Bf16(m) => matmul_bf16(y, x, m, input),
        WMat::I8R(m) => matmul_q8(y, x, m, input),
        WMat::F32(m) => matmul(y, x, m, input),
    }
}

/// `y[rows] = W[rows][in] . x[in]`, with `W` read straight out of packed MXFP4 and never
/// materialised as floats. This is not an optimisation; it is what makes streaming
/// experts possible at all: one expert is 33,030,144 parameters, so the 1,472 a single
/// token touches would be 194 GB of fp32. As packed nibbles the same expert is 17.55 MB,
/// and a matrix-vector product is memory bound, so reading 7.5x fewer bytes makes this
/// kernel FASTER than dequantising first.
///
/// ACCURACY CONTRACT. Deliberately NOT bit-identical to dequantise-then-`matmul`: it sums
/// each group of 32 and applies that group's scale before accumulating, while
/// dequantise-then-matmul sums every term of the row under one set of accumulators. Every
/// individual product is EXACT in f64 (an E2M1 value carries 3 mantissa bits and `x`
/// carries 24, so 27 of the available 53), so only the additions round and reassociating
/// exact terms moves the result by roughly 1 ULP of f64. The required agreement is 1e-6,
/// gated by `tests/expert.rs`; the margin is nine orders of magnitude. k3_ops.c:1243.
pub fn matmul_mxfp4(
    y: &mut [f32],
    x: &[f32],
    packed: &[u8],
    scales: &[u8],
    input: usize,
    group: usize,
) {
    let dot = dispatch::kernels().mxfp4;
    let pcols = input / 2;
    let ngrp = input.div_ceil(group);
    let body = |r: usize, yr: &mut f32| {
        *yr = unsafe {
            dot(
                &packed[r * pcols..(r + 1) * pcols],
                &scales[r * ngrp..(r + 1) * ngrp],
                x,
                input,
                group,
            )
        } as f32;
    };
    if y.len() > 64 {
        y.par_iter_mut().enumerate().for_each(|(r, yr)| body(r, yr));
    } else {
        for (r, yr) in y.iter_mut().enumerate() {
            body(r, yr);
        }
    }
}

/// Dequantise OCP MX FP4.
/// ```text
/// packed [rows][pcols]       u8, TWO 4-bit elements per byte
/// scales [rows][pcols*2/32]  u8, one E8M0 exponent per 32 elements
/// out    [rows][pcols*2]     f32
/// value = E2M1[nibble] * 2^(scale - 127)
/// ```
/// NIBBLE ORDER IS A CONVENTION, NOT A RULE: the low nibble of each byte is the EVEN
/// element. Reversing it yields a matrix with exactly the right values in the wrong
/// places, so every statistic looks perfect and the model is wrong. E8M0 255 is NaN by
/// spec and maps to zero so one bad byte cannot poison a row. k3_ops.c:1331.
pub fn mxfp4_dequant(
    out: &mut [f32],
    packed: &[u8],
    scales: &[u8],
    rows: usize,
    pcols: usize,
    group: usize,
) {
    let width = pcols * 2;
    let ngrp = width.div_ceil(group);

    for r in 0..rows {
        let pr = &packed[r * pcols..(r + 1) * pcols];
        let sr = &scales[r * ngrp..(r + 1) * ngrp];
        let orow = &mut out[r * width..(r + 1) * width];

        for g in 0..ngrp {
            let mult = dispatch::E8M0[sr[g] as usize];
            let lo = g * group;
            let hi = core::cmp::min(lo + group, width);
            for i in lo..hi {
                let byte = pr[i >> 1];
                let nib = if i & 1 != 0 { byte >> 4 } else { byte & 0x0F };
                orow[i] = dispatch::E2M1[nib as usize] * mult;
            }
        }
    }
}

// ----------------------------------------------------------------------- router ----

/// MoE routing, one token.
/// ```text
/// logits = W x               f32, no bias, W is [n_experts, hidden]
/// scores = sigmoid(logits)   independent; they do NOT sum to 1
/// sel    = topk(scores + bias)   the frozen bias steers SELECTION ONLY
/// w      = scores[sel]           gathered from the UNBIASED scores
/// w     /= sum(w) + 1e-20        when renorm
/// w     *= routed_scale
/// ```
/// Reading the weights from the biased scores instead is the classic silent error: it
/// still routes to the same experts and only perturbs the mixture. The router reads the
/// FULL hidden width, before the latent down-projection. `idx` and `w` are written in
/// DESCENDING score order.
///
/// PARALLEL over experts, and bit-identical because of it: each iteration writes only its
/// own score, and the accumulation order INSIDE an expert is untouched. k3_ops.c:397.
pub fn router(
    idx: &mut [i32],
    w_out: &mut [f32],
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    hidden: usize,
    n_experts: usize,
    topk: usize,
    renorm: bool,
    routed_scale: f32,
) {
    let mut score = vec![0.0f32; n_experts];
    let mut choice = vec![0.0f32; n_experts];

    score
        .par_iter_mut()
        .zip(choice.par_iter_mut())
        .enumerate()
        .for_each(|(e, (sc, ch))| {
            let row = &w[e * hidden..e * hidden + hidden];
            let mut acc = 0.0f64;
            for i in 0..hidden {
                acc += row[i] as f64 * x[i] as f64;
            }
            *sc = 1.0 / (1.0 + (-(acc as f32)).exp());
            *ch = *sc + bias.map_or(0.0, |b| b[e]);
        });

    // top-k by repeated max: 896 experts and topk 16 is 14k comparisons, cheap next to an
    // 18 MB expert read, and it avoids a sort. Marking taken entries -inf keeps ties
    // deterministic in first-index order, matching a stable selection.
    for j in 0..topk {
        let mut best: i32 = -1;
        let mut bv = f32::NEG_INFINITY;
        for e in 0..n_experts {
            if choice[e] > bv {
                bv = choice[e];
                best = e as i32;
            }
        }
        if best < 0 {
            idx[j] = 0;
            w_out[j] = 0.0;
            continue;
        }
        idx[j] = best;
        w_out[j] = score[best as usize]; // UNBIASED score, not choice[best]
        choice[best as usize] = f32::NEG_INFINITY;
    }

    if renorm && topk > 1 {
        let mut s = 0.0f64;
        for j in 0..topk {
            s += w_out[j] as f64;
        }
        let inv = (1.0 / (s + 1e-20)) as f32;
        for j in 0..topk {
            w_out[j] *= inv;
        }
    }
    for j in 0..topk {
        w_out[j] *= routed_scale;
    }
}

// ---------------------------------------------------------------------- AttnRes ----

/// AttnRes aggregation over `nsrc` sources of width `n`.
/// ```text
/// keys  = RMSNorm(sources)      normalised
/// score = dot(key, fold)        fold = norm.weight * proj.weight, ONE vector
/// out   = softmax(score) @ sources    the RAW, UNNORMALISED sources
/// ```
/// modeling_kimi_linear.py:1075-1088, k3_ops.c:459.
pub fn attn_res(out: &mut [f32], src: &[f32], fold: &[f32], nsrc: usize, n: usize, eps: f32) {
    let mut score = vec![0.0f32; nsrc];

    for s in 0..nsrc {
        let v = &src[s * n..s * n + n];
        let mut ss = 0.0f64;
        for i in 0..n {
            ss += v[i] as f64 * v[i] as f64;
        }
        let inv = (1.0 / (ss / n as f64 + eps as f64).sqrt()) as f32;
        // key is the NORMALISED source; fold already carries norm.weight * proj.weight
        let mut acc = 0.0f64;
        for i in 0..n {
            acc += (v[i] * inv) as f64 * fold[i] as f64;
        }
        score[s] = acc as f32;
    }

    let mut m = score[0];
    for s in 1..nsrc {
        if score[s] > m {
            m = score[s];
        }
    }
    let mut z = 0.0f64;
    for s in 0..nsrc {
        score[s] = (score[s] - m).exp();
        z += score[s] as f64;
    }

    for i in 0..n {
        out[i] = 0.0;
    }
    for s in 0..nsrc {
        let p = (score[s] as f64 / z) as f32;
        let v = &src[s * n..s * n + n]; // the RAW source, not the key
        for i in 0..n {
            out[i] += p * v[i];
        }
    }
}

// -------------------------------------------------------------------- Gated MLA ----

/// Scratch for MLA. `cap` is the highest position that will be attended over plus one;
/// without a cache that is just `T`. `cached_mode` is whether a KV cache is supplied, in
/// which case the keys and values do not live in scratch. k3_ops.c:497.
pub fn mla_scratch_cached(c: &Cfg, t_len: usize, cap: usize, cached_mode: bool) -> usize {
    let h = c.n_heads as usize;
    let qh = (c.qk_nope + c.qk_rope) as usize;
    let vh = c.v_head as usize;
    let kvd = (c.qk_nope + c.v_head) as usize;
    let mut n = t_len * h * qh
        + (c.kv_lora + c.qk_rope) as usize
        + c.q_lora as usize
        + 2 * h * vh
        + core::cmp::max(cap, t_len);
    if !cached_mode {
        n += t_len * h * kvd + t_len * c.qk_rope as usize;
    }
    n
}

pub fn mla_scratch(c: &Cfg, t_len: usize) -> usize {
    mla_scratch_cached(c, t_len, t_len, false)
}

/// Gated MLA, NoPE, with an optional KV cache.
///
/// NoPE: no rotation is ever applied, yet the `qk_rope` slots STILL EXIST and are still
/// concatenated onto both query and key. Dropping them changes the head width from 192 to
/// 128 and silently produces a different model. The softmax scale is
/// `(qk_nope + qk_rope)^-0.5`, i.e. `192^-0.5`, NOT `qk_nope^-0.5`.
///
/// The output gate multiplies the attention output BEFORE `o_proj`, with no norm, unlike
/// KDA which norms first then gates.
///
/// WHAT IS CACHED, AND WHY NOT THE COMPRESSED LATENT. Caching the 576-float latent and
/// re-expanding through `kv_b` each step is 42x smaller and far slower: `kv_b` is
/// 24576x512, so re-expanding every cached position costs an O(T) sweep of 12.6M-MAC
/// matmuls per layer per token. The EXPANDED per-head keys and values are cached instead,
/// 2.37 MB per position across the 24 MLA layers. The rope slot is cached separately
/// because it is SHARED across heads.
///
/// `kvc == None` selects the self-contained path, which recomputes all keys and values
/// from `x` and caches nothing; both paths must produce identical output. k3_ops.c:299.
#[allow(clippy::too_many_arguments)]
pub fn mla_cached(
    out: &mut [f32],
    x: &[f32],
    w: &MlaW,
    c: &Cfg,
    t_len: usize,
    scratch: &mut [f32],
    kvc: Option<&mut [f32]>,
    ropec: Option<&mut [f32]>,
    cached: usize,
    cap: usize,
) {
    let e = c.hidden as usize;
    let h = c.n_heads as usize;
    let qn = c.qk_nope as usize;
    let qr = c.qk_rope as usize;
    let vh = c.v_head as usize;
    let qh = qn + qr; // 192: the FULL head width
    let kvw = c.kv_lora as usize + qr; // 576: latent + shared rope slot
    let kvd = qn + vh; // 256: cached width per head
    let scale = 1.0f32 / (qh as f32).sqrt();
    let has_cache = kvc.is_some();
    let cached = if has_cache { cached } else { 0 };
    let last = cached + t_len - 1; // highest absolute position
    if has_cache && last >= cap {
        panic!(
            "k3: FATAL, MLA KV cache position is {}, which exceeds the limit of {}.\n\
             \x20   Aborting rather than returning without writing the output buffer,\n\
             \x20   which would fold the previous layer's values into the residual and\n\
             \x20   produce plausible-looking but meaningless output.\n\
             \x20   Shorten the prompt, lower --gen, or drop --incremental.",
            last,
            cap as i64 - 1
        );
    }

    // Scratch layout. Every region is DISJOINT, which `split_at_mut` now enforces rather
    // than merely documenting. Size the buffer with `mla_scratch_cached`.
    let (q, rest) = scratch.split_at_mut(t_len * h * qh);
    let (ct, rest) = rest.split_at_mut(kvw);
    let (ql, rest) = rest.split_at_mut(c.q_lora as usize);
    let (acc, rest) = rest.split_at_mut(h * vh);
    let (gbuf, rest) = rest.split_at_mut(h * vh);
    let (sc, rest) = rest.split_at_mut(last + 1);
    let (kvs, rps) = rest.split_at_mut(if has_cache { 0 } else { t_len * h * kvd });

    // Without a cache the keys/values live in scratch and `cached` is 0, so absolute
    // position `p` indexes both stores identically. This is the C K3_KV_AT/K3_ROPE_AT
    // pair, resolved once.
    let kv: &mut [f32] = match kvc {
        Some(k) => k,
        None => kvs,
    };
    let rope: &mut [f32] = match ropec {
        Some(r) => r,
        None => rps,
    };

    // ---- per-token projections ----
    for t in 0..t_len {
        let p = cached + t;
        let xt = &x[t * e..t * e + e];
        mmw(ql, xt, w.q_a, e);
        rmsnorm_ip(ql, w.q_a_norm, c.rms_eps);
        mmw(
            &mut q[t * h * qh..(t + 1) * h * qh],
            ql,
            w.q_b,
            c.q_lora as usize,
        );

        // ONE projection emits the compressed latent AND the shared rope slot
        mmw(ct, xt, w.kv_a, e);
        // the norm covers the latent only, never the rope slot
        rmsnorm_ip(&mut ct[..c.kv_lora as usize], w.kv_a_norm, c.rms_eps);
        rope[p * qr..p * qr + qr].copy_from_slice(&ct[c.kv_lora as usize..c.kv_lora as usize + qr]);
        mmw(
            &mut kv[p * h * kvd..(p + 1) * h * kvd],
            &ct[..c.kv_lora as usize],
            w.kv_b,
            c.kv_lora as usize,
        );
    }

    // The stores are final; read them immutably for the attention sweep.
    let kv = &*kv;
    let rope = &*rope;
    let q = &*q;

    // ---- attention, per head, causal ----
    for t in 0..t_len {
        let p = cached + t;
        for head in 0..h {
            let qt = &q[(t * h + head) * qh..(t * h + head) * qh + qh];
            let mut m = f32::NEG_INFINITY;
            for s in 0..=p {
                let ks = &kv[s * h * kvd + head * kvd..s * h * kvd + head * kvd + kvd];
                let kr = &rope[s * qr..s * qr + qr]; // shared slot
                let mut d = 0.0f64;
                for i in 0..qn {
                    d += qt[i] as f64 * ks[i] as f64;
                }
                // the rope slot is UNROTATED but still scored, and the SAME 64 values
                // serve every head. Dropping this term is the silent bug.
                for i in 0..qr {
                    d += qt[qn + i] as f64 * kr[i] as f64;
                }
                sc[s] = d as f32 * scale;
                if sc[s] > m {
                    m = sc[s];
                }
            }
            let mut z = 0.0f64;
            for s in 0..=p {
                sc[s] = (sc[s] - m).exp();
                z += sc[s] as f64;
            }

            let o = &mut acc[head * vh..head * vh + vh];
            for j in 0..vh {
                o[j] = 0.0;
            }
            for s in 0..=p {
                let pr = (sc[s] as f64 / z) as f32;
                let vs = &kv[s * h * kvd + head * kvd + qn..s * h * kvd + head * kvd + kvd];
                for j in 0..vh {
                    o[j] += pr * vs[j];
                }
            }
        }

        // ---- output gate then projection. Gate BEFORE o_proj, and no norm on it,
        // unlike KDA which norms first. ----
        if let Some(g) = w.g {
            mmw(gbuf, &x[t * e..t * e + e], g, e);
            for i in 0..h * vh {
                acc[i] *= 1.0 / (1.0 + (-gbuf[i]).exp());
            }
        }
        mmw(&mut out[t * e..t * e + e], acc, w.o, h * vh);
    }
}

pub fn mla(out: &mut [f32], x: &[f32], w: &MlaW, c: &Cfg, t_len: usize, scratch: &mut [f32]) {
    mla_cached(out, x, w, c, t_len, scratch, None, None, 0, 0);
}

// ------------------------------------------------------------ Stable LatentMoE ----

pub fn moe_scratch(c: &Cfg) -> usize {
    let si = (c.moe_inter * c.n_shared) as usize;
    2 * c.latent as usize        // z, accL
        + 3 * c.moe_inter as usize  // gu (2*I) + act (I)
        + c.latent as usize         // edn
        + 3 * si                    // sgu (2*SI) + sact
        + c.hidden as usize // sdn
}

/// Stable LatentMoE. The ORDER is load bearing:
/// 1. route on the FULL hidden width, before any projection
/// 2. down-project to the latent width
/// 3. run the selected experts IN LATENT SPACE and sum, weighted
/// 4. RMSNorm the AGGREGATE, never per expert
/// 5. up-project back to hidden
/// 6. add the shared expert computed on the ORIGINAL input, with NO routing weight and
///    NO scaling
///
/// Routed experts live at the latent width (`latent -> moe_inter -> latent`); the shared
/// expert is one wider MLP at full width with intermediate `moe_inter * n_shared`.
/// k3_ops.c:534.
#[allow(clippy::too_many_arguments)]
pub fn moe(
    out: &mut [f32],
    x: &[f32],
    w: &MoeW,
    c: &Cfg,
    t_len: usize,
    idx: &mut [i32],
    wt: &mut [f32],
    scratch: &mut [f32],
    src: Option<&mut dyn ExpertSrc>,
) {
    let e = c.hidden as usize;
    let l = c.latent as usize;
    let i_dim = c.moe_inter as usize;
    let si = i_dim * c.n_shared as usize;
    let topk = c.topk as usize;

    let (z, rest) = scratch.split_at_mut(l);
    let (accl, rest) = rest.split_at_mut(l);
    let (gu, rest) = rest.split_at_mut(2 * i_dim);
    let (act, rest) = rest.split_at_mut(i_dim);
    let (edn, rest) = rest.split_at_mut(l);
    let (sgu, rest) = rest.split_at_mut(2 * si);
    let (sact, rest) = rest.split_at_mut(si);
    let (sdn, _) = rest.split_at_mut(e);

    let mut src = src;

    for t in 0..t_len {
        let xt = &x[t * e..t * e + e];

        // 1. route on the FULL width, before the down-projection
        router(
            idx,
            wt,
            xt,
            w.gate,
            w.bias,
            e,
            c.n_experts as usize,
            topk,
            c.moe_renorm != 0,
            c.routed_scale,
        );

        let mut nk = topk;
        // Draft cache-only routing: keep only the top-k experts already resident and
        // renormalise their weights so the mixture still sums as intended. This makes a
        // draft token read ZERO new expert bytes. It is an approximation, which is exactly
        // what a draft is; the exact model verifies every proposed token.
        if w.cache_only {
            if let Some(s) = src.as_mut() {
                let mut m = 0usize;
                let mut wsum = 0.0f32;
                for j in 0..topk {
                    if s.is_resident(w.layer, idx[j] as usize) {
                        idx[m] = idx[j];
                        wt[m] = wt[j];
                        wsum += wt[j];
                        m += 1;
                    }
                }
                nk = m;
                if wsum > 0.0 {
                    for j in 0..nk {
                        wt[j] /= wsum;
                    }
                }
            }
        }

        // 2. down-project into the latent space
        mmw(z, xt, w.down, e);

        // 3. the selected experts, in latent space, weighted and summed
        accl.fill(0.0);
        // Hand the WHOLE top-k to the source first, so its reads can overlap. Without
        // this the loop below misses, blocks on a 17.55 MB read, computes, misses again:
        // a queue depth of one against a drive that needs depth to reach its rated
        // bandwidth.
        if !w.cache_only {
            if let Some(s) = src.as_mut() {
                s.get_many(w.layer, &idx[..nk]);
            }
        }
        for j in 0..nk {
            match src.as_mut() {
                Some(s) => {
                    // Streamed: the expert stays MXFP4 and the matmul reads nibbles. In
                    // cache-only mode every idx[j] is known resident, so `resident` serves
                    // it with no disk read; otherwise `get` may read it.
                    let expert = idx[j] as usize;
                    let q = if w.cache_only {
                        s.resident(w.layer, expert)
                    } else {
                        s.get(w.layer, expert)
                    };
                    let Some(q) = q else {
                        // A cache-only draft filtered to resident experts already, so a
                        // miss here is a benign race at worst; skip it, since the draft is
                        // approximate by construction and the exact model verifies. On the
                        // exact path a miss is the unacceptable silent-corruption case:
                        // count it so the caller fails the run.
                        if w.cache_only {
                            continue;
                        }
                        EXPERT_DROPS.fetch_add(1, Ordering::Relaxed);
                        eprintln!(
                            "EXPERT DROP: layer {} expert {} failed to load; this token is CORRUPT",
                            w.layer, expert
                        );
                        continue;
                    };
                    let (gu_gate, gu_up) = gu.split_at_mut(i_dim);
                    matmul_mxfp4(gu_gate, z, q.p1, q.s1, l, MXFP4_GROUP);
                    matmul_mxfp4(gu_up, z, q.p3, q.s3, l, MXFP4_GROUP);
                    situ_glu(act, gu, c.situ_b1, c.situ_b2);
                    matmul_mxfp4(edn, act, q.p2, q.s2, i_dim, MXFP4_GROUP);
                }
                None => {
                    let ei = idx[j] as usize;
                    let e1 = &w.w1.expect("resident expert bank w1")[ei * i_dim * l..];
                    let e3 = &w.w3.expect("resident expert bank w3")[ei * i_dim * l..];
                    let e2 = &w.w2.expect("resident expert bank w2")[ei * l * i_dim..];
                    let (gu_gate, gu_up) = gu.split_at_mut(i_dim);
                    matmul(gu_gate, z, e1, l);
                    matmul(gu_up, z, e3, l);
                    situ_glu(act, gu, c.situ_b1, c.situ_b2);
                    matmul(edn, act, e2, i_dim);
                }
            }
            let wj = wt[j];
            for i in 0..l {
                accl[i] += wj * edn[i];
            }
        }

        // 4. RMSNorm the AGGREGATE (not per expert), then 5. up-project
        if c.latent_norm != 0 {
            rmsnorm_ip(accl, w.latent_norm.expect("latent_norm weight"), c.rms_eps);
        }
        let ot = &mut out[t * e..t * e + e];
        mmw(ot, accl, w.up, l);

        // 6. shared expert on the ORIGINAL full-width input, added UNWEIGHTED
        let (sgu_gate, sgu_up) = sgu.split_at_mut(si);
        mmw(sgu_gate, xt, w.sh1, e);
        mmw(sgu_up, xt, w.sh3, e);
        situ_glu(sact, sgu, c.situ_b1, c.situ_b2);
        mmw(sdn, sact, w.sh2, si);
        for i in 0..e {
            ot[i] += sdn[i];
        }
    }
}

/// Batched MoE for PREFILL over a chunk of tokens, streamed experts only.
///
/// `moe` walks the top-k for each token independently, so across a T-token chunk it
/// fetches an expert once per token that routes to it. Under near-uniform routing that is
/// mostly waste: reading each unique expert ONCE and reusing it for every token in the
/// chunk cuts prefill expert bytes ~3-4x.
///
/// Exactness is preserved to the last bit. Per token the arithmetic is identical to `moe`:
/// the routed latent contributions are accumulated in the ORIGINAL top-k order from a
/// per-(token, slot) buffer, then normalised, up-projected and given the shared expert
/// exactly as before. Only the ORDER in which experts are fetched from disk changes, and
/// that touches no floating-point result. k3_ops.c:667.
#[allow(clippy::too_many_arguments)]
pub fn moe_prefill(
    out: &mut [f32],
    x: &[f32],
    w: &MoeW,
    c: &Cfg,
    t_len: usize,
    idx: &mut [i32],
    wt: &mut [f32],
    scratch: &mut [f32],
    src: Option<&mut dyn ExpertSrc>,
) {
    // K3_NO_BATCH_PREFILL forces the per-token path, so one binary can produce both the
    // batched and the reference token streams for a bit-identity A/B.
    let no_batch = no_batch_prefill();
    // cache_only renormalises per token over the resident subset, which the per-token path
    // already does; the draft's prompt prefill is one-time, so defer rather than duplicate
    // the renorm in the batch.
    if src.is_none() || t_len <= 1 || no_batch || w.cache_only {
        moe(out, x, w, c, t_len, idx, wt, scratch, src);
        return;
    }
    let mut src = src;
    let e = c.hidden as usize;
    // Fixed sub-chunks bound the contribution buffer (14.7 MB at 64 tokens) no matter how
    // long the prompt is; a 32k prefill would otherwise want 7.3 GB of it. Most of the
    // dedup is already captured at this width: the unique-expert count grows far slower
    // than the request count under near-uniform routing.
    const CHUNK: usize = 64;
    let mut t0 = 0usize;
    while t0 < t_len {
        let n = core::cmp::min(t_len - t0, CHUNK);
        // Fresh reborrow per chunk: `Option<&mut dyn Trait>` is not `Copy`, and the C code
        // hands the same source to every chunk.
        let s: Option<&mut dyn ExpertSrc> = match &mut src {
            Some(r) => Some(&mut **r),
            None => None,
        };
        if n == 1 {
            moe(
                &mut out[t0 * e..(t0 + 1) * e],
                &x[t0 * e..(t0 + 1) * e],
                w,
                c,
                1,
                idx,
                wt,
                scratch,
                s,
            );
        } else {
            moe_prefill_chunk(
                &mut out[t0 * e..(t0 + n) * e],
                &x[t0 * e..(t0 + n) * e],
                w,
                c,
                n,
                scratch,
                s,
            );
        }
        t0 += CHUNK;
    }
}

fn no_batch_prefill() -> bool {
    static NO_BATCH: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("K3_NO_BATCH_PREFILL").is_some());
    *NO_BATCH
}

fn moe_prefill_chunk(
    out: &mut [f32],
    x: &[f32],
    w: &MoeW,
    c: &Cfg,
    t_len: usize,
    scratch: &mut [f32],
    src: Option<&mut dyn ExpertSrc>,
) {
    let e = c.hidden as usize;
    let ll = c.latent as usize;
    let i_dim = c.moe_inter as usize;
    let si = i_dim * c.n_shared as usize;
    let k = c.topk as usize;
    let src = src.expect("moe_prefill_chunk requires a streamed source");

    // Per-token routing decisions and latent inputs, plus a contribution buffer holding
    // every routed expert's latent output for every token: [T][K][Ll].
    let mut ridx = vec![0i32; t_len * k];
    let mut rwt = vec![0.0f32; t_len * k];
    let mut zz = vec![0.0f32; t_len * ll];
    let mut contrib = vec![0.0f32; t_len * k * ll];

    // 1. route every token and down-project it, and collect the batch's unique experts.
    let mut uniq: Vec<i32> = Vec::with_capacity(t_len * k);
    let mut seen = vec![false; c.n_experts as usize];
    for t in 0..t_len {
        let xt = &x[t * e..t * e + e];
        router(
            &mut ridx[t * k..(t + 1) * k],
            &mut rwt[t * k..(t + 1) * k],
            xt,
            w.gate,
            w.bias,
            e,
            c.n_experts as usize,
            k,
            c.moe_renorm != 0,
            c.routed_scale,
        );
        mmw(&mut zz[t * ll..(t + 1) * ll], xt, w.down, e);
        for j in 0..k {
            let ex = ridx[t * k + j];
            if ex >= 0 && (ex as usize) < c.n_experts as usize && !seen[ex as usize] {
                seen[ex as usize] = true;
                uniq.push(ex);
            }
        }
    }

    // 2. expert-major: fetch each unique expert ONCE, apply it to every (token, slot) that
    // selected it. gu/act/edn are reused per (expert, token).
    {
        let (gu, rest) = scratch.split_at_mut(2 * i_dim);
        let (act, rest) = rest.split_at_mut(i_dim);
        let (edn, _) = rest.split_at_mut(ll);
        src.get_many(w.layer, &uniq);
        for &ex in &uniq {
            let Some(q) = src.get(w.layer, ex as usize) else {
                EXPERT_DROPS.fetch_add(1, Ordering::Relaxed);
                eprintln!(
                    "EXPERT DROP: layer {} expert {} failed to load; this chunk is CORRUPT",
                    w.layer, ex
                );
                continue;
            };
            for t in 0..t_len {
                for j in 0..k {
                    if ridx[t * k + j] != ex {
                        continue;
                    }
                    let zt = &zz[t * ll..(t + 1) * ll];
                    let (gu_gate, gu_up) = gu.split_at_mut(i_dim);
                    matmul_mxfp4(gu_gate, zt, q.p1, q.s1, ll, MXFP4_GROUP);
                    matmul_mxfp4(gu_up, zt, q.p3, q.s3, ll, MXFP4_GROUP);
                    situ_glu(act, gu, c.situ_b1, c.situ_b2);
                    matmul_mxfp4(edn, act, q.p2, q.s2, i_dim, MXFP4_GROUP);
                    contrib[(t * k + j) * ll..(t * k + j + 1) * ll].copy_from_slice(edn);
                }
            }
        }
    }

    // 3. per token, sum contributions in the ORIGINAL top-k order, then the tail of the
    // MoE exactly as `moe` does it, so every float matches the per-token path.
    //
    // The shared-expert buffers reuse the same scratch base the gu/act/edn trio just
    // finished with, exactly as k3_ops.c:775 does; carving them in a second pass is the
    // same aliasing with no pointer arithmetic. `moe_scratch` covers both layouts.
    let (sgu, rest) = scratch.split_at_mut(2 * si);
    let (sact, rest) = rest.split_at_mut(si);
    let (sdn, _) = rest.split_at_mut(e);

    for t in 0..t_len {
        let xt = &x[t * e..t * e + e];
        // Reuse this token's now-dead down-projection slot as the aggregate.
        let acc = &mut zz[t * ll..(t + 1) * ll];
        acc.fill(0.0);
        for j in 0..k {
            let wj = rwt[t * k + j];
            let cb = &contrib[(t * k + j) * ll..(t * k + j + 1) * ll];
            for i in 0..ll {
                acc[i] += wj * cb[i];
            }
        }
        if c.latent_norm != 0 {
            rmsnorm_ip(acc, w.latent_norm.expect("latent_norm weight"), c.rms_eps);
        }
        let ot = &mut out[t * e..t * e + e];
        mmw(ot, acc, w.up, ll);

        let (sgu_gate, sgu_up) = sgu.split_at_mut(si);
        mmw(sgu_gate, xt, w.sh1, e);
        mmw(sgu_up, xt, w.sh3, e);
        situ_glu(sact, sgu, c.situ_b1, c.situ_b2);
        mmw(sdn, sact, w.sh2, si);
        for i in 0..e {
            ot[i] += sdn[i];
        }
    }
}

// ------------------------------------------------------------- KDA full layer ----

/// L2 normalisation over the last dimension. The reference uses the SUM of squares with
/// eps inside the rsqrt, NOT the mean: k3_ref.py `l2norm()`. Using the mean here scales
/// every q and k by `sqrt(d_k)` and quietly changes the attention temperature.
/// k3_ops.c:792.
pub fn l2norm(v: &mut [f32], eps: f32) {
    let n = v.len();
    let mut ss = 0.0f64;
    for i in 0..n {
        ss += v[i] as f64 * v[i] as f64;
    }
    let inv = (1.0 / (ss + eps as f64).sqrt()) as f32;
    for i in 0..n {
        v[i] *= inv;
    }
}

pub fn kda_scratch(c: &Cfg, t_len: usize) -> usize {
    let p = (c.kda_heads * c.kda_head_dim) as usize;
    3 * t_len * p                    // q, k, v after conv
        + 2 * t_len * p              // z then alpha
        + t_len * c.kda_heads as usize   // beta
        + t_len * p                  // recurrence output
        + 2 * p                      // gate buffer and one work row
        + c.kda_head_dim as usize // f_a output
}

/// Kimi Delta Attention, one full layer, one sequence.
///
/// Order, and every step of it is load bearing:
/// 1. q,k,v = Linear(x), separate projections
/// 2. ShortConv on each, SiLU FUSED inside
/// 3. L2Norm on q and k ONLY, never on v. Sum of squares, not mean.
/// 4. beta = sigmoid(b_proj(x)), a PER-HEAD SCALAR, not per channel
/// 5. z = f_b(f_a(x)) + dt_bias, ONE shared low-rank pair for all heads;
///    g = lb*sigmoid(exp(A_log[h]) * z) with A_log indexed BY HEAD; alpha = exp(g)
/// 6. recurrence, q pre-scaled by d_k^-0.5: decay, delta write, read UPDATED
/// 7. head-wise RMSNorm on the output, over d_v, per head
/// 8. multiply by sigmoid(g_proj(x)): norm FIRST, then gate
/// 9. o_proj
///
/// Step 8 is the opposite order to MLA, which gates before its projection with no norm at
/// all. Sharing one code path between them is wrong.
///
/// `state`, when supplied, holds `H*D*D` recurrent floats followed by
/// `3*H*D*(conv_k-1)` convolution floats, and is UPDATED IN PLACE so a decode loop can
/// carry it. k3_ops.c:811.
pub fn kda_layer(
    out: &mut [f32],
    x: &[f32],
    w: &KdaW,
    c: &Cfg,
    t_len: usize,
    state: Option<&mut [f32]>,
    scratch: &mut [f32],
) {
    let e = c.hidden as usize;
    let h_count = c.kda_heads as usize;
    let d = c.kda_head_dim as usize;
    let p = h_count * d;
    let k_taps = c.conv_k as usize;
    let hist = k_taps - 1;

    let (q, rest) = scratch.split_at_mut(t_len * p);
    let (k, rest) = rest.split_at_mut(t_len * p);
    let (v, rest) = rest.split_at_mut(t_len * p);
    let (z, rest) = rest.split_at_mut(t_len * p);
    let (al, rest) = rest.split_at_mut(t_len * p);
    let (bt, rest) = rest.split_at_mut(t_len * h_count);
    let (o, rest) = rest.split_at_mut(t_len * p);
    let (gb, rest) = rest.split_at_mut(p);
    let (wr, rest) = rest.split_at_mut(p);
    let (fa, _) = rest.split_at_mut(d);

    // 1. projections
    for t in 0..t_len {
        let xt = &x[t * e..t * e + e];
        mmw(&mut q[t * p..(t + 1) * p], xt, w.q, e);
        mmw(&mut k[t * p..(t + 1) * p], xt, w.k, e);
        mmw(&mut v[t * p..(t + 1) * p], xt, w.v, e);
        mmw(&mut bt[t * h_count..(t + 1) * h_count], xt, w.b, e);
        // ONE shared low-rank pair feeds every head: [E->D] then [D->H*D]
        mmw(fa, xt, w.f_a, e);
        mmw(&mut z[t * p..(t + 1) * p], fa, w.f_b, d);
    }

    // 2. ShortConv with fused SiLU, carrying state across calls
    let (s_recur, conv_state) = match state {
        Some(s) => {
            let (a, b) = s.split_at_mut(h_count * d * d);
            (Some(a), Some(b))
        }
        None => (None, None),
    };
    match conv_state {
        Some(cs) => {
            let (cq, rest) = cs.split_at_mut(p * hist);
            let (ck, rest) = rest.split_at_mut(p * hist);
            let (cv, _) = rest.split_at_mut(p * hist);
            shortconv_ip(q, w.q_conv, Some(cq), p, k_taps, t_len);
            shortconv_ip(k, w.k_conv, Some(ck), p, k_taps, t_len);
            shortconv_ip(v, w.v_conv, Some(cv), p, k_taps, t_len);
        }
        None => {
            shortconv_ip(q, w.q_conv, None, p, k_taps, t_len);
            shortconv_ip(k, w.k_conv, None, p, k_taps, t_len);
            shortconv_ip(v, w.v_conv, None, p, k_taps, t_len);
        }
    }

    // 3. L2Norm on q and k ONLY, per head. v is deliberately left alone.
    for t in 0..t_len {
        for head in 0..h_count {
            l2norm(&mut q[t * p + head * d..t * p + head * d + d], 1e-6);
            l2norm(&mut k[t * p + head * d..t * p + head * d + d], 1e-6);
        }
    }

    // 4/5. beta and the decay chain
    for t in 0..t_len {
        for head in 0..h_count {
            bt[t * h_count + head] = sigmoidf(bt[t * h_count + head]);
        }
        kda_decay_ip(
            &mut z[t * p..(t + 1) * p],
            &mut al[t * p..(t + 1) * p],
            w.a_log,
            w.dt_bias,
            h_count,
            d,
            c.gate_lb,
        );
    }

    // 6. recurrence, per head, with q pre-scaled by d_k^-0.5.
    //
    // Heads are independent: each reads and writes only its own S block, its own D-wide
    // slice of q/k/v/al/o, and its own beta column. The recurrence is sequential in t
    // WITHIN a head, so each head walks its own t in order; per-head arithmetic is
    // untouched and the results are bit-identical to the serial form. The recurrence is
    // 0.4% of FLOPs but, serial, it is a majority of non-matmul wall time at high core
    // counts.
    let mut owned_state: Vec<f32>;
    let s_all: &mut [f32] = match s_recur {
        Some(s) => s,
        None => {
            owned_state = vec![0.0f32; h_count * d * d];
            &mut owned_state
        }
    };
    let qscale = 1.0f32 / (d as f32).sqrt();
    let (qr, kr, vr, alr) = (&*q, &*k, &*v, &*al);
    let btr = &*bt;
    {
        // `o` is head-strided within each t, so it cannot be chunked by head directly.
        // Regroup its D-wide chunks: chunk index i is t*H + h, so i % H is the head.
        let mut per_head: Vec<Vec<&mut [f32]>> =
            (0..h_count).map(|_| Vec::with_capacity(t_len)).collect();
        for (i, ch) in o.chunks_mut(d).enumerate() {
            per_head[i % h_count].push(ch);
        }
        let tasks: Vec<_> = s_all
            .chunks_mut(d * d)
            .zip(wr.chunks_mut(d))
            .zip(per_head)
            .collect();
        tasks
            .into_par_iter()
            .enumerate()
            .for_each(|(head, ((sh, wh), mut o_rows))| {
                for t in 0..t_len {
                    let off = t * p + head * d;
                    for i in 0..d {
                        wh[i] = qr[off + i] * qscale;
                    }
                    kda_step(
                        sh,
                        o_rows[t],
                        wh,
                        &kr[off..off + d],
                        &vr[off..off + d],
                        &alr[off..off + d],
                        btr[t * h_count + head],
                        d,
                        d,
                    );
                }
            });
    }

    // 7/8/9. head-wise RMSNorm, THEN the gate, THEN the output projection
    for t in 0..t_len {
        let xt = &x[t * e..t * e + e];
        for head in 0..h_count {
            let lo = t * p + head * d;
            rmsnorm_ip(&mut o[lo..lo + d], w.o_norm, c.rms_eps);
        }
        mmw(gb, xt, w.g, e);
        for i in 0..p {
            o[t * p + i] *= sigmoidf(gb[i]);
        }
        mmw(&mut out[t * e..t * e + e], &o[t * p..(t + 1) * p], w.o, p);
    }
}

// ------------------------------------------------------------- decoder layer ----

pub fn layer_scratch(c: &Cfg, t_len: usize) -> usize {
    let a = mla_scratch(c, t_len);
    let b = kda_scratch(c, t_len);
    let m = moe_scratch(c);
    let sub = a.max(b).max(m);
    // prefix_sum, tmp, fold vectors, one attn_res source stack, plus the sub-block
    3 * t_len * c.hidden as usize
        + 2 * c.hidden as usize
        + (c.n_layers / c.attn_res_block + 2) as usize * c.hidden as usize
        + 2 * c.dense_inter as usize
        + sub
}

/// One decoder layer, reproducing `_forward_attn_residual` statement for statement.
/// ```text
/// prefix_sum = h
/// if block_residual is NON-EMPTY:
///     h = attn_res([blocks..., prefix_sum], self_attention_res)   <- REPLACES h
/// if layer_idx % attn_res_block == 0:
///     push prefix_sum onto block_residual
///     prefix_sum = NONE                                           <- the reset
/// h = input_layernorm(h)
/// h = attention(h)
/// prefix_sum = (prefix_sum == NONE) ? h : prefix_sum + h
/// h = attn_res([blocks..., prefix_sum], mlp_res)                  <- UNCONDITIONAL
/// h = post_attention_layernorm(h)
/// h = moe(h) or dense_mlp(h)
/// prefix_sum = (prefix_sum == NONE) ? h : prefix_sum + h
/// return prefix_sum
/// ```
/// THE SUBTLETY: on a boundary layer the running residual is pushed into the snapshot
/// stack and then CLEARED, so it does NOT also survive as a separate softmax source
/// there. On every other layer it does. Getting this wrong is silent.
///
/// The second aggregation has NO emptiness guard, unlike the first. That is safe only
/// because layer 0 is itself a boundary and has already pushed one snapshot.
///
/// The incremental form: everything except MLA already carries its own state, so only
/// softmax attention has to see every earlier position. `kvc == None` gives exactly the
/// full-recompute behaviour. k3_ops.c:925.
#[allow(clippy::too_many_arguments)]
pub fn decoder_layer_inc(
    h: &mut [f32],
    block_residual: &mut [f32],
    n_blocks: &mut usize,
    w: &LayerW,
    c: &Cfg,
    layer_idx: i32,
    t_len: usize,
    state: Option<&mut [f32]>,
    scratch: &mut [f32],
    kvc: Option<&mut [f32]>,
    ropec: Option<&mut [f32]>,
    cached: usize,
    cap: usize,
    src: Option<&mut dyn ExpertSrc>,
) {
    let e = c.hidden as usize;
    let maxb = (c.n_layers / c.attn_res_block + 2) as usize;

    let (pref, rest) = scratch.split_at_mut(t_len * e);
    let (tmp, rest) = rest.split_at_mut(t_len * e);
    let (hin, rest) = rest.split_at_mut(t_len * e);
    let (fold_a, rest) = rest.split_at_mut(e);
    let (fold_m, rest) = rest.split_at_mut(e);
    let (src_stack, rest) = rest.split_at_mut(maxb * e);
    let (dgu, sub) = rest.split_at_mut(2 * c.dense_inter as usize);

    // The norm gain and the scoring projection collapse to ONE vector. Folding them here
    // costs 2*hidden multiplies per layer; a real engine folds at load time.
    for i in 0..e {
        fold_a[i] = w.attn_res_norm[i] * w.attn_res_proj[i];
        fold_m[i] = w.mlp_res_norm[i] * w.mlp_res_proj[i];
    }

    pref.copy_from_slice(&h[..t_len * e]);
    let mut have_prefix = true; // mirrors "prefix_sum is not None"

    // aggregation before attention, only when snapshots already exist
    if *n_blocks > 0 {
        for t in 0..t_len {
            for b in 0..*n_blocks {
                src_stack[b * e..(b + 1) * e]
                    .copy_from_slice(&block_residual[(t * maxb + b) * e..(t * maxb + b + 1) * e]);
            }
            src_stack[*n_blocks * e..(*n_blocks + 1) * e]
                .copy_from_slice(&pref[t * e..(t + 1) * e]);
            attn_res(
                &mut h[t * e..(t + 1) * e],
                src_stack,
                fold_a,
                *n_blocks + 1,
                e,
                c.rms_eps,
            );
        }
    }

    // block boundary: snapshot the running residual, then CLEAR it
    if layer_idx % c.attn_res_block == 0 {
        for t in 0..t_len {
            let dst = (t * maxb + *n_blocks) * e;
            block_residual[dst..dst + e].copy_from_slice(&pref[t * e..(t + 1) * e]);
        }
        *n_blocks += 1;
        have_prefix = false;
    }

    // attention
    for t in 0..t_len {
        rmsnorm(
            &mut hin[t * e..(t + 1) * e],
            &h[t * e..(t + 1) * e],
            w.in_norm,
            c.rms_eps,
        );
    }
    match &w.attn {
        Attn::Kda(kda) => kda_layer(tmp, hin, kda, c, t_len, state, sub),
        Attn::Mla(mla_w) => mla_cached(tmp, hin, mla_w, c, t_len, sub, kvc, ropec, cached, cap),
    }

    if have_prefix {
        for i in 0..t_len * e {
            pref[i] += tmp[i];
        }
    } else {
        // C sets have_prefix = 1 here; nothing reads it again, so the assignment is
        // dropped rather than carried as a dead store.
        pref.copy_from_slice(&tmp[..t_len * e]);
    }

    // aggregation before the MLP. NO emptiness guard in the reference.
    for t in 0..t_len {
        for b in 0..*n_blocks {
            src_stack[b * e..(b + 1) * e]
                .copy_from_slice(&block_residual[(t * maxb + b) * e..(t * maxb + b + 1) * e]);
        }
        src_stack[*n_blocks * e..(*n_blocks + 1) * e].copy_from_slice(&pref[t * e..(t + 1) * e]);
        attn_res(
            &mut h[t * e..(t + 1) * e],
            src_stack,
            fold_m,
            *n_blocks + 1,
            e,
            c.rms_eps,
        );
    }

    for t in 0..t_len {
        rmsnorm(
            &mut hin[t * e..(t + 1) * e],
            &h[t * e..(t + 1) * e],
            w.post_norm,
            c.rms_eps,
        );
    }

    match &w.moe {
        Some(moe_w) => {
            let mut idx = [0i32; crate::cfg::MAX_TOPK];
            let mut wt = [0.0f32; crate::cfg::MAX_TOPK];
            let topk = c.topk as usize;
            // Prefill batches (T > 1, streamed source) fetch each unique expert once for
            // the whole chunk; decode (T == 1) and the resident path fall straight through
            // to `moe` inside, byte-identical.
            moe_prefill(
                tmp,
                hin,
                moe_w,
                c,
                t_len,
                &mut idx[..topk],
                &mut wt[..topk],
                sub,
                src,
            );
        }
        None => {
            let di = c.dense_inter as usize;
            for t in 0..t_len {
                let (dg, du) = dgu.split_at_mut(di);
                mmw(
                    dg,
                    &hin[t * e..(t + 1) * e],
                    w.dense_gate.expect("dense_gate on the dense layer"),
                    e,
                );
                mmw(
                    du,
                    &hin[t * e..(t + 1) * e],
                    w.dense_up.expect("dense_up on the dense layer"),
                    e,
                );
                situ_glu(&mut sub[..di], dgu, c.situ_b1, c.situ_b2);
                mmw(
                    &mut tmp[t * e..(t + 1) * e],
                    &sub[..di],
                    w.dense_down.expect("dense_down on the dense layer"),
                    di,
                );
            }
        }
    }

    for i in 0..t_len * e {
        pref[i] += tmp[i];
    }
    h[..t_len * e].copy_from_slice(&pref[..t_len * e]);
}

#[allow(clippy::too_many_arguments)]
pub fn decoder_layer(
    h: &mut [f32],
    block_residual: &mut [f32],
    n_blocks: &mut usize,
    w: &LayerW,
    c: &Cfg,
    layer_idx: i32,
    t_len: usize,
    state: Option<&mut [f32]>,
    scratch: &mut [f32],
    src: Option<&mut dyn ExpertSrc>,
) {
    decoder_layer_inc(
        h,
        block_residual,
        n_blocks,
        w,
        c,
        layer_idx,
        t_len,
        state,
        scratch,
        None,
        None,
        0,
        0,
        src,
    );
}
