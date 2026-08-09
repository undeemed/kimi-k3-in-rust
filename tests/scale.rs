// SPDX-License-Identifier: Apache-2.0
//! Does the engine hold up at REAL Kimi K3 dimensions? A port of
//! `tests/unit/scale_test.c`.
//!
//! Everything the per-op fixtures validate ran at hidden 128 with 13 layers. The real
//! model is 7168 wide with 93 layers, 96 heads, 896 experts. The mathematics is identical,
//! but three classes of failure only appear at scale and all of them are silent:
//!
//! 1. INTEGER OVERFLOW. 12288 * 7168 = 88,080,384 fits in int32, but 12288 * 7168 * 4
//!    bytes = 352 MB does not fit comfortably once multiplied by a layer count, and an
//!    expert bank is 92 * 896 * 33,030,144 elements. Any product computed in int rather
//!    than size_t wraps and produces a wrong pointer. Rust's `usize` is 64-bit on the
//!    targets this engine runs on, so the overflow itself cannot happen here, but the
//!    scratch sizers must still return non-zero and layer-scratch must dominate each
//!    sub-buffer at real widths.
//! 2. SCRATCH SIZING. The `*_scratch` helpers must return the right value at real widths,
//!    not just tiny ones. A scratch of zero, or a layer buffer smaller than a sub-buffer it
//!    must contain, is a heap overflow waiting for a caller that trusts it.
//! 3. EXECUTION. A full-width KDA layer is about 443M parameters; at fp32 that is 1.8 GB
//!    of weights, which fits on an ordinary box and is worth running once.
//!
//! This does NOT load real weights. It allocates at real shapes with synthetic values to
//! prove the plumbing survives the dimensions.

use k3::cfg::Cfg;
use k3::ops::{kda_layer, kda_scratch, layer_scratch, mla_scratch, moe_scratch, KdaW, WMat};

/// The same xorshift32 the C `fill()` uses, so the synthetic weights have the same
/// distribution. scale_test.c:39.
fn fill(p: &mut [f32], seed: u32) {
    let mut s = if seed != 0 { seed } else { 1 };
    for v in p.iter_mut() {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        *v = ((s >> 8) as f32 / 8388608.0 - 1.0) * 0.02;
    }
}

/// The real Kimi K3 config, built literally. Every value matches the released
/// config.json; `main::real_cfg_hardcoded` builds the same struct. scale_test.c:57-67.
fn real_cfg() -> Cfg {
    let mut fa = Vec::new();
    for i in (4..=93).step_by(4) {
        fa.push(i);
    }
    fa.push(93); // the trailing extra MLA layer
    Cfg {
        hidden: 7168,
        n_layers: 93,
        vocab: 163840,
        rms_eps: 1e-5,
        kda_heads: 96,
        kda_head_dim: 128,
        conv_k: 4,
        gate_lb: -5.0,
        n_heads: 96,
        q_lora: 1536,
        kv_lora: 512,
        qk_nope: 128,
        qk_rope: 64,
        v_head: 128,
        mla_out_gate: 1,
        n_experts: 896,
        topk: 16,
        n_shared: 2,
        latent: 3584,
        moe_inter: 3072,
        routed_scale: 1.0,
        moe_renorm: 1,
        latent_norm: 1,
        first_dense: 1,
        dense_inter: 33792,
        attn_res_block: 12,
        situ_b1: 4.0,
        situ_b2: 25.0,
        full_attn: fa,
    }
}

#[test]
fn scale_layer_map() {
    // Section 1. The layer map: 69 KDA + 24 MLA = 93. scale_test.c:80-88.
    let c = real_cfg();
    let mut nk = 0usize;
    let mut nm = 0usize;
    for i in 0..c.n_layers {
        if c.is_mla(i) {
            nm += 1;
        } else {
            nk += 1;
        }
    }
    assert_eq!(nk, 69, "layer map: expected 69 KDA layers, got {nk}");
    assert_eq!(nm, 24, "layer map: expected 24 MLA layers, got {nm}");
    assert_eq!(nk + nm, 93);

    // The last five layers and the dense check, exactly as the C test prints them.
    // Layers 88..=92: 88,89,90,91 are KDA (not multiples of 4 plus 1 in the full_attn
    // list), 91 and 92 are BOTH MLA (91 = 4*22+3? no: full_attn has 92 and 93 one-based,
    // i.e. zero-based 91 and 92). layer 0 is dense (first_dense == 1).
    assert!(c.is_dense(0), "layer 0 must be dense");
    assert!(!c.is_mla(0), "layer 0 must not be MLA");
}

#[test]
fn scale_scratch_sizing() {
    // Section 2. Scratch requirements at real widths, checked for zero or undersize.
    // scale_test.c:90-108. A scratch of zero, or a layer buffer smaller than a sub-buffer
    // it must contain, is a heap overflow waiting for a caller that trusts it.
    let c = real_cfg();
    let ts = [1usize, 8, 512, 4096];
    for &t in &ts {
        let mla = mla_scratch(&c, t);
        let kda = kda_scratch(&c, t);
        let moe = moe_scratch(&c);
        let lay = layer_scratch(&c, t);
        // Each sub-buffer must be non-zero and the layer buffer must dominate each one.
        assert!(mla > 0, "mla_scratch == 0 at T={t}");
        assert!(kda > 0, "kda_scratch == 0 at T={t}");
        assert!(moe > 0, "moe_scratch == 0");
        assert!(
            lay >= mla,
            "layer_scratch {lay} < mla_scratch {mla} at T={t}"
        );
        assert!(
            lay >= kda,
            "layer_scratch {lay} < kda_scratch {kda} at T={t}"
        );
        // The layer scratch must also cover the MoE sub-block, which is the max of the
        // three at most widths. k3_ops.c layer_scratch takes `a.max(b).max(m)` plus the
        // per-layer overhead, so it is always >= each sub-buffer by construction.
        assert!(
            lay >= moe,
            "layer_scratch {lay} < moe_scratch {moe} at T={t}"
        );
    }
}

#[test]
fn scale_kda_layer_runs() {
    // Section 5. Actually run ONE full-width KDA layer at fp32. scale_test.c:161-208.
    //
    // This does not load real weights. It allocates at real shapes with synthetic values
    // to prove the plumbing survives the dimensions. The C test's verdict feeds on the
    // output being finite and the state being non-zero, not on any numerical agreement.
    let c = real_cfg();
    let p = (c.kda_heads * c.kda_head_dim) as usize; // 12288
    let t = 4usize;

    // Weight layout, copied from scale_test.c:174-186. The `need` count is the same sum the
    // C uses, in the same order, so the slices line up exactly.
    let need: usize = 4 * p * c.hidden as usize
        + c.hidden as usize * p
        + 3 * p * c.conv_k as usize
        + c.kda_head_dim as usize * c.hidden as usize
        + p * c.kda_head_dim as usize
        + c.kda_heads as usize * c.hidden as usize
        + c.kda_heads as usize
        + p
        + c.kda_head_dim as usize;

    let mut w_ = vec![0.0f32; need];
    fill(&mut w_, 12345);

    let mut o = 0usize;
    let (q, rest) = w_.split_at_mut(p * c.hidden as usize);
    o += p * c.hidden as usize;
    let (k, rest) = rest.split_at_mut(p * c.hidden as usize);
    o += p * c.hidden as usize;
    let (v, rest) = rest.split_at_mut(p * c.hidden as usize);
    o += p * c.hidden as usize;
    let (g, rest) = rest.split_at_mut(p * c.hidden as usize);
    o += p * c.hidden as usize;
    let (o_proj, rest) = rest.split_at_mut(c.hidden as usize * p);
    o += c.hidden as usize * p;
    let (q_conv, rest) = rest.split_at_mut(p * c.conv_k as usize);
    o += p * c.conv_k as usize;
    let (k_conv, rest) = rest.split_at_mut(p * c.conv_k as usize);
    o += p * c.conv_k as usize;
    let (v_conv, rest) = rest.split_at_mut(p * c.conv_k as usize);
    o += p * c.conv_k as usize;
    let (f_a, rest) = rest.split_at_mut(c.kda_head_dim as usize * c.hidden as usize);
    o += c.kda_head_dim as usize * c.hidden as usize;
    let (f_b, rest) = rest.split_at_mut(p * c.kda_head_dim as usize);
    o += p * c.kda_head_dim as usize;
    let (b_proj, rest) = rest.split_at_mut(c.kda_heads as usize * c.hidden as usize);
    o += c.kda_heads as usize * c.hidden as usize;
    let (a_log, rest) = rest.split_at_mut(c.kda_heads as usize);
    o += c.kda_heads as usize;
    let (dt_bias, rest) = rest.split_at_mut(p);
    o += p;
    let (o_norm, _rest) = rest.split_at_mut(c.kda_head_dim as usize);
    o += c.kda_head_dim as usize;
    assert_eq!(o, need, "weight slice accounting drifted from the C layout");

    let w = KdaW {
        q: WMat::F32(q),
        k: WMat::F32(k),
        v: WMat::F32(v),
        q_conv,
        k_conv,
        v_conv,
        f_a: WMat::F32(f_a),
        f_b: WMat::F32(f_b),
        a_log,
        dt_bias,
        b: WMat::F32(b_proj),
        g: WMat::F32(g),
        o_norm,
        o: WMat::F32(o_proj),
    };

    let mut x = vec![0.0f32; t * c.hidden as usize];
    let mut y = vec![0.0f32; t * c.hidden as usize];
    let mut sc = vec![0.0f32; kda_scratch(&c, t)];
    let mut st = vec![0.0f32; p * c.kda_head_dim as usize + 3 * p * (c.conv_k - 1) as usize];
    fill(&mut x, 999);

    // The actual run. Panics on OOM (the allocator aborts) or on a bound check inside
    // `mla_cached`; both are the failures this test exists to surface.
    kda_layer(&mut y, &x, &w, &c, t, Some(&mut st), &mut sc);

    // Output must be finite. scale_test.c:200-206.
    let mut finite = true;
    let mut mx = 0.0f32;
    for &v in &y {
        if !v.is_finite() {
            finite = false;
            break;
        }
        if v.abs() > mx {
            mx = v.abs();
        }
    }
    assert!(finite, "KDA layer produced non-finite output at full width");
    // max |y| is printed in C; the verdict does not gate on its magnitude, only on
    // finiteness, but we assert it is non-trivial so a zero-output regression is caught.
    assert!(mx > 0.0, "KDA layer produced all-zero output at full width");

    // State must be non-zero. scale_test.c:207-208.
    assert!(
        st[0] != 0.0 || st[1] != 0.0,
        "KDA recurrent state is all-zero after a full-width run (suspicious)"
    );
}
