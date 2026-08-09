// SPDX-License-Identifier: Apache-2.0
//! Full-model oracle gate. Port of `tests/unit/k3_model.c`.
//!
//! Builds the tiny model's weight store from `fixtures/tiny_k3.json` + `tiny_k3.bin`,
//! assembles `k3::ops::LayerW` per layer, and runs the three gates the C test runs:
//!
//! GATE 1 - teacher forcing over `full_ids`, comparing argmax against `tf_pred`.
//! GATE 2 - greedy decode from `prompt_ids`, comparing generated ids against `full_ids`.
//! GATE 3 - incremental decode with the MLA KV cache and carried KDA state, which must
//!          produce the identical ids to GATE 2.
//!
//! All three gates must be EXACT. The final-position logits are dumped as raw
//! little-endian f32 to `target/rust_logits.bin` for byte-comparison against a C dump.

// This harness mirrors `tests/unit/k3_model.c` statement for statement: the index loops
// and the wide per-layer signatures are what keep the two diffable side by side.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use k3::cfg::Cfg;
use k3::ops::{
    self, attn_res, decoder_layer, decoder_layer_inc, layer_scratch, rmsnorm, Attn, KdaW, LayerW,
    MlaW, MoeW, WMat,
};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;

// ------------------------------------------------------------------- weight store ----

/// The tiny-model weight store: a flat f32 blob plus a name -> offset index, exactly as
/// the C `Store`. The fixture stores every tensor as f32 (the `.bin` is
/// `total_floats * 4` bytes), so every matrix view is `WMat::F32` and every vector view is
/// `&[f32]`. There is no bf16 in the tiny model.
struct Store {
    blob: Vec<f32>,
    tensors: HashMap<String, TensorEntry>,
}

#[derive(Clone, Copy, Debug)]
struct TensorEntry {
    off: usize, // FLOAT index into `blob` (the C `offset` field)
    numel: usize,
}

impl Store {
    /// Open `tiny_k3.json` + `tiny_k3.bin` from `dir`. The `.json` carries the tensor
    /// index; the `.bin` is a flat little-endian f32 array of `total_floats` entries.
    fn open(dir: &Path) -> std::io::Result<Store> {
        let jpath = dir.join("tiny_k3.json");
        let bpath = dir.join("tiny_k3.bin");
        let jtext = fs::read_to_string(&jpath)?;
        let root: serde_json::Value = serde_json::from_str(&jtext)?;
        let ts = root
            .get("tensors")
            .and_then(|t| t.as_object())
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "tiny_k3.json: missing 'tensors' object",
                )
            })?;

        let mut tensors = HashMap::with_capacity(ts.len());
        for (name, entry) in ts {
            let off = entry
                .get("offset")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("tiny_k3.json: tensor {name} has no offset"),
                    )
                })? as usize;
            let numel = entry.get("numel").and_then(|v| v.as_i64()).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("tiny_k3.json: tensor {name} has no numel"),
                )
            })? as usize;
            tensors.insert(name.clone(), TensorEntry { off, numel });
        }

        let bytes = fs::read(&bpath)?;
        if bytes.len() % 4 != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "tiny_k3.bin: length is not a multiple of 4",
            ));
        }
        let n = bytes.len() / 4;
        let mut blob = Vec::with_capacity(n);
        for chunk in bytes.chunks_exact(4) {
            blob.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }

        Ok(Store { blob, tensors })
    }

    /// Look a tensor up by name and return its f32 slice. The C `W()` returns NULL on
    /// absence and the caller treats that as fatal; here we panic with the same message
    /// prefix so a missing weight is loud.
    fn w(&self, name: &str) -> &[f32] {
        let e = self.tensors.get(name).unwrap_or_else(|| {
            panic!("MISSING WEIGHT: {name}");
        });
        let start = e.off;
        let end = start + e.numel;
        &self.blob[start..end]
    }

    /// Build the contiguous per-expert bank the MoE indexes at a fixed stride, mirroring
    /// C `pack()`. `per` is the float count of one expert's matrix.
    fn pack(&self, layer: usize, which: &str, n: usize, per: usize) -> Vec<f32> {
        let mut bank = vec![0.0f32; per * n];
        for e in 0..n {
            let key = format!("layers_{layer}_mlp_experts_{e}_{which}_weight");
            let src = self.w(&key);
            assert_eq!(src.len(), per, "expert tensor {key} has wrong size");
            bank[e * per..(e + 1) * per].copy_from_slice(src);
        }
        bank
    }
}

// ---------------------------------------------------------------- model assembly ----

/// One decoder layer's weight data. The `Attn` and `MoeW` views are NOT stored here (they
/// would be self-referential against the expert banks); they are materialised at call
/// time by `Model::layer_w`, which borrows both `Store` (for the matrix/vector slices) and
/// this struct's owned expert banks, all under one shared lifetime.
struct LayerData {
    layer: usize,
    is_mla: bool,
    // shared norm / res-proj vectors, f32
    in_norm: Vec<f32>,
    post_norm: Vec<f32>,
    attn_res_norm: Vec<f32>,
    attn_res_proj: Vec<f32>,
    mlp_res_norm: Vec<f32>,
    mlp_res_proj: Vec<f32>,
    // MLA attention matrices (empty on KDA layers)
    mla_q_a: Vec<f32>,
    mla_q_b: Vec<f32>,
    mla_kv_a: Vec<f32>,
    mla_kv_b: Vec<f32>,
    mla_o: Vec<f32>,
    mla_g: Vec<f32>,
    mla_q_a_norm: Vec<f32>,
    mla_kv_a_norm: Vec<f32>,
    // KDA attention matrices (empty on MLA layers)
    kda_q: Vec<f32>,
    kda_k: Vec<f32>,
    kda_v: Vec<f32>,
    kda_q_conv: Vec<f32>,
    kda_k_conv: Vec<f32>,
    kda_v_conv: Vec<f32>,
    kda_f_a: Vec<f32>,
    kda_f_b: Vec<f32>,
    kda_a_log: Vec<f32>,
    kda_dt_bias: Vec<f32>,
    kda_b: Vec<f32>,
    kda_g: Vec<f32>,
    kda_o_norm: Vec<f32>,
    kda_o: Vec<f32>,
    // MoE / dense
    is_dense: bool,
    moe_gate: Vec<f32>,
    moe_bias: Vec<f32>,
    moe_down: Vec<f32>,
    moe_up: Vec<f32>,
    moe_latent_norm: Vec<f32>,
    moe_sh1: Vec<f32>,
    moe_sh3: Vec<f32>,
    moe_sh2: Vec<f32>,
    moe_w1: Vec<f32>, // packed [n_experts * per]
    moe_w3: Vec<f32>,
    moe_w2: Vec<f32>,
    dense_gate: Vec<f32>,
    dense_up: Vec<f32>,
    dense_down: Vec<f32>,
}

/// The whole model: owned embedding/head/norm, the per-layer weight data, and the
/// model-level attn-res vectors. All views into this struct borrow `&self`, so there are
/// no self-referential lifetimes and no unsafe.
struct Model {
    embed: Vec<f32>,
    lm_head: Vec<f32>,
    final_norm: Vec<f32>,
    out_res_norm: Vec<f32>,
    out_res_proj: Vec<f32>,
    layers: Vec<LayerData>,
}

fn vec_of(s: &[f32]) -> Vec<f32> {
    s.to_vec()
}

fn model_build(st: &Store, c: &Cfg) -> Model {
    let embed = vec_of(st.w("embed_tokens_weight"));
    let lm_head = vec_of(st.w("lm_head_weight"));
    let final_norm = vec_of(st.w("norm_weight"));
    let out_res_norm = vec_of(st.w("output_attn_res_norm_weight"));
    let out_res_proj = vec_of(st.w("output_attn_res_proj_weight"));

    let p13 = (c.moe_inter as usize) * (c.latent as usize);
    let p2 = (c.latent as usize) * (c.moe_inter as usize);

    let mut layers = Vec::with_capacity(c.n_layers as usize);
    for l in 0..c.n_layers as usize {
        let is_mla = c.is_mla(l as i32);
        let is_dense = c.is_dense(l as i32);

        let in_norm = vec_of(st.w(&format!("layers_{l}_input_layernorm_weight")));
        let post_norm = vec_of(st.w(&format!("layers_{l}_post_attention_layernorm_weight")));
        let attn_res_norm = vec_of(st.w(&format!("layers_{l}_self_attention_res_norm_weight")));
        let attn_res_proj = vec_of(st.w(&format!("layers_{l}_self_attention_res_proj_weight")));
        let mlp_res_norm = vec_of(st.w(&format!("layers_{l}_mlp_res_norm_weight")));
        let mlp_res_proj = vec_of(st.w(&format!("layers_{l}_mlp_res_proj_weight")));

        let (mla_q_a, mla_q_b, mla_kv_a, mla_kv_b, mla_o, mla_g, mla_q_a_norm, mla_kv_a_norm);
        let (
            kda_q,
            kda_k,
            kda_v,
            kda_q_conv,
            kda_k_conv,
            kda_v_conv,
            kda_f_a,
            kda_f_b,
            kda_a_log,
            kda_dt_bias,
            kda_b,
            kda_g,
            kda_o_norm,
            kda_o,
        );

        if is_mla {
            mla_q_a = vec_of(st.w(&format!("layers_{l}_self_attn_q_a_proj_weight")));
            mla_q_b = vec_of(st.w(&format!("layers_{l}_self_attn_q_b_proj_weight")));
            mla_kv_a = vec_of(st.w(&format!("layers_{l}_self_attn_kv_a_proj_with_mqa_weight")));
            mla_kv_b = vec_of(st.w(&format!("layers_{l}_self_attn_kv_b_proj_weight")));
            mla_o = vec_of(st.w(&format!("layers_{l}_self_attn_o_proj_weight")));
            mla_g = vec_of(st.w(&format!("layers_{l}_self_attn_g_proj_weight")));
            mla_q_a_norm = vec_of(st.w(&format!("layers_{l}_self_attn_q_a_layernorm_weight")));
            mla_kv_a_norm = vec_of(st.w(&format!("layers_{l}_self_attn_kv_a_layernorm_weight")));
            // std implements Default for tuples only up to 12 elements, so the KDA
            // slots are cleared one at a time.
            kda_q = Vec::new();
            kda_k = Vec::new();
            kda_v = Vec::new();
            kda_q_conv = Vec::new();
            kda_k_conv = Vec::new();
            kda_v_conv = Vec::new();
            kda_f_a = Vec::new();
            kda_f_b = Vec::new();
            kda_a_log = Vec::new();
            kda_dt_bias = Vec::new();
            kda_b = Vec::new();
            kda_g = Vec::new();
            kda_o_norm = Vec::new();
            kda_o = Vec::new();
        } else {
            (
                mla_q_a,
                mla_q_b,
                mla_kv_a,
                mla_kv_b,
                mla_o,
                mla_g,
                mla_q_a_norm,
                mla_kv_a_norm,
            ) = Default::default();
            kda_q = vec_of(st.w(&format!("layers_{l}_self_attn_q_proj_weight")));
            kda_k = vec_of(st.w(&format!("layers_{l}_self_attn_k_proj_weight")));
            kda_v = vec_of(st.w(&format!("layers_{l}_self_attn_v_proj_weight")));
            kda_q_conv = vec_of(st.w(&format!("layers_{l}_self_attn_q_conv1d_weight")));
            kda_k_conv = vec_of(st.w(&format!("layers_{l}_self_attn_k_conv1d_weight")));
            kda_v_conv = vec_of(st.w(&format!("layers_{l}_self_attn_v_conv1d_weight")));
            kda_f_a = vec_of(st.w(&format!("layers_{l}_self_attn_f_a_proj_weight")));
            kda_f_b = vec_of(st.w(&format!("layers_{l}_self_attn_f_b_proj_weight")));
            kda_a_log = vec_of(st.w(&format!("layers_{l}_self_attn_A_log")));
            kda_dt_bias = vec_of(st.w(&format!("layers_{l}_self_attn_dt_bias")));
            kda_b = vec_of(st.w(&format!("layers_{l}_self_attn_b_proj_weight")));
            kda_g = vec_of(st.w(&format!("layers_{l}_self_attn_g_proj_weight")));
            kda_o_norm = vec_of(st.w(&format!("layers_{l}_self_attn_o_norm_weight")));
            kda_o = vec_of(st.w(&format!("layers_{l}_self_attn_o_proj_weight")));
        }

        let (
            is_dense_flag,
            moe_gate,
            moe_bias,
            moe_down,
            moe_up,
            moe_latent_norm,
            moe_sh1,
            moe_sh3,
            moe_sh2,
            moe_w1,
            moe_w3,
            moe_w2,
            dense_gate,
            dense_up,
            dense_down,
        );
        if is_dense {
            is_dense_flag = true;
            dense_gate = vec_of(st.w(&format!("layers_{l}_mlp_gate_proj_weight")));
            dense_up = vec_of(st.w(&format!("layers_{l}_mlp_up_proj_weight")));
            dense_down = vec_of(st.w(&format!("layers_{l}_mlp_down_proj_weight")));
            (
                moe_gate,
                moe_bias,
                moe_down,
                moe_up,
                moe_latent_norm,
                moe_sh1,
                moe_sh3,
                moe_sh2,
                moe_w1,
                moe_w3,
                moe_w2,
            ) = Default::default();
        } else {
            is_dense_flag = false;
            moe_gate = vec_of(st.w(&format!("layers_{l}_mlp_gate_weight")));
            moe_bias = vec_of(st.w(&format!("layers_{l}_mlp_e_score_correction_bias")));
            moe_down = vec_of(st.w(&format!("layers_{l}_mlp_down_weight")));
            moe_up = vec_of(st.w(&format!("layers_{l}_mlp_up_weight")));
            moe_latent_norm = vec_of(st.w(&format!("layers_{l}_mlp_norm_weight")));
            moe_sh1 = vec_of(st.w(&format!("layers_{l}_mlp_shared_w1_weight")));
            moe_sh3 = vec_of(st.w(&format!("layers_{l}_mlp_shared_w3_weight")));
            moe_sh2 = vec_of(st.w(&format!("layers_{l}_mlp_shared_w2_weight")));
            moe_w1 = st.pack(l, "w1", c.n_experts as usize, p13);
            moe_w3 = st.pack(l, "w3", c.n_experts as usize, p13);
            moe_w2 = st.pack(l, "w2", c.n_experts as usize, p2);
            (dense_gate, dense_up, dense_down) = Default::default();
        }

        layers.push(LayerData {
            layer: l,
            is_mla,
            in_norm,
            post_norm,
            attn_res_norm,
            attn_res_proj,
            mlp_res_norm,
            mlp_res_proj,
            mla_q_a,
            mla_q_b,
            mla_kv_a,
            mla_kv_b,
            mla_o,
            mla_g,
            mla_q_a_norm,
            mla_kv_a_norm,
            kda_q,
            kda_k,
            kda_v,
            kda_q_conv,
            kda_k_conv,
            kda_v_conv,
            kda_f_a,
            kda_f_b,
            kda_a_log,
            kda_dt_bias,
            kda_b,
            kda_g,
            kda_o_norm,
            kda_o,
            is_dense: is_dense_flag,
            moe_gate,
            moe_bias,
            moe_down,
            moe_up,
            moe_latent_norm,
            moe_sh1,
            moe_sh3,
            moe_sh2,
            moe_w1,
            moe_w3,
            moe_w2,
            dense_gate,
            dense_up,
            dense_down,
        });
    }

    Model {
        embed,
        lm_head,
        final_norm,
        out_res_norm,
        out_res_proj,
        layers,
    }
}

impl Model {
    /// Build a `LayerW` borrowing this layer's owned weight data. All slices borrow
    /// `&self`, so the `LayerW<'_>` is valid for as long as the caller holds the borrow.
    /// No unsafe: the expert banks and the matrix/vector views all live in `LayerData`,
    /// owned by `self`.
    fn layer_w(&self, l: usize) -> LayerW<'_> {
        let s = &self.layers[l];
        let attn = if s.is_mla {
            let g = if !s.mla_g.is_empty() {
                Some(WMat::F32(&s.mla_g[..]))
            } else {
                None
            };
            Attn::Mla(MlaW {
                q_a: WMat::F32(&s.mla_q_a[..]),
                q_b: WMat::F32(&s.mla_q_b[..]),
                kv_a: WMat::F32(&s.mla_kv_a[..]),
                kv_b: WMat::F32(&s.mla_kv_b[..]),
                o: WMat::F32(&s.mla_o[..]),
                g,
                q_a_norm: &s.mla_q_a_norm[..],
                kv_a_norm: &s.mla_kv_a_norm[..],
            })
        } else {
            Attn::Kda(KdaW {
                q: WMat::F32(&s.kda_q[..]),
                k: WMat::F32(&s.kda_k[..]),
                v: WMat::F32(&s.kda_v[..]),
                q_conv: &s.kda_q_conv[..],
                k_conv: &s.kda_k_conv[..],
                v_conv: &s.kda_v_conv[..],
                f_a: WMat::F32(&s.kda_f_a[..]),
                f_b: WMat::F32(&s.kda_f_b[..]),
                a_log: &s.kda_a_log[..],
                dt_bias: &s.kda_dt_bias[..],
                b: WMat::F32(&s.kda_b[..]),
                g: WMat::F32(&s.kda_g[..]),
                o_norm: &s.kda_o_norm[..],
                o: WMat::F32(&s.kda_o[..]),
            })
        };

        let moe = if s.is_dense {
            None
        } else {
            Some(MoeW {
                gate: &s.moe_gate[..],
                bias: Some(&s.moe_bias[..]),
                w1: Some(&s.moe_w1[..]),
                w3: Some(&s.moe_w3[..]),
                w2: Some(&s.moe_w2[..]),
                latent_norm: Some(&s.moe_latent_norm[..]),
                down: WMat::F32(&s.moe_down[..]),
                up: WMat::F32(&s.moe_up[..]),
                sh1: WMat::F32(&s.moe_sh1[..]),
                sh3: WMat::F32(&s.moe_sh3[..]),
                sh2: WMat::F32(&s.moe_sh2[..]),
                layer: s.layer,
                cache_only: false,
            })
        };

        LayerW {
            in_norm: &s.in_norm[..],
            post_norm: &s.post_norm[..],
            attn_res_norm: &s.attn_res_norm[..],
            attn_res_proj: &s.attn_res_proj[..],
            mlp_res_norm: &s.mlp_res_norm[..],
            mlp_res_proj: &s.mlp_res_proj[..],
            attn,
            moe,
            dense_gate: if s.is_dense {
                Some(WMat::F32(&s.dense_gate[..]))
            } else {
                None
            },
            dense_up: if s.is_dense {
                Some(WMat::F32(&s.dense_up[..]))
            } else {
                None
            },
            dense_down: if s.is_dense {
                Some(WMat::F32(&s.dense_down[..]))
            } else {
                None
            },
        }
    }
}

// ---------------------------------------------------------------- forward ----

/// The full forward pass over `T` tokens, writing `logits[T * vocab]`. Mirrors C
/// `forward()` statement for statement: embed, zero block-residual and KDA state, run
/// every layer, the model-level attn-res aggregator, then final norm + lm_head.
fn forward(
    m: &Model,
    c: &Cfg,
    ids: &[i32],
    t_len: usize,
    logits: &mut [f32],
    scratch: &mut [f32],
    h: &mut [f32],
    br: &mut [f32],
    kstate: &mut [f32],
) {
    let e = c.hidden as usize;
    let maxb = (c.n_layers / c.attn_res_block + 2) as usize;
    let p = (c.kda_heads * c.kda_head_dim) as usize;
    let kper = p * c.kda_head_dim as usize + 3 * p * (c.conv_k as usize - 1);

    for t in 0..t_len {
        let row = ids[t] as usize * e;
        h[t * e..(t + 1) * e].copy_from_slice(&m.embed[row..row + e]);
    }

    for x in br[..t_len * maxb * e].iter_mut() {
        *x = 0.0;
    }
    for x in kstate[..kper * c.n_layers as usize].iter_mut() {
        *x = 0.0;
    }

    let mut nb = 0usize;
    for l in 0..c.n_layers as usize {
        let w = m.layer_w(l);
        decoder_layer(
            h,
            br,
            &mut nb,
            &w,
            c,
            l as i32,
            t_len,
            Some(&mut kstate[kper * l..kper * (l + 1)]),
            scratch,
            None,
        );
    }

    // The model-level aggregator: output_attn_res_{norm,proj}. One more beyond the 2 per
    // layer. Skipping it is silent.
    let (fold, src) = scratch.split_at_mut(e);
    for i in 0..e {
        fold[i] = m.out_res_norm[i] * m.out_res_proj[i];
    }
    for t in 0..t_len {
        for b in 0..nb {
            src[b * e..(b + 1) * e]
                .copy_from_slice(&br[(t * maxb + b) * e..(t * maxb + b + 1) * e]);
        }
        src[nb * e..(nb + 1) * e].copy_from_slice(&h[t * e..(t + 1) * e]);
        attn_res(
            &mut h[t * e..(t + 1) * e],
            &src[..(nb + 1) * e],
            fold,
            nb + 1,
            e,
            c.rms_eps,
        );
    }

    let nrm = &mut scratch[..e];
    for t in 0..t_len {
        rmsnorm(nrm, &h[t * e..(t + 1) * e], &m.final_norm, c.rms_eps);
        let out = &mut logits[t * c.vocab as usize..(t + 1) * c.vocab as usize];
        ops::matmul(out, nrm, &m.lm_head, e);
    }
}

/// The incremental forward pass for GATE 3. Mirrors the C loop: the first call feeds the
/// whole prompt, later calls feed one token, carrying the MLA KV cache and KDA state.
/// Only the LAST new position's logits are computed. Writes `vocab` logits into `lg`.
fn forward_inc(
    m: &Model,
    c: &Cfg,
    ids: &[i32],
    base: usize,
    n_t: usize,
    cap: usize,
    lg: &mut [f32],
    scratch: &mut [f32],
    h: &mut [f32],
    br: &mut [f32],
    kstate: &mut [f32],
    kvc: &mut [f32],
    rpc: &mut [f32],
) {
    let e = c.hidden as usize;
    let maxb = (c.n_layers / c.attn_res_block + 2) as usize;
    let p = (c.kda_heads * c.kda_head_dim) as usize;
    let kper = p * c.kda_head_dim as usize + 3 * p * (c.conv_k as usize - 1);
    let h_heads = c.n_heads as usize;
    let kvd = (c.qk_nope + c.v_head) as usize;
    let kvper = cap * h_heads * kvd;
    let rpper = cap * c.qk_rope as usize;

    for t in 0..n_t {
        let row = ids[base + t] as usize * e;
        h[t * e..(t + 1) * e].copy_from_slice(&m.embed[row..row + e]);
    }

    for x in br[..n_t * maxb * e].iter_mut() {
        *x = 0.0;
    }

    let mut nb = 0usize;
    for l in 0..c.n_layers as usize {
        let w = m.layer_w(l);
        decoder_layer_inc(
            h,
            br,
            &mut nb,
            &w,
            c,
            l as i32,
            n_t,
            Some(&mut kstate[kper * l..kper * (l + 1)]),
            scratch,
            Some(&mut kvc[kvper * l..kvper * (l + 1)]),
            Some(&mut rpc[rpper * l..rpper * (l + 1)]),
            base,
            cap,
            None,
        );
    }

    let lastt = n_t - 1;
    let (fold, src) = scratch.split_at_mut(e);
    for i in 0..e {
        fold[i] = m.out_res_norm[i] * m.out_res_proj[i];
    }
    for b in 0..nb {
        src[b * e..(b + 1) * e]
            .copy_from_slice(&br[(lastt * maxb + b) * e..(lastt * maxb + b + 1) * e]);
    }
    src[nb * e..(nb + 1) * e].copy_from_slice(&h[lastt * e..(lastt + 1) * e]);
    attn_res(
        &mut h[lastt * e..(lastt + 1) * e],
        &src[..(nb + 1) * e],
        fold,
        nb + 1,
        e,
        c.rms_eps,
    );

    let nrm = &mut scratch[..e];
    rmsnorm(
        nrm,
        &h[lastt * e..(lastt + 1) * e],
        &m.final_norm,
        c.rms_eps,
    );
    ops::matmul(lg, nrm, &m.lm_head, e);
}

fn argmax(v: &[f32]) -> i32 {
    let mut b = 0;
    for i in 1..v.len() {
        if v[i] > v[b] {
            b = i;
        }
    }
    b as i32
}

// ---------------------------------------------------------------- config ----

/// Build a `Cfg` from `ref_k3.json`'s nested `config` object. `Cfg::load` is not present
/// in `src/cfg.rs` (the reader lives in the C `k3_cfg.h`, not yet ported), so we parse with
/// `serde_json` and construct `Cfg` literally, mirroring `k3_cfg_load` field for field.
fn load_cfg(dir: &Path) -> std::io::Result<Cfg> {
    let text = fs::read_to_string(dir.join("ref_k3.json"))?;
    let root: serde_json::Value = serde_json::from_str(&text)?;
    let jc = root.get("config").ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "ref_k3.json: no config")
    })?;

    let i = |k: &str| -> i32 {
        jc.get(k)
            .and_then(|v| v.as_i64())
            .unwrap_or_else(|| panic!("ref_k3.json: config missing {k}")) as i32
    };
    let f = |k: &str| -> f32 {
        jc.get(k)
            .and_then(|v| v.as_f64())
            .unwrap_or_else(|| panic!("ref_k3.json: config missing {k}")) as f32
    };
    let b = |k: &str, dflt: i32| -> i32 {
        match jc.get(k) {
            Some(serde_json::Value::Bool(true)) => 1,
            Some(serde_json::Value::Bool(false)) => 0,
            Some(v) => v
                .as_i64()
                .map(|n| if n != 0 { 1 } else { 0 })
                .unwrap_or(dflt),
            None => dflt,
        }
    };

    let full_attn = jc
        .get("full_attn_layers")
        .and_then(|v| v.as_array())
        .unwrap_or_else(|| panic!("ref_k3.json: config missing full_attn_layers"))
        .iter()
        .map(|v| {
            v.as_i64()
                .unwrap_or_else(|| panic!("ref_k3.json: full_attn_layers entry not a number"))
                as i32
        })
        .collect::<Vec<i32>>();

    let cfg = Cfg {
        hidden: i("hidden_size"),
        n_layers: i("num_hidden_layers"),
        vocab: i("vocab_size"),
        rms_eps: f("rms_norm_eps"),
        kda_heads: i("kda_num_heads"),
        kda_head_dim: i("kda_head_dim"),
        conv_k: i("short_conv_kernel_size"),
        gate_lb: f("gate_lower_bound"),
        n_heads: i("num_attention_heads"),
        q_lora: i("q_lora_rank"),
        kv_lora: i("kv_lora_rank"),
        qk_nope: i("qk_nope_head_dim"),
        qk_rope: i("qk_rope_head_dim"),
        v_head: i("v_head_dim"),
        mla_out_gate: b("mla_use_output_gate", 1),
        n_experts: i("num_experts"),
        topk: i("num_experts_per_token"),
        n_shared: i("num_shared_experts"),
        latent: i("routed_expert_hidden_size"),
        moe_inter: i("moe_intermediate_size"),
        routed_scale: f("routed_scaling_factor"),
        moe_renorm: b("moe_renormalize", 1),
        latent_norm: b("latent_moe_use_norm", 1),
        first_dense: i("first_k_dense_replace"),
        dense_inter: i("intermediate_size"),
        attn_res_block: i("attn_res_block_size"),
        situ_b1: f("situ_beta"),
        situ_b2: f("situ_linear_beta"),
        full_attn,
    };

    // Structural checks, mirroring k3_cfg_load. Any failure here is a real config error.
    assert!(
        cfg.n_layers > 0 && cfg.hidden > 0 && cfg.vocab > 0,
        "non-positive dims"
    );
    assert!(
        (cfg.full_attn.len() as i32) < cfg.n_layers,
        "all layers marked full attention"
    );
    for (idx, &l) in cfg.full_attn.iter().enumerate() {
        assert!(
            l >= 1 && l <= cfg.n_layers,
            "full_attn_layers[{idx}] = {l} outside 1..{}",
            cfg.n_layers
        );
    }
    assert!(
        cfg.topk <= k3::cfg::MAX_TOPK as i32,
        "topk exceeds MAX_TOPK"
    );
    assert!(cfg.topk <= cfg.n_experts, "topk > n_experts");
    assert!(cfg.attn_res_block > 0, "attn_res_block_size <= 0");
    assert!(cfg.conv_k >= 1, "short_conv_kernel_size < 1");

    Ok(cfg)
}

// ---------------------------------------------------------------- the gate ----

#[test]
fn model_oracle() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");

    let st = Store::open(&dir).unwrap_or_else(|e| {
        eprintln!(
            "GATE ABORTED: cannot load checkpoint from '{}'",
            dir.display()
        );
        eprintln!("  {e}");
        panic!("cannot load tiny model");
    });

    let c = load_cfg(&dir).unwrap_or_else(|e| {
        eprintln!(
            "GATE ABORTED: config in {}/ref_k3.json could not be read",
            dir.display()
        );
        eprintln!("  {e}");
        panic!("cannot load config");
    });

    // The reference arrays.
    let rtext = fs::read_to_string(dir.join("ref_k3.json")).unwrap();
    let rref: serde_json::Value = serde_json::from_str(&rtext).unwrap();
    let prompt_ids: Vec<i32> = rref["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    let full_ids: Vec<i32> = rref["full_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    let tf_pred: Vec<i32> = rref["tf_pred"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();

    println!("Kimi K3 pure-C engine, full-model oracle gate");
    println!("fixtures: {}\n", dir.display());

    // The config summary line, matching k3_cfg_load's banner.
    println!(
        "config: {}/ref_k3.json (flat shape) | hidden={} layers={} vocab={} | {} MLA + {} KDA | \
         experts {} top{} shared{} | latent={}",
        dir.display(),
        c.hidden,
        c.n_layers,
        c.vocab,
        c.full_attn.len(),
        c.n_layers as usize - c.full_attn.len(),
        c.n_experts,
        c.topk,
        c.n_shared,
        c.latent
    );

    // Layer map, matching the C banner.
    print!("layer map (0-based): ");
    for i in 0..c.n_layers as usize {
        print!("{}", if c.is_mla(i as i32) { "M" } else { "K" });
    }
    println!("   (M=MLA, K=KDA; dense layer = 0)");
    print!("attn_res boundaries at: ");
    for i in 0..c.n_layers as usize {
        if i % c.attn_res_block as usize == 0 {
            print!("{i} ");
        }
    }
    println!("\n");

    println!("checkpoint: {} tensors loaded", st.tensors.len());
    println!(
        "prompt_ids {}, full_ids {}, tf_pred {}\n",
        prompt_ids.len(),
        full_ids.len(),
        tf_pred.len()
    );

    let m = model_build(&st, &c);
    println!("all layer weights bound\n");

    let t_len = full_ids.len();
    let np = prompt_ids.len();
    let e = c.hidden as usize;
    let maxb = (c.n_layers / c.attn_res_block + 2) as usize;
    let p = (c.kda_heads * c.kda_head_dim) as usize;
    let kper = p * c.kda_head_dim as usize + 3 * p * (c.conv_k as usize - 1);

    let mut need = layer_scratch(&c, t_len);
    let alt = (maxb + 2) * e + c.vocab as usize;
    if alt > need {
        need = alt;
    }

    let mut scratch = vec![0.0f32; need];
    let mut h = vec![0.0f32; t_len * e];
    let mut br = vec![0.0f32; t_len * maxb * e];
    let mut ks = vec![0.0f32; kper * c.n_layers as usize];
    let mut lg = vec![0.0f32; t_len * c.vocab as usize];

    // ---- GATE 1: teacher forcing ----
    forward(
        &m,
        &c,
        &full_ids,
        t_len,
        &mut lg,
        &mut scratch,
        &mut h,
        &mut br,
        &mut ks,
    );
    let mut tf_all = 0;
    let mut tf_gen = 0;
    let mut tf_gen_ok = 0;
    for i in 0..t_len {
        let got = argmax(&lg[i * c.vocab as usize..(i + 1) * c.vocab as usize]);
        let want = tf_pred[i];
        if got == want {
            tf_all += 1;
        }
        if i >= np - 1 && i < t_len - 1 {
            tf_gen += 1;
            if got == want {
                tf_gen_ok += 1;
            }
        }
    }
    println!("GATE 1  teacher forcing : {tf_all}/{t_len} positions match tf_pred");
    println!("        generated span  : {tf_gen_ok}/{tf_gen}  <- must be exact");

    // ---- GATE 2: greedy decode ----
    let mut gen = vec![0i32; t_len];
    gen[..np].copy_from_slice(&prompt_ids);
    let mut cur = np;
    let mut gok = 0;
    while cur < t_len {
        forward(
            &m,
            &c,
            &gen,
            cur,
            &mut lg,
            &mut scratch,
            &mut h,
            &mut br,
            &mut ks,
        );
        gen[cur] = argmax(&lg[(cur - 1) * c.vocab as usize..cur * c.vocab as usize]);
        if gen[cur] == full_ids[cur] {
            gok += 1;
        }
        cur += 1;
    }
    println!(
        "GATE 2  greedy decode   : {gok}/{} generated tokens match full_ids",
        t_len - np
    );

    // ---- GATE 3: incremental decode ----
    let h_heads = c.n_heads as usize;
    let kvd = (c.qk_nope + c.v_head) as usize;
    let kvper = t_len * h_heads * kvd;
    let rpper = t_len * c.qk_rope as usize;
    let mut kvc = vec![0.0f32; kvper * c.n_layers as usize];
    let mut rpc = vec![0.0f32; rpper * c.n_layers as usize];
    let mut sc_i = vec![0.0f32; need];
    let mut h_i = vec![0.0f32; t_len * e];
    let mut br_i = vec![0.0f32; t_len * maxb * e];
    let mut ks_i = vec![0.0f32; kper * c.n_layers as usize];
    let mut lg_i = vec![0.0f32; c.vocab as usize];
    let mut gi = vec![0i32; t_len];
    gi[..np].copy_from_slice(&prompt_ids);
    for x in ks_i.iter_mut() {
        *x = 0.0;
    }
    let mut iok = 0;
    let mut cached = 0usize;
    let mut step = 0usize;
    while cached < t_len - 1 || step == 0 {
        let base = cached;
        let n_t = if step == 0 { np } else { 1 };

        forward_inc(
            &m, &c, &gi, base, n_t, t_len, &mut lg_i, &mut sc_i, &mut h_i, &mut br_i, &mut ks_i,
            &mut kvc, &mut rpc,
        );
        cached = base + n_t;
        if cached >= t_len {
            break;
        }
        gi[cached] = argmax(&lg_i);
        step += 1;
    }
    for i in np..t_len {
        if gi[i] == full_ids[i] {
            iok += 1;
        }
    }
    println!(
        "GATE 3  incremental    : {iok}/{} generated tokens match full_ids  <- KV cache + carried KDA state",
        t_len - np
    );
    if iok != t_len - np {
        gok = -1;
    }

    let pass = (tf_gen_ok == tf_gen) && (gok == (t_len - np) as i32);
    println!();
    println!(
        "VERDICT: {}",
        if pass {
            "ENGINE MATCHES THE REFERENCE EXACTLY"
        } else {
            "MISMATCH, see the counts above"
        }
    );
    if !pass {
        print!("\n  ref  ");
        for i in np..t_len.min(np + 14) {
            print!("{:4}", full_ids[i]);
        }
        print!("\n  got  ");
        for i in np..t_len.min(np + 14) {
            print!("{:4}", gen[i]);
        }
        println!();
    }

    // ---- logits dump ----
    // The final-position logits from the GATE 1 teacher-forcing sweep, as raw
    // little-endian f32, for byte-comparison against a C dump on the same machine.
    let last_logits = &lg[(t_len - 1) * c.vocab as usize..t_len * c.vocab as usize];
    let target_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target");
    let dump_path = target_dir.join("rust_logits.bin");
    let dump_ok = (|| -> std::io::Result<()> {
        fs::create_dir_all(&target_dir)?;
        let mut f = fs::File::create(&dump_path)?;
        for &v in last_logits {
            f.write_all(&v.to_le_bytes())?;
        }
        f.flush()?;
        Ok(())
    })();
    match &dump_ok {
        Ok(()) => println!("\nlogits dumped to {}", dump_path.display()),
        Err(e) => eprintln!(
            "WARNING: could not write logits dump to {}: {}",
            dump_path.display(),
            e
        ),
    }

    assert!(pass, "ORACLE MISMATCH: gates did not all pass exactly");
    assert_eq!(tf_gen_ok, tf_gen, "GATE 1 generated span not exact");
    assert_eq!(gok, (t_len - np) as i32, "GATE 2 greedy decode not exact");
    assert_eq!(iok, t_len - np, "GATE 3 incremental decode not exact");
}
