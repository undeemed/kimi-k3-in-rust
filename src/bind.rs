// SPDX-License-Identifier: Apache-2.0
//! Bind real checkpoint tensors into the engine's weight structs.
//!
//! Port of `src/model/k3_bind.c` and `src/model/k3_bind.h`.
//!
//! WHAT THIS SOLVES
//!   `st` finds any tensor by name; `ops` describes what the kernels want. Nothing
//!   connects the two. This does, for one layer at a time, and it validates as it goes:
//!   every tensor's element count is checked against what the config implies BEFORE the
//!   bytes are read. A wrong assumption about a shape then fails loudly at load instead
//!   of silently producing a transposed or truncated matrix.
//!
//! TWO STORAGE CLASSES (k3_bind.h:33)
//!   Large matrices keep the checkpoint's own bf16 bytes and are widened inside `mmw`.
//!   Small vectors are widened here to fp32, because the kernels that consume them
//!   (`rmsnorm`, `shortconv`, `kda_decay`, `router`) index them ELEMENTWISE, where a
//!   silently wrong element type is not a crash but a different model.
//!
//! The split is a 1:1 port of the C `Plan`/`Req` system. `LayerPlan` holds the per-tensor
//! decisions; `LayerBind::from_shards` and the streaming trunk both go through the same
//! `widen`/`views` so the wide/narrow decisions cannot diverge.
//!
//! THE TWO PATHS. The trunk path stores the run as raw disk bytes (bf16 for narrow AND
//! for wide vectors) and `widen` copies the bf16 vectors into a separate fp32 buffer.
//! The shard path reads the small vectors directly as fp32 via `St::read_f32`, so the
//! blob already holds f32 for them and `widen` is a no-op (it points into the run). The
//! `run_dtype` per slot records which is which, so `widen`/`views` work identically over
//! both: a wide tensor with `run_dtype == F32` is pointed at in place; a wide tensor
//! with `run_dtype == BF16` is widened into the buffer. The demoted fallback (a large
//! matrix that is not bf16) reads every matrix as f32, so `run_dtype == F32` everywhere
//! and the widen buffer is unused.

use crate::cfg::Cfg;
use crate::io_util::AlignedBuf;
use crate::ops::{Attn, KdaW, LayerW, MlaW, MoeW, WMat};
use crate::st::{Dtype, St};
use std::io::{Error, ErrorKind, Result};

const PRE: &str = "language_model.model.";

/// The byte offset that means "this wide tensor lives in the run, not the widen buffer".
/// `widen_off == RUN_SENTINEL` tells `views` to point straight into `run` (already f32).
const RUN_SENTINEL: i64 = -1;

/// One tensor's role in the layer, so `views` knows which field of `LayerW`/`KdaW`/`MlaW`/
/// `MoeW` it materialises. Mirrors the `reqw`/`reqn` calls in C `plan_layer`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    // Always-wide fold vectors, read elementwise.
    InNorm,
    PostNorm,
    AttnResNorm,
    AttnResProj,
    MlpResNorm,
    MlpResProj,
    // MLA.
    MlaQA,
    MlaQaNorm,
    MlaQB,
    MlaKvA,
    MlaKvANorm,
    MlaKvB,
    MlaO,
    MlaG,
    // KDA.
    KdaQ,
    KdaK,
    KdaV,
    KdaG,
    KdaO,
    KdaQConv,
    KdaKConv,
    KdaVConv,
    KdaFA,
    KdaFB,
    KdaB,
    KdaALog,
    KdaDtBias,
    KdaONorm,
    // Dense MLP (layer 0).
    DenseGate,
    DenseUp,
    DenseDown,
    // MoE.
    MoeGate,
    MoeBias,
    MoeDown,
    MoeUp,
    MoeLatentNorm,
    MoeSh1,
    MoeSh3,
    MoeSh2,
}

/// One requested tensor. `off` is the within-run/blob offset used by `widen`/`views`;
/// `nbytes`/`dtype` are the checkpoint's (from `find`); `run_dtype` is what is ACTUALLY
/// in the run at `off` (BF16 raw, F32 already-widened, or I8R), which is what `widen` and
/// `views` dispatch on.
#[derive(Clone, Debug)]
struct Slot {
    name: String,
    role: Role,
    want: i64,        // elements the engine expects; -1 accepts whatever.
    take: i64,        // elements actually used (A_log takes a prefix).
    narrow: bool,     // true = keep the checkpoint's bf16/i8r bytes in the run.
    off: i64,         // within-run offset used by widen/views.
    nbytes: i64,      // tensor nbytes from find (raw bytes on disk).
    dtype: Dtype,     // checkpoint dtype (from find).
    run_dtype: Dtype, // what is actually in the run at `off`.
    widen_off: i64,   // offset in the widen buffer, or RUN_SENTINEL if it lives in the run.
}

/// The plan half of a layer binding. Holds the per-tensor decisions; both `from_shards`
/// and the streaming trunk go through it so the wide/narrow decisions cannot diverge.
pub struct LayerPlan {
    slots: Vec<Slot>,
    layer: usize,
    is_mla: bool,
    is_dense: bool,
}

impl LayerPlan {
    /// Resolve every tensor the layer needs via `find`, validate element counts, and
    /// lay out the blob. `find(name)` returns `(off, nbytes, dtype)` where `off` is the
    /// within-run offset (trunk path) or the absolute shard offset (from_shards path,
    /// where it is only used to read and is then overwritten with the blob offset).
    pub fn build(
        c: &Cfg,
        layer: usize,
        find: &dyn Fn(&str) -> Option<(i64, i64, Dtype)>,
    ) -> Result<LayerPlan> {
        let is_mla = c.is_mla(layer as i32);
        let is_dense = c.is_dense(layer as i32);
        let mut plan = LayerPlan {
            slots: Vec::new(),
            layer,
            is_mla,
            is_dense,
        };
        plan.plan_layer(c, true);
        let demoted = plan.resolve(find)?;
        // If any narrow tensor is not bf16, redo the whole layer at fp32 (narrow_ok =
        // false). The dtype tag is per struct in C, so a mixed layer cannot be described;
        // the wholesale fallback is the only correct option. k3_bind.c:262-274.
        if demoted > 0 {
            let mut plan2 = LayerPlan {
                slots: Vec::new(),
                layer,
                is_mla,
                is_dense,
            };
            plan2.plan_layer(c, false);
            plan2.resolve(find)?;
            return Ok(plan2);
        }
        Ok(plan)
    }

    /// Resolve every slot via `find`, validate element counts, and demote narrow tensors
    /// that are not bf16. Returns the demotion count so `build` can trigger the wholesale
    /// fp32 fallback. k3_bind.c:71.
    fn resolve(&mut self, find: &dyn Fn(&str) -> Option<(i64, i64, Dtype)>) -> Result<usize> {
        let mut demoted = 0usize;
        for s in &mut self.slots {
            let (off, nbytes, dtype) = find(&s.name).ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidData,
                    format!("k3_bind: missing tensor {}", s.name),
                )
            })?;
            let esz = dtype.elemsize() as i64;
            if esz == 0 {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!("k3_bind: {} has unsupported dtype {:?}", s.name, dtype),
                ));
            }
            let have = nbytes / esz;
            // The check that earns its keep: a shape the engine did not expect means the
            // config and the checkpoint disagree, and every kernel downstream would read
            // the wrong strides while producing plausible numbers. k3_bind.c:86.
            if s.want >= 0 && have != s.want {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "k3_bind: {} has {} elements, engine expects {}",
                        s.name, have, s.want
                    ),
                ));
            }
            if s.take > have {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "k3_bind: {}: asked for {} of {} elements",
                        s.name, s.take, have
                    ),
                ));
            }
            // Narrow storage is only legal when the checkpoint really holds bf16. If a
            // tensor ships F32, keeping "its own bytes" would mean handing 4-byte floats
            // to a kernel that reads 2-byte elements. Demoting just this tensor is NOT
            // enough because the dtype tag lives on the STRUCT, not the field: one
            // demoted tensor inside a struct tagged BF16 would be read as bf16 anyway.
            // So record it and let the caller fall back wholesale. k3_bind.c:108.
            if s.narrow && dtype != Dtype::Bf16 && dtype != Dtype::I8R {
                s.narrow = false;
                demoted += 1;
            }
            // reqn always takes the whole tensor; a partial take of a narrow tensor is not
            // implemented. k3_bind.c:112.
            if s.narrow && s.take != have {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "k3_bind: {}: a partial take of a narrow tensor is not implemented ({} of {})",
                        s.name, s.take, have
                    ),
                ));
            }
            s.off = off;
            s.nbytes = nbytes;
            s.dtype = dtype;
            // Default: the run holds the checkpoint's own bytes (trunk path). from_shards
            // overrides run_dtype to F32 for the small vectors it reads via read_f32.
            s.run_dtype = dtype;
            s.widen_off = RUN_SENTINEL;
        }
        Ok(demoted)
    }

    /// Widen the bf16 vectors that kernels read elementwise into `widen`, and record where
    /// they landed. Call once per (plan, run). A wide tensor whose `run_dtype` is already
    /// F32 stays in the run (`widen_off = RUN_SENTINEL`); a wide tensor whose `run_dtype`
    /// is BF16 is widened into the buffer. k3_bind.c:320 (layer_mem).
    pub fn widen(&mut self, run: &[u8], widen: &mut [u8]) -> Result<usize> {
        let mut w: usize = 0;
        for s in &mut self.slots {
            if s.narrow {
                continue; // narrow tensors stay in the run; views points at them.
            }
            match s.run_dtype {
                Dtype::F32 => {
                    // Already f32 in the run; point at it. A prefix take (A_log) is the
                    // front of the same array, so that is free too. k3_bind.c:408.
                    s.widen_off = RUN_SENTINEL;
                }
                Dtype::Bf16 => {
                    w = align8(w as i64) as usize;
                    let need = s.take as usize * 4;
                    if w + need > widen.len() {
                        return Err(Error::new(
                            ErrorKind::InvalidData,
                            format!(
                                "k3_bind: widen area too small at {} ({} of {})",
                                s.name,
                                w + need,
                                widen.len()
                            ),
                        ));
                    }
                    let src_off = s.off as usize;
                    let sp = &run[src_off..src_off + s.take as usize * 2];
                    let dst = &mut widen[w..w + need];
                    for k in 0..s.take as usize {
                        let h = u16::from_le_bytes([sp[k * 2], sp[k * 2 + 1]]);
                        dst[k * 4..k * 4 + 4].copy_from_slice(&f32::to_le_bytes(bf16f(h)));
                    }
                    s.widen_off = w as i64;
                    w += need;
                }
                Dtype::I8R => {
                    // Per-row int8 draft weight, wanted as fp32: dequantise into widen.
                    // take is the logical element count (rows*cols); each row is
                    // [4 bytes scale][cols int8], nb = rows*(4+cols) with rows*cols == take.
                    // k3_bind.c:354.
                    w = align8(w as i64) as usize;
                    let take = s.take;
                    let nb = s.nbytes;
                    if (nb - take) % 4 != 0 {
                        return Err(Error::new(
                            ErrorKind::InvalidData,
                            format!("k3_bind_mem: {} bad int8 layout", s.name),
                        ));
                    }
                    let rows = (nb - take) / 4;
                    if rows <= 0 || take % rows != 0 {
                        return Err(Error::new(
                            ErrorKind::InvalidData,
                            format!("k3_bind_mem: {} bad int8 shape", s.name),
                        ));
                    }
                    let cols = take / rows;
                    let need = s.take as usize * 4;
                    if w + need > widen.len() {
                        return Err(Error::new(
                            ErrorKind::InvalidData,
                            format!("k3_bind_mem: widen area too small at {}", s.name),
                        ));
                    }
                    let dst = &mut widen[w..w + need];
                    let rp = &run[s.off as usize..];
                    let rowb = 4 + cols as usize;
                    for r in 0..rows as usize {
                        let scale = f32::from_le_bytes([
                            rp[r * rowb],
                            rp[r * rowb + 1],
                            rp[r * rowb + 2],
                            rp[r * rowb + 3],
                        ]);
                        for k in 0..cols as usize {
                            let q8 = rp[r * rowb + 4 + k] as i8 as f32;
                            dst[(r * cols as usize + k) * 4..(r * cols as usize + k) * 4 + 4]
                                .copy_from_slice(&f32::to_le_bytes(q8 * scale));
                        }
                    }
                    s.widen_off = w as i64;
                    w += need;
                }
                _ => {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        format!(
                            "k3_bind: {} has run_dtype {:?}, cannot widen",
                            s.name, s.run_dtype
                        ),
                    ));
                }
            }
        }
        Ok(w)
    }

    /// Materialise a borrow-checked `LayerW` over a run buffer and a widen buffer. The
    /// run holds raw checkpoint bytes (bf16 for narrow, f32 or bf16 for wide); the widen
    /// buffer holds the widened f32 vectors that `widen` produced.
    pub fn views<'r>(&self, _c: &Cfg, run: &'r [u8], widen: &'r [u8]) -> LayerW<'r> {
        let in_norm = self.f32_view(Role::InNorm, run, widen);
        let post_norm = self.f32_view(Role::PostNorm, run, widen);
        let attn_res_norm = self.f32_view(Role::AttnResNorm, run, widen);
        let attn_res_proj = self.f32_view(Role::AttnResProj, run, widen);
        let mlp_res_norm = self.f32_view(Role::MlpResNorm, run, widen);
        let mlp_res_proj = self.f32_view(Role::MlpResProj, run, widen);

        let attn = if self.is_mla {
            Attn::Mla(MlaW {
                q_a: self.wmat_view(Role::MlaQA, run, widen),
                q_b: self.wmat_view(Role::MlaQB, run, widen),
                kv_a: self.wmat_view(Role::MlaKvA, run, widen),
                kv_b: self.wmat_view(Role::MlaKvB, run, widen),
                o: self.wmat_view(Role::MlaO, run, widen),
                g: self.find(Role::MlaG).map(|s| self.wmat_from(s, run, widen)),
                q_a_norm: self.f32_view(Role::MlaQaNorm, run, widen),
                kv_a_norm: self.f32_view(Role::MlaKvANorm, run, widen),
            })
        } else {
            Attn::Kda(KdaW {
                q: self.wmat_view(Role::KdaQ, run, widen),
                k: self.wmat_view(Role::KdaK, run, widen),
                v: self.wmat_view(Role::KdaV, run, widen),
                q_conv: self.f32_view(Role::KdaQConv, run, widen),
                k_conv: self.f32_view(Role::KdaKConv, run, widen),
                v_conv: self.f32_view(Role::KdaVConv, run, widen),
                f_a: self.wmat_view(Role::KdaFA, run, widen),
                f_b: self.wmat_view(Role::KdaFB, run, widen),
                a_log: self.f32_view(Role::KdaALog, run, widen),
                dt_bias: self.f32_view(Role::KdaDtBias, run, widen),
                b: self.wmat_view(Role::KdaB, run, widen),
                g: self.wmat_view(Role::KdaG, run, widen),
                o_norm: self.f32_view(Role::KdaONorm, run, widen),
                o: self.wmat_view(Role::KdaO, run, widen),
            })
        };

        let moe = if self.is_dense {
            None
        } else {
            Some(MoeW {
                gate: self.f32_view(Role::MoeGate, run, widen),
                bias: self
                    .find(Role::MoeBias)
                    .map(|_| self.f32_view(Role::MoeBias, run, widen)),
                w1: None,
                w3: None,
                w2: None,
                latent_norm: self
                    .find(Role::MoeLatentNorm)
                    .map(|_| self.f32_view(Role::MoeLatentNorm, run, widen)),
                down: self.wmat_view(Role::MoeDown, run, widen),
                up: self.wmat_view(Role::MoeUp, run, widen),
                sh1: self.wmat_view(Role::MoeSh1, run, widen),
                sh3: self.wmat_view(Role::MoeSh3, run, widen),
                sh2: self.wmat_view(Role::MoeSh2, run, widen),
                layer: self.layer,
                cache_only: false,
            })
        };

        let (dense_gate, dense_up, dense_down) = if self.is_dense {
            (
                Some(self.wmat_view(Role::DenseGate, run, widen)),
                Some(self.wmat_view(Role::DenseUp, run, widen)),
                Some(self.wmat_view(Role::DenseDown, run, widen)),
            )
        } else {
            (None, None, None)
        };

        LayerW {
            in_norm,
            post_norm,
            attn_res_norm,
            attn_res_proj,
            mlp_res_norm,
            mlp_res_proj,
            attn,
            moe,
            dense_gate,
            dense_up,
            dense_down,
        }
    }

    /// Generate the request list for one layer. `narrow_ok` is the C `Plan.narrow_ok`:
    /// false forces every tensor to fp32 (the demoted fallback). k3_bind.c:163.
    fn plan_layer(&mut self, c: &Cfg, narrow_ok: bool) {
        let h = c.hidden as i64;
        let p = (c.kda_heads as i64) * (c.kda_head_dim as i64); // 12288

        // Norms and the attn-res projections are folded ELEMENTWISE, never through a
        // matmul, so they must stay fp32. k3_bind.c:170.
        self.reqw(
            Role::InNorm,
            h,
            -1,
            &fmt_name("layers.%d.input_layernorm.weight", self.layer),
        );
        self.reqw(
            Role::PostNorm,
            h,
            -1,
            &fmt_name("layers.%d.post_attention_layernorm.weight", self.layer),
        );
        self.reqw(
            Role::AttnResNorm,
            h,
            -1,
            &fmt_name("layers.%d.self_attention_res_norm.weight", self.layer),
        );
        self.reqw(
            Role::AttnResProj,
            h,
            -1,
            &fmt_name("layers.%d.self_attention_res_proj.weight", self.layer),
        );
        self.reqw(
            Role::MlpResNorm,
            h,
            -1,
            &fmt_name("layers.%d.mlp_res_norm.weight", self.layer),
        );
        self.reqw(
            Role::MlpResProj,
            h,
            -1,
            &fmt_name("layers.%d.mlp_res_proj.weight", self.layer),
        );

        if self.is_mla {
            let qh = (c.qk_nope as i64) + (c.qk_rope as i64); // 192
            self.reqn(
                Role::MlaQA,
                (c.q_lora as i64) * h,
                &fmt_name("layers.%d.self_attn.q_a_proj.weight", self.layer),
                narrow_ok,
            );
            self.reqw(
                Role::MlaQaNorm,
                c.q_lora as i64,
                -1,
                &fmt_name("layers.%d.self_attn.q_a_layernorm.weight", self.layer),
            );
            self.reqn(
                Role::MlaQB,
                (c.n_heads as i64) * qh * (c.q_lora as i64),
                &fmt_name("layers.%d.self_attn.q_b_proj.weight", self.layer),
                narrow_ok,
            );
            self.reqn(
                Role::MlaKvA,
                ((c.kv_lora as i64) + (c.qk_rope as i64)) * h,
                &fmt_name("layers.%d.self_attn.kv_a_proj_with_mqa.weight", self.layer),
                narrow_ok,
            );
            self.reqw(
                Role::MlaKvANorm,
                c.kv_lora as i64,
                -1,
                &fmt_name("layers.%d.self_attn.kv_a_layernorm.weight", self.layer),
            );
            self.reqn(
                Role::MlaKvB,
                (c.n_heads as i64) * ((c.qk_nope as i64) + (c.v_head as i64)) * (c.kv_lora as i64),
                &fmt_name("layers.%d.self_attn.kv_b_proj.weight", self.layer),
                narrow_ok,
            );
            self.reqn(
                Role::MlaO,
                h * (c.n_heads as i64) * (c.v_head as i64),
                &fmt_name("layers.%d.self_attn.o_proj.weight", self.layer),
                narrow_ok,
            );
            if c.mla_out_gate != 0 {
                self.reqn(
                    Role::MlaG,
                    (c.n_heads as i64) * (c.v_head as i64) * h,
                    &fmt_name("layers.%d.self_attn.g_proj.weight", self.layer),
                    narrow_ok,
                );
            }
        } else {
            self.reqn(
                Role::KdaQ,
                p * h,
                &fmt_name("layers.%d.self_attn.q_proj.weight", self.layer),
                narrow_ok,
            );
            self.reqn(
                Role::KdaK,
                p * h,
                &fmt_name("layers.%d.self_attn.k_proj.weight", self.layer),
                narrow_ok,
            );
            self.reqn(
                Role::KdaV,
                p * h,
                &fmt_name("layers.%d.self_attn.v_proj.weight", self.layer),
                narrow_ok,
            );
            self.reqn(
                Role::KdaG,
                p * h,
                &fmt_name("layers.%d.self_attn.g_proj.weight", self.layer),
                narrow_ok,
            );
            self.reqn(
                Role::KdaO,
                h * p,
                &fmt_name("layers.%d.self_attn.o_proj.weight", self.layer),
                narrow_ok,
            );
            // Rank 3 on disk, [H*D][1][conv_k]; the element count is what matters. Read
            // elementwise by shortconv, so fp32. k3_bind.c:201.
            self.reqw(
                Role::KdaQConv,
                p * (c.conv_k as i64),
                -1,
                &fmt_name("layers.%d.self_attn.q_conv1d.weight", self.layer),
            );
            self.reqw(
                Role::KdaKConv,
                p * (c.conv_k as i64),
                -1,
                &fmt_name("layers.%d.self_attn.k_conv1d.weight", self.layer),
            );
            self.reqw(
                Role::KdaVConv,
                p * (c.conv_k as i64),
                -1,
                &fmt_name("layers.%d.self_attn.v_conv1d.weight", self.layer),
            );
            self.reqn(
                Role::KdaFA,
                (c.kda_head_dim as i64) * h,
                &fmt_name("layers.%d.self_attn.f_a_proj.weight", self.layer),
                narrow_ok,
            );
            self.reqn(
                Role::KdaFB,
                p * (c.kda_head_dim as i64),
                &fmt_name("layers.%d.self_attn.f_b_proj.weight", self.layer),
                narrow_ok,
            );
            self.reqn(
                Role::KdaB,
                (c.kda_heads as i64) * h,
                &fmt_name("layers.%d.self_attn.b_proj.weight", self.layer),
                narrow_ok,
            );
            // PER HEAD. The checkpoint ships kda_head_dim values and zeroes the tail; take
            // the first kda_heads. Read elementwise by kda_decay, so fp32. k3_bind.c:211.
            self.reqw(
                Role::KdaALog,
                c.kda_head_dim as i64,
                c.kda_heads as i64,
                &fmt_name("layers.%d.self_attn.A_log", self.layer),
            );
            self.reqw(
                Role::KdaDtBias,
                p,
                -1,
                &fmt_name("layers.%d.self_attn.dt_bias", self.layer),
            );
            self.reqw(
                Role::KdaONorm,
                c.kda_head_dim as i64,
                -1,
                &fmt_name("layers.%d.self_attn.o_norm.weight", self.layer),
            );
        }

        if self.is_dense {
            self.reqn(
                Role::DenseGate,
                (c.dense_inter as i64) * h,
                &fmt_name("layers.%d.mlp.gate_proj.weight", self.layer),
                narrow_ok,
            );
            self.reqn(
                Role::DenseUp,
                (c.dense_inter as i64) * h,
                &fmt_name("layers.%d.mlp.up_proj.weight", self.layer),
                narrow_ok,
            );
            self.reqn(
                Role::DenseDown,
                h * (c.dense_inter as i64),
                &fmt_name("layers.%d.mlp.down_proj.weight", self.layer),
                narrow_ok,
            );
        } else {
            let si = (c.moe_inter as i64) * (c.n_shared as i64); // fused: 6144
                                                                 // gate stays fp32: router has its own inline matmul. k3_bind.c:223.
            self.reqw(
                Role::MoeGate,
                (c.n_experts as i64) * h,
                -1,
                &fmt_name("layers.%d.block_sparse_moe.gate.weight", self.layer),
            );
            self.reqw(
                Role::MoeBias,
                c.n_experts as i64,
                -1,
                &fmt_name(
                    "layers.%d.block_sparse_moe.gate.e_score_correction_bias",
                    self.layer,
                ),
            );
            self.reqn(
                Role::MoeDown,
                (c.latent as i64) * h,
                &fmt_name(
                    "layers.%d.block_sparse_moe.routed_expert_down_proj.weight",
                    self.layer,
                ),
                narrow_ok,
            );
            self.reqn(
                Role::MoeUp,
                h * (c.latent as i64),
                &fmt_name(
                    "layers.%d.block_sparse_moe.routed_expert_up_proj.weight",
                    self.layer,
                ),
                narrow_ok,
            );
            self.reqw(
                Role::MoeLatentNorm,
                c.latent as i64,
                -1,
                &fmt_name(
                    "layers.%d.block_sparse_moe.routed_expert_norm.weight",
                    self.layer,
                ),
            );
            self.reqn(
                Role::MoeSh1,
                si * h,
                &fmt_name(
                    "layers.%d.block_sparse_moe.shared_experts.gate_proj.weight",
                    self.layer,
                ),
                narrow_ok,
            );
            self.reqn(
                Role::MoeSh3,
                si * h,
                &fmt_name(
                    "layers.%d.block_sparse_moe.shared_experts.up_proj.weight",
                    self.layer,
                ),
                narrow_ok,
            );
            self.reqn(
                Role::MoeSh2,
                h * si,
                &fmt_name(
                    "layers.%d.block_sparse_moe.shared_experts.down_proj.weight",
                    self.layer,
                ),
                narrow_ok,
            );
        }
    }

    /// WIDE: widened to fp32. k3_bind.c:52.
    fn reqw(&mut self, role: Role, want: i64, take: i64, name: &str) {
        self.slots.push(Slot {
            name: name.to_string(),
            role,
            want,
            take: if take < 0 { want } else { take },
            narrow: false,
            off: 0,
            nbytes: 0,
            dtype: Dtype::Unknown,
            run_dtype: Dtype::Unknown,
            widen_off: RUN_SENTINEL,
        });
    }

    /// NARROW: kept as the checkpoint's bf16, unless `narrow_ok` is false (demoted
    /// fallback). k3_bind.c:61.
    fn reqn(&mut self, role: Role, want: i64, name: &str, narrow_ok: bool) {
        self.slots.push(Slot {
            name: name.to_string(),
            role,
            want,
            take: want, // reqn always takes the whole tensor.
            narrow: narrow_ok,
            off: 0,
            nbytes: 0,
            dtype: Dtype::Unknown,
            run_dtype: Dtype::Unknown,
            widen_off: RUN_SENTINEL,
        });
    }

    /// Find the first slot with a given role.
    fn find(&self, role: Role) -> Option<&Slot> {
        self.slots.iter().find(|s| s.role == role)
    }

    /// Borrow a wide (fp32) vector from the run or the widen buffer.
    fn f32_view<'r>(&self, role: Role, run: &'r [u8], widen: &'r [u8]) -> &'r [f32] {
        let s = self
            .find(role)
            .unwrap_or_else(|| panic!("k3_bind: role {:?} not in plan", role));
        let len = s.take as usize;
        if s.widen_off == RUN_SENTINEL {
            // Lives in the run as f32.
            unsafe { f32_slice(run, s.off as usize, len) }
        } else {
            unsafe { f32_slice(widen, s.widen_off as usize, len) }
        }
    }

    /// Borrow a matrix from the run as a `WMat`. For a narrow slot, the bytes are the
    /// checkpoint's own (bf16 or i8r). For a wide (demoted) slot, the bytes are f32 in the
    /// widen buffer or the run.
    fn wmat_view<'r>(&self, role: Role, run: &'r [u8], widen: &'r [u8]) -> WMat<'r> {
        let s = self
            .find(role)
            .unwrap_or_else(|| panic!("k3_bind: role {:?} not in plan", role));
        self.wmat_from(s, run, widen)
    }

    fn wmat_from<'r>(&self, s: &Slot, run: &'r [u8], widen: &'r [u8]) -> WMat<'r> {
        if s.narrow {
            // The checkpoint's own bytes, in the run.
            let off = s.off as usize;
            let bytes = s.nbytes as usize;
            let raw = &run[off..off + bytes];
            match s.run_dtype {
                Dtype::Bf16 => {
                    let n = bytes / 2;
                    let ptr = raw.as_ptr() as *const u16;
                    // SAFETY: the run outlives 'r; the slice is 2-byte aligned (the blob is
                    // hugepage-aligned and every tensor is 8-byte aligned within it).
                    let u16s = unsafe { std::slice::from_raw_parts(ptr, n) };
                    WMat::Bf16(u16s)
                }
                Dtype::F32 => {
                    let n = bytes / 4;
                    let ptr = raw.as_ptr() as *const f32;
                    // SAFETY: 4-byte aligned.
                    let f32s = unsafe { std::slice::from_raw_parts(ptr, n) };
                    WMat::F32(f32s)
                }
                Dtype::I8R => WMat::I8R(raw),
                _ => panic!(
                    "k3_bind: {} has unsupported narrow run_dtype {:?}",
                    s.name, s.run_dtype
                ),
            }
        } else {
            // Demoted: the matrix was widened to f32. It is in the widen buffer or the run.
            let numel = s.take as usize;
            if s.widen_off == RUN_SENTINEL {
                unsafe { WMat::F32(f32_slice(run, s.off as usize, numel)) }
            } else {
                unsafe { WMat::F32(f32_slice(widen, s.widen_off as usize, numel)) }
            }
        }
    }

    /// Read every tensor from the shards into `blob`, laying the blob out with the small
    /// vectors at fp32 (via `read_f32`) and the matrices at their checkpoint dtype. Sets
    /// `off` to the blob offset and `run_dtype` to what is actually stored, so the shared
    /// `widen`/`views` see the data correctly. k3_bind.c:249 (bind_layer).
    fn read_into_blob(&mut self, st: &St, blob: &mut [u8]) -> Result<()> {
        // Lay out the blob: narrow at the checkpoint's elemsize, wide at 4 (f32).
        let mut off: i64 = 0;
        let mut layout: Vec<(i64, Dtype)> = Vec::with_capacity(self.slots.len());
        for s in &self.slots {
            off = align8(off);
            let esz = if s.narrow {
                s.dtype.elemsize() as i64
            } else {
                4 // wide is stored as f32
            };
            let slot_bytes = s.nbytes / s.dtype.elemsize() as i64 * esz;
            layout.push((off, if s.narrow { s.dtype } else { Dtype::F32 }));
            off += slot_bytes;
        }

        for (i, s) in self.slots.iter_mut().enumerate() {
            let (blob_off, run_dtype) = layout[i];
            let t = st.find(&s.name).ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidData,
                    format!("k3_bind: missing tensor {}", s.name),
                )
            })?;
            let have = t.numel();
            if s.narrow {
                // Raw bytes, straight from the shard. k3_bind.c:138.
                let dst = &mut blob[blob_off as usize..blob_off as usize + t.nbytes as usize];
                let got = st.read(t, dst)?;
                if got != t.nbytes {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        format!("k3_bind: short read of {}", s.name),
                    ));
                }
            } else {
                // Widened to f32 on read. k3_bind.c:143.
                let n_f32 = have as usize;
                let dst_bytes = &mut blob[blob_off as usize..blob_off as usize + n_f32 * 4];
                // SAFETY: dst_bytes is 8-byte aligned within the hugepage-aligned blob, so
                // 4-byte alignment for f32 is guaranteed; the length is n_f32 * 4.
                let dst_f32 = unsafe {
                    std::slice::from_raw_parts_mut(dst_bytes.as_mut_ptr() as *mut f32, n_f32)
                };
                let got = st.read_f32(t, dst_f32)?;
                if got != have {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        format!("k3_bind: short read of {}", s.name),
                    ));
                }
            }
            s.off = blob_off;
            s.run_dtype = run_dtype;
        }
        Ok(())
    }

    /// Total bytes the owned blob must hold (for `from_shards`). Uses the per-slot layout
    /// above: narrow at the checkpoint's elemsize, wide at 4.
    fn blob_size(&self) -> usize {
        let mut off: i64 = 0;
        for s in &self.slots {
            off = align8(off);
            let esz = if s.narrow {
                s.dtype.elemsize() as i64
            } else {
                4
            };
            off += s.nbytes / s.dtype.elemsize() as i64 * esz;
        }
        off as usize
    }
}

/// One decoder layer's weights, read from the shards into one owned blob plus a widen
/// buffer. `views` materialises the borrow-checked `LayerW` on demand.
pub struct LayerBind {
    plan: LayerPlan,
    run: AlignedBuf,
    widen: AlignedBuf,
    cfg: Cfg,
    layer: usize,
}

impl LayerBind {
    /// Read one decoder layer's tensors from the shards into an owned blob, then widen
    /// the bf16 vectors. Routed expert pointers (`w1`/`w2`/`w3`) are left `None`: those
    /// are streamed per token by `load`, not resident. k3_bind.c:249.
    pub fn from_shards(s: &St, c: &Cfg, layer: usize) -> Result<LayerBind> {
        let find = |name: &str| -> Option<(i64, i64, Dtype)> {
            s.find(name).map(|t| (t.off, t.nbytes, t.dtype))
        };
        let mut plan = LayerPlan::build(c, layer, &find)?;

        let blob_size = plan.blob_size();
        let mut run = AlignedBuf::new(blob_size)?;
        let widen_cap = widen_bytes(c);
        let mut widen = AlignedBuf::new(widen_cap)?;

        // Read every tensor's raw bytes into the blob, then point the plan at the blob.
        // For the shard path the small vectors are read as f32 via read_f32, so the blob
        // already holds f32 for them and `widen` is a no-op for those slots.
        plan.read_into_blob(s, &mut run)?;
        let _widen_used = plan.widen(&run, &mut widen)?;
        // The widen buffer is sized once by widen_bytes; usage is bounded by it.

        Ok(LayerBind {
            plan,
            run,
            widen,
            cfg: c.clone(),
            layer,
        })
    }

    /// Materialise the borrow-checked `LayerW` over the owned blob and widen buffer.
    pub fn views(&self) -> LayerW<'_> {
        self.plan.views(&self.cfg, &self.run, &self.widen)
    }

    pub fn nbytes(&self) -> usize {
        self.run.len() + self.widen.len()
    }

    pub fn layer(&self) -> usize {
        self.layer
    }
}

// ------------------------------------------------------------------ model level ----

/// Model-level weights: embedding, final norm, lm_head, and the one model-level
/// attn-res aggregator. `embed` and `lm_head` are 2.35 GB each at bf16, so pass
/// `want_lm_head = false` when only the trunk is being exercised.
pub struct ModelBind {
    blob: AlignedBuf,
    embed_off: i64,
    embed_dtype: Dtype,
    norm_off: i64,
    out_res_norm_off: i64,
    out_res_proj_off: i64,
    lm_head: Option<(i64, Dtype)>,
    hidden: i32,
    vocab: i32,
}

impl ModelBind {
    /// Load the model-level weights from the shards. embed and lm_head stay bf16 (they
    /// are 2.35 GB each; widening to fp32 would double that); the norms are widened to
    /// fp32 because they are read elementwise. k3_bind.c:461.
    pub fn load(s: &St, c: &Cfg, want_lm_head: bool) -> Result<ModelBind> {
        // First pass: resolve, validate, handle the demoted fallback.
        let (mut slots, mut narrow_ok) = plan_model(s, c, want_lm_head, true)?;
        let demoted = slots.iter().any(|q| q.narrow && q.dtype != Dtype::Bf16);
        if demoted {
            eprintln!(
                "k3_bind: model-level tensor(s) are not BF16; binding the model-level weights at fp32 instead"
            );
            let (s2, n2) = plan_model(s, c, want_lm_head, false)?;
            slots = s2;
            narrow_ok = n2;
        }

        // Size the blob: narrow at 2 bytes, wide at 4. k3_bind.c:124.
        let mut off: i64 = 0;
        let mut layout: Vec<i64> = Vec::with_capacity(slots.len());
        for q in &slots {
            off = align8(off);
            layout.push(off);
            off += q.take * (if q.narrow { 2 } else { 4 });
        }
        let need = off as usize;
        let mut blob = AlignedBuf::new(need)?;

        // Read each tensor into the blob.
        for (i, q) in slots.iter().enumerate() {
            let blob_off = layout[i];
            let t = s.find(&q.name).ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidData,
                    format!("k3_bind: missing tensor {}", q.name),
                )
            })?;
            let have = t.numel();
            let dst_off = blob_off as usize;
            if q.narrow {
                let dst = &mut blob[dst_off..dst_off + t.nbytes as usize];
                let got = s.read(t, dst)?;
                if got != t.nbytes {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        format!("k3_bind: short read of {}", q.name),
                    ));
                }
            } else if q.take == have {
                let n_f32 = have as usize;
                let dst_bytes = &mut blob[dst_off..dst_off + n_f32 * 4];
                let dst_f32 = unsafe {
                    std::slice::from_raw_parts_mut(dst_bytes.as_mut_ptr() as *mut f32, n_f32)
                };
                let got = s.read_f32(t, dst_f32)?;
                if got != have {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        format!("k3_bind: short read of {}", q.name),
                    ));
                }
            } else {
                // A prefix: read the whole tensor into scratch, keep the front. Only
                // A_log needs this, and it is 128 floats. k3_bind.c:150.
                let mut tmp = vec![0f32; have as usize];
                let got = s.read_f32(t, &mut tmp)?;
                if got != have {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        format!("k3_bind: short read of {}", q.name),
                    ));
                }
                let dst = &mut blob[dst_off..dst_off + q.take as usize * 4];
                for k in 0..q.take as usize {
                    dst[k * 4..k * 4 + 4].copy_from_slice(&f32::to_le_bytes(tmp[k]));
                }
            }
        }

        let embed_off = layout[0];
        let embed_dtype = if narrow_ok {
            slots[0].dtype
        } else {
            Dtype::F32
        };
        let norm_off = layout[1];
        let out_res_norm_off = layout[2];
        let out_res_proj_off = layout[3];
        let lm_head = if want_lm_head {
            Some((
                layout[4],
                if narrow_ok {
                    slots[4].dtype
                } else {
                    Dtype::F32
                },
            ))
        } else {
            None
        };

        Ok(ModelBind {
            blob,
            embed_off,
            embed_dtype,
            norm_off,
            out_res_norm_off,
            out_res_proj_off,
            lm_head,
            hidden: c.hidden,
            vocab: c.vocab,
        })
    }

    /// Bytes this binding holds, which is what the CLI banner reports. k3_bind.h:76.
    pub fn nbytes(&self) -> usize {
        self.blob.len()
    }

    /// The embedding table. bf16 when the checkpoint is bf16, f32 after the demoted
    /// fallback. k3_bind.c:453.
    pub fn embed(&self) -> WMat<'_> {
        self.wmat_at(
            self.embed_off,
            self.embed_dtype,
            (self.vocab as i64) * (self.hidden as i64),
        )
    }

    /// `lm_head`, when it was loaded. k3_bind.c:458.
    pub fn lm_head(&self) -> Option<WMat<'_>> {
        self.lm_head
            .map(|(off, dt)| self.wmat_at(off, dt, (self.vocab as i64) * (self.hidden as i64)))
    }

    /// Final RMSNorm gain, `[hidden]`, fp32. k3_bind.c:454.
    pub fn norm(&self) -> &[f32] {
        unsafe { f32_slice(&self.blob, self.norm_off as usize, self.hidden as usize) }
    }

    /// Model-level attn-res norm, `[hidden]`, fp32. k3_bind.c:455.
    pub fn out_res_norm(&self) -> &[f32] {
        unsafe {
            f32_slice(
                &self.blob,
                self.out_res_norm_off as usize,
                self.hidden as usize,
            )
        }
    }

    /// Model-level attn-res projection, `[hidden]`, fp32. k3_bind.c:456.
    pub fn out_res_proj(&self) -> &[f32] {
        unsafe {
            f32_slice(
                &self.blob,
                self.out_res_proj_off as usize,
                self.hidden as usize,
            )
        }
    }

    /// Gather one embedding row into `dst[hidden]`, widening if the table is bf16. The
    /// table is INDEXED rather than multiplied, so it cannot go through `mmw`: a plain
    /// memcpy with a float stride would read half a row of the wrong values. k3_bind.h:115.
    pub fn embed_row(&self, dst: &mut [f32], row: i64) {
        let hidden = self.hidden as usize;
        match self.embed_dtype {
            Dtype::Bf16 => {
                let off = self.embed_off as usize + (row as usize) * hidden * 2;
                let p = &self.blob[off..off + hidden * 2];
                for i in 0..hidden {
                    let h = u16::from_le_bytes([p[i * 2], p[i * 2 + 1]]);
                    dst[i] = bf16f(h);
                }
            }
            _ => {
                let off = self.embed_off as usize + (row as usize) * hidden * 4;
                let f32s = unsafe {
                    std::slice::from_raw_parts(self.blob.as_ptr().add(off) as *const f32, hidden)
                };
                dst.copy_from_slice(f32s);
            }
        }
    }

    fn wmat_at(&self, off: i64, dtype: Dtype, numel: i64) -> WMat<'_> {
        let off = off as usize;
        let n = numel as usize;
        let raw = &self.blob[off..off + n * dtype.elemsize()];
        match dtype {
            Dtype::Bf16 => {
                let ptr = raw.as_ptr() as *const u16;
                unsafe { WMat::Bf16(std::slice::from_raw_parts(ptr, n)) }
            }
            Dtype::F32 => {
                let ptr = raw.as_ptr() as *const f32;
                unsafe { WMat::F32(std::slice::from_raw_parts(ptr, n)) }
            }
            Dtype::I8R => WMat::I8R(raw),
            _ => panic!("k3_bind: unsupported model-level dtype {:?}", dtype),
        }
    }
}

/// BYTES one layer needs, without loading it. Matches C `k3_bind_layer_bytes`: the
/// logical resident-weight size, narrow at 2 bytes and wide at 4. k3_bind.c:239.
pub fn layer_bytes(s: &St, c: &Cfg, layer: usize) -> Result<i64> {
    let find = |name: &str| -> Option<(i64, i64, Dtype)> {
        s.find(name).map(|t| (t.off, t.nbytes, t.dtype))
    };
    let plan = LayerPlan::build(c, layer, &find)?;
    let mut total: i64 = 0;
    for slot in &plan.slots {
        total = align8(total);
        total += slot.take * (if slot.narrow { 2 } else { 4 });
    }
    Ok(total)
}

/// Upper bound on the widen area one layer needs, so a trunk slot can be sized once.
/// Only the bf16 vectors that kernels read elementwise are copied; everything else is
/// pointed at in place. The router gate dominates. k3_bind.c:307.
pub fn widen_bytes(c: &Cfg) -> usize {
    let h = c.hidden as usize;
    let n = 6 * h // in/post norm, attn-res and mlp-res pair
        + (c.q_lora as usize + c.kv_lora as usize) // MLA q_a/kv_a layernorms
        + c.latent as usize // routed_expert_norm
        + (c.n_experts as usize) * h; // router gate
    n * 4 + 4096 // slack for per-tensor 8-byte alignment
}

// -------------------------------------------------------------------- helpers ----

/// Align to 8 bytes. k3_bind.c:68.
fn align8(x: i64) -> i64 {
    (x + 7) & !7
}

/// bf16 -> f32 is a pure left shift by 16 bits: no rounding, no table. k3.h:274.
fn bf16f(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// Format a C-style `%d` layer name, fully qualified.
///
/// C spells every layer template as `PRE "layers.%d..."`, concatenating the prefix into
/// the literal at each of its 39 call sites (k3_bind.c:170). Prefixing here keeps the 39
/// Rust templates readable while producing byte-identical names; dropping it made the
/// binder look up `layers.0.input_layernorm.weight`, which no released checkpoint holds.
fn fmt_name(template: &str, layer: usize) -> String {
    format!("{PRE}{}", template.replace("%d", &layer.to_string()))
}

/// Reinterpret a byte slice as f32. SAFETY: `off` is 4-byte aligned within the buffer
/// (every tensor is 8-byte aligned) and `len * 4` bytes are available.
unsafe fn f32_slice(buf: &[u8], off: usize, len: usize) -> &[f32] {
    let ptr = buf.as_ptr().add(off) as *const f32;
    std::slice::from_raw_parts(ptr, len)
}

/// One model-level request, parallel to `Slot` but simpler (no run/widen split). The
/// expected element count is checked in `plan_model` before the request is built, so it
/// is not carried here.
struct ModelReq {
    name: String,
    take: i64,
    narrow: bool,
    dtype: Dtype,
}

/// The C `plan_model` + `plan_resolve` for model-level weights. Returns the requests and
/// whether narrow storage is allowed. k3_bind.c:450.
fn plan_model(
    s: &St,
    c: &Cfg,
    want_lm_head: bool,
    narrow_ok: bool,
) -> Result<(Vec<ModelReq>, bool)> {
    let h = c.hidden as i64;
    // Order matters: embed, norm, out_res_norm, out_res_proj, [lm_head]. k3_bind.c:453.
    // embed is gathered a row at a time rather than multiplied, so k3_run widens the row
    // it needs. lm_head goes through mmw. Both are 2.35 GB as bf16 and 4.70 GB widened,
    // which is why neither is widened here. k3_bind.c:447.
    let mut reqs: Vec<(String, i64, i64, bool)> = Vec::new();
    reqs.push((
        format!("{}embed_tokens.weight", PRE),
        (c.vocab as i64) * h,
        -1,
        true,
    ));
    reqs.push((format!("{}norm.weight", PRE), h, -1, false));
    reqs.push((format!("{}output_attn_res_norm.weight", PRE), h, -1, false));
    reqs.push((format!("{}output_attn_res_proj.weight", PRE), h, -1, false));
    if want_lm_head {
        reqs.push((
            "language_model.lm_head.weight".to_string(),
            (c.vocab as i64) * h,
            -1,
            true,
        ));
    }

    let mut out = Vec::new();
    for (name, want, take, narrow_pref) in reqs {
        let t = s.find(&name).ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidData,
                format!("k3_bind: missing tensor {}", name),
            )
        })?;
        let have = t.numel();
        let take = if take < 0 { want } else { take };
        if want >= 0 && have != want {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "k3_bind: {} has {} elements, engine expects {}",
                    name, have, want
                ),
            ));
        }
        if take > have {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("k3_bind: {}: asked for {} of {} elements", name, take, have),
            ));
        }
        let narrow = narrow_pref && narrow_ok;
        out.push(ModelReq {
            name: name.clone(),
            take,
            narrow,
            dtype: t.dtype,
        });
    }
    Ok((out, narrow_ok))
}
