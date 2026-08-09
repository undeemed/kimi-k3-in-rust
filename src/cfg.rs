// SPDX-License-Identifier: Apache-2.0
//! Model configuration and the layer map. Port of `include/k3/k3_cfg.h` and the
//! `K3Cfg` struct in `include/k3/k3.h`.

/// Upper bound on `num_experts_per_token`. K3 selects 16. A HARD limit: `router` fills
/// the caller's `idx` with `topk` entries. k3.h:557.
pub const MAX_TOPK: usize = 64;

/// `quantization_config.group_size`. Named rather than spelled 32 at each call site so a
/// checkpoint that changed it fails in one place. k3.h:548.
pub const MXFP4_GROUP: usize = 32;

/// Context ceilings: a sanity bound, not the binding constraint. k3.h:411.
pub const MAX_PROMPT: usize = 32768;
pub const MAX_GEN: usize = 4096;

/// Bytes of MLA KV cache per position, measured on the released checkpoint. k3.h:415.
pub const KV_BYTES_PER_POS: f64 = 2_370_000.0;

/// Every value verified against the released config.json. See k3.h:73.
#[derive(Clone, Debug, PartialEq)]
pub struct Cfg {
    pub hidden: i32,
    pub n_layers: i32,
    pub vocab: i32,
    pub rms_eps: f32,

    // Kimi Delta Attention. 69 of the 93 layers.
    pub kda_heads: i32,
    pub kda_head_dim: i32,
    pub conv_k: i32,
    pub gate_lb: f32,

    // Gated MLA. 24 of the 93 layers.
    pub n_heads: i32,
    pub q_lora: i32,
    pub kv_lora: i32,
    pub qk_nope: i32,
    /// 64, PRESENT BUT NEVER ROTATED.
    pub qk_rope: i32,
    pub v_head: i32,
    pub mla_out_gate: i32,

    // Stable LatentMoE. 92 of the 93 layers.
    pub n_experts: i32,
    pub topk: i32,
    /// 2, full width, added UNWEIGHTED.
    pub n_shared: i32,
    pub latent: i32,
    pub moe_inter: i32,
    pub routed_scale: f32,
    pub moe_renorm: i32,
    /// 1, RMSNorm on the AGGREGATE, not per expert.
    pub latent_norm: i32,

    // the single dense layer, layer 0
    pub first_dense: i32,
    pub dense_inter: i32,

    /// 12. Boundaries fire when `layer_idx % this == 0`.
    pub attn_res_block: i32,

    // SiTU-GLU. The sigmoid takes the UNCAPPED gate. Bound is b1*b2 = 100.
    pub situ_b1: f32,
    pub situ_b2: f32,

    /// ONE-BASED layer indices, as the released config lists them. Zero-based MLA layers
    /// are therefore 3, 7, 11, ..., 87, 91, 92, and 91 and 92 are BOTH MLA. k3.h:115.
    pub full_attn: Vec<i32>,
}

impl Cfg {
    /// `layer` is ZERO-based. The released config lists `full_attn_layers` ONE-BASED and
    /// configuration_kimi_k3.py:152-156 tests `(layer_idx + 1)`; getting this off by one
    /// silently swaps KDA and MLA layers throughout the stack. k3_ops.c:81.
    pub fn is_mla(&self, layer: i32) -> bool {
        self.full_attn.iter().any(|&l| l == layer + 1)
    }
    pub fn is_kda(&self, layer: i32) -> bool {
        !self.is_mla(layer)
    }
    pub fn is_dense(&self, layer: i32) -> bool {
        layer < self.first_dense
    }
}
