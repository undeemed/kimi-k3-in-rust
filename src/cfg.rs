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

// ----------------------------------------------------------------- loader ----
// Port of k3_cfg.h's k3_cfg_load / k3_cfg_load_file. Two JSON shapes describe this
// model: the released config.json (nested under text_config, with the KDA/MLA map
// under text_config.linear_attn_config) and the flat ref_k3.json fixture. Every lookup
// tries its primary name then its alias across the candidate objects in order, so a
// nested released config and a flat fixture both load through one path.
//
// THE ONE RULE: an absent field is an error, never a default. k3_cfg.h:18-34 states
// why plainly: defaulting situ_beta/situ_linear_beta to 4.0/25.0 looks correct, and an
// empty full_attn_layers makes every layer run as KDA, producing fluent tokens from the
// wrong architecture. So every missing key is collected and reported together.

/// Candidate objects searched, in order, for every field. k3_cfg.h:46-57.
struct Src<'a> {
    txt: Option<&'a serde_json::Value>,
    lin: Option<&'a serde_json::Value>,
    root: &'a serde_json::Value,
    nested: bool,
    missing: Vec<&'static str>,
}

impl<'a> Src<'a> {
    /// Mirror k3cfg_find: search txt, then lin, then root; within each, try primary then
    /// alias. k3_cfg.h:59-72.
    fn find(
        &self,
        primary: &'static str,
        alias: Option<&'static str>,
    ) -> Option<&'a serde_json::Value> {
        for o in [self.txt, self.lin, Some(self.root)].into_iter().flatten() {
            for n in [Some(primary), alias].into_iter().flatten() {
                if let Some(v) = o.get(n) {
                    return Some(v);
                }
            }
        }
        None
    }

    /// Record a missing required field. k3_cfg.h:74-79.
    fn miss(&mut self, name: &'static str) {
        self.missing.push(name);
    }

    /// k3cfg_i: a required integer. A JSON bool is NOT accepted here (k3cfg_i rejects
    /// non-J_NUM); only k3cfg_b takes bools. k3_cfg.h:81-86.
    fn i(&mut self, primary: &'static str, alias: Option<&'static str>) -> i32 {
        match self.find(primary, alias) {
            Some(v) if v.is_i64() => v.as_i64().unwrap() as i32,
            Some(v) if v.is_u64() => v.as_u64().unwrap() as i32,
            _ => {
                self.miss(primary);
                0
            }
        }
    }

    /// k3cfg_f: a required float. k3_cfg.h:88-93.
    fn f(&mut self, primary: &'static str, alias: Option<&'static str>) -> f32 {
        match self.find(primary, alias) {
            Some(v) if v.is_number() => v.as_f64().unwrap() as f32,
            _ => {
                self.miss(primary);
                0.0
            }
        }
    }

    /// k3cfg_b: an optional boolean with a real default. JSON true/false accepted, and a
    /// numeric 0/1 because hand-edited configs use it. k3_cfg.h:98-105.
    fn b(&mut self, primary: &'static str, alias: Option<&'static str>, dflt: i32) -> i32 {
        match self.find(primary, alias) {
            Some(v) if v.is_boolean() => {
                if v.as_bool().unwrap() {
                    1
                } else {
                    0
                }
            }
            Some(v) => {
                // Accept numeric 0/1. k3_cfg.h:103 takes any J_NUM != 0.0.
                let n = v.as_i64().or_else(|| v.as_u64().map(|x| x as i64));
                match n {
                    Some(x) => {
                        if x != 0 {
                            1
                        } else {
                            0
                        }
                    }
                    None => dflt,
                }
            }
            _ => dflt,
        }
    }
}

impl Cfg {
    /// Load and validate a config.json in either shape. Port of k3_cfg_load_file plus
    /// k3_cfg_load. Returns the fully-checked Cfg, or an io::Error whose message is the
    /// same diagnostic the C loader writes to stderr. k3_cfg.h:114-248, 253-274.
    pub fn load(path: &std::path::Path) -> std::io::Result<Cfg> {
        let bytes = std::fs::read(path).map_err(|e| {
            // k3_cfg_load_file uses perror(path), which writes "<path>: <strerror>".
            std::io::Error::new(e.kind(), format!("{}: {}", path.display(), e))
        })?;
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{}: not valid JSON", path.display()),
            )
        })?;
        // The released config.json is the config object at the root (with text_config
        // nested inside it). The ref_k3.json fixture wraps the config under a "config"
        // member; test_cfg.c:46 unwraps that before calling k3_cfg_load. Do the same
        // here so Cfg::load handles both files without the caller branching.
        let root = match parsed.get("config") {
            Some(c) if c.is_object() => c,
            _ => &parsed,
        };
        Cfg::from_value(root, &path.display().to_string())
    }

    fn from_value(root: &serde_json::Value, whence: &str) -> std::io::Result<Cfg> {
        let txt = root.get("text_config");
        let nested = txt.is_some();
        let base = txt.unwrap_or(root);
        let lin = base.get("linear_attn_config");

        let mut s = Src {
            txt,
            lin,
            root,
            nested,
            missing: Vec::new(),
        };

        let mut c = Cfg {
            hidden: 0,
            n_layers: 0,
            vocab: 0,
            rms_eps: 0.0,
            kda_heads: 0,
            kda_head_dim: 0,
            conv_k: 0,
            gate_lb: 0.0,
            n_heads: 0,
            q_lora: 0,
            kv_lora: 0,
            qk_nope: 0,
            qk_rope: 0,
            v_head: 0,
            mla_out_gate: 0,
            n_experts: 0,
            topk: 0,
            n_shared: 0,
            latent: 0,
            moe_inter: 0,
            routed_scale: 0.0,
            moe_renorm: 0,
            latent_norm: 0,
            first_dense: 0,
            dense_inter: 0,
            attn_res_block: 0,
            situ_b1: 0.0,
            situ_b2: 0.0,
            full_attn: Vec::new(),
        };

        c.hidden = s.i("hidden_size", None);
        c.n_layers = s.i("num_hidden_layers", None);
        c.vocab = s.i("vocab_size", None);
        c.rms_eps = s.f("rms_norm_eps", None);

        // KDA. Released: linear_attn_config.num_heads / .head_dim. Fixture:
        // kda_num_heads / kda_head_dim at top level. Searching lin before root is what
        // disambiguates num_heads, which would otherwise read ambiguously. k3_cfg.h:130.
        c.kda_heads = s.i("num_heads", Some("kda_num_heads"));
        c.kda_head_dim = s.i("head_dim", Some("kda_head_dim"));
        c.conv_k = s.i("short_conv_kernel_size", None);
        c.gate_lb = s.f("gate_lower_bound", None);

        // Gated MLA. mla_use_output_gate is genuinely optional with default 1.
        c.n_heads = s.i("num_attention_heads", None);
        c.q_lora = s.i("q_lora_rank", None);
        c.kv_lora = s.i("kv_lora_rank", None);
        c.qk_nope = s.i("qk_nope_head_dim", None);
        c.qk_rope = s.i("qk_rope_head_dim", None);
        c.v_head = s.i("v_head_dim", None);
        c.mla_out_gate = s.b("mla_use_output_gate", None, 1);

        // Stable LatentMoE. moe_renormalize and latent_moe_use_norm default to 1.
        c.n_experts = s.i("num_experts", None);
        c.topk = s.i("num_experts_per_token", None);
        c.n_shared = s.i("num_shared_experts", None);
        c.latent = s.i("routed_expert_hidden_size", None);
        c.moe_inter = s.i("moe_intermediate_size", None);
        c.routed_scale = s.f("routed_scaling_factor", None);
        c.moe_renorm = s.b("moe_renormalize", None, 1);
        c.latent_norm = s.b("latent_moe_use_norm", None, 1);

        c.first_dense = s.i("first_k_dense_replace", None);
        c.dense_inter = s.i("intermediate_size", None);
        c.attn_res_block = s.i("attn_res_block_size", None);

        // SiTU-GLU. The released and fixture spellings differ; both accepted, neither
        // defaulted. These two are the most dangerous fields to default: 4.0/25.0 are
        // correct, so a defaulting reader hides that it read nothing else either.
        c.situ_b1 = s.f("activation_situ_beta", Some("situ_beta"));
        c.situ_b2 = s.f("activation_situ_linear_beta", Some("situ_linear_beta"));

        // Layer map. Released: text_config.linear_attn_config.full_attn_layers.
        // Fixture: full_attn_layers at top level. k3_cfg.h:168-188.
        match s.find("full_attn_layers", None) {
            Some(fal) if fal.is_array() && !fal.as_array().unwrap().is_empty() => {
                let arr = fal.as_array().unwrap();
                let mut fa = Vec::with_capacity(arr.len());
                for (i, e) in arr.iter().enumerate() {
                    let n = e.as_i64().ok_or_else(|| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("k3_cfg: {} full_attn_layers[{}] is not a number", whence, i),
                        )
                    })?;
                    fa.push(n as i32);
                }
                c.full_attn = fa;
            }
            _ => s.miss("full_attn_layers"),
        }

        // Report ALL missing fields at once, never one per run. k3_cfg.h:190-200.
        if !s.missing.is_empty() {
            let mut msg = format!(
                "k3_cfg: {} is missing {} required field(s):\n",
                whence,
                s.missing.len()
            );
            for name in &s.missing {
                msg.push_str(&format!("    {}\n", name));
            }
            msg.push_str(
                "  refusing to substitute defaults: a config this reader cannot\n  fully understand would silently produce a DIFFERENT model.\n",
            );
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, msg));
        }

        // Structural checks. Each has been seen or is one typo away. k3_cfg.h:202-240.
        if c.n_layers <= 0 || c.hidden <= 0 || c.vocab <= 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("k3_cfg: {} has non-positive layers/hidden/vocab", whence),
            ));
        }
        let n_full = c.full_attn.len() as i32;
        if n_full >= c.n_layers {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "k3_cfg: {} marks {} of {} layers as full attention, leaving no KDA layers",
                    whence, n_full, c.n_layers
                ),
            ));
        }
        for (i, &l) in c.full_attn.iter().enumerate() {
            if l < 1 || l > c.n_layers {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "k3_cfg: {} full_attn_layers[{}] = {} is outside 1..{} (the list is ONE-based)",
                        whence, i, l, c.n_layers
                    ),
                ));
            }
        }
        if c.topk > MAX_TOPK as i32 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "k3_cfg: {} selects top-{}, but this build supports at most {}\n  (K3_MAX_TOPK in k3.h bounds the fixed-size routing arrays)",
                    whence, c.topk, MAX_TOPK
                ),
            ));
        }
        if c.topk > c.n_experts {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "k3_cfg: {} selects {} of {} experts",
                    whence, c.topk, c.n_experts
                ),
            ));
        }
        if c.attn_res_block <= 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "k3_cfg: {} has attn_res_block_size {}; layer_idx % 0 would divide by zero",
                    whence, c.attn_res_block
                ),
            ));
        }
        if c.conv_k < 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("k3_cfg: {} has short_conv_kernel_size {}", whence, c.conv_k),
            ));
        }

        // The C loader prints this summary to stdout on success; it is informational, not
        // part of the contract, and the caller cannot reconstruct the nested/flat label,
        // so emit it to stderr to match. k3_cfg.h:242-246.
        eprintln!(
            "config: {} ({} shape) | hidden={} layers={} vocab={} | {} MLA + {} KDA | experts {} top{} shared{} | latent={}",
            whence,
            if s.nested { "nested" } else { "flat" },
            c.hidden,
            c.n_layers,
            c.vocab,
            c.full_attn.len(),
            c.n_layers - c.full_attn.len() as i32,
            c.n_experts,
            c.topk,
            c.n_shared,
            c.latent
        );

        Ok(c)
    }
}
