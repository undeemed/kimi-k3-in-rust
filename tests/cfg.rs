// SPDX-License-Identifier: Apache-2.0
// Port of tests/unit/test_cfg.c. The C test has three modes: fixture (asserts the
// tiny-oracle config's fields), real (asserts the released checkpoint's constants, run
// only when the checkpoint is present), and reject (expects a refusal). This file covers
// the fixture accept case and all three refusal fixtures. The real-config path needs the
// 1.56 TB checkpoint and is out of scope here; the field values it checks are the same
// ones the C test_cfg.c "real" mode asserts, documented in test_cfg.c:68-116.
use std::path::{Path, PathBuf};

use k3::cfg::Cfg;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

/// The accept case: fixtures/ref_k3.json wraps a flat config under "config" and must load
/// with the tiny-oracle numbers from make_k3_oracle.py. Mirrors test_cfg.c:37-66.
#[test]
fn fixture_loads() {
    let path = root().join("fixtures/ref_k3.json");
    let c = Cfg::load(&path).expect("ref_k3.json should load");

    assert_eq!(c.hidden, 128, "hidden");
    assert_eq!(c.n_layers, 13, "layers");
    assert_eq!(c.vocab, 256, "vocab");
    assert_eq!(c.kda_heads, 4, "kda heads");
    assert_eq!(c.kda_head_dim, 16, "kda head_dim");
    assert_eq!(c.n_experts, 8, "experts");
    assert_eq!(c.topk, 2, "topk");
    assert_eq!(c.latent, 64, "latent");
    assert_eq!(c.attn_res_block, 3, "attn_res_block");
    assert_eq!(c.situ_b1, 4.0, "situ b1");
    assert_eq!(c.situ_b2, 25.0, "situ b2");
    assert_eq!(c.full_attn.len(), 4, "full_attn count");
    assert_eq!(c.gate_lb, -5.0, "gate_lb");

    // The one-based full_attn list, exactly as the config lists it.
    assert_eq!(c.full_attn, vec![4, 8, 12, 13]);

    // The derived split the engine branches on. k3_is_mla compares against layer+1, so
    // one-based 4,8,12,13 means zero-based 3,7,11,12 are MLA; layer 0 is KDA.
    let nmla = (0..c.n_layers).filter(|&l| c.is_mla(l)).count();
    let nkda = (0..c.n_layers).filter(|&l| c.is_kda(l)).count();
    assert_eq!(nmla, 4, "MLA layer count");
    assert_eq!(nkda, 9, "KDA layer count");
    assert!(c.is_mla(3) && c.is_mla(12), "layers 3 and 12 both MLA");
    assert!(!c.is_mla(0), "layer 0 is KDA");
}

/// fixtures/cfg/no_layermap.json is a complete flat config MINUS full_attn_layers. The
/// loader must refuse it for the missing-field reason, and the message must name
/// full_attn_layers specifically (and only that field, since nothing else is missing).
#[test]
fn rejects_no_layermap() {
    let path = root().join("fixtures/cfg/no_layermap.json");
    let err = Cfg::load(&path).expect_err("no_layermap.json should be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("missing 1 required field(s)"),
        "expected a single missing field, got: {msg}"
    );
    assert!(
        msg.contains("full_attn_layers"),
        "expected the missing field to be full_attn_layers, got: {msg}"
    );
    assert!(
        msg.contains("refusing to substitute defaults"),
        "expected the refuse-defaults banner, got: {msg}"
    );
}

/// fixtures/cfg/bad_layer_index.json lists full_attn_layers with 999, which is outside
/// 1..n_layers. The loader must refuse it for the out-of-range reason, NOT the
/// missing-field reason, and the message must name the offending index.
#[test]
fn rejects_bad_layer_index() {
    let path = root().join("fixtures/cfg/bad_layer_index.json");
    let err = Cfg::load(&path).expect_err("bad_layer_index.json should be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("full_attn_layers[2] = 999 is outside 1..13"),
        "expected the out-of-range diagnostic naming index 2 and value 999, got: {msg}"
    );
    assert!(
        msg.contains("ONE-based"),
        "expected the ONE-based hint, got: {msg}"
    );
    assert!(
        !msg.contains("missing"),
        "bad_layer_index is not a missing-field case: {msg}"
    );
}

/// fixtures/cfg/bad_topk.json sets num_experts_per_token = 128, which exceeds MAX_TOPK
/// (64). The loader must refuse it for the topk-too-large reason. 128 also exceeds
/// num_experts (8), so either the MAX_TOPK check or the topk>experts check catches it
/// first; the C check order has MAX_TOPK first, so the message must mention "at most".
#[test]
fn rejects_bad_topk() {
    let path = root().join("fixtures/cfg/bad_topk.json");
    let err = Cfg::load(&path).expect_err("bad_topk.json should be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("selects top-128"),
        "expected the diagnostic to name top-128, got: {msg}"
    );
    assert!(
        msg.contains("at most 64"),
        "expected the MAX_TOPK=64 bound in the message, got: {msg}"
    );
    assert!(
        !msg.contains("missing"),
        "bad_topk is not a missing-field case: {msg}"
    );
}
