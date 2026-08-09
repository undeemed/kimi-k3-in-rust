// SPDX-License-Identifier: Apache-2.0
//! The binder must ask the checkpoint for FULLY QUALIFIED tensor names.
//!
//! C concatenates `PRE "language_model.model."` into each of its 39 layer templates at
//! the call site (k3_bind.c:11,170). The Rust port keeps the templates bare and prefixes
//! them once in `fmt_name`, so a single missing `PRE` silently unqualifies all 39 names.
//! That shipped: the binder looked up `layers.0.input_layernorm.weight`, which no
//! released checkpoint holds, so `k3` could not bind a single layer of the real model.
//!
//! Nothing caught it. `tests/model_oracle.rs` has its own fixture-named weight store and
//! never calls the binder, and `tests/real_layer.rs` is gated on `K3_SHARD_DIR`. This
//! test is the ungated gate: it drives the real `layer_bytes` planner against a shard set
//! that deliberately holds NO layer tensors, and reads the name out of the resulting
//! error. Every attention and MLP branch is covered, because the templates differ per
//! branch and a partial prefix would be just as broken as none.

use std::path::{Path, PathBuf};

use k3::bind;
use k3::cfg::Cfg;
use k3::st::St;

/// The prefix every layer tensor carries in the released checkpoint. k3_bind.c:11.
const PRE: &str = "language_model.model.";

/// A scratch shard set holding one tensor that is not part of any layer, so `St::open`
/// succeeds and every layer lookup misses. Written under `target/`, never `fixtures/`.
///
/// `who` MUST be unique per test. Cargo runs the tests in one binary concurrently, and a
/// shared path here is a real race: one test read the file while the other was still
/// writing it and got "too short for a header length". Separate directories remove the
/// sharing rather than synchronising it.
fn scratch_shards(who: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("bind_names_test")
        .join(who);
    std::fs::create_dir_all(&dir).expect("mkdir scratch");

    let header = br#"{"unrelated.tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut pad = header.to_vec();
    while pad.len() % 8 != 0 {
        pad.push(b' ');
    }
    let mut blob = Vec::new();
    blob.extend_from_slice(&(pad.len() as u64).to_le_bytes());
    blob.extend_from_slice(&pad);
    blob.extend_from_slice(&0f32.to_le_bytes());
    std::fs::write(dir.join("model-00001-of-00001.safetensors"), blob).expect("write shard");
    dir
}

fn tiny_cfg() -> Cfg {
    Cfg::load(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("tiny_k3.json"),
    )
    .expect("load fixtures/tiny_k3.json")
}

/// The name the planner asked for first, recovered from the miss it reports.
fn first_requested_name(st: &St, c: &Cfg, layer: usize) -> String {
    let err = bind::layer_bytes(st, c, layer)
        .expect_err("a shard set with no layer tensors must fail to plan")
        .to_string();
    err.strip_prefix("k3_bind: missing tensor ")
        .unwrap_or_else(|| panic!("unexpected planner error, cannot recover the name: {err}"))
        .to_string()
}

#[test]
fn every_layer_branch_requests_fully_qualified_names() {
    let st = St::open(&scratch_shards("layer")).expect("open scratch shards");
    let c = tiny_cfg();

    // The tiny config is 13 layers: first_k_dense_replace = 1 makes layer 0 dense, and
    // the one-based full_attn_layers [4,8,12,13] make layers 3,7,11,12 MLA. So layer 0 is
    // KDA + dense, layer 1 is KDA + MoE, layer 3 is MLA + MoE, and between them they
    // reach all four template groups.
    assert!(c.is_dense(0) && !c.is_mla(0), "layer 0 is KDA + dense");
    assert!(!c.is_dense(1) && !c.is_mla(1), "layer 1 is KDA + MoE");
    assert!(!c.is_dense(3) && c.is_mla(3), "layer 3 is MLA + MoE");

    for layer in [0usize, 1, 3] {
        let name = first_requested_name(&st, &c, layer);
        assert!(
            name.starts_with(PRE),
            "layer {layer} planner asked for `{name}`, which is missing the `{PRE}` prefix; \
             no released checkpoint holds that name"
        );
        assert_eq!(
            name,
            format!("{PRE}layers.{layer}.input_layernorm.weight"),
            "layer {layer} first request must match the C template PRE \"layers.%d.input_layernorm.weight\""
        );
    }
}

#[test]
fn model_level_names_are_fully_qualified() {
    // The model-level planner builds its names with `PRE` inline rather than through
    // `fmt_name`, so it never carried the bug; this pins that it stays that way.
    let st = St::open(&scratch_shards("model")).expect("open scratch shards");
    let c = tiny_cfg();
    let err = match bind::ModelBind::load(&st, &c, true) {
        Ok(_) => panic!("a shard set with no model tensors must fail to plan"),
        Err(e) => e.to_string(),
    };
    let name = err
        .strip_prefix("k3_bind: missing tensor ")
        .unwrap_or_else(|| panic!("unexpected planner error: {err}"));
    assert_eq!(name, format!("{PRE}embed_tokens.weight"));
}
