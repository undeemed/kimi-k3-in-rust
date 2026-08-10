// SPDX-License-Identifier: Apache-2.0
//! Safetensors reader tests.
//!
//! Port of `tests/unit/test_st.c` and the `test-st` recipe in the Makefile. The C test's
//! own pass/fail is only the round-trip and ghost-name checks; the six named tensors are
//! read and their stats printed for `tools/verify_st.py` to compare externally. Here we
//! assert the dtype, shape, numel, and a prefix/suffix of the widened f32 values for each
//! of the six named cases, plus the round-trip and ghost-name semantics the C test checks.

// The expected values are the FULL decimal expansions `tools/verify_st.py` prints for the
// same fixture bytes, kept verbatim so the two can be compared by eye. They already parse
// to the exact f32 asserted here; shortening them would only make the comparison harder.
#![allow(clippy::excessive_precision)]

use std::path::Path;

use k3::st::{Dtype, St};

/// The fixture directory, relative to the crate root.
fn fixtures_st() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join("st")
}

/// A scratch directory under `target/` that never touches `fixtures/`.
fn scratch_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("st_test")
}

fn f32_bits(x: f32) -> u32 {
    x.to_bits()
}

#[test]
fn opens_two_shards_and_indexes_all_tensors() {
    let st = St::open(&fixtures_st()).expect("open fixtures/st");
    // Two shards, sorted: model-00001-of-00002 then model-00002-of-00002.
    assert_eq!(st.nshard(), 2);
    // The fixture headers list these tensors (minus __metadata__):
    // shard 0: plain.f32.2d, plain.bf16.1d, tricky.f16.1d, packed.u8.2d, scalar.f32,
    //         empty.f32, 2 expert packed tensors, rank3.f32, rank4.u8
    // shard 1: second.shard.f32, second.shard.u8, weird\name."with"quotes
    // __metadata__ is skipped, not indexed.
    let total = st.tensors().len();
    assert_eq!(
        total, 13,
        "expected 13 indexed tensors (all minus __metadata__), got {}",
        total
    );
}

#[test]
fn round_trip_every_name_resolves_to_itself() {
    let st = St::open(&fixtures_st()).expect("open");
    let mut bad = 0;
    for (i, t) in st.tensors().iter().enumerate() {
        match st.find(&t.name) {
            Some(f) if std::ptr::eq(f as *const _, &st.tensors()[i] as *const _) => {}
            _ => {
                bad += 1;
                eprintln!("round-trip miss on index {} name {}", i, t.name);
            }
        }
    }
    assert_eq!(bad, 0, "{} names did not resolve to themselves", bad);
}

#[test]
fn absent_names_return_none() {
    let st = St::open(&fixtures_st()).expect("open");
    // The C test uses these three ghost names. k3_st.c uses the same set in test_st.c:102.
    let ghosts = [
        "",
        "no.such.tensor",
        "language_model.model.layers.999.self_attn.A_log",
    ];
    for g in &ghosts {
        assert!(st.find(g).is_none(), "ghost name {:?} should be absent", g);
    }
}

#[test]
fn weird_name_with_escapes_is_indexed() {
    let st = St::open(&fixtures_st()).expect("open");
    // The fixture has a tensor whose name contains a backslash and quotes; the C
    // scanner and serde_json both unescape it to this.
    let t = st
        .find(r#"weird\name."with"quotes"#)
        .expect("weird-name tensor must be indexed");
    assert_eq!(t.dtype, Dtype::F32);
    assert_eq!(t.shape(), vec![3]);
    assert_eq!(t.numel(), 3);
    assert_eq!(t.shard, 1);
}

#[test]
fn plain_f32_2d() {
    let st = St::open(&fixtures_st()).expect("open");
    let t = st.find("plain.f32.2d").expect("plain.f32.2d present");
    assert_eq!(t.dtype, Dtype::F32);
    assert_eq!(t.shape(), vec![16, 16]);
    assert_eq!(t.numel(), 256);
    assert_eq!(t.shard, 0);
    assert_eq!(t.nbytes, 1024);

    let mut out = vec![0.0f32; 256];
    let got = st.read_f32(t, &mut out).expect("read_f32");
    assert_eq!(got, 256);

    // First four and last four, from the fixture (see tools/verify_st.py parity).
    assert_eq!(f32_bits(out[0]), f32_bits(0.0036904599983245134));
    assert_eq!(f32_bits(out[1]), f32_bits(0.8962365984916687));
    assert_eq!(f32_bits(out[2]), f32_bits(-0.8224135637283325));
    assert_eq!(f32_bits(out[3]), f32_bits(-2.6717755794525146));
    assert_eq!(f32_bits(out[252]), f32_bits(4.0006794929504395));
    assert_eq!(f32_bits(out[253]), f32_bits(0.14135971665382385));
    assert_eq!(f32_bits(out[254]), f32_bits(-3.517637014389038));
    assert_eq!(f32_bits(out[255]), f32_bits(-2.8220996856689453));
}

#[test]
fn plain_bf16_1d() {
    let st = St::open(&fixtures_st()).expect("open");
    let t = st.find("plain.bf16.1d").expect("plain.bf16.1d present");
    assert_eq!(t.dtype, Dtype::Bf16);
    assert_eq!(t.shape(), vec![128]);
    assert_eq!(t.numel(), 128);
    assert_eq!(t.shard, 0);
    assert_eq!(t.nbytes, 256);

    let mut out = vec![0.0f32; 128];
    let got = st.read_f32(t, &mut out).expect("read_f32");
    assert_eq!(got, 128);

    // bf16 -> f32 is a pure left shift by 16 bits, so the bit patterns are exact.
    // The fixture's first four bf16 codes decode to inf, -inf, nan, 0.0.
    assert!(
        out[0].is_infinite() && out[0].is_sign_positive(),
        "out[0] = +inf, got {}",
        out[0]
    );
    assert!(
        out[1].is_infinite() && out[1].is_sign_negative(),
        "out[1] = -inf, got {}",
        out[1]
    );
    assert!(out[2].is_nan(), "out[2] = nan, got {}", out[2]);
    assert_eq!(f32_bits(out[3]), f32_bits(0.0));
    // The tail carries a mix of normals; spot-check the last.
    assert_eq!(f32_bits(out[127]), f32_bits(1.5643308870494366e-10));
}

#[test]
fn tricky_f16_1d() {
    let st = St::open(&fixtures_st()).expect("open");
    let t = st.find("tricky.f16.1d").expect("tricky.f16.1d present");
    assert_eq!(t.dtype, Dtype::F16);
    assert_eq!(t.shape(), vec![64]);
    assert_eq!(t.numel(), 64);
    assert_eq!(t.shard, 0);
    assert_eq!(t.nbytes, 128);

    let mut out = vec![0.0f32; 64];
    let got = st.read_f32(t, &mut out).expect("read_f32");
    assert_eq!(got, 64);

    // The fixture deliberately carries inf/-inf/nan/0 at the head, and the f16 -> f32
    // port must preserve them bit-for-bit. test_st.c:159 notes non-finite values here
    // are the reader working, not failing.
    assert!(
        out[0].is_infinite() && out[0].is_sign_positive(),
        "out[0] = +inf, got {}",
        out[0]
    );
    assert!(
        out[1].is_infinite() && out[1].is_sign_negative(),
        "out[1] = -inf, got {}",
        out[1]
    );
    assert!(out[2].is_nan(), "out[2] = nan, got {}", out[2]);
    assert_eq!(f32_bits(out[3]), f32_bits(0.0));
    // The tail is finite; spot-check against the fixture's widened values.
    assert_eq!(f32_bits(out[60]), f32_bits(0.25927734375));
    assert_eq!(f32_bits(out[61]), f32_bits(0.55322265625));
    assert_eq!(f32_bits(out[62]), f32_bits(1.9521484375));
    assert_eq!(f32_bits(out[63]), f32_bits(-0.19677734375));
}

#[test]
fn packed_u8_2d() {
    let st = St::open(&fixtures_st()).expect("open");
    let t = st.find("packed.u8.2d").expect("packed.u8.2d present");
    assert_eq!(t.dtype, Dtype::U8);
    assert_eq!(t.shape(), vec![7, 32]);
    assert_eq!(t.numel(), 224);
    assert_eq!(t.shard, 0);
    assert_eq!(t.nbytes, 224);

    let mut out = vec![0.0f32; 224];
    let got = st.read_f32(t, &mut out).expect("read_f32");
    assert_eq!(got, 224);

    // U8 widens to the raw byte value as a float. k3_st.c:558.
    assert_eq!(out[0], 23.0);
    assert_eq!(out[1], 157.0);
    assert_eq!(out[2], 148.0);
    assert_eq!(out[3], 190.0);
    assert_eq!(out[220], 133.0);
    assert_eq!(out[221], 105.0);
    assert_eq!(out[222], 168.0);
    assert_eq!(out[223], 129.0);
}

#[test]
fn scalar_f32() {
    let st = St::open(&fixtures_st()).expect("open");
    let t = st.find("scalar.f32").expect("scalar.f32 present");
    assert_eq!(t.dtype, Dtype::F32);
    // A scalar has shape [] in safetensors; numel is 1.
    assert_eq!(t.shape(), Vec::<i64>::new());
    assert_eq!(t.numel(), 1);
    assert_eq!(t.shard, 0);
    assert_eq!(t.nbytes, 4);

    let mut out = vec![0.0f32; 1];
    let got = st.read_f32(t, &mut out).expect("read_f32");
    assert_eq!(got, 1);
    assert_eq!(f32_bits(out[0]), f32_bits(3.5));
}

#[test]
fn second_shard_f32() {
    let st = St::open(&fixtures_st()).expect("open");
    let t = st
        .find("second.shard.f32")
        .expect("second.shard.f32 present");
    assert_eq!(t.dtype, Dtype::F32);
    assert_eq!(t.shape(), vec![10, 10]);
    assert_eq!(t.numel(), 100);
    // This one lives in the SECOND shard, which is the whole point of the case.
    assert_eq!(t.shard, 1);
    assert_eq!(t.nbytes, 400);

    let mut out = vec![0.0f32; 100];
    let got = st.read_f32(t, &mut out).expect("read_f32");
    assert_eq!(got, 100);

    assert_eq!(f32_bits(out[0]), f32_bits(-0.3837008774280548));
    assert_eq!(f32_bits(out[1]), f32_bits(-0.29570621252059937));
    assert_eq!(f32_bits(out[2]), f32_bits(-1.1264570951461792));
    assert_eq!(f32_bits(out[3]), f32_bits(2.5369296073913574));
    assert_eq!(f32_bits(out[96]), f32_bits(-0.030903251841664314));
    assert_eq!(f32_bits(out[97]), f32_bits(-0.0840543583035469));
    assert_eq!(f32_bits(out[98]), f32_bits(-0.09379192441701889));
    assert_eq!(f32_bits(out[99]), f32_bits(-1.1218096017837524));
}

#[test]
fn raw_read_returns_stored_bytes() {
    let st = St::open(&fixtures_st()).expect("open");
    let t = st.find("scalar.f32").expect("scalar.f32 present");
    let mut buf = [0u8; 4];
    let got = st.read(t, &mut buf).expect("read");
    assert_eq!(got, 4);
    // 3.5f32 little-endian is 0x40600000.
    assert_eq!(buf, [0x00, 0x00, 0x60, 0x40]);
}

#[test]
fn rebuilds_index_under_target_without_touching_fixtures() {
    // The Makefile recipe writes the rebuilt index to $(BUILD); we mirror that by writing
    // under target/ so two concurrent runs cannot race and `fixtures/` stays read-only.
    let dir = scratch_dir();
    std::fs::create_dir_all(&dir).expect("mkdir scratch");
    let out = dir.join("st_index.json");

    let st = St::open(&fixtures_st()).expect("open");
    // A minimal index dump: one line per tensor, name + dtype + shape + shard.
    let mut s = String::new();
    s.push_str("{\n");
    for (i, t) in st.tensors().iter().enumerate() {
        if i > 0 {
            s.push_str(",\n");
        }
        s.push_str(&format!(
            "  \"{}\": {{\"shard\":{},\"dtype\":\"{:?}\",\"shape\":{:?},\"off\":{},\"nbytes\":{}}}",
            t.name,
            t.shard,
            t.dtype,
            t.shape(),
            t.off,
            t.nbytes
        ));
    }
    s.push_str("\n}\n");
    std::fs::write(&out, s).expect("write index");
    assert!(out.exists(), "index written to {}", out.display());
}

#[test]
fn empty_tensor_has_zero_numel_and_zero_bytes() {
    let st = St::open(&fixtures_st()).expect("open");
    let t = st.find("empty.f32").expect("empty.f32 present");
    assert_eq!(t.dtype, Dtype::F32);
    assert_eq!(t.shape(), vec![0, 8]);
    assert_eq!(t.numel(), 0);
    assert_eq!(t.nbytes, 0);
    // Reading zero bytes into a zero-length buffer must succeed and return 0.
    let mut out: Vec<f32> = Vec::new();
    let got = st.read_f32(t, &mut out).expect("read_f32 on empty");
    assert_eq!(got, 0);
}

#[test]
fn rank3_and_rank4_tensors_index() {
    let st = St::open(&fixtures_st()).expect("open");
    let t = st.find("rank3.f32").expect("rank3.f32 present");
    assert_eq!(t.dtype, Dtype::F32);
    assert_eq!(t.shape(), vec![2, 3, 4]);
    assert_eq!(t.numel(), 24);

    let t = st.find("rank4.u8").expect("rank4.u8 present");
    assert_eq!(t.dtype, Dtype::U8);
    assert_eq!(t.shape(), vec![2, 2, 2, 2]);
    assert_eq!(t.numel(), 16);
}

#[test]
fn expert_packed_tensors_resolve() {
    let st = St::open(&fixtures_st()).expect("open");
    // The fixture carries two expert w1.weight_packed entries, exercising the long-name
    // path that motivated the FNV-1a hash in the C code.
    let t = st
        .find("language_model.model.layers.0.block_sparse_moe.experts.0.w1.weight_packed")
        .expect("expert 0 w1 packed");
    assert_eq!(t.dtype, Dtype::U8);
    assert_eq!(t.shape(), vec![2, 32]);
    assert_eq!(t.numel(), 64);
    assert_eq!(t.nbytes, 64);

    let t = st
        .find("language_model.model.layers.0.block_sparse_moe.experts.1.w1.weight_packed")
        .expect("expert 1 w1 packed");
    assert_eq!(t.dtype, Dtype::U8);
    assert_eq!(t.shape(), vec![2, 32]);
}

/// Write a one-shard directory holding exactly the given header entries, sized so every
/// tensor's span lies inside the file. Used to drive the reader's refusal paths, which no
/// fixture can express: a committed fixture with a duplicate name would break every other
/// test that opens the directory.
fn write_header_only(dir: &Path, shards: &[(&str, &str)]) {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).expect("mkdir");
    for (fname, header) in shards {
        let mut blob = header.as_bytes().to_vec();
        while blob.len() % 8 != 0 {
            blob.push(b' ');
        }
        let mut out = (blob.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&blob);
        out.resize(out.len() + 4096, 0); // room for any span the header declares
        std::fs::write(dir.join(fname), out).expect("write shard");
    }
}

#[test]
fn duplicate_name_across_shards_is_refused() {
    // C's hash insert refuses a second copy of a name (k3_st.c:422). The index is keyed by
    // FNV-1a rather than by an owned string, so this refusal runs through a hash hit plus a
    // name comparison; without the comparison a collision would false-positive, and without
    // the refusal a duplicate would silently shadow the first tensor.
    let dir = scratch_dir().join("dup");
    let entry = r#"{"same.name":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    write_header_only(
        &dir,
        &[
            ("model-00001-of-00002.safetensors", entry),
            ("model-00002-of-00002.safetensors", entry),
        ],
    );

    let err = match St::open(&dir) {
        Ok(_) => panic!("duplicate name must be refused"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("duplicate tensor name") && msg.contains("same.name"),
        "expected a duplicate-name refusal naming the tensor, got: {msg}"
    );
}

#[test]
fn rank_above_four_is_refused_not_truncated() {
    // Shape is stored inline as `[i64; 4]`, matching C's `int64_t shape[4]`. A rank-5 tensor
    // has to be refused: truncating to the first four dims would leave numel disagreeing
    // with the byte span, and every later read of that tensor misaligned.
    let dir = scratch_dir().join("rank5");
    write_header_only(
        &dir,
        &[(
            "model-00001-of-00001.safetensors",
            r#"{"too.deep":{"dtype":"F32","shape":[2,2,2,2,2],"data_offsets":[0,128]}}"#,
        )],
    );

    let err = match St::open(&dir) {
        Ok(_) => panic!("rank 5 must be refused"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("rank 5") && msg.contains("max 4"),
        "expected a rank-ceiling refusal, got: {msg}"
    );
}
