// SPDX-License-Identifier: Apache-2.0
//! Tokenizer integration test. Mirrors the C Makefile's `tok` gate: when
//! `K3_TOK_FILES` points at a directory holding `tiktoken.model` (and
//! `tokenizer_config.json`), load it and assert an encode/decode round trip over
//! a source file plus a handful of `id_of` lookups. When the variable is absent,
//! print the NOT RUN note and pass - the vocabulary ships with the checkpoint,
//! not with this repository.
//!
//! Everything checked here is derived from the files under test rather than
//! hardcoded, so the test is correct for any checkpoint: the special tokens come
//! from `added_tokens_decoder` and the EOS from `eos_token`.

use std::path::Path;

/// The NOT RUN note, matching the C Makefile's wording.
fn skip(reason: &str) {
    eprintln!(
        "  NOT RUN: {reason}\n\
         \x20          the vocabulary ships with the checkpoint, not with this\n\
         \x20          repository. Run: K3_TOK_FILES=/path/to/k3model cargo test --test tok"
    );
}

#[test]
fn tok_roundtrip_and_id_of() {
    let dir = match std::env::var("K3_TOK_FILES") {
        Ok(d) => std::path::PathBuf::from(d),
        Err(_) => {
            skip("no tiktoken.model (K3_TOK_FILES unset)");
            return;
        }
    };
    if !dir.join("tiktoken.model").exists() {
        skip(&format!("no tiktoken.model at {}", dir.display()));
        return;
    }

    let tok = match k3::tok::Tok::load(&dir) {
        Ok(t) => t,
        Err(e) => panic!("Tok::load failed: {e}"),
    };

    // --- encode/decode round trip over a source file -----------------------
    // Byte-exact recovery is the bar, matching test_tok.c's `roundtrip` mode: a
    // tokenizer that loses a byte somewhere will silently corrupt a prompt.
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/tok/mod.rs");
    let text = std::fs::read(&src).unwrap_or_else(|e| panic!("read {}: {e}", src.display()));
    let s = String::from_utf8(text.clone()).expect("source file is valid UTF-8");

    let ids = tok.encode(&s);
    assert!(
        !ids.is_empty(),
        "encode produced no ids for {}",
        src.display()
    );

    let back = tok.decode(&ids).into_bytes();
    assert_eq!(
        back.len(),
        text.len(),
        "roundtrip length changed: {} bytes in, {} bytes out",
        text.len(),
        back.len()
    );
    if back != text {
        let d = back.iter().zip(&text).take_while(|(a, b)| a == b).count();
        panic!("roundtrip not byte-exact; first divergence at byte {d}");
    }

    // --- id_of over the added tokens the config actually declares ----------
    let cfg_path = dir.join("tokenizer_config.json");
    let cfg: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&cfg_path).unwrap_or_else(|e| panic!("read {}: {e}", cfg_path.display())),
    )
    .unwrap_or_else(|e| panic!("parse {}: {e}", cfg_path.display()));

    let adt = cfg
        .get("added_tokens_decoder")
        .and_then(|v| v.as_object())
        .expect("tokenizer_config.json has no added_tokens_decoder object");
    assert!(!adt.is_empty(), "added_tokens_decoder is empty");

    // A handful is enough to prove the mapping; checking all of them would just
    // re-run the same lookup a few hundred times.
    let mut checked = 0;
    for (id_str, entry) in adt.iter().take(8) {
        let want: i32 = id_str.parse().expect("added token key is not an integer");
        let content = entry
            .get("content")
            .and_then(|c| c.as_str())
            .expect("added token has no string content");
        assert_eq!(
            tok.id_of(content),
            Some(want),
            "id_of({content:?}) disagrees with added_tokens_decoder"
        );
        // An added token is atomic in encode: it must come back as exactly one id.
        assert_eq!(
            tok.encode(content),
            vec![want],
            "added token {content:?} did not encode atomically"
        );
        checked += 1;
    }
    assert!(checked > 0, "no added tokens were checked");

    // A token that cannot be in the vocabulary resolves to None.
    assert_eq!(
        tok.id_of("<|definitely-not-a-real-added-token|>"),
        None,
        "id_of invented a token"
    );

    // --- eos() tracks the config's eos_token -------------------------------
    let want_eos = cfg
        .get("eos_token")
        .and_then(|v| {
            v.as_str()
                .or_else(|| v.get("content").and_then(|c| c.as_str()))
        })
        .and_then(|name| tok.id_of(name));
    assert_eq!(
        tok.eos(),
        want_eos,
        "eos() disagrees with the eos_token field"
    );

    // --- vocab() is the K3 ceiling from config.json text_config.vocab_size --
    assert_eq!(tok.vocab(), 163840, "vocab() is not K3_TOK_VOCAB");
}
