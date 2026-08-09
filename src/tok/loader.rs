// SPDX-License-Identifier: Apache-2.0
//! `k3_tok.h` port: construct a `Tok` directly from the released Kimi K3 files
//! (`tiktoken.model` + `tokenizer_config.json`), with no synthesised
//! `tokenizer.json`.
//!
//! THREE SILENT INVARIANTS, each enforced here, each silent when violated (the
//! tokenizer runs, emits ids, and is wrong):
//!
//! 1. The vocabulary is keyed by the GPT-2 BYTE-LEVEL string, each byte mapped to
//!    a printable codepoint via `byte2str`, not by raw token bytes. `tiktoken.model`
//!    supplies RAW bytes. Omitting the conversion yields a hash table that never
//!    hits, so every piece degrades to single bytes and the model runs on garbage
//!    ids. `Tok::build_bytemap` runs before any key is constructed - enforced at
//!    the top of `load` (k3_tok.h:118).
//! 2. `rankbpe` must be 1. There is no merges list in this format; tiktoken merges
//!    the adjacent pair whose CONCATENATION has the lowest id. With `rankbpe` at 0
//!    the encoder consults an empty merges map and emits one token per codepoint.
//!    Enforced at `load` (k3_tok.h:121); the `merges` map is left empty.
//! 3. `kimi` must be 1 (which makes the o200k flag irrelevant). The K3 pre-tokenizer
//!    adds a leading `\p{Han}`-run rule and excludes Han from its letter classes.
//!    `tok.h` normally infers the family by inspecting the pre-tokenizer regex
//!    inside `tokenizer.json`; with no such file the flag must be set explicitly.
//!    Enforced at `load` (k3_tok.h:119-120).

use std::io::{Error, ErrorKind, Result};
use std::path::Path;

use serde_json::Value;

use super::Tok;

/// Vocabulary ceiling from `config.json text_config.vocab_size`. Ranks occupy
/// `[0, 163584)` and the added tokens land above that, so one array of this size
/// covers both. Asserted against the files rather than assumed. (k3_tok.h:66)
pub const K3_TOK_VOCAB: usize = 163840;

/// Standard base64 alphabet, '=' padded. Returns decoded length, or an error on a
/// malformed group. Port of `k3_b64` (k3_tok.h:71). `out` must hold at least
/// `3*((n+3)/4)` bytes.
fn b64_decode(input: &[u8], out: &mut [u8]) -> Result<usize> {
    // 256-entry lookup; -1 (i8) for non-alphabet bytes.
    static TABLE: std::sync::LazyLock<[i8; 256]> = std::sync::LazyLock::new(|| {
        let mut t = [-1i8; 256];
        const A: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (i, &c) in A.iter().enumerate() {
            t[c as usize] = i as i8;
        }
        t
    });
    let t = *TABLE;
    let mut o = 0usize;
    let mut acc: u32 = 0;
    let mut bits: u32 = 0;
    for &c in input {
        if c == b'=' {
            break;
        }
        let v = t[c as usize];
        if v < 0 {
            return Err(Error::new(ErrorKind::InvalidData, "k3_tok: bad base64"));
        }
        acc = (acc << 6) | (v as u32);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out[o] = ((acc >> bits) & 0xFF) as u8;
            o += 1;
        }
    }
    Ok(o)
}

/// Read a whole file into a `Vec<u8>`. Hard error (no silent half-load) on any
/// I/O failure - a tokenizer that loads half-correctly is worse than one that
/// refuses, because the failure surfaces as subtly wrong text hundreds of tokens
/// later. Port of `tk_read_file` (tok.h:104).
fn read_file(path: &Path) -> Result<Vec<u8>> {
    let buf = std::fs::read(path)
        .map_err(|e| Error::new(e.kind(), format!("{}: {}", path.display(), e)))?;
    // sanity cap vs a hostile size (tok.h:109: 1 GiB)
    if buf.len() > (1usize << 30) {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("{}: file too large ({} bytes)", path.display(), buf.len()),
        ));
    }
    Ok(buf)
}

/// Load `tiktoken.model` and `tokenizer_config.json` from `dir` into a fresh
/// `Tok`. Port of `k3_tok_load` (k3_tok.h:113). Returns an error on any malformed
/// input; the C version `exit(1)`s, this returns `Err` so the library API stays
/// panic-free on parse failures.
pub fn load(dir: &Path) -> Result<Tok> {
    let mut t = Tok {
        vocab: std::collections::HashMap::new(),
        merges: std::collections::HashMap::new(),
        id2str: Vec::new(),
        id_added: Vec::new(),
        id_special: Vec::new(),
        sp: Vec::new(),
        byte2cp: [0; 256],
        byte2cp_len: [0; 256],
        byte2str: [[0; 4]; 256],
        cp2byte: [-1; 1024],
        o200k: false,
        kimi: false,
        rankbpe: false,
        n_ids: 0,
        eos_id: None,
    };

    // Order matters: byte2str must exist before any vocab key is built (trap #1).
    t.build_bytemap();
    t.kimi = true; // trap #3
    t.o200k = true; // K3 builds on the o200k rules; kimi refines them
    t.rankbpe = true; // trap #2

    t.n_ids = K3_TOK_VOCAB;
    t.id2str = vec![Vec::new(); t.n_ids];
    t.id_added = vec![false; t.n_ids];
    t.id_special = vec![false; t.n_ids];

    // ---- ranks ----
    let model_path = dir.join("tiktoken.model");
    let buf = read_file(&model_path)?;

    // Power-of-two capacity, ~2x load factor, as tok_load does. The std HashMap
    // resizes itself, but reserving the same capacity avoids rehashing.
    let vc = {
        let mut v = 1usize;
        while v < 163584 * 2 {
            v <<= 1;
        }
        v
    };
    t.vocab.reserve(vc);

    let mut nrank = 0usize;
    let mut maxrank: i32 = -1;
    let mut i = 0usize;
    let nbuf = buf.len();
    while i < nbuf {
        let ls = i;
        while i < nbuf && buf[i] != b'\n' {
            i += 1;
        }
        let mut le = i;
        if i < nbuf {
            i += 1; // step over '\n'
        }
        if le > ls && buf[le - 1] == b'\r' {
            le -= 1;
        }
        if le <= ls {
            continue; // blank line
        }

        // split on the single space: "<base64> <rank>"
        let mut sp = ls;
        while sp < le && buf[sp] != b' ' {
            sp += 1;
        }
        if sp >= le {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("k3_tok: tiktoken.model line {} has no rank field", nrank),
            ));
        }
        let blen = sp - ls;
        // atoi(buf + sp + 1): parse the rank as decimal, possibly with leading
        // whitespace/sign, up to the end of line.
        let rank = parse_i32(&buf[sp + 1..le]);
        let rank = match rank {
            Some(r) => r,
            None => {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!("k3_tok: bad rank at line {}", nrank),
                ))
            }
        };
        if rank < 0 || rank as usize >= t.n_ids {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("k3_tok: rank {} out of range at line {}", rank, nrank),
            ));
        }

        // raw token bytes: at most 384 bytes (512 raw -> 683 b64 chars). A
        // 1024-byte scratch covers any plausible token.
        let mut raw = [0u8; 1024];
        if blen > raw.len() * 4 / 3 {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("k3_tok: implausibly long token at rank {}", rank),
            ));
        }
        let rn = b64_decode(&buf[ls..ls + blen], &mut raw)?;

        // byte-level string: worst case two output bytes per input byte.
        let mut key = vec![0u8; 2 * rn + 1];
        let kl = bytelevel(&t, &raw[..rn], &mut key);
        key.truncate(kl);

        t.vocab.insert(key.clone(), rank);
        t.id2str[rank as usize] = key; // decode reverses this via cp2byte
        if rank > maxrank {
            maxrank = rank;
        }
        nrank += 1;
    }
    if nrank == 0 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "k3_tok: tiktoken.model is empty",
        ));
    }

    // ---- specials ----
    let cfg_path = dir.join("tokenizer_config.json");
    let cfg = read_file(&cfg_path)?;
    let root: Value = serde_json::from_slice(&cfg).map_err(|e| {
        Error::new(
            ErrorKind::InvalidData,
            format!("k3_tok: tokenizer_config.json: {}", e),
        )
    })?;
    let adt = root.get("added_tokens_decoder").ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidData,
            "k3_tok: tokenizer_config.json has no added_tokens_decoder",
        )
    })?;
    let adt_obj = adt.as_object().ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidData,
            "k3_tok: added_tokens_decoder is not an object",
        )
    })?;

    let nsp = adt_obj.len();
    let mut sp: Vec<super::Special> = Vec::with_capacity(nsp);

    for (key_str, e) in adt_obj {
        // keys are the ids, as strings: {"163584": {"content": "[BOS]", ...}}
        let id = match key_str.parse::<i32>() {
            Ok(v) => v,
            Err(_) => {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!("k3_tok: added token id '{}' not an integer", key_str),
                ))
            }
        };
        let jc = e.get("content").ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidData,
                format!("k3_tok: added token {} has no content", key_str),
            )
        })?;
        let content = jc.as_str().ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidData,
                format!("k3_tok: added token {} content is not a string", key_str),
            )
        })?;
        if id < 0 || id as usize >= t.n_ids {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("k3_tok: added token id {} out of range", id),
            ));
        }
        let content_bytes = content.as_bytes().to_vec();
        t.id2str[id as usize] = content_bytes.clone(); // added tokens decode literally
        t.id_added[id as usize] = true;
        if let Some(sf) = e.get("special") {
            if sf.as_bool().unwrap_or(false) {
                t.id_special[id as usize] = true;
            }
        }
        sp.push(super::Special {
            str_: content_bytes,
            id,
        });
    }
    // longest match first, so "<|end_of_msg|>" wins over any prefix of it
    sp.sort_by_key(|s| std::cmp::Reverse(s.str_.len()));
    t.sp = sp;

    // EOS: read the standard HuggingFace `eos_token` field and resolve it through
    // the added tokens, rather than assuming a name. The field is either a bare
    // string or an object carrying `content`. The C loader never resolves EOS (the
    // run path is driven by `--ids`), so there is no C behaviour to match here;
    // reading the file is the only way to be right for any checkpoint.
    t.eos_id = root
        .get("eos_token")
        .and_then(|v| match v {
            Value::String(s) => Some(s.as_str()),
            _ => v.get("content").and_then(|c| c.as_str()),
        })
        .and_then(|name| {
            t.sp.iter()
                .find(|s| s.str_.as_slice() == name.as_bytes())
                .map(|s| s.id)
        });

    eprintln!(
        "[TOK] {} ranks (max id {}) + {} added tokens | kimi={} rankbpe={}",
        nrank,
        maxrank,
        t.sp.len(),
        t.kimi as i32,
        t.rankbpe as i32
    );

    Ok(t)
}

/// Raw token bytes -> the byte-level string `Tok` hashes on. Port of
/// `k3_bytelevel` (k3_tok.h:97). `out` must hold `2*n+1`. Returns the string
/// length.
fn bytelevel(t: &Tok, b: &[u8], out: &mut [u8]) -> usize {
    let mut o = 0;
    for &bb in b {
        let bb = bb as usize;
        let l = t.byte2cp_len[bb];
        out[o..o + l].copy_from_slice(&t.byte2str[bb][..l]);
        o += l;
    }
    o
}

/// Parse a possibly-signed decimal integer from a byte slice, mirroring C
/// `atoi`: leading whitespace, optional sign, digits. Returns None if no digit
/// is found. (atoi stops at the first non-digit; we do too.)
fn parse_i32(s: &[u8]) -> Option<i32> {
    let mut i = 0;
    while i < s.len() && (s[i] == b' ' || s[i] == b'\t' || s[i] == b'\r') {
        i += 1;
    }
    let mut neg = false;
    if i < s.len() && (s[i] == b'+' || s[i] == b'-') {
        neg = s[i] == b'-';
        i += 1;
    }
    let mut v: i64 = 0;
    let mut any = false;
    while i < s.len() && s[i].is_ascii_digit() {
        v = v * 10 + (s[i] - b'0') as i64;
        any = true;
        i += 1;
    }
    if !any {
        return None;
    }
    if neg {
        v = -v;
    }
    if v > i32::MAX as i64 || v < i32::MIN as i64 {
        return None;
    }
    Some(v as i32)
}
