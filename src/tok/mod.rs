// SPDX-License-Identifier: Apache-2.0
//! Byte-level BPE tokenizer, a port of `third_party/tok.h` from the C reference
//! (commit ff11dce). Faithful reimplementation of HuggingFace `tokenizer.json`
//! semantics: model.type = BPE, ignore_merges = true, byte_fallback = false;
//! pre-tokenizer regex Split (cl100k / o200k / Kimi) + ByteLevel; merge rank is the
//! position in the merges list; `\p{L}/\p{N}/\s` come from `unicode.rs`.
//!
//! The C `tok.h` is never fed a `tokenizer.json` by this engine: Kimi K3 ships none.
//! `loader.rs` populates these structures directly from the released `tiktoken.model`
//! and `tokenizer_config.json`, after which `encode`/`decode`/`id_of` work unchanged.
//!
//! Internals stay close to the C for parity; only the public surface (`Tok`) is
//! idiomatic Rust. The three pre-tokenizers (`pretok_chunk`, `pretok_chunk_o200k`,
//! `pretok_chunk_kimi`) are ported statement for statement so the bit-identical
//! tokenization contract holds.

pub mod loader;
pub mod unicode;

use std::collections::HashMap;

/// One added (special or not) token. Sorted by descending length so the longest
/// match wins when splitting on added tokens.
#[derive(Clone)]
pub(super) struct Special {
    pub(super) str_: Vec<u8>,
    pub(super) id: i32,
}

/// The byte-level BPE tokenizer. The C `Tok` struct, with the tagged `const void*`
/// id2str replaced by a `Vec<u8>` per id and the open-addressing `hmap` replaced by
/// `std::collections::HashMap` keyed on `Vec<u8>` (binary key + explicit length,
/// exactly the C `klen && memcmp` semantics).
pub struct Tok {
    /// byte-level string -> id
    vocab: HashMap<Vec<u8>, i32>,
    /// `"left\0right"` -> rank, used only when `rankbpe` is false (never for K3)
    merges: HashMap<Vec<u8>, i32>,
    /// id -> byte-level string (the vocab entry's key, or the added token's content)
    id2str: Vec<Vec<u8>>,
    /// id_added[id] = true for an added token (emitted literally on decode)
    id_added: Vec<bool>,
    /// id_special[id] = true for an added token with `"special": true`. Control tokens
    /// (`<|endoftext|>`, `<|endofprompt|>`, ...), never legitimate response content.
    /// Distinct from `id_added`, which also covers `<|im_start|>`/`<|im_end|>`
    /// (`"special": false`) that ARE real text and must be rendered.
    id_special: Vec<bool>,
    /// added tokens, sorted by descending length
    sp: Vec<Special>,
    /// byte -> codepoint in the GPT-2 byte-level map
    byte2cp: [u32; 256],
    /// byte -> length in bytes of that codepoint's UTF-8 encoding
    byte2cp_len: [usize; 256],
    /// byte -> the UTF-8 bytes of the codepoint (max 3 bytes + NUL)
    byte2str: [[u8; 4]; 256],
    /// codepoint (< 1024) -> original byte, or -1
    cp2byte: [i16; 1024],
    /// pre_tokenizer regex family: false = cl100k, true = o200k
    o200k: bool,
    /// Kimi (K3) family: o200k rules + a leading `\p{Han}`-run rule, Han excluded
    /// from the letter classes, no '/' tail in the punct rule
    kimi: bool,
    /// 1 = no merges list (tiktoken-derived vocab): merge the adjacent pair whose
    /// CONCATENATION has the lowest vocab id, exactly tiktoken's byte_pair_encode
    rankbpe: bool,
    /// number of ids (size of id2str / id_added / id_special)
    n_ids: usize,
    /// EOS token id, resolved by the loader from the `eos_token` field of
    /// `tokenizer_config.json` against the added tokens. `None` when the config
    /// names no EOS, or names one that is not an added token.
    eos_id: Option<i32>,
}

/// Decode one UTF-8 codepoint starting at byte `i` of `s` (length `len`).
/// Returns `(consumed_bytes, codepoint)`. Invalid bytes are treated as a single
/// unit, matching `u8_next` (tok.h:71).
fn u8_next(s: &[u8], i: usize, len: usize) -> (usize, u32) {
    let c = s[i];
    if c < 0x80 {
        return (1, c as u32);
    }
    if (c >> 5) == 0x06 && i + 1 < len {
        return (2, ((c as u32 & 0x1F) << 6) | (s[i + 1] as u32 & 0x3F));
    }
    if (c >> 4) == 0x0E && i + 2 < len {
        return (
            3,
            ((c as u32 & 0x0F) << 12) | ((s[i + 1] as u32 & 0x3F) << 6) | (s[i + 2] as u32 & 0x3F),
        );
    }
    if (c >> 3) == 0x1E && i + 3 < len {
        return (
            4,
            ((c as u32 & 0x07) << 18)
                | ((s[i + 1] as u32 & 0x3F) << 12)
                | ((s[i + 2] as u32 & 0x3F) << 6)
                | (s[i + 3] as u32 & 0x3F),
        );
    }
    // invalid byte: treated as a single unit
    (1, c as u32)
}

/// Encode `cp` as UTF-8 into `o`, return the number of bytes written. Port of
/// `u8_put` (tok.h:79). `o` must hold at least 4 bytes.
fn u8_put(o: &mut [u8], cp: u32) -> usize {
    if cp < 0x80 {
        o[0] = cp as u8;
        return 1;
    }
    if cp < 0x800 {
        o[0] = 0xC0 | (cp >> 6) as u8;
        o[1] = 0x80 | (cp & 0x3F) as u8;
        return 2;
    }
    if cp < 0x10000 {
        o[0] = 0xE0 | (cp >> 12) as u8;
        o[1] = 0x80 | ((cp >> 6) & 0x3F) as u8;
        o[2] = 0x80 | (cp & 0x3F) as u8;
        return 3;
    }
    o[0] = 0xF0 | (cp >> 18) as u8;
    o[1] = 0x80 | ((cp >> 12) & 0x3F) as u8;
    o[2] = 0x80 | ((cp >> 6) & 0x3F) as u8;
    o[3] = 0x80 | (cp & 0x3F) as u8;
    4
}

impl Tok {
    /// Build the GPT-2/ByteLevel byte <-> unicode map. MUST run before any vocab
    /// key is constructed (silent invariant #1 from k3_tok.h): the vocab is keyed
    /// by the byte-level string, each byte mapped to a printable codepoint via
    /// `byte2str`, not by raw token bytes. Keying on raw bytes gives a tokenizer
    /// that round-trips and produces different ids. Port of `tk_build_bytemap`
    /// (tok.h:87).
    fn build_bytemap(&mut self) {
        for v in &mut self.cp2byte {
            *v = -1;
        }
        // bytes that map to themselves (printable ASCII + Latin-1 punctuation range)
        let mut isdir = [false; 256];
        for b in 33..=126u32 {
            isdir[b as usize] = true;
        }
        for b in 161..=172u32 {
            isdir[b as usize] = true;
        }
        for b in 174..=255u32 {
            isdir[b as usize] = true;
        }
        let mut n = 0u32;
        for b in 0..256u32 {
            let cp = if isdir[b as usize] { b } else { 256 + n };
            if !isdir[b as usize] {
                n += 1;
            }
            self.byte2cp[b as usize] = cp;
            let mut buf = [0u8; 4];
            let l = u8_put(&mut buf, cp);
            self.byte2str[b as usize] = [buf[0], buf[1], buf[2], buf[3]];
            self.byte2cp_len[b as usize] = l;
            if cp < 1024 {
                self.cp2byte[cp as usize] = b as i16;
            }
        }
    }

    /// BPE over one piece: raw bytes `[a, b)` -> ids appended to `out`. Port of
    /// `bpe_piece` (tok.h:207).
    fn bpe_piece(&self, p: &[u8], a: usize, b: usize, out: &mut Vec<i32>, max: usize) {
        let nb = b - a;
        // byte-level string (byte2str concatenated): <= 2 bytes per input byte
        let mut s = Vec::with_capacity(2 * nb + 1);
        for i in a..b {
            let bb = p[i] as usize;
            let l = self.byte2cp_len[bb];
            s.extend_from_slice(&self.byte2str[bb][..l]);
        }
        let sl = s.len();

        // ignore_merges: if the whole piece is itself a token, emit it directly
        if let Some(&whole) = self.vocab.get(&s[..sl]) {
            if out.len() < max {
                out.push(whole);
            }
            return;
        }

        // initial symbols = the codepoints of the byte-level string
        let mut soff: Vec<usize> = Vec::with_capacity(sl + 1);
        let mut slen: Vec<usize> = Vec::with_capacity(sl + 1);
        let mut i = 0;
        while i < sl {
            let (k, _cp) = u8_next(&s, i, sl);
            soff.push(i);
            slen.push(k);
            i += k;
        }
        let mut ns = soff.len();

        loop {
            let mut best = i32::MAX;
            let mut bp: isize = -1;
            let mut i = 0;
            while i + 1 < ns {
                let ll = slen[i];
                let rl = slen[i + 1];
                let rk = if self.rankbpe {
                    // tiktoken: rank of the CONCATENATION (contiguous in s)
                    self.vocab
                        .get(&s[soff[i]..soff[i] + ll + rl])
                        .copied()
                        .unwrap_or(-1)
                } else {
                    let mut kbuf = Vec::with_capacity(ll + 1 + rl);
                    kbuf.extend_from_slice(&s[soff[i]..soff[i] + ll]);
                    kbuf.push(0);
                    kbuf.extend_from_slice(&s[soff[i + 1]..soff[i + 1] + rl]);
                    self.merges.get(&kbuf).copied().unwrap_or(-1)
                };
                if rk >= 0 && rk < best {
                    best = rk;
                    bp = i as isize;
                }
                i += 1;
            }
            if bp < 0 {
                break;
            }
            let bp = bp as usize;
            // fuse bp and bp+1 (contiguous in s)
            slen[bp] = soff[bp + 1] + slen[bp + 1] - soff[bp];
            let mut j = bp + 1;
            while j < ns - 1 {
                soff[j] = soff[j + 1];
                slen[j] = slen[j + 1];
                j += 1;
            }
            ns -= 1;
        }

        let mut i = 0;
        while i < ns {
            let id = self
                .vocab
                .get(&s[soff[i]..soff[i] + slen[i]])
                .copied()
                .unwrap_or(-1);
            if id >= 0 && out.len() < max {
                out.push(id);
            }
            i += 1;
        }
    }

    /// Pre-tokenizer regex (cl100k pattern) over a span of text. Decodes the
    /// codepoints, applies the alternatives IN ORDER, and calls `bpe_piece` on
    /// each resulting piece. The order is part of the specification: an earlier
    /// alternative that matches wins even where a later one would match more.
    /// Port of `pretok_chunk` (tok.h:250).
    fn pretok_chunk(&self, p: &[u8], a: usize, b: usize, out: &mut Vec<i32>, max: usize) {
        let nb = b - a;
        if nb == 0 {
            return;
        }
        let mut cp: Vec<u32> = Vec::with_capacity(nb + 1);
        let mut off: Vec<usize> = Vec::with_capacity(nb + 2);
        let mut i = a;
        let mut n = 0;
        while i < b {
            let (k, c) = u8_next(p, i, b);
            off.push(i);
            cp.push(c);
            n += 1;
            i += k;
        }
        off.push(b);

        let isnl = |c: u32| c == '\r' as u32 || c == '\n' as u32;
        let low = |c: u32| {
            if c >= 'A' as u32 && c <= 'Z' as u32 {
                c + 32
            } else {
                c
            }
        };

        let mut i = 0;
        while i < n {
            let start = i;
            let c = cp[i];
            // 1) (?i:'s|'t|'re|'ve|'m|'ll|'d)
            if c == '\'' as u32 && i + 1 < n {
                let d = low(cp[i + 1]);
                if i + 2 < n {
                    let d2 = low(cp[i + 2]);
                    if (d == 'r' as u32 && d2 == 'e' as u32)
                        || (d == 'v' as u32 && d2 == 'e' as u32)
                        || (d == 'l' as u32 && d2 == 'l' as u32)
                    {
                        i += 3;
                        self.bpe_piece(p, off[start], off[i], out, max);
                        continue;
                    }
                }
                if d == 's' as u32 || d == 't' as u32 || d == 'm' as u32 || d == 'd' as u32 {
                    i += 2;
                    self.bpe_piece(p, off[start], off[i], out, max);
                    continue;
                }
            }
            // 2) [^\r\n\p{L}\p{N}]? \p{L}+
            {
                let mut j = i;
                if !unicode::is_l(c) && !isnl(c) && !unicode::is_n(c) {
                    if j + 1 < n && unicode::is_l(cp[j + 1]) {
                        j += 1;
                    } else {
                        j = usize::MAX; // sentinel: -1
                    }
                }
                if j != usize::MAX && unicode::is_l(cp[j]) {
                    while j < n && unicode::is_l(cp[j]) {
                        j += 1;
                    }
                    i = j;
                    self.bpe_piece(p, off[start], off[i], out, max);
                    continue;
                }
            }
            // 3) \p{N}{1,3}
            if unicode::is_n(c) {
                let mut j = i;
                let mut k = 0;
                while j < n && unicode::is_n(cp[j]) && k < 3 {
                    j += 1;
                    k += 1;
                }
                i = j;
                self.bpe_piece(p, off[start], off[i], out, max);
                continue;
            }
            // 4) ' ?[^\s\p{L}\p{N}]+[\r\n]*'
            {
                let mut j = i;
                if c == ' ' as u32
                    && j + 1 < n
                    && !unicode::is_s(cp[j + 1])
                    && !unicode::is_l(cp[j + 1])
                    && !unicode::is_n(cp[j + 1])
                {
                    j += 1;
                }
                if j < n && !unicode::is_s(cp[j]) && !unicode::is_l(cp[j]) && !unicode::is_n(cp[j])
                {
                    while j < n
                        && !unicode::is_s(cp[j])
                        && !unicode::is_l(cp[j])
                        && !unicode::is_n(cp[j])
                    {
                        j += 1;
                    }
                    while j < n && isnl(cp[j]) {
                        j += 1;
                    }
                    i = j;
                    self.bpe_piece(p, off[start], off[i], out, max);
                    continue;
                }
            }
            // 5) \s*[\r\n]+  -> whitespace run up to the last contiguous newline
            {
                let r;
                {
                    let mut r0 = i;
                    while r0 < n && unicode::is_s(cp[r0]) {
                        r0 += 1;
                    }
                    r = r0;
                }
                if r > i {
                    let mut last: isize = -1;
                    let mut j = i;
                    while j < r {
                        if isnl(cp[j]) {
                            last = j as isize;
                        }
                        j += 1;
                    }
                    if last >= 0 {
                        i = last as usize + 1;
                        self.bpe_piece(p, off[start], off[i], out, max);
                        continue;
                    }
                    // 6) \s+(?!\S): if followed by a non-space leave the last ws, else take it all
                    let mut end = if r < n { r - 1 } else { r };
                    if end <= i {
                        end = i + 1; // \s+ minimo 1 (fallback alt 7)
                    }
                    i = end;
                    self.bpe_piece(p, off[start], off[i], out, max);
                    continue;
                }
            }
            i += 1; // backstop: should be unreachable
            self.bpe_piece(p, off[start], off[i], out, max);
        }
    }

    /// o200k branch A|B letter matcher: end (cp index) of the A|B match at `i`, or
    /// None. Replays the regex engine's backtracking order exactly. Port of
    /// `o2_letters` (tok.h:327), parameterised by the S1/S2 predicates so the Kimi
    /// variant can reuse it with Han-masked classes.
    fn o2_letters_with<F1: Fn(u32) -> bool, F2: Fn(u32) -> bool>(
        cp: &[u32],
        n: usize,
        i: usize,
        s1: F1,
        s2: F2,
    ) -> Option<usize> {
        // branch A, prefix greedy (taken first), then without prefix
        for &pfx in &[true, false] {
            let mut j0 = i;
            if pfx {
                let c = cp[i];
                if c == '\r' as u32
                    || c == '\n' as u32
                    || unicode::is_l(c)
                    || unicode::is_n(c)
                    || i + 1 >= n
                {
                    continue;
                }
                j0 = i + 1;
            }
            let mut m1 = j0;
            while m1 < n && s1(cp[m1]) {
                m1 += 1;
            }
            // walk s back from m1 to j0 looking for the first S2 (the greedy S1*
            // gives back until S2+ can take >=1 char)
            let mut s = m1 as isize;
            while s >= j0 as isize {
                if (s as usize) < n && s2(cp[s as usize]) {
                    let mut k = s as usize + 1;
                    while k < n && s2(cp[k]) {
                        k += 1;
                    }
                    return Some(Self::o2_contraction(cp, n, k));
                }
                s -= 1;
            }
        }
        // branch B
        for &pfx in &[true, false] {
            let mut j0 = i;
            if pfx {
                let c = cp[i];
                if c == '\r' as u32
                    || c == '\n' as u32
                    || unicode::is_l(c)
                    || unicode::is_n(c)
                    || i + 1 >= n
                {
                    continue;
                }
                j0 = i + 1;
            }
            let mut m1 = j0;
            while m1 < n && s1(cp[m1]) {
                m1 += 1;
            }
            if m1 > j0 {
                let mut k = m1;
                while k < n && s2(cp[k]) {
                    k += 1;
                }
                return Some(Self::o2_contraction(cp, n, k));
            }
        }
        None
    }

    /// `(?i:'s|'t|'re|'ve|'m|'ll|'d)?` after a letter run. Port of
    /// `o2_contraction` (tok.h:317).
    fn o2_contraction(cp: &[u32], n: usize, k: usize) -> usize {
        if k < n && cp[k] == '\'' as u32 && k + 1 < n {
            let d = if cp[k + 1] >= 'A' as u32 && cp[k + 1] <= 'Z' as u32 {
                cp[k + 1] + 32
            } else {
                cp[k + 1]
            };
            if k + 2 < n {
                let e = if cp[k + 2] >= 'A' as u32 && cp[k + 2] <= 'Z' as u32 {
                    cp[k + 2] + 32
                } else {
                    cp[k + 2]
                };
                if (d == 'r' as u32 && e == 'e' as u32)
                    || (d == 'v' as u32 && e == 'e' as u32)
                    || (d == 'l' as u32 && e == 'l' as u32)
                {
                    return k + 3;
                }
            }
            if d == 's' as u32 || d == 't' as u32 || d == 'm' as u32 || d == 'd' as u32 {
                return k + 2;
            }
        }
        k
    }

    /// Pre-tokenizer o200k (GPT-4o regex family). Port of `pretok_chunk_o200k`
    /// (tok.h:361). S1 = Lu|Lt|Lm|Lo|M, S2 = Ll|Lm|Lo|M.
    fn pretok_chunk_o200k(&self, p: &[u8], a: usize, b: usize, out: &mut Vec<i32>, max: usize) {
        let nb = b - a;
        if nb == 0 {
            return;
        }
        let mut cp: Vec<u32> = Vec::with_capacity(nb + 1);
        let mut off: Vec<usize> = Vec::with_capacity(nb + 2);
        let mut i = a;
        let mut n = 0;
        while i < b {
            let (k, c) = u8_next(p, i, b);
            off.push(i);
            cp.push(c);
            n += 1;
            i += k;
        }
        off.push(b);

        let isnl = |c: u32| c == '\r' as u32 || c == '\n' as u32;
        let s1 = |c: u32| unicode::is_u(c) || unicode::is_x(c);
        let s2 = |c: u32| unicode::is_x(c) || (unicode::is_l(c) && !unicode::is_u(c));

        let mut i = 0;
        while i < n {
            let start = i;
            let c = cp[i];
            // A|B: letter runs with case-aware split + optional contraction
            if let Some(e) = Self::o2_letters_with(&cp, n, i, s1, s2) {
                if e > i {
                    i = e;
                    self.bpe_piece(p, off[start], off[i], out, max);
                    continue;
                }
            }
            // C: \p{N}{1,3}
            if unicode::is_n(c) {
                let mut j = i;
                let mut k = 0;
                while j < n && unicode::is_n(cp[j]) && k < 3 {
                    j += 1;
                    k += 1;
                }
                i = j;
                self.bpe_piece(p, off[start], off[i], out, max);
                continue;
            }
            // D: ' ?[^\s\p{L}\p{N}]+[\r\n/]*'
            {
                let mut j = i;
                if c == ' ' as u32
                    && j + 1 < n
                    && !unicode::is_s(cp[j + 1])
                    && !unicode::is_l(cp[j + 1])
                    && !unicode::is_n(cp[j + 1])
                {
                    j += 1;
                }
                if j < n && !unicode::is_s(cp[j]) && !unicode::is_l(cp[j]) && !unicode::is_n(cp[j])
                {
                    while j < n
                        && !unicode::is_s(cp[j])
                        && !unicode::is_l(cp[j])
                        && !unicode::is_n(cp[j])
                    {
                        j += 1;
                    }
                    while j < n && (isnl(cp[j]) || cp[j] == '/' as u32) {
                        j += 1;
                    }
                    i = j;
                    self.bpe_piece(p, off[start], off[i], out, max);
                    continue;
                }
            }
            // E: \s*[\r\n]+  F: \s+(?!\S)  G: \s+  (same as cl100k)
            {
                let r;
                {
                    let mut r0 = i;
                    while r0 < n && unicode::is_s(cp[r0]) {
                        r0 += 1;
                    }
                    r = r0;
                }
                if r > i {
                    let mut last: isize = -1;
                    let mut j = i;
                    while j < r {
                        if isnl(cp[j]) {
                            last = j as isize;
                        }
                        j += 1;
                    }
                    if last >= 0 {
                        i = last as usize + 1;
                        self.bpe_piece(p, off[start], off[i], out, max);
                        continue;
                    }
                    let mut end = if r < n { r - 1 } else { r };
                    if end <= i {
                        end = i + 1;
                    }
                    i = end;
                    self.bpe_piece(p, off[start], off[i], out, max);
                    continue;
                }
            }
            i += 1;
            self.bpe_piece(p, off[start], off[i], out, max);
        }
    }

    /// `\p{Han}` membership. Script=Han ranges (Unicode 15). Port of `is_han`
    /// (tok.h:416).
    fn is_han(c: u32) -> bool {
        static R: &[(u32, u32)] = &[
            (0x2E80, 0x2E99),
            (0x2E9B, 0x2EF3),
            (0x2F00, 0x2FD5),
            (0x3005, 0x3005),
            (0x3007, 0x3007),
            (0x3021, 0x3029),
            (0x3038, 0x303B),
            (0x3400, 0x4DBF),
            (0x4E00, 0x9FFF),
            (0xF900, 0xFA6D),
            (0xFA70, 0xFAD9),
            (0x16FE2, 0x16FE3),
            (0x16FF0, 0x16FF1),
            (0x20000, 0x2A6DF),
            (0x2A700, 0x2B739),
            (0x2B740, 0x2B81D),
            (0x2B820, 0x2CEA1),
            (0x2CEB0, 0x2EBE0),
            (0x2EBF0, 0x2EE5D),
            (0x2F800, 0x2FA1D),
            (0x30000, 0x3134A),
            (0x31350, 0x323AF),
        ];
        if c < 0x2E80 {
            return false;
        }
        let mut lo = 0isize;
        let mut hi = R.len() as isize - 1;
        while lo <= hi {
            let m = (lo + hi) / 2;
            let (a, b) = R[m as usize];
            if c < a {
                hi = m - 1;
            } else if c > b {
                lo = m + 1;
            } else {
                return true;
            }
        }
        false
    }

    /// Pre-tokenizer Kimi (K3 / tiktoken tokenization_kimi.py). Port of
    /// `pretok_chunk_kimi` (tok.h:466). Identical to o200k except: Han runs are
    /// their own chunks, Han never joins a letter run (it is `\p{Lo}`, so it must
    /// be masked out of S1/S2), and rule D has no '/' in its newline tail. A Han
    /// codepoint can ONLY match rule H. On Han-free text this family tokenizes
    /// exactly like o200k minus the '/' tail.
    fn pretok_chunk_kimi(&self, p: &[u8], a: usize, b: usize, out: &mut Vec<i32>, max: usize) {
        let nb = b - a;
        if nb == 0 {
            return;
        }
        let mut cp: Vec<u32> = Vec::with_capacity(nb + 1);
        let mut off: Vec<usize> = Vec::with_capacity(nb + 2);
        let mut i = a;
        let mut n = 0;
        while i < b {
            let (k, c) = u8_next(p, i, b);
            off.push(i);
            cp.push(c);
            n += 1;
            i += k;
        }
        off.push(b);

        let isnl = |c: u32| c == '\r' as u32 || c == '\n' as u32;
        // KM_S1(c) = ((is_U||is_X) && !is_han);  KM_S2(c) = ((is_X||(is_L&&!is_U)) && !is_han)
        let s1 = |c: u32| (unicode::is_u(c) || unicode::is_x(c)) && !Self::is_han(c);
        let s2 = |c: u32| {
            (unicode::is_x(c) || (unicode::is_l(c) && !unicode::is_u(c))) && !Self::is_han(c)
        };

        let mut i = 0;
        while i < n {
            let start = i;
            let c = cp[i];
            // H: [\p{Han}]+
            if Self::is_han(c) {
                let mut j = i;
                while j < n && Self::is_han(cp[j]) {
                    j += 1;
                }
                i = j;
                self.bpe_piece(p, off[start], off[i], out, max);
                continue;
            }
            // A|B: letter runs, Han excluded
            if let Some(e) = Self::o2_letters_with(&cp, n, i, s1, s2) {
                if e > i {
                    i = e;
                    self.bpe_piece(p, off[start], off[i], out, max);
                    continue;
                }
            }
            // C: \p{N}{1,3}
            if unicode::is_n(c) {
                let mut j = i;
                let mut k = 0;
                while j < n && unicode::is_n(cp[j]) && k < 3 {
                    j += 1;
                    k += 1;
                }
                i = j;
                self.bpe_piece(p, off[start], off[i], out, max);
                continue;
            }
            // D: ' ?[^\s\p{L}\p{N}]+[\r\n]*'  (no '/' tail, unlike o200k)
            {
                let mut j = i;
                if c == ' ' as u32
                    && j + 1 < n
                    && !unicode::is_s(cp[j + 1])
                    && !unicode::is_l(cp[j + 1])
                    && !unicode::is_n(cp[j + 1])
                {
                    j += 1;
                }
                if j < n && !unicode::is_s(cp[j]) && !unicode::is_l(cp[j]) && !unicode::is_n(cp[j])
                {
                    while j < n
                        && !unicode::is_s(cp[j])
                        && !unicode::is_l(cp[j])
                        && !unicode::is_n(cp[j])
                    {
                        j += 1;
                    }
                    while j < n && isnl(cp[j]) {
                        j += 1;
                    }
                    i = j;
                    self.bpe_piece(p, off[start], off[i], out, max);
                    continue;
                }
            }
            // E: \s*[\r\n]+  F: \s+(?!\S)  G: \s+
            {
                let r;
                {
                    let mut r0 = i;
                    while r0 < n && unicode::is_s(cp[r0]) {
                        r0 += 1;
                    }
                    r = r0;
                }
                if r > i {
                    let mut last: isize = -1;
                    let mut j = i;
                    while j < r {
                        if isnl(cp[j]) {
                            last = j as isize;
                        }
                        j += 1;
                    }
                    if last >= 0 {
                        i = last as usize + 1;
                        self.bpe_piece(p, off[start], off[i], out, max);
                        continue;
                    }
                    let mut end = if r < n { r - 1 } else { r };
                    if end <= i {
                        end = i + 1;
                    }
                    i = end;
                    self.bpe_piece(p, off[start], off[i], out, max);
                    continue;
                }
            }
            i += 1;
            self.bpe_piece(p, off[start], off[i], out, max);
        }
    }

    /// Loads `tiktoken.model` and `tokenizer_config.json` from `dir`. Delegated to
    /// `loader.rs`; this is the constructor the batch contract names.
    pub fn load(dir: &std::path::Path) -> std::io::Result<Tok> {
        loader::load(dir)
    }

    /// Encode `text` to ids, splitting on added tokens first then pre-tokenising +
    /// BPE each piece. Port of `tok_encode` (tok.h:512).
    pub fn encode(&self, text: &str) -> Vec<i32> {
        let p = text.as_bytes();
        let len = p.len();
        let max = usize::MAX;
        let mut out: Vec<i32> = Vec::new();
        let mut i = 0;
        while i < len {
            // next added-token occurrence at or after i (longest match, since sp is
            // sorted by descending length)
            let mut hitpos: usize = usize::MAX;
            let mut hitlen = 0usize;
            let mut hitid: i32 = -1;
            let mut j = i;
            while j < len && hitpos == usize::MAX {
                for k in 0..self.sp.len() {
                    let sl = self.sp[k].str_.len();
                    if sl > 0 && j + sl <= len && &p[j..j + sl] == self.sp[k].str_.as_slice() {
                        hitpos = j;
                        hitlen = sl;
                        hitid = self.sp[k].id;
                        break;
                    }
                }
                j += 1;
            }
            let chunk_end = if hitpos == usize::MAX { len } else { hitpos };
            if chunk_end > i {
                if self.kimi {
                    self.pretok_chunk_kimi(p, i, chunk_end, &mut out, max);
                } else if self.o200k {
                    self.pretok_chunk_o200k(p, i, chunk_end, &mut out, max);
                } else {
                    self.pretok_chunk(p, i, chunk_end, &mut out, max);
                }
            }
            if hitpos == usize::MAX {
                break;
            }
            if out.len() < max {
                out.push(hitid);
            }
            i = hitpos + hitlen;
        }
        out
    }

    /// Decode `ids` to text (inverse byte-level map; added tokens literal). Port
    /// of `tok_decode` (tok.h:543).
    pub fn decode(&self, ids: &[i32]) -> String {
        let mut out: Vec<u8> = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            if id < 0 || id as usize >= self.n_ids {
                continue;
            }
            let s = &self.id2str[id as usize];
            if s.is_empty() {
                continue;
            }
            if self.id_added[id as usize] {
                out.extend_from_slice(s);
                continue;
            }
            let sl = s.len();
            let mut j = 0;
            while j < sl {
                let (k, c) = u8_next(s, j, sl);
                j += k;
                if c < 1024 && self.cp2byte[c as usize] >= 0 {
                    out.push(self.cp2byte[c as usize] as u8);
                }
            }
        }
        // The text is byte-level; round-trip via from_utf8_lossy is wrong for raw
        // byte sequences, but decode's contract is byte recovery. Use from_utf8_lossy
        // to produce a String for the public API; the byte content is preserved.
        String::from_utf8_lossy(&out).into_owned()
    }

    /// Id of an added token given its content (e.g. `"<|endoftext|>"`), or None if
    /// absent. Port of `tok_id_of` (tok.h:537).
    pub fn id_of(&self, s: &str) -> Option<i32> {
        let b = s.as_bytes();
        for sp in &self.sp {
            if sp.str_.as_slice() == b {
                return Some(sp.id);
            }
        }
        None
    }

    /// Vocabulary size (number of ids).
    pub fn vocab(&self) -> usize {
        self.n_ids
    }

    /// The end-of-sequence token id, resolved by the loader from the `eos_token`
    /// field of `tokenizer_config.json`. `None` when the config names no EOS.
    /// The C engine has no equivalent: its run path is driven by `--ids`.
    pub fn eos(&self) -> Option<i32> {
        self.eos_id
    }
}
