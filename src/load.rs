// SPDX-License-Identifier: Apache-2.0
//! Resolve Kimi K3 checkpoint tensors and stream routed experts.
//!
//! Port of `src/io/k3_load.c` and `k3_load.h`.
//!
//! The six tensors making up one routed expert (w1, w2, w3, each with `weight_packed` and
//! `weight_scale`) are laid out as ONE contiguous run of exactly 17,547,264 bytes, in
//! the order `w1.packed w1.scale w2.packed w2.scale w3.packed w3.scale` with no gaps.
//! That is what makes streaming cheap: fetching an expert is a single coalesced pread,
//! not six scattered ones. `expert_ref` verifies this per expert rather than assuming it.
//!
//! The fallback is not dead code: if a future checkpoint interleaves tensors
//! differently, `expert_ref` reports `contiguous = false` and `expert_load` issues six
//! preads into the same buffer layout. The result is identical; only the seek count
//! changes.

use std::io;

use crate::cfg::MXFP4_GROUP;
use crate::st::St;

/// Geometry of one quantised matrix inside an expert's byte run.
#[derive(Clone, Copy, Debug, Default)]
pub struct QMat {
    /// Packed nibbles, offset WITHIN the run.
    pub p_off: i64,
    pub p_bytes: i64,
    /// E8M0 scales, offset WITHIN the run.
    pub s_off: i64,
    pub s_bytes: i64,
    /// Packed is `[rows][pcols]`; logical width is `pcols*2`.
    pub rows: usize,
    pub pcols: usize,
    /// Scales are `[rows][scols]`, `scols == pcols*2/32`.
    pub scols: usize,
}

/// One routed expert: its six tensors resolved and validated.
#[derive(Clone, Debug)]
pub struct ExpertRef {
    pub layer: usize,
    pub expert: usize,
    pub shard: usize,
    /// Absolute file offset of the run start.
    pub off: i64,
    /// Total bytes, 17,547,264 for real K3.
    pub nbytes: i64,
    pub contiguous: bool,
    /// w1, w2, w3 in that order.
    pub m: [QMat; 3],
}

/// The name template, matching `EXPERT_FMT` in k3_load.c:11.
const EXPERT_FMT: &str = "language_model.model.layers.{L}.block_sparse_moe.experts.{E}.{W}.{S}";

/// The three weight names, in the order the C code uses. k3_load.c:17.
const W: [&str; 3] = ["w1", "w2", "w3"];

/// Elements in each dequantised matrix, for sizing the destination buffers.
/// k3_load.c:13.
pub fn qmat_numel(m: &QMat) -> i64 {
    (m.rows as i64) * (m.pcols as i64) * 2
}

/// Resolve the six tensors of one expert and validate their geometry. k3_load.c:15.
pub fn expert_ref(s: &St, layer: usize, expert: usize) -> io::Result<ExpertRef> {
    let mut pk = [None, None, None];
    let mut sc = [None, None, None];

    let mut r = ExpertRef {
        layer,
        expert,
        shard: 0,
        off: 0,
        nbytes: 0,
        contiguous: false,
        m: [QMat::default(); 3],
    };

    for i in 0..3 {
        let name = format_name(layer, expert, W[i], "weight_packed");
        let pki = s.find(&name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("k3_load: missing {}", name),
            )
        })?;
        let name = format_name(layer, expert, W[i], "weight_scale");
        let sci = s.find(&name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("k3_load: missing {}", name),
            )
        })?;
        pk[i] = Some(pki);
        sc[i] = Some(sci);

        let pki = pk[i].unwrap();
        let sci = sc[i].unwrap();

        // Both must be U8. k3_load.c:32.
        if pki.dtype != crate::st::Dtype::U8 || sci.dtype != crate::st::Dtype::U8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("k3_load: L{} expert {} {} is not U8", layer, expert, W[i]),
            ));
        }
        // Both must be 2D. k3_load.c:36.
        if pki.shape().len() != 2 || sci.shape().len() != 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("k3_load: L{} expert {} {} is not 2D", layer, expert, W[i]),
            ));
        }
        // Row counts must agree. k3_load.c:40.
        if pki.shape()[0] != sci.shape()[0] {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "k3_load: L{} expert {} {} row mismatch {} vs {}",
                    layer,
                    expert,
                    W[i],
                    pki.shape()[0],
                    sci.shape()[0]
                ),
            ));
        }
        // The scale count must equal the logical width divided by the group size. If it
        // does not, the group size is not 32 for this tensor and every scale after the
        // first would be applied to the wrong 32 weights. Silent, and catastrophic.
        // k3_load.c:46.
        let logical = pki.shape()[1] * 2;
        if sci.shape()[1] * MXFP4_GROUP as i64 != logical {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "k3_load: L{} expert {} {}: {} scales for {} elements implies group size {:.2}, not {}",
                    layer,
                    expert,
                    W[i],
                    sci.shape()[1],
                    logical,
                    if sci.shape()[1] != 0 {
                        logical as f64 / sci.shape()[1] as f64
                    } else {
                        0.0
                    },
                    MXFP4_GROUP
                ),
            ));
        }
        // Packed and scale must live in the same shard. k3_load.c:57.
        if pki.shard != sci.shard {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "k3_load: L{} expert {} {} is split across shards",
                    layer, expert, W[i]
                ),
            ));
        }
        // All three matrices must live in the same shard. k3_load.c:62.
        if i > 0 && pki.shard != pk[0].unwrap().shard {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("k3_load: L{} expert {} spans shards", layer, expert),
            ));
        }

        r.m[i].rows = pki.shape()[0] as usize;
        r.m[i].pcols = pki.shape()[1] as usize;
        r.m[i].scols = sci.shape()[1] as usize;
        r.m[i].p_bytes = pki.nbytes;
        r.m[i].s_bytes = sci.nbytes;
    }

    r.shard = pk[0].unwrap().shard;

    // Is the whole expert one run? Take the lowest offset as the base and check that the
    // six spans tile it exactly. k3_load.c:76.
    let all = [
        pk[0].unwrap(),
        sc[0].unwrap(),
        pk[1].unwrap(),
        sc[1].unwrap(),
        pk[2].unwrap(),
        sc[2].unwrap(),
    ];
    let mut lo = all[0].off;
    let mut hi: i64 = 0;
    let mut own: i64 = 0;
    for t in &all {
        if t.off < lo {
            lo = t.off;
        }
        if t.off + t.nbytes > hi {
            hi = t.off + t.nbytes;
        }
        own += t.nbytes;
    }
    r.contiguous = hi - lo == own;

    if r.contiguous {
        // k3_load.c:87.
        r.off = lo;
        r.nbytes = own;
        for i in 0..3 {
            r.m[i].p_off = pk[i].unwrap().off - lo;
            r.m[i].s_off = sc[i].unwrap().off - lo;
        }
    } else {
        // Pack sequentially in the canonical order; six preads will fill it.
        // k3_load.c:94.
        let mut o: i64 = 0;
        for i in 0..3 {
            r.m[i].p_off = o;
            o += r.m[i].p_bytes;
            r.m[i].s_off = o;
            o += r.m[i].s_bytes;
        }
        r.off = lo;
        r.nbytes = o;
    }
    Ok(r)
}

/// Fetch the raw bytes. `buf` must hold `r.nbytes`. One pread when contiguous.
/// k3_load.c:107.
pub fn expert_load(s: &St, r: &ExpertRef, buf: &mut [u8]) -> io::Result<i64> {
    if r.contiguous {
        // The whole point: one coalesced read. k3_load.c:109.
        let t = crate::st::Tensor::byte_span("expert", r.shard, r.off, r.nbytes);
        return s.read(&t, buf);
    }

    // Six scattered reads into the packed layout. k3_load.c:123.
    let mut got = 0i64;
    for i in 0..3 {
        let name = format_name(r.layer, r.expert, W[i], "weight_packed");
        let p = s.find(&name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("k3_load: missing {}", name),
            )
        })?;
        let name = format_name(r.layer, r.expert, W[i], "weight_scale");
        let c = s.find(&name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("k3_load: missing {}", name),
            )
        })?;
        got += s.read(p, &mut buf[r.m[i].p_off as usize..])?;
        got += s.read(c, &mut buf[r.m[i].s_off as usize..])?;
    }
    Ok(got)
}

/// As `expert_load` but with O_DIRECT, bypassing the page cache. Only valid for a
/// contiguous expert; falls back to `expert_load` (payload_off 0) otherwise.
///
/// Returns `(payload_bytes, payload_off)`. k3_load.c:138.
pub fn expert_load_direct(s: &St, r: &ExpertRef, buf: &mut [u8]) -> io::Result<(i64, i64)> {
    if !r.contiguous {
        return expert_load(s, r, buf).map(|g| (g, 0));
    }
    s.read_aligned(r.shard, r.off, r.nbytes, buf)
}

/// Dequantise from a buffer filled by `expert_load`. Each output must hold
/// `rows * pcols * 2` floats. Any of the three may be `None` to skip it. k3_load.c:148.
pub fn expert_dequant(
    r: &ExpertRef,
    buf: &[u8],
    w1: Option<&mut [f32]>,
    w2: Option<&mut [f32]>,
    w3: Option<&mut [f32]>,
) {
    dequant_one(r, buf, 0, w1);
    dequant_one(r, buf, 1, w2);
    dequant_one(r, buf, 2, w3);
}

/// Dequantise one of the three matrices; a no-op when its output is `None`.
fn dequant_one(r: &ExpertRef, buf: &[u8], i: usize, out: Option<&mut [f32]>) {
    if let Some(out) = out {
        let packed = &buf[r.m[i].p_off as usize..(r.m[i].p_off + r.m[i].p_bytes) as usize];
        let scales = &buf[r.m[i].s_off as usize..(r.m[i].s_off + r.m[i].s_bytes) as usize];
        crate::ops::mxfp4_dequant(out, packed, scales, r.m[i].rows, r.m[i].pcols, MXFP4_GROUP);
    }
}

/// Build one of the six tensor names. Matches `snprintf(name, ..., EXPERT_FMT, ...)`.
fn format_name(layer: usize, expert: usize, w: &str, suffix: &str) -> String {
    EXPERT_FMT
        .replace("{L}", &layer.to_string())
        .replace("{E}", &expert.to_string())
        .replace("{W}", w)
        .replace("{S}", suffix)
}
