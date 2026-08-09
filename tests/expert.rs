// SPDX-License-Identifier: Apache-2.0
//! Load real routed experts from the real checkpoint. Port of `tests/unit/test_expert.c`.
//!
//! `test_ops` already proves the dequantiser is bit-exact against `fixtures/mxfp4.json`,
//! but that fixture was built from bytes pulled over HTTP. It proves the dequantiser, not
//! the reader. This closes the last link: it reads the same expert out of a local shard
//! through `st` and `load`, checks the geometry and contiguity across the whole bank, and
//! asserts the fused MXFP4 matmul agrees with dequantise-then-multiply.
//!
//! THE FUSED MATMUL IS THE ONLY CORRECTNESS ASSERTION HERE, and it must assert rather
//! than merely print: a broken nibble order or a mis-biased E8M0 scale once printed
//! "DISAGREE <-- BUG" while the process still exited 0. test_expert.c:214-219.
//!
//! The two implementations accumulate in a different ORDER (the fused version sums within
//! a 32-element group before applying the shared scale), so the comparison is against
//! double-precision rounding at 1e-6 relative, not against zero.
//!
//! The C program's three I/O timing regimes (sequential cold, random cold, warm) are not
//! ported: they need `/proc/sys/vm/drop_caches` and root, and they measure the storage
//! device rather than this code. `benches/kernels.rs` covers the compute side.
//!
//! Needs the released checkpoint. Run with:
//!   K3_SHARD_DIR=/path/to/shards cargo test --test expert -- --ignored --nocapture

use k3::cfg::MXFP4_GROUP;
use k3::load::{expert_load, expert_ref, qmat_numel};
use k3::ops::{matmul, matmul_mxfp4, mxfp4_dequant};
use k3::st::St;
use std::path::Path;

/// xorshift, so the activation vector is reproducible across runs and machines.
/// test_expert.c:63.
struct Rs(u32);
impl Rs {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
}

#[test]
#[ignore = "needs the real 1.56 TB checkpoint; set K3_SHARD_DIR"]
fn expert_streaming_matches_dequantised() {
    let Some(dir) = std::env::var_os("K3_SHARD_DIR") else {
        eprintln!("  NOT RUN: K3_SHARD_DIR is unset, so there is no checkpoint to read.");
        eprintln!(
            "           Run: K3_SHARD_DIR=/path/to/shards cargo test --test expert -- --ignored"
        );
        return;
    };
    let dir = Path::new(&dir);
    let layer: usize = std::env::var("K3_EXPERT_LAYER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);

    let s = St::open(dir).expect("open shard directory");
    println!(
        "indexed {} tensors from {} shard(s)\n",
        s.tensors().len(),
        s.nshard()
    );

    // How many experts does this layer actually have? test_expert.c:79.
    let mut have = 0usize;
    while have < 1024 {
        let nm = format!(
            "language_model.model.layers.{}.block_sparse_moe.experts.{}.w1.weight_packed",
            layer, have
        );
        if s.find(&nm).is_none() {
            break;
        }
        have += 1;
    }
    println!("layer {layer} has {have} routed experts present in this shard set");
    assert!(
        have > 0,
        "no experts for layer {layer} in {}",
        dir.display()
    );

    // Geometry and contiguity, checked across every expert. test_expert.c:92.
    let r0 = expert_ref(&s, layer, 0).expect("resolve expert 0");
    println!("expert 0 geometry:");
    for (i, name) in ["w1", "w2", "w3"].iter().enumerate() {
        let m = &r0.m[i];
        println!(
            "  {} packed [{}][{}] -> logical [{}][{}], scales [{}][{}], group {}",
            name,
            m.rows,
            m.pcols,
            m.rows,
            m.pcols * 2,
            m.rows,
            m.scols,
            MXFP4_GROUP
        );
    }
    println!(
        "  run: shard {}, offset {}, {} bytes, contiguous {}",
        r0.shard,
        r0.off,
        r0.nbytes,
        if r0.contiguous { "YES" } else { "NO" }
    );

    let params: i64 = r0.m.iter().map(qmat_numel).sum();
    println!(
        "  {} parameters in {} bytes = {:.6} bytes/param (MXFP4 predicts {:.6})",
        params,
        r0.nbytes,
        r0.nbytes as f64 / params as f64,
        0.5 + 1.0 / 32.0
    );

    let mut noncontig = 0usize;
    let mut badgeom = 0usize;
    for e in 0..have {
        match expert_ref(&s, layer, e) {
            Ok(r) => {
                if !r.contiguous {
                    noncontig += 1;
                }
                if r.nbytes != r0.nbytes {
                    badgeom += 1;
                }
            }
            Err(_) => badgeom += 1,
        }
    }
    println!(
        "  across all {} experts: {} non-contiguous, {} with unexpected geometry{}\n",
        have,
        noncontig,
        badgeom,
        if noncontig > 0 || badgeom > 0 {
            "   <-- INVESTIGATE"
        } else {
            ""
        }
    );

    // A non-uniform expert bank is a legitimate failure in its own right: the streaming
    // cache sizes every slot from one probe and assumes the rest match. test_expert.c:120.
    assert_eq!(
        badgeom, 0,
        "{badgeom} of {have} experts do not match expert 0's geometry. The expert cache \
         sizes all slots from a single probe, so a non-uniform bank cannot be streamed \
         safely."
    );

    let mut buf = vec![0u8; r0.nbytes as usize];
    let got = expert_load(&s, &r0, &mut buf).expect("load expert 0");
    assert_eq!(got, r0.nbytes, "short read on expert 0");

    // The fused MXFP4 matmul must equal dequantise-then-multiply. test_expert.c:184.
    let rows = r0.m[0].rows;
    let wid = r0.m[0].pcols * 2;
    let p_off = r0.m[0].p_off as usize;
    let s_off = r0.m[0].s_off as usize;
    let p_bytes = r0.m[0].p_bytes as usize;
    let s_bytes = r0.m[0].s_bytes as usize;

    let mut rs = Rs(12345);
    let mut x = vec![0.0f32; wid];
    for v in x.iter_mut() {
        *v = (rs.next() >> 8) as f32 / 8_388_608.0 - 1.0;
    }

    let mut w = vec![0.0f32; rows * wid];
    let mut ya = vec![0.0f32; rows];
    let mut yb = vec![0.0f32; rows];

    let t1 = std::time::Instant::now();
    mxfp4_dequant(
        &mut w,
        &buf[p_off..p_off + p_bytes],
        &buf[s_off..s_off + s_bytes],
        rows,
        r0.m[0].pcols,
        MXFP4_GROUP,
    );
    matmul(&mut ya, &x, &w, wid);
    let t_deq = t1.elapsed().as_secs_f64();

    let t1 = std::time::Instant::now();
    matmul_mxfp4(
        &mut yb,
        &x,
        &buf[p_off..p_off + p_bytes],
        &buf[s_off..s_off + s_bytes],
        wid,
        MXFP4_GROUP,
    );
    let t_fus = t1.elapsed().as_secs_f64();

    let mut maxabs = 0.0f64;
    let mut scale = 0.0f64;
    for i in 0..rows {
        let d = (ya[i] as f64 - yb[i] as f64).abs();
        if d > maxabs {
            maxabs = d;
        }
        if (ya[i] as f64).abs() > scale {
            scale = (ya[i] as f64).abs();
        }
    }
    let maxrel = if scale > 0.0 { maxabs / scale } else { 0.0 };

    println!("fused MXFP4 matmul vs dequantise-then-matmul, w1 [{rows}][{wid}]:");
    println!(
        "  max abs diff {:.3e}, relative to max |y| {:.3e}   {}",
        maxabs,
        maxrel,
        if maxrel < 1e-6 {
            "AGREE"
        } else {
            "DISAGREE  <-- BUG"
        }
    );
    println!(
        "  dequant+matmul {:.1} ms (plus {:.1} MB materialised), fused {:.1} ms",
        t_deq * 1000.0,
        (rows * wid * 4) as f64 / 1e6,
        t_fus * 1000.0
    );
    println!(
        "  fused reads {:.2} MB instead of {:.2} MB: {:.1}x less memory traffic\n",
        (p_bytes + s_bytes) as f64 / 1e6,
        (rows * wid * 4) as f64 / 1e6,
        (rows * wid * 4) as f64 / (p_bytes + s_bytes) as f64
    );

    assert!(
        maxrel < 1e-6,
        "fused MXFP4 matmul disagrees with dequantise-then-matmul: relative {maxrel:.3e}"
    );
}
