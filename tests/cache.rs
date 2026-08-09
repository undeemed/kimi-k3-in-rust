// SPDX-License-Identifier: Apache-2.0
//! The streaming expert cache, which nothing else tests. Port of
//! `tests/unit/test_cache.c`, run against `fixtures/cache` as the `test-cache` Makefile
//! recipe does.
//!
//! WHY THIS FILE EXISTS
//!
//! Every op fixture and all three oracle gates drive the RESIDENT expert bank
//! (`MoeW.w1/w3/w2`).
//! The streaming path (`ExpertSrc` -> `Cache`) is reached only when running the real
//! 1.56 TB checkpoint.
//! So the component the entire project depends on had no test at all, and it showed: a
//! batch-prefetch bug that handed one slot to several experts simultaneously passed 22/22
//! fixtures and all three gates, and was caught only because the real model emitted token
//! 65 where it should have emitted 2494.
//! A test that cannot fail is worse than no test; a path with no test is worse still.
//!
//! WHAT IS CHECKED
//!
//! 1. IDENTITY - every expert read back through the cache is byte-identical to the same
//!    expert read straight off disk. This is the check the aliasing bug fails.
//! 2. EQUIVALENCE - the batch prefetch and the serial path return the SAME bytes. The
//!    prefetch is an optimisation; if it changes a byte it is wrong.
//! 3. PRESSURE - with a cache far smaller than the working set, so eviction runs
//!    constantly and slots are recycled aggressively. The bug only appeared under
//!    pressure, because with a roomy cache `pick_victim` returns genuinely free slots and
//!    the aliasing never happens.
//! 4. ACCOUNTING - requests, hits and `prefetch_reads` stay mutually consistent.

use std::path::{Path, PathBuf};

use k3::cache::Cache;
use k3::cfg::Cfg;
use k3::load::{expert_load, expert_ref};
use k3::ops::{ExpertQ, ExpertSrc};
use k3::st::St;

/// The C harness counts failures in a global and returns 1 if any fired. A case that
/// could not run counts as a FAILURE, never as a skip.
struct Checks {
    fail: usize,
}

impl Checks {
    fn ck(&mut self, ok: bool, what: &str, detail: &str) {
        println!(
            "  {}  {:<34} {}",
            if ok { "PASS" } else { "FAIL" },
            what,
            detail
        );
        if !ok {
            self.fail += 1;
        }
    }
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/cache")
}

/// Everything zero but the three fields the cache reads, matching the C test's
/// `memset(&c, 0, sizeof c)`. `first_dense` of 0 leaves layer 0 routed, which is what the
/// fixture holds.
fn zero_cfg(n_layers: i32, n_experts: i32, topk: i32) -> Cfg {
    Cfg {
        hidden: 0,
        n_layers,
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
        n_experts,
        topk,
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
    }
}

/// Read an expert straight from the store, bypassing the cache entirely. This is the
/// ground truth the cache is measured against.
fn direct_read(st: &St, layer: usize, e: usize) -> Option<Vec<u8>> {
    let r = expert_ref(st, layer, e).ok()?;
    let mut b = vec![0u8; r.nbytes as usize];
    if expert_load(st, &r, &mut b).ok()? != r.nbytes {
        return None;
    }
    Some(b)
}

/// The three (packed, scale) pairs the cache hands out must point at bytes equal to the
/// direct read. Compare through the SAME offsets the kernels use, so a wrong pad or a
/// wrong slot base is caught, not just a wrong buffer.
fn same_expert(st: &St, layer: usize, e: usize, q: &ExpertQ<'_>) -> bool {
    let Some(truth) = direct_read(st, layer, e) else {
        return false;
    };
    let Ok(r) = expert_ref(st, layer, e) else {
        return false;
    };

    let got: [&[u8]; 6] = [q.p1, q.s1, q.p2, q.s2, q.p3, q.s3];
    let mut ok = true;
    for i in 0..3 {
        if !ok {
            break;
        }
        let m = &r.m[i];
        let pn = m.rows * m.pcols;
        let sn = m.rows * m.scols;
        let tp = &truth[m.p_off as usize..m.p_off as usize + pn];
        let ts = &truth[m.s_off as usize..m.s_off as usize + sn];
        if &got[i * 2][..pn] != tp {
            ok = false;
        }
        if &got[i * 2 + 1][..sn] != ts {
            ok = false;
        }
    }
    ok
}

#[test]
fn streaming_expert_cache() {
    let dir = fixture_dir();
    const NE: usize = 24;

    let st = match St::open(&dir) {
        Ok(s) => s,
        Err(e) => panic!(
            "TEST ABORTED: cannot open {}: {e}\n  build it with: python3 tools/make_cache_fixture.py",
            dir.display()
        ),
    };
    println!(
        "streaming expert cache, {} tensors from {} shard(s)\n",
        st.tensors().len(),
        st.nshard()
    );

    let cfg = zero_cfg(1, NE as i32, 4);
    let topk = cfg.topk as usize;
    let mut t = Checks { fail: 0 };

    // Size the budget by asking, not by arithmetic. The cache rounds a slot up to an
    // O_DIRECT-widened, page-aligned size, so for these deliberately tiny fixture experts
    // the requested budget and the usable slot count part company badly. Grow until the
    // constructor accepts, then confirm PRESSURE remains: fewer slots than experts, so
    // eviction actually runs. A roomy cache hides slot-recycling bugs entirely.
    let probe = match expert_ref(&st, 0, 0) {
        Ok(r) => r,
        Err(e) => panic!("no expert 0: {e}"),
    };
    let mut cache: Option<Cache<'_>> = None;
    let mut budget = probe.nbytes * 8;
    while budget <= probe.nbytes * 4096 {
        if let Ok(c) = Cache::new(&st, &cfg, budget) {
            cache = Some(c);
            break;
        }
        budget *= 2;
    }
    let Some(mut cache) = cache else {
        panic!("cache init failed at every budget");
    };
    let nslot = cache.stats().nslot;
    t.ck(
        nslot > topk && nslot < NE,
        "cache under pressure",
        &format!("{nslot} slots for {NE} experts, top-{topk}"),
    );

    // ---- 1+3: serial path, every expert, under eviction pressure ----
    let mut bad = 0usize;
    for _pass in 0..3 {
        for e in 0..NE {
            match cache.get(0, e) {
                None => bad += 1,
                Some(q) => {
                    if !same_expert(&st, 0, e, &q) {
                        bad += 1;
                    }
                }
            }
        }
    }
    t.ck(
        bad == 0,
        "serial reads are byte-exact",
        &format!("{} of {} reads wrong", bad, 3 * NE),
    );

    // ---- 2: batch prefetch must agree with the serial path ----
    //
    // This is the check the aliasing bug fails. Ask for a whole top-k at once, with the
    // cache too small to hold the previous batch, then verify EVERY expert.
    //
    // C tests `cache.src.getmany` for NULL; the Rust port keeps the method and gates it on
    // the same environment variable, so this is the same "the batch path is not compiled
    // out" check, and a disabled path is a FAILURE here exactly as it is in C.
    cache.reset_stats();
    let batch_on = std::env::var_os("K3_NOPREFETCH").is_none();
    if !batch_on {
        t.ck(
            false,
            "batch prefetch present",
            "K3_NOPREFETCH disables get_many",
        );
    } else {
        let mut bad2 = 0usize;
        let mut batches = 0usize;
        let mut start = 0usize;
        while start + topk <= NE {
            let ids: Vec<i32> = (0..topk).map(|j| (start + j) as i32).collect();
            cache.get_many(0, &ids);
            batches += 1;
            for &id in &ids {
                let e = id as usize;
                match cache.get(0, e) {
                    None => bad2 += 1,
                    Some(q) => {
                        if !same_expert(&st, 0, e, &q) {
                            bad2 += 1;
                        }
                    }
                }
            }
            start += topk;
        }
        t.ck(
            bad2 == 0,
            "batch prefetch is byte-exact",
            &format!("{batches} batches of {topk}, {bad2} wrong"),
        );

        // ACCOUNTING, checked HERE and only here. The loop above is exactly the pattern
        // `moe` uses - prefetch a top-k, then consume that same top-k - and for that
        // pattern every prefetched expert is still resident when get() asks, so it is
        // recorded as a hit and prefetch_reads can never exceed hits. If it does, the
        // report's "true resident hit rate" underflows and the whole figure is a lie. The
        // mixed test below deliberately prefetches sets it never consumes, so the
        // invariant does NOT hold there and asserting it would be wrong.
        let s = cache.stats();
        t.ck(
            s.prefetch_reads <= s.hits,
            "prefetch_reads <= hits",
            &format!(
                "requests {}, hits {}, prefetch {}",
                s.hits + s.misses,
                s.hits,
                s.prefetch_reads
            ),
        );
    }

    // ---- 2b: interleaving the two paths must not corrupt either ----
    cache.reset_stats();
    let mut bad3 = 0usize;
    for e in 0..NE {
        if batch_on && e % 2 == 0 {
            let ids: Vec<i32> = (0..4).map(|j| ((e + j) % NE) as i32).collect();
            cache.get_many(0, &ids);
        }
        match cache.get(0, e) {
            None => bad3 += 1,
            Some(q) => {
                if !same_expert(&st, 0, e, &q) {
                    bad3 += 1;
                }
            }
        }
    }
    t.ck(
        bad3 == 0,
        "mixed batch and serial",
        &format!("{bad3} of {NE} wrong"),
    );

    println!(
        "\n{}",
        if t.fail != 0 {
            "CACHE TESTS FAILED"
        } else {
            "CACHE TESTS PASSED"
        }
    );
    assert_eq!(t.fail, 0, "CACHE TESTS FAILED");
}
