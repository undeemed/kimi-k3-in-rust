// SPDX-License-Identifier: Apache-2.0
//! Run a REAL Kimi K3 layer from the REAL checkpoint.
//!
//! Port of `tests/unit/test_real_layer.c`.
//!
//! WHAT IS ACTUALLY NEW HERE
//!   The arithmetic is already validated by the op fixtures and the full-model oracle.
//!   What has never been tested is the BINDING: whether `bind` attaches the right
//!   checkpoint tensor to the right kernel argument. That failure mode is invisible to
//!   every weightless test, because a transposed or swapped weight still has the right
//!   shape and produces finite, plausible numbers. It can only be caught by running real
//!   weights and comparing against an independent implementation reading the same file.
//!
//!   So this dumps its input and intermediate outputs in stages, and
//!   `tools/verify_real_layer.py` reproduces each stage in torch from the same shards:
//!     stage A  attention output       -> the attention tensors are wired correctly
//!     stage B  router indices/weights  -> gate and e_score_correction_bias are correct
//!     stage C  MoE output              -> latent down/up, shared experts, and the
//!                                         streamed MXFP4 experts are all correct
//!
//! This test is `#[ignore]`-gated: it needs the real 1.56 TB checkpoint, whose directory
//! comes from the `K3_SHARD_DIR` environment variable. Without it, the test prints a note
//! and returns (a skip, not a failure, matching the Makefile's NOT RUN convention).

use k3::bind::LayerBind;
use k3::cache::Cache;
use k3::cfg::Cfg;
use k3::ops::{
    kda_layer, layer_scratch, mla, mmw, moe, moe_scratch, rmsnorm, router, situ_glu, ExpertSrc,
};
use k3::st::St;
use std::io::Write;
use std::path::Path;

/// The real Kimi K3 configuration, from config.json and confirmed against the shipped
/// tensor shapes in every case. test_real_layer.c:63.
fn real_cfg() -> Cfg {
    let mut full_attn: Vec<i32> = Vec::new();
    let mut i = 4;
    while i <= 93 {
        full_attn.push(i); // one-based in config
        i += 4;
    }
    full_attn.push(93);
    Cfg {
        hidden: 7168,
        n_layers: 93,
        vocab: 163840,
        rms_eps: 1e-5,
        kda_heads: 96,
        kda_head_dim: 128,
        conv_k: 4,
        gate_lb: -5.0,
        n_heads: 96,
        q_lora: 1536,
        kv_lora: 512,
        qk_nope: 128,
        qk_rope: 64,
        v_head: 128,
        mla_out_gate: 1,
        n_experts: 896,
        topk: 16,
        n_shared: 2,
        latent: 3584,
        moe_inter: 3072,
        routed_scale: 1.0,
        moe_renorm: 1,
        latent_norm: 1,
        first_dense: 1,
        dense_inter: 33792,
        attn_res_block: 12,
        situ_b1: 4.0,
        situ_b2: 25.0,
        full_attn,
    }
}

/// xorshift32 PRNG matching the C test's `rnd_f`. test_real_layer.c:44.
struct Rng {
    rs: u32,
}
impl Rng {
    fn new() -> Self {
        Rng { rs: 20260728 }
    }
    fn next_f32(&mut self) -> f32 {
        self.rs ^= self.rs << 13;
        self.rs ^= self.rs >> 17;
        self.rs ^= self.rs << 5;
        (self.rs >> 8) as f32 / 8388608.0 - 1.0
    }
}

/// Write an f32 array as a JSON array of bit patterns (exact, NaN-safe).
/// test_real_layer.c:51.
fn dump_f32<W: Write>(w: &mut W, name: &str, v: &[f32], comma: bool) -> std::io::Result<()> {
    write!(w, "{}\"{}\":[", if comma { "," } else { "" }, name)?;
    for (i, &x) in v.iter().enumerate() {
        if i > 0 {
            write!(w, ",")?;
        }
        write!(w, "{}", x.to_bits())?;
    }
    write!(w, "]")
}

#[test]
#[ignore = "needs the real 1.56 TB checkpoint; set K3_SHARD_DIR"]
fn real_layer() {
    let dir = match std::env::var("K3_SHARD_DIR") {
        Ok(d) => d,
        Err(_) => {
            eprintln!("test_real_layer: K3_SHARD_DIR not set; this test needs the real checkpoint");
            return;
        }
    };
    let dir = Path::new(&dir);

    let args: Vec<String> = std::env::args().collect();
    let layer: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let t: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let cache_gb: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2.0);

    let c = real_cfg();
    println!("Kimi K3, released weights, layer {}, T={}", layer, t);
    println!(
        "  layer {} is {} and {}",
        layer,
        if c.is_mla(layer as i32) { "MLA" } else { "KDA" },
        if c.is_dense(layer as i32) {
            "dense"
        } else {
            "MoE"
        }
    );

    let st = match St::open(dir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("k3_st_open failed: {}", e);
            std::process::exit(1);
        }
    };
    println!(
        "  indexed {} tensors from {} shard(s)\n",
        st.tensors().len(),
        st.nshard()
    );

    // ---- bind ----
    let need = match k3::bind::layer_bytes(&st, &c, layer) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("binding plan FAILED: {}", e);
            std::process::exit(1);
        }
    };
    println!(
        "layer {} resident weights: {:.2} GB in RAM, held in the checkpoint's own bf16",
        layer,
        need as f64 / 1e9
    );

    let b = match LayerBind::from_shards(&st, &c, layer) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("BIND FAILED: {}", e);
            std::process::exit(1);
        }
    };

    // ---- the expert cache ----
    let cache_budget = (cache_gb * 1e9) as i64;
    let mut cache = match Cache::new(&st, &c, cache_budget) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("cache init failed: {}", e);
            std::process::exit(1);
        }
    };

    // ---- input ----
    let mut rng = Rng::new();
    let mut x = vec![0f32; t * c.hidden as usize];
    for v in &mut x {
        *v = rng.next_f32() * 0.5;
    }

    let mut f = std::fs::File::create("real_layer.json").unwrap();
    write!(
        f,
        "{{\"layer\":{},\"T\":{},\"hidden\":{},\"topk\":{},\"is_mla\":{}",
        layer,
        t,
        c.hidden,
        c.topk,
        c.is_mla(layer as i32) as i32
    )
    .unwrap();
    dump_f32(&mut f, "x", &x, true).unwrap();

    // ---- stage A: attention ----
    let mut att = vec![0f32; t * c.hidden as usize];
    let mut xn = vec![0f32; t * c.hidden as usize];
    let views = b.views();
    let in_norm = views.in_norm;
    for tok in 0..t {
        let (xo, xno) = (
            &x[tok * c.hidden as usize..(tok + 1) * c.hidden as usize],
            &mut xn[tok * c.hidden as usize..(tok + 1) * c.hidden as usize],
        );
        rmsnorm(xno, xo, in_norm, c.rms_eps);
    }
    dump_f32(&mut f, "x_norm", &xn, true).unwrap();

    let p = (c.kda_heads as usize) * (c.kda_head_dim as usize);
    let state_size = p * c.kda_head_dim as usize + 3 * p * (c.conv_k as usize - 1);
    let mut state = vec![0f32; state_size];
    let scr_len = layer_scratch(&c, t);
    let mut scr = vec![0f32; scr_len];

    let is_mla = c.is_mla(layer as i32);
    let t0 = std::time::Instant::now();
    if is_mla {
        let mla_w = match &views.attn {
            k3::ops::Attn::Mla(w) => w,
            _ => unreachable!(),
        };
        mla(&mut att, &xn, mla_w, &c, t, &mut scr);
    } else {
        let kda_w = match &views.attn {
            k3::ops::Attn::Kda(w) => w,
            _ => unreachable!(),
        };
        kda_layer(&mut att, &xn, kda_w, &c, t, Some(&mut state), &mut scr);
    }
    let t_attn = t0.elapsed().as_secs_f64();
    dump_f32(&mut f, "attn_out", &att, true).unwrap();
    println!(
        "stage A  {} attention : {:.2} s for {} tokens ({:.0} ms/token)",
        if is_mla { "MLA" } else { "KDA" },
        t_attn,
        t,
        t_attn * 1000.0 / t as f64
    );

    // ---- stage B: routing, on the post-attention normalised hidden ----
    let mut hn = vec![0f32; t * c.hidden as usize];
    let post_norm = views.post_norm;
    for tok in 0..t {
        let (ao, hno) = (
            &att[tok * c.hidden as usize..(tok + 1) * c.hidden as usize],
            &mut hn[tok * c.hidden as usize..(tok + 1) * c.hidden as usize],
        );
        rmsnorm(hno, ao, post_norm, c.rms_eps);
    }
    dump_f32(&mut f, "moe_in", &hn, true).unwrap();

    // LAYER 0 IS DENSE and has no router at all. test_real_layer.c:165.
    if c.is_dense(layer as i32) {
        let di = c.dense_inter as usize;
        let mut dgu = vec![0f32; 2 * di];
        let mut dsub = vec![0f32; di];
        let mut dout = vec![0f32; t * c.hidden as usize];
        let dense_gate = views.dense_gate.unwrap();
        let dense_up = views.dense_up.unwrap();
        let dense_down = views.dense_down.unwrap();
        let t0 = std::time::Instant::now();
        for tok in 0..t {
            let h = &hn[tok * c.hidden as usize..(tok + 1) * c.hidden as usize];
            mmw(&mut dgu[..di], h, dense_gate, c.hidden as usize);
            mmw(&mut dgu[di..], h, dense_up, c.hidden as usize);
            situ_glu(&mut dsub, &dgu, c.situ_b1, c.situ_b2);
            let out = &mut dout[tok * c.hidden as usize..(tok + 1) * c.hidden as usize];
            mmw(out, &dsub, dense_down, di);
        }
        let t_dense = t0.elapsed().as_secs_f64();
        dump_f32(&mut f, "dense_out", &dout, true).unwrap();
        writeln!(f, "}}").unwrap();
        drop(f);
        println!("stage B  (skipped: dense layer, no router)");
        println!(
            "stage C  dense MLP, inter {}: {:.2} s for {} tokens ({:.0} ms/token)\n",
            di,
            t_dense,
            t,
            t_dense * 1000.0 / t as f64
        );
        let mut dfin = true;
        let mut dmx = 0.0f64;
        for &v in &dout {
            if !v.is_finite() {
                dfin = false;
                break;
            }
            if (v as f64).abs() > dmx {
                dmx = (v as f64).abs();
            }
        }
        println!("output finite: {}, max |y| = {:.6}", dfin, dmx);
        println!("wrote real_layer.json for tools/verify_real_layer.py");
        if dfin {
            return;
        } else {
            std::process::exit(1);
        }
    }

    let topk = c.topk as usize;
    let mut idx = vec![0i32; topk];
    let mut wt = vec![0f32; topk];
    write!(&mut f, ",\"route_idx\":[").unwrap();
    let mut allw = vec![0f32; t * topk];
    let moe_w = views.moe.as_ref().unwrap();
    let gate = moe_w.gate;
    let bias = moe_w.bias;
    for tok in 0..t {
        let h = &hn[tok * c.hidden as usize..(tok + 1) * c.hidden as usize];
        router(
            &mut idx,
            &mut wt,
            h,
            gate,
            bias,
            c.hidden as usize,
            c.n_experts as usize,
            topk,
            c.moe_renorm != 0,
            c.routed_scale,
        );
        for j in 0..topk {
            if tok > 0 || j > 0 {
                write!(&mut f, ",").unwrap();
            }
            write!(&mut f, "{}", idx[j]).unwrap();
            allw[tok * topk + j] = wt[j];
        }
        if tok == 0 {
            print!("stage B  router, token 0: experts");
            for j in 0..8.min(topk) {
                print!(" {}({:.4})", idx[j], wt[j]);
            }
            println!(" ...");
        }
    }
    write!(&mut f, "]").unwrap();
    dump_f32(&mut f, "route_wt", &allw, true).unwrap();

    // ---- stage C: MoE with STREAMED experts ----
    let mut moe_out = vec![0f32; t * c.hidden as usize];
    let mscr_len = moe_scratch(&c);
    let mut mscr = vec![0f32; mscr_len];

    cache.reset_stats();
    let t0 = std::time::Instant::now();
    moe(
        &mut moe_out,
        &hn,
        moe_w,
        &c,
        t,
        &mut idx,
        &mut wt,
        &mut mscr,
        Some(&mut cache as &mut dyn ExpertSrc),
    );
    let t_moe = t0.elapsed().as_secs_f64();
    dump_f32(&mut f, "moe_out", &moe_out, true).unwrap();
    writeln!(f, "}}").unwrap();
    drop(f);

    println!(
        "stage C  MoE (streamed): {:.2} s for {} tokens ({:.0} ms/token)\n",
        t_moe,
        t,
        t_moe * 1000.0 / t as f64
    );
    cache.report("after one layer, cold");

    // Second pass: same tokens, so every expert is already resident. The gap between
    // the two is exactly what caching buys.
    cache.reset_stats();
    let t0 = std::time::Instant::now();
    moe(
        &mut moe_out,
        &hn,
        moe_w,
        &c,
        t,
        &mut idx,
        &mut wt,
        &mut mscr,
        Some(&mut cache as &mut dyn ExpertSrc),
    );
    let t_moe2 = t0.elapsed().as_secs_f64();
    println!(
        "\nsecond pass, fully cached: {:.2} s ({:.0} ms/token), {:.1}x faster",
        t_moe2,
        t_moe2 * 1000.0 / t as f64,
        t_moe / t_moe2
    );
    cache.report("second pass");

    let mut finite = true;
    let mut mx = 0.0f64;
    for &v in &moe_out {
        if !v.is_finite() {
            finite = false;
            break;
        }
        if (v as f64).abs() > mx {
            mx = (v as f64).abs();
        }
    }
    println!("\noutput finite: {}, max |y| = {:.6}", finite, mx);
    println!("wrote real_layer.json for tools/verify_real_layer.py");

    let _ = cache.dump_hist(Path::new("expert_hist.json"));
    let _ = cache.dump_trace(Path::new("expert_trace.bin"));

    if !finite {
        std::process::exit(1);
    }
}
