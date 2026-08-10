// SPDX-License-Identifier: Apache-2.0
//! Run the real Kimi K3, all 93 layers, from the released checkpoint.
//!
//! Port of `src/cli/k3_run.c`. Token ids in, token ids out.
//!
//! MEMORY. The banner this program prints before allocating is a PLAN, not a
//! measurement. It reports requested budgets rather than actual reservations, and in
//! practice it OVERSTATES: across the 12-rung ladder in docs/data/ the planned total
//! exceeded measured peak RSS by 0.13-1.84 GB, because both budgets round down to whole
//! slots and that rounding outweighs the safetensors index it omits. Quote the "PEAK RSS"
//! line instead, which comes from `getrusage` after the run.
//!
//! THIS ENGINE IS I/O BOUND at small budgets and roughly balanced at large ones. The
//! measured I/O share runs 40.9%-60.6% across the 12-rung ladder, dropping below 50% at
//! 96 GB and above. The "I/O share" line printed at the end of every run reports it for
//! that run.
//!
//! DECODE STRATEGY. By default each step re-runs the whole prefix rather than carrying
//! state forward. That is O(T^2), but it is the path the full-model oracle validates.
//! `--incremental` switches to prefill-then-one-token-at-a-time, carrying the KDA
//! recurrent state and an MLA KV cache. GATE 3 of the tiny-model oracle requires it to
//! produce the SAME tokens as full recompute.

// Same rationale as `src/lib.rs`: the CLI's forward path mirrors `k3_run.c` argument for
// argument and index for index, so the two can be diffed line by line.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use k3::bind::{self, LayerBind, ModelBind};
use k3::cache::Cache;
use k3::cfg::{self, Cfg, KV_BYTES_PER_POS, MAX_GEN, MAX_PROMPT};
use k3::ops::{self, decoder_layer_inc, mla_scratch_cached, ExpertSrc};
use k3::st::St;
use k3::tok::Tok;
use k3::trunk::Trunk;
use std::env;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::Instant;

const VERSION: &str = "1.0.0";
const SPEC_MAX: usize = 8;
static START: LazyLock<Instant> = LazyLock::new(Instant::now);

// ----------------------------------------------------------- host introspection ----

fn now_s() -> f64 {
    START.elapsed().as_secs_f64()
}

fn human(b: f64) -> String {
    let units = ["B", "KB", "MB", "GB", "TB"];
    let mut i = 0usize;
    let mut b = b;
    while b >= 1000.0 && i < 4 {
        b /= 1000.0;
        i += 1;
    }
    format!("{:.2} {}", b, units[i])
}

/// Peak resident set, in bytes. `ru_maxrss` is kilobytes on Linux and BYTES on Darwin, so
/// the scale factor differs by platform. This is the authoritative memory figure.
fn peak_rss_bytes() -> f64 {
    let mut ru = libc::rusage {
        ru_utime: libc::timeval {
            tv_sec: 0,
            tv_usec: 0,
        },
        ru_stime: libc::timeval {
            tv_sec: 0,
            tv_usec: 0,
        },
        ru_maxrss: 0,
        ru_ixrss: 0,
        ru_idrss: 0,
        ru_isrss: 0,
        ru_minflt: 0,
        ru_majflt: 0,
        ru_nswap: 0,
        ru_inblock: 0,
        ru_oublock: 0,
        ru_msgsnd: 0,
        ru_msgrcv: 0,
        ru_nsignals: 0,
        ru_nvcsw: 0,
        ru_nivcsw: 0,
    };
    unsafe {
        if libc::getrusage(libc::RUSAGE_SELF, &mut ru as *mut _) != 0 {
            return 0.0;
        }
    }
    #[cfg(target_os = "macos")]
    {
        ru.ru_maxrss as f64 // already bytes
    }
    #[cfg(not(target_os = "macos"))]
    {
        ru.ru_maxrss as f64 * 1024.0 // kilobytes
    }
}

/// Bytes this process may still allocate before it is killed.
///
/// `/proc/meminfo` alone is not that number. Under a cgroup memory cap - a container, or
/// `systemd-run -p MemoryMax=8G` - the host can report tens of GB free while the process is
/// killed at 8 GiB. That is not hypothetical: it is how a 93-layer run died 16 MiB over its
/// cap with this function reporting ~55 GB available, so the refusal below never fired.
/// The binding constraint is the cgroup's remaining headroom, so take the smaller.
fn mem_available_bytes() -> f64 {
    let host = fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines().find_map(|l| {
                l.strip_prefix("MemAvailable:")
                    .and_then(|r| r.split_whitespace().next()?.parse::<f64>().ok())
            })
        })
        .map(|kb| kb * 1024.0);

    match (host, cgroup_headroom_bytes()) {
        (Some(h), Some(c)) => h.min(c),
        (Some(h), None) => h,
        (None, Some(c)) => c,
        (None, None) => 0.0,
    }
}

/// This process's current resident set, from `/proc/self/status` VmRSS. Zero where that
/// does not exist (macOS), which is also where the availability check is skipped anyway.
/// Used to undo a double count: the plan's TOTAL includes what is already resident, while
/// MemAvailable and cgroup headroom both exclude it.
fn own_rss_bytes() -> f64 {
    fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines().find_map(|l| {
                l.strip_prefix("VmRSS:")
                    .and_then(|r| r.split_whitespace().next()?.parse::<f64>().ok())
            })
        })
        .map_or(0.0, |kb| kb * 1024.0)
}

/// The memory cgroup's `(cap, current_charge)` for this process, or `None` when uncapped.
///
/// Handles cgroup v2 (`memory.max`, `memory.current`) and v1 (`memory.limit_in_bytes`,
/// `memory.usage_in_bytes`). v2 writes the literal `max` when unlimited; v1 writes a
/// sentinel near `u64::MAX`, which is why an absurd limit is treated as uncapped rather
/// than as headroom.
fn cgroup_mem_bytes() -> Option<(f64, f64)> {
    // v2 puts this process's cgroup after `0::`; the controller files hang off that path.
    let rel = fs::read_to_string("/proc/self/cgroup")
        .ok()
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("0::").map(|p| p.trim().to_string()))
        })
        .unwrap_or_default();

    cgroup_mem_under(&[
        format!("/sys/fs/cgroup{rel}"),
        "/sys/fs/cgroup".to_string(),
        "/sys/fs/cgroup/memory".to_string(),
    ])
}

/// Remaining headroom in this process's memory cgroup, or `None` when uncapped.
fn cgroup_headroom_bytes() -> Option<f64> {
    cgroup_mem_bytes().map(|(cap, cur)| (cap - cur).max(0.0))
}

/// The parsing behind `cgroup_mem_bytes`, over explicit roots so it can be tested without
/// a container. Tries each root in order and takes the first real cap.
fn cgroup_mem_under(roots: &[String]) -> Option<(f64, f64)> {
    // "Unlimited" has three spellings in the wild: v2's literal `max`, and v1's numeric
    // sentinels - u64::MAX, i64::MAX, and (most commonly) i64::MAX rounded down to a page
    // multiple, 0x7FFF_FFFF_FFFF_F000. That last one sits just BELOW u64::MAX / 2, so a
    // `< u64::MAX / 2` test lets it through and reports 9.2 exabytes of headroom, disabling
    // the very check this exists for. Threshold on physical plausibility instead: no machine
    // has 8 PiB of RAM, so anything this large is a sentinel, not a cap.
    const IMPLAUSIBLE: u64 = 1 << 53; // 8 PiB

    let num = |p: String| -> Option<u64> {
        let s = fs::read_to_string(p).ok()?;
        let s = s.trim();
        if s == "max" {
            return None;
        }
        let v = s.parse::<u64>().ok()?;
        (v < IMPLAUSIBLE).then_some(v)
    };

    for base in roots {
        // v2 first, then v1, since a v2 host may still carry a v1 compatibility tree.
        for (lim, use_) in [
            ("memory.max", "memory.current"),
            ("memory.limit_in_bytes", "memory.usage_in_bytes"),
        ] {
            if let Some(max) = num(format!("{base}/{lim}")) {
                let cur = num(format!("{base}/{use_}")).unwrap_or(0);
                return Some((max as f64, cur as f64));
            }
        }
    }
    None
}

// ----------------------------------------------------------------- presets ----

struct Preset {
    name: &'static str,
    trunk_gb: f64,
    cache_gb: f64,
    note: &'static str,
}

const PRESETS: &[Preset] = &[
    Preset {
        name: "laptop",
        trunk_gb: 3.0,
        cache_gb: 1.0,
        note: "8.2 GB peak RSS. The floor. Runs, slowly.",
    },
    Preset {
        name: "desktop",
        trunk_gb: 16.0,
        cache_gb: 10.0,
        note: "31.9 GB peak RSS.",
    },
    Preset {
        name: "workstation",
        trunk_gb: 60.0,
        cache_gb: 30.0,
        note: "95.5 GB peak RSS; the expert cache starts to matter here.",
    },
    Preset {
        name: "server",
        trunk_gb: 110.0,
        cache_gb: 13.0,
        note: "~128 GB peak RSS; 90 of 93 trunk layers pinned. Fastest.",
    },
    Preset {
        name: "max",
        trunk_gb: 110.0,
        cache_gb: 109.0,
        note: "~224 GB peak RSS; trunk pinned and a large expert cache.",
    },
];

fn preset_find(name: &str) -> Option<&'static Preset> {
    PRESETS.iter().find(|p| p.name == name)
}

fn preset_list<W: Write>(f: &mut W) {
    let _ = writeln!(f, "presets (trunk / expert-cache, in GB):");
    for p in PRESETS {
        let _ = writeln!(
            f,
            "  {:<12} {:6.1} / {:<6.1}  {}",
            p.name, p.trunk_gb, p.cache_gb, p.note
        );
    }
    let _ = writeln!(
        f,
        "  {:<12} {:>6} / {:<6}  sizes both from this machine's free RAM, trunk-first. Recommended.",
        "auto", "fit", "fit"
    );
    let _ = writeln!(
        f,
        "\nAll presets stream the trunk, so they need --trunk <packed_dir>.\n\
                         Run scripts/k3-doctor.sh to see which one this machine fits."
    );
}

// ----------------------------------------------------------------- usage ----

fn usage<W: Write>(f: &mut W) {
    let _ = writeln!(f, "k3 {VERSION}, Kimi K3 inference engine\n");
    let _ = writeln!(f, "usage: k3 <model_dir> [options]\n");
    let _ = writeln!(f, "prompt (exactly one):");
    let _ = writeln!(f, "  --prompt TEXT         tokenize TEXT and run it");
    let _ = writeln!(
        f,
        "  --prompt-file PATH    read the prompt from a file; use this for non-ASCII, since"
    );
    let _ = writeln!(f, "                        argv is re-encoded by the shell");
    let _ = writeln!(
        f,
        "  --ids 1,2,3           raw token ids; the reproducible channel used by the tests\n"
    );
    let _ = writeln!(f, "memory:");
    let _ = writeln!(
        f,
        "  --preset NAME         auto | laptop | desktop | workstation | server | max"
    );
    let _ = writeln!(
        f,
        "                        auto sizes both budgets from this machine's free RAM,"
    );
    let _ = writeln!(
        f,
        "                        trunk-first; also spelled --trunk-gb auto"
    );
    let _ = writeln!(
        f,
        "  --list-presets        show each preset's split and expected speed"
    );
    let _ = writeln!(
        f,
        "  --trunk DIR           packed trunk directory; enables streaming (see scripts/)"
    );
    let _ = writeln!(
        f,
        "  --trunk-gb X          trunk ring / pinned-layer budget"
    );
    let _ = writeln!(f, "  --cache-gb X           routed-expert cache budget\n");
    let _ = writeln!(f, "generation:");
    let _ = writeln!(f, "  --gen N               tokens to generate (default 8)");
    let _ = writeln!(
        f,
        "  --incremental         carry KV cache and recurrent state between tokens"
    );
    let _ = writeln!(
        f,
        "  --save-state PATH     write the carried state after the run, so the next turn of a"
    );
    let _ = writeln!(
        f,
        "                        conversation resumes instead of re-reading the whole prompt"
    );
    let _ = writeln!(
        f,
        "  --load-state PATH     resume from a saved state; the prompt given now is treated as"
    );
    let _ = writeln!(
        f,
        "                        the CONTINUATION of the saved sequence. Needs --incremental"
    );
    let _ = writeln!(
        f,
        "  --draft-trunk DIR     hybrid decode: a second packed trunk (typically a quantized"
    );
    let _ = writeln!(
        f,
        "                        derivation of the real one, see tools/qdq_trunk.py) DRAFTS"
    );
    let _ = writeln!(
        f,
        "                        tokens which the exact model verifies in batched sweeps."
    );
    let _ = writeln!(
        f,
        "                        Output remains exactly the exact model's greedy decode; the"
    );
    let _ = writeln!(
        f,
        "                        draft only proposes. Needs --incremental; implies --spec 4"
    );
    let _ = writeln!(
        f,
        "  --draft-trunk-gb X    trunk budget for the draft model (default 6)"
    );
    let _ = writeln!(
        f,
        "  --spec N              speculative decode: draft up to N tokens by n-gram lookup and"
    );
    let _ = writeln!(
        f,
        "                        verify them in ONE batched sweep. Output is identical to"
    );
    let _ = writeln!(
        f,
        "                        serial decode by construction; needs --incremental. An extra"
    );
    let _ = writeln!(
        f,
        "                        verified position costs ~22% of a serial token when the trunk"
    );
    let _ = writeln!(
        f,
        "                        streams, so repetitive text decodes up to several times faster"
    );
    let _ = writeln!(
        f,
        "  --tok DIR             directory with tiktoken.model and tokenizer_config.json\n"
    );
    let _ = writeln!(f, "diagnostics:");
    let _ = writeln!(
        f,
        "  --config PATH         model config; defaults to <model_dir>/config.json"
    );
    let _ = writeln!(
        f,
        "  --layers N            bind only the first N layers (partial shard sets)"
    );
    let _ = writeln!(
        f,
        "  --dump-logits PATH    write float32 logits for the first step"
    );
    let _ = writeln!(
        f,
        "  --dump-cache-trace D  write expert_hist.json and expert_trace.bin into D, for"
    );
    let _ = writeln!(
        f,
        "                        offline analysis with tools/sim_cache.py"
    );
    let _ = writeln!(
        f,
        "  --out FILE            JSON results (default k3_run.json)"
    );
    let _ = writeln!(f, "  --version, --help\n");
    let _ = writeln!(
        f,
        "Memory is a dial, not a floor: the same model runs in 8 GB and in 224 GB and produces"
    );
    let _ = writeln!(
        f,
        "identical output. Give memory to the trunk before the expert cache, see"
    );
    let _ = writeln!(
        f,
        "docs/TUNING.md for why, and scripts/k3-doctor.sh to size this machine."
    );
}

// ----------------------------------------------------------- config loading ----

/// Prefer the checkpoint's own config; fall back to the built-in Kimi K3 constants only
/// when there is no config.json. A config that was found but could not be parsed is
/// evidence the checkpoint is not what the fallback describes, which is exactly when the
/// fallback is most dangerous, so a found-but-bad config aborts rather than falls back.
fn real_cfg(dir: &str, cfg_path: Option<&str>) -> io::Result<Cfg> {
    let p = match cfg_path {
        Some(p) => PathBuf::from(p),
        None => {
            let guess = Path::new(dir).join("config.json");
            if guess.is_file() {
                guess
            } else {
                let c = real_cfg_hardcoded();
                println!("config: NO config.json found under {}\n\
                         \x20        falling back to the built-in Kimi K3 constants (93 layers, 24 MLA).\n\
                         \x20        These match the released checkpoint but are NOT read from it; pass\n\
                         \x20        --config PATH to validate against the real file.", dir);
                return Ok(c);
            }
        }
    };
    cfg::Cfg::load(&p)
}

fn real_cfg_hardcoded() -> Cfg {
    // The released constants. Every value matches the released config.json, but a
    // hardcoded table cannot notice a checkpoint revision.
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
        // config lists are ONE-based: 4, 8, ..., 92, 93.
        full_attn: (4..=93).step_by(4).chain(std::iter::once(93)).collect(),
    }
}

fn argmax(v: &[f32]) -> i32 {
    let mut b = 0usize;
    for i in 1..v.len() {
        if v[i] > v[b] {
            b = i;
        }
    }
    b as i32
}

// ------------------------------------------------------- conversation state ----
//
// Everything the engine carries between tokens, on disk. Three things and only three:
// the KDA recurrent matrices plus ShortConv history (fixed size, independent of
// context), the MLA KV cache, and the shared rope rows. The AttnRes block buffer is NOT
// carried because `forward` clears it on entry and rebuilds it from the layer outputs
// every pass.
//
// The KV cache is stored position-major inside each MLA layer's slice, so only the
// OCCUPIED positions are written and a resumed run may size its cache differently. The
// header carries a config fingerprint: restoring state built by a different architecture
// would produce fluent, wrong output with nothing to indicate it.

const STATE_MAGIC: [u8; 4] = *b"K3ST";
const STATE_VER: i32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct StateHdr {
    magic: [u8; 4],
    version: i32,
    fp: [i32; 12],
    n_bound: i32,
    n_mla: i32,
    cached: i32,
    nseq: i32,
    kper: i64,
    kvpp: i64,
    ropepp: i64,
}

fn cfg_fingerprint(c: &Cfg) -> [i32; 12] {
    [
        c.hidden,
        c.n_layers,
        c.vocab,
        c.kda_heads,
        c.kda_head_dim,
        c.conv_k,
        c.n_heads,
        c.qk_nope,
        c.qk_rope,
        c.v_head,
        c.n_experts,
        c.topk,
    ]
}

/// Read only the header, so the caller can size buffers before committing to a load.
fn state_peek(path: &str) -> io::Result<StateHdr> {
    let mut f = fs::File::open(path)?;
    let mut buf = [0u8; std::mem::size_of::<StateHdr>()];
    f.read_exact(&mut buf)?;
    let hd: StateHdr = unsafe { std::mem::transmute_copy(&buf) };
    if hd.magic != STATE_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{path} is not a k3 state file"),
        ));
    }
    if hd.version != STATE_VER {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{path} is state version {}, this build writes {}",
                hd.version, STATE_VER
            ),
        ));
    }
    Ok(hd)
}

/// Restore the sequence, the per-layer KDA+conv state, and the MLA KV cache. The
/// destination slices are caller-owned; a mismatch in layer count, MLA count, or KV
/// capacity is a hard refusal.
fn state_load(
    path: &str,
    c: &Cfg,
    hd: &StateHdr,
    seq: &mut [i32],
    ks: &mut [f32],
    kvc: &mut [f32],
    ropec: &mut [f32],
    n_bound: usize,
    n_mla: usize,
    kv_cap: usize,
) -> io::Result<()> {
    let fp = cfg_fingerprint(c);
    if fp != hd.fp {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "REFUSING: {path} was written by a different model architecture.\n  \
             Restoring it would produce fluent, wrong output."
            ),
        ));
    }
    if hd.n_bound as usize != n_bound || hd.n_mla as usize != n_mla {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "REFUSING: {path} holds {} bound layers and {} MLA layers, this run has {} and {}",
                hd.n_bound, hd.n_mla, n_bound, n_mla
            ),
        ));
    }
    if hd.cached as usize > kv_cap {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "REFUSING: {path} holds {} positions, this run's KV cache is {}.\n  \
             Raise --gen or shorten the prompt.",
                hd.cached, kv_cap
            ),
        ));
    }
    let mut f = fs::File::open(path)?;
    // Skip the header.
    let mut head = [0u8; std::mem::size_of::<StateHdr>()];
    f.read_exact(&mut head)?;

    let nseq = hd.nseq as usize;
    if nseq > seq.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{path} is truncated"),
        ));
    }
    read_i32_exact(&mut f, &mut seq[..nseq])?;
    let kper_total = (hd.kper as usize) * n_bound;
    if kper_total > ks.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{path} is truncated"),
        ));
    }
    read_f32_exact(&mut f, &mut ks[..kper_total])?;
    // Position-major inside each layer slice.
    let kvpp = hd.kvpp as usize;
    let ropepp = hd.ropepp as usize;
    let cached = hd.cached as usize;
    for mi in 0..n_mla {
        let dst = &mut kvc[mi * kv_cap * kvpp..mi * kv_cap * kvpp + cached * kvpp];
        read_f32_exact(&mut f, dst)?;
    }
    for mi in 0..n_mla {
        let dst = &mut ropec[mi * kv_cap * ropepp..mi * kv_cap * ropepp + cached * ropepp];
        read_f32_exact(&mut f, dst)?;
    }
    Ok(())
}

fn state_save(
    path: &str,
    c: &Cfg,
    seq: &[i32],
    nseq: usize,
    ks: &[f32],
    kvc: &[f32],
    ropec: &[f32],
    n_bound: usize,
    n_mla: usize,
    kv_cap: usize,
    cached: usize,
    kper: i64,
    kvpp: i64,
    ropepp: i64,
) -> io::Result<()> {
    let hd = StateHdr {
        magic: STATE_MAGIC,
        version: STATE_VER,
        fp: cfg_fingerprint(c),
        n_bound: n_bound as i32,
        n_mla: n_mla as i32,
        cached: cached as i32,
        nseq: nseq as i32,
        kper,
        kvpp,
        ropepp,
    };
    let mut f = fs::File::create(path)?;
    let buf: &[u8] = unsafe {
        std::slice::from_raw_parts(
            &hd as *const StateHdr as *const u8,
            std::mem::size_of::<StateHdr>(),
        )
    };
    f.write_all(buf)?;
    write_i32(&mut f, &seq[..nseq])?;
    let kper_total = kper as usize * n_bound;
    write_f32(&mut f, &ks[..kper_total])?;
    for mi in 0..n_mla {
        let src =
            &kvc[mi * kv_cap * kvpp as usize..mi * kv_cap * kvpp as usize + cached * kvpp as usize];
        write_f32(&mut f, src)?;
    }
    for mi in 0..n_mla {
        let src = &ropec[mi * kv_cap * ropepp as usize
            ..mi * kv_cap * ropepp as usize + cached * ropepp as usize];
        write_f32(&mut f, src)?;
    }
    Ok(())
}

fn read_i32_exact(f: &mut fs::File, buf: &mut [i32]) -> io::Result<()> {
    let bytes = buf.as_bytes_mut();
    f.read_exact(bytes)
}
fn read_f32_exact(f: &mut fs::File, buf: &mut [f32]) -> io::Result<()> {
    let bytes = buf.as_bytes_mut();
    f.read_exact(bytes)
}
fn write_i32(f: &mut fs::File, buf: &[i32]) -> io::Result<()> {
    let bytes = buf.as_bytes();
    f.write_all(bytes)
}
fn write_f32(f: &mut fs::File, buf: &[f32]) -> io::Result<()> {
    let bytes = buf.as_bytes();
    f.write_all(bytes)
}

// Trait extension: read/write the raw byte representation of a numeric slice.
trait RawBytes {
    fn as_bytes(&self) -> &[u8];
    fn as_bytes_mut(&mut self) -> &mut [u8];
}
impl RawBytes for [i32] {
    fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * 4) }
    }
    fn as_bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr() as *mut u8, self.len() * 4) }
    }
}
impl RawBytes for [f32] {
    fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * 4) }
    }
    fn as_bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr() as *mut u8, self.len() * 4) }
    }
}

// ------------------------------------------------------------ spec drafting ----
//
// Longest-suffix n-gram drafting: if the last n ids (n=4, then 3) already appeared earlier
// in the sequence, propose the ids that followed them there. Costs nothing when it
// misses: no draft means the step runs exactly as without --spec. Evidence-gated: a draft
// only fires when the suffix n-gram's occurrences AGREE on what follows; weak drafts are
// worse than no drafts because partial acceptances pay a replay sweep.

fn spec_draft(seq: &[i32], t_len: usize, cap: usize, out: &mut [i32]) -> usize {
    let cap = cap.min(SPEC_MAX);
    for n in (3..=4).rev() {
        if t_len < n + 1 {
            continue;
        }
        let mut m1: i32 = -1;
        let mut m2: i32 = -1; // two most recent matches
        for j in (0..t_len - n - 1).rev() {
            let mut hit = true;
            for i in 0..n {
                if seq[j + i] != seq[t_len - n + i] {
                    hit = false;
                    break;
                }
            }
            if !hit {
                continue;
            }
            if m1 < 0 {
                m1 = j as i32;
            } else {
                m2 = j as i32;
                break;
            }
        }
        if m1 < 0 {
            continue;
        }
        let m1 = m1 as usize;
        let mut nd = 0usize;
        let mut i = 0;
        while nd < cap && m1 + n + i < t_len {
            let cand = seq[m1 + n + i];
            if m2 >= 0 {
                let m2 = m2 as usize;
                if m2 + n + i >= m1 || seq[m2 + n + i] != cand {
                    break;
                }
            }
            out[nd] = cand;
            nd += 1;
            i += 1;
        }
        if nd > 0 {
            return nd;
        }
    }
    0
}

// -------------------------------------------------------------- the Weights ----

/// Everything the forward pass needs, beyond the per-call buffers. Built once, reused
/// every step.
struct Weights {
    /// `n_bound` layer bindings. When the trunk is streamed these stay empty and are
    /// filled per-step from `trunk`; when resident they hold the whole layer.
    lay: Vec<LayerBind>,
    mb: ModelBind,
    n_bound: usize,
    trunk: Option<Trunk>,
    /// Incremental decode state. Only MLA layers need a KV cache, so the 24 of them are
    /// numbered densely rather than indexing all 93 and wasting 74% of the allocation.
    kvc: Option<Vec<f32>>,
    ropec: Option<Vec<f32>>,
    mla_slot: Vec<i32>, // [n_layers] -> dense MLA index, or -1
    n_mla: usize,
    kv_cap: usize,
    cached: usize,
    /// 1 for the hybrid draft: cache-only expert routing.
    draft_mode: bool,
}

impl Weights {
    /// kvc/ropec are present iff incremental.
    fn incremental(&self) -> bool {
        self.kvc.is_some()
    }
}

// ----------------------------------------------------------------- forward ----
//
// One full forward over T tokens, writing logits for the LAST position only. Every step
// rebuilds state from scratch, matching the path the oracle validates.
//
// `arg_all`, when provided, receives `argmax(logits)` for EVERY position 0..T-1, which is
// what batched greedy verification consumes. `logits_last` still gets the final position's
// full vector either way. The extra cost is one `lm_head` matmul per additional position,
// pure RAM-resident compute.

#[allow(clippy::too_many_arguments)]
fn forward(
    w: &mut Weights,
    c: &Cfg,
    cache: &mut Cache,
    ids: &[i32],
    t_len: usize,
    logits_last: &mut [f32],
    scratch: &mut [f32],
    h: &mut [f32],
    br: &mut [f32],
    kstate: &mut [f32],
    arg_all: Option<&mut [i32]>,
) -> Result<(), ()> {
    let e = c.hidden as usize;
    let maxb = (c.n_layers / c.attn_res_block + 2) as usize;
    let p = (c.kda_heads * c.kda_head_dim) as usize;
    let kper = (p * c.kda_head_dim as usize) + 3 * p * (c.conv_k as usize - 1);

    for t in 0..t_len {
        w.mb.embed_row(&mut h[t * e..(t + 1) * e], ids[t] as i64);
    }

    for x in br[..t_len * maxb * e].iter_mut() {
        *x = 0.0;
    }
    // Incremental decode carries the KDA recurrent matrix and ShortConv history across
    // steps, so it must NOT be cleared here; the full-recompute path rebuilds from
    // scratch every step and must be.
    if !w.incremental() {
        for x in kstate[..kper * w.n_bound].iter_mut() {
            *x = 0.0;
        }
    }
    let mut nb = 0usize;

    for l in 0..w.n_bound {
        // Streaming: bring this layer in, and hint the next one so its read overlaps this
        // layer's arithmetic. The order is fixed 0..92 every token, so the hint is never
        // wrong.
        let layer_view = if let Some(trunk) = &w.trunk {
            match trunk.bind(c, l) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("trunk bind failed at layer {l}: {e}");
                    return Err(());
                }
            }
        } else {
            w.lay[l].views()
        };
        if let Some(trunk) = &w.trunk {
            trunk.prefetch(l + 1);
        }

        // The MoE source is shared across layers, so the cache is passed as the expert
        // source for the decoder layer only when this layer has a MoE (the dense layer
        // has none). `moe.layer` already records `l`; the draft model's `cache_only`
        // routing is set when the draft weights are built.
        let has_moe = layer_view.moe.is_some();
        let src_opt: Option<&mut dyn ExpertSrc> = if has_moe {
            Some(cache as &mut dyn ExpertSrc)
        } else {
            None
        };

        // KDA state is per-layer; MLA KV cache and rope are per-MLA-layer.
        let (kvc_l, ropec_l, cached, cap): (Option<&mut [f32]>, Option<&mut [f32]>, usize, usize) =
            if w.incremental() && w.mla_slot[l] >= 0 {
                let mi = w.mla_slot[l] as usize;
                let kvper =
                    w.kv_cap * c.n_heads as usize * (c.qk_nope as usize + c.v_head as usize);
                let rpper = w.kv_cap * c.qk_rope as usize;
                let (kvc, ropec) = (w.kvc.as_mut().unwrap(), w.ropec.as_mut().unwrap());
                (
                    Some(&mut kvc[mi * kvper..(mi + 1) * kvper]),
                    Some(&mut ropec[mi * rpper..(mi + 1) * rpper]),
                    w.cached,
                    w.kv_cap,
                )
            } else {
                (None, None, 0, 0)
            };

        let state_slice: Option<&mut [f32]> = Some(&mut kstate[l * kper..(l + 1) * kper]);
        let h_slice = &mut h[..t_len * e];
        let br_slice = &mut br[..t_len * maxb * e];

        decoder_layer_inc(
            h_slice,
            br_slice,
            &mut nb,
            &layer_view,
            c,
            l as i32,
            t_len,
            state_slice,
            scratch,
            kvc_l,
            ropec_l,
            cached,
            cap,
            src_opt,
        );
    }

    // The model-level aggregator, beyond the two per layer. Exactly one pair exists in
    // the checkpoint; skipping it is silent.
    let (fold, src_rest) = scratch.split_at_mut(e);
    let (src_stack, _) = src_rest.split_at_mut(maxb * e);
    for i in 0..e {
        fold[i] = w.mb.out_res_norm()[i] * w.mb.out_res_proj()[i];
    }
    for t in 0..t_len {
        for b in 0..nb {
            src_stack[b * e..(b + 1) * e]
                .copy_from_slice(&br[(t * maxb + b) * e..(t * maxb + b + 1) * e]);
        }
        src_stack[nb * e..(nb + 1) * e].copy_from_slice(&h[t * e..(t + 1) * e]);
        ops::attn_res(
            &mut h[t * e..(t + 1) * e],
            src_stack,
            fold,
            nb + 1,
            e,
            c.rms_eps,
        );
    }

    let vocab = c.vocab as usize;
    let nrm = &mut scratch[..e];
    if let Some(arg_all) = arg_all {
        for t in 0..t_len {
            ops::rmsnorm(nrm, &h[t * e..(t + 1) * e], w.mb.norm(), c.rms_eps);
            let lm_head = match w.mb.lm_head() {
                Some(m) => m,
                None => {
                    eprintln!("lm_head was not bound; cannot produce logits");
                    return Err(());
                }
            };
            ops::mmw(&mut logits_last[..vocab], nrm, lm_head, e);
            arg_all[t] = argmax(&logits_last[..vocab]);
        }
        // logits_last now holds the FINAL position's vector, same as the plain path.
        return Ok(());
    }
    ops::rmsnorm(nrm, &h[(t_len - 1) * e..t_len * e], w.mb.norm(), c.rms_eps);
    let lm_head = match w.mb.lm_head() {
        Some(m) => m,
        None => {
            eprintln!("lm_head was not bound; cannot produce logits");
            return Err(());
        }
    };
    ops::mmw(&mut logits_last[..vocab], nrm, lm_head, e);
    Ok(())
}

// ------------------------------------------------------------------- main ----

fn run() -> i32 {
    let argv: Vec<String> = env::args().collect();

    // Informational flags work without a model directory.
    for a in argv.iter().skip(1) {
        if a == "--help" || a == "-h" {
            let mut out = io::stdout();
            usage(&mut out);
            return 0;
        }
        if a == "--version" {
            println!("k3 {VERSION}");
            return 0;
        }
        if a == "--list-presets" {
            let mut out = io::stdout();
            preset_list(&mut out);
            return 0;
        }
    }
    if argv.len() < 2 {
        let mut e = io::stderr();
        usage(&mut e);
        return 2;
    }

    let dir = &argv[1];
    if dir.starts_with('-') {
        eprintln!("the first argument must be the model directory, got '{dir}'\n");
        let mut e = io::stderr();
        usage(&mut e);
        return 2;
    }

    let mut ids_s: Option<String> = None;
    let mut outp = String::from("k3_run.json");
    let mut trunk_dir: Option<String> = None;
    let mut trace_dir: Option<String> = None;
    let mut logits_path: Option<String> = None;
    let mut prompt_text: Option<String> = None;
    let mut prompt_file: Option<String> = None;
    let mut tok_dir: Option<String> = None;
    let mut cfg_path: Option<String> = None;
    let mut gen: i32 = 8;
    let mut want_layers: i32 = -1;
    let mut cache_gb: f64 = 64.0;
    let mut trunk_gb: f64 = 16.0;
    let mut budget_auto = false;
    let mut spec_n: i32 = 0;
    let mut tf_check = false;
    let mut draft_dir: Option<String> = None;
    let mut draft_gb: f64 = 6.0;
    let mut load_state: Option<String> = None;
    let mut save_state: Option<String> = None;
    let mut preset_name: Option<String> = None;
    let mut incremental = false;
    let mut plan_only = false;

    // Flags are applied in argv order, so an explicit --trunk-gb/--cache-gb after a
    // --preset still wins, exactly as the C parser behaves.
    let mut i = 2usize;
    while i < argv.len() {
        let a = argv[i].clone();
        // Every valued flag consumes the next argv entry; a missing value is a usage
        // error rather than a silent default.
        let mut value = |name: &str| -> Option<String> {
            if i + 1 >= argv.len() {
                eprintln!("missing value for {name}");
                return None;
            }
            i += 1;
            Some(argv[i].clone())
        };
        match a.as_str() {
            "--ids" => match value("--ids") {
                Some(v) => ids_s = Some(v),
                None => return 2,
            },
            "--prompt" => match value("--prompt") {
                Some(v) => prompt_text = Some(v),
                None => return 2,
            },
            "--prompt-file" => match value("--prompt-file") {
                Some(v) => prompt_file = Some(v),
                None => return 2,
            },
            "--tok" => match value("--tok") {
                Some(v) => tok_dir = Some(v),
                None => return 2,
            },
            "--config" => match value("--config") {
                Some(v) => cfg_path = Some(v),
                None => return 2,
            },
            "--out" => match value("--out") {
                Some(v) => outp = v,
                None => return 2,
            },
            "--trunk" => match value("--trunk") {
                Some(v) => trunk_dir = Some(v),
                None => return 2,
            },
            "--load-state" => match value("--load-state") {
                Some(v) => load_state = Some(v),
                None => return 2,
            },
            "--save-state" => match value("--save-state") {
                Some(v) => save_state = Some(v),
                None => return 2,
            },
            "--draft-trunk" => match value("--draft-trunk") {
                Some(v) => draft_dir = Some(v),
                None => return 2,
            },
            "--dump-logits" => match value("--dump-logits") {
                Some(v) => logits_path = Some(v),
                None => return 2,
            },
            "--dump-cache-trace" => match value("--dump-cache-trace") {
                Some(v) => trace_dir = Some(v),
                None => return 2,
            },
            // C uses atoi/atof, which yield 0 on a non-numeric argument; the range checks
            // downstream then reject it. Mirror that rather than inventing a new error.
            "--gen" => match value("--gen") {
                Some(v) => gen = v.parse().unwrap_or(0),
                None => return 2,
            },
            "--layers" => match value("--layers") {
                Some(v) => want_layers = v.parse().unwrap_or(0),
                None => return 2,
            },
            "--spec" => match value("--spec") {
                Some(v) => spec_n = v.parse().unwrap_or(0),
                None => return 2,
            },
            "--cache-gb" => match value("--cache-gb") {
                Some(v) => cache_gb = v.parse().unwrap_or(0.0),
                None => return 2,
            },
            "--draft-trunk-gb" => match value("--draft-trunk-gb") {
                Some(v) => draft_gb = v.parse().unwrap_or(0.0),
                None => return 2,
            },
            "--trunk-gb" => match value("--trunk-gb") {
                Some(v) => {
                    if v == "auto" {
                        budget_auto = true;
                    } else {
                        trunk_gb = v.parse().unwrap_or(0.0);
                        budget_auto = false;
                    }
                }
                None => return 2,
            },
            "--preset" => match value("--preset") {
                Some(v) => {
                    if v == "auto" {
                        // Not in the table: the table is fixed budgets, auto is computed
                        // from this machine's MemAvailable once parsing is complete.
                        budget_auto = true;
                        preset_name = Some("auto".to_string());
                    } else if let Some(p) = preset_find(&v) {
                        trunk_gb = p.trunk_gb;
                        cache_gb = p.cache_gb;
                        preset_name = Some(p.name.to_string());
                    } else {
                        eprintln!("unknown preset '{v}'\n");
                        let mut e = io::stderr();
                        preset_list(&mut e);
                        return 2;
                    }
                }
                None => return 2,
            },
            "--tf-check" => tf_check = true,
            "--incremental" => incremental = true,
            "--plan-only" => plan_only = true,
            "--list-presets" => {
                let mut o = io::stdout();
                preset_list(&mut o);
                return 0;
            }
            "--version" => {
                println!("k3 {VERSION}");
                return 0;
            }
            "--help" | "-h" => {
                let mut o = io::stdout();
                usage(&mut o);
                return 0;
            }
            _ => {
                eprintln!("unknown option {a}\n");
                let mut e = io::stderr();
                usage(&mut e);
                return 2;
            }
        }
        i += 1;
    }

    // Exactly one prompt source.
    {
        let nsrc = [
            ids_s.is_some(),
            prompt_text.is_some(),
            prompt_file.is_some(),
        ]
        .iter()
        .filter(|&&x| x)
        .count();
        if nsrc == 0 {
            eprintln!("one of --ids, --prompt or --prompt-file is required");
            return 2;
        }
        if nsrc > 1 {
            // Refuse rather than pick: silently preferring one source would run the WRONG
            // prompt for tens of minutes.
            eprintln!("--ids, --prompt and --prompt-file are mutually exclusive");
            return 2;
        }
    }

    // ---- auto budget: RAM-first ----
    if budget_auto {
        let avail = mem_available_bytes();
        if avail <= 0.0 {
            eprintln!("--preset auto needs /proc/meminfo; pass explicit --trunk-gb/--cache-gb on this platform");
            return 2;
        }
        let reserve = 2.0 + 0.02 * (avail / 1e9) + 4.70 + 1.70;
        let usable = avail / 1e9 - reserve;
        let slot_min = 2.5;
        let cache_min = 0.5;
        if usable < slot_min + cache_min {
            eprintln!("auto: only {:.1} GB usable after the {:.1} GB reserve; below the {:.1} GB floor. Pass explicit budgets.",
                usable, reserve, slot_min + cache_min);
            return 2;
        }
        let trunk_full = 111.0;
        if usable - cache_min >= trunk_full {
            trunk_gb = trunk_full;
            cache_gb = usable - trunk_full;
        } else {
            let memtotal = fs::read_to_string("/proc/meminfo")
                .ok()
                .and_then(|s| {
                    s.lines().find_map(|l| {
                        l.strip_prefix("MemTotal:")
                            .and_then(|r| r.trim().parse::<f64>().ok())
                    })
                })
                .map(|kb| kb * 1024.0)
                .unwrap_or(0.0);
            let rss_ceiling = if memtotal > 0.0 {
                0.55 * memtotal / 1e9
            } else {
                usable
            };
            let mut cap = rss_ceiling - reserve - cache_min;
            if cap < slot_min {
                cap = slot_min;
            }
            trunk_gb = usable - cache_min;
            if trunk_gb > cap {
                trunk_gb = cap;
            }
            cache_gb = cache_min;
        }
        println!("auto budget: {:.1} GB available, {:.1} GB reserved -> trunk {:.1} GB / expert cache {:.1} GB",
            avail / 1e9, reserve, trunk_gb, cache_gb);
    }

    let c = match real_cfg(dir, cfg_path.as_deref()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("ABORTED: the model config could not be read with confidence.\n{e}");
            return 2;
        }
    };
    if want_layers > 0 && want_layers < c.n_layers {
        println!("NOTE: binding only the first {} of {} layers. Output is NOT the full model; it is a partial stack for testing the machinery.\n",
            want_layers, c.n_layers);
    }

    // ---- prompt ----
    let mut prompt = vec![0i32; MAX_PROMPT];
    let mut np = 0usize;
    let mut tok: Option<Tok> = None;
    if prompt_text.is_some() || prompt_file.is_some() {
        if tok_dir.is_none() {
            eprintln!("--prompt/--prompt-file need --tok DIR (the directory with tiktoken.model and tokenizer_config.json)");
            return 2;
        }
        tok = Some(
            Tok::load(Path::new(tok_dir.as_deref().unwrap())).unwrap_or_else(|e| {
                eprintln!("tokenizer load failed: {e}");
                std::process::exit(1);
            }),
        );
        let ptext: Vec<u8> = if let Some(pf) = &prompt_file {
            fs::read(pf).unwrap_or_else(|e| {
                eprintln!("cannot read {pf}: {e}");
                std::process::exit(1);
            })
        } else {
            prompt_text.as_ref().unwrap().as_bytes().to_vec()
        };
        let ids = tok
            .as_ref()
            .unwrap()
            .encode(&String::from_utf8_lossy(&ptext));
        for id in &ids {
            if np >= MAX_PROMPT {
                break;
            }
            prompt[np] = *id;
            np += 1;
        }
        println!("  tokenized: {} bytes -> {} ids", ptext.len(), np);
    } else {
        let s = ids_s.unwrap();
        let mut p = s.as_bytes();
        while !p.is_empty() && np < MAX_PROMPT {
            // parse a comma- or space-separated int
            let mut val: i32 = 0;
            let mut sign = 1;
            let mut j = 0;
            while j < p.len() && (p[j] == b',' || p[j] == b' ') {
                j += 1;
            }
            if j < p.len() && p[j] == b'-' {
                sign = -1;
                j += 1;
            }
            let start = j;
            while j < p.len() && p[j].is_ascii_digit() {
                val = val * 10 + (p[j] - b'0') as i32;
                j += 1;
            }
            if j == start {
                break;
            }
            prompt[np] = sign * val;
            np += 1;
            p = &p[j..];
        }
    }
    if np == 0 {
        eprintln!("no prompt ids parsed");
        return 2;
    }
    for id in &prompt[..np] {
        if *id < 0 || *id >= c.vocab {
            eprintln!("token id {} is outside the vocabulary of {}", id, c.vocab);
            return 2;
        }
    }

    if gen < 0 || gen > MAX_GEN as i32 {
        eprintln!(
            "--gen {} is out of range: this build generates at most {} tokens (outtok[{}])",
            gen, MAX_GEN, MAX_GEN
        );
        return 2;
    }
    if np > MAX_PROMPT {
        eprintln!(
            "prompt of {} ids exceeds the {}-id ceiling (seq[{}])",
            np,
            MAX_PROMPT,
            MAX_PROMPT + MAX_GEN
        );
        return 2;
    }
    if np + gen as usize + 1 > MAX_PROMPT + MAX_GEN {
        eprintln!(
            "prompt {} + gen {} + 1 exceeds the {}-position ceiling",
            np,
            gen,
            MAX_PROMPT + MAX_GEN
        );
        return 2;
    }
    if incremental {
        let kv_need = (np + gen as usize + 1) as f64 * KV_BYTES_PER_POS;
        let avail = mem_available_bytes();
        println!(
            "  KV cache : {} for {} positions ({:.2} MB/position)",
            human(kv_need),
            np + gen as usize + 1,
            KV_BYTES_PER_POS / 1e6
        );
        if avail > 0.0 && kv_need > avail * 0.9 {
            eprintln!(
                "\nREFUSING: the KV cache for {} positions needs {} but only {} is\n\
                      available. This is a MEMORY limit, not an engine ceiling: MLA caches\n\
                      expanded k and v in fp32 across 24 layers, so context costs ~2.37 MB per\n\
                      position regardless of budget. Shorten the request, or use full\n\
                      recompute (drop --incremental), which carries no KV cache at all.",
                np + gen as usize + 1,
                human(kv_need),
                human(avail)
            );
            return 2;
        }
    }

    println!("Kimi K3, pure Rust, released checkpoint");
    println!("  model    : {}", dir);
    println!("  prompt   : {} tokens, generating {}", np, gen);
    if let Some(pn) = &preset_name {
        println!(
            "  preset   : {} (trunk {:.1} GB / expert cache {:.1} GB)",
            pn, trunk_gb, cache_gb
        );
    }
    println!();

    let st = match St::open(Path::new(dir)) {
        Ok(st) => st,
        Err(e) => {
            eprintln!("failed to open shards: {e}");
            return 1;
        }
    };
    let t0 = now_s();
    println!(
        "indexed {} tensors from {} shards in {:.2} s",
        st.tensors().len(),
        st.nshard(),
        now_s() - t0
    );

    let nl = (if want_layers > 0 && want_layers < c.n_layers {
        want_layers
    } else {
        c.n_layers
    }) as usize;
    let mut total: i64 = 0;
    let mut missing = 0usize;
    for l in 0..nl {
        match bind::layer_bytes(&st, &c, l) {
            Ok(n) => total += n,
            Err(_) => missing += 1,
        }
    }
    if missing > 0 {
        eprintln!("\n{} of {} layers are missing tensors in this shard set. A partial download cannot run the model.", missing, nl);
        return 1;
    }
    if let Some(td) = &trunk_dir {
        println!(
            "trunk on disk : {} total (STREAMED from {}, not held in RAM)",
            human(total as f64),
            td
        );
    } else {
        println!("resident trunk: {} in RAM (large matrices kept in the checkpoint's bf16,\n  fp32 only for the norms and biases that kernels read elementwise)", human(total as f64));
    }

    // ---- memory plan ----
    {
        let e64 = c.hidden as i64;
        let w_trunk = if trunk_dir.is_some() {
            trunk_gb * 1e9
        } else {
            total as f64
        };
        let w_model = 2.0 * c.vocab as f64 * e64 as f64 * 2.0 + 3.0 * e64 as f64 * 4.0;
        let w_cache = cache_gb * 1e9;
        let tm = (np + gen as usize + 1) as f64;
        let mb = (c.n_layers / c.attn_res_block + 2) as f64;
        let pp = (c.kda_heads * c.kda_head_dim) as f64;
        let w_state =
            (pp * c.kda_head_dim as f64 + 3.0 * pp * (c.conv_k as f64 - 1.0)) * nl as f64 * 4.0;
        let w_buf = (tm * e64 as f64
            + tm * mb * e64 as f64
            + ops::layer_scratch(&c, tm as usize) as f64
            + c.vocab as f64)
            * 4.0;
        let n_mla = (0..c.n_layers).filter(|&l| c.is_mla(l)).count();
        let w_kv = if incremental {
            tm * n_mla as f64
                * (c.n_heads as f64 * (c.qk_nope as f64 + c.v_head as f64) + c.qk_rope as f64)
                * 4.0
        } else {
            0.0
        };
        // The tensor index is resident for the whole run and scales with the checkpoint,
        // not the config: 497,220 tensors on the released weights. Leaving it out of the
        // forecast is what let a run start 16 MiB short of its cap and get OOM-killed.
        let w_index = st.index_bytes() as f64;
        let need_b = w_trunk + w_model + w_cache + w_state + w_buf + w_kv + w_index;
        let have = mem_available_bytes();
        println!("\nmemory plan");
        println!("  trunk {:<10} {}\n  embed + lm_head  {}\n  expert cache     {}\n  recurrent state  {}\n  buffers          {}\n  KV cache         {}\n  tensor index     {}\n  TOTAL            {}",
            if trunk_dir.is_some() { "(STREAMED)" } else { "(resident)" },
            human(w_trunk), human(w_model), human(w_cache), human(w_state), human(w_buf), human(w_kv), human(w_index), human(need_b));
        // A machine-readable copy, emitted before the local-availability check, because a
        // forecast must not depend on the machine doing the forecasting: the point of
        // --plan-only is to size a DIFFERENT box, usually one not yet rented. The caller
        // compares this against the cap it intends to impose.
        if plan_only {
            println!("plan_bytes {}", need_b as u64);
            println!(
                "\nplan only, nothing was loaded. This forecast needs no tensor data, so it can be\n\
                 taken against a header-only replica of a checkpoint before renting anything."
            );
            return 0;
        }
        // The margin covers what the forecast cannot see: page tables (16 MB measured at
        // the OOM kill), kernel accounting, and forecast error (13 MB observed against the
        // killed run's actual peak). A percentage is the wrong shape - 5% of this plan is
        // 430 MB, which refused a run on a box where the C build had just completed under
        // the same cap, and where the plan measurably fit with ~116 MB to spare. A wrong
        // refusal and a mid-run kill now cost the same thing (the box holds either way),
        // so the guard is sized to measurements, not to fear.
        const MARGIN: f64 = 64.0 * 1024.0 * 1024.0;
        if let Some((cap, _)) = cgroup_mem_bytes() {
            // Under a cgroup cap, the plan compares against the CAP, not against
            // remaining headroom. Two reasons, both from the forensics of the run this
            // guard failed to save and the run it then wrongly refused:
            //   - The plan is a forecast of ANON memory, and anon is what kills: at the
            //     OOM the process held anon-rss 7.98 GiB and file-rss 2 MB. File pages
            //     yield under pressure, so page cache charged to the cgroup (headers read
            //     while indexing) reduces headroom but not what the run can actually hold.
            //   - Headroom double-counts: by plan time the index is built, so it is both
            //     inside `need_b` and already subtracted from headroom. 8.47 GB of need
            //     was refused at an 8.59 GB cap on that arithmetic.
            println!("  cgroup cap       {}", human(cap));
            if need_b + MARGIN > cap {
                eprintln!("\nREFUSING TO START: this needs {} (+ {} margin) against a {} memory cap.\nThe cap is the limit here, not the host's free memory.\nOptions: a larger cap, a smaller --cache-gb or --trunk-gb, or fewer --layers.",
                    human(need_b), human(MARGIN), human(cap));
                return 1;
            }
        } else if have > 0.0 {
            // Uncapped: compare against what the host can still hand out, plus what this
            // process already holds (the plan's total includes it; MemAvailable does not).
            let usable = have + own_rss_bytes();
            println!("  available        {}", human(usable));
            if need_b + MARGIN > usable {
                eprintln!("\nREFUSING TO START: this needs {} (+ {} margin) and only {} is available.\nOptions: more free memory, a smaller --cache-gb or --trunk-gb, or fewer --layers.",
                    human(need_b), human(MARGIN), human(usable));
                return 1;
            }
        }
        println!();
    }

    let mut w = Weights {
        lay: Vec::new(),
        mb: ModelBind::load(&st, &c, true).unwrap_or_else(|e| {
            eprintln!("bind_model failed: {e}");
            std::process::exit(1);
        }),
        n_bound: nl,
        trunk: None,
        kvc: None,
        ropec: None,
        mla_slot: vec![-1; c.n_layers as usize],
        n_mla: 0,
        kv_cap: 0,
        cached: 0,
        draft_mode: false,
    };

    let t0 = now_s();
    if let Some(td) = &trunk_dir {
        let trunk = match Trunk::open(Path::new(td), &c, (trunk_gb * 1e9) as i64) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("trunk open failed: {e}");
                return 1;
            }
        };
        if trunk.n_layers() < nl {
            eprintln!("packed trunk has {} layers, need {}", trunk.n_layers(), nl);
            return 1;
        }
        w.trunk = Some(trunk);
        println!(
            "trunk streaming enabled from {} in {:.1} s",
            td,
            now_s() - t0
        );
    } else {
        w.lay.reserve(nl);
        for l in 0..nl {
            match LayerBind::from_shards(&st, &c, l) {
                Ok(b) => w.lay.push(b),
                Err(e) => {
                    eprintln!("bind failed at layer {l}: {e}");
                    return 1;
                }
            }
            if (l + 1) % 10 == 0 || l + 1 == nl {
                println!(
                    "  bound {}/{} layers, {:.1} s elapsed",
                    l + 1,
                    nl,
                    now_s() - t0
                );
                let _ = io::stdout().flush();
            }
        }
        let t_bind = now_s() - t0;
        println!(
            "trunk loaded in {:.1} s ({:.0} MB/s from disk)",
            t_bind,
            total as f64 / 1e6 / t_bind
        );
    }

    let t0 = now_s();
    println!(
        "embedding, final norm and lm_head: {} in {:.1} s\n",
        human(w.mb.nbytes() as f64),
        now_s() - t0
    );

    let mut cache = match Cache::new(&st, &c, (cache_gb * 1e9) as i64) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("cache init failed: {e}");
            return 1;
        }
    };
    println!(
        "peak RSS after loading weights: {}  (the plan above is a forecast, this is measured)",
        human(peak_rss_bytes())
    );
    let stats = cache.stats();
    println!(
        "expert cache: {} slots x {:.2} MB = {:.2} GB ({:.2}% of the 1.45 TB expert pool)\n",
        stats.nslot,
        stats.slot_bytes as f64 / 1e6,
        stats.nslot as f64 * stats.slot_bytes as f64 / 1e9,
        100.0 * stats.nslot as f64 / (92.0 * c.n_experts as f64)
    );

    // ---- buffers ----
    let prior = if let Some(ls) = &load_state {
        if !incremental {
            eprintln!("--load-state needs --incremental");
            return 2;
        }
        match state_peek(ls) {
            Ok(h) => {
                println!(
                    "resuming from {}: {} prior positions, {} new\n",
                    ls, h.nseq, np
                );
                h.nseq as usize
            }
            Err(e) => {
                eprintln!("{e}");
                return 1;
            }
        }
    } else {
        0
    };
    let tmax = prior + np + gen as usize + 1;
    let e = c.hidden as usize;
    let maxb = (c.n_layers / c.attn_res_block + 2) as usize;
    let p = (c.kda_heads * c.kda_head_dim) as usize;
    let kper = p * c.kda_head_dim as usize + 3 * p * (c.conv_k as usize - 1);

    {
        let need_scratch = (maxb + 2) * e;
        let have_scratch = ops::layer_scratch(&c, tmax);
        if have_scratch < need_scratch {
            eprintln!(
                "scratch is {} floats, the attn-res aggregator needs {}",
                have_scratch, need_scratch
            );
            return 1;
        }
    }

    let mut h = vec![0f32; tmax * e];
    let mut br = vec![0f32; tmax * maxb * e];
    let mut ks = vec![0f32; kper * nl];
    let mut sc_need = ops::layer_scratch(&c, tmax);
    {
        let ic = mla_scratch_cached(&c, tmax, tmax, true);
        if ic > sc_need {
            sc_need = ic;
        }
    }
    let mut sc = vec![0f32; sc_need];
    let mut lg = vec![0f32; c.vocab as usize];
    println!(
        "recurrent state for {} layers: {}\n",
        nl,
        human((kper * nl) as f64 * 4.0)
    );

    let mut seq = vec![0i32; prior + np + gen as usize + 8];
    let mut outtok: Vec<i32> = Vec::with_capacity(gen as usize + 8);
    seq[prior..prior + np].copy_from_slice(&prompt[..np]);
    let mut t_len = prior + np;
    let mut nout = 0usize;

    if incremental {
        w.n_mla = 0;
        for l in 0..nl {
            if c.is_mla(l as i32) {
                w.mla_slot[l] = w.n_mla as i32;
                w.n_mla += 1;
            } else {
                w.mla_slot[l] = -1;
            }
        }
        w.kv_cap = tmax;
        let kvper = w.kv_cap * c.n_heads as usize * (c.qk_nope as usize + c.v_head as usize);
        let rpper = w.kv_cap * c.qk_rope as usize;
        println!(
            "incremental decode: KV cache {} for {} MLA layers at {} positions\n",
            human((kvper + rpper) as f64 * w.n_mla as f64 * 4.0),
            w.n_mla,
            w.kv_cap
        );
        w.kvc = Some(vec![0f32; kvper * w.n_mla]);
        w.ropec = Some(vec![0f32; rpper * w.n_mla]);
        for x in ks[..kper * nl].iter_mut() {
            *x = 0.0;
        }
        w.cached = 0;
        if let Some(ls) = &load_state {
            let tl = now_s();
            if let Err(e) = state_load(
                ls,
                &c,
                &state_peek(ls).unwrap(),
                &mut seq,
                &mut ks,
                w.kvc.as_mut().unwrap(),
                w.ropec.as_mut().unwrap(),
                w.n_bound,
                w.n_mla,
                w.kv_cap,
            ) {
                eprintln!("{e}");
                return 1;
            }
            // state_load already set w.cached? No: peek gives the header.
            w.cached = state_peek(ls).unwrap().cached as usize;
            println!("restored {} positions in {:.2} s: decode continues without re-reading the prior context\n",
                w.cached, now_s() - tl);
        }
    }

    let kper_p = (c.kda_heads * c.kda_head_dim) as usize;
    let kper_f = kper_p * c.kda_head_dim as usize + 3 * kper_p * (c.conv_k as usize - 1);
    let mut spec_snap: Option<Vec<f32>> = None;
    if spec_n > 0 {
        if !incremental {
            eprintln!("--spec needs --incremental; ignoring --spec");
            spec_n = 0;
        } else {
            let sn = spec_n.min(SPEC_MAX as i32);
            spec_n = sn;
            spec_snap = Some(vec![0f32; kper_f * w.n_bound]);
            println!("speculative decode: up to {} drafted tokens per sweep, n-gram lookup, verified batched\n", spec_n);
        }
    }

    // Hybrid draft trunk: a second packed trunk drafts, the exact model verifies. Output
    // is exactly the exact model's greedy decode by construction.
    let mut dw = Weights {
        lay: Vec::new(),
        mb: ModelBind::load(&st, &c, false).unwrap_or_else(|e| {
            eprintln!("draft bind_model failed: {e}");
            std::process::exit(1);
        }),
        n_bound: nl,
        trunk: None,
        kvc: None,
        ropec: None,
        mla_slot: w.mla_slot.clone(),
        n_mla: w.n_mla,
        kv_cap: w.kv_cap,
        cached: 0,
        draft_mode: false,
    };
    let mut dks: Option<Vec<f32>> = None;
    let mut dsnap: Option<Vec<f32>> = None;
    let mut hyb_rounds = 0i64;
    let mut hyb_drafted = 0i64;
    let mut hyb_accepted = 0i64;
    if let Some(dd) = &draft_dir {
        if !incremental || trunk_dir.is_none() {
            // C also clears draft_dir here; nothing downstream reads it, because every
            // later branch tests `dw.trunk` instead, which this path leaves unset.
            eprintln!("--draft-trunk needs --incremental and --trunk; ignoring");
        } else {
            if spec_n <= 0 {
                spec_n = 4;
            }
            let sn = spec_n.min(SPEC_MAX as i32);
            spec_n = sn;
            if spec_snap.is_none() {
                spec_snap = Some(vec![0f32; kper_f * w.n_bound]);
            }
            let trunk_d = match Trunk::open(Path::new(dd), &c, (draft_gb * 1e9) as i64) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("draft trunk open failed: {e}");
                    return 1;
                }
            };
            dks = Some(vec![0f32; kper_f * w.n_bound]);
            dsnap = Some(vec![0f32; kper_f * w.n_bound]);
            let kvperd = w.kv_cap * c.n_heads as usize * (c.qk_nope as usize + c.v_head as usize);
            let rpperd = w.kv_cap * c.qk_rope as usize;
            dw.kvc = Some(vec![0f32; kvperd * w.n_mla]);
            dw.ropec = Some(vec![0f32; rpperd * w.n_mla]);
            dw.trunk = Some(trunk_d);
            dw.draft_mode = true;
            println!("hybrid decode: draft trunk {} ({:.1} GB budget) proposes up to {} tokens per sweep;\n               the exact model verifies every one before it is emitted\n",
                dd, draft_gb, spec_n);
        }
    }

    if tf_check {
        if np < 2 {
            eprintln!("--tf-check needs at least 2 ids");
            return 2;
        }
        let mut arg = vec![0i32; np];
        let t0c = now_s();
        if forward(
            &mut w,
            &c,
            &mut cache,
            &seq,
            np,
            &mut lg,
            &mut sc,
            &mut h,
            &mut br,
            &mut ks,
            Some(&mut arg),
        )
        .is_err()
        {
            eprintln!("forward failed in --tf-check");
            return 1;
        }
        let mut m = 0usize;
        for i in 0..np - 1 {
            if arg[i] == seq[i + 1] {
                m += 1;
            }
        }
        println!(
            "teacher-forced agreement: {}/{} positions ({:.1}%) in {:.1} s",
            m,
            np - 1,
            100.0 * m as f64 / (np - 1) as f64,
            now_s() - t0c
        );
        print!("  per-position (p=predicted a=actual): ");
        for i in 0..np - 1 {
            if arg[i] != seq[i + 1] {
                print!("[{} p={} a={}] ", i, arg[i], seq[i + 1]);
            }
        }
        println!();
        if let Ok(mut tf) = fs::File::create(&outp) {
            let _ = writeln!(
                tf,
                "{{\"tf_positions\":{},\"tf_matches\":{},\"tf_agreement\":{:.4}}}",
                np - 1,
                m,
                m as f64 / (np - 1) as f64
            );
        }
        return 0;
    }

    println!(
        "{:<6} {:<10} {:<12} {:<10} {:<10} TOK/S",
        "STEP", "TOKEN", "SECONDS", "CACHE HIT", "READ GB"
    );
    println!("--------------------------------------------------------------------");
    let mut t_total = 0.0f64;
    let mut expert_s_total = 0.0f64;
    let mut expert_gb_total = 0.0f64;
    let mut expert_reqs_total = 0u64;
    let mut expert_evict_total = 0u64;

    let mut g = 0i32;
    while nout < gen as usize {
        cache.reset_stats();
        let ts = now_s();
        let mut emit = [0i32; SPEC_MAX + 1];
        let mut emitn = 0usize;
        let frc_ok;

        if incremental && g == 0 {
            let base = w.cached;
            let nt0 = t_len - base;
            frc_ok = forward(
                &mut w,
                &c,
                &mut cache,
                &seq[base..],
                nt0,
                &mut lg,
                &mut sc,
                &mut h,
                &mut br,
                &mut ks,
                None,
            )
            .is_ok();
            if frc_ok {
                w.cached = base + nt0;
                emit[emitn] = argmax(&lg[..c.vocab as usize]);
                emitn += 1;
            }
            if dw.trunk.is_some() && frc_ok {
                let db = if load_state.is_some() { 0 } else { base };
                if forward(
                    &mut dw,
                    &c,
                    &mut cache,
                    &seq[db..],
                    base + nt0 - db,
                    &mut lg,
                    &mut sc,
                    &mut h,
                    &mut br,
                    dks.as_mut().unwrap(),
                    None,
                )
                .is_ok()
                {
                    dw.cached = base + nt0;
                }
            }
        } else if incremental {
            let base = w.cached;
            let mut d = [0i32; SPEC_MAX];
            let mut nd = 0usize;
            if spec_snap.is_some()
                && t_len + spec_n as usize + 1 < tmax
                && base + (spec_n as usize) < w.kv_cap
            {
                if dw.trunk.is_some() {
                    // The draft model proposes: sequential one-token steps through the draft
                    // trunk, chaining its own argmax. Its state is snapshotted first so a
                    // partial acceptance can rewind it.
                    dsnap
                        .as_mut()
                        .unwrap()
                        .copy_from_slice(dks.as_ref().unwrap());
                    let mut prev = seq[base];
                    while nd < spec_n as usize {
                        if forward(
                            &mut dw,
                            &c,
                            &mut cache,
                            std::slice::from_ref(&prev),
                            1,
                            &mut lg,
                            &mut sc,
                            &mut h,
                            &mut br,
                            dks.as_mut().unwrap(),
                            None,
                        )
                        .is_err()
                        {
                            break;
                        }
                        dw.cached += 1;
                        prev = argmax(&lg[..c.vocab as usize]);
                        d[nd] = prev;
                        nd += 1;
                    }
                    hyb_rounds += 1;
                    hyb_drafted += nd as i64;
                } else {
                    nd = spec_draft(&seq, t_len, spec_n as usize, &mut d);
                }
            }
            if nd > 0 {
                let mut arg = [0i32; SPEC_MAX + 1];
                spec_snap
                    .as_mut()
                    .unwrap()
                    .copy_from_slice(&ks[..kper_f * w.n_bound]);
                seq[t_len..t_len + nd].copy_from_slice(&d[..nd]);
                frc_ok = forward(
                    &mut w,
                    &c,
                    &mut cache,
                    &seq[base..],
                    nd + 1,
                    &mut lg,
                    &mut sc,
                    &mut h,
                    &mut br,
                    &mut ks,
                    Some(&mut arg[..nd + 1]),
                )
                .is_ok();
                if frc_ok {
                    let mut m = 0usize;
                    while m < nd && arg[m] == d[m] {
                        m += 1;
                    }
                    if m == nd {
                        // every fed position had true context; state is exact
                        w.cached = base + nd + 1;
                    } else {
                        // the recurrent state absorbed rejected tokens: restore, then replay
                        // only the accepted prefix. The replay rewrites the KV rows too.
                        ks[..kper_f * w.n_bound].copy_from_slice(spec_snap.as_ref().unwrap());
                        w.cached = base;
                        if forward(
                            &mut w,
                            &c,
                            &mut cache,
                            &seq[base..],
                            m + 1,
                            &mut lg,
                            &mut sc,
                            &mut h,
                            &mut br,
                            &mut ks,
                            None,
                        )
                        .is_ok()
                        {
                            w.cached = base + m + 1;
                        }
                    }
                    if dw.trunk.is_some() {
                        hyb_accepted += m as i64;
                        if m == nd {
                            let last = d[nd - 1];
                            if forward(
                                &mut dw,
                                &c,
                                &mut cache,
                                std::slice::from_ref(&last),
                                1,
                                &mut lg,
                                &mut sc,
                                &mut h,
                                &mut br,
                                dks.as_mut().unwrap(),
                                None,
                            )
                            .is_ok()
                            {
                                dw.cached += 1;
                            }
                        } else {
                            dks.as_mut()
                                .unwrap()
                                .copy_from_slice(dsnap.as_ref().unwrap());
                            dw.cached = base;
                            if forward(
                                &mut dw,
                                &c,
                                &mut cache,
                                &seq[base..],
                                m + 1,
                                &mut lg,
                                &mut sc,
                                &mut h,
                                &mut br,
                                dks.as_mut().unwrap(),
                                None,
                            )
                            .is_ok()
                            {
                                dw.cached = base + m + 1;
                            }
                        }
                    }
                    for i in 0..m {
                        emit[emitn] = d[i];
                        emitn += 1;
                    }
                    emit[emitn] = arg[m];
                    emitn += 1;
                }
            } else {
                frc_ok = forward(
                    &mut w,
                    &c,
                    &mut cache,
                    &seq[base..],
                    1,
                    &mut lg,
                    &mut sc,
                    &mut h,
                    &mut br,
                    &mut ks,
                    None,
                )
                .is_ok();
                if frc_ok {
                    w.cached = base + 1;
                    emit[emitn] = argmax(&lg[..c.vocab as usize]);
                    emitn += 1;
                }
                // keep the draft in lockstep through non-drafted steps
                if dw.trunk.is_some()
                    && frc_ok
                    && forward(
                        &mut dw,
                        &c,
                        &mut cache,
                        &seq[base..],
                        1,
                        &mut lg,
                        &mut sc,
                        &mut h,
                        &mut br,
                        dks.as_mut().unwrap(),
                        None,
                    )
                    .is_ok()
                {
                    dw.cached = base + 1;
                }
            }
        } else {
            frc_ok = forward(
                &mut w, &c, &mut cache, &seq, t_len, &mut lg, &mut sc, &mut h, &mut br, &mut ks,
                None,
            )
            .is_ok();
            if frc_ok {
                emit[emitn] = argmax(&lg[..c.vocab as usize]);
                emitn += 1;
            }
        }
        if !frc_ok || emitn == 0 {
            eprintln!("forward pass failed at generation step {g}; aborting.");
            return 1;
        }
        let nxt = emit[emitn - 1];
        if let Some(lp) = &logits_path {
            if g == 0 {
                match fs::File::create(lp) {
                    Ok(mut lf) => {
                        let _ = lf.write_all(lg.as_bytes());
                        println!("wrote {} ({} float32 logits)", lp, c.vocab);
                    }
                    Err(_) => eprintln!("cannot open {} for the logits dump", lp),
                }
            }
        }
        let dt = now_s() - ts;
        t_total += dt;
        let s = cache.stats();
        let req = s.hits + s.misses;
        println!(
            "{:<6} {:<10} {:<12.2} {:<10.1} {:<10.2} {:.3}",
            g,
            nxt,
            dt,
            if req > 0 {
                100.0 * s.hits as f64 / req as f64
            } else {
                0.0
            },
            s.bytes_read as f64 / 1e9,
            1.0 / dt
        );
        let _ = io::stdout().flush();
        expert_s_total += s.load_seconds;
        expert_gb_total += s.bytes_read as f64 / 1e9;
        expert_reqs_total += s.hits + s.misses;
        expert_evict_total += s.evictions;
        for i in 0..emitn {
            if nout >= gen as usize || t_len >= tmax {
                break;
            }
            seq[t_len] = emit[i];
            t_len += 1;
            outtok.push(emit[i]);
            nout += 1;
        }
        if t_len >= tmax {
            break;
        }
        g += 1;
    }

    if let Some(ss) = &save_state {
        if !incremental {
            eprintln!("--save-state needs --incremental; nothing written");
        } else {
            let tsv = now_s();
            let kvpp = (c.n_heads * (c.qk_nope + c.v_head)) as i64;
            let ropepp = c.qk_rope as i64;
            if let Ok(()) = state_save(
                ss,
                &c,
                &seq,
                t_len,
                &ks,
                w.kvc.as_ref().unwrap(),
                w.ropec.as_ref().unwrap(),
                w.n_bound,
                w.n_mla,
                w.kv_cap,
                w.cached,
                kper as i64,
                kvpp,
                ropepp,
            ) {
                let bytes = std::mem::size_of::<StateHdr>() as f64
                    + t_len as f64 * 4.0
                    + kper as f64 * w.n_bound as f64 * 4.0
                    + w.cached as f64 * (kvpp as f64 + ropepp as f64) * w.n_mla as f64 * 4.0;
                println!(
                    "wrote {} ({}, {} positions) in {:.2} s",
                    ss,
                    human(bytes),
                    w.cached,
                    now_s() - tsv
                );
            }
        }
    }

    if dw.trunk.is_some() && hyb_rounds > 0 {
        println!(
            "\nhybrid decode: {} rounds, {} drafted, {} accepted ({:.1}%), mean accepted run {:.2}",
            hyb_rounds,
            hyb_drafted,
            hyb_accepted,
            if hyb_drafted > 0 {
                100.0 * hyb_accepted as f64 / hyb_drafted as f64
            } else {
                0.0
            },
            hyb_accepted as f64 / hyb_rounds as f64
        );
    }
    println!("--------------------------------------------------------------------");
    if nout > 0 {
        println!(
            "{} tokens in {:.1} s, {:.2} s/token average",
            nout,
            t_total,
            t_total / nout as f64
        );
    } else {
        println!("0 tokens in {:.1} s", t_total);
    }

    if let Some(tok) = &tok {
        if nout > 0 {
            let txt = tok.decode(&outtok);
            println!(
                "\n--- generated text ---\n{}\n----------------------\n",
                txt
            );
        }
    }
    println!(
        "PEAK RSS for the whole run: {}   <- quote this, not the plan\n",
        human(peak_rss_bytes())
    );
    cache.report("final step");

    if let Ok(mut f) = fs::File::create(&outp) {
        let _ = write!(f, "{{\"prompt_ids\":[");
        for (i, id) in prompt[..np].iter().enumerate() {
            let _ = write!(f, "{}{}", if i == 0 { "" } else { "," }, id);
        }
        let _ = write!(f, "],\"generated_ids\":[");
        for (i, id) in outtok.iter().enumerate() {
            let _ = write!(f, "{}{}", if i == 0 { "" } else { "," }, id);
        }
        let _ = write!(f, "],\"full_ids\":[");
        for (i, id) in seq[..t_len].iter().enumerate() {
            let _ = write!(f, "{}{}", if i == 0 { "" } else { "," }, id);
        }
        let _ = writeln!(
            f,
            "],\"layers\":{},\"seconds_per_token\":{:.4}}}",
            nl,
            if nout > 0 { t_total / nout as f64 } else { 0.0 }
        );
        println!("\nwrote {}", outp);
    }
    if let Some(td) = &trace_dir {
        let p = Path::new(td).join("expert_hist.json");
        let _ = cache.dump_hist(&p);
        let p = Path::new(td).join("expert_trace.bin");
        let _ = cache.dump_trace(&p);
    }

    // I/O share, measured rather than inferred.
    {
        let trunk_s = if let Some(t) = &w.trunk {
            t.stats().load_seconds
        } else {
            0.0
        };
        let io_s = trunk_s + expert_s_total;
        let share = if t_total > 0.0 {
            100.0 * io_s / t_total
        } else {
            0.0
        };
        println!(
            "I/O share of wall clock: {:.1}%  (trunk {:.1} s + experts {:.1} s of {:.1} s)",
            share, trunk_s, expert_s_total, t_total
        );
        println!("  both figures are WHOLE-RUN totals over {} steps", nout);
        if share > 100.0 {
            println!("  over 100% because trunk reads overlap compute on the reader thread;");
            println!(
                "  {:.1} s of device time was hidden behind arithmetic",
                io_s - t_total
            );
        }
        let retained = expert_reqs_total.saturating_sub(expert_evict_total);
        println!("  experts, whole run: {:.2} GB read | {} of {} requests retained in RAM ({:.2}%) | {} evictions",
            expert_gb_total, retained, expert_reqs_total,
            if expert_reqs_total > 0 { 100.0 * retained as f64 / expert_reqs_total as f64 } else { 0.0 },
            expert_evict_total);
        println!("    (retention = requests - evictions; the raw `hits` counter includes");
        println!("     experts the prefetcher had just read from disk, so it is not a");
        println!("     measure of avoided I/O)\n");
    }

    let drops = ops::expert_drops();
    if drops > 0 {
        eprintln!("\nRUN INVALID: {} routed expert load(s) failed and were dropped from\nthe MoE sum. The token ids above are CORRUPT. Re-run; if this repeats,\nthe shard set or the storage is at fault.", drops);
        return 4;
    }
    0
}

fn main() {
    std::process::exit(run());
}

#[cfg(test)]
mod tests {
    use super::now_s;
    use std::time::Duration;

    #[test]
    fn monotonic_timer_advances() {
        let before = now_s();
        std::thread::sleep(Duration::from_millis(1));
        assert!(now_s() > before);
    }
}

#[cfg(test)]
mod cgroup_tests {
    use super::cgroup_mem_under;

    /// Writes a fake cgroup tree and returns its root.
    fn tree(files: &[(&str, &str)]) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("k3cg-{}", std::process::id()));
        let d = d.join(format!("{:p}", files));
        std::fs::create_dir_all(&d).unwrap();
        for (n, v) in files {
            std::fs::write(d.join(n), v).unwrap();
        }
        d
    }

    fn under(d: &std::path::Path) -> Option<f64> {
        cgroup_mem_under(&[d.display().to_string()]).map(|(cap, cur)| (cap - cur).max(0.0))
    }

    #[test]
    fn v2_cap_reports_headroom_not_the_limit() {
        // 8 GiB cap with 1 GiB already charged leaves 7 GiB, not 8.
        let d = tree(&[
            ("memory.max", "8589934592\n"),
            ("memory.current", "1073741824\n"),
        ]);
        assert_eq!(under(&d), Some(7.0 * 1024.0 * 1024.0 * 1024.0));
    }

    #[test]
    fn v2_literal_max_is_uncapped() {
        // The whole point: `max` must not parse as a number, or an uncapped cgroup would
        // report zero headroom and the engine would refuse to start on an idle machine.
        let d = tree(&[("memory.max", "max\n"), ("memory.current", "1024\n")]);
        assert_eq!(under(&d), None);
    }

    /// cgroup v1 spells "unlimited" as a huge number rather than a word, and it has three
    /// spellings. The page-aligned one sits just below `u64::MAX / 2`, so the obvious
    /// threshold lets it through and reports exabytes of headroom - which shipped here once,
    /// and this case caught it. All three must read as uncapped.
    #[test]
    fn every_v1_unlimited_sentinel_is_uncapped() {
        for sentinel in [
            "9223372036854771712",  // i64::MAX rounded to a page multiple, the common one
            "9223372036854775807",  // i64::MAX
            "18446744073709551615", // u64::MAX
        ] {
            let d = tree(&[
                ("memory.limit_in_bytes", sentinel),
                ("memory.usage_in_bytes", "1024\n"),
            ]);
            assert_eq!(under(&d), None, "sentinel {sentinel} must read as uncapped");
        }
    }

    #[test]
    fn v1_real_limit_is_honoured() {
        let d = tree(&[
            ("memory.limit_in_bytes", "2147483648\n"),
            ("memory.usage_in_bytes", "147483648\n"),
        ]);
        assert_eq!(under(&d), Some(2_000_000_000.0));
    }

    #[test]
    fn usage_above_limit_saturates_to_zero_not_underflow() {
        // Charged over the cap is possible transiently. Unsigned underflow here would report
        // ~18 exabytes free, which is precisely the failure this guard exists to prevent.
        let d = tree(&[("memory.max", "1024\n"), ("memory.current", "4096\n")]);
        assert_eq!(under(&d), Some(0.0));
    }

    #[test]
    fn missing_tree_is_uncapped() {
        assert_eq!(
            cgroup_mem_under(&["/nonexistent/k3/cgroup".to_string()]),
            None
        );
    }
}
