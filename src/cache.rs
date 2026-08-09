// SPDX-License-Identifier: Apache-2.0
//! The streaming expert cache. Port of `src/cache/k3_cache.c` and `k3_cache.h`.
//!
//! THE PROBLEM, IN MEASURED NUMBERS
//!
//! Routed experts are 1.45 TB of the 1.56 TB checkpoint.
//! A decode step selects top-16 in each of 92 MoE layers, so 1,472 experts, 17,547,264
//! bytes each: 25.83 GB of weights per token if nothing is cached.
//! At the ~1.2 GB/s a commodity NVMe device sustains on cold random reads of this size,
//! that alone is about 21 s/token.
//! The cache exists to reduce it, and its hit rate is the single number that decides
//! whether the whole approach works.
//!
//! WHY THE CACHE HOLDS MXFP4 AND NOT FLOATS
//!
//! Dequantised, one expert is 132 MB rather than 17.55 MB.
//! Caching floats would cut the number of resident experts by 7.5x for no benefit,
//! because `matmul_mxfp4` consumes the packed form directly and a matrix-vector product
//! is memory-bound: reading a seventh of the bytes is faster, not slower.
//!
//! REPLACEMENT POLICY
//!
//! LRU with pinning. Two things make plain LRU safe here:
//!
//! - `moe` fetches an expert and immediately uses it, so a slot handed out cannot be
//!   evicted before use as long as capacity exceeds topk. The constructor enforces that
//!   rather than trusting it.
//! - The victim search is a linear scan over slots. At a few hundred slots that is a few
//!   hundred comparisons against a 17.55 MB read; it is not worth a heap.
//!
//! THE HISTOGRAM IS NOT INSTRUMENTATION
//!
//! Which experts are hot is not knowable in advance and is the input to any sensible
//! pinning strategy.
//! The cache counts every request per (layer, expert), 82,432 counters costing 330 KB, so
//! a real workload can be profiled and the hot set pinned permanently.
//! Without that measurement, pinning is guesswork.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use rayon::prelude::*;

use crate::cfg::{Cfg, MAX_TOPK};
use crate::io_util::{AlignedBuf, ST_ALIGN};
use crate::load::ExpertRef;
use crate::ops::{ExpertQ, ExpertSrc};
use crate::st::St;

/// Slot states for `key_of`. EMPTY must stay -1: several places test "< 0" meaning "not
/// holding a key", which both sentinels satisfy. INFLIGHT is distinct so `pick_victim`
/// can refuse to hand out a slot whose read has not landed yet. k3_cache.h:41.
pub const SLOT_EMPTY: i32 = -1;
pub const SLOT_INFLIGHT: i32 = -2;

/// A snapshot of the counters, so callers can read them without the fields being public.
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub bytes_read: u64,
    /// Experts brought resident by the BATCH prefetch rather than by `get`.
    ///
    /// This has to be counted separately or the hit rate becomes a lie.
    /// A prefetched expert is resident by the time `get` asks for it, so `get` records a
    /// hit - but the bytes still came off the disk during this token.
    /// Without this counter the report would show the hit rate climbing while the I/O did
    /// not fall at all.
    /// Effective hit rate is `(hits - prefetch_reads) / requests`. k3_cache.h:70.
    pub prefetch_reads: u64,
    pub load_seconds: f64,
    pub nslot: usize,
    pub slot_bytes: i64,
}

/// Hands one reserved slot of the shared arena to a parallel reader.
///
/// Only used inside `get_many`'s phase 2. The pointer is the arena base; a task is given
/// the index of the slot phase 1 reserved for it and nothing else.
struct SlotWriter {
    base: *mut u8,
    slot_bytes: usize,
}

// SAFETY: `SlotWriter` is a bare pointer into an arena the `Cache` owns and keeps alive
// across the whole parallel region, plus a stride. It grants no aliasing by itself; every
// dereference goes through `slot`, whose contract is discharged at each call site.
unsafe impl Send for SlotWriter {}
unsafe impl Sync for SlotWriter {}

impl SlotWriter {
    /// The base of one slot.
    ///
    /// This deliberately returns a raw pointer rather than a `&mut [u8]`. A method taking
    /// `&self` and returning `&mut` would invent a mutable borrow out of a shared one, so
    /// two tasks holding the same `&SlotWriter` could each obtain a `&mut` to the same
    /// bytes with nothing in the type system objecting. The aliasing here is legitimate,
    /// but the argument for it is the phase-1 reservation, which lives at the call site -
    /// so the call site is where the slice is built and where the argument is written down.
    ///
    /// # Safety
    ///
    /// `slot` must be a slot index that phase 1 reserved for this task and handed to no
    /// other task, and `slot * slot_bytes + slot_bytes` must be within the arena.
    unsafe fn slot_ptr(&self, slot: usize) -> *mut u8 {
        self.base.add(slot * self.slot_bytes)
    }
}

/// One expert reserved by phase 1 and read by phase 2. Bounded by top-k, as in C.
struct Work {
    slot: usize,
    expert: usize,
    r: ExpertRef,
    got: i64,
    pad: i64,
}

/// One `(offset, length)` pair of a slot's payload, as the C code's pointer arithmetic
/// `b + m[i].p_off` expresses it.
#[inline]
fn sub(arena: &[u8], base: usize, off: i64, nbytes: i64) -> &[u8] {
    let s = base + off as usize;
    &arena[s..s + nbytes as usize]
}

pub struct Cache<'a> {
    st: &'a St,
    n_layers: usize,
    n_experts: usize,

    /// `nslot * slot_bytes`, page aligned.
    arena: AlignedBuf,
    slot_bytes: i64,
    nslot: usize,

    /// `[n_layers*n_experts]` -> slot, or -1.
    slot_of: Vec<i32>,
    /// `[nslot]` -> `layer*n_experts+expert`, or `SLOT_EMPTY` / `SLOT_INFLIGHT`.
    key_of: Vec<i32>,
    /// `[nslot]` LRU stamp.
    used_at: Vec<u64>,
    /// `[nslot]` never evict while set.
    pinned: Vec<bool>,
    /// `[nslot]` geometry of the resident expert.
    ref_: Vec<Option<ExpertRef>>,
    /// `[nslot]` where the payload starts in the slot; non-zero only on the O_DIRECT
    /// path, where the read is widened to aligned bounds.
    pad: Vec<i32>,

    clock: u64,

    hits: u64,
    misses: u64,
    evictions: u64,
    bytes_read: u64,
    prefetch_reads: u64,
    load_seconds: f64,
    /// `[n_layers*n_experts]` request counts.
    hist: Vec<u32>,

    /// THE ACCESS TRACE, and why it is worth recording.
    ///
    /// The question this project exists to answer is how much RAM Kimi K3 actually needs,
    /// which is a question about hit rate versus cache size.
    /// Measuring that directly would mean re-running the model once per cache size, and
    /// each run costs real money on a machine big enough to hold the trunk.
    ///
    /// The routing decisions do not depend on the cache at all, so one run yields the
    /// whole curve: record (layer, expert) in request order, then simulate any replacement
    /// policy at any capacity offline.
    /// 1,472 requests per token, 8 bytes each, is 12 KB per token.
    trace: Vec<i32>,

    /// `K3_NOPREFETCH=1` disables the batch path at runtime.
    ///
    /// An A/B between two BUILDS compares two binaries; an A/B on one binary compares one
    /// decision, which is the only way to attribute a timing difference to the prefetch
    /// rather than to the compiler, the layout, or the weather. C nulls the `getmany`
    /// vtable slot; here `get_many` returns 0, which callers already treat as "nothing
    /// prefetched, miss and read it the slow way". k3_cache.c:264.
    batch: bool,
}

impl<'a> Cache<'a> {
    /// `budget_bytes` is the arena size; it is rounded down to whole experts. Fails if
    /// that leaves fewer than `topk + 1` slots, because a smaller cache cannot serve one
    /// token without evicting an expert that is still in use. k3_cache.c:258.
    pub fn new(st: &'a St, cfg: &Cfg, budget_bytes: i64) -> io::Result<Cache<'a>> {
        let batch = std::env::var_os("K3_NOPREFETCH").is_none();
        if !batch {
            eprintln!("k3_cache: batch prefetch DISABLED by K3_NOPREFETCH");
        }
        let n_layers = cfg.n_layers as usize;
        let n_experts = cfg.n_experts as usize;

        // Size a slot from the checkpoint rather than from arithmetic: find any expert and
        // ask how many bytes it actually occupies.
        let mut probe: Option<ExpertRef> = None;
        for l in 0..cfg.n_layers {
            if cfg.is_dense(l) {
                continue;
            }
            if let Ok(r) = crate::load::expert_ref(st, l as usize, 0) {
                probe = Some(r);
                break;
            }
        }
        let probe = match probe {
            Some(p) => p,
            None => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "k3_cache: no routed experts in this shard set",
                ))
            }
        };

        // Room for an O_DIRECT read widened outward to 4096 boundaries at both ends.
        //
        // Round the SLOT STRIDE up to the O_DIRECT alignment, not just the arena base.
        //
        // The allocation below aligns the arena, which aligns slot 0 and nothing else:
        // slot N starts at arena + N*slot_bytes, so every slot is aligned only if
        // slot_bytes is itself a multiple of ST_ALIGN. On the real checkpoint an expert is
        // 17,547,264 bytes, which is exactly 4284 * 4096, so this held BY COINCIDENCE and
        // the engine worked. With any other expert size - another model, a repacked
        // container, or the few-KB experts in fixtures/cache - every O_DIRECT read into
        // every slot after the first returns 0 bytes and the cache silently serves
        // nothing. The fixture deliberately uses a non-conforming expert size so this is
        // gated rather than left to the real checkpoint's coincidence (tests/cache.rs).
        // k3_cache.c:295.
        let align = ST_ALIGN as i64;
        let mut slot_bytes = probe.nbytes + 2 * align;
        slot_bytes = (slot_bytes + align - 1) & !(align - 1);

        // i64 throughout: a negative budget must refuse, not wrap into a huge slot count.
        let nslot_i = budget_bytes / slot_bytes;
        if nslot_i < (cfg.topk + 1) as i64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "k3_cache: budget {:.2} GB gives {} slots of {:.2} MB, but top-{} needs at \
                     least {}. A cache smaller than one token's working set would evict an \
                     expert that is still being multiplied.",
                    budget_bytes as f64 / 1e9,
                    nslot_i,
                    slot_bytes as f64 / 1e6,
                    cfg.topk,
                    cfg.topk + 1
                ),
            ));
        }
        let nslot = nslot_i as usize;

        // Page aligned so the arena can be read into with O_DIRECT unchanged, 2 MB aligned
        // and hugepage-advised for the same reason as the trunk arena: every O_DIRECT
        // expert read pins its destination pages, and a 17.55 MB slot on 4 KB pages is
        // 4,284 pins per read, 1,472 reads per token. `K3_NOHUGE=1` restores 4 KB so the
        // two can be compared on one binary. `AlignedBuf` owns that policy. k3_cache.c:318.
        let want = nslot * slot_bytes as usize;
        let arena = AlignedBuf::new(want).map_err(|e| {
            io::Error::new(
                e.kind(),
                format!(
                    "k3_cache: cannot allocate {:.2} GB arena: {e}",
                    want as f64 / 1e9
                ),
            )
        })?;

        let nkey = n_layers * n_experts;
        Ok(Cache {
            st,
            n_layers,
            n_experts,
            arena,
            slot_bytes,
            nslot,
            slot_of: vec![-1; nkey],
            key_of: vec![SLOT_EMPTY; nslot],
            used_at: vec![0; nslot],
            pinned: vec![false; nslot],
            ref_: (0..nslot).map(|_| None).collect(),
            pad: vec![0; nslot],
            clock: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
            bytes_read: 0,
            prefetch_reads: 0,
            load_seconds: 0.0,
            hist: vec![0; nkey],
            trace: Vec::new(),
            batch,
        })
    }

    /// Pin or unpin whatever slot currently holds this expert. Pinning a resident hot set
    /// is the payoff from the histogram. Returns false if the expert was not resident.
    /// k3_cache.c:370.
    pub fn pin(&mut self, layer: usize, expert: usize, pin: bool) -> bool {
        let key = layer * self.n_experts + expert;
        if key >= self.n_layers * self.n_experts {
            return false;
        }
        let slot = self.slot_of[key];
        if slot < 0 {
            return false;
        }
        self.pinned[slot as usize] = pin;
        true
    }

    /// Load an expert without returning it, so a prefetcher can warm the cache.
    /// k3_cache.c:380.
    pub fn prefetch(&mut self, layer: usize, expert: usize) -> bool {
        self.admit(layer, expert).is_some()
    }

    /// k3_cache.c:385.
    pub fn reset_stats(&mut self) {
        self.hits = 0;
        self.misses = 0;
        self.evictions = 0;
        self.bytes_read = 0;
        self.load_seconds = 0.0;
        // prefetch_reads belongs to the same window as hits and misses.
        //
        // `report` derives the effective hit rate as (hits - prefetch_reads), so both
        // counters must cover the same interval. Resetting one without the other compares
        // a per-window numerator against a since-startup subtrahend, which drives the
        // result negative and clamps it to zero at every cache size.
        self.prefetch_reads = 0;
    }

    /// k3_cache.c:398.
    pub fn report(&self, label: &str) {
        let n = self.hits + self.misses;
        let mut resident = 0usize;
        let mut pinned = 0usize;
        for i in 0..self.nslot {
            if self.key_of[i] >= 0 {
                resident += 1;
            }
            if self.pinned[i] {
                pinned += 1;
            }
        }
        println!("cache [{label}]");
        println!(
            "  slots        : {} of {:.2} MB = {:.2} GB arena ({} resident, {} pinned)",
            self.nslot,
            self.slot_bytes as f64 / 1e6,
            self.nslot as f64 * self.slot_bytes as f64 / 1e9,
            resident,
            pinned
        );
        println!(
            "  requests     : {}  hits {} ({:.2}%)  misses {}  evictions {}",
            n,
            self.hits,
            if n != 0 {
                100.0 * self.hits as f64 / n as f64
            } else {
                0.0
            },
            self.misses,
            self.evictions
        );
        // The prefetch makes the raw hit rate above flattering: an expert the batch read
        // from disk moments earlier is resident by the time get() asks, so it counts as a
        // hit. Report what was actually served from RAM without touching the disk.
        if self.prefetch_reads != 0 {
            let served = self.hits.saturating_sub(self.prefetch_reads);
            // Two calls, one line each: the C printf embeds the newline mid-literal and
            // aligns the continuation under "came" with 18 leading spaces. k3_cache.c:415.
            println!(
                "  of those hits : {} came from the batch prefetch, i.e. read from disk",
                self.prefetch_reads
            );
            println!(
                "                  this token; TRUE resident hit rate {:.2}%",
                if n != 0 {
                    100.0 * served as f64 / n as f64
                } else {
                    0.0
                }
            );
        }
        println!(
            "  read from disk: {:.2} GB in {:.2} s ({:.0} MB/s while loading)",
            self.bytes_read as f64 / 1e9,
            self.load_seconds,
            if self.load_seconds > 0.0 {
                self.bytes_read as f64 / 1e6 / self.load_seconds
            } else {
                0.0
            }
        );
    }

    /// Write the request histogram as JSON, for offline analysis of the hot set.
    /// k3_cache.c:426.
    pub fn dump_hist(&self, path: &Path) -> io::Result<()> {
        let mut f = BufWriter::new(File::create(path)?);
        write!(
            f,
            "{{\"n_layers\":{},\"n_experts\":{},\"counts\":{{",
            self.n_layers, self.n_experts
        )?;
        let mut first = true;
        for l in 0..self.n_layers {
            for e in 0..self.n_experts {
                let v = self.hist[l * self.n_experts + e];
                if v == 0 {
                    continue; // sparse: most are zero
                }
                write!(f, "{}\"{},{}\":{}", if first { "" } else { "," }, l, e, v)?;
                first = false;
            }
        }
        writeln!(f, "}}}}")?;
        f.flush()
    }

    /// Write the access trace as a flat binary file of int32 pairs (layer, expert) in
    /// request order. `tools/sim_cache.py` replays it at any capacity. k3_cache.c:358.
    pub fn dump_trace(&self, path: &Path) -> io::Result<()> {
        if self.trace.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "k3_cache: no access trace recorded",
            ));
        }
        let ntrace = self.trace.len();
        {
            let mut f = BufWriter::new(File::create(path)?);
            for v in &self.trace {
                f.write_all(&v.to_le_bytes())?;
            }
            f.flush()?;
        }
        println!(
            "wrote {}: {} requests ({:.1} KB)",
            path.display(),
            ntrace / 2,
            ntrace as f64 * 4.0 / 1024.0
        );
        Ok(())
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            bytes_read: self.bytes_read,
            prefetch_reads: self.prefetch_reads,
            load_seconds: self.load_seconds,
            nslot: self.nslot,
            slot_bytes: self.slot_bytes,
        }
    }

    /// Resolve a slot to the three (packed, scale) pairs the kernels want. k3_cache.c:19.
    fn fill_q(&self, slot: usize) -> Option<ExpertQ<'_>> {
        let r = self.ref_[slot].as_ref()?;
        // pad is where the expert really begins: an O_DIRECT read starts at the enclosing
        // 4096 boundary, which is at or before the expert's own offset.
        let b = slot * self.slot_bytes as usize + self.pad[slot] as usize;
        let a: &[u8] = &self.arena;
        Some(ExpertQ {
            p1: sub(a, b, r.m[0].p_off, r.m[0].p_bytes),
            s1: sub(a, b, r.m[0].s_off, r.m[0].s_bytes),
            p2: sub(a, b, r.m[1].p_off, r.m[1].p_bytes),
            s2: sub(a, b, r.m[1].s_off, r.m[1].s_bytes),
            p3: sub(a, b, r.m[2].p_off, r.m[2].p_bytes),
            s3: sub(a, b, r.m[2].s_off, r.m[2].s_bytes),
        })
    }

    /// Least recently used unpinned slot. Linear, deliberately: a few hundred comparisons
    /// against a 17.55 MB read is not where the time goes.
    ///
    /// `key_of` has THREE states, not two:
    ///
    /// | value | meaning |
    /// |---|---|
    /// | `>= 0` | holds that key |
    /// | `SLOT_EMPTY` | holds nothing, free to take |
    /// | `SLOT_INFLIGHT` | reserved by a batch prefetch whose read has not finished |
    ///
    /// The third state exists because of a real bug.
    /// The batch prefetch marks a slot empty before reading into it, so that a failed read
    /// cannot leave the slot claiming an expert it does not hold.
    /// But the empty test below is a FAST PATH that returns immediately, ahead of the
    /// pinned check and the LRU scan - so the next expert in the same batch was handed the
    /// SAME slot, several parallel reads wrote into one buffer, and the MoE multiplied
    /// garbage.
    /// It cost one wrong token (65 instead of 2494) on the real model and nothing at all
    /// in the fixtures, because no fixture exercises the streaming cache. k3_cache.c:44.
    fn pick_victim(&self) -> Option<usize> {
        let mut best: Option<usize> = None;
        let mut oldest = u64::MAX;
        for i in 0..self.nslot {
            if self.key_of[i] == SLOT_INFLIGHT {
                continue; // being read into RIGHT NOW
            }
            if self.key_of[i] == SLOT_EMPTY {
                return Some(i); // free, take it
            }
            if self.pinned[i] {
                continue;
            }
            if self.used_at[i] < oldest {
                oldest = self.used_at[i];
                best = Some(i);
            }
        }
        best
    }

    /// Bring (layer, expert) resident and return its slot. k3_cache.c:58.
    fn admit(&mut self, layer: usize, expert: usize) -> Option<usize> {
        let key = layer * self.n_experts + expert;
        let slot = self.slot_of[key];
        if slot >= 0 {
            self.hits += 1;
            self.clock += 1;
            self.used_at[slot as usize] = self.clock;
            return Some(slot as usize);
        }
        self.misses += 1;

        let r = crate::load::expert_ref(self.st, layer, expert).ok()?;
        if r.nbytes > self.slot_bytes {
            eprintln!(
                "k3_cache: L{} expert {} is {} bytes, slot holds {}",
                layer, expert, r.nbytes, self.slot_bytes
            );
            return None;
        }

        let slot = match self.pick_victim() {
            Some(s) => s,
            None => {
                eprintln!(
                    "k3_cache: every slot is pinned, cannot admit L{} expert {}",
                    layer, expert
                );
                return None;
            }
        };
        if self.key_of[slot] >= 0 {
            self.slot_of[self.key_of[slot] as usize] = -1;
            self.evictions += 1;
        }

        let t0 = Instant::now();
        let sb = self.slot_bytes as usize;
        let dst = &mut self.arena[slot * sb..slot * sb + sb];
        let (got, pad) = crate::load::expert_load_direct(self.st, &r, dst).unwrap_or((-1, 0));
        self.load_seconds += t0.elapsed().as_secs_f64();
        if got != r.nbytes {
            eprintln!(
                "k3_cache: short load of L{} expert {} ({} of {})",
                layer, expert, got, r.nbytes
            );
            self.key_of[slot] = SLOT_EMPTY;
            return None;
        }
        self.bytes_read += got as u64;

        self.pad[slot] = pad as i32;
        self.ref_[slot] = Some(r); // after pad: `r` is moved here
        self.key_of[slot] = key as i32;
        self.slot_of[key] = slot as i32;
        self.clock += 1;
        self.used_at[slot] = self.clock;
        Some(slot)
    }
}

impl ExpertSrc for Cache<'_> {
    /// k3_cache.c:230.
    fn get(&mut self, layer: usize, expert: usize) -> Option<ExpertQ<'_>> {
        if layer >= self.n_layers || expert >= self.n_experts {
            eprintln!("k3_cache: out of range L{layer} expert {expert}");
            return None;
        }
        self.hist[layer * self.n_experts + expert] += 1;

        // Record the request before serving it. The trace must reflect what the MODEL
        // asked for, independent of what the cache happened to hold, or replaying it at a
        // different capacity would be meaningless.
        self.trace.push(layer as i32);
        self.trace.push(expert as i32);

        let slot = self.admit(layer, expert)?;
        self.fill_q(slot)
    }

    /// Bring a whole top-k resident, with the reads issued CONCURRENTLY.
    ///
    /// The serial path admits one expert per call, so the drive sees a queue depth of one:
    /// 17.55 MB, wait, repeat, 16 times per layer.
    /// NVMe needs depth to reach rated bandwidth, so that pattern leaves most of the drive
    /// idle. This hands the whole set over at once.
    ///
    /// THREE PHASES, and the split is not cosmetic:
    ///
    /// 1. SERIAL resolve each miss and reserve it a slot. Slot allocation touches the LRU
    ///    bookkeeping, which is shared mutable state and must not race.
    /// 2. PARALLEL do the reads. Every read targets a distinct, already-assigned buffer
    ///    and goes through a positioned read, which takes its offset as an argument and so
    ///    does not touch any shared file position. Nothing here is shared for writing.
    /// 3. SERIAL publish. A slot is registered to its key ONLY after its read succeeded.
    ///
    /// Phase 3 is where the danger was.
    /// Registering the key up front, then reading, would leave a failed read with a slot
    /// that claims to hold an expert it does not - and the next request for that expert
    /// would count a HIT and multiply garbage.
    /// That exact bug existed in the trunk ring and is why the order here is deliberate.
    /// k3_cache.c:126.
    fn get_many(&mut self, layer: usize, experts: &[i32]) -> usize {
        if experts.is_empty() || !self.batch {
            return 0;
        }

        // One entry per expert in a batch prefetch, so it is bounded by top-k.
        let cap = MAX_TOPK;
        let mut w: Vec<Work> = Vec::with_capacity(cap.min(experts.len()));

        // ---- phase 1: reserve, serially ----
        for &id in experts {
            if w.len() >= cap {
                break;
            }
            if id < 0 || id as usize >= self.n_experts {
                continue;
            }
            let e = id as usize;
            let key = layer * self.n_experts + e;
            if self.slot_of[key] >= 0 {
                continue; // already resident
            }
            // the same id twice in one top-k
            if w.iter().any(|x| x.expert == e) {
                continue;
            }

            let r = match crate::load::expert_ref(self.st, layer, e) {
                Ok(r) => r,
                Err(_) => continue,
            };
            if r.nbytes > self.slot_bytes {
                continue;
            }

            let slot = match self.pick_victim() {
                Some(s) => s,
                None => break,
            };
            if self.key_of[slot] >= 0 {
                self.slot_of[self.key_of[slot] as usize] = -1;
                self.evictions += 1;
            }
            // INFLIGHT, not EMPTY. Marking it empty made pick_victim's fast path hand the
            // same slot to the next expert in this very batch.
            self.key_of[slot] = SLOT_INFLIGHT;
            self.clock += 1;
            self.used_at[slot] = self.clock;

            w.push(Work {
                slot,
                expert: e,
                r,
                got: -1,
                pad: 0,
            });
        }
        if w.is_empty() {
            return 0;
        }

        // Issue in DISK-OFFSET order. Experts are not stored id-ordered inside a shard, so
        // sorting by where the bytes actually live turns a scattered set of seeks into a
        // mostly forward sweep. A stable sort by (shard, off) is exactly the permutation
        // the C insertion sort produces. k3_cache.c:161.
        w.sort_by_key(|a| (a.r.shard, a.r.off));

        // ---- phase 2: read, concurrently ----
        let t0 = Instant::now();
        {
            let writer = SlotWriter {
                base: self.arena.as_mut_ptr(),
                slot_bytes: self.slot_bytes as usize,
            };
            let st = self.st;
            w.par_iter_mut().for_each(|it| {
                // SAFETY: phase 1 already reserved this slot exclusively - it was taken
                // from `pick_victim`, which refuses SLOT_INFLIGHT, and immediately marked
                // SLOT_INFLIGHT, so no later reservation in this same batch and no
                // concurrent `admit` can hand it out again. Reservations are unique, so no
                // two tasks receive the same index and the `&mut [u8]` slices are pairwise
                // disjoint. And no reader can observe the slot until phase 3 publishes it:
                // `slot_of` still says -1 for this key, so `get`, `resident` and `fill_q`
                // cannot reach these bytes. The arena outlives the parallel region because
                // it is owned by `self`, which is borrowed for the whole call. This is the
                // same argument the C source makes at k3_cache.c:107-124.
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(writer.slot_ptr(it.slot), writer.slot_bytes)
                };
                let (got, pad) = crate::load::expert_load_direct(st, &it.r, dst).unwrap_or((-1, 0));
                it.got = got;
                it.pad = pad;
            });
        }
        self.load_seconds += t0.elapsed().as_secs_f64();

        // ---- phase 3: publish only what actually arrived ----
        let mut ok = 0usize;
        for it in w {
            if it.got != it.r.nbytes {
                eprintln!(
                    "k3_cache: short prefetch of L{} expert {} ({} of {}); leaving the slot \
                     empty so it cannot be served as a hit",
                    layer, it.expert, it.got, it.r.nbytes
                );
                self.key_of[it.slot] = SLOT_EMPTY; // release the reservation
                continue;
            }
            let key = layer * self.n_experts + it.expert;
            self.ref_[it.slot] = Some(it.r); // partial move; `it.got`/`it.pad` are Copy
            self.pad[it.slot] = it.pad as i32;
            self.key_of[it.slot] = key as i32;
            self.slot_of[key] = it.slot as i32;
            self.clock += 1;
            self.used_at[it.slot] = self.clock;
            self.bytes_read += it.got as u64;
            self.prefetch_reads += 1;
            ok += 1;
        }
        ok
    }

    /// Is this expert already resident, i.e. would `get` serve it with no disk read? Used
    /// by the draft model's cache-only routing to propose tokens without any expert I/O;
    /// if it is resident, `fill_q` hands back the same bytes `get` would. k3_cache.c:218.
    fn resident(&mut self, layer: usize, expert: usize) -> Option<ExpertQ<'_>> {
        if layer >= self.n_layers || expert >= self.n_experts {
            return None;
        }
        let key = layer * self.n_experts + expert;
        let slot = self.slot_of[key];
        if slot < 0 {
            return None;
        }
        self.fill_q(slot as usize)
    }
}
