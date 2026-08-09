// SPDX-License-Identifier: Apache-2.0
//! Stream the resident trunk, so RAM becomes a dial instead of a floor.
//!
//! Port of `src/io/k3_trunk.c` and `src/io/k3_trunk.h`.
//!
//! WHY STREAM THE TRUNK AT ALL (k3_trunk.h:1)
//!   The engine holds 110 GB of trunk plus 4.70 GB of embed/lm_head. That is the floor
//!   that forces a 128 GB machine. Quantising it down is the obvious idea and it is the
//!   wrong one: Kimi K3's technical report says the experts are MXFP4 with
//!   quantisation-aware training "while all non-expert components remain in higher
//!   precision". That list IS this trunk. Streaming costs zero error: the bytes are the
//!   checkpoint's own bytes.
//!
//! WHY IT IS AFFORDABLE
//!   The trunk access order is FIXED: layer 0, 1, ..., 92, every single token, so the
//!   next read is always known in advance. `prefetch` hands layer L+1 to a reader thread
//!   while the main thread computes on layer L.
//!
//! WHY LRU WOULD BE THE WORST POLICY
//!   A cyclic sequential scan is the classic LRU pathology. With N < 93 slots, by the
//!   time the walk returns to layer 0 it is exactly the least recently used thing and has
//!   just been evicted, so the hit rate is ZERO. This therefore PINS a prefix of layers
//!   and streams the rest through a small ring: pin K layers and the hit rate is exactly
//!   K/93, deterministically.

use crate::bind::{widen_bytes, LayerPlan};
use crate::cfg::Cfg;
use crate::io_util::{open_direct, pread_full, AlignedBuf};
use crate::ops::LayerW;
use crate::st::Dtype;
use std::fs::File;
use std::io::{Error, ErrorKind, Result};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

/// `pack_trunk.py` pads runs to this so O_DIRECT works. k3_trunk.h:59.
pub const TRUNK_ALIGN: usize = 4096;

/// One tensor recorded in `trunk.json`. k3_trunk.h:61.
struct TrunkTensor {
    name: String,
    off: i64, // byte offset WITHIN the layer run.
    nbytes: i64,
    dtype: Dtype,
}

/// One layer's run: where it sits in `trunk.bin` and its tensor map. k3_trunk.h:68.
struct TrunkLayer {
    file_off: i64,
    nbytes: i64,
    tensors: Vec<TrunkTensor>,
}

/// Bookkeeping + the asynchronous-reader protocol + stats, all behind one mutex so the
/// main thread and the reader thread coordinate through one lock (mirroring C's single
/// `K3TrunkIO.mu`). The reader touches only the protocol fields; the main thread touches
/// `layer_of`/`slot_of`/`ring`/stats.
struct Inner {
    layer_of: Vec<i32>, // [nslot] which layer occupies each ring slot, -1 empty.
    slot_of: Vec<i32>,  // [n_layers] -1 when not resident.
    ring: usize,        // next ring slot to reuse.
    // reader protocol
    busy: bool,
    done: bool,
    io_layer: i32, // -1 = idle.
    io_slot: i32,
    io_result: i32,
    // stats
    hits: u64,
    misses: u64,
    bytes_read: u64,
    load_seconds: f64,
    prefetch_hits: u64,
}

/// The parts the reader thread and the main thread share: the file, the arena, the
/// per-layer metadata, and the mutex/condvar. Held in an `Arc` so the reader thread can
/// outlive the `Trunk` constructor's stack frame.
struct Shared {
    file: File,
    layers: Vec<TrunkLayer>,
    arena: AlignedBuf,
    slot_bytes: i64,
    inner: Mutex<Inner>,
    cv: Condvar,
    stop: AtomicBool,
}

/// Stats snapshot, copied out under the lock. k3_trunk.h:75.
#[derive(Clone, Copy, Debug, Default)]
pub struct TrunkStats {
    pub pinned: usize,
    pub ring: usize,
    pub slot_bytes: i64,
    pub hits: u64,
    pub misses: u64,
    pub bytes_read: u64,
    pub load_seconds: f64,
    pub prefetch_hits: u64,
}

/// The streaming trunk. `bind` takes `&self` because the C call site binds layer L and
/// then prefetches L+1 while the bound views are still live; interior mutability backs
/// the bookkeeping and the arena.
pub struct Trunk {
    shared: Arc<Shared>,
    npin: usize,
    nslot: usize,
    slot_bytes: i64,
    widen_bytes: i64,
    /// Pinned layers: exact-size allocations, written once during `open`, read-only
    /// after. Safe to hand out `&[u8]` into these without `UnsafeCell`.
    pin: Vec<AlignedBuf>,
    /// The reader handle is `None` when the ring has fewer than 2 slots.
    io: Option<std::thread::JoinHandle<()>>,
}

impl Trunk {
    /// `budget_bytes` sizes the slot array. Layers 0..K-1 are pinned, where K is as large
    /// as the budget allows minus a small streaming ring. k3_trunk.c:98.
    pub fn open(dir: &Path, c: &Cfg, budget_bytes: i64) -> Result<Trunk> {
        let trunk_json = dir.join("trunk.json");
        let txt = std::fs::read_to_string(&trunk_json).map_err(|_| {
            Error::new(
                ErrorKind::NotFound,
                format!("k3_trunk: cannot read {}", trunk_json.display()),
            )
        })?;
        let root: serde_json::Value = serde_json::from_str(&txt).map_err(|_| {
            Error::new(
                ErrorKind::InvalidData,
                format!("k3_trunk: {} is not valid JSON", trunk_json.display()),
            )
        })?;

        let jl = root
            .get("layers")
            .and_then(|v| v.as_array())
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, "k3_trunk: no layers array"))?;
        let n_layers = jl.len();
        if n_layers == 0 {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "k3_trunk: empty layers array",
            ));
        }
        let mut layers: Vec<TrunkLayer> = Vec::with_capacity(n_layers);
        for (i, e) in jl.iter().enumerate() {
            let file_off = e.get("file_off").and_then(|v| v.as_i64()).unwrap_or(0);
            let nbytes = e.get("nbytes").and_then(|v| v.as_i64()).unwrap_or(0);
            let ts = e
                .get("tensors")
                .and_then(|v| v.as_object())
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::InvalidData,
                        format!("k3_trunk: layer {} has no tensors", i),
                    )
                })?;
            let mut tensors: Vec<TrunkTensor> = Vec::with_capacity(ts.len());
            for (name, o) in ts {
                let off = o.get("off").and_then(|v| v.as_i64()).unwrap_or(0);
                let t_nbytes = o.get("nbytes").and_then(|v| v.as_i64()).unwrap_or(0);
                let dtype = o
                    .get("dtype")
                    .and_then(|v| v.as_str())
                    .map(dt_of)
                    .unwrap_or(Dtype::Unknown);
                tensors.push(TrunkTensor {
                    name: name.clone(),
                    off,
                    nbytes: t_nbytes,
                    dtype,
                });
            }
            layers.push(TrunkLayer {
                file_off,
                nbytes,
                tensors,
            });
        }

        // Open trunk.bin, O_DIRECT when available. k3_trunk.c:157.
        let trunk_bin = dir.join("trunk.bin");
        let df = open_direct(&trunk_bin)?;
        let direct = df.direct;
        // If the json reports an align that is not TRUNK_ALIGN, a trunk packed before the
        // alignment change cannot be read with O_DIRECT. Fall back to buffered. k3_trunk.c:169.
        let want_align = root.get("align").and_then(|v| v.as_i64()).unwrap_or(0);
        let file = if direct && want_align != TRUNK_ALIGN as i64 {
            eprintln!(
                "k3_trunk: trunk.json reports align {}, expected {}; falling back to buffered reads (repack to enable O_DIRECT)",
                want_align, TRUNK_ALIGN
            );
            File::open(&trunk_bin)?
        } else {
            df.file
        };
        let direct = if direct && want_align != TRUNK_ALIGN as i64 {
            false
        } else {
            direct
        };

        let widen = widen_bytes(c) as i64;
        let mut total: i64 = 0;
        for l in &layers {
            total += l.nbytes;
        }

        // Two ring slots: the layer being computed on, plus one asynchronous read in
        // flight. This is a REQUEST, not a guarantee. k3_trunk.c:193.
        const RING_WANT: usize = 2;
        let mut ring_slot: i64 = 0;
        let mut spent: i64 = 0;
        let mut npin: usize = 0;
        let mut ring = RING_WANT;

        // Ring size and pin count are mutually dependent: a smaller ring frees budget,
        // which pins more layers, which can shrink the ring again. Iterate to a fixed
        // point. It converges in two or three passes and is monotone. k3_trunk.c:220.
        for _pass in 0..4 {
            let mut big: i64 = 0;
            for i in npin..n_layers {
                if layers[i].nbytes > big {
                    big = layers[i].nbytes;
                }
            }
            if big == 0 {
                big = layers[n_layers - 1].nbytes; // all pinned
            }
            let mut rs = align_up(big, TRUNK_ALIGN as i64);
            rs += widen;
            rs = align_up(rs, 4096);

            // The ring itself must fit the budget before any layer is pinned. Drop to
            // one slot rather than overshoot. k3_trunk.c:234.
            ring = RING_WANT;
            while ring > 1 && (ring as i64) * rs > budget_bytes {
                ring -= 1;
            }

            let mut sp = (ring as i64) * rs;
            let mut np = 0;
            while np < n_layers {
                let need = layers[np].nbytes + widen;
                if sp + need > budget_bytes {
                    break;
                }
                sp += need;
                np += 1;
            }
            if np >= n_layers {
                np = n_layers;
            }
            if rs == ring_slot && np == npin {
                ring_slot = rs;
                spent = sp;
                break;
            }
            ring_slot = rs;
            npin = np;
            spent = sp;
        }

        // Pinned layers: exact-size allocations. k3_trunk.c:256.
        let mut pin: Vec<AlignedBuf> = Vec::with_capacity(npin);
        for i in 0..npin {
            let need = align_up(layers[i].nbytes, TRUNK_ALIGN as i64) as usize + widen as usize;
            pin.push(AlignedBuf::new(need)?);
        }

        // The ring arena: one allocation of RING * slot_bytes. k3_trunk.c:265.
        let arena_bytes = ring * ring_slot as usize;
        let arena = AlignedBuf::new(arena_bytes)?;

        let layer_of = vec![-1i32; ring];
        let slot_of = vec![-1i32; n_layers];

        let inner = Inner {
            layer_of,
            slot_of,
            ring: 0,
            busy: false,
            done: false,
            io_layer: -1,
            io_slot: -1,
            io_result: 0,
            hits: 0,
            misses: 0,
            bytes_read: 0,
            load_seconds: 0.0,
            prefetch_hits: 0,
        };

        let shared = Arc::new(Shared {
            file,
            layers,
            arena,
            slot_bytes: ring_slot,
            inner: Mutex::new(inner),
            cv: Condvar::new(),
            stop: AtomicBool::new(false),
        });

        // THE READER IS STARTED ONLY WHEN THERE ARE AT LEAST TWO SLOTS, and this is a
        // correctness requirement rather than an optimisation.
        //
        // k3_trunk_prefetch claims tr->ring for the incoming layer. With one slot,
        // tr->ring is necessarily the slot k3_trunk_bind just returned to the caller, so
        // the worker preads layer L+1 straight over layer L's bytes while the caller is
        // still computing on them. Nothing detects it: the read succeeds, no bound
        // pointer changes, and the run completes and emits fluent, wrong tokens.
        //
        // Measured on the released checkpoint. With one slot and the reader running, the
        // same prompt that gives 17374 20829 10 427 414 1008 606 142957 instead produced
        // 32609 2329 146429 2539 11 152834 44449 7569, with no diagnostic of any kind.
        //
        // With the reader absent, io_wait returns 0 and prefetch returns immediately,
        // which is exactly the synchronous path this file had before the reader existed.
        // k3_trunk.c:274-289.
        let io = if ring >= 2 {
            let shared2 = Arc::clone(&shared);
            Some(std::thread::spawn(move || reader_main(shared2)))
        } else {
            None
        };

        // The stdout messages the C prints. k3_trunk.c:309.
        println!(
            "trunk stream: {:.2} GB packed, {}/{} layers PINNED ({:.2} GB), ring {} x {:.2} GB",
            total as f64 / 1e9,
            npin,
            n_layers,
            (spent - (ring as i64) * ring_slot) as f64 / 1e9,
            ring,
            ring_slot as f64 / 1e9
        );
        println!(
            "              reads use {}",
            if direct {
                "O_DIRECT (page cache bypassed)"
            } else {
                "buffered I/O"
            }
        );
        if ring < RING_WANT {
            println!(
                "              ring held at {} slot: a second slot needs {:.2} GB and the trunk budget is {:.2} GB,\n              so reads are NOT overlapped with compute. Raise --trunk-gb above {:.2} GB to enable it.",
                ring,
                (RING_WANT as f64) * ring_slot as f64 / 1e9,
                budget_bytes as f64 / 1e9,
                (RING_WANT as f64) * ring_slot as f64 / 1e9
            );
        }
        println!(
            "              deterministic hit rate {:.1}% (a cyclic scan defeats LRU, so a pinned prefix is used instead)",
            100.0 * npin as f64 / n_layers as f64
        );

        Ok(Trunk {
            shared,
            npin,
            nslot: ring,
            slot_bytes: ring_slot,
            widen_bytes: widen,
            pin,
            io,
        })
    }

    /// Make layer L resident and return views into its slot. Takes `&self`, not
    /// `&mut self`, because the C call site binds layer L and then prefetches L+1 while
    /// the bound views are still live. k3_trunk.c:453.
    ///
    /// SAFETY argument for the `&'r [u8]` handed out of the arena: `bind` waits for any
    /// in-flight read of the slot it is about to return (via `io_wait`), so the slot's
    /// bytes are fully written and stable. `prefetch` only ever targets a DIFFERENT ring
    /// slot (it claims `ring`, which `bind` just advanced past), so the reader thread
    /// never writes to the slot the caller is computing from. A pinned layer's buffer is
    /// never written after `open`. The `LayerW<'_>` therefore borrows bytes that no
    /// other thread mutates for the duration of the borrow.
    pub fn bind(&self, c: &Cfg, layer: usize) -> Result<LayerW<'_>> {
        if layer >= self.shared.layers.len() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "k3_trunk: layer out of range",
            ));
        }

        let (base_ptr, widen_ptr, slot_run_bytes) = self.acquire_slot(layer)?;

        // Build the plan for this layer (same wide/narrow decisions as from_shards) and
        // widen the bf16 vectors into the slot's widen area. k3_trunk.c:494.
        let find = |name: &str| -> Option<(i64, i64, Dtype)> { self.find_in_layer(layer, name) };
        let mut plan = LayerPlan::build(c, layer, &find)?;

        // SAFETY: base_ptr points into either a pinned buffer (never written after open)
        // or a ring slot whose in-flight read has completed (acquire_slot waited for it).
        // The widen area follows the run, aligned to TRUNK_ALIGN. Both live for the
        // duration of the LayerW borrow (tied to &self).
        let run: &[u8] = unsafe { std::slice::from_raw_parts(base_ptr, slot_run_bytes as usize) };
        let widen_area: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(widen_ptr, self.widen_bytes as usize) };
        plan.widen(run, widen_area)?;

        Ok(plan.views(c, run, widen_area))
    }

    /// Start an asynchronous read of layer L into its slot, if it is not resident. Safe
    /// to call for a layer that is pinned or already loaded (it becomes a no-op).
    /// k3_trunk.c:508.
    pub fn prefetch(&self, layer: usize) {
        if layer >= self.shared.layers.len() || layer < self.npin {
            return;
        }
        if self.io.is_none() {
            return; // no reader thread; synchronous path only.
        }
        // Already resident in a ring slot or slot_of? k3_trunk.c:511.
        {
            let inner = self.shared.inner.lock().unwrap();
            if inner.slot_of[layer] >= 0 {
                return;
            }
            for i in 0..self.nslot {
                if inner.layer_of[i] == layer as i32 {
                    return;
                }
            }
        }

        let mut inner = self.shared.inner.lock().unwrap();
        if inner.busy || inner.slot_of[layer] >= 0 {
            return;
        }
        let slot = inner.ring;
        inner.ring = (inner.ring + 1) % self.nslot;
        let prev = inner.layer_of[slot];
        if prev >= 0 {
            inner.slot_of[prev as usize] = -1;
        }
        inner.layer_of[slot] = -1; // mark EMPTY before reading into it. k3_trunk.c:523.
        inner.io_layer = layer as i32;
        inner.io_slot = slot as i32;
        inner.done = false;
        inner.busy = true;
        self.shared.cv.notify_one();
        drop(inner);
    }

    /// Print a human-readable report. k3_trunk.c:532.
    pub fn report(&self, label: &str) {
        let inner = self.shared.inner.lock().unwrap();
        let n = inner.hits + inner.misses;
        println!("trunk [{}]", label);
        println!(
            "  pinned {}/{} layers, ring {} slots",
            self.npin,
            self.shared.layers.len(),
            self.nslot
        );
        println!(
            "  binds {}, hits {} ({:.1}%), reads {}",
            n,
            inner.hits,
            if n > 0 {
                100.0 * inner.hits as f64 / n as f64
            } else {
                0.0
            },
            inner.misses
        );
        println!(
            "  read {:.2} GB in {:.2} s ({:.0} MB/s)",
            inner.bytes_read as f64 / 1e9,
            inner.load_seconds,
            if inner.load_seconds > 0.0 {
                inner.bytes_read as f64 / 1e6 / inner.load_seconds
            } else {
                0.0
            }
        );
        let _ = inner.prefetch_hits; // tracked; not in the C report's device-rate lines.
    }

    pub fn stats(&self) -> TrunkStats {
        let inner = self.shared.inner.lock().unwrap();
        TrunkStats {
            pinned: self.npin,
            ring: self.nslot,
            slot_bytes: self.slot_bytes,
            hits: inner.hits,
            misses: inner.misses,
            bytes_read: inner.bytes_read,
            load_seconds: inner.load_seconds,
            prefetch_hits: inner.prefetch_hits,
        }
    }

    pub fn n_layers(&self) -> usize {
        self.shared.layers.len()
    }

    /// Acquire the run buffer for layer L: return `(base_ptr, widen_ptr, run_bytes)`.
    /// For a pinned layer, the base is the pin buffer and it is loaded once on first
    /// touch. For a streaming layer, wait for any in-flight prefetch, or claim a ring
    /// slot and load synchronously. k3_trunk.c:453.
    fn acquire_slot(&self, layer: usize) -> Result<(*const u8, *mut u8, i64)> {
        if layer < self.npin {
            // Pinned: loaded once, kept forever. k3_trunk.c:460.
            {
                let mut inner = self.shared.inner.lock().unwrap();
                if inner.slot_of[layer] >= 0 {
                    inner.hits += 1;
                } else {
                    drop(inner);
                    let base = self.pin[layer].as_ptr();
                    self.load_run(layer, base as *mut u8)?;
                    let mut inner = self.shared.inner.lock().unwrap();
                    inner.slot_of[layer] = layer as i32;
                    inner.misses += 1;
                }
            }
            let run_bytes = align_up(self.shared.layers[layer].nbytes, TRUNK_ALIGN as i64);
            let base = self.pin[layer].as_ptr();
            let widen_ptr = unsafe { (base as *mut u8).add(run_bytes as usize) };
            return Ok((base, widen_ptr, run_bytes));
        }

        // Streaming. Wait for an in-flight prefetch of this layer. k3_trunk.c:471.
        let prefetched = self.io_wait(layer);
        if prefetched < 0 {
            return Err(Error::other("k3_trunk: async read failed"));
        }

        let slot = if prefetched > 0 {
            let inner = self.shared.inner.lock().unwrap();
            inner.slot_of[layer] as usize
        } else {
            // Not prefetched: scan the ring for this layer, or claim a new slot.
            // k3_trunk.c:476.
            let mut inner = self.shared.inner.lock().unwrap();
            let mut slot = -1i32;
            for i in 0..self.nslot {
                if inner.layer_of[i] == layer as i32 {
                    slot = i as i32;
                    break;
                }
            }
            if slot >= 0 {
                inner.hits += 1;
                slot as usize
            } else {
                let s = inner.ring;
                inner.ring = (inner.ring + 1) % self.nslot;
                let prev = inner.layer_of[s];
                if prev >= 0 {
                    inner.slot_of[prev as usize] = -1;
                }
                inner.layer_of[s] = -1; // mark EMPTY before reading into it. k3_trunk.c:485.
                drop(inner);
                let base = self.arena_slot_ptr(s) as *mut u8;
                self.load_run(layer, base)?;
                let mut inner = self.shared.inner.lock().unwrap();
                inner.layer_of[s] = layer as i32;
                inner.slot_of[layer] = s as i32;
                inner.misses += 1;
                s
            }
        };

        let base = self.arena_slot_ptr(slot);
        let run_bytes = align_up(self.shared.layers[layer].nbytes, TRUNK_ALIGN as i64);
        let widen_ptr = unsafe { (base as *mut u8).add(run_bytes as usize) };
        Ok((base, widen_ptr, run_bytes))
    }

    /// Raw pointer to the start of ring slot `s`.
    fn arena_slot_ptr(&self, slot: usize) -> *const u8 {
        unsafe {
            self.shared
                .arena
                .as_ptr()
                .add(slot * self.shared.slot_bytes as usize)
        }
    }

    /// Wait for an in-flight read of layer L to complete. Returns 1 if a read completed,
    /// 0 if no read was in flight for L, -1 on read failure. k3_trunk.c:430.
    fn io_wait(&self, layer: usize) -> i32 {
        if self.io.is_none() {
            return 0;
        }
        let mut inner = self.shared.inner.lock().unwrap();
        if (inner.busy || inner.done) && inner.io_layer == layer as i32 {
            while !inner.done && !self.shared.stop.load(Ordering::Relaxed) {
                inner = self.shared.cv.wait(inner).unwrap();
            }
            let rc = inner.io_result;
            let slot = inner.io_slot;
            if !self.shared.stop.load(Ordering::Relaxed) && rc == 0 {
                inner.layer_of[slot as usize] = layer as i32;
                inner.slot_of[layer] = slot;
                inner.misses += 1;
            }
            inner.done = false;
            return if rc == 0 { 1 } else { -1 };
        }
        0
    }

    /// Read one layer's run into `dst` (a raw pointer into a pin buffer or ring slot).
    /// k3_trunk.c:387. Updates stats under the lock.
    fn load_run(&self, layer: usize, dst: *mut u8) -> Result<()> {
        load_run_into(
            &self.shared.file,
            &self.shared.layers,
            layer,
            dst,
            &self.shared.inner,
        )
    }

    /// Look up a tensor by name within layer L's run. Returns `(off, nbytes, dtype)`.
    /// k3_trunk.c:86.
    fn find_in_layer(&self, layer: usize, name: &str) -> Option<(i64, i64, Dtype)> {
        let lay = &self.shared.layers[layer];
        for t in &lay.tensors {
            if t.name == name {
                return Some((t.off, t.nbytes, t.dtype));
            }
        }
        None
    }
}

impl Drop for Trunk {
    fn drop(&mut self) {
        // Stop the reader thread. k3_trunk.c:332.
        if self.io.is_some() {
            self.shared.stop.store(true, Ordering::Relaxed);
            self.shared.cv.notify_all();
            if let Some(handle) = self.io.take() {
                let _ = handle.join();
            }
        }
    }
}

/// The reader thread loop. Mirrors C `trunk_io_main` (k3_trunk.c:403): wait for a
/// request, read the layer into the assigned ring slot, signal done, repeat until stop.
fn reader_main(shared: Arc<Shared>) {
    loop {
        let mut inner = shared.inner.lock().unwrap();
        while !inner.busy && !shared.stop.load(Ordering::Relaxed) {
            inner = shared.cv.wait(inner).unwrap();
        }
        if shared.stop.load(Ordering::Relaxed) {
            return;
        }
        let layer = inner.io_layer as usize;
        let slot = inner.io_slot as usize;
        drop(inner);

        // SAFETY: the arena is owned by `shared`; slot `slot` is reserved for this read
        // (the main thread marked it INFLIGHT before signalling). No other thread reads
        // or writes this slot until the read completes and `done` is signalled.
        let dst =
            unsafe { shared.arena.as_ptr().add(slot * shared.slot_bytes as usize) as *mut u8 };
        let rc = load_run_into(&shared.file, &shared.layers, layer, dst, &shared.inner);

        let mut inner = shared.inner.lock().unwrap();
        inner.io_result = if rc.is_ok() { 0 } else { -1 };
        inner.done = true;
        inner.busy = false;
        shared.cv.notify_all();
    }
}

/// Read one layer's run into `dst`, updating stats under the lock. k3_trunk.c:387.
/// Shared between the main thread (synchronous loads) and the reader thread.
fn load_run_into(
    file: &File,
    layers: &[TrunkLayer],
    layer: usize,
    dst: *mut u8,
    inner: &Mutex<Inner>,
) -> Result<()> {
    let lay = &layers[layer];
    let t0 = Instant::now();
    let mut got: i64 = 0;
    let buf: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(dst, lay.nbytes as usize) };
    while got < lay.nbytes {
        let r = pread_full(
            file,
            &mut buf[got as usize..lay.nbytes as usize],
            (lay.file_off + got) as u64,
        )?;
        if r == 0 {
            return Err(Error::new(
                ErrorKind::UnexpectedEof,
                format!("k3_trunk: short read on layer {}", layer),
            ));
        }
        got += r as i64;
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let mut g = inner.lock().unwrap();
    g.load_seconds += elapsed;
    g.bytes_read += got as u64;
    Ok(())
}

/// Parse the dtype string the way C's `dt_of` does. k3_trunk.c:59.
fn dt_of(s: &str) -> Dtype {
    match s {
        "BF16" => Dtype::Bf16,
        "F32" => Dtype::F32,
        "U8" => Dtype::U8,
        "F16" => Dtype::F16,
        "I8R" => Dtype::I8R,
        _ => Dtype::Unknown,
    }
}

/// Align `x` up to `a`. k3_trunk.c:227.
fn align_up(x: i64, a: i64) -> i64 {
    (x + a - 1) & !(a - 1)
}
