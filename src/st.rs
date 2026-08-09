// SPDX-License-Identifier: Apache-2.0
//! Safetensors reader for the real Kimi K3 checkpoint.
//!
//! Port of `src/io/k3_st.c` and `k3_st.h`.
//!
//! FORMAT, verified against `model-00002-of-000096.safetensors` (k3_st.h:7):
//! `[8 bytes little-endian N][N bytes of JSON header][tensor data]`.
//! Each header entry: `{"dtype": ..., "shape": [...], "data_offsets": [start, end]}`.
//! `data_offsets` are relative to the END of the header, so the absolute file offset
//! is `8 + N + start`. The data region is fully contiguous with no gaps between tensors.
//!
//! This port replaces the C hand-rolled JSON scanner and FNV-1a open-addressed hash table
//! with `serde_json` and a `HashMap<String, usize>` index into a `Vec<Tensor>`. Every
//! observable behaviour is preserved: shards are visited in sorted filename order,
//! `__metadata__` is skipped, a duplicate tensor name is a hard error, the byte span
//! must equal `numel * elemsize`, and `read_aligned` widens to aligned bounds on the
//! direct path.

use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

use crate::io_util::{open_direct, pread_full, ST_ALIGN};

/// The packed trunk's per-row int8 draft format: each row is `[f32 scale][int8 * cols]`.
/// It only appears in a draft trunk written by `tools/int8_trunk.py`. k3_st.h:33.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dtype {
    Unknown,
    U8,
    Bf16,
    F16,
    F32,
    I8R,
}

impl Dtype {
    /// `U8`/`I8R` 1, `BF16`/`F16` 2, `F32` 4, `Unknown` 0.
    pub fn elemsize(self) -> usize {
        match self {
            Dtype::U8 | Dtype::I8R => 1,
            Dtype::Bf16 | Dtype::F16 => 2,
            Dtype::F32 => 4,
            Dtype::Unknown => 0,
        }
    }

    /// Parse a dtype string as the C `dtype_of` does. k3_st.c:51.
    pub fn from_str_name(s: &str) -> Dtype {
        match s {
            "U8" => Dtype::U8,
            "BF16" => Dtype::Bf16,
            "F16" => Dtype::F16,
            "F32" => Dtype::F32,
            "I8R" => Dtype::I8R,
            _ => Dtype::Unknown,
        }
    }
}

/// One indexed tensor. `off` is the ABSOLUTE byte offset within its shard file.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub name: String,
    pub shard: usize,
    pub dtype: Dtype,
    pub shape: Vec<i64>,
    pub off: i64,
    pub nbytes: i64,
}

impl Tensor {
    /// Product of all shape dims, or 1 for a scalar (`shape == []`). k3_st.c:44.
    pub fn numel(&self) -> i64 {
        let mut n: i64 = 1;
        for &d in &self.shape {
            n *= d;
        }
        n
    }
}

/// An open safetensors store: every shard indexed, every tensor addressable by name.
pub struct St {
    /// Buffered descriptor per shard (`K3St.fd[]`). Used by `read` and `read_f32`.
    files: Vec<File>,
    /// O_DIRECT descriptor per shard (`K3St.dfd[]`), or `None` when unavailable.
    dfiles: Vec<Option<File>>,
    #[allow(dead_code)]
    paths: Vec<PathBuf>,
    tensors: Vec<Tensor>,
    index: HashMap<String, usize>,
}

impl St {
    /// Open every `*.safetensors` in `dir` and index every tensor. k3_st.c:364.
    pub fn open(dir: &Path) -> io::Result<St> {
        // Collect and sort shard filenames so shard indices are stable across runs and
        // machines; readdir order is not. k3_st.c:385.
        let mut shard_paths: Vec<PathBuf> = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.ends_with(".safetensors") {
                shard_paths.push(entry.path());
            }
        }
        shard_paths.sort();

        if shard_paths.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("k3_st: no .safetensors files in {}", dir.display()),
            ));
        }

        let nshard = shard_paths.len();
        let mut files = Vec::with_capacity(nshard);
        let mut dfiles: Vec<Option<File>> = Vec::with_capacity(nshard);
        let mut tensors: Vec<Tensor> = Vec::new();
        let mut index: HashMap<String, usize> = HashMap::new();

        for (shard, path) in shard_paths.iter().enumerate() {
            // Buffered descriptor, kept for `read`/`read_f32`. k3_st.c:199.
            let file = File::open(path).map_err(|e| {
                io::Error::new(
                    e.kind(),
                    format!("k3_st: cannot open {}: {}", path.display(), e),
                )
            })?;

            // Read the 8-byte LE header length. k3_st.c:202.
            let mut lenbuf = [0u8; 8];
            let got = pread_full(&file, &mut lenbuf, 0)?;
            if got != 8 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("k3_st: {} is too short for a header length", path.display()),
                ));
            }
            let hlen = u64::from_le_bytes(lenbuf);

            // Bounds-check the header against the file size. k3_st.c:210.
            let fsize = file.metadata()?.len() as i64;
            if hlen == 0 || fsize < 8 + hlen as i64 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "k3_st: {} header length {} is impossible (file {} bytes)",
                        path.display(),
                        hlen,
                        fsize
                    ),
                ));
            }

            // Read the full header bytes. k3_st.c:217.
            let mut json = vec![0u8; hlen as usize];
            let mut filled = 0usize;
            while filled < json.len() {
                let n = pread_full(&file, &mut json[filled..], 8 + filled as u64)?;
                if n == 0 {
                    break;
                }
                filled += n;
            }
            if filled != json.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("k3_st: short read of {} header", path.display()),
                ));
            }

            // `data_offsets` are relative to the end of the header. k3_st.c:231.
            let base: i64 = 8 + hlen as i64;

            // Parse the JSON. serde_json replaces the hand scanner; the entry order in a
            // serde_json Value::Object matches source order for the default parser, and
            // the C code does not depend on order (it scans every key), so this is a
            // strict superset.
            let root: serde_json::Value = serde_json::from_slice(&json).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("k3_st: {} header is not valid JSON: {}", path.display(), e),
                )
            })?;
            let obj = root.as_object().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("k3_st: {} header is not a JSON object", path.display()),
                )
            })?;

            for (name, val) in obj {
                // __metadata__ is not a tensor; its shape is arbitrary. k3_st.c:256.
                if name == "__metadata__" {
                    continue;
                }
                let entry = val.as_object().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("k3_st: {}: entry {} is not an object", path.display(), name),
                    )
                })?;

                let t = build_tensor(entry, name, shard, base, path)?;

                // Consistency: the byte span must equal elements times element size. A
                // mismatch means the shape and the data disagree, and every later read
                // of this tensor would be silently misaligned. Refuse rather than load
                // it. k3_st.c:316.
                let want = t.numel() * t.dtype.elemsize() as i64;
                if t.nbytes != want {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "k3_st: {}: {} spans {} bytes but shape implies {}",
                            path.display(),
                            name,
                            t.nbytes,
                            want
                        ),
                    ));
                }
                // The C code checks `base + o1 > fsize`, i.e. the absolute end past EOF.
                // Here `t.off == base + o0` and the end is `t.off + t.nbytes == base + o1`.
                if t.off + t.nbytes > fsize {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("k3_st: {}: {} ends past EOF", path.display(), name),
                    ));
                }

                // Duplicate-name check: the C hash insert refuses a second copy of the
                // same name. k3_st.c:422.
                if index.contains_key(name) {
                    let prev = &tensors[*index.get(name).unwrap()];
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "k3_st: duplicate tensor name {} (shard {} and shard {})",
                            name, prev.shard, shard
                        ),
                    ));
                }
                index.insert(name.clone(), tensors.len());
                tensors.push(t);
            }

            // The C code notes (does not fail on) trailing bytes after the last tensor.
            // k3_st.c:337. We skip the note here; it is not load-bearing.

            // Second descriptor on the same file, for streamed reads that must not go
            // through the page cache. Optional: falls back to the buffered `file`.
            // k3_st.c:346.
            let df = open_direct(path)?;
            files.push(file);
            dfiles.push(if df.direct { Some(df.file) } else { None });
        }

        Ok(St {
            files,
            dfiles,
            paths: shard_paths,
            tensors,
            index,
        })
    }

    /// O(1) lookup. Returns `None` when absent. k3_st.c:442.
    pub fn find(&self, name: &str) -> Option<&Tensor> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    /// Raw bytes, exactly as stored. `buf` must hold `t.nbytes`. Returns bytes read.
    /// k3_st.c:497.
    pub fn read(&self, t: &Tensor, buf: &mut [u8]) -> io::Result<i64> {
        let mut got = 0i64;
        while got < t.nbytes {
            let n = pread_full(
                &self.files[t.shard],
                &mut buf[got as usize..(t.nbytes as usize)],
                (t.off + got) as u64,
            )?;
            if n == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("k3_st: short read on {} at +{}", t.name, got),
                ));
            }
            got += n as i64;
        }
        Ok(got)
    }

    /// Read `[off, off+nbytes)` from a shard with O_DIRECT, bypassing the page cache.
    ///
    /// Returns `(payload_bytes, payload_off)`. On the direct path the read is WIDENED to
    /// aligned bounds, so `buf` must hold `nbytes + 2*ST_ALIGN` and be page aligned; on
    /// the buffered fallback the payload lands at offset 0 (`payload_off == 0`).
    /// k3_st.c:457.
    pub fn read_aligned(
        &self,
        shard: usize,
        off: i64,
        nbytes: i64,
        buf: &mut [u8],
    ) -> io::Result<(i64, i64)> {
        if shard >= self.files.len() {
            return Ok((0, 0));
        }

        match &self.dfiles[shard] {
            None => {
                // No O_DIRECT: plain buffered read, payload at offset 0. k3_st.c:463.
                if (buf.len() as i64) < nbytes {
                    return Ok((0, 0));
                }
                let mut got = 0i64;
                while got < nbytes {
                    let n = pread_full(
                        &self.files[shard],
                        &mut buf[got as usize..nbytes as usize],
                        (off + got) as u64,
                    )?;
                    if n == 0 {
                        break;
                    }
                    got += n as i64;
                }
                Ok((got, 0))
            }
            Some(dfd) => {
                // Widen outward to the enclosing aligned window. k3_st.c:476.
                let lo = off & !(ST_ALIGN as i64 - 1);
                let hi = (off + nbytes + ST_ALIGN as i64 - 1) & !(ST_ALIGN as i64 - 1);
                let len = hi - lo;
                let pad = off - lo;
                if len > buf.len() as i64 {
                    return Ok((0, 0));
                }

                let mut got = 0i64;
                while got < len {
                    let n =
                        pread_full(dfd, &mut buf[got as usize..len as usize], (lo + got) as u64)?;
                    if n == 0 {
                        // The final window of a shard can extend past EOF, which is a
                        // short read rather than an error. Accept it once the payload
                        // itself is covered. k3_st.c:487.
                        break;
                    }
                    got += n as i64;
                }
                // k3_st.c:494.
                let avail = if got >= pad + nbytes {
                    nbytes
                } else if got > pad {
                    got - pad
                } else {
                    0
                };
                Ok((avail, pad))
            }
        }
    }

    /// Read and widen to f32. Handles F32 (bit copy), BF16 (16-bit left shift), F16, and
    /// U8 (raw byte value). `out` must hold `numel` floats. k3_st.c:524.
    pub fn read_f32(&self, t: &Tensor, out: &mut [f32]) -> io::Result<i64> {
        let n = t.numel();
        if t.dtype == Dtype::F32 {
            // k3_st.c:527: a single raw read into the output buffer reinterpreted as
            // bytes, then return the element count (= bytes / 4).
            let got = self.read(t, out_as_bytes_mut(out))?;
            return Ok(got / 4);
        }

        let esz = t.dtype.elemsize() as i64;
        if esz <= 0 {
            return Ok(0);
        }

        // Widen in bounded chunks rather than reading the whole tensor first. A whole
        // number of elements per chunk, so no element straddles a boundary. k3_st.c:531.
        let chunk_elems: i64 = WIDEN_CHUNK as i64 / esz;
        let mut raw = vec![0u8; (chunk_elems * esz) as usize];

        let mut done = 0i64;
        while done < n {
            let take = if n - done < chunk_elems {
                n - done
            } else {
                chunk_elems
            };
            let want = take * esz;
            let mut got = 0i64;
            while got < want {
                let r = pread_full(
                    &self.files[t.shard],
                    &mut raw[got as usize..want as usize],
                    (t.off + done * esz + got) as u64,
                )?;
                if r == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        format!("k3_st: short read widening {} at element {}", t.name, done),
                    ));
                }
                got += r as i64;
            }

            let o = &mut out[done as usize..(done + take) as usize];
            match t.dtype {
                Dtype::Bf16 => {
                    // bf16 IS the top 16 bits of an f32: pure left shift, no rounding.
                    // k3_st.c:555, k3_st.h:107.
                    let p = cast_slice::<u16>(&raw[..want as usize]);
                    for i in 0..take as usize {
                        o[i] = crate::ops::dispatch::bf16f(p[i]);
                    }
                }
                Dtype::U8 => {
                    // Raw byte value as a float. k3_st.c:558.
                    for i in 0..take as usize {
                        o[i] = raw[i] as f32;
                    }
                }
                Dtype::F16 => {
                    // Port the C f16 -> f32 bit manipulation VERBATIM. k3_st.c:561.
                    let p = cast_slice::<u16>(&raw[..want as usize]);
                    for i in 0..take as usize {
                        o[i] = f16_to_f32(p[i]);
                    }
                }
                _ => unreachable!("non-widening dtype reached widen loop"),
            }
            done += take;
        }
        Ok(n)
    }

    pub fn nshard(&self) -> usize {
        self.files.len()
    }

    pub fn tensors(&self) -> &[Tensor] {
        &self.tensors
    }

    /// The buffered descriptor for a shard.
    pub fn file(&self, shard: usize) -> &File {
        &self.files[shard]
    }
}

/// Widen chunk size, 4 MiB. k3_st.c:522.
const WIDEN_CHUNK: usize = 4 << 20;

/// Build a `Tensor` from one parsed header entry, validating dtype and shape.
fn build_tensor(
    entry: &serde_json::Map<String, serde_json::Value>,
    name: &str,
    shard: usize,
    base: i64,
    path: &Path,
) -> io::Result<Tensor> {
    // dtype. k3_st.c:275.
    let dv = entry.get("dtype").and_then(|v| v.as_str()).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "k3_st: {}: {} is missing dtype or data_offsets",
                path.display(),
                name
            ),
        )
    })?;
    let dtype = Dtype::from_str_name(dv);
    if dtype == Dtype::Unknown {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "k3_st: {}: unsupported dtype '{}' on {}",
                path.display(),
                dv,
                name
            ),
        ));
    }

    // shape. A scalar is `[]`. k3_st.c:284.
    let shape: Vec<i64> = match entry.get("shape") {
        Some(serde_json::Value::Array(a)) => {
            let mut s = Vec::with_capacity(a.len());
            for d in a {
                let v = d.as_i64().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("k3_st: {}: {} has non-integer shape", path.display(), name),
                    )
                })?;
                s.push(v);
            }
            s
        }
        Some(serde_json::Value::Null) | None => Vec::new(),
        Some(_) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("k3_st: {}: {} shape is not an array", path.display(), name),
            ));
        }
    };

    // data_offsets. k3_st.c:299.
    let offs = entry.get("data_offsets").ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "k3_st: {}: {} is missing dtype or data_offsets",
                path.display(),
                name
            ),
        )
    })?;
    let arr = offs.as_array().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "k3_st: {}: {} data_offsets is not an array",
                path.display(),
                name
            ),
        )
    })?;
    if arr.len() != 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "k3_st: {}: {} data_offsets is not a pair",
                path.display(),
                name
            ),
        ));
    }
    let o0 = arr[0].as_i64().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "k3_st: {}: {} data_offsets has non-integer start",
                path.display(),
                name
            ),
        )
    })?;
    let o1 = arr[1].as_i64().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "k3_st: {}: {} data_offsets has non-integer end",
                path.display(),
                name
            ),
        )
    })?;

    Ok(Tensor {
        name: name.to_string(),
        shard,
        dtype,
        shape,
        off: base + o0,
        nbytes: o1 - o0,
    })
}

/// f16 -> f32, the C bit manipulation transcribed verbatim. k3_st.c:563.
///
/// This deliberately does NOT use a `half` crate or any f16 intrinsic; it reproduces the
/// exact exponent rebias, subnormal renormalisation, and inf/nan handling the C does.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let man = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            // Subnormal: renormalise. k3_st.c:570.
            let m = man;
            let mut sh = 0i32;
            let mut mm = m;
            while mm & 0x400 == 0 {
                mm <<= 1;
                sh += 1;
            }
            let mm = mm & 0x3FF;
            sign | (((127 - 15 - sh + 1) as u32) << 23) | (mm << 13)
        }
    } else if exp == 31 {
        // inf / nan. k3_st.c:576.
        sign | 0x7F800000u32 | (man << 13)
    } else {
        // Normal: rebias the exponent from f16 (bias 15) to f32 (bias 127). k3_st.c:577.
        sign | ((exp - 15 + 127) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// Reinterpret a byte slice as a typed slice. Caller guarantees the length is a whole
/// number of `T` and that the bytes came from a file read (so alignment through a fresh
/// `Vec<u8>` is fine).
fn cast_slice<T>(bytes: &[u8]) -> &[T] {
    assert!(bytes.len() % std::mem::size_of::<T>() == 0);
    let ptr = bytes.as_ptr() as *const T;
    let len = bytes.len() / std::mem::size_of::<T>();
    // SAFETY: `bytes` is a slice of a freshly-read `Vec<u8>`; the element count is whole
    // and the pointee lifetime matches `bytes`.
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

/// View an `&mut [f32]` as `&mut [u8]` for a raw byte copy on the F32 path.
fn out_as_bytes_mut(out: &mut [f32]) -> &mut [u8] {
    let ptr = out.as_mut_ptr() as *mut u8;
    let len = out.len() * 4;
    // SAFETY: `out` is a mutable borrowed slice of f32; reinterpreting as u8 of the same
    // byte width is valid for the borrow's lifetime and never violates aliasing.
    unsafe { std::slice::from_raw_parts_mut(ptr, len) }
}
