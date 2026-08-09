// SPDX-License-Identifier: Apache-2.0
//! Platform I/O shims and aligned allocation.
//!
//! Port of `src/io/k3_portable_io.h`. The C engine asks for two Linux-only things:
//!
//! - `O_DIRECT`, to bypass the page cache on the trunk and expert reads. Darwin's
//!   equivalent is not an open flag but `fcntl(F_NOCACHE)` after the fact, so the open
//!   path differs per OS. Any failure, or any unsupported target, falls back to a plain
//!   buffered open with `direct = false`, which is correct (just slower) - exactly what
//!   the C shims do.
//! - `posix_fadvise(WILLNEED)`, a page-cache prefetch hint with no Darwin equivalent.
//!   Callers already treat it as advisory, so the non-Linux body is a no-op.

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::fs::File;
use std::io;
use std::ops::{Deref, DerefMut};
use std::path::Path;

/// Default hugepage alignment: 2 MiB. On Linux the allocation additionally gets
/// `madvise(MADV_HUGEPAGE)`.
const HUGE_ALIGN: usize = 2 * 1024 * 1024;

/// O_DIRECT alignment: offset, length and buffer must all be multiples of this.
/// `K3_ST_ALIGN` from `k3_st.h:49`.
pub const ST_ALIGN: usize = 4096;

/// A page- or hugepage-aligned heap buffer. All the unsafe allocation in the crate lives
/// here. `len` is rounded UP to the alignment. Alignment is 2 MiB unless `K3_NOHUGE` is
/// set in the environment, in which case 4 KiB; on Linux the buffer gets
/// `madvise(MADV_HUGEPAGE)`.
///
/// The allocation is zeroed so a short read never exposes uninitialised bytes to a caller
/// that treats the tail as defined.
pub struct AlignedBuf {
    ptr: *mut u8,
    layout: Layout,
    len: usize,
}

impl AlignedBuf {
    /// Allocate with the default alignment (2 MiB, or 4 KiB when `K3_NOHUGE` is set).
    pub fn new(len: usize) -> io::Result<AlignedBuf> {
        let align = if std::env::var_os("K3_NOHUGE").is_some() {
            ST_ALIGN
        } else {
            HUGE_ALIGN
        };
        Self::with_align(len, align)
    }

    /// Allocate with an explicit alignment. `len` is rounded UP to `align`.
    pub fn with_align(len: usize, align: usize) -> io::Result<AlignedBuf> {
        // align must be a power of two and a valid allocation alignment.
        debug_assert!(align.is_power_of_two() && align != 0);
        let len = (len + align - 1) & !(align - 1);
        let len = if len == 0 { align } else { len };
        // SAFETY: align is a non-zero power of two; size is non-zero and aligned up.
        let layout = Layout::from_size_align(len, align)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "bad layout"))?;
        // SAFETY: layout is valid (constructed and checked above). alloc_zeroed returns
        // null only on allocation failure, which we turn into an io error.
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                "alloc_zeroed failed",
            ));
        }
        // On Linux, advise the kernel to back this with huge pages. Advisory: failure is
        // not fatal (the buffer still works, just with 4 KiB pages), so the result is
        // ignored exactly like the C path.
        #[cfg(target_os = "linux")]
        unsafe {
            let _ = libc::madvise(ptr as *mut libc::c_void, len, libc::MADV_HUGEPAGE);
        }
        Ok(AlignedBuf { ptr, layout, len })
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    /// The rounded length this buffer exposes through `Deref`.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when this buffer exposes no bytes.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Deref for AlignedBuf {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        // SAFETY: ptr owns `len` initialised bytes for the lifetime of `&self`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl DerefMut for AlignedBuf {
    fn deref_mut(&mut self) -> &mut [u8] {
        // SAFETY: same as Deref; exclusive borrow proven by `&mut self`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: this layout is the one the buffer was allocated with.
        unsafe { dealloc(self.ptr, self.layout) }
    }
}

// SAFETY: AlignedBuf owns its allocation exclusively; the raw pointer is never shared
// across threads and the buffer has no interior mutability beyond `&mut self`.
unsafe impl Send for AlignedBuf {}
// SAFETY: same - no shared mutable state; `&AlignedBuf` only yields `&[u8]`.
unsafe impl Sync for AlignedBuf {}

/// A shard file plus whether the un-cached path is actually available on it.
pub struct Dfile {
    pub file: File,
    pub direct: bool,
}

/// Linux: `O_DIRECT` via `OpenOptions::custom_flags`. macOS: open then
/// `fcntl(F_NOCACHE)`. Anything else, or any failure: a plain buffered open with
/// `direct = false`. This is the port of `src/io/k3_portable_io.h`.
pub fn open_direct(path: &Path) -> io::Result<Dfile> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        match std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(path)
        {
            Ok(f) => Ok(Dfile {
                file: f,
                direct: true,
            }),
            Err(_) => {
                // O_DIRECT can be refused (filesystem does not support it, tmpfs, etc).
                // Fall back to a buffered open, exactly like the C path's `open` retry.
                let file = File::open(path)?;
                Ok(Dfile {
                    file,
                    direct: false,
                })
            }
        }
    }
    #[cfg(target_os = "macos")]
    {
        let file = File::open(path)?;
        use std::os::unix::io::AsRawFd;
        let fd = file.as_raw_fd();
        // F_NOCACHE is the Darwin O_DIRECT equivalent. Failure is not fatal: the caller
        // keeps the descriptor and reads through the page cache instead. `direct` is true
        // only when this returns 0.
        let r = unsafe { libc::fcntl(fd, libc::F_NOCACHE, 1) };
        Ok(Dfile {
            file,
            direct: r == 0,
        })
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let file = File::open(path)?;
        Ok(Dfile {
            file,
            direct: false,
        })
    }
}

/// Positioned read that loops until `buf` is full or EOF. Returns bytes read.
pub fn pread_full(f: &File, buf: &mut [u8], off: u64) -> io::Result<usize> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        let mut got = 0usize;
        let mut o = off;
        while got < buf.len() {
            let n = f.read_at(&mut buf[got..], o)?;
            if n == 0 {
                break;
            }
            got += n;
            o += n as u64;
        }
        Ok(got)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut got = 0usize;
        let mut o = off;
        while got < buf.len() {
            let n = f.seek_read(&mut buf[got..], o)?;
            if n == 0 {
                break;
            }
            got += n;
            o += n as u64;
        }
        Ok(got)
    }
}

/// `posix_fadvise(POSIX_FADV_WILLNEED)` on Linux, a no-op elsewhere.
pub fn fadvise_willneed(f: &File, off: u64, len: u64) {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;
        let fd = f.as_raw_fd();
        // Advisory: ignore the result, matching the C shim which treats it as a hint.
        unsafe {
            libc::posix_fadvise(
                fd,
                off as libc::off_t,
                len as libc::off_t,
                libc::POSIX_FADV_WILLNEED,
            );
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (f, off, len);
    }
}
