// Measure what indexing the real checkpoint costs in resident memory.
//
// The full-model run was OOM-killed at an 8 GB cgroup cap with anon-rss 8.37 GB, while the
// C build survived the same config at 8.28 GB peak. `indexed 497220 tensors from 96 shards`
// is the obvious suspect: C keeps names in an arena behind an FNV table, while this port
// holds a `String` per `Tensor` plus a `HashMap<String, usize>` that stores every name a
// SECOND time, and gives each name and each shape its own heap allocation.
//
// Run against a header-only replica so this needs 76 MB rather than 1.56 TB:
//   /tmp/k3venv/bin/python /tmp/make_header_replica.py
//   cargo run --release --bin measure_index -- /tmp/k3-headers

use std::path::Path;

fn rss_bytes() -> u64 {
    // ru_maxrss is BYTES on Darwin and kibibytes on Linux.
    let mut u: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut u) };
    let v = u.ru_maxrss as u64;
    if cfg!(target_os = "macos") {
        v
    } else {
        v * 1024
    }
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/k3-headers".into());
    let before = rss_bytes();
    let st = k3::st::St::open(Path::new(&dir)).expect("open header replica");
    let n = st.tensors().len();
    let after = rss_bytes();

    let used = after.saturating_sub(before);
    println!("shards            {}", st.nshard());
    println!("tensors           {n}");
    println!("RSS before        {:.1} MB", before as f64 / 1e6);
    println!("RSS after         {:.1} MB", after as f64 / 1e6);
    println!("index cost        {:.1} MB", used as f64 / 1e6);
    println!("per tensor        {:.0} bytes", used as f64 / n as f64);

    // Keep it alive so nothing is dropped before the measurement is printed.
    std::hint::black_box(&st);
}
