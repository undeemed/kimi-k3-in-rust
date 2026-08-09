// SPDX-License-Identifier: Apache-2.0
//! Shared harness for the per-op fixture tests. A port of the utility half of
//! `tests/unit/test_ops.c`.
//!
//! What lives here:
//! - loading a fixture JSON with `serde_json` and pulling a named `{ "shape": [...],
//!   "data": [...] }` entry (or a bare array) into a `Vec<f32>`,
//! - reading scalar config fields out of a fixture the way C's `num()` / `boolean()` do,
//! - the tolerance contract: `ATOL` / `RTOL` come from `fixtures/ops/MANIFEST.json` and fall
//!   back to the C defaults `1e-4` / `1e-3` only when the manifest is absent, and
//! - `assert_close`, which reports the worst element with its index and both values, the
//!   way C's `report()` does.
//!
//! A SKIPPED case is a FAILURE, exactly as the C harness counts a skip as a non-zero exit.
//! Tests here do not skip; if a fixture is absent the test panics, which is the Rust
//! equivalent of the C `g_fail++` on a missing file.

use serde_json::Value;
use std::fs;
use std::path::Path;

/// The C fallback tolerances. The manifest is the authority; these are used only if the
/// manifest cannot be read, and they are deliberately looser than the published contract
/// so the fallback can never be mistaken for a pass at the real tolerance. test_ops.c:29.
pub const ATOL_FALLBACK: f64 = 1e-4;
pub const RTOL_FALLBACK: f64 = 1e-3;

/// A loaded fixture directory together with the tolerances in force. The tolerances come
/// from `MANIFEST.json`'s `tolerance.fp32_abs` / `fp32_rel` keys (the manifest's own names)
/// and fall back to `ATOL_FALLBACK` / `RTOL_FALLBACK` when the manifest is missing.
/// test_ops.c:858.
pub struct Fixtures {
    pub atol: f64,
    pub rtol: f64,
    pub dir: std::path::PathBuf,
}

impl Fixtures {
    /// Load every fixture the per-op tests need. The directory is `fixtures/ops` under the
    /// crate root, resolved from `CARGO_MANIFEST_DIR` so the tests run from anywhere.
    pub fn ops() -> Fixtures {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("ops");
        let (atol, rtol) = manifest_tolerances(&dir);
        Fixtures { atol, rtol, dir }
    }

    /// Read and parse a fixture JSON by name (e.g. `"rmsnorm"`). Absent file or malformed
    /// JSON is a hard failure: the C harness counts a missing fixture as `g_fail`, and a
    /// skip is not a pass.
    pub fn load(&self, name: &str) -> Value {
        let path = self.dir.join(format!("{name}.json"));
        let txt = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot open {}: {e}", path.display()));
        serde_json::from_str(&txt)
            .unwrap_or_else(|e| panic!("malformed JSON in {}: {e}", path.display()))
    }

    /// The `mxfp4` fixture lives beside the `ops/` directory, not inside it, because it is
    /// built from released checkpoint bytes rather than from the reference implementation.
    /// test_ops.c:744-751. The Rust copy sits at `fixtures/mxfp4.json`.
    pub fn load_mxfp4(&self) -> Option<Value> {
        let beside = self.dir.join("mxfp4.json");
        let parent = self.dir.join("..").join("mxfp4.json");
        for p in [beside, parent] {
            if let Ok(txt) = fs::read_to_string(&p) {
                return serde_json::from_str(&txt).ok();
            }
        }
        None
    }
}

/// Pull `tolerance.fp32_abs` / `fp32_rel` from `MANIFEST.json`, falling back to the C
/// defaults if the manifest is absent. test_ops.c:866-871.
fn manifest_tolerances(dir: &Path) -> (f64, f64) {
    let path = dir.join("MANIFEST.json");
    match fs::read_to_string(&path) {
        Ok(txt) => {
            let v: Value = serde_json::from_str(&txt).unwrap_or(Value::Null);
            let tolo = v.get("tolerance");
            let fa = tolo
                .and_then(|t| t.get("fp32_abs"))
                .and_then(|x| x.as_f64())
                .unwrap_or(ATOL_FALLBACK);
            let fr = tolo
                .and_then(|t| t.get("fp32_rel"))
                .and_then(|x| x.as_f64())
                .unwrap_or(RTOL_FALLBACK);
            (fa, fr)
        }
        Err(_) => (ATOL_FALLBACK, RTOL_FALLBACK),
    }
}

/// Pull `"key": {"shape": [...], "data": [...]}` into a flat `Vec<f32>`. A bare JSON array
/// is also accepted (the `mxfp4` fixture uses bare arrays for `packed` / `scales`). This
/// mirrors C's `arr()`, which flattens regardless of the declared shape. test_ops.c:66.
pub fn arr(root: &Value, key: &str) -> Vec<f32> {
    let o = root.get(key).unwrap_or_else(|| panic!("missing key {key}"));
    let d = if o.is_object() {
        o.get("data")
            .unwrap_or_else(|| panic!("key {key} has no data array"))
    } else {
        o
    };
    let a = d
        .as_array()
        .unwrap_or_else(|| panic!("key {key} is not an array"));
    a.iter()
        .map(|v| {
            v.as_f64()
                .unwrap_or_else(|| panic!("element of {key} is not a number")) as f32
        })
        .collect()
}

/// `arr` for `u8` payloads (the `mxfp4` packed nibbles and E8M0 scales arrive as floats in
/// the JSON but represent byte values). test_ops.c:771-772.
pub fn arr_u8(root: &Value, key: &str) -> Vec<u8> {
    let o = root.get(key).unwrap_or_else(|| panic!("missing key {key}"));
    let d = if o.is_object() {
        o.get("data").unwrap_or(o)
    } else {
        o
    };
    let a = d
        .as_array()
        .unwrap_or_else(|| panic!("key {key} is not an array"));
    a.iter()
        .map(|v| {
            let f = v
                .as_f64()
                .unwrap_or_else(|| panic!("element of {key} is not a number"))
                as f32;
            f as u8
        })
        .collect()
}

/// A scalar number with a default, like C's `num()`. test_ops.c:79. JSON booleans are NOT
/// numbers; use `boolean` for those. `num_i32` and `num_f32` are the casts the C applies
/// at each call site, so they go through here rather than re-reading the value.
fn num(root: &Value, key: &str, dflt: f64) -> f64 {
    root.get(key).and_then(|v| v.as_f64()).unwrap_or(dflt)
}

/// A scalar number coerced to `i32`, the way C casts `num(...)` to `int` at each call site.
pub fn num_i32(root: &Value, key: &str, dflt: i32) -> i32 {
    num(root, key, dflt as f64) as i32
}

/// A scalar number coerced to `f32`.
pub fn num_f32(root: &Value, key: &str, dflt: f32) -> f32 {
    num(root, key, dflt as f64) as f32
}

/// A boolean, like C's `boolean()`. JSON booleans are `J_BOOL`, not `J_NUM`; reading one
/// with `num()` silently yields the default, which is how `layer_mla` was first mistaken
/// for a KDA layer. test_ops.c:88.
pub fn boolean(root: &Value, key: &str, dflt: i32) -> i32 {
    match root.get(key) {
        Some(Value::Bool(b)) => i32::from(*b),
        Some(Value::Number(n)) => n.as_f64().map(|x| x as i32).unwrap_or(dflt),
        _ => dflt,
    }
}

/// Read `"key": {"shape": [...]}` into the shape vector, or an empty vector if absent.
/// Guessing row counts from element totals is how the first version of the C test went
/// wrong; the fixtures carry the shape, so use it. test_ops.c:100.
pub fn shape(root: &Value, key: &str) -> Vec<i64> {
    match root.get(key) {
        Some(Value::Object(map)) => {
            if let Some(s) = map.get("shape").and_then(|v| v.as_array()) {
                s.iter()
                    .map(|v| {
                        v.as_f64()
                            .unwrap_or_else(|| panic!("shape entry of {key} is not a number"))
                            as i64
                    })
                    .collect()
            } else {
                Vec::new()
            }
        }
        Some(Value::Array(a)) => vec![a.len() as i64],
        _ => Vec::new(),
    }
}

/// Compare two float arrays and panic with the worst element if any element exceeds
/// `atol + rtol * |want|`. This is C's `report()` inlined into a pass/fail assertion: the
/// worst element's index and both its values appear in the message. test_ops.c:111.
pub fn assert_close(name: &str, got: &[f32], want: &[f32], atol: f64, rtol: f64) {
    assert_eq!(
        got.len(),
        want.len(),
        "{name}: length mismatch got {} want {}",
        got.len(),
        want.len()
    );
    let mut worst = 0.0f64;
    let mut at = 0usize;
    for i in 0..got.len() {
        let a = ((got[i] as f64) - (want[i] as f64)).abs();
        let tol = atol + rtol * (want[i] as f64).abs();
        let r = if tol > 0.0 { a / tol } else { f64::INFINITY };
        if r > worst {
            worst = r;
            at = i;
        }
    }
    assert!(
        worst <= 1.0,
        "{name}: worst {:.2}x tol at i={} got {:.7} want {:.7}",
        worst,
        at,
        got[at],
        want[at]
    );
}

/// Like `assert_close` but for the KDA decay test, which uses an absolute `1e-5` bound
/// rather than the manifest tolerances (the C test's own `wg <= 1e-5 && wa <= 1e-5`).
/// test_ops.c:264.
pub fn assert_abs_max(name: &str, got: &[f32], want: &[f32], bound: f64) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    let mut worst = 0.0f64;
    let mut at = 0usize;
    for i in 0..got.len() {
        let d = ((got[i] as f64) - (want[i] as f64)).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    assert!(
        worst <= bound,
        "{name}: max|delta|={:.3e} at i={} got {:.7} want {:.7}",
        worst,
        at,
        got[at],
        want[at]
    );
}
