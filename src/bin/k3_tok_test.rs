// SPDX-License-Identifier: Apache-2.0
//! `test_tok.c` port: exercise the Rust tokenizer with the same argv contract so
//! `tools/tok_parity.py` runs unchanged against this binary.
//!
//!   k3-tok-test <files_dir> encode     "text"     -> comma-separated ids, one line
//!   k3-tok-test <files_dir> encodefile <path>    -> same, but text read as raw bytes
//!   k3-tok-test <files_dir> decode     1,2,3     -> the text
//!   k3-tok-test <files_dir> roundtrip  <path>    -> encode then decode; PASS/FAIL on
//!                                                   byte-exact recovery of the input
//!
//! `encode` output is byte-identical in format to `tok.py encode` so the two can be
//! diffed directly. USE `encodefile` FOR ANYTHING NON-ASCII: argv is re-encoded by
//! the host shell, so non-ASCII arrives as different bytes than the oracle saw.
//! `encodefile` reads bytes verbatim and is the one that can fail. (test_tok.c:17)

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

const MAXIDS: usize = 1 << 20;
const MAXTEXT: usize = 1 << 22;

fn parse_ids(s: &str, out: &mut Vec<i32>, max: usize) -> usize {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() && out.len() < max {
        // skip separators
        while i < bytes.len() && (bytes[i] == b',' || bytes[i] == b' ') {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        // atoi: optional sign then digits
        let mut neg = false;
        if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
            neg = bytes[i] == b'-';
            i += 1;
        }
        let mut v: i64 = 0;
        let mut any = false;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            v = v * 10 + (bytes[i] - b'0') as i64;
            any = true;
            i += 1;
        }
        if !any {
            // no digit; skip this char to avoid an infinite loop
            if i < bytes.len() {
                i += 1;
            }
            continue;
        }
        if neg {
            v = -v;
        }
        out.push(v as i32);
        // skip to next separator
        while i < bytes.len() && bytes[i] != b',' {
            i += 1;
        }
    }
    out.len()
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: k3-tok-test <files_dir> encode \"text\"\n\
             \x20      k3-tok-test <files_dir> decode 1,2,3\n\
             \x20      k3-tok-test <files_dir> roundtrip <textfile>"
        );
        return ExitCode::from(2);
    }
    let dir = PathBuf::from(&args[1]);
    let mode = &args[2];

    let tok = match k3::tok::Tok::load(&dir) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{}", e);
            return ExitCode::from(1);
        }
    };

    let stdout = std::io::stdout();

    if mode == "encode" && args.len() > 3 {
        let ids = tok.encode(&args[3]);
        let mut h = stdout.lock();
        for (i, id) in ids.iter().enumerate() {
            let _ = write!(h, "{}{}", if i == 0 { "" } else { "," }, id);
        }
        let _ = writeln!(h);
        return ExitCode::from(0);
    }

    if mode == "encodefile" && args.len() > 3 {
        let mut text = Vec::with_capacity(MAXTEXT);
        let mut f = match std::fs::File::open(&args[3]) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("{}: {}", args[3], e);
                return ExitCode::from(1);
            }
        };
        use std::io::Read;
        let mut buf = vec![0u8; 65536];
        loop {
            match f.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    if text.len() + n > MAXTEXT - 1 {
                        let room = MAXTEXT - 1 - text.len();
                        text.extend_from_slice(&buf[..room]);
                        break;
                    }
                    text.extend_from_slice(&buf[..n]);
                }
                Err(e) => {
                    eprintln!("{}: {}", args[3], e);
                    return ExitCode::from(1);
                }
            }
        }
        let s = String::from_utf8_lossy(&text);
        let ids = tok.encode(&s);
        let mut h = stdout.lock();
        for (i, id) in ids.iter().enumerate() {
            let _ = write!(h, "{}{}", if i == 0 { "" } else { "," }, id);
        }
        let _ = writeln!(h);
        return ExitCode::from(0);
    }

    if mode == "decode" && args.len() > 3 {
        let mut ids: Vec<i32> = Vec::with_capacity(MAXIDS);
        parse_ids(&args[3], &mut ids, MAXIDS);
        let out = tok.decode(&ids);
        let mut h = stdout.lock();
        let _ = h.write_all(out.as_bytes());
        let _ = writeln!(h);
        return ExitCode::from(0);
    }

    if mode == "roundtrip" && args.len() > 3 {
        let mut text = Vec::with_capacity(MAXTEXT);
        let mut f = match std::fs::File::open(&args[3]) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("{}: {}", args[3], e);
                return ExitCode::from(1);
            }
        };
        use std::io::Read;
        let mut buf = vec![0u8; 65536];
        loop {
            match f.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    if text.len() + n > MAXTEXT - 1 {
                        let room = MAXTEXT - 1 - text.len();
                        text.extend_from_slice(&buf[..room]);
                        break;
                    }
                    text.extend_from_slice(&buf[..n]);
                }
                Err(e) => {
                    eprintln!("{}: {}", args[3], e);
                    return ExitCode::from(1);
                }
            }
        }
        let n = text.len();
        let s = String::from_utf8_lossy(&text);
        let ni = tok.encode(&s);
        let back = tok.decode(&ni);
        let back_bytes = back.as_bytes();
        let nb = back_bytes.len();

        // Byte-exact recovery is the bar.
        let ok = nb == n && back_bytes[..nb] == text[..n];
        println!(
            "roundtrip: {} bytes -> {} ids -> {} bytes : {}",
            n,
            ni.len(),
            nb,
            if ok { "PASS" } else { "FAIL" }
        );
        if !ok {
            let lim = if nb < n { nb } else { n };
            let mut d = 0;
            while d < lim && back_bytes[d] == text[d] {
                d += 1;
            }
            eprintln!("first divergence at byte {}", d);
            eprintln!(
                "  in : {}",
                String::from_utf8_lossy(&text[d..(d + 40).min(n)])
            );
            eprintln!(
                "  out: {}",
                String::from_utf8_lossy(&back_bytes[d..(d + 40).min(nb)])
            );
            return ExitCode::from(1);
        }
        return ExitCode::from(0);
    }

    eprintln!("unknown mode '{}'", mode);
    ExitCode::from(2)
}
