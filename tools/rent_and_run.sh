#!/usr/bin/env bash
# rent_and_run.sh - stand up the full Kimi K3 comparison on a fresh rented box.
#
# Clones both engines, pulls the whole 1.56 TB checkpoint, builds both, packs the trunk,
# then runs C and Rust at the same 8 GB memory ceiling and byte-compares their output.
#
# RUN THIS ON TWO BOXES, ONE PER ARCHITECTURE, BOTH BINARIES ON EACH:
#   im4gn.xlarge   arm64   4 vCPU, 16 GiB, 1,875 GB NVMe, ~$0.36/hr
#                          tests whether the measured 1.6x holds on the full 93 layers
#   i3en.xlarge    x86-64  4 vCPU, 32 GiB, 2,500 GB NVMe, ~$0.45/hr
#                          tests the prediction that x86-64 comes out at parity
#
# DO NOT split the two engines across two boxes. The reference's own data: the same code
# on two devices differed 2.2x (31.71 against 70.62 s/token) with nothing in the source to
# explain it, and one identical configuration measured twice pulled 2,709 against 5,874
# MB/s off the disk on byte-for-byte identical work. That is a 2.17x device spread against
# a 1.6x effect, so a cross-machine comparison measures the storage. Both engines run here,
# on one device, back to back.
#
# DISK IS THE BINDING CONSTRAINT AND THE MARGIN IS THIN:
#   checkpoint          1,561 GB
#   packed trunk          109 GB   (not optional at an 8 GB ceiling)
#   total               1,670 GB
# A 1,875 GB device is decimal, and ext4 reserves 5% of blocks for root by default, which
# would take another ~94 GB and can strand the trunk pack at hour five. So this formats
# with `-m 0`, and the downloader writes each shard straight to its final path rather than
# assembling parts, which would transiently double the largest files.
#
# No pread cap patch is needed here. That bug is macOS-only: Linux caps a large read at
# 0x7ffff000 and returns short, which the C reader's loop already absorbs.
#
# Usage:  sudo bash rent_and_run.sh [device]      # device defaults to /dev/nvme1n1
set -euo pipefail

# cloud-init runs user-data with no HOME, and `set -u` turns the rustup PATH line into a
# fatal "HOME: unbound variable" after the toolchain has already installed. Verified on a
# live box: this was the only unset variable the script reads.
export HOME="${HOME:-/root}"

DEV="${1:-/dev/nvme1n1}"
MNT=/data
REPO=https://huggingface.co/moonshotai/Kimi-K3/resolve/main
C_COMMIT=ff11dce858a2eb8a781224facdffd33a1fa48d25
JOBS="${JOBS:-8}"            # parallel shard downloads
# Compared in BYTES. `df -BG` prints GiB despite the G, so comparing it against a figure
# computed in decimal GB silently demands ~7% more space than intended: a 1,875 GB device
# reports 1,711 and a 1750 threshold rejects it, which killed the first launch on a disk
# with 168 GB to spare.
NEED_BYTES=$((1670 * 1000 * 1000 * 1000))   # 1,561 GB checkpoint + 109 GB packed trunk

say() { printf '\n=== %s ===\n' "$*"; }
ncpu() { nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1; }

# ------------------------------------------------------------ what this box tests ----
ARCH=$(uname -m)
case "$ARCH" in
    aarch64|arm64)
        EXPECT="Rust ahead. The gap lives in matmul_bf16, which the reference left to the
  autovectoriser and the port hand-wrote in NEON. Two layers measured 1.54-1.62x."
        ;;
    x86_64)
        EXPECT="parity. Both projects hand-write AVX2 here and the instruction mix is
  identical, 4, 4 and 2 vfmadd*pd per kernel. A large gap either way is a finding."
        ;;
    *)  EXPECT="unknown for $ARCH: neither project hand-writes intrinsics for it, so both
  fall back to their compiler's autovectoriser." ;;
esac

say "this box is $ARCH, $(ncpu) vCPU"
echo "Both engines will run here, on this one device, back to back."
echo "Expected: $EXPECT"
echo
echo "Not reproduced here: the reference's 32.69 s/token floor came from 124 cores, and"
echo "roughly 43% of a token at that configuration is arithmetic. Fewer cores will be"
echo "slower by an amount this script cannot predict. What transfers is the 8 GB ceiling,"
echo "the byte-identical output, and the C-against-Rust ratio on ONE device."

# ---------------------------------------------------------------- disk ----
say "formatting $DEV without reserved blocks"
if ! mountpoint -q "$MNT"; then
    mkfs.ext4 -F -m 0 -E lazy_itable_init=0,lazy_journal_init=0 "$DEV"
    mkdir -p "$MNT"
    mount -o noatime "$DEV" "$MNT"
fi
avail_bytes=$(df -B1 --output=avail "$MNT" | tail -1 | tr -dc '0-9')
printf 'usable: %s GB, need %s GB\n' "$((avail_bytes/1000000000))" "$((NEED_BYTES/1000000000))"
[ "$avail_bytes" -ge "$NEED_BYTES" ] || { echo "NOT ENOUGH DISK, stopping before wasting hours" >&2; exit 1; }
chmod 777 "$MNT"

# ---------------------------------------------------------------- deps ----
say "installing toolchain"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq build-essential git curl python3 pkg-config >/dev/null
command -v cargo >/dev/null || {
    curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null
}
export PATH="$HOME/.cargo/bin:$PATH"
cc --version | head -1; cargo --version

# ---------------------------------------------------------------- repos ----
say "cloning both engines"
cd "$MNT"
[ -d kimi-k3-in-c ]    || git clone -q https://github.com/FareedKhan-dev/kimi-k3-in-c.git
[ -d kimi-k3-in-rust ] || git clone -q https://github.com/undeemed/kimi-k3-in-rust.git
# On a re-run the clone already exists, so PULL it: two reruns once executed a stale
# guard because "skips what is done" silently included this checkout. The C clone is
# not pulled - it is the frozen reference, pinned below.
git -C kimi-k3-in-rust pull -q --ff-only 2>/dev/null || true
git -C kimi-k3-in-c checkout -q "$C_COMMIT"
echo "C at $(git -C kimi-k3-in-c rev-parse --short HEAD), Rust at $(git -C kimi-k3-in-rust rev-parse --short HEAD)"

# ---------------------------------------------------------------- weights ----
# One curl per shard, straight to the final path, JOBS files at a time. Chunking inside a
# file would need a concatenate step, and 8 of those in flight is 136 GB of transient
# copies against ~170 GB of headroom.
say "downloading the checkpoint, 1.56 TB, resumable"
MODEL="$MNT/k3model"
mkdir -p "$MODEL"
for f in config.json tiktoken.model tokenizer_config.json; do
    [ -s "$MODEL/$f" ] || curl -sSL -o "$MODEL/$f" "$REPO/$f"
done

fetch() {
    f="model-$(printf '%05d' "$1")-of-000096.safetensors"
    want=$(curl -sSLI "$REPO/$f" | tr -d '\r' | awk 'tolower($1)=="content-length:"{n=$2} END{print n}')
    have=$(stat -c%s "$MODEL/$f" 2>/dev/null || echo 0)
    [ "$have" = "$want" ] && { echo "  have $f"; return 0; }
    curl -sSL --retry 100 --retry-delay 2 --retry-all-errors \
         --speed-limit 65536 --speed-time 30 -C - -o "$MODEL/$f" "$REPO/$f"
    got=$(stat -c%s "$MODEL/$f")
    [ "$got" = "$want" ] || { echo "  SHORT $f: $got of $want" >&2; return 1; }
    echo "  ok   $f  $((got/1000000)) MB"
}
export -f fetch; export REPO MODEL
seq 1 96 | xargs -P "$JOBS" -I{} bash -c 'fetch {}'
echo "checkpoint: $(du -sh "$MODEL" | cut -f1)"

# ---------------------------------------------------------------- build ----
say "building both engines"
make -C kimi-k3-in-c bin/k3 -j"$(ncpu)" >/dev/null
( cd kimi-k3-in-rust && cargo build --release --quiet )
ls -l kimi-k3-in-c/bin/k3 kimi-k3-in-rust/target/release/k3 | awk '{print $NF, $5, "bytes"}'

# ---------------------------------------------------------------- trunk ----
# Required at an 8 GB ceiling: resident-trunk mode wants 113 GB of RAM for 93 layers.
say "packing the trunk, ~109 GB, stdlib only"
TRUNK="$MNT/k3trunk"
mkdir -p "$TRUNK"
[ -s "$TRUNK/trunk.bin" ] || python3 kimi-k3-in-c/tools/pack_trunk.py "$MODEL" "$TRUNK"
printf 'disk left after packing: %s GB\n' "$(($(df -B1 --output=avail "$MNT" | tail -1 | tr -dc '0-9')/1000000000))"

# ---------------------------------------------------------------- run ----
# His 8 GB floor configuration, and the ceiling is enforced rather than hoped for.
# MemorySwapMax=0 matters as much as MemoryMax: without it an over-budget run swaps
# instead of dying, and its s/token measures swap bandwidth.
say "running both at an 8 GB ceiling"
IDS=1,2,3,4,5,6,7,8
ARGS=(--ids "$IDS" --gen 8 --trunk "$TRUNK" --trunk-gb 2.5 --cache-gb 0.5 --incremental)

run_capped() {  # name binary outfile logitsfile threads_env
    systemd-run --scope -q -p MemoryMax=8G -p MemorySwapMax=0 \
        env "$5=$(ncpu)" "$2" "$MODEL" "${ARGS[@]}" \
        --dump-logits "$4" --out "$3" 2>&1 | tail -30
}
run_capped C    kimi-k3-in-c/bin/k3                 /tmp/c.json    /tmp/c.bin    OMP_NUM_THREADS
run_capped Rust kimi-k3-in-rust/target/release/k3   /tmp/rust.json /tmp/rust.bin RAYON_NUM_THREADS

# ---------------------------------------------------------------- verdict ----
# One CSV line per box, so the two runs merge without retyping anything.
say "verdict on $ARCH"
if cmp -s /tmp/c.bin /tmp/rust.bin; then LOGITS=identical; else LOGITS=DIFFER; fi
INSTANCE=$(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/instance-type || echo "$ARCH-box")
ARCH="$ARCH" INSTANCE="$INSTANCE" LOGITS="$LOGITS" EXPECT="$EXPECT" NPROC="$(ncpu)" python3 - <<'PY'
import json, os
c = json.load(open('/tmp/c.json')); r = json.load(open('/tmp/rust.json'))
cs, rs = c['seconds_per_token'], r['seconds_per_token']
same = c['generated_ids'] == r['generated_ids']

print(f"  arch          {os.environ['ARCH']} on {os.environ['INSTANCE']}, {os.environ['NPROC']} vCPU")
nl = c['layers']
print(f"  layers        {nl}" + (" (full model)" if nl == 93 else f" of 93  <- PARTIAL"))
print(f"  tokens        {'IDENTICAL' if same else 'MISMATCH  <- STOP, the engines diverged'}")
print(f"  logits        {os.environ['LOGITS']}")
print(f"  C             {cs:8.2f} s/token")
print(f"  Rust          {rs:8.2f} s/token")
if rs:
    print(f"  speedup       {cs / rs:8.2f}x")
print(f"\n  expected: {os.environ['EXPECT']}")

# The line to paste back. Correctness first: a speed number from diverged engines is junk.
line = (f"{os.environ['INSTANCE']},{os.environ['ARCH']},{os.environ['NPROC']},"
        f"{c['layers']},{cs:.4f},{rs:.4f},{cs / rs if rs else 0:.4f},"
        f"{'yes' if same else 'NO'},{os.environ['LOGITS']}")
hdr = "instance,arch,vcpu,layers,c_s_per_token,rust_s_per_token,speedup,tokens_match,logits"
print(f"\n  paste this back:\n  {hdr}\n  {line}")
open('/tmp/result.csv', 'w').write(hdr + "\n" + line + "\n")
PY
echo
echo "also saved to /tmp/result.csv"
