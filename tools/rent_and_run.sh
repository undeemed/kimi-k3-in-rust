#!/usr/bin/env bash
# rent_and_run.sh - stand up the full Kimi K3 comparison on a fresh rented box.
#
# Clones both engines, pulls the whole 1.56 TB checkpoint, builds both, packs the trunk,
# then runs C and Rust at the same 8 GB memory ceiling and byte-compares their output.
#
# TARGET: im4gn.xlarge (4 vCPU, 16 GiB, 1,875 GB NVMe, arm64) or anything with >= 1.75 TB
# of local disk. arm64 is deliberate: the port's measured advantage is a NEON result, and
# on x86-64 both projects hand-write their intrinsics to the same instruction mix.
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

DEV="${1:-/dev/nvme1n1}"
MNT=/data
REPO=https://huggingface.co/moonshotai/Kimi-K3/resolve/main
C_COMMIT=ff11dce858a2eb8a781224facdffd33a1fa48d25
JOBS="${JOBS:-8}"            # parallel shard downloads
NEED_GB=1750

say() { printf '\n=== %s ===\n' "$*"; }

# ---------------------------------------------------------------- disk ----
say "formatting $DEV without reserved blocks"
if ! mountpoint -q "$MNT"; then
    mkfs.ext4 -F -m 0 -E lazy_itable_init=0,lazy_journal_init=0 "$DEV"
    mkdir -p "$MNT"
    mount -o noatime "$DEV" "$MNT"
fi
avail_gb=$(df -BG --output=avail "$MNT" | tail -1 | tr -dc '0-9')
echo "usable: ${avail_gb} GB (need ${NEED_GB} GB)"
[ "$avail_gb" -ge "$NEED_GB" ] || { echo "NOT ENOUGH DISK, stopping before wasting hours" >&2; exit 1; }
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
make -C kimi-k3-in-c bin/k3 -j"$(nproc)" >/dev/null
( cd kimi-k3-in-rust && cargo build --release --quiet )
ls -l kimi-k3-in-c/bin/k3 kimi-k3-in-rust/target/release/k3 | awk '{print $NF, $5, "bytes"}'

# ---------------------------------------------------------------- trunk ----
# Required at an 8 GB ceiling: resident-trunk mode wants 113 GB of RAM for 93 layers.
say "packing the trunk, ~109 GB, stdlib only"
TRUNK="$MNT/k3trunk"
mkdir -p "$TRUNK"
[ -s "$TRUNK/trunk.bin" ] || python3 kimi-k3-in-c/tools/pack_trunk.py "$MODEL" "$TRUNK"
df -BG --output=avail "$MNT" | tail -1 | xargs echo "disk left after packing:"

# ---------------------------------------------------------------- run ----
# His 8 GB floor configuration, and the ceiling is enforced rather than hoped for.
# MemorySwapMax=0 matters as much as MemoryMax: without it an over-budget run swaps
# instead of dying, and its s/token measures swap bandwidth.
say "running both at an 8 GB ceiling"
IDS=1,2,3,4,5,6,7,8
ARGS=(--ids "$IDS" --gen 8 --trunk "$TRUNK" --trunk-gb 2.5 --cache-gb 0.5 --incremental)

run_capped() {  # name binary outfile logitsfile threads_env
    systemd-run --scope -q -p MemoryMax=8G -p MemorySwapMax=0 \
        env "$5=$(nproc)" "$2" "$MODEL" "${ARGS[@]}" \
        --dump-logits "$4" --out "$3" 2>&1 | tail -30
}
run_capped C    kimi-k3-in-c/bin/k3                 /tmp/c.json    /tmp/c.bin    OMP_NUM_THREADS
run_capped Rust kimi-k3-in-rust/target/release/k3   /tmp/rust.json /tmp/rust.bin RAYON_NUM_THREADS

# ---------------------------------------------------------------- verdict ----
say "verdict"
python3 - <<'PY'
import json
c = json.load(open('/tmp/c.json')); r = json.load(open('/tmp/rust.json'))
print('C    ids:', c['generated_ids'])
print('Rust ids:', r['generated_ids'])
print('tokens:', 'IDENTICAL' if c['generated_ids'] == r['generated_ids'] else 'MISMATCH')
print(f"C    {c['seconds_per_token']:.2f} s/token")
print(f"Rust {r['seconds_per_token']:.2f} s/token")
if r['seconds_per_token']:
    print(f"speedup {c['seconds_per_token'] / r['seconds_per_token']:.2f}x")
PY
cmp /tmp/c.bin /tmp/rust.bin && echo "logits BYTE-IDENTICAL (163,840 f32)" || echo "logits DIFFER"
