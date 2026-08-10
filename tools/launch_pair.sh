#!/usr/bin/env bash
# launch_pair.sh - launch one arm64 and one x86-64 box, each running the full comparison.
#
# Both engines run on each box, back to back on one device. The two boxes answer two
# different questions rather than splitting one measurement:
#   arm64   does the measured 1.6x hold on the full 93 layers
#   x86-64  does it come out at parity, as the identical instruction mix predicts
#
# Splitting C onto one box and Rust onto the other would measure the storage: the
# reference records the same code differing 2.2x across two devices.
#
# Each box runs unattended from user-data, so nothing needs an SSH session held open.
# Results land in /tmp/result.csv on the box and in the console log.
#
# Usage:
#   aws sso login --profile myorg
#   PROFILE=myorg bash tools/launch_pair.sh check     # quotas and AMIs only, free
#   PROFILE=myorg bash tools/launch_pair.sh launch    # spends money
#   PROFILE=myorg bash tools/launch_pair.sh status
#   PROFILE=myorg bash tools/launch_pair.sh kill      # terminate both
set -euo pipefail

PROFILE="${PROFILE:?set PROFILE to an aws profile, e.g. PROFILE=myorg}"
# Instance profile granting AmazonSSMManagedInstanceCore, so a box that fails is
# reachable with `aws ssm start-session` instead of being a black box. Create once with
# tools/make_ssm_profile.sh. Set SSM_PROFILE= to launch without it.
SSM_PROFILE="${SSM_PROFILE-kimi-bench-ssm}"
REGION="${REGION:-us-east-1}"
TAG=kimi-k3-bench
RUST_REPO="${RUST_REPO:-https://github.com/undeemed/kimi-k3-in-rust.git}"
A=(aws --profile "$PROFILE" --region "$REGION")

# im4gn is Graviton2, so the arm64 box is where the NEON kernels actually get tested.
# Plain case statements rather than associative arrays: macOS still ships bash 3.2, which
# has no `declare -A`, and this half runs on the laptop.
# Core count is a measurement decision, not just a cost one. The reference measured on 124
# cores where ~80% of a token is disk wait. On 4 cores the compute half inflates while the
# disk half does not, so the run turns compute-bound and reports something near the raw
# kernel ratio rather than what a deployment sees. Since the Rust advantage lives entirely
# in compute, a small box overstates it. Default to the largest that fits a 16 vCPU quota.
ARM_TYPE="${ARM_TYPE:-im4gn.4xlarge}"     # 16 vCPU, 64 GiB, 7.5 TB
X86_TYPE="${X86_TYPE:-i3en.3xlarge}"      # 12 vCPU, 96 GiB, 7.5 TB
inst_for()     { case "$1" in arm64) echo "$ARM_TYPE" ;; x86_64) echo "$X86_TYPE" ;; esac; }
ami_suffix()   { case "$1" in arm64) echo arm64 ;; x86_64) echo amd64 ;; esac; }

say() { printf '\n=== %s ===\n' "$*"; }

ami_for() {  # canonical's published Ubuntu 24.04 AMI for this arch
    "${A[@]}" ssm get-parameters \
        --names "/aws/service/canonical/ubuntu/server/24.04/stable/current/$(ami_suffix "$1")/hvm/ebs-gp3/ami-id" \
        --query 'Parameters[0].Value' --output text
}

# Storage-optimized families very often sit at a zero On-Demand quota on a fresh account,
# which fails at RunInstances after everything else looks fine. Check before spending.
quota_for() {
    case "$1" in
        arm64)  code=L-1216C47A ;;   # Running On-Demand Standard (A, C, D, H, I, M, R, T, Z)
        x86_64) code=L-1216C47A ;;
    esac
    "${A[@]}" service-quotas get-service-quota --service-code ec2 --quota-code "$code" \
        --query 'Quota.Value' --output text 2>/dev/null || echo "unknown"
}

userdata() {  # runs as root on first boot, no SSH needed
    cat <<EOF
#!/usr/bin/env bash
exec > >(tee -a /var/log/kimi-bench.log | tee /dev/console) 2>&1
set -x

# Guaranteed end, whatever happens. Without this a box that fails early, or wedges on a
# stalled download, runs until someone remembers it.
#
# 12 hours, not 8, because the two costs are lopsided. Measured on the live box: 242 MB/s
# download, and modelling the runs off the reference's own 57.3% I/O split puts the whole
# job at 2.3-2.8 h. But a backstop that fires mid-run stops the instance, and stopping
# discards the instance store, so the 1.56 TB has to be pulled again: ~2 h lost against
# ~\$6 of idle billing for the extra headroom. The run stops itself when it finishes, so
# this only ever fires on a hang.
shutdown -h +720 "kimi-bench backstop" &

# Shut down on the way out, success or failure. Instance-initiated shutdown is set to STOP,
# not terminate, which is what makes this safe to do on failure too: a stopped instance
# keeps its root volume and its console log, so /tmp/result.csv and the full log are still
# readable afterwards, and it bills only ~40 GB of gp3. Terminating would have destroyed
# the very output this run exists to produce.
on_exit() {
  rc=\$?
  echo "EXIT status \$rc"
  if [ "\$rc" -ne 0 ]; then
    # Do NOT stop on failure. Stopping discards the instance store, and that is where the
    # 1.56 TB checkpoint lives, so a stop turns any late failure into another ~2.5 h
    # download. Leave the box up and let the 12 h backstop bound it: that is hours to fix
    # the cause and re-run against weights that are already on disk.
    echo "FAILED, rc=\$rc. LEAVING THIS BOX RUNNING so the checkpoint survives."
    echo "  aws ssm start-session --target \$(cloud-init query instance_id 2>/dev/null || echo INSTANCE)"
    echo "  cat /var/log/kimi-bench.log        # what went wrong"
    echo "  ls -la /data/k3model /data/k3trunk # 1.56 TB + 109 GB, still here"
    echo "  bash /root/rust-port/tools/rent_and_run.sh \$DEV   # re-runs, skips what is done"
    echo "The 12 h backstop will stop it if nobody intervenes."
    exit "\$rc"
  fi
  shutdown -h now
}
trap on_exit EXIT

apt-get update -qq && apt-get install -y -qq git curl
cd /root
git clone -q $RUST_REPO rust-port

# Pick the instance store by SIZE, never by enumeration order. Nitro does not guarantee
# that the local NVMe is nvme1n1, so \`tail -1\` can hand back the 40 GB EBS root, and
# then this either refuses to format and dies at first boot or, worse, formats the root.
# The instance store is 1.87 TB or 2.5 TB against a 40 GB root, so largest-wins is
# unambiguous.
DEV=\$(lsblk -dbn -o NAME,SIZE,TYPE | awk '\$3=="disk"{print \$2, \$1}' | sort -rn | head -1 | awk '{print "/dev/"\$2}')
ROOT=\$(lsblk -no PKNAME "\$(findmnt -no SOURCE /)")
echo "chose \$DEV, root is on /dev/\$ROOT"
[ "\$DEV" = "/dev/\$ROOT" ] && { echo "REFUSING: largest disk is the root device"; exit 1; }

bash rust-port/tools/rent_and_run.sh "\$DEV"
echo "BENCH COMPLETE"
cat /tmp/result.csv
EOF
}

case "${1:-check}" in
check)
    say "AMIs"
    for arch in arm64 x86_64; do printf '  %-8s %-16s %s\n' "$arch" "$(inst_for "$arch")" "$(ami_for "$arch")"; done
    say "On-Demand vCPU quota (need 4 per box, 8 total)"
    printf '  standard families: %s\n' "$(quota_for arm64)"
    echo
    echo "If that is 0 or below 8, request an increase before launching:"
    echo "  aws --profile $PROFILE service-quotas request-service-quota-increase \\"
    echo "      --service-code ec2 --quota-code L-1216C47A --desired-value 16"
    ;;
launch)
    # A fresh account often sits at 5 On-Demand vCPUs, and each box wants 4, so both at
    # once fails the second RunInstances. `launch arm64` runs one at a time.
    WHICH="${2:-arm64 x86_64}"
    say "launching: $WHICH (tagged $TAG)"
    for arch in $WHICH; do
        id=$("${A[@]}" ec2 run-instances \
            --image-id "$(ami_for "$arch")" \
            --instance-type "$(inst_for "$arch")" \
            --instance-initiated-shutdown-behavior stop \
            ${SSM_PROFILE:+--iam-instance-profile Name=$SSM_PROFILE} \
            --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=40,VolumeType=gp3,DeleteOnTermination=true}' \
            --user-data "$(userdata)" \
            --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG-$arch},{Key=Project,Value=$TAG}]" \
            --query 'Instances[0].InstanceId' --output text)
        echo "  $arch  $(inst_for "$arch")  $id"
    done
    echo
    echo "Each box downloads 1.56 TB then runs. Budget 4-6 hours."
    echo "Each STOPS itself when done, or after 12 hours whatever happens. On FAILURE it"
    echo "stays UP so the downloaded checkpoint survives for a retry. A stopped box"
    echo "keeps /tmp/result.csv and its console log and bills only its 40 GB root volume."
    echo "Poll with:  PROFILE=$PROFILE bash tools/launch_pair.sh status"
    echo "Clean up:   PROFILE=$PROFILE bash tools/launch_pair.sh kill"
    ;;
status)
    # `stopped` matters as much as `running`: a box that finished has stopped itself, and
    # filtering it out would report success as "nothing there".
    LIVE="pending,running,stopping,stopped"
    "${A[@]}" ec2 describe-instances \
        --filters "Name=tag:Project,Values=$TAG" "Name=instance-state-name,Values=$LIVE" \
        --query 'Reservations[].Instances[].[InstanceId,InstanceType,Architecture,State.Name,PublicIpAddress]' \
        --output table
    say "result line, or the tail of the log if it is not done"
    for id in $("${A[@]}" ec2 describe-instances --filters "Name=tag:Project,Values=$TAG" \
                 "Name=instance-state-name,Values=$LIVE" \
                 --query 'Reservations[].Instances[].InstanceId' --output text); do
        echo "--- $id"
        log=$("${A[@]}" ec2 get-console-output --instance-id "$id" --output text 2>/dev/null || true)
        if [ -z "$log" ]; then
            echo "  (console not published yet, first boot takes a few minutes)"
        elif printf '%s' "$log" | grep -q 'BENCH COMPLETE'; then
            printf '%s' "$log" | sed -n '/instance,arch,vcpu/,$p' | head -3
        else
            printf '%s' "$log" | tail -12
        fi
    done
    echo
    echo "A stopped box still holds /tmp/result.csv on its root volume."
    echo "Done with them:  PROFILE=$PROFILE bash $0 kill"
    ;;
kill)
    ids=$("${A[@]}" ec2 describe-instances --filters "Name=tag:Project,Values=$TAG" \
           "Name=instance-state-name,Values=pending,running,stopping,stopped" \
           --query 'Reservations[].Instances[].InstanceId' --output text)
    [ -z "$ids" ] && { echo "nothing to terminate"; exit 0; }
    echo "terminating: $ids"
    "${A[@]}" ec2 terminate-instances --instance-ids $ids --query 'TerminatingInstances[].InstanceId' --output text
    ;;
userdata)  # print the boot script for an arch without launching anything.
    # Exists because a $-expansion bug in this heredoc once emptied the userdata and a box
    # booted to do nothing; this makes that checkable before money moves.
    userdata "${2:-arm64}" ;;

*)  echo "usage: $0 {check|launch [arm64|x86_64]|status|kill|userdata [arch]}" >&2; exit 2 ;;
esac
