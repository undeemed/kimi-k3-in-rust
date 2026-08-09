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
#   aws sso login --profile prox-dev-sso
#   PROFILE=prox-dev-sso bash tools/launch_pair.sh check     # quotas and AMIs only, free
#   PROFILE=prox-dev-sso bash tools/launch_pair.sh launch    # spends money
#   PROFILE=prox-dev-sso bash tools/launch_pair.sh status
#   PROFILE=prox-dev-sso bash tools/launch_pair.sh kill      # terminate both
set -euo pipefail

PROFILE="${PROFILE:?set PROFILE, e.g. PROFILE=prox-dev-sso}"
REGION="${REGION:-us-east-1}"
TAG=kimi-k3-bench
RUST_REPO="${RUST_REPO:-https://github.com/undeemed/kimi-k3-in-rust.git}"
A=(aws --profile "$PROFILE" --region "$REGION")

# im4gn is Graviton2, so the arm64 box is where the NEON kernels actually get tested.
# Plain case statements rather than associative arrays: macOS still ships bash 3.2, which
# has no `declare -A`, and this half runs on the laptop.
inst_for()     { case "$1" in arm64) echo im4gn.xlarge ;; x86_64) echo i3en.xlarge ;; esac; }
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
exec > >(tee -a /var/log/kimi-bench.log) 2>&1
set -x
apt-get update -qq && apt-get install -y -qq git curl
cd /root
git clone -q $RUST_REPO rust-port
DEV=\$(lsblk -dn -o NAME,TYPE | awk '\$2=="disk" && \$1 ~ /nvme/ {print "/dev/"\$1}' | tail -1)
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
    say "launching two boxes, tagged $TAG"
    for arch in arm64 x86_64; do
        id=$("${A[@]}" ec2 run-instances \
            --image-id "$(ami_for "$arch")" \
            --instance-type "$(inst_for "$arch")" \
            --instance-initiated-shutdown-behavior terminate \
            --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=40,VolumeType=gp3,DeleteOnTermination=true}' \
            --user-data "$(userdata)" \
            --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG-$arch},{Key=Project,Value=$TAG}]" \
            --query 'Instances[0].InstanceId' --output text)
        echo "  $arch  $(inst_for "$arch")  $id"
    done
    echo
    echo "Each box downloads 1.56 TB then runs. Budget 4-6 hours."
    echo "Poll with: PROFILE=$PROFILE bash tools/launch_pair.sh status"
    ;;
status)
    "${A[@]}" ec2 describe-instances \
        --filters "Name=tag:Project,Values=$TAG" "Name=instance-state-name,Values=pending,running" \
        --query 'Reservations[].Instances[].[InstanceId,InstanceType,Architecture,State.Name,PublicIpAddress]' \
        --output table
    say "last lines of each console log"
    for id in $("${A[@]}" ec2 describe-instances --filters "Name=tag:Project,Values=$TAG" \
                 "Name=instance-state-name,Values=running" \
                 --query 'Reservations[].Instances[].InstanceId' --output text); do
        echo "--- $id"
        "${A[@]}" ec2 get-console-output --instance-id "$id" --output text \
            | tail -15 || echo "  (console not published yet)"
    done
    ;;
kill)
    ids=$("${A[@]}" ec2 describe-instances --filters "Name=tag:Project,Values=$TAG" \
           "Name=instance-state-name,Values=pending,running" \
           --query 'Reservations[].Instances[].InstanceId' --output text)
    [ -z "$ids" ] && { echo "nothing running"; exit 0; }
    echo "terminating: $ids"
    "${A[@]}" ec2 terminate-instances --instance-ids $ids --query 'TerminatingInstances[].InstanceId' --output text
    ;;
*)  echo "usage: $0 {check|launch|status|kill}" >&2; exit 2 ;;
esac
