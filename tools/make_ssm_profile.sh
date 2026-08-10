#!/usr/bin/env bash
# make_ssm_profile.sh - one-time IAM setup so a benchmark box is diagnosable.
#
# The first launch of launch_pair.sh died two minutes in and left nothing to read: no key
# pair, no SSM, and an empty console log. Stopping rather than terminating preserved the
# root volume, which was the right call, but with no way in it did not help. This grants
# the instance AmazonSSMManagedInstanceCore, so `aws ssm start-session --target <id>`
# works with no inbound port, no key material and no security group changes.
#
# Idempotent. Run once per account.
#
#   PROFILE=myorg bash tools/make_ssm_profile.sh
set -euo pipefail

PROFILE="${PROFILE:?set PROFILE to an aws profile}"
NAME="${NAME:-kimi-bench-ssm}"
A=(aws --profile "$PROFILE")

TRUST='{"Version":"2012-10-17","Statement":[{"Effect":"Allow",
        "Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

"${A[@]}" iam get-role --role-name "$NAME" >/dev/null 2>&1 \
    || "${A[@]}" iam create-role --role-name "$NAME" \
         --assume-role-policy-document "$TRUST" --query 'Role.RoleName' --output text

"${A[@]}" iam attach-role-policy --role-name "$NAME" \
    --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore

"${A[@]}" iam get-instance-profile --instance-profile-name "$NAME" >/dev/null 2>&1 || {
    "${A[@]}" iam create-instance-profile --instance-profile-name "$NAME" \
        --query 'InstanceProfile.InstanceProfileName' --output text
    "${A[@]}" iam add-role-to-instance-profile --instance-profile-name "$NAME" --role-name "$NAME"
}

echo "ready:"
"${A[@]}" iam get-instance-profile --instance-profile-name "$NAME" \
    --query 'InstanceProfile.[InstanceProfileName,Roles[0].RoleName]' --output text
echo
echo "To read a box that failed:"
echo "  aws --profile $PROFILE ec2 start-instances --instance-ids <id>"
echo "  aws --profile $PROFILE ssm start-session --target <id>"
echo "  sudo cat /var/log/kimi-bench.log"
