#!/usr/bin/env bash
# Applies the four timeout edits to the Arbor fork (F67, F70, F72) and checks them.
# The fork lives in a gitignored folder, so every checkout has to do this once:
#   bash scripts/snellius/apply_arbor_timeouts.sh
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ARBOR=$REPO/gepa_artifact/utils/arbor
V=$ARBOR/arbor/server/services/inference
check() {
  grep -q "timeout=7200)" "$V/vllm_client.py" && grep -q "default=21600," "$V/vllm_serve.py" \
    && [ "$(grep -c 'timeout=600.0)' "$V/vllm_serve.py")" = 3 ] && ! grep -q "timeout=60.0)" "$V/vllm_serve.py"
}
if check; then echo "arbor timeouts already applied"; exit 0; fi
git -C "$ARBOR" apply --check "$REPO/scripts/snellius/arbor_timeouts.patch" || { echo "patch does not apply cleanly, the fork is not at the expected state (113fc35 + nothing else)"; exit 1; }
git -C "$ARBOR" apply "$REPO/scripts/snellius/arbor_timeouts.patch"
check && echo "arbor timeouts applied: forwarder 7200 s, worker 21600 s, chunk call 600 s" || { echo "patch applied but the check failed"; exit 1; }
