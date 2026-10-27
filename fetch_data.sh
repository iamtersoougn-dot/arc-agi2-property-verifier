#!/usr/bin/env bash
# Downloads the public inputs (not redistributed here): ARC-AGI-2 data and the NVARC harness candidate exports.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/cands
[ -d data/ARC-AGI-2 ] || git clone --depth 1 https://github.com/arcprize/ARC-AGI-2.git data/ARC-AGI-2
[ -d data/harness ] || git clone --depth 1 https://github.com/Actualjudgepresident/arc2026-nvarc-harness.git data/harness
tar -xf data/harness/runs/s0/cands_seed0.tar -C data/cands
tar -xf data/harness/runs/s1/cands_seed1.tar -C data/cands
echo "data ready"
