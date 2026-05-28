#!/usr/bin/env bash
# Download the TUM RGB-D subsets used by this project.
#   xyz          -> epipolar pipeline (freiburg1)
#   pioneer_slam -> ICP pipeline       (freiburg2)
#
# Usage: bash scripts/download_tum.sh
set -euo pipefail

BASE="https://cvg.cit.tum.de/rgbd/dataset"
DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
mkdir -p "$DATA_DIR"

fetch() {
    local sequence="$1"   # e.g. freiburg1_xyz
    local family="${sequence%%_*}"   # freiburg1
    local url="${BASE}/${family}/rgbd_dataset_${sequence}.tgz"
    local tgz="${DATA_DIR}/rgbd_dataset_${sequence}.tgz"
    local out="${DATA_DIR}/rgbd_dataset_${sequence}"

    if [ -d "$out" ]; then
        echo "[skip] $out already exists"
        return
    fi

    echo "[fetch] $url"
    curl -L --fail -o "$tgz" "$url"
    echo "[extract] $tgz"
    tar -xzf "$tgz" -C "$DATA_DIR"
    rm -f "$tgz"
}

fetch freiburg1_xyz
fetch freiburg2_pioneer_slam

echo "Done. Contents of $DATA_DIR:"
ls -1 "$DATA_DIR"
