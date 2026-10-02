#!/usr/bin/env bash
# 00_reencode_features.sh — (Re)encode the clean per-scene ResNet-18 feature
# caches for every SDD scene.
#
# This is a one-time maintenance step, not a research-suite step (no 0X report
# is produced — see the suite docs). It exists because the PRE-FIX caches were
# corrupted: SDDFrameFeatureLookup (single-video handling) reports rows shared
# across videos as missing, i.e. the old encoder collapsed frame ids. These
# caches must be regenerated BEFORE any video-conditioned training / Q1-Q4 run.
#
# For each scene it runs:
#   python -m video_encoder encode-sdd \
#       --sdd-root <SDD_ROOT> --out <FEATURES_ROOT> --scenes <scene>
# which writes clean <scene>.npy + <scene>.manifest.parquet.
#
# One scene per subprocess so a single bad video cannot poison the others;
# by default other scenes continue and a failure summary is printed at the
# end (rc=1 if anything failed). Set FAIL_FAST=1 to abort on the first error.
#
# Usage:
#   scripts/research/00_reencode_features.sh                 # all 8 scenes
#   scripts/research/00_reencode_features.sh --scene hyang   # one scene
#   FAIL_FAST=1 scripts/research/00_reencode_features.sh
#   RESEARCH_DRY_RUN=1 scripts/research/00_reencode_features.sh
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=00_common.sh
source "${DIR}/00_common.sh"

LABEL="reencode-features"
SCENE=""
FAIL_FAST="${FAIL_FAST:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene) SCENE="$2"; shift 2 ;;
        --fail-fast) FAIL_FAST=1; shift ;;
        --dry-run) RESEARCH_DRY_RUN=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 3 ;;
    esac
done

OUTDIR="${RESEARCH_ROOT}/features"
mkdir -p "$OUTDIR"
LOG="${OUTDIR}/reencode-run.log"
: > "$LOG"
write_provenance "$OUTDIR" "scripts/research/00_reencode_features.sh"

SCENE_LIST="$(loso_scenes "$LABEL" "$SCENE")"
log_to "$LOG" INFO "re-encoding feature caches. scenes=(${SCENE_LIST}) features_root=${FEATURES_ROOT} fail_fast=${FAIL_FAST}"

FAILED=0
for scene in ${SCENE_LIST}; do
    log_to "$LOG" INFO "[${scene}] encode-sdd begin"
    if [[ "${RESEARCH_DRY_RUN:-0}" == "1" ]]; then
        log_to "$LOG" INFO "[${scene}] (dry-run) would run: python -m video_encoder encode-sdd --sdd-root ${SDD_ROOT} --out ${FEATURES_ROOT} --scenes ${scene}"
        continue
    fi
    if ! (cd "$REPO_ROOT" && "$PYTHON_BIN" -m video_encoder encode-sdd \
            --sdd-root "$SDD_ROOT" \
            --out "$FEATURES_ROOT" \
            --scenes "$scene"); then
        log_to "$LOG" ERROR "[${scene}] encode-sdd FAILED"
        FAILED=1
        if ((FAIL_FAST)); then
            log_to "$LOG" ERROR "[${scene}] FAIL_FAST set — aborting"
            exit 8
        fi
        continue
    fi
    if [[ ! -f "${FEATURES_ROOT}/${scene}.npy" || ! -f "${FEATURES_ROOT}/${scene}.manifest.parquet" ]]; then
        log_to "$LOG" ERROR "[${scene}] encode reported success but cache files missing: ${FEATURES_ROOT}/${scene}.npy / ${scene}.manifest.parquet"
        FAILED=1
        continue
    fi
    log_to "$LOG" INFO "[${scene}] OK: ${FEATURES_ROOT}/${scene}.npy + ${scene}.manifest.parquet"
done

if ((FAILED)); then
    log_to "$LOG" ERROR "one or more scenes FAILED; re-run for the failed scenes only, e.g. ./00_reencode_features.sh --scene hyang"
    exit 1
fi
log_to "$LOG" INFO "all ${SCENE_LIST} feature caches re-encoded"