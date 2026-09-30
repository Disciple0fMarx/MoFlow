# Research Script Suite — Empirical Evidence for the Thesis Defense / Journal

This suite generates the empirical evidence answering each of the supervisor's
seven review questions about the SDD video-conditioned FlowMatching model. All
commands follow the **strict LOSO protocol**: every script loops over all 8
SDD scenes by default, and accepts a single optional `--scene <name>` to
restrict the loop to one held-out scene.

```
00_common.sh                 shared env + helpers (paths, seeds, LOSO loop)
01_video_mode_ablation.sh    Q6 : video vs static vs trajectory-only
02_zvid_attribution.sh       Q1 : is the gain actually from z_vid?
03_transfer_matrix.sh        Q2 : is the gain scene-specific or transferable?
04_gain_geometry.sh          Q5/Q7 : where video helps + roundabout intent
```

## Question → Script map

| # | Supervisor question | Script | Evidence produced |
|---|--------------------|--------|-------------------|
| Q1 | Gain comes from z_vid? | `02_zvid_attribution.sh` | ADE/FDE under baseline / zeroed / permuted `z_video_global` on the *same* seeded noise; `d_traj`/`d_ade` deltas per scene |
| Q2 | Gain matrix diagonal? | `03_transfer_matrix.sh` | 8×8 and gain matrices; diag = LOSO effect, off-diag = transferable component |
| Q3 | Strict LOSO protocol + ETH/UCY reproducibility | built into every script + `PROVENANCE.md` | per-run dir per held-out scene, fixed seeds, sampling schedule, git SHA recorded |
| Q4 | Justify the 64×64 crop | (design note) | crop size is a hyperparameter of the agent crop path (`AGENT_CROP_SIZE`); revisit if agent-crop ablations are needed |
| Q5 | Draw gain by scene ↔ complex geometry? | `04_gain_geometry.sh` | per-scene mean gain table, ranked; low-gain scenes cross-referenced with non-linear/roundabout corridors |
| Q6 | Video vs mere visual info? | `01_video_mode_ablation.sh` | off / static / full ADE — isolates per-window visual *content* from any-image conditioning |
| Q7 | Roundabout exit = intent? | `04_gain_geometry.sh` `--scene deathCircle` | gain-ranked roundabout windows + multi-hypothesis renders |

## Methodology notes (publication-grade)

- **Fixed randomness.** Every arm of a family uses the same `--seed` (default
  42) and `--sampling_steps` (10), so ADE differences are attributable to the
  conditioning channel, not sampling. `PROVENANCE.md` records seed, SHA,
  paths per report.
- **Attribution RNG discipline** (`tools/attrib_video_conditioning.py`): the
  torch RNG is re-seeded immediately before *every* `sample()` call, so
  baseline/zeroed/permuted arms draw **identical** initial noise and differ
  only in `z_video_global`.
- **Norm-stats decoupling (Q2).** When evaluating a checkpoint trained with
  held-out scene A on an unrelated scene B, inputs are normalized with **A's**
  train-split statistics (`--norm-scene A --held-out-scene B`), never B's —
  otherwise cross-scene numbers measure normalization shift, not transfer.
- **LOSO correctness (Q3).** Each run directory is named
  `_SDD_ho<scene>_<variant>` and only the held-out scene's windows form the
  test split; norm stats are always derived from the *training* split.

## Report schema

```
report/research/
├── q1/  <scene>_attrib.csv            (baseline/zeroed/permuted ADE+FDE+deltas)
├── q2/  <train>__eval_<eval>__{full,novid}.csv   (per-cell baseline)
│        q2_transfer_ade.csv           (8×8 ADE matrix)
│        q2_transfer_gain.csv          (8×8 gain matrix, ade_novid − ade_full)
├── q4/  <scene>_windows.csv           (per-window baseline/zeroed ADE)
│        q4_window_gains.csv           (concatenated per-window gains)
│        q4_scene_gain_summary.csv     (scenes ranked by mean video gain)
│        figs/*.png                    (gain-ranked trajectory renders)
└── q6/  q6_video_mode_ade.csv         (off/static/full per scene × horizon)
```

## Quick start

```bash
# Q6 — require the three trained checkpoints per scene first:
scripts/research/01_video_mode_ablation.sh            # train + eval all scenes
scripts/research/01_video_mode_ablation.sh --train-only --scene coupa

# Q1 — attribution on each scene's `full` checkpoint:
scripts/research/02_zvid_attribution.sh --scene coupa

# Q2 — quick transfer matrix (3 batches/cell), full sweep = drop --n-batches:
scripts/research/03_transfer_matrix.sh --n-batches 3
scripts/research/03_transfer_matrix.sh --n-batches 0   # full 8×8

# Q5/Q7 — per-window gains + gain-ranked renders (+ roundabout pass):
scripts/research/04_gain_geometry.sh --n-batches 5 --top-k 3
scripts/research/04_gain_geometry.sh --scene deathCircle

# Dry-run any script to print the exact commands without executing:
RESEARCH_DRY_RUN=1 scripts/research/02_zvid_attribution.sh
```

## Lab machine facts baked into defaults (override via env)

The default `SDD_ROOT`, `FEATURES_ROOT`, and `CUDA_VISIBLE_DEVICES=0` match the
lab box (single RTX 4080). Override with env vars, e.g.
`SDD_ROOT=/data/SDD FEATURES_ROOT=/data/feats scripts/research/01_…sh`.