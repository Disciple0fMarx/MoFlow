# Video-Conditioned Flow Matching for Human Trajectory Prediction on SDD
## A Question-by-Question Empirical Response to Supervisor Review Q1–Q7

> **Working outline** — copy/paste-ready skeleton. Complements the existing
> LaTeX report (`report/main.tex`, `report/bilan.tex`); reuse its prose,
> tables, and architecture figures where noted.

- **Authors / Date:** <fill>
- **Repo SHA:** <from `report/research/*/PROVENANCE*.md`>
- **Defended protocol:** Geographic LOSO over 8 SDD scenes
  (bookstore, coupa, deathCircle, gates, hyang, little, nexus, quad);
  seed 42, 10 sampling steps, K=20, horizons 1.2–4.8 s.
- **Secondary angle:** single-video `video0` replicate (robustness).

---

## Guiding principles (apply throughout)

- **Q-sequential core, narrative frame.** Keep the supervisor's Q1→Q7 order in
  the results, but bookend with front matter (setup) and a **Synthesis** that
  re-tells the causal chain:
  *Q6 establishes that per-window content matters → Q1 proves it flows through
  `z_vid` → Q2 shows it generalizes → Q5/Q7 locate where/why → Q4 bounds the
  mechanism → Q3 secures the protocol.*
- **Lead with the defended protocol.** Geographic LOSO (unsuffixed artifacts) is
  the main line; `_vidvideo0` runs are a robustness replicate → put them in the
  appendix or as a paired sub-panel.
- **One claim per figure.** Each result states a
  Claim → Method → Metric (table) → Figure → Interpretation → Answer.
- **Every figure is referenced in prose before it appears**, and its caption =
  *what is shown + key number + takeaway*, with the source path in the appendix.
- **Do not overstate the `static` prior.** Call it the *scene-level prior*
  (mean of ≤8 coarse keyframes), not "a single keyframe".

---

# Outline

## 0. Answers at a glance  `[ ]`
Preview table so a reader gets the spine before the detail.

| Q | Question (short) | One-line answer | Key metric | Evidence |
|---|---|---|---|---|
| Q1 | Is the gain from `z_vid`? | <…> | ΔADE (zeroed/permuted vs baseline) | `q1/<scene>_attrib.csv` · Fig Q1.1–Q1.2 |
| Q2 | Is the 8×8 matrix diagonal (transferable)? | <…> | diag vs off-diag gain | `q2_transfer_gain.csv` · Fig Q2.1 |
| Q3 | Strict LOSO + reproducibility? | <…> | protocol audit | `PROVENANCE.md` · Fig Q3.1 |
| Q4 | Why a 64×64 crop? | <…> | design note (inactive in global suite) | `Q4_crop_size_design_note.md` · Fig Q4.1 |
| Q5 | Does gain track scene geometry? | <…> | per-scene mean gain | `q4_scene_gain_summary.csv` · Fig Q5.1–Q5.2 |
| Q6 | Video vs. mere visual info? | <…> | off/static/full ADE | `q6_video_mode_ade.csv` · Fig Q6.1–Q6.3 |
| Q7 | Roundabout exit = intent? | <…> | qualitative | `q4/figs/*` · Fig Q7.1 |

---

## 1. Introduction  `[ ]`
- 1.1 Motivation: why visual scene context for trajectory prediction.
- 1.2 Contributions: causal attribution, content-vs-presence ablation,
  transfer analysis, geometry/intent case study.
- 1.3 Map of the seven questions → section pointer (mirror §0).
- *Reuse:* `report/main.tex` §"Introduction and System Motivation".

## 2. Method recap (brief; cite, don't re-derive)  `[ ]`
- 2.1 Conditional Flow Matching objective + Euler sampler (K=20, 10 steps).
- 2.2 Architecture: frozen ResNet-18 global scene encoder → `z_video_global` →
  per-timestep concat.
- 2.3 The three arms: `off` (video branch not built), `static` (one fixed scene
  prior), `full` (per-window mean-pooled features).
- 2.4 Coordinate note: identity homography (annotation pixels ≈ metres).
- *Reuse:* `report/main.tex` §"Model Architecture", `report/arch_*.tex`,
  `report/arch_*.pdf`.

## 3. Experimental setup  `[ ]`
- 3.1 Dataset + geographic LOSO (train = 7 scenes, val = 10 % holdout of train,
  test = held-out scene).
- 3.2 Single-video `video0` ablation as the secondary angle.
- 3.3 Metrics (minADE/minFDE over K=20, per horizon), norm-stats decoupling.
- 3.4 Provenance & fixed randomness.
- **Table 3.1 Setup summary** `[ ]`
- **Fig Q3.1 / Fig 3.1** = LOSO + val-holdout schematic `[ ]`
- *Reuse:* `report/main.tex` §"Leave-One-Scene-Out Evaluation Protocol".

---

## 4. Results: Q1–Q7
Each as: **Claim → Method → Metric (table) → Figure → Interpretation → Answer.**

### Q1 — Is the gain attributable to `z_vid`?  [mechanism]  `[ ]`
- Method: baseline / zeroed / permuted `z_video_global` on identical seeded noise.
- **Table Q1.1** baseline·zeroed·permuted ADE/FDE + Δ per scene
  → `report/research/q1/<scene>_attrib.csv`
- **Fig Q1.1 — Mechanism (Grad-CAM).** 8-frame overlay on the anchor frame
  → `results_sdd/gradcam_visualizations/<scene>/w<idx>/frame_f*_overlay.jpg`
- **Fig Q1.2 — Attribution bars.** ΔADE per condition (zeroed/permuted collapse
  the gain).
- Answer: <one sentence>.

### Q2 — Is the transfer matrix diagonal?  [generalization]  `[ ]`
- Method: cross-scene eval with train-scene norm stats (`--norm-scene A`).
- **Fig Q2.1 — 8×8 gain heatmap** (diag = LOSO effect, off-diag = transferable)
  → `report/research/q2/q2_transfer_gain.csv` (+ `q2_transfer_ade.csv`).
- **Table Q2.1** diag vs off-diag mean.
- Answer: <…>.

### Q3 — Strict LOSO + reproducibility  [rigor]  `[ ]`
- **Fig Q3.1 — Protocol diagram** (scene-geography split; val carved from train
  only).
- **Table Q3.1 — Provenance ledger** (SHA, seed, steps, paths)
  → `report/research/q3/*` + `report/research/features/PROVENANCE*.md`.
- Answer: <…>.

### Q4 — Justifying the 64×64 crop  [design]  `[ ]`
- Note: crop path is **inactive** in the global-video suite; state rationale +
  when it activates.
- **Fig Q4.1 — Crop design schematic**; **Table Q4.1 — config snapshot**
  → `docs/Q4_crop_size_design_note.md`,
  `report/research/q4/Q4_hyperparameter_config_*.yml`.
- Answer: <…>.

### Q5 — Does gain track scene geometry?  [where]  `[ ]`
- **Table Q5.1 — scenes ranked by mean gain**
  → `report/research/q4/q4_scene_gain_summary.csv`
- **Fig Q5.1 — Ranked gain bar chart.**
- **Fig Q5.2 — Gain vs. corridor complexity** (low-gain scenes ↔
  non-linear/roundabout corridors) → `q4/q4_window_gains.csv`.
- Answer: <…>.

### Q6 — Video vs. mere visual information?  [content vs presence]  `[ ]`
- **Table Q6.1 — off/static/full ADE per scene × horizon**
  → `report/research/q6/q6_video_mode_ade.csv`
- **Fig Q6.1 — Ablation bars** (off < static < full) — the headline.
- **Fig Q6.2 — static vs full paired panels**
  → `results_sdd/visualizations/{global,video0}/<scene>[_video0]/sample_NNN_{static,full}_video.png`
- **Fig Q6.3 — static-prior vs full Grad-CAM**
  → `sample_NNN_{static,full}_gradcam.png`
- Answer: <… per-window content, not just any image>.

### Q7 — Roundabout exit as intent?  [qualitative case study]  `[ ]`
- `deathCircle` gain-ranked windows + multi-hypothesis renders
  → `report/research/q4/figs/*.png`
  (i.e. `<scene>_w<win>_trajectory_comparison.png`).
- **Fig Q7.1** off vs full at a roundabout exit (correct branch vs shortcut).
- Answer: <…>.

---

## 5. Synthesis  `[ ]`
- The causal chain across Q6→Q1→Q2→Q5→Q7→Q4→Q3 (one paragraph + a small flow
  figure).
- What is settled vs. what remains open.

## 6. Limitations & threats to validity  `[ ]`
- Single dataset (SDD); static prior is a control, not a method; `video0` as
  partial robustness; crop path dormant.

## 7. Conclusion  `[ ]`

## 8. Appendix  `[ ]`
- A. Full per-scene metric tables (all horizons, both angles).
- B. Repro commands (exact CLI per Q).
- C. Figure source-path index.

## 9. References  `[ ]`
- *Reuse:* `report/references.bib`.

---

# Visualization & asset map

| Figure | Section | Pull from | Purpose / caption must say |
|---|---|---|---|
| **Q1.1** Grad-CAM overlays | Q1 (echo Q6) | `results_sdd/gradcam_visualizations/<scene>/w<idx>/frame_f*_overlay.jpg` | Which scene regions the video branch attends to |
| **Q1.2** Attribution bars | Q1 | `report/research/q1/<scene>_attrib.csv` | ΔADE collapses when `z_vid` zeroed/permuted |
| **Q2.1** 8×8 gain heatmap | Q2 | `report/research/q2/q2_transfer_gain.csv` | Diagonal dominance = location-specific vs transferable |
| **Q3.1** LOSO diagram | Q3 | schematic (draw) | No test leak; val from train only |
| **Q4.1** Crop schematic | Q4 | `docs/Q4_crop_size_design_note.md` | Rationale; path inactive here |
| **Q5.1** Ranked gain bars | Q5 | `report/research/q4/q4_scene_gain_summary.csv` | Which scenes benefit |
| **Q5.2** Gain vs geometry | Q5 | `report/research/q4/q4_window_gains.csv` | Non-linear corridors ↔ low gain |
| **Q6.1** Ablation bars | Q6 | `report/research/q6/q6_video_mode_ade.csv` | off/static/full ordering |
| **Q6.2** static vs full (raw frame) | Q6 | `results_sdd/visualizations/**/sample_NNN_{static,full}_video.png` | Content, not mere presence |
| **Q6.3** static-prior vs full Grad-CAM | Q6 | `results_sdd/visualizations/**/sample_NNN_{static,full}_gradcam.png` | Per-window vs constant attention |
| **Q7.1** Roundabout renders | Q7 | `report/research/q4/figs/*.png` | Intent-level branch choice |

## Embedding rules
1. **Bind figure IDs to questions** (`Fig Qn.k`) so every visual maps to an answer.
2. **Pair the metric with the picture:** never show a render without the ADE
   numbers it illustrates (put ADE in the panel title — the tools already
   annotate `ADE_min`).
3. **Three-panel rhythm for Q6:** bar chart (aggregate) → static-vs-full frames
   (qualitative) → Grad-CAM pair (mechanism). Aggregate → instance → cause.
4. **Caption template:** *"Figure Qn.k — \<title\>. \<what is shown\>.
   full=red / static=orange / off=blue, GT=green, K=20 bundle faint.
   \<key number\>. Source: `<path>`."*
5. **Main vs appendix:** geographic-LOSO figures in the body; label the
   `video0` replicate "(robustness angle)" and cross-reference the appendix.
6. **Keep 300 DPI**, consistent variant colours, identical axis units ("[m]")
   across panels.

---

# Appendix B — reproduction commands (fill as you go)

```bash
# Q6 — off/static/full (train + eval; note: 01_* deletes vmstatic at the end)
scripts/research/01_video_mode_ablation.sh

# Q1 — z_vid attribution on each scene's `full` checkpoint
scripts/research/02_zvid_attribution.sh --scene <scene>

# Q2 — transfer matrix (3 batches/cell; 0 = full 8x8)
scripts/research/03_transfer_matrix.sh --n-batches 3

# Q5/Q7 — per-window gains + gain-ranked renders (+ roundabout pass)
scripts/research/04_gain_geometry.sh --n-batches 5 --top-k 3
scripts/research/04_gain_geometry.sh --scene deathCircle

# Figures — qualitative variants + Grad-CAM
python tools/generate_qualitative_figures.py --protocol both --num-samples 5
python tools/gradcam_sdd.py --scene <scene> --window-index 0
```
