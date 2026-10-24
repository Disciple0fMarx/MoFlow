# [System_Architecture] Codebase Context for NotebookLM

Technical implementation reference for the Video-MoFlow SDD trajectory-prediction pipeline. Every statement below is grounded in the frozen code at commit `6d9c445` (`feature/sdd-phase1-refactor`). File paths are relative to the repo root; the SDD training entry point is `fm_sdd_global.py`, the flow-matching engine is `models/flow_matching.py`, the network is `models/backbone_eth_ucy.py`, and the LOSO dataloader is `data/dataloader_sdd_global.py`.

## 0. Pipeline snapshot (key constants)

| Quantity | Value | Source |
|---|---|---|
| Observation horizon | P = 8 frames | `SDD_PAST_FRAMES` |
| Prediction horizon | F = 12 frames (≈4 s at 3 frames/s) | `SDD_FUTURE_FRAMES` |
| Agents per window | A = 1 (`SDD_AGENTS_PER_WINDOW`) | `data/dataloader_sdd_global.py` |
| Trajectory channels per step | 6 = abs_xy + rel_xy + vel_xy | `NUM_INPUT_CONTEXT` |
| Raw video feature dim | 512 (ResNet-18) | `VIDEO_DIM_RAW` |
| Projected video dim | 32 | `VIDEO_DIM` |
| Context dim | D = 128 | `D_MODEL` |
| Candidate hypotheses | K = 20 | `denoising_head_preds` / `k_preds` |
| Sampling steps | S = 10, Euler (`solver='euler'`) | `sampling_steps` |
| Objective / wrapper | `pred_data` / `fm_wrapper='direct'` | defaults in `fm_sdd_global.py` |
| Time schedule | t ~ LogitNormal(μ=-0.5, σ=1.5) | `t_schedule`, `logit_norm_mean/std` |
| Noise level scale | σ_data = 0.0663 | `sigma_data` |
| Train / test batch | 128 / 256 | `train_batch_size`, `test_batch_size` |
| Epochs / optimizer | 150, AdamW lr 1e-4, wd 0.01, cosine+clip 1.0 | `OPTIMIZATION` |
| Val / eval seeds | val split seed 0; eval sampling reseeded 42 | `VAL_SEED`, `set_random_seed(42)` in `test()` |

The three architecture variants are: `_novid` (video branch absent), `_vmfull` (per-window ResNet-18 features, video branch on), `_vmstatic` (one fixed per-scene vector, video branch on) — set via `video_mode` in `fm_sdd_global.py` (`'off'→USE_VIDEO=False`; `'static'/'full'→USE_VIDEO=True`).

## 1. How the visual feature (z_vid) is injected into the flow-matching framework

**Feature construction (dataloader, `SDDGlobalDataset.__getitem__`).**
For each test/train window, `z_video_global` is produced as: per-frame ResNet-18 scene features `<scene>.npy` (one 512-d vector per (video, frame) row, via `SDDFrameFeatureLookup.window(...)`), sampled over the P=8 observed frames `past_frame_ids = [anchor-7, ..., anchor]` (stride 1, `policy="nearest"`), optionally Gaussian-smoothed in time (`VIDEO_SMOOTH_SIGMA`, 0.0 by default — identity), and then **mean-pooled over the 8 frames** → one `[512]` vector per window (equal-timestep weighting). The collate stacks these to `[B, 512]`. In `_vmstatic`, `_static_video(...)` instead serves one fixed scene-level mean vector for every window (video present but non-informative). If the cache is missing, the sample carries the scalar zero-sentinel `tensor([0.])`, which the collate detects and drops (`z_video_global=None`).

**Projection and concatenation (variance-A path; `models/context_encoder/eth_encoder.py`).**
The shipped Q1–Q7 configs run the **variant-A per-timestep concatenation** (tri-modal fusion disabled). Because the SDD contract is the 2-D pooled vector:

- `z = video_proj(z_video)` — a single linear layer `Linear(512→32)`, `[B,32]`;
- broadcast `z[:, None, None, :].expand(B, A, P, 32)` → `[B, A, P, 32]` (SAME video vector repeated over every agent and every observation timestep);
- `past_traj = cat([past_traj, z], dim=-1)` → trajectory channels become `[B, A, P, 6+32]`.

The 38-channel per-timestep token then flows through the agent MLP encoder (3-layer, 256 hiddens) → `[B,A,D=128]`, transformer self-attention (4 layers × 8 heads), to the context `z_ctx ∈ [B,A,128]`.

**Conditioning of the generative head (`models/backbone_eth_ucy.py`, `forward`).**
The denoiser network `D_θ(y_t, t, x_ctx)` receives the noisy future `y_t ∈ [B,K,A,24]` (24 = 12 frames × 2 coords) and:

1. `encoder_out` (containing all video influence) is broadcast to every hypothesis: `[B,K,A,128]`;
2. the noisy vector is embedded (`noisy_y_mlp`) and summed with `t-embedding` (FM time upscaled by ×1000; `time*1000.0`), hypothesis embedding `k_pe`, and agent embedding `a_pe`;
3. `emb_fusion = init_emb_fusion_mlp(cat(encoder_out_batch, y_emb, t_emb_batch))` — the context **gates/fuses** the noisy candidate;
4. an MTR-style decoder (4 blocks × 8 heads, cross-attention) reads out `reg_head → [B,K,A,24]` (the predicted x₁) and `cls_head → [B,K,A]` (one scalar logit per hypothesis, jointly trained for candidate selection).

`z_vid` therefore enters generatively **only through the past-context**, never directly into `y_t`; all K=20 hypotheses share one context vector.

**Flow-matching mechanics (for the math sections).**
Training (per-sample): sample `t~LogitNormal`, per-hypothesis noise `ϵ~N(0,I)` (independent heads, `tied_noise=False`), interpolate `x_t = t·x₁ + (1-t)·ϵ`, target `x₁` (true future; `pred_data`), weight `l_weight=1` (`fm_wrapper='direct'`), loss = winner-takes-all best-of-K L2 **scene- or agent-wise** (`LOSS_NN_MODE='agent'`, `WTA` over K via `CrossEntropy` on `cls_head`, `loss = reg·1.0 + cls·1.0 + vel·0.2 + div·0.1`). Sampling: `y(x=0) ~ N(0,I)`, 10 Euler steps `y ← y + v(x_t,t)·Δt` with `v = (x̂₁ - x_t)/(1-t)`.

## 2. How the dataloader enforces Leave-One-Out (LOSO/LOO) across SDD scenes

All scene logic is centralized in `data/dataloader_sdd_global.py`; both training (`fm_sdd_global.py:296`) and every attribution/viz tool build datasets through the identical constructor:

```python
if split == "train": scenes = [s for s in SDD_SCENES if s != held_out_scene]
elif split == "test": scenes = [held_out_scene]
```

- `SDD_SCENES = [bookstore coupa deathCircle gates hyang little nexus quad]` (8 scenes), hardcoded module constant.
- A run is parameterized by **one `--held_out_scene`**; the training set is the union of the other **seven** scenes; the test split is that scene **alone**. Every Q1–Q7 run therefore evaluates on a scene whose windows were never seen in training (geographic LOSO).
- Norm statistics are computed **only on the train-scene union** (`training=True` branch computes `min/max` over train windows; the held-out scene is excluded from stats → no normalization leakage). At eval, `_ensure_norm_stats` loads the stored train stats to unnormalize.
- Window indexing (`build_window_index`) walks `annotations/<scene>/<video>/annotations.txt` (bbox centers), keeps only **contiguous frame runs** of length ≥20, emits one window per run anchored at its observed horizon (`anchor_frame`), with `(scene_idx, video_idx, track_id, anchor_frame)` per row.
- Optional `video_id_filter` (the Angle-2 `video0` ablation) is applied **after** scene selection: it filters the video folders of whichever scene(s) survived LOSO, so the geographic train/test boundary is preserved even on the video subset.
- **Model selection is disjoint from reporting** (`trainer/denoising_model_trainers.py:52`): a deterministic 10% validation split (`VAL_FRACTION=0.1`, generator seed `VAL_SEED=0`) is carved from the **train** dataset via `random_split`; `checkpoint_best.pt` is chosen on that validation split and the reported ADE/FDE come from the held-out test scene (blind). This prevents the classic train/test model-selection leak.

## 3. How ADE is computed in evaluation (`trainer/denoising_model_trainers.py`)

The ADE reported in Q1–Q7 tables is the **best-of-K** (K=20) minimum displacement error, averaged over agents — an INDICATIVE minimINED oracle metric, not an expectation over the model's distribution.

Mechanics, in order:

1. `test(mode="best")` loads `checkpoint_best.pt`, calls `set_random_seed(42)` (so the noise draws are reproducible), then `sample(data, num_trajs=20)`.
2. Sample → K=20 hypotheses `[B,A,K,12,2]` (continuous, **stochastic** in noise `y_0~N(0,I)`, K independent since `tied_noise=False`), unnormalized back to original scale via `unnormalize_min_max` (min/max stats from the train union; SDD coords are pixels ≈ meters under identity homography).
3. `distances = ‖pred − GT‖₂` per (agent, hypothesis, frame) → `[N, K, 12]`.
4. `ADF_min(h) = mean over frames ≤ h of distances, then min over K`; `FDE_min(h) = distance at the horizon frame, min over K`.
5. Metrics stored in `eval_test_metrics.csv` as **sums** over `num_trajs` agents; `python tools/parse_eval_csv.py` (and the q2/q6 aggregators) divide by `num_trajs` to recover the exact means.
6. Horizon buckets: SDD uses `freq=3` (3 frames ≈ 1 s; `assert 3·4 = 12`), so columns `ADE_min_{1s..4s}`, `FDE_min_{1s..4s}`. The headline ADE is the 4 s bucket.
7. Robustness: any OOM/shape-failed batch is **skipped** and flagged `partial`; the q6 aggregator selects the complete row (`partial=0`).

Note for the report: `ADE_min` here is *min-over-20-samples*, which is optimistically biased by construction (higher K ⇒ lower score) and is the standard SDD/ETH-UCY benchmark convention; the **deltas** used in Q1 (z_vid attribution) are computed identically on the paired baseline/zeroed conditions of the same windows, so the *gains* inherit the same convention.

## 4. Why intent detection (deathCircle roundabout exit choice) is hard in this formulation

These are architectural facts of the shipped config that jointly explain the difficulty (they are not opinions; each maps to a specific line/flag):

1. **Single-agent windows** — `SDD_AGENTS_PER_WINDOW=1`, `AGENTS: 1`. Each SDD window contains exactly one pedestrian; the graph/social encoder has no cross-agent edges, and there is no vehicle/agent type encoding. Roundabout exit choice is an inherently **interactive** decision (who yields to whom per arm); the model only ever sees the focal agent's own kinematics — intent-relevant interaction context is structurally absent.

2. **Scene conditioning is global, pooled, and spatially ungrounded** — `z_vid` is the mean of 8 ResNet-18 frame vectors broadcast identically to all timesteps/agents. It carries no per-agent spatial grounding: no cross-attention between "where the agent IS" and the scene layout, no localization of the roundabout entrance/arms, and no goal/exit signal. Coordinate grounding itself is an **identity homography (pixels ≈ meters)** with predictions in pixel space, so there is no metric/world alignment to reason about arms or bearings.

3. **No discrete intent/exit latent** — the head regresses a continuous 12×2 **relative displacement** via a single flow matching ODE. Multimodality is represented only *implicitly*: K=20 hypotheses distinguished solely by a positional embedding `k_pe` and independent Gaussian noise, all sharing the identical context. The only anti-collapse mechanisms are the winner-takes-all mining (selecting argmin best-of-K; `cls_head` is a scalar per-hypothesis likelihood, not a per-exit classifier) and a hinge diversity term (`div` weight 0.1, margin 0.25). There is no semantic notion of "exit" anywhere in the output space.

4. **Smooth single-field assumption** — one continuous conditional vector field `v_t` from `N(0,I)` must place probability across separated branches of the future (roundabout fingers). With a shared context and a winner-takes-all loss that optimises only the best hypothesis, rare exits receive vanishing mass — the ODE concentrates on the dominant mode, and "intent" (which exit) is exactly the least-calibrated quantity. The 8-frame observation horizon (≈2.7 s at the 3 frames/s convention) also means the merge decision is observed only in its early, pre-decisive phase.

**Implication for Q1–Q7 write-up:** video conditioning (z_vid) improves scene-level hypotheses and typical-window ADE, but cannot by itself recover intent because the *incentive* (winner-takes-all, shared context) and the *representation* (no discrete latent, no interaction edges, pooled/pixel-space scene prior) do not express the exit decision. The deathCircle window-gain evidence (Q4/Q5/Q7) should be discussed under this lens: gains concentrate where the scene prior disambiguates branches, i.e. at windows where one arm dominates.