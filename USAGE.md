# USAGE — MoFlow (Video-Conditioned Flow Matching)

This guide documents the end-to-end command-line pipeline for training and
evaluating the **Correlated Flow Matching (CFM) Teacher** on the Stanford
Drone Dataset (SDD), with the three conditioning variants:

| Variant        | Conditioning                                 | Entry point       | Run tag suffix        |
|----------------|----------------------------------------------|-------------------|-----------------------|
| Baseline (`no_video`) | Trajectory-only (no visual information)     | `fm_sdd_global.py` | `_SDD_ho<scene>_novid` |
| Global Video   | Scene-level features (`z_video_global`)     | `fm_sdd_global.py` | `_SDD_ho<scene>_vid`    |
| Agent-Centric  | Per-agent crops + tri-modal fusion          | `fm_sdd_agent.py`  | `_SDD_ho<scene>_agent`  |

All variants share the same training stack (`ETHMotionTransformer` +
`FlowMatcher` + `Trainer`), so the checkpoints are byte-for-byte compatible and
can be compared directly with the shared visualization tool.

---

## 1. Environment

Requires Python ≥ 3.10, a PyTorch installation with CUDA, and the dependencies
in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Run every command from the repository root. On machines without the lab SDD
mount, pass `--sdd_root <path>` explicitly (see Section 2).

---

## 2. Dataset layout

The SDD dataset must be laid out as follows (the adapter auto-detects either
the Remote Lab Machine root or the Kaggle mount; an explicit `--sdd_root`
always wins):

```
<SDD_ROOT>/
├── annotations/<scene>/*.txt      # one file per video; rows = xmin, ymin, xmax, ymax, frame_id, track_id, ...
└── videos/<scene>/<video_id>/     # "videos/" on the lab machine, "video/" on Kaggle
    └── frames/frameNNNNNN.jpg     # pre-extracted JPEG frames
```

The eight SDD scenes are:

```text
bookstore  coupa  deathCircle  gates  hyang  little  nexus  quad
```

> The repository intentionally does **not** ship a copy of SDD. Point
> `--sdd_root` at your dataset root.

---

## 3. Data preparation — extracting spatio-temporal features

Global video features are extracted **once** and cached per scene. The encoder
scans the annotated frames of each scene, encodes them with a ResNet-18
backbone, and writes a per-scene cache consumed by the dataloaders at training
time.

### 3.1 Encode all SDD scenes

```bash
python -m video_encoder encode-sdd \
    --sdd-root /path/to/SDD \
    --out ./features/resnet18
```

Optional flags:

| Flag           | Default              | Meaning                                   |
|----------------|----------------------|-------------------------------------------|
| `--scenes`     | all 8 scenes         | Scene subset, e.g. `--scenes bookstore nexus` |
| `--out`        | `./features/resnet18`| Output directory                          |
| `--device`     | auto                 | `cuda` / `cpu`                            |
| `--batch-size` | 32                   | Batch size                                |

### 3.2 Output

For each scene the command writes:

```text
./features/resnet18/<scene>.npy               # [N_frames, D] float32 features
./features/resnet18/<scene>.manifest.parquet  # scene, video_id, frame_id, row_idx
```

These caches are read by the Global Video dataloader and by the visualization
tooling. Only the **Global** variant requires them; the baseline and
agent-centric variants do not.

### 3.3 Encode a single arbitrary video

For quick experiments on a standalone clip:

```bash
python -m video_encoder encode-video \
    --video /path/to/clip.mov \
    --out ./features/from_video \
    --stride 10
```

### 3.4 Note on SDD raw annotations

The schedulers read the raw SDD annotations directly from
`<data_root>/annotations/<scene>/<video_id>/annotations.txt` (this is the same
root layout used by the encoder in Section 3.1). `eval_sdd.py` points at it
with `--data_dir` (Section 5.3).

---

## 4. Training

Training uses **Leave-One-Scene-Out (LOSO)**: hold out one scene for testing
and train on the remaining seven. Replace `<SCENE>` with one of the eight
scene names and `<SDD_ROOT>` with your dataset root.

### 4.1 Baseline (no video)

```bash
python fm_sdd_global.py \
    --cfg cfg/sdd/cor_fm.yml \
    --held_out_scene <SCENE> \
    --no_video \
    --sdd_root /path/to/SDD
```

`--no_video` disables the global branch (trajectory-only). Point
`--sdd_root` at your SDD root if it is not the auto-detected default.

### 4.2 Global Video Encoder

```bash
python fm_sdd_global.py \
    --cfg cfg/sdd/cor_fm.yml \
    --held_out_scene <SCENE> \
    --sdd_root /path/to/SDD \
    --video_features_root ./features/resnet18
```

If `--video_features_root` is omitted, the model receives zero placeholders (OK
for smoke tests, meaningless for real evaluation).

### 4.3 Agent-Centric Video Encoder

```bash
python fm_sdd_agent.py \
    --cfg cfg/sdd/cor_fm.yml \
    --held_out_scene <SCENE> \
    --sdd_root /path/to/SDD \
    --crop_size 64 \
    --padding 0
```

The agent-centric script forces `USE_AGENT_VIDEO=True` and
`USE_TRI_MODAL_FUSION=True` internally; it does not need the global feature
cache. Key tuning flags: `--crop_size` (default 64), `--padding`,
`--drop_lost`, `--agent-encoder {auto,compact,resnet18}`.

### 4.4 Shared training flags

| Flag                | Default | Meaning                                          |
|---------------------|---------|--------------------------------------------------|
| `--exp`             | ``      | Experiment label appended to the run directory   |
| `--epochs`          | cfg     | Override number of epochs                        |
| `--batch_size`      | cfg     | Override batch size                              |
| `--sampling_steps`  | 10      | Euler steps used at inference                    |
| `--seed`            | 42      | RNG seed                                         |
| `--fix-random-seed` | off     | Reproducible seeding                             |
| `--init-lr`         | cfg     | Override peak learning rate                      |
| `--checkpt-freq`    | 5       | Checkpoint interval (epochs)                     |
| `--max-num-ckpts`   | 5       | Max kept checkpoints                             |

### 4.5 Output layout

Runs are written under the config's `results_root_dir` (default `results_sdd/`):

```text
results_sdd/cor_fm/_SDD_ho<SCENE>_novid/models/checkpoint_best.pt   # baseline
results_sdd/cor_fm/_SDD_ho<SCENE>_vid/models/checkpoint_best.pt     # global
results_sdd/cor_fm/_SDD_ho<SCENE>_agent/models/checkpoint_best.pt   # agent
```

Each run also contains `log/` (training log), `samples/`, and periodic
checkpoints. `checkpoint_best.pt` is selected by best min-ADE and is the file
used by evaluation and visualization.

---

## 5. Evaluation (ADE/FDE metrics)

The trainers compute per-scene min/avg ADE and FDE (plus JADE/JFDE and
variance indicators) during validation. To run a dedicated evaluation pass on
the best checkpoint:

### 5.1 Baseline and Global

```bash
# Baseline
python fm_sdd_global.py \
    --cfg cfg/sdd/cor_fm.yml \
    --held_out_scene <SCENE> \
    --no_video \
    --sdd_root /path/to/SDD \
    --eval

# Global
python fm_sdd_global.py \
    --cfg cfg/sdd/cor_fm.yml \
    --held_out_scene <SCENE> \
    --sdd_root /path/to/SDD \
    --video_features_root ./features/resnet18 \
    --eval
```

### 5.2 Agent-Centric

```bash
python fm_sdd_agent.py \
    --cfg cfg/sdd/cor_fm.yml \
    --held_out_scene <SCENE> \
    --sdd_root /path/to/SDD \
    --crop_size 64 \
    --eval
```

Metrics (ADE_min, FDE_min, ADE_avg, FDE_avg, JADE, JFDE, AVar/FVar, MASD) are
printed to the log and stored in `log/log_eval_*.txt`.

### 5.3 Evaluate a specific checkpoint

To evaluate an explicit checkpoint regardless of run tag, use `eval_sdd.py`
(reads the raw SDD annotations directly):

```bash
python eval_sdd.py \
    --cfg auto \
    --ckpt_path results_sdd/cor_fm/_SDD_ho<SCENE>_vid/models/checkpoint_best.pt \
    --data_dir /path/to/SDD \
    --batch_size 1024
```

---

## 6. Visualization

`run_visualize_trajectories.py` renders publication-quality figures (observed
trajectory, ground-truth future, and the sampled `K`-hypothesis bundle) on the
exact video frame at the end of the observation window, one PNG per
scene/variant/window at 300 DPI.

### 6.1 Checkpoint discovery

By default the tool discovers checkpoints by scene run-tag under
`--checkpoint-dir`:

```text
no_video:      <dir>/_SDD_ho<scene>{_novid,_novideo,_baseline}/models/checkpoint_best.pt
global_video:  <dir>/_SDD_ho<scene>{_vid,_globalvideo,_global}/models/checkpoint_best.pt
agent_centric: <dir>/_SDD_ho<scene>_agent/models/checkpoint_best.pt
```

Print the resolved plan without rendering anything:

```bash
python run_visualize_trajectories.py \
    --checkpoint-dir results_sdd/cor_fm \
    --video-features-root ./features/resnet18 \
    --discover-only
```

### 6.2 Random window sampling (per scene)

By default one random observation window per scene is rendered with a seedable
RNG; the same windows are used for every variant so figures are directly
comparable:

```bash
python run_visualize_trajectories.py \
    --checkpoint-dir results_sdd/cor_fm \
    --video-features-root ./features/resnet18 \
    --sdd-root /path/to/SDD \
    --num-samples 1 \
    --seed 42 \
    --variants no_video global_video agent_centric
```

- `--num-samples N` — number of random windows **per scene** (default 1).
- `--seed S` — RNG seed for deterministic random sampling.
- `--variants ...` — subset of `{no_video, global_video, agent_centric}`.

### 6.3 Explicit target filtering

Random sampling is bypassed when explicit filters are given; the single
(first, deterministic) matching window is rendered.

Render one specific pedestrian track, every scene:

```bash
python run_visualize_trajectories.py \
    --checkpoint-dir results_sdd/cor_fm \
    --video-features-root ./features/resnet18 \
    --sdd-root /path/to/SDD \
    --ped-id 77
```

Render the window anchored at a specific frame number, every scene:

```bash
python run_visualize_trajectories.py \
    --checkpoint-dir results_sdd/cor_fm \
    --video-features-root ./features/resnet18 \
    --sdd-root /path/to/SDD \
    --window-idx 2519
```

Restrict to a single scene (repeatable flag) and combine both filters:

```bash
python run_visualize_trajectories.py \
    --checkpoint-dir results_sdd/cor_fm \
    --video-features-root ./features/resnet18 \
    --sdd-root /path/to/SDD \
    --scene nexus \
    --ped-id 77 \
    --window-idx 2519
```

> `--window-idx` is an alias of `--frame-id` (the anchor frame number), and
> `--ped-id` is an alias of `--agent-id` (the track id). Both override random
> sampling.

### 6.4 Explicit checkpoints

Instead of discovery, pass checkpoints per variant (repeatable):

```bash
python run_visualize_trajectories.py \
    --scene nexus \
    --variant-ckpt no_video=results_sdd/cor_fm/_SDD_honexus_novid/models/checkpoint_best.pt \
    --variant-ckpt global_video=results_sdd/cor_fm/_SDD_honexus_vid/models/checkpoint_best.pt \
    --variant-ckpt agent_centric=results_sdd/cor_fm/_SDD_honexus_agent/models/checkpoint_best.pt \
    --video-features-root ./features/resnet18 \
    --sdd-root /path/to/SDD \
    --ped-id 77 --window-idx 2519
```

### 6.5 Output

Figures are written to `--out-dir` (default `visualizations/`) with the naming
scheme:

```text
{out_dir}/{scene}_ped{agent_id}_{video_id}_f{anchor_frame}_{variant}.png
# e.g. nexus_ped77_video9_f2519_agent_centric.png
```

Rendering options: `--sampling-steps` (default 10), `--use-ema`, `--no-background`
(white placeholder), `--background <image>`, `--zoom-margin` (default 90),
`--max-agents` (cap windows per scene-variant), `--demo` (checkpoint-free
styling self-check).

Peer tools: `visualize_trajectories.py` is the standalone rendering backend;
`viz_agent_sdd.py` renders the agent-centric variant for a single tracked
pedestrian.

---

## 7. Config file reference

`cfg/sdd/cor_fm.yml` controls the denoising/generation setup
(`objective`, `sigma_data`, `t_schedule`), the optimizer, and the model. The
video-specific switches under `MODEL.CONTEXT_ENCODER` are:

```yaml
USE_VIDEO: False             # Global video branch
USE_AGENT_VIDEO: False       # Agent-centric branch
USE_TRI_MODAL_FUSION: False  # Cascaded cross-attention (Step A + B)
VIDEO_DIM_RAW: 512           # ResNet-18 feature dimension
VIDEO_DIM: 32                # Projected video dimension
AGENT_CROP_SIZE: [64, 64]    # Crop height/width [px]
RESNET_FREEZE_BLOCKS: 2      # Frozen ResNet blocks
AGENT_ENCODER_TYPE: resnet18 # resnet18 | compact
USE_VIDEO_GATING: True       # Learnable kinematic-video gate
```

The entry points override these switches internally (e.g. `fm_sdd_agent.py`
forces `USE_AGENT_VIDEO=True`), so the config keeps them `False` by default.

---

## 8. Quick reference — full pipeline at a glance

```bash
# 1. Data preparation (global features, run once)
python -m video_encoder encode-sdd --sdd-root /path/to/SDD --out ./features/resnet18

# 2. Train the three variants
python fm_sdd_global.py --cfg cfg/sdd/cor_fm.yml --held_out_scene <SCENE> \
    --sdd_root /path/to/SDD --no_video
python fm_sdd_global.py --cfg cfg/sdd/cor_fm.yml --held_out_scene <SCENE> \
    --sdd_root /path/to/SDD --video_features_root ./features/resnet18
python fm_sdd_agent.py  --cfg cfg/sdd/cor_fm.yml --held_out_scene <SCENE> \
    --sdd_root /path/to/SDD --crop_size 64

# 3. Evaluate
python fm_sdd_global.py --cfg cfg/sdd/cor_fm.yml --held_out_scene <SCENE> \
    --sdd_root /path/to/SDD --video_features_root ./features/resnet18 --eval
python fm_sdd_agent.py  --cfg cfg/sdd/cor_fm.yml --held_out_scene <SCENE> \
    --sdd_root /path/to/SDD --eval

# 4. Visualize a specific window in one scene
python run_visualize_trajectories.py \
    --checkpoint-dir results_sdd/cor_fm \
    --video-features-root ./features/resnet18 \
    --sdd-root /path/to/SDD \
    --scene <SCENE> --ped-id <ID> --window-idx <FRAME>
```

---

## Troubleshooting

- **`YAML file [...] does not exist`** — run from the repository root and pass
  `--cfg cfg/sdd/cor_fm.yml`.
- **Global variant trains with zero placeholders** — `--video_features_root` is
  missing or does not contain `<scene>.npy`; run Section 3.1 first.
- **`FileNotFoundError: ... checkpoint missing`** in visualization — the run
  tag does not match the discovery patterns in Section 6.1; pass
  `--variant-ckpt VARIANT=PATH` explicitly.
- **CUDA out of memory** — lower `--batch_size`, or for the agent-centric
  variant reduce `--crop_size`, or pass `--grad_accum_steps` (native AMP and
  memory guards are applied automatically).