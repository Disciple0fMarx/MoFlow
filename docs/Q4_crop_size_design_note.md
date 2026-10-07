# Q4 — Justifying the 64×64 agent crop size

**Supervisor question 4:** justify the `64×64` crop.

**Short answer:** `AGENT_CROP_SIZE: [64, 64]` is the crop hyperparameter of the
**agent-centric** video branch (`AgentVideoEncoder`). It is **inert in the Q1–Q7
experiments this suite produces** — `USE_AGENT_VIDEO: False` and
`USE_TRI_MODAL_FUSION: False` in the shipped `cfg/sdd/cor_fm.yml`, and the SDD
dataloader returns a `torch.zeros(1)` placeholder for `agent_crops` on every
sample (`data/dataloader_sdd.py:368`). Every reported number is produced by the
**global scene-video** branch (mean-pooled ResNet-18 scene features), which does
no spatial cropping. The knob is inherited from the ETH/UCY variant where agent
crops are genuinely extracted, and 64×64 was chosen there as follows.

## Where the knob lives

```yaml
# cfg/sdd/cor_fm.yml  (also cfg/eth_ucy/cor_fm.yml)
MODEL:
  CONTEXT_ENCODER:
    USE_AGENT_VIDEO: False       # agent branch off by default
    USE_TRI_MODAL_FUSION: False  # cascaded cross-attention OFF
    AGENT_CROP_SIZE: [64, 64]    # [height, width] px, consumed by AgentVideoEncoder
    AGENT_ENCODER_CHUNK_SIZE: 32 # micro-batch guard for the backbone forward
    AGENT_ENCODER_TYPE: resnet18
    VIDEO_DIM_RAW: 512           # ResNet-18 feature dim (global + agent)
    VIDEO_DIM: 32                # projected video dim
```

The agent branch is only activated by `fm_sdd_agent.py` / `fm_sdd_eth.py`
(these force `USE_AGENT_VIDEO=True`, `USE_TRI_MODAL_FUSION=True` and read
`--crop_size`, default 64; see `USAGE.md §4.3`).

## Why 64×64 (design rationale for the branch that uses it)

1. **Backbone spatial budget.** ResNet-18 down-samples by 2× per stage
   (conv1-stem + 4 stages) for a total stride of 16. A `64×64` crop therefore
   collapses to a `4×4` spatial feature grid before global average pooling —
   exactly the `VIDEO_DIM_RAW = 512`-dim vector `video_proj` maps to
   `VIDEO_DIM = 32` (`models/context_encoder/eth_encoder.py:98-102`). Any
   multiple of 16 would satisfy the stride contract; 64 is the smallest that
   keeps the agent's pixels well inside the backbone's receptive field.
2. **Scene scale.** SDD/ETH tracks are recorded from a bird's-eye camera; an
   agent spans on the order of tens of pixels. A 64 px window centered on the
   agent (world position mapped through the scene homography via
   `world_to_pixel`) covers the actor plus its immediate interaction
   neighbourhood without drowning the network in unrelated background.
3. **Compute cost.** The agent branch runs the backbone once per agent per
   observation time-step, so the crop is the dominant cost. 64×64 is the
   cheapest power-of-16 crop that does not alias the actor out of the kernel
   footprint; `AGENT_ENCODER_CHUNK_SIZE: 32` additionally micro-batches those
   forwards to bound peak memory.
4. **Edge handling.** Crops that would exceed the frame boundary are
   zero-padded (`dataloader_eth_ucy.py` `_extract_agent_crops`). Within a scene
   all agents are visible in-frame for the annotated windows, so the padding is
   a defensive path rather than a frequent one.

## Status in the Q1–Q7 report

| Fact | Evidence |
|------|----------|
| Agent branch inactive | `cfg/sdd/cor_fm.yml:58` `USE_AGENT_VIDEO: False` |
| No crops are emitted | `data/dataloader_sdd.py:368` returns `torch.zeros(1)` placeholder for `agent_crops`; the SDD collate drops them |
| Active conditioning channel | **global scene video**: per-window mean-pooled ResNet-18 frame features, `video_proj` → 32-d, gated with the trajectory stream |
| Conclusion | the 64×64 crop contributes **zero** to any q1…q7 number |

`build_supervisor_layout` snapshots this note plus the two shipped configs into
`{RESEARCH_ROOT}/q4/` (`Q4_crop_size_design_note.md`,
`Q4_hyperparameter_config_cor_fm.yml`, `Q4_hyperparameter_config_eth_ucy_cor_fm.yml`).

## When 64×64 would matter (and how to revisit it)

If the reviewer asks for an **agent-centric** follow-up, the knob becomes live
and is then a real hyperparameter requiring an LOSO-style ablation:

```bash
for cs in 32 48 64 96 128; do
  python fm_sdd_agent.py --cfg cfg/sdd/cor_fm.yml \
      --held_out_scene <SCENE> --crop_size $cs ...
done
```

and re-running `02_zvid_attribution.sh` per crop size to check whether the
agent-crop channel shows the same attribution signature the global channel
shows in Q1. Until then, the honest statement is: *the 64×64 crop is a dormant
hyperparameter of a disabled branch; the reported video conditioning is the
global scene-video channel.*