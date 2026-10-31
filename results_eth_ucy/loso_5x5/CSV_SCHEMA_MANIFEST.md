# CSV Schema Manifest (ETH/UCY LOSO 5×5)

Generated from Kaggle runs (kernel: dhyaelbahri/moflow-eth-ucy-tu-clausthal-m1-m6-pipeline, branch fix/eth-ucy-alignment) using strict TU-Clausthal raw data (no pickles). Splits follow raw SocialGAN frame ranges; UCY uses annotation.vsp + H.txt; ETH/hotel use obsmat.txt. Video features built from full-range visual_data JPEGs (stride 10) via video_encoder/scripts/build_ethucy_features.py.

## Aggregated (summary)
- `loso_5x5_metrics.csv` / `results_loso_5x5_metrics.csv` (same): per (held_out_scene, video_mode, split) with ADE_min/FDE_min/ADE_avg/FDE_avg (K = cfg.denoising_head_preds = 20). `*_min` = best-of-K, `*_avg` = mean over all K. Columns: held_out_scene,video_mode,split,ade_min,fde_min,ade_avg,fde_avg,n_agents,status,checkpoint
- `loso_5x5_summary.csv` / `results_loso_5x5_summary.csv`: macro averages over 5 scenes by video_mode (off/static/full)

## Per-scene test evaluations (from evaluator)
Files: `results_{scene}_{mode}.csv` for all scenes {eth,hotel,univ,zara1,zara2} × modes {off,static,full}
Schema (from tools/eval_eth_transfer.py): scene, mode, split, ade_min, fde_min, ade_avg, fde_avg, n_samples/trajs (n_agents as reported). Values are LOSO test metrics on held-out scene.

## Validation/eval detail CSVs (extracted)
- `eth_full_eval_val_metrics.csv`, `hotel_static_eval_val_metrics.csv`, `univ_off_eval_val_metrics.csv`
- `MoFlow_results_eth_ucy_loso_5x5_zara2_static_eval_val_metrics.csv` (zara2/static val detail)
These contain per-timestep/diagnostic rows (long format) produced during evaluation/validation; see headers in files for full column set.

## Notes
- All 15 (5×5) test runs completed (status ok). 
- Checkpoints referenced under `/kaggle/working/checkpoints/{scene}_{mode}.pt`.
- Metrics computed with strict raw LOSO (held-out scene fully excluded), honoring checkpoint VIDEO_MODE.
