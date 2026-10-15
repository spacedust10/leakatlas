# Tool notes (Phase 1 recon)

Everything below was read from source at the pinned commits or measured on this machine; nothing is assumed.

| Repo | Commit | License | How it is used |
|---|---|---|---|
| `privacytrustlab/ml_privacy_meter` | `e384af8` (2025-04-26) | MIT | git submodule `third_party/ml_privacy_meter`; `attacks.py` imported directly |
| `najeebjebreel/lira_analysis` | `193896f` (2026-04-07) | MIT | reference only (gitignored clone); protocol and formulas re-implemented with citation |
| MIAShield code: `ijarin/MIAShield-Code` | `2858c11` (2024-03-20) | none stated | reference only; the brief's `um-dsp/MIAShield` does **not exist** (404); the um-dsp org has no MIAShield repo. `ijarin` is the first author's account and its README cites the PoPETs'23 paper. `MIAShield/MIAShield` has the same two notebooks. |

## 1. ML Privacy Meter

**Not a pip package.** No `setup.py` or `pyproject.toml`; it is a set of scripts (`run_mia.py`, `run_audit_dp.py`, ...) whose `requirements.txt` pins an old stack (torch 2.4.1, numpy 1.26, jupyter, transformers, peft). We pin it as a submodule and import only `attacks.py`, whose sole imports are numpy and scikit-learn. `audit.py` also works, except it imports `visualize.py` → matplotlib, which this project does not install.

### YAML config (`configs/*.yaml`, documented in `configs/README.md`)
- `run`: `random_seed`, `log_dir` (models in an existing `log_dir` are **reloaded, not retrained**), `time_log`, `num_experiments` (= number of target models attacked).
- `audit`: `privacy_game: privacy_loss_model`, `algorithm` (`RMIA` | `LOSS`), `num_ref_models`, `device`, `report_log`, `batch_size`, `data_size` (balanced audit subsample, must be even), `population_size`.
- `train`: `model_name` (`CNN`, `wrn28-1/2/10`, `vgg16`, `mlp`, `gpt2`, `speedyresnet`), `device`, `batch_size`, `optimizer` (`SGD`/`Adam`/`AdamW`), `learning_rate`, `weight_decay`, `epochs`, optional `peft`.
- `data`: `dataset` (`cifar10`, `cifar100`, `purchase100`, `texas100`, `agnews`; dispatched by name in `dataset/utils.py:get_dataset`), `data_dir`, `tokenize`/`tokenizer`.
- **Neither Adult nor Fashion-MNIST is supported.** Adding them means editing `dataset/utils.py`; we load data ourselves instead.

### Attacks actually implemented (`audit.py:audit_models`, `attacks.py`)
Only **two**: `RMIA` (Zarifzadeh et al. 2024, offline variant) and `LOSS`. Anything else raises `NotImplementedError`. **No LiRA, and no separate "reference-model" or "population" attack.** RMIA is itself the reference-model attack, and it uses population data `z`. (`run_range_mia.py` adds a range-membership variant of RMIA; `run_audit_dp.py` does DP auditing. Neither is needed here.)

### Per-sample scores
- Signal = true-class softmax probability per sample per model (`get_signals.get_softmax`), shape `[N, M]`, cached to `{log_dir}/signals/{algorithm}_signals.npy`.
- `run_rmia(target_idx, all_signals, population_signals, all_memberships, num_ref, offline_a)` → **one score per sample** (fraction of population `z` with likelihood ratio below the sample's). Higher means more likely a member.
- `run_loss(target_signals)` returns `-target_signals`. **Bug when fed softmax probabilities, which is what Privacy Meter's own pipeline does:** members get *lower* scores. Synthetic check: AUC 0.127 vs 0.873 with the correct sign. We pass per-sample **loss** into `run_loss`, so `-loss` ranks members higher.
- `audit.compute_attack_results` reports AUC and TPR at 1%, 0.1% and 0% FPR from `sklearn.roc_curve` (evaluation only).

### Number of reference models
- `num_model_pairs = max(ceil(num_experiments/2), num_ref_models + 1)`. Models are trained in **pairs on complementary halves** (`models/utils.py:split_dataset_for_training`), so the pipeline trains `2 × num_model_pairs` models (≥ 4 for `num_ref_models=1`).
- RMIA needs ≥ 1 OUT reference model per sample (the configs ship with `num_ref_models: 1`). LOSS needs **0**.
- **Calibration leak.** `tune_offline_a` picks RMIA's `offline_a` by maximising AUC against the *paired* model's membership. The paired model is trained on the complement of the target's data, so its labels are exactly the target's non-membership labels. That breaks the brief's "never calibrate on target members" rule. **Fix:** tune `offline_a` with a reference model as pseudo-target, never with the target or its pair.
- `run_rmia` drops columns `target_idx` and `target_idx±1` (its pair), so in our layout we place the target in column 0, a duplicate in column 1, and the references after them. Every reference is OUT on every audited record (offline setting). `run_rmia` is then used **unmodified**.

### Device
`device` is a plain torch device string from the config. `speedyresnet`/`fast_train.py` is CUDA-only. Our own MLP/CNN run on CPU and MPS (benchmark below).
