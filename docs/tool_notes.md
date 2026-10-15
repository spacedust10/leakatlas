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

## 2. lira_analysis (Jebreel et al., PoPETs 2026)

Reusable ideas, verified in the code:
- **Shadow-only threshold** (`comprehensive_analysis/analysis_utils.py:compute_shadow_thresholds`): for a target FPR, compute each *shadow* model's ROC threshold on its own known membership, then use the **median of the other models' thresholds** for the target. This is the protocol we adopt for every "flagged at 1% / 0.1% FPR" decision.
- **TPR at FPR** (`metrics.py:compute_tpr_at_fpr`): the largest threshold with FPR ≤ target. **Precision under a prior** (`compute_precision`, Bayes with π).
- **Loss ratio** (`utils/common.py:~389`): `test_loss / train_loss` from the final-epoch metrics row, per model. That is the model-level Proxy B.
- **LiRA** (`attacks/lira.py`): online/offline with per-sample or fixed variance; score = `log(p_true) − log(1 − p_true)` averaged over augmentations. Offline LiRA needs many OUT models (paper: 256 shadow models). Quirk: online fixed-variance sets `sigma_out = np.std(dat_in)`, copied from Carlini et al.'s reference code.
- **Per-sample reproducibility**: a record counts as vulnerable only if flagged in ≥ k of the runs ("support thresholding"). That is our Stability Strip.
- Config scale: `num_shadow_models: 256`, `pkeep: 0.5`. Infeasible here, which is why we use RMIA (strong with few references).

## 3. MIAShield (Jarin & Eshete, PoPETs 2023)

- One Colab notebook (TensorFlow/Keras, CIFAR-10/100, plus `tensorflow_privacy`, `imagehash`, `opencv`). Nothing can be imported; the logic is re-implemented in PyTorch.
- Mechanism (README + notebook): split the training set into **K disjoint subsets**, train one model per subset, and at query time an **exclusion oracle** drops the sub-model(s) whose subset contains the query; the remaining models' outputs are aggregated.
- Oracles named in the README: **EO2-ESE** (exact signature, a hash of the input), **EO3-ASE** (approximate signature, perceptual image hash), **EO4-MCE** (model-confidence based), **EO5-COE** (collusion-based).
- **Plan:** implement **ESE** (exact hash, so it works for tabular and image data with no new dependency). ASE needs `imagehash` (a new dependency); MCE and COE are out of scope unless you ask.

## 4. Datasets

| | Adult | Fashion-MNIST |
|---|---|---|
| Source | UCI ML Repository id 2, `archive.ics.uci.edu/static/public/2/adult.zip` (Becker & Kohavi 1996) | Zalando Research via `torchvision.datasets.FashionMNIST` (Xiao, Rasul, Vollgraf 2017) |
| License | CC BY 4.0 | MIT |
| Version / hash | sha256 of zip `7537312dd56c2b98…` (`adult.data` + `adult.test`) | sha256 of the 4 raw `.gz` files `e71a29c25d6786b8…` |
| Records | 48,842 → **45,222** after dropping rows containing `?` (the standard count) | **70,000** (train + test pooled; we make our own splits) |
| Preprocessing | strip whitespace; label `income ∈ {>50K, >50K.}` = 1 (24.8% positive); z-score 6 numeric columns; one-hot 8 categorical columns → **104 features**. Sex and race stay as model inputs, as in standard Adult usage. | scale to [0,1], normalise mean 0.2860 / std 0.3530; 10 balanced classes |
| Subgroups | sex (M 30,527 / F 14,695); race (White 38,903, Black 4,228, Asian-Pac-Islander 1,303, **Amer-Indian-Eskimo 435, Other 353**); age band (<30, 30–44, 45–59, 60+ with 3,246) | class name (for the gallery) |

The two smallest race groups will produce wide bootstrap CIs; the subgroup views will show their n.

## 5. Timing benchmark (`scripts/bench_training.py` → `docs/bench_training.jsonl`)

Apple M3 Pro, torch 2.14.1, Opacus 1.6.0. No CUDA; MPS (Apple GPU, part of the standard macOS wheel) available. Mean of 3 epochs after warm-up.

| Dataset (train size) | Device | s / epoch, standard | s / epoch, DP-SGD (Opacus) | per-sample grad-norm pass over 2N records (torch.func) |
|---|---|---|---|---|
| Adult MLP (10k) | **CPU** | **0.032** | **0.354** | 0.66 s |
| Adult MLP (10k) | MPS | 0.056 | 0.750 | 1.75 s |
| F-MNIST CNN (15k) | CPU | 2.553 | 4.926 | 10.3 s |
| F-MNIST CNN (15k) | **MPS** | **0.701** | **4.544** | 6.15 s |

The tiny MLP is faster on CPU (kernel-launch overhead dominates on MPS). The CNN is 3.6× faster on MPS for standard training, but DP-SGD gains little because Opacus's per-sample-gradient hooks dominate.
