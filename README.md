# LeakAtlas

**Which training records does a model leak through membership inference, whose records are they, and could a
cheap, attack-free signal have warned us in advance?**

LeakAtlas audits every training record of small Adult and Fashion-MNIST models under five defenses, stores
per-record attack scores, losses and gradient norms in DuckDB, tests whether two attack-free proxies predict
which records (and which demographic subgroups) are exposed, and presents it all in a linked, explorable
dashboard where changing the seed, defense, ε or FPR shows whether "this record is at risk" survives.

CIS 545 Data Security & Privacy, University of Michigan-Dearborn.

![Record Atlas](results/figures/viewer_01_record_atlas.png)

Results with confidence intervals: **[results/summary.md](results/summary.md)** (generated from the stored data,
no hand-typed numbers).

## Research question

Can attack-free proxies, (A) the per-sample gradient norm at the end of training (Tobaben et al.) and (B) the loss
ratio (Jebreel et al.), predict which records and subgroups are most exposed well enough for a compliance reviewer
to triage risk without running a full attack? And how does each defense reshape that per-record risk map, not just
the average?

## Threat model

| | |
|---|---|
| **Attacker access** | Black-box query access to the target model's outputs (per-class probabilities), nothing else: no weights, gradients or training code beyond the public recipe. |
| **Attacker knowledge** | Shadow data from the same distribution as the training data (a disjoint pool), used to train reference models with the same recipe as the target, plus a population sample for RMIA. Thresholds are calibrated on those reference models only, never on the target's members (realistic-LiRA protocol). |
| **Protected information** | Whether a specific record was in the training set (membership). For Adult this links a person's census record to the training set; subgroup attributes (sex, race, age band) are analysed for disparate exposure. |
| **Trust boundary** | The model's prediction interface. Everything behind it (training data, weights, MIAShield's exclusion oracle, DP noise) is trusted; everything that crosses it (queries and returned probabilities) is attacker-observable. |
| **Defender** | Can measure the attack-free proxies (it owns its training data and a hold-out) and apply a defense: anti-overfitting training, DP-SGD (ε ∈ {1, 4, 8}, δ = 1/N, RDP accountant) or a MIAShield ensemble. |
| **Out of scope** | White-box and label-only attacks, adaptive attacks against MIAShield's oracle, data poisoning, attribute inference. |

## What is built

| Stage | Module | What it does |
|---|---|---|
| Data | `src/leakatlas/data.py` | UCI Adult (45,222 rows after dropping missing values, CC BY 4.0) and Fashion-MNIST (70,000 images, MIT); preprocessing and subgroup labels. |
| Models | `src/leakatlas/models.py` | MLP (Adult) and small CNN (Fashion-MNIST), no BatchNorm (Opacus-compatible). |
| Defenses | `src/leakatlas/defenses.py` | Fixed target/shadow/population split; overfit baseline, anti-overfitting (weight decay + label smoothing + early stopping), DP-SGD (Opacus, RDP), MIAShield (5 disjoint sub-models + exact-signature exclusion oracle); resumable training cache. |
| Attacks | `src/leakatlas/attacks.py` | RMIA and LOSS through ML Privacy Meter's `attacks.py` (pinned submodule, unmodified); reference-only threshold calibration; underflow-free log-loss and per-sample log gradient norms (torch.func). |
| Store | `src/leakatlas/store.py` | DuckDB, one row per record × configuration × seed; Pydantic row schemas; Polars for analysis. |
| Proxies | `src/leakatlas/proxies.py` | Proxy evaluation (tuned on seed 0 × dev records, evaluated once on seeds 1–2 × test records), stability across seeds, Adult subgroup gaps with bootstrap CIs. |
| Analysis | `src/leakatlas/analysis.py` | Per-configuration privacy/utility/cost table and `results/summary.md`, all from queries. |
| Dashboard | `src/leakatlas/viewer.py`, `theme.py` | Panel + HoloViews/Bokeh + Datashader; 7 linked views, one editorial theme ([design notes](docs/design_principles.md)). |

Recon notes on the tools (verified from source): [docs/tool_notes.md](docs/tool_notes.md). Pipeline design and
compute plan: [docs/architecture.md](docs/architecture.md).

## Setup

Python 3.12, [uv](https://docs.astral.sh/uv/). CPU or Apple-silicon MPS; no CUDA needed.

```bash
git clone --recurse-submodules https://github.com/spacedust10/leakatlas
```
```bash
cd leakatlas && uv sync
```
```bash
uv run playwright install chromium
```

The datasets are downloaded on first use into `data/raw/` (gitignored).

## Commands

| Command | What it does | Time (M3 Pro) |
|---|---|---|
| `make train` | Full pipeline: train every target and reference model, audit, analyse, render figures. | ≈ 2.5 h training + ≈ 15 min audit/analysis |
| `make reproduce` | Rebuild every table, `results/summary.md` and every figure **from `data/leakatlas.duckdb` only** (no training, no network). | ≈ 6 min |
| `make serve` | Open the dashboard: `panel serve src/leakatlas/viewer.py`. | instant |
| `make test` | Check the underflow-free gradient-norm computation against float64 autograd. | ≈ 1 min |

Dashboard views are addressable by URL, e.g. `/viewer?dataset=adult&defense=dp&eps=4&seed=all&fpr=0.01&record=6620`.

## Config → results → figures

| Config | Produces | Shown in |
|---|---|---|
| `configs/adult.yaml`: S ∈ {2,000, 10,000}, 16 reference models, 6 defenses × 3 seeds | 36 target runs, 432,000 record rows | All views with `dataset=adult`; subgroup ridgelines (view 3) are Adult-only |
| `configs/fmnist.yaml`: S ∈ {5,000, 15,000}, 8 reference models, 6 defenses × 3 seeds | 36 target runs, 720,000 record rows | All views with `dataset=fmnist`; image record cards (view 6) |
| — | `results/config_table.md` | privacy (AUC, TPR@0.1%/1% FPR, reference-calibrated TPR + realized FPR), utility, cost per configuration |
| — | `results/phase3_*.md` | proxy evaluation, model-level loss ratio vs TPR, stability, subgroup gaps |
| — | `results/figures/viewer_01_record_atlas.png` | Record Atlas: gradient norm vs attack score, flags across seeds |
| — | `results/figures/viewer_02_log_roc.png` | Log-log ROC per defense with seed bands |
| — | `results/figures/viewer_03_subgroup_ridgelines_adult.png` | Subgroup score distributions + flag-rate CIs |
| — | `results/figures/viewer_04_proxy_vs_truth.png` | Per-record proxy density and model-level scatter |
| — | `results/figures/viewer_05_stability_strip.png` | Records by number of seeds that flag them |
| — | `results/figures/viewer_06_record_inspector*.png` | Record cards (Fashion-MNIST and Adult) |
| — | `results/figures/viewer_07_defense_tradeoff.png` | Accuracy vs TPR@FPR, Pareto front, cost |

## Repository layout

```
src/leakatlas/{data,models,defenses,attacks,proxies,store,analysis,viewer,theme}.py
configs/            experiment matrices (YAML)
data/               DuckDB, model cache, raw data (gitignored)
results/            summary.md, tables, figures
docs/               tool notes, architecture, design principles
scripts/            timing benchmark, screenshot capture
tests/              gradient-norm check
third_party/ml_privacy_meter   pinned submodule (e384af8)
```

## Deviations from the original plan (all documented in the notes)

- **RMIA instead of LiRA**: ML Privacy Meter implements only RMIA and LOSS; RMIA is its reference-model attack and works with few references.
- **Privacy Meter fixes**: its LOSS pipeline is sign-inverted when fed probabilities (we feed log-loss) and its `offline_a` tuner calibrates on the model trained on the target's complement (we tune on reference models only).
- **MIAShield**: the brief's `um-dsp/MIAShield` repository does not exist; the authors' code (`ijarin/MIAShield-Code`, TensorFlow notebook) was re-implemented in PyTorch with the exact-signature oracle only.
- **Numerics**: per-record loss and gradient norm are computed in log space; float32 (and even float64) underflow had tied thousands of records at exactly zero.

## References

- Tobaben et al., arXiv:2402.06674 — per-sample gradient norms at the end of training correlate with membership-inference vulnerability.
- N. Jebreel, M. Khalil, D. Sánchez, J. Domingo-Ferrer, "Revisiting the LiRA Membership Inference Attack Under Realistic Assumptions", PoPETs 2026, arXiv:2603.07567. Code: github.com/najeebjebreel/lira_analysis.
- N. Carlini, S. Chien, M. Nasr, S. Song, A. Terzis, F. Tramèr, "Membership Inference Attacks From First Principles", IEEE S&P 2022 (LiRA; TPR at low FPR on log-scale ROC curves).
- I. Jarin, B. Eshete, "MIAShield: Defending Membership Inference Attacks via Preemptive Exclusion of Members", PoPETs 2023, arXiv:2203.00915.
- S. Zarifzadeh, P. Liu, P. Bhatt, R. Shokri, "Low-Cost High-Power Membership Inference Attacks", ICML 2024 (RMIA).
- ML Privacy Meter, privacytrustlab, github.com/privacytrustlab/ml_privacy_meter (MIT).
- Opacus, github.com/pytorch/opacus (DP-SGD with the RDP accountant).
- B. Becker, R. Kohavi, "Adult", UCI Machine Learning Repository, 1996 (CC BY 4.0). H. Xiao, K. Rasul, R. Vollgraf, "Fashion-MNIST", arXiv:1708.07747, 2017 (MIT).
- Design: C. Wilke, *Fundamentals of Data Visualization*; R. Alexander, *Telling Stories with Data*; Distill, "How to Use t-SNE Effectively" (2016).
