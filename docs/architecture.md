# Pipeline architecture and compute plan

Single machine, offline batch pipeline. There is no serving load, so there are no queues, caches or services: plain Python modules that write to one DuckDB file, which the Panel app reads.

## 1. Requirements

- **Functional:** for each (dataset × defense × training size × seed), train a target model; score every audited record with LOSS and RMIA; store per-record score, loss, final-epoch gradient norm, subgroup and true membership; flag records at 0.1% / 1% FPR using thresholds learned **only from reference models**.
- **Non-functional:** reproducible (fixed seeds, `uv.lock`, pinned submodule); resumable (each trained model is cached on disk and skipped if present); total compute under `[COMPUTE_HOURS]`; dashboard runs offline from DuckDB.

## 2. Data split (fixed once per dataset, seed 0)

```
all records ──► target pool  T (2·S_large) ──► members M_S (S) | non-members (S)   ← audited records
            ├─► shadow pool  H (2·S_large) ──► each reference model trains on a random S-subset (pkeep 0.5)
            └─► population   P (remainder) ──► RMIA's z-samples (OUT for every model)
```
- Adult: S ∈ {2,000, 10,000}, so T = H = 20,000 and P = 5,222. Fashion-MNIST: S ∈ {5,000, 15,000}, so T = H = 30,000 and P = 10,000.
- The small size uses a fixed subset of the large size's members and non-members, so records are comparable across sizes.
- Target **members are fixed across seeds**; seeds change only initialisation and batch order. Per-record flag variation across seeds is therefore pure training randomness (the Jebreel stability question).
- Test accuracy is measured on the target's non-members. The proxy **dev / test record split** (Phase 3) is a fixed 50/50 split of T by record id.

## 3. Calibration without touching target members

For each reference model r: treat it as a pseudo-target, attack it with the remaining R−1 references on its own shadow IN/OUT records, take its threshold at the target FPR. The target's threshold is the **median over r** (lira_analysis protocol). The real target is attacked with the same R−1 references, so pseudo-targets and target see identical attack strength. RMIA's `offline_a` is tuned the same way (fixing Privacy Meter's paired-model leak).

## 4. Storage (DuckDB, `data/leakatlas.duckdb`)

| table | grain | key columns |
|---|---|---|
| `records` | record × dataset | dataset, record_id, sex, race, age_band, class_name |
| `runs` | model × seed | run_id, dataset, defense, epsilon, train_size, seed, test_acc, train_loss, test_loss, train_sec, eps_spent (RDP) |
| `record_scores` | record × run | run_id, record_id, member, loss, grad_norm, score_loss, score_rmia |
| `thresholds` | run × attack × FPR | run_id, attack, fpr, threshold (from references only) |

Size estimate: Adult 6 defenses × 3 seeds × (4k + 20k) audited records ≈ 0.43M rows; Fashion-MNIST ≈ 0.72M rows. About 1.2M rows in total, tens of MB in DuckDB. Datashader is needed only for the per-record density views (> 10k points).

## 5. Compute projection (from `docs/bench_training.jsonl`)

Per model: overfit baseline 100 (Adult) / 60 (F-MNIST) epochs; anti-overfit ≤ 60 / 40 epochs with early stopping, counted at the cap; DP-SGD 30 epochs; MIAShield 1.3 × baseline (K = 5 sub-models on S/5 each, plus overhead). Plus a gradient-norm pass and evaluation per model. Cells per dataset: 6 defenses × 2 sizes = 12, and each cell trains 3 targets + R references.

| R references | Adult (CPU) | F-MNIST (MPS) |
|---|---|---|
| 4 | 0.13 h | 1.50 h |
| 8 | 0.20 h | **2.36 h** |
| 16 | **0.35 h** | 4.08 h |
| 32 | 0.64 h | 7.51 h |

Bold = proposed. DP-SGD is ≈ 75% of F-MNIST time (3 ε values × 30 epochs × 4.5 s). Adult runs on CPU and F-MNIST on MPS, so the two can run concurrently.
