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
