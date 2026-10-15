"""Membership-inference audit of every trained target, via ML Privacy Meter's attack code.

Usage: uv run python -m leakatlas.attacks configs/adult.yaml

- LOSS and RMIA scores come from Privacy Meter's `run_loss` / `run_rmia` (unmodified, pinned submodule).
  `run_loss` is fed per-sample *loss* (its own pipeline feeds softmax, which inverts the score).
- Every threshold and RMIA's `offline_a` is fit on reference models only (shadow-only calibration,
  Jebreel et al.): each reference is attacked as a pseudo-target by the others, and the target uses
  the median of their thresholds. Target membership is used only to *evaluate*.
"""

import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.func import functional_call, grad, vmap

from leakatlas.data import LOADERS
from leakatlas.defenses import MIAShield, load_config, log_probs, make_split, model_path
from leakatlas.store import FPR_GRID, AttackRow, RunRow, connect, replace_cell, replace_records

_pm_spec = importlib.util.spec_from_file_location(
    "privacy_meter_attacks", Path(__file__).resolve().parents[2] / "third_party/ml_privacy_meter/attacks.py")
pm = importlib.util.module_from_spec(_pm_spec)
_pm_spec.loader.exec_module(pm)
log = logging.getLogger("leakatlas.attacks")
CHUNK = 2048  # rows per run_rmia call; its [rows x population] ratio matrix is the memory peak


def signals(model, x, y, dev):
    """True-class prob, loss and correctness, computed in float64 from logits.

    float32 log_softmax saturates to loss == 0 for confident predictions, which ties members with
    non-members at the top of the score range and zeroes TPR at low FPR. Here
    loss_k = softplus(logsumexp_{j!=y} z_j - z_y) per (sub-)model, and MIAShield's served loss is
    -log(mean of kept p_k), done as a log-mean-exp.
    """
    shield = isinstance(model, MIAShield)
    subs = list(model.subs) if shield else [model]
    excl = model.oracle(x) if shield else torch.full((len(x),), -1)
    loss, correct = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x), 4096):
            xb, eb = x[i:i + 4096].to(dev), excl[i:i + 4096]
            z = torch.stack([m(xb) for m in subs], 1).cpu().double()                # [B, K, C]
            idx = y[i:i + 4096][:, None, None].expand(-1, len(subs), 1)
            zy = z.gather(2, idx).squeeze(2)
            margin = torch.logsumexp(z.scatter(2, idx, float("-inf")), 2) - zy      # [B, K]
            keep = torch.arange(len(subs))[None] != eb[:, None]
            logp = torch.logsumexp(torch.where(keep, -F.softplus(margin), float("-inf")), 1) - keep.sum(1).log()
            probs = (F.softmax(z, 2) * keep[..., None]).sum(1)
            loss.append(-logp)
            correct.append(probs.argmax(1) == y[i:i + 4096])
    loss = torch.cat(loss).numpy()
    return np.exp(-loss), loss, torch.cat(correct).numpy()


def grad_norms(model, x, y, dev, bs=512):
    """L2 norm of each record's loss gradient w.r.t. all parameters at the final weights (Tobaben et al.)."""
    model = model.to(dev).eval()
    params = {k: v.detach() for k, v in model.named_parameters()}
    shield = isinstance(model, MIAShield)
    excl_all = model.oracle(x) if shield else torch.zeros(len(x), dtype=torch.long)

    def loss_one(p, xi, yi, ei):
        if shield:
            lp = functional_call(model, p, (xi[None], ei[None]))
        else:
            lp = F.log_softmax(functional_call(model, p, (xi[None],)), -1)
        return F.nll_loss(lp, yi[None])

    per_sample = vmap(grad(loss_one), in_dims=(None, 0, 0, 0))
    out = []
    for i in range(0, len(x), bs):
        g = per_sample(params, x[i:i + bs].to(dev), y[i:i + bs].to(dev), excl_all[i:i + bs].to(dev))
        out.append(torch.sqrt(sum((v.flatten(1) ** 2).sum(1) for v in g.values())).cpu())
    return torch.cat(out).numpy()


def threshold_at(scores, member, fpr):
    """Largest threshold with FPR <= fpr (same rule as lira_analysis compute_tpr_at_fpr)."""
    f, _, t = roc_curve(member, scores)
    return t[np.where(f <= fpr)[0][-1]]


def shadow_thresholds(pseudo_scores, keep):
    """Median over reference models (each attacked as pseudo-target) of its own FPR threshold."""
    return {f: float(np.median([threshold_at(pseudo_scores[r], keep[r], f) for r in range(len(keep))]))
            for f in FPR_GRID}


def chunked_rmia(target_idx, sig, pop, mem, k, a):
    return np.concatenate([pm.run_rmia(target_idx, sig[i:i + CHUNK], pop, mem[i:i + CHUNK], k, a)
                           for i in range(0, len(sig), CHUNK)])


def eval_attack(run_id, name, scores, member, thr):
    f, t, _ = roc_curve(member, scores)
    flag = scores >= thr[0.01]
    return AttackRow(run_id=run_id, attack=name, auc=roc_auc_score(member, scores),
                     tpr_at_0p1=t[np.where(f <= 0.001)[0][-1]], tpr_at_1=t[np.where(f <= 0.01)[0][-1]],
                     shadow_tpr_at_1=flag[member].mean(), shadow_fpr_at_1=flag[~member].mean())


def audit_cell(cfg, d, size, defense, con):
    split = make_split(len(d.y), cfg, size)
    dev = torch.device(cfg.device)
    R, k = cfg.num_references, cfg.num_references // 2 - 1  # each record has exactly k OUT refs in both games
    # weights_only=False: these .pt files are this pipeline's own local cache (gitignored, never
    # downloaded); MIAShield pickles its signature map. Never point this at untrusted checkpoints.
    load = lambda role: torch.load(model_path(cfg, size, defense, role), weights_only=False).to(dev)
    meta = lambda role: json.loads(model_path(cfg, size, defense, role).with_suffix(".json").read_text())

    # Reference-model signals on the shadow pool, the audited records and the population.
    sets = {"H": split.shadow, "T": split.audit, "Z": split.population}
    ref = {s: [] for s in sets}
    for r in range(R):
        m = load(f"ref{r}")
        for s, idx in sets.items():
            ref[s].append(signals(m, d.x[idx], d.y[idx], dev)[:2])
    p = {s: np.column_stack([v[0] for v in ref[s]]) for s in sets}      # true-class prob [n, R]
    loss = {s: np.column_stack([v[1] for v in ref[s]]) for s in sets}

    # RMIA offline_a: Privacy Meter's tuner on references only (pseudo-target = ref0, attacking its pair).
    sub = np.random.default_rng(0).permutation(len(split.shadow))[:4000]
    offline_a = float(pm.tune_offline_a(0, p["H"][sub], p["Z"], split.ref_keep.T[sub], log)[0])

    # Pseudo-target games on the shadow pool -> thresholds.
    rmia_pseudo = np.stack([chunked_rmia(r, p["H"], p["Z"], split.ref_keep.T, k, offline_a) for r in range(R)])
    loss_pseudo = np.stack([pm.run_loss(loss["H"][:, r]) for r in range(R)])
    thr = {"rmia": shadow_thresholds(rmia_pseudo, split.ref_keep),
           "loss": shadow_thresholds(loss_pseudo, split.ref_keep)}

    # Target game: k references (one per pair 1..R/2-1) are OUT for every audited record, matching the
    # pseudo-target game. Column 1 duplicates the target so run_rmia's "paired model" slot is inert.
    sel = [c for q in range(1, R // 2) for c in (2 * q, 2 * q + 1)]
    member = np.r_[np.ones(size, bool), np.zeros(size, bool)]
    ref_sec = sum(meta(f"ref{r}")["train_sec"] for r in range(R))
    runs, attacks, thr_rows, score_frames, run_ids = [], [], [], [], []
    for seed in cfg.target_seeds:
        role = f"target_seed{seed}"
        m, info = load(role), meta(role)
        run_id = f"{cfg.dataset}|{defense}|S{size}|seed{seed}"
        pt_T, loss_T, correct = signals(m, d.x[split.audit], d.y[split.audit], dev)
        pt_Z = signals(m, d.x[split.population], d.y[split.population], dev)[0]
        sig = np.column_stack([pt_T, pt_T, p["T"][:, sel]])
        mem = np.zeros(sig.shape, bool)
        mem[:, 3::2] = True  # mark the second model of each pair as "IN" so exactly k OUT refs are used
        s_rmia = chunked_rmia(0, sig, np.column_stack([pt_Z, pt_Z, p["Z"][:, sel]]), mem, k, offline_a)
        s_loss = pm.run_loss(loss_T)
        gn = grad_norms(m, d.x[split.audit], d.y[split.audit], dev)

        eps = cfg.defenses[defense].get("epsilon")
        runs.append(RunRow(
            run_id=run_id, dataset=cfg.dataset, defense=defense, epsilon=eps, train_size=size, seed=seed,
            test_acc=correct[~member].mean(), train_acc=correct[member].mean(),
            train_loss=loss_T[member].mean(), test_loss=loss_T[~member].mean(),
            loss_ratio=loss_T[~member].mean() / loss_T[member].mean(), train_sec=info["train_sec"],
            ref_train_sec=ref_sec, epochs_run=info["epochs_run"], eps_spent=info.get("eps_spent"),
            delta=info.get("delta"), offline_a=offline_a, num_refs_used=k))
        attacks += [eval_attack(run_id, "loss", s_loss, member, thr["loss"]),
                    eval_attack(run_id, "rmia", s_rmia, member, thr["rmia"])]
        thr_rows += [(run_id, a, f, t) for a in thr for f, t in thr[a].items()]
        score_frames.append(pl.DataFrame({
            "run_id": run_id, "record_id": split.audit.astype(np.int64), "member": member, "loss": loss_T,
            "p_true": pt_T, "grad_norm": gn, "score_loss": s_loss, "score_rmia": s_rmia}))
        run_ids.append(run_id)
    replace_cell(con, run_ids, runs, attacks,
                 pl.DataFrame(thr_rows, schema=["run_id", "attack", "fpr", "threshold"], orient="row"),
                 pl.concat(score_frames))
    return attacks


def audit_all(cfg):
    d = LOADERS[cfg.dataset]()
    con = connect()
    groups = d.groups.with_columns(pl.lit(cfg.dataset).alias("dataset"), pl.col("record_id").cast(pl.Int64))
    replace_records(con, cfg.dataset, groups)
    for size in cfg.sizes:
        for defense in cfg.defenses:
            for a in audit_cell(cfg, d, size, defense, con):
                print(f"{a.run_id:32s} {a.attack:4s} AUC {a.auc:.3f}  TPR@0.1% {a.tpr_at_0p1:.4f}  "
                      f"TPR@1% {a.tpr_at_1:.4f}  shadow-thr TPR/FPR@1% {a.shadow_tpr_at_1:.4f}/{a.shadow_fpr_at_1:.4f}",
                      flush=True)
    con.close()


if __name__ == "__main__":
    audit_all(load_config(sys.argv[1]))
