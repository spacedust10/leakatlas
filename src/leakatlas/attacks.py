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


def _served(model, x, y, dev):
    """Per-(sub)model float64 logits, the oracle's keep-mask and the true-class index for a batch."""
    shield = isinstance(model, MIAShield)
    subs = list(model.subs) if shield else [model]
    excl = model.oracle(x) if shield else torch.full((len(x),), -1)
    with torch.no_grad():
        z = torch.stack([m(x.to(dev)) for m in subs], 1).cpu().double()            # [B, K, C]
    keep = torch.arange(len(subs))[None] != excl[:, None]                           # [B, K]
    return subs, z, keep, y[:, None, None].expand(-1, len(subs), 1)


def signals(model, x, y, dev, bs=4096):
    """True-class prob, loss, log(loss) and correctness, computed in float64 from logits.

    Confident predictions underflow: float32 loss hits 0 and even float64 loss hits 0 once the
    logit margin m = logsumexp_{j!=y} z_j - z_y drops below ~-745 (seen for Adult DP-SGD). Ties at
    0 break rankings, so we also return log(loss), which for tiny losses equals log q with
    q = mean_kept sigmoid(m_k) (= 1 - served p_y), computed with logsigmoid and never underflows.
    MIAShield's served loss is -log(mean of kept p_k); -log1p(-q) avoids a K-dependent rounding floor
    (members keep 4 sub-models, non-members 5).
    """
    model.eval()
    out = {"loss": [], "log_loss": [], "correct": []}
    for i in range(0, len(x), bs):
        _, z, keep, idx = _served(model, x[i:i + bs], y[i:i + bs], dev)
        margin = torch.logsumexp(z.scatter(2, idx, float("-inf")), 2) - z.gather(2, idx).squeeze(2)
        log_k = keep.sum(1).double().log()
        log_q = torch.logsumexp(torch.where(keep, F.logsigmoid(margin), float("-inf")), 1) - log_k
        q = log_q.exp()
        lme = torch.logsumexp(torch.where(keep, -F.softplus(margin), float("-inf")), 1) - log_k
        loss = -torch.where(q < 0.5, torch.log1p(-q), lme)
        out["loss"].append(loss)
        out["log_loss"].append(torch.where(q < 1e-6, log_q, loss.log()))
        out["correct"].append((F.softmax(z, 2) * keep[..., None]).sum(1).argmax(1) == y[i:i + bs])
    loss, log_loss = torch.cat(out["loss"]).numpy(), torch.cat(out["log_loss"]).numpy()
    return np.exp(-loss), loss, log_loss, torch.cat(out["correct"]).numpy()


def grad_norms(model, x, y, dev, bs=512):
    """log L2 norm of each record's loss gradient w.r.t. all parameters at the final weights (Tobaben et al.).

    Computed without underflow: for served P_y = mean_kept p_{k,y},
        grad L = sum_k (p_{k,y} / (K' P_y)) J_k^T (p_k - e_y) = sum_k exp(a_k) J_k^T u_k
    with unit vectors u_k and log-weights a_k in float64. With A = max a_k,
        log||grad L|| = A + log|| grad_theta sum_k exp(a_k - A) u_k . z_k(theta) ||
    where the surrogate's gradient is O(1), so float32 on the device is safe. K = 1 for plain models.
    """
    model = model.to(dev).eval()
    subs = list(model.subs) if isinstance(model, MIAShield) else [model]
    params = [{k: v.detach() for k, v in m.named_parameters()} for m in subs]

    def surrogate(ps, xi, wi):  # wi: [K, C] = exp(a_k - A) * u_k
        return sum((wi[k] * functional_call(m, p, (xi[None],))[0]).sum() for k, (m, p) in enumerate(zip(subs, ps)))

    per_sample = vmap(grad(surrogate), in_dims=(None, 0, 0))
    out = []
    for i in range(0, len(x), bs):
        _, z, keep, idx = _served(model, x[i:i + bs], y[i:i + bs], dev)
        logp = torch.log_softmax(z, 2)
        is_y = torch.zeros_like(z, dtype=torch.bool).scatter(2, idx, True)
        lp_other = logp.masked_fill(is_y, float("-inf"))
        lse_other = torch.logsumexp(lp_other, 2)                                    # log(1 - p_y)
        log_v = 0.5 * torch.logaddexp(torch.logsumexp(2 * lp_other, 2), 2 * lse_other)  # log||p - e_y||
        u = torch.where(is_y, -(lse_other - log_v).exp()[..., None], (lp_other - log_v[..., None]).exp())
        log_py = logp.gather(2, idx).squeeze(2)
        log_k = keep.sum(1).double().log()
        log_P = torch.logsumexp(torch.where(keep, log_py, float("-inf")), 1) - log_k
        a = torch.where(keep, log_py - log_k[:, None] - log_P[:, None] + log_v, float("-inf"))
        A = a.max(1).values
        w = ((a - A[:, None]).exp()[..., None] * u).float()
        g = per_sample(params, x[i:i + bs].to(dev), w.to(dev))
        sq = sum((v.flatten(1) ** 2).sum(1) for gk in g for v in gk.values())
        out.append(A + 0.5 * sq.cpu().double().log())
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
            ref[s].append(signals(m, d.x[idx], d.y[idx], dev))
    p = {s: np.column_stack([v[0] for v in ref[s]]) for s in sets}         # true-class prob [n, R]
    log_loss = {s: np.column_stack([v[2] for v in ref[s]]) for s in sets}  # LOSS ranks by log(loss): no ties

    # RMIA offline_a: Privacy Meter's tuner on references only (pseudo-target = ref0, attacking its pair).
    sub = np.random.default_rng(0).permutation(len(split.shadow))[:4000]
    offline_a = float(pm.tune_offline_a(0, p["H"][sub], p["Z"], split.ref_keep.T[sub], log)[0])

    # Pseudo-target games on the shadow pool -> thresholds.
    rmia_pseudo = np.stack([chunked_rmia(r, p["H"], p["Z"], split.ref_keep.T, k, offline_a) for r in range(R)])
    loss_pseudo = np.stack([pm.run_loss(log_loss["H"][:, r]) for r in range(R)])
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
        pt_T, loss_T, log_loss_T, correct = signals(m, d.x[split.audit], d.y[split.audit], dev)
        pt_Z = signals(m, d.x[split.population], d.y[split.population], dev)[0]
        sig = np.column_stack([pt_T, pt_T, p["T"][:, sel]])
        mem = np.zeros(sig.shape, bool)
        mem[:, 3::2] = True  # mark the second model of each pair as "IN" so exactly k OUT refs are used
        s_rmia = chunked_rmia(0, sig, np.column_stack([pt_Z, pt_Z, p["Z"][:, sel]]), mem, k, offline_a)
        s_loss = pm.run_loss(log_loss_T)
        log_gn = grad_norms(m, d.x[split.audit], d.y[split.audit], dev)

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
            "log_loss": log_loss_T, "p_true": pt_T, "log_grad_norm": log_gn, "score_loss": s_loss,
            "score_rmia": s_rmia}))
        run_ids.append(run_id)
    replace_cell(con, run_ids, runs, attacks,
                 pl.DataFrame(thr_rows, schema=["run_id", "attack", "fpr", "threshold"], orient="row"),
                 pl.concat(score_frames))
    return attacks


def audit_all(cfg, only: str | None = None):
    d = LOADERS[cfg.dataset]()
    con = connect()
    groups = d.groups.with_columns(pl.lit(cfg.dataset).alias("dataset"), pl.col("record_id").cast(pl.Int64))
    replace_records(con, cfg.dataset, groups)
    for size in cfg.sizes:
        for defense in cfg.defenses:
            if only and defense != only:
                continue
            for a in audit_cell(cfg, d, size, defense, con):
                print(f"{a.run_id:32s} {a.attack:4s} AUC {a.auc:.3f}  TPR@0.1% {a.tpr_at_0p1:.4f}  "
                      f"TPR@1% {a.tpr_at_1:.4f}  shadow-thr TPR/FPR@1% {a.shadow_tpr_at_1:.4f}/{a.shadow_fpr_at_1:.4f}",
                      flush=True)
    con.close()


if __name__ == "__main__":
    audit_all(load_config(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else None)
