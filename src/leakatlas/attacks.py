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
    lp = log_probs(model, x, dev)
    loss = F.nll_loss(lp, y, reduction="none").numpy()
    return np.exp(-loss), loss, (lp.argmax(1) == y).numpy()


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
