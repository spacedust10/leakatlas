"""Check: log_grad_norm (underflow-free surrogate) equals log||grad|| from direct float64 autograd."""
import copy

import numpy as np
import torch
import torch.nn.functional as F

import leakatlas.defenses as D
from leakatlas.attacks import grad_norms, signals
from leakatlas.data import LOADERS


def direct(model, x, y):
    m = copy.deepcopy(model).cpu().double()
    out = []
    for xi, yi in zip(x.double(), y):
        m.zero_grad()
        if isinstance(m, D.MIAShield):
            lp = m(xi[None], m.oracle(xi[None].float()))
        else:
            lp = F.log_softmax(m(xi[None]), -1)
        F.nll_loss(lp, yi[None]).backward()
        out.append(0.5 * np.log(sum(float((p.grad ** 2).sum()) for p in m.parameters())))
    return np.array(out)


def check(path, size, defense, n=40):
    cfg = D.load_config(path)
    d = LOADERS[cfg.dataset]()
    sp = D.make_split(len(d.y), cfg, size)
    m = torch.load(D.model_path(cfg, size, defense, "target_seed0"), weights_only=False)
    idx = np.r_[sp.members[:n // 2], sp.non_members[:n // 2]]
    x, y = d.x[idx], d.y[idx]
    ours = grad_norms(m, x, y, torch.device(cfg.device))
    ref = direct(m, x, y)
    _, loss, log_loss, _ = signals(m, x, y, torch.device(cfg.device))
    # Direct autograd forms p_y - 1, which rounds to 0 once 1 - p_y < ~1e-16, so it is only a valid
    # reference for log_loss > -30; below that it drops the true-class term (that is the bug we avoid).
    ok = np.isfinite(ref) & (log_loss > -30)
    assert np.isfinite(ours).all(), "log_grad_norm must be finite everywhere"
    assert np.allclose(ours[ok], ref[ok], atol=1e-3), np.abs(ours[ok] - ref[ok]).max()
    pos = loss > 0
    assert np.allclose(log_loss[pos], np.log(loss[pos]), atol=1e-6) and np.isfinite(log_loss).all()
    print(f"{cfg.dataset} {defense}: {ok.sum()}/{n} comparable, max |diff| {np.abs(ours[ok] - ref[ok]).max():.2e}, "
          f"loss==0 for {int((~pos).sum())} but log_loss finite")


if __name__ == "__main__":
    check("configs/adult.yaml", 10000, "dp_eps1")
    check("configs/adult.yaml", 10000, "baseline")
    check("configs/fmnist.yaml", 15000, "miashield", n=20)
