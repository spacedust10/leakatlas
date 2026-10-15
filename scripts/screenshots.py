"""Headless capture of every dashboard view (Checkpoint 4). Starts `panel serve`, opens two URL states
(Fashion-MNIST and Adult, each with the most exposed record selected), screenshots each view section.

Usage: uv run python scripts/screenshots.py
"""

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from leakatlas.store import connect  # noqa: E402

OUT = ROOT / "results" / "figures"
PORT = 5017
NAMES = ["record_atlas", "stability_strip", "record_inspector", "log_roc", "defense_tradeoff",
         "proxy_vs_truth", "subgroup_ridgelines"]  # dashboard order (investigation order)
WANT = {"fmnist": [n for n in NAMES if n != "subgroup_ridgelines"],     # ridgelines are Adult-only
        "adult": ["subgroup_ridgelines", "record_inspector"]}


def most_exposed(dataset, size):
    con = connect(read_only=True)
    rid = con.execute("""SELECT s.record_id FROM record_scores s JOIN runs r USING (run_id)
                         JOIN thresholds t ON t.run_id = s.run_id AND t.attack = 'rmia' AND t.fpr = 0.01
                         WHERE r.dataset = ? AND r.train_size = ? AND r.defense = 'baseline' AND s.member
                         GROUP BY s.record_id ORDER BY sum((s.score_rmia >= t.threshold)::INT) DESC,
                         avg(s.score_rmia) DESC LIMIT 1""", [dataset, size]).fetchone()[0]
    con.close()
    return int(rid)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONWARNINGS="ignore")
    server = subprocess.Popen([sys.executable, "-m", "panel", "serve", str(ROOT / "src/leakatlas/viewer.py"),
                               "--port", str(PORT), "--allow-websocket-origin", f"localhost:{PORT}"],
                              env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://localhost:{PORT}/viewer", timeout=2)
                break
            except OSError:
                time.sleep(1)
        states = {
            "fmnist": f"dataset=fmnist&size=15000&defense=baseline&seed=all&fpr=0.01&record={most_exposed('fmnist', 15000)}",
            "adult": f"dataset=adult&defense=baseline&seed=all&fpr=0.01&attribute=race&record={most_exposed('adult', 10000)}",
        }
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=2)
            page.set_default_timeout(180_000)  # cold Adult load: 6 defenses x subgroup bootstraps
            for tag, query in states.items():
                page.goto(f"http://localhost:{PORT}/viewer?{query}", wait_until="networkidle", timeout=180_000)
                page.wait_for_timeout(12_000)  # let Bokeh + datashader finish drawing
                for i, name in enumerate(NAMES):
                    if name not in WANT[tag]:
                        continue
                    v = page.locator(f".la-anchor-{i + 1}").first  # nth(.la-view) is unordered across shadow roots
                    v.scroll_into_view_if_needed()
                    page.wait_for_timeout(1500)
                    path = OUT / f"viewer_{i + 1:02d}_{name}{'_adult' if tag == 'adult' else ''}.png"
                    v.screenshot(path=str(path))
                    print("saved", path.relative_to(ROOT))
            page.goto(f"http://localhost:{PORT}/viewer?{states['fmnist']}", wait_until="networkidle", timeout=180_000)
            page.wait_for_timeout(12_000)
            page.screenshot(path=str(OUT / "viewer_full_page.png"), full_page=True)
            browser.close()
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
