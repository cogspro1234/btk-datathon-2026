# OOF blend: oof/ klasorundeki deneyleri yukler, MSE'yi minimize eden
# negatif-olmayan agirliklari bulur, blend submission yazar.
# Kullanim: python blend.py fe_text cat_fe_text xgb_fe_text
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).parent
OOF = ROOT / "oof"
SUBS = ROOT / "submissions"

names = sys.argv[1:]
assert len(names) >= 2, "en az 2 deney adi ver"

tr = pd.read_csv(ROOT / "datathon-2026" / "train.csv", encoding="utf-8-sig")
te = pd.read_csv(ROOT / "datathon-2026" / "test_x.csv", encoding="utf-8-sig")
y = tr["career_success_score"].to_numpy()

# LB, test'in yil dagilimiyla skorlaniyor -> ayni agirlikla optimize et
te_p = te["application_year"].value_counts(normalize=True)
tr_p = tr["application_year"].value_counts(normalize=True)
sw = tr["application_year"].map(te_p / tr_p).fillna(0).to_numpy()
sw = sw / sw.mean()

oofs = np.column_stack([np.load(OOF / f"{n}_oof.npy") for n in names])
tests = np.column_stack([np.load(OOF / f"{n}_test.npy") for n in names])

for i, n in enumerate(names):
    print(f"{n:24s} lb-tahmini = "
          f"{mean_squared_error(y, oofs[:, i], sample_weight=sw):.4f}")


def loss(w):
    w = np.abs(w) / np.abs(w).sum()
    return mean_squared_error(y, oofs @ w, sample_weight=sw)


best = min((minimize(loss, w0, method="Nelder-Mead")
            for w0 in [np.ones(len(names)), *np.eye(len(names))]),
           key=lambda r: r.fun)
w = np.abs(best.x) / np.abs(best.x).sum()
print("\nagirliklar:", dict(zip(names, np.round(w, 4))))
blend_lb = mean_squared_error(y, np.clip(oofs @ w, 0, 100), sample_weight=sw)
print(f"blend lb-tahmini = {blend_lb:.4f}")

pred = np.clip(tests @ w, 0, 100)
sub = pd.DataFrame({"student_id": te["student_id"], "career_success_score": pred})
path = SUBS / f"sub_blend{len(names)}_lb{blend_lb:.3f}.csv"
sub.to_csv(path, index=False)
print("yazildi:", path)
