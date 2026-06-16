# =====================================================================
# Datathon 2026 - PySR symbolic regression: f(features) formul recovery
# Kurulum (Colab): !pip install -q pysr   (ilk import Julia'yi kurar ~3-5 dk)
# train.csv yukle, bu hucreyi calistir (~25-40 dk).
# Cikti: kesfedilen formul + clean-year ve all-data R2. Eger yuksek R2'li
# kompakt formul bulursa -> leader edge olabilir; pysr_oof/test.npy (Drive).
# =====================================================================
import os, shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

OUTDIR = "/content/drive/MyDrive/datathon_night"
os.makedirs(OUTDIR, exist_ok=True)

cands = [".", "datathon-2026", "/kaggle/input/datathon-2026", "/content"]
base = next(p for p in cands if os.path.exists(os.path.join(p, "train.csv")))
tr = pd.read_csv(os.path.join(base, "train.csv"), encoding="utf-8-sig")
te = pd.read_csv(os.path.join(base, "test_x.csv"), encoding="utf-8-sig")
y = tr["career_success_score"].to_numpy()
yil = tr["application_year"].to_numpy()
te_p = te["application_year"].value_counts(normalize=True)
tr_p = tr["application_year"].value_counts(normalize=True)
sw = tr["application_year"].map(te_p / tr_p).to_numpy(); sw = sw / sw.mean()

# en bilgilendirici sayisal feature'lar (skill + birkac guclu)
FEATS = ["coding_score","problem_solving_score","data_structures_score","sql_score",
         "machine_learning_score","backend_score","frontend_score","cloud_score","devops_score",
         "project_quality_score","technical_interview_score","communication_score",
         "portfolio_score","cgpa","real_client_project_count"]
X = tr[FEATS].fillna(tr[FEATS].median()).to_numpy()
Xte = te[FEATS].fillna(tr[FEATS].median()).to_numpy()

from pysr import PySRRegressor


def make():
    return PySRRegressor(
        niterations=200, populations=24, population_size=40,
        maxsize=28, binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "sqrt_abs(x)=sqrt(abs(x))",
                         "logp(x)=log(abs(x)+1)", "sig(x)=100/(1+exp(-x))"],
        extra_sympy_mappings={"sqrt_abs": lambda x: x, "logp": lambda x: x, "sig": lambda x: x},
        elementwise_loss="loss(p,t) = (p-t)^2",
        model_selection="best", random_state=0, deterministic=False,
        procs=2, progress=False, verbosity=0,
    )


# 1) clean years (2019-21) - dusuk gurultu, formul daha gorunur
cl = yil <= 2021
m_clean = make(); m_clean.fit(X[cl], y[cl])
pred_clean = np.clip(m_clean.predict(X), 0, 100)
r2c = 1 - ((y[cl]-pred_clean[cl])**2).sum()/((y[cl]-y[cl].mean())**2).sum()
print("=== CLEAN-YEAR formul ===")
print(m_clean.get_best())
print(f"clean R2 (egitildigi): {r2c:.3f}  | tum-veri uzerinde uygulaninca lb-tahmini: "
      f"{mean_squared_error(y, pred_clean, sample_weight=sw):.3f}")

# 2) tum veri - gurultu icinden formul
m_all = make(); m_all.fit(X, y)
print("\n=== ALL-DATA formul ===")
print(m_all.get_best())

# OOF olarak degerlendir (5-fold, all-data formul yapisini yeniden fit)
kf = KFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(y)); tp = np.zeros(len(Xte))
for fold, (itr, iva) in enumerate(kf.split(X)):
    m = make(); m.fit(X[itr], y[itr])
    oof[iva] = m.predict(X[iva]); tp += m.predict(Xte)/5
    print(f"fold {fold} ok", flush=True)
oof = np.clip(oof, 0, 100); tp = np.clip(tp, 0, 100)
np.save(f"{OUTDIR}/pysr_oof.npy", oof); np.save(f"{OUTDIR}/pysr_test.npy", tp)
print(f"\nPySR OOF lb-tahmini: {mean_squared_error(y, oof, sample_weight=sw):.3f} (referans tek-LGBM 88.36)")
print("pysr_oof/test.npy Drive'a kaydedildi")
