# =====================================================================
# Datathon 2026 - TabPFN config denemesi (kenarda kalan son fikir)
# tabpfn_llm girdisi + (1) native categorical_features_indices
# (2) n_estimators=24 (daha cok ic-ensemble) (3) preprocess varsa 'all'
# Kurulum (Colab L4): !pip install -q tabpfn + drive mount + train/test
#   os.environ["TABPFN_TOKEN"]="..." + bu dosya
# Cikti: tabpfn_cfg_*  -> Drive. Mevcut tabpfn_llm 86.63 ile kiyas.
# =====================================================================
import os, shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD

OUTDIR = "/content/drive/MyDrive/datathon_night"
def save(name, arr):
    np.save(f"{name}.npy", arr)
    try: shutil.copy(f"{name}.npy", f"{OUTDIR}/{name}.npy")
    except Exception: pass

cands = [".", "datathon-2026", "/kaggle/input/datathon-2026", "/content"]
base = next(p for p in cands if os.path.exists(os.path.join(p, "train.csv")))
tr = pd.read_csv(os.path.join(base, "train.csv"), encoding="utf-8-sig")
te = pd.read_csv(os.path.join(base, "test_x.csv"), encoding="utf-8-sig")
y = tr["career_success_score"].to_numpy()
te_p = te["application_year"].value_counts(normalize=True)
tr_p = tr["application_year"].value_counts(normalize=True)
sw = tr["application_year"].map(te_p / tr_p).to_numpy(); sw = sw / sw.mean()

CATS = ["department", "university_tier", "target_role", "hobby",
        "preferred_social_media_platform"]
Rtr = tr.drop(columns=["student_id", "career_success_score", "mentor_feedback_text"]).copy()
Rte = te.drop(columns=["student_id", "mentor_feedback_text"]).copy()
cat_idx = [Rtr.columns.get_loc(c) for c in CATS]   # kategorik kolon indeksleri
for c in CATS:
    cc = pd.Categorical(Rtr[c]).categories
    Rtr[c] = pd.Categorical(Rtr[c], categories=cc).codes
    Rte[c] = pd.Categorical(Rte[c], categories=cc).codes
print("kategorik indeksler:", cat_idx)

E_tr = np.load(f"{OUTDIR}/e5_train.npy"); E_te = np.load(f"{OUTDIR}/e5_test.npy")
svd = TruncatedSVD(n_components=64, random_state=42)
Ztr = svd.fit_transform(E_tr); Zte = svd.transform(E_te)
L_tr = np.load(f"{OUTDIR}/llm_train.npy"); L_te = np.load(f"{OUTDIR}/llm_test.npy")
Xtr = np.nan_to_num(np.hstack([Rtr.to_numpy(np.float32), Ztr.astype(np.float32), L_tr.astype(np.float32)]), nan=-1.0)
Xte = np.nan_to_num(np.hstack([Rte.to_numpy(np.float32), Zte.astype(np.float32), L_te.astype(np.float32)]), nan=-1.0)
# kategorik indeksler Rtr icinde; e5/llm sonra eklendigi icin ayni indeksler gecerli
print("girdi:", Xtr.shape)

from tabpfn import TabPFNRegressor
import inspect
sig = set(inspect.signature(TabPFNRegressor.__init__).parameters)
print("TabPFNRegressor parametreleri:", sorted(sig))

def make(seed):
    kw = dict(device="cuda", n_estimators=8, random_state=seed,
              ignore_pretraining_limits=True)
    if "categorical_features_indices" in sig:
        kw["categorical_features_indices"] = cat_idx
    return TabPFNRegressor(**kw)

bo = np.zeros(len(y)); bt = np.zeros(len(Xte))
for s in [7, 21, 101]:
    kf = KFold(5, shuffle=True, random_state=s)
    oof = np.zeros(len(y)); tp = np.zeros(len(Xte))
    for itr, iva in kf.split(Xtr):
        m = make(s); m.fit(Xtr[itr], y[itr])
        oof[iva] = m.predict(Xtr[iva]); tp += m.predict(Xte) / 5
    bo += np.clip(oof, 0, 100) / 3; bt += np.clip(tp, 0, 100) / 3
bo = np.clip(bo, 0, 100); bt = np.clip(bt, 0, 100)
save("tabpfn_cfg_oof", bo); save("tabpfn_cfg_test", bt)
print("tabpfn_cfg (native cat, n_est8):", round(mean_squared_error(y, bo, sample_weight=sw), 3),
      "| referans tabpfn_llm 86.625")
