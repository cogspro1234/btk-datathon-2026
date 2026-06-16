# =====================================================================
# Datathon 2026 - tabpfn_llm: TabPFN (tabular + e5_svd64 + LLM-judge feat)
# + TabPFN-v3 denemesi (varsa). Yeni base'ler stack5/blend icin.
# Kurulum (Colab L4): !pip install -q tabpfn + drive mount + train/test yukle
#   os.environ["TABPFN_TOKEN"]="..." + bu dosya
# e5_train/test.npy ve llm_train/test.npy Drive'dan (datathon_night/).
# Cikti: tabpfn_llm_*, (varsa) tabpfn_v3_*  -> Drive.
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
for c in CATS:
    cc = pd.Categorical(Rtr[c]).categories
    Rtr[c] = pd.Categorical(Rtr[c], categories=cc).codes
    Rte[c] = pd.Categorical(Rte[c], categories=cc).codes

E_tr = np.load(f"{OUTDIR}/e5_train.npy"); E_te = np.load(f"{OUTDIR}/e5_test.npy")
svd = TruncatedSVD(n_components=64, random_state=42)
Ztr = svd.fit_transform(E_tr); Zte = svd.transform(E_te)
L_tr = np.load(f"{OUTDIR}/llm_train.npy"); L_te = np.load(f"{OUTDIR}/llm_test.npy")

Xtr = np.nan_to_num(np.hstack([Rtr.to_numpy(np.float32), Ztr.astype(np.float32),
                               L_tr.astype(np.float32)]), nan=-1.0)
Xte = np.nan_to_num(np.hstack([Rte.to_numpy(np.float32), Zte.astype(np.float32),
                               L_te.astype(np.float32)]), nan=-1.0)
print("tabpfn_llm girdisi:", Xtr.shape)

from tabpfn import TabPFNRegressor


def run(tag, **kw):
    bo = np.zeros(len(y)); bt = np.zeros(len(Xte))
    for s in [7, 21, 101]:
        kf = KFold(5, shuffle=True, random_state=s)
        oof = np.zeros(len(y)); tp = np.zeros(len(Xte))
        for itr, iva in kf.split(Xtr):
            m = TabPFNRegressor(device="cuda", n_estimators=8,
                                ignore_pretraining_limits=True, **kw)
            m.fit(Xtr[itr], y[itr])
            oof[iva] = m.predict(Xtr[iva]); tp += m.predict(Xte) / 5
        bo += np.clip(oof, 0, 100) / 3; bt += np.clip(tp, 0, 100) / 3
    bo = np.clip(bo, 0, 100); bt = np.clip(bt, 0, 100)
    save(f"{tag}_oof", bo); save(f"{tag}_test", bt)
    print(f"{tag}: lb-tahmini {mean_squared_error(y, bo, sample_weight=sw):.3f}", flush=True)


run("tabpfn_llm")  # mevcut (v2.5) default

# --- TabPFN-v3 denemesi (varsa) ---
try:
    import tabpfn
    print("tabpfn surum:", getattr(tabpfn, "__version__", "?"))
    # v3 default model adi paket surumune gore degisebilir; varsa model_path/version
    run("tabpfn_v3_llm", model_path="auto")  # bazı surumler bunu kabul eder
except Exception as e:
    print("TabPFN-v3 denemesi atlandi:", str(e)[:200])
print("bitti")
