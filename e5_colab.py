# =====================================================================
# Datathon 2026 - multilingual-e5-large embedding denemesi (Colab L4)
# Farkli/guclu text gorusu. Cikti: e5 embedding (ham + SVD) + iki base model:
#   e5lgb (tabular+e5svd LGBM), e5tab (tabular+e5svd TabPFN)
# Kurulum: drive mount + !pip install -q sentence-transformers tabpfn
#   train/test yukle + os.environ["TABPFN_TOKEN"]="..." + bu dosya
# Ciktilar HEM lokale HEM Drive'a (datathon_night/).
# =====================================================================
import os, shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD

OUTDIR = "/content/drive/MyDrive/datathon_night"
os.makedirs(OUTDIR, exist_ok=True)
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

# ---- e5-large embedding ----
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("intfloat/multilingual-e5-large", device="cuda")
# e5 "query:" prefix bekler
tr_txt = ["query: " + t for t in tr["mentor_feedback_text"]]
te_txt = ["query: " + t for t in te["mentor_feedback_text"]]
E_tr = model.encode(tr_txt, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
E_te = model.encode(te_txt, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
save("e5_train", E_tr.astype(np.float32))
save("e5_test", E_te.astype(np.float32))
print("e5 embedding:", E_tr.shape)

svd = TruncatedSVD(n_components=64, random_state=42)
Ztr = svd.fit_transform(E_tr); Zte = svd.transform(E_te)

# ---- tabular (fe + kw benzeri minimal) ----
CATS = ["department", "university_tier", "target_role", "hobby",
        "preferred_social_media_platform"]
Rtr = tr.drop(columns=["student_id", "career_success_score", "mentor_feedback_text"]).copy()
Rte = te.drop(columns=["student_id", "mentor_feedback_text"]).copy()
for c in CATS:
    cc = pd.Categorical(Rtr[c]).categories
    Rtr[c] = pd.Categorical(Rtr[c], categories=cc).codes
    Rte[c] = pd.Categorical(Rte[c], categories=cc).codes
Xtr = np.nan_to_num(np.hstack([Rtr.to_numpy(np.float32), Ztr.astype(np.float32)]), nan=-1.0)
Xte = np.nan_to_num(np.hstack([Rte.to_numpy(np.float32), Zte.astype(np.float32)]), nan=-1.0)

# ---- e5lgb ----
import lightgbm as lgb
kf = KFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(y)); tp = np.zeros(len(Xte))
for itr, iva in kf.split(Xtr):
    m = lgb.LGBMRegressor(n_estimators=20000, learning_rate=0.01, num_leaves=31,
                          max_depth=6, min_child_samples=50, colsample_bytree=0.8,
                          subsample=0.8, subsample_freq=1, reg_lambda=1.0, verbose=-1)
    m.fit(Xtr[itr], y[itr], eval_set=[(Xtr[iva], y[iva])],
          callbacks=[lgb.early_stopping(400, verbose=False), lgb.log_evaluation(0)])
    oof[iva] = m.predict(Xtr[iva]); tp += m.predict(Xte) / 5
oof = np.clip(oof, 0, 100)
save("e5lgb_oof", oof); save("e5lgb_test", np.clip(tp, 0, 100))
print("e5lgb lb-tahmini:", round(mean_squared_error(y, oof, sample_weight=sw), 3))

# ---- e5tab (TabPFN) ----
from tabpfn import TabPFNRegressor
oof = np.zeros(len(y)); tp = np.zeros(len(Xte))
for itr, iva in kf.split(Xtr):
    m = TabPFNRegressor(device="cuda", n_estimators=8, ignore_pretraining_limits=True)
    m.fit(Xtr[itr], y[itr])
    oof[iva] = m.predict(Xtr[iva]); tp += m.predict(Xte) / 5
oof = np.clip(oof, 0, 100)
save("e5tab_oof", oof); save("e5tab_test", np.clip(tp, 0, 100))
print("e5tab lb-tahmini:", round(mean_squared_error(y, oof, sample_weight=sw), 3))
print("\nbitti - e5lgb/e5tab + e5_train/test embedding Drive'da")
