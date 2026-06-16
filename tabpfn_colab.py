# =====================================================================
# Datathon 2026 - TabPFN 2.5 denemesi (Colab L4 icin)
# Kullanim: L4 GPU runtime -> train.csv + test_x.csv yukle ->
#   ILK hucrede:  !pip install -q tabpfn
#   sonra bu dosyayi ikinci hucreye yapistir, calistir (~10-20 dk)
# Cikti: tabpfn_oof.npy, tabpfn_test.npy (indir -> oof/ klasorune)
# =====================================================================
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack

candidates = [".", "datathon-2026", "/kaggle/input/datathon-2026", "/content"]
base = next(p for p in candidates if os.path.exists(os.path.join(p, "train.csv")))
tr = pd.read_csv(os.path.join(base, "train.csv"), encoding="utf-8-sig")
te = pd.read_csv(os.path.join(base, "test_x.csv"), encoding="utf-8-sig")
y = tr["career_success_score"].to_numpy()
print("veri:", tr.shape, te.shape)

# --- feature'lar: tabular + kategorik kodlama + TF-IDF/SVD64 metin ---
CATS = ["department", "university_tier", "target_role", "hobby",
        "preferred_social_media_platform"]
Xtr = tr.drop(columns=["student_id", "career_success_score", "mentor_feedback_text"]).copy()
Xte = te.drop(columns=["student_id", "mentor_feedback_text"]).copy()
for c in CATS:
    cats = pd.Categorical(Xtr[c]).categories
    Xtr[c] = pd.Categorical(Xtr[c], categories=cats).codes
    Xte[c] = pd.Categorical(Xte[c], categories=cats).codes

word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                       max_features=50000, sublinear_tf=True)
W = hstack([word.fit_transform(tr["mentor_feedback_text"]),
            char.fit_transform(tr["mentor_feedback_text"])]).tocsr()
Wt = hstack([word.transform(te["mentor_feedback_text"]),
             char.transform(te["mentor_feedback_text"])]).tocsr()
svd = TruncatedSVD(n_components=64, random_state=42)
Ztr = svd.fit_transform(W)
Zte = svd.transform(Wt)
X = np.hstack([Xtr.to_numpy(dtype=np.float32), Ztr.astype(np.float32)])
X_te = np.hstack([Xte.to_numpy(dtype=np.float32), Zte.astype(np.float32)])
X = np.nan_to_num(X, nan=-1.0)
X_te = np.nan_to_num(X_te, nan=-1.0)
print("feature matrisi:", X.shape)

from tabpfn import TabPFNRegressor

kf = KFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(y))
test_pred = np.zeros(len(X_te))
for fold, (itr, iva) in enumerate(kf.split(X)):
    m = TabPFNRegressor(device="cuda", ignore_pretraining_limits=True)
    m.fit(X[itr], y[itr])
    oof[iva] = m.predict(X[iva])
    test_pred += m.predict(X_te) / 5
    print(f"fold {fold}: val mse={mean_squared_error(y[iva], np.clip(oof[iva],0,100)):.3f}",
          flush=True)

oof = np.clip(oof, 0, 100)
test_pred = np.clip(test_pred, 0, 100)
np.save("tabpfn_oof.npy", oof)
np.save("tabpfn_test.npy", test_pred)
print("\nOOF MSE (esit):", round(mean_squared_error(y, oof), 3))
te_p = te["application_year"].value_counts(normalize=True)
tr_p = tr["application_year"].value_counts(normalize=True)
sw = tr["application_year"].map(te_p / tr_p).to_numpy(); sw = sw / sw.mean()
print("LB-tahmini (yil-agirlikli):", round(mean_squared_error(y, oof, sample_weight=sw), 3))
print("kaydedildi: tabpfn_oof.npy, tabpfn_test.npy")
