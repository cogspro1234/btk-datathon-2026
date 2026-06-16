# =====================================================================
# Datathon 2026 - TabPFN varyant turu (Colab L4)
# Kullanim: ayni oturumda (tabpfn kurulu + TABPFN_TOKEN ayarli + csv'ler yuklu)
#   bu dosyayi yeni hucreye yapistir, calistir (~45-60 dk / 3 varyant)
# Cikti: tabpfn_tabonly_*, tabpfn_rich_*, tabpfn_ne16_* npy'leri -> oof/ klasorune
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

CATS = ["department", "university_tier", "target_role", "hobby",
        "preferred_social_media_platform"]
MISS_COLS = ["english_exam_score", "internship_duration_months", "portfolio_score",
             "github_avg_stars", "open_source_contribution_count",
             "linkedin_profile_score", "hr_interview_score"]

POS = ["dikkat çek", "güçlü", "ileri düzey", "umut verici", "potansiyel",
       "etkili", "başarılı", "yetkin", "uzman", "tutku", "üst düzey"]
NEG = ["geliştirme", "gelişim göster", "çalışması gerek", "ihtiyaç",
       "eksik", "zayıf", "yetersiz", "daha fazla"]


def tabular(df):
    X = df.drop(columns=["student_id", "mentor_feedback_text"], errors="ignore")
    X = X.drop(columns=["career_success_score"], errors="ignore").copy()
    for c in CATS:
        cats = pd.Categorical(tr[c]).categories
        X[c] = pd.Categorical(X[c], categories=cats).codes
    return X


def kw_feats(df):
    txt = df["mentor_feedback_text"].str.lower()
    out = pd.DataFrame(index=df.index)
    out["kw_pos"] = sum(txt.str.count(p) for p in POS)
    out["kw_neg"] = sum(txt.str.count(p) for p in NEG)
    out["kw_bal"] = out["kw_pos"] - out["kw_neg"]
    out["t_len"] = df["mentor_feedback_text"].str.len()
    return out


# SVD64 metin (v1 ile ayni)
word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                       max_features=50000, sublinear_tf=True)
W = hstack([word.fit_transform(tr["mentor_feedback_text"]),
            char.fit_transform(tr["mentor_feedback_text"])]).tocsr()
Wt = hstack([word.transform(te["mentor_feedback_text"]),
             char.transform(te["mentor_feedback_text"])]).tocsr()
svd = TruncatedSVD(n_components=64, random_state=42)
Ztr, Zte = svd.fit_transform(W), svd.transform(Wt)

Ttr, Tte = tabular(tr), tabular(te)
Ktr, Kte = kw_feats(tr), kw_feats(te)
Mtr = tr[MISS_COLS].isna().astype(int).to_numpy()
Mte = te[MISS_COLS].isna().astype(int).to_numpy()


def prep(*parts):
    X = np.hstack([p.to_numpy(dtype=np.float32) if hasattr(p, "to_numpy")
                   else p.astype(np.float32) for p in parts])
    return np.nan_to_num(X, nan=-1.0)


views = {
    "tabpfn_tabonly": (prep(Ttr), prep(Tte), 8),
    "tabpfn_rich": (prep(Ttr, Ktr, Mtr, Ztr), prep(Tte, Kte, Mte, Zte), 8),
    "tabpfn_ne16": (prep(Ttr, Ztr), prep(Tte, Zte), 16),
}

from tabpfn import TabPFNRegressor

te_p = te["application_year"].value_counts(normalize=True)
tr_p = tr["application_year"].value_counts(normalize=True)
sw = tr["application_year"].map(te_p / tr_p).to_numpy(); sw = sw / sw.mean()

kf = KFold(5, shuffle=True, random_state=42)
for name, (X, X_te, ne) in views.items():
    oof = np.zeros(len(y)); test_pred = np.zeros(len(X_te))
    for fold, (itr, iva) in enumerate(kf.split(X)):
        m = TabPFNRegressor(device="cuda", n_estimators=ne,
                            ignore_pretraining_limits=True)
        m.fit(X[itr], y[itr])
        oof[iva] = m.predict(X[iva])
        test_pred += m.predict(X_te) / 5
    oof = np.clip(oof, 0, 100)
    np.save(f"{name}_oof.npy", oof)
    np.save(f"{name}_test.npy", np.clip(test_pred, 0, 100))
    print(f"{name}: esit={mean_squared_error(y, oof):.3f}  "
          f"lb-tahmini={mean_squared_error(y, oof, sample_weight=sw):.3f}", flush=True)
print("bitti - 6 npy dosyasini indir, oof/ klasorune koy")
