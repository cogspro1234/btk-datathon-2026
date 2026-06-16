# Hipotez testi: hedef ozelliklerin lineer/formulsel bir fonksiyonu mu?
# Ridge varyantlarini yil-agirlikli OOF MSE (LB-tahmini) ile karsilastirir.
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

ROOT = Path(__file__).parent
tr = pd.read_csv(ROOT / "datathon-2026" / "train.csv", encoding="utf-8-sig")
te = pd.read_csv(ROOT / "datathon-2026" / "test_x.csv", encoding="utf-8-sig")
y = tr["career_success_score"].to_numpy()

te_p = te["application_year"].value_counts(normalize=True)
tr_p = tr["application_year"].value_counts(normalize=True)
sw = tr["application_year"].map(te_p / tr_p).fillna(0).to_numpy()
sw = sw / sw.mean()

CATS = ["department", "university_tier", "target_role", "hobby",
        "preferred_social_media_platform"]
num_cols = [c for c in tr.columns
            if pd.api.types.is_numeric_dtype(tr[c])
            and c not in ("career_success_score",)]

# sayisal: median impute + eksiklik bayragi + standardize
Xn = tr[num_cols].copy()
miss = Xn.isna().astype(int)
miss.columns = [c + "_miss" for c in miss.columns]
Xn = Xn.fillna(Xn.median())
Xn = pd.concat([Xn, miss.loc[:, miss.sum() > 0]], axis=1)
Xn = csr_matrix(StandardScaler().fit_transform(Xn))

# kategorik: one-hot
Xc = csr_matrix(pd.get_dummies(tr[CATS]).to_numpy(dtype=float))

# metin: tam TF-IDF (SVD'siz)
word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                       max_features=50000, sublinear_tf=True)
Xw = word.fit_transform(tr["mentor_feedback_text"])
Xch = char.fit_transform(tr["mentor_feedback_text"])

variants = {
    "ridge_tabular": hstack([Xn, Xc]).tocsr(),
    "ridge_text_only": hstack([Xw, Xch]).tocsr(),
    "ridge_tab+text": hstack([Xn, Xc, Xw, Xch]).tocsr(),
}

kf = KFold(5, shuffle=True, random_state=42)
for name, X in variants.items():
    for alpha in [1.0, 10.0]:
        oof = np.zeros(len(y))
        for itr, iva in kf.split(X):
            m = Ridge(alpha=alpha)
            m.fit(X[itr], y[itr])
            oof[iva] = m.predict(X[iva])
        oof = np.clip(oof, 0, 100)
        print(f"{name:18s} a={alpha:5.1f}  esit={mean_squared_error(y, oof):8.3f}  "
              f"lb-tahmini={mean_squared_error(y, oof, sample_weight=sw):8.3f}")
