# Train/test kaymasi teshisi: kolon dagilimlari + adversarial validation
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent
tr = pd.read_csv(ROOT / "datathon-2026" / "train.csv", encoding="utf-8-sig")
te = pd.read_csv(ROOT / "datathon-2026" / "test_x.csv", encoding="utf-8-sig")

num_cols = [c for c in te.columns
            if pd.api.types.is_numeric_dtype(te[c]) and c != "student_id"]
print("=== sayisal kolonlar: train mean/std vs test mean/std (fark buyukse drift) ===")
rows = []
for c in num_cols:
    m1, s1 = tr[c].mean(), tr[c].std()
    m2, s2 = te[c].mean(), te[c].std()
    z = abs(m1 - m2) / (s1 / np.sqrt(len(te)) + 1e-9)
    rows.append((c, m1, m2, s1, s2, z))
rows.sort(key=lambda r: -r[5])
for c, m1, m2, s1, s2, z in rows[:15]:
    print(f"{c:32s} tr={m1:8.2f}/{s1:6.2f}  te={m2:8.2f}/{s2:6.2f}  z={z:8.1f}")

print("\n=== kategorik dagilimlar ===")
for c in ["application_year", "graduation_year", "department", "university_tier", "target_role"]:
    p1 = tr[c].value_counts(normalize=True).sort_index()
    p2 = te[c].value_counts(normalize=True).sort_index()
    d = pd.concat([p1, p2], axis=1, keys=["train", "test"]).fillna(0)
    d["fark"] = (d["train"] - d["test"]).abs()
    if d["fark"].max() > 0.01:
        print(f"\n{c}:")
        print(d.round(3).to_string())

print("\n=== adversarial validation (train=0/test=1 ayirt edilebiliyor mu) ===")
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

X = pd.concat([tr[num_cols], te[num_cols]], axis=0).reset_index(drop=True)
yav = np.r_[np.zeros(len(tr)), np.ones(len(te))]
clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, verbose=-1)
auc = cross_val_score(clf, X, yav, cv=3, scoring="roc_auc").mean()
print(f"adversarial AUC (sayisal kolonlar): {auc:.4f}  (0.5 = drift yok)")

clf.fit(X, yav)
imp = pd.Series(clf.feature_importances_, index=num_cols).sort_values(ascending=False)
print("\nen ayirt edici kolonlar:")
print(imp.head(10).to_string())

print("\n=== hedef ve tahmin dagilimi ===")
sub = pd.read_csv(ROOT / "yuklenecek" / "sub3_blend8_cv76.72.csv")
print("train hedef:", tr["career_success_score"].describe().round(2).to_dict())
print("blend tahmin:", sub["career_success_score"].describe().round(2).to_dict())
