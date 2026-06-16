# =====================================================================
# Datathon 2026 - AutoGluon multimodal (Colab L4 icin)
# 82-83 kumesi hipotez testi: alan standart guclu otomatik cozume mi yigildi?
# Kullanim: T4/L4 GPU runtime -> train.csv + test_x.csv yukle ->
#   ILK once ayri bir hucrede:  !pip install -q autogluon.tabular[all]
#   sonra bu dosyanin tamamini ikinci hucreye yapistir, calistir (~2-2.5 saat)
# Cikti: ag_oof.npy, ag_test.npy (indir -> projede oof/ klasorune koy)
# =====================================================================
import os
import numpy as np
import pandas as pd

candidates = [".", "datathon-2026", "/kaggle/input/datathon-2026", "/content"]
base = next(p for p in candidates if os.path.exists(os.path.join(p, "train.csv")))
tr = pd.read_csv(os.path.join(base, "train.csv"), encoding="utf-8-sig")
te = pd.read_csv(os.path.join(base, "test_x.csv"), encoding="utf-8-sig")
print("veri:", tr.shape, te.shape)

from autogluon.tabular import TabularPredictor

train_data = tr.drop(columns=["student_id"])
test_data = te.drop(columns=["student_id"])

predictor = TabularPredictor(
    label="career_success_score",
    eval_metric="mean_squared_error",
    problem_type="regression",
).fit(
    train_data,
    presets="best_quality",
    time_limit=7200,          # 2 saat
    num_bag_folds=5,
    num_bag_sets=1,
    dynamic_stacking=False,
)

print(predictor.leaderboard(silent=True).head(15).to_string())

# OOF (bagging sayesinde leak-free) + test tahminleri
oof = predictor.predict_oof().to_numpy() if hasattr(predictor, "predict_oof") \
    else predictor.get_oof_pred().to_numpy()
test_pred = predictor.predict(test_data).to_numpy()

oof = np.clip(oof, 0, 100)
test_pred = np.clip(test_pred, 0, 100)
np.save("ag_oof.npy", oof)
np.save("ag_test.npy", test_pred)

y = tr["career_success_score"].to_numpy()
from sklearn.metrics import mean_squared_error
print("\nAutoGluon OOF MSE (esit agirlik):", round(mean_squared_error(y, oof), 3))
# yil-agirlikli (LB-esdegeri) metrik
te_p = te["application_year"].value_counts(normalize=True)
tr_p = tr["application_year"].value_counts(normalize=True)
sw = tr["application_year"].map(te_p / tr_p).to_numpy(); sw = sw / sw.mean()
print("AutoGluon LB-tahmini (yil-agirlikli):", round(mean_squared_error(y, oof, sample_weight=sw), 3))
print("kaydedildi: ag_oof.npy, ag_test.npy -> indir, oof/ klasorune koy")
