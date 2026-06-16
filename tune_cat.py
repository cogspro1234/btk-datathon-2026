# CatBoost hiperparametre taramasi (GPU): sabit seed-42 KFold, paired kiyas,
# metrik = yil-agirlikli OOF MSE. En iyi config'in oof/test'i kaydedilir.
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from pathlib import Path
from train import load_data, build_xy, year_weights, OOF, CAT_COLS, TEXT_COL

ROOT = Path(__file__).parent

tr, te = load_data()
sw = year_weights(tr, te)
X, y, X_te, feats = build_xy(tr, te, True, True, use_kw=True, keep_text=True)
cat_feats = [c for c in CAT_COLS if c in X.columns]


def to_pool(Xd, yd=None):
    Xd = Xd.copy()
    for c in cat_feats:
        Xd[c] = Xd[c].astype(str)
    return Pool(Xd, yd, cat_features=cat_feats, text_features=[TEXT_COL])


base = dict(iterations=5000, learning_rate=0.03, depth=6, loss_function="RMSE",
            random_seed=42, verbose=0, task_type="GPU", early_stopping_rounds=200)

configs = {
    "base_d6": {},
    "d7": dict(depth=7),
    "d8_lr02": dict(depth=8, learning_rate=0.02),
    "d6_l2_9": dict(l2_leaf_reg=9.0),
    "d5_lr05": dict(depth=5, learning_rate=0.05),
}

kf = KFold(5, shuffle=True, random_state=42)
results = {}
for name, upd in configs.items():
    params = {**base, **upd}
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_te))
    pool_te = to_pool(X_te)
    for itr, iva in kf.split(X):
        m = CatBoostRegressor(**params)
        m.fit(to_pool(X.iloc[itr], y[itr]), eval_set=to_pool(X.iloc[iva], y[iva]))
        oof[iva] = m.predict(to_pool(X.iloc[iva]))
        test_pred += m.predict(pool_te) / 5
    oof = np.clip(oof, 0, 100)
    lb = mean_squared_error(y, oof, sample_weight=sw)
    results[name] = (lb, oof, np.clip(test_pred, 0, 100))
    print(f"{name:16s} esit={mean_squared_error(y, oof):8.3f}  lb-tahmini={lb:8.3f}",
          flush=True)

best = min(results, key=lambda k: results[k][0])
lb, oof, test_pred = results[best]
np.save(OOF / "cat_tuned_oof.npy", oof)
np.save(OOF / "cat_tuned_test.npy", test_pred)
print(f"\nen iyi: {best} ({lb:.3f}) -> oof/cat_tuned_*.npy kaydedildi")
