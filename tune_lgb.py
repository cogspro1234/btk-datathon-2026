# LGBM hiperparametre taramasi: sabit seed-42 KFold ile paired kiyas,
# metrik = yil-agirlikli OOF MSE (LB-tahmini). En iyi config'in oof/test'i kaydedilir.
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from pathlib import Path
from train import load_data, build_xy, year_weights, OOF

ROOT = Path(__file__).parent

tr, te = load_data()
sw = year_weights(tr, te)
X, y, X_te, feats = build_xy(tr, te, True, True, use_kw=True)

base = dict(objective="regression", n_estimators=5000, learning_rate=0.03,
            num_leaves=63, colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
            min_child_samples=30, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1)

configs = {
    "base": {},
    "lr02_leaves127": dict(learning_rate=0.02, num_leaves=127),
    "lr02_leaves31_mcs60": dict(learning_rate=0.02, num_leaves=31, min_child_samples=60),
    "leaves31": dict(num_leaves=31),
    "mcs80_l2_5": dict(min_child_samples=80, reg_lambda=5.0),
    "feat06": dict(colsample_bytree=0.6),
    "depth_ctrl": dict(num_leaves=31, max_depth=6, min_child_samples=50),
}

kf = KFold(5, shuffle=True, random_state=42)
results = {}
for name, upd in configs.items():
    params = {**base, **upd}
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_te))
    for itr, iva in kf.split(X):
        m = lgb.LGBMRegressor(**params)
        m.fit(X.iloc[itr], y[itr], eval_set=[(X.iloc[iva], y[iva])],
              callbacks=[lgb.early_stopping(200, verbose=False),
                         lgb.log_evaluation(0)])
        oof[iva] = m.predict(X.iloc[iva])
        test_pred += m.predict(X_te) / 5
    oof = np.clip(oof, 0, 100)
    lb = mean_squared_error(y, oof, sample_weight=sw)
    results[name] = (lb, oof, np.clip(test_pred, 0, 100))
    print(f"{name:24s} esit={mean_squared_error(y, oof):8.3f}  lb-tahmini={lb:8.3f}",
          flush=True)

best = min(results, key=lambda k: results[k][0])
lb, oof, test_pred = results[best]
np.save(OOF / "lgb_tuned_oof.npy", oof)
np.save(OOF / "lgb_tuned_test.npy", test_pred)
print(f"\nen iyi: {best} ({lb:.3f}) -> oof/lgb_tuned_*.npy kaydedildi")
