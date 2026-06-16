# XGBoost hiperparametre taramasi (GPU): sabit seed-42 KFold, paired kiyas,
# metrik = yil-agirlikli OOF MSE. En iyi config'in oof/test'i kaydedilir.
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from train import load_data, build_xy, year_weights, OOF

tr, te = load_data()
sw = year_weights(tr, te)
X, y, X_te, feats = build_xy(tr, te, True, True, use_kw=True)

base = dict(n_estimators=5000, learning_rate=0.03, max_depth=6,
            colsample_bytree=0.8, subsample=0.8, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.0, tree_method="hist", device="cuda",
            enable_categorical=True, early_stopping_rounds=200,
            random_state=42, verbosity=0)

configs = {
    "base": {},
    "d4_lr02": dict(max_depth=4, learning_rate=0.02),
    "d5_mcw20": dict(max_depth=5, min_child_weight=20),
    "d6_l2_8": dict(reg_lambda=8.0),
    "d8_sub06": dict(max_depth=8, subsample=0.6, colsample_bytree=0.6),
}

kf = KFold(5, shuffle=True, random_state=42)
results = {}
for name, upd in configs.items():
    params = {**base, **upd}
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_te))
    for itr, iva in kf.split(X):
        m = XGBRegressor(**params)
        m.fit(X.iloc[itr], y[itr], eval_set=[(X.iloc[iva], y[iva])], verbose=False)
        oof[iva] = m.predict(X.iloc[iva])
        test_pred += m.predict(X_te) / 5
    oof = np.clip(oof, 0, 100)
    lb = mean_squared_error(y, oof, sample_weight=sw)
    results[name] = (lb, oof, np.clip(test_pred, 0, 100))
    print(f"{name:12s} esit={mean_squared_error(y, oof):8.3f}  lb-tahmini={lb:8.3f}",
          flush=True)

best = min(results, key=lambda k: results[k][0])
lb, oof, test_pred = results[best]
np.save(OOF / "xgb_tuned_oof.npy", oof)
np.save(OOF / "xgb_tuned_test.npy", test_pred)
print(f"\nen iyi: {best} ({lb:.3f}) -> oof/xgb_tuned_*.npy kaydedildi")
