# Cesitlilik modelleri: MLP (sklearn) + ExtraTrees, ayni 5-fold seed-42 protokolu.
# Girdi: fe+kw tabular (one-hot) + TF-IDF/SVD64 metin. Cikti: oof/{mlp,et}_*.npy
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor
from train import load_data, build_xy, year_weights, OOF, CAT_COLS

tr, te = load_data()
sw = year_weights(tr, te)
X, y, X_te, feats = build_xy(tr, te, True, True, use_kw=True)

# one-hot + impute + scale (MLP icin sart, ET icin zararsiz)
Xall = pd.concat([X, X_te])
Xall = pd.get_dummies(Xall, columns=[c for c in CAT_COLS if c in Xall.columns])
Xall = Xall.fillna(Xall.median())
Xn = pd.DataFrame(StandardScaler().fit_transform(Xall), columns=Xall.columns)
Xtr_n, Xte_n = Xn.iloc[:len(X)].reset_index(drop=True), Xn.iloc[len(X):].reset_index(drop=True)

models = {
    "mlp": lambda seed: MLPRegressor(hidden_layer_sizes=(256, 128), alpha=1e-3,
                                     learning_rate_init=1e-3, batch_size=256,
                                     max_iter=300, early_stopping=True,
                                     n_iter_no_change=20, random_state=seed),
    "et": lambda seed: ExtraTreesRegressor(n_estimators=600, min_samples_leaf=10,
                                           max_features=0.5, n_jobs=-1,
                                           random_state=seed),
}

kf = KFold(5, shuffle=True, random_state=42)
for name, make in models.items():
    oof = np.zeros(len(y)); test_pred = np.zeros(len(Xte_n))
    for itr, iva in kf.split(Xtr_n):
        m = make(42)
        m.fit(Xtr_n.iloc[itr], y[itr])
        oof[iva] = m.predict(Xtr_n.iloc[iva])
        test_pred += m.predict(Xte_n) / 5
    oof = np.clip(oof, 0, 100)
    np.save(OOF / f"{name}_oof.npy", oof)
    np.save(OOF / f"{name}_test.npy", np.clip(test_pred, 0, 100))
    print(f"{name}: esit={mean_squared_error(y, oof):.3f}  "
          f"lb-tahmini={mean_squared_error(y, oof, sample_weight=sw):.3f}", flush=True)
