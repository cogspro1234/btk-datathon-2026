# Final zincir: bag'lar -> NM agirlik optimizasyonu (yil-agirlikli) ->
# yil-bazli dogrusal kalibrasyon -> submission.
# Rapor: kalibrasyonun nested (durust) kazanci + final dosyanin LB-tahmini.
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).parent
OOF = ROOT / "oof"
SUBS = ROOT / "submissions"

tr = pd.read_csv(ROOT / "datathon-2026" / "train.csv", encoding="utf-8-sig")
te = pd.read_csv(ROOT / "datathon-2026" / "test_x.csv", encoding="utf-8-sig")
y = tr["career_success_score"].to_numpy()
yil = tr["application_year"].to_numpy()
te_p = te["application_year"].value_counts(normalize=True)
tr_p = tr["application_year"].value_counts(normalize=True)
sw = tr["application_year"].map(te_p / tr_p).to_numpy(); sw = sw / sw.mean()


def load(n):
    return np.load(OOF / f"{n}_oof.npy"), np.load(OOF / f"{n}_test.npy")


def bag(names):
    o = np.mean([load(n)[0] for n in names], axis=0)
    t = np.mean([load(n)[1] for n in names], axis=0)
    return o, t


components = {
    "cat_bag_cpu": bag(["cat_fekw_text", "cat_fekw_text_s1cpu", "cat_fekw_text_s2cpu",
                        "cat_fekw_text_s3cpu", "cat_fekw_text_s4cpu"]),
    "catl2_bag": bag(["cat_l2", "cat_l2_s1", "cat_l2_s2", "cat_l2_s3", "cat_l2_s4"]),
    "lgb_bag": bag(["lgb_fekw_text", "lgb_fekw_text_s1", "lgb_fekw_text_s2"]),
    "lgbt_bag": bag(["lgb_tuned", "lgb_tuned_s1", "lgb_tuned_s2",
                     "lgb_tuned_s3", "lgb_tuned_s4"]),
    "xgbt_bag": bag(["xgb_tuned", "xgb_tuned_s1", "xgb_tuned_s2", "xgb_tuned_s3",
                     "xgb_tuned_s4", "xgb_tuned_s5", "xgb_tuned_s6", "xgb_tuned_s7",
                     "xgb_tuned_s8", "xgb_tuned_s9", "xgb_tuned_s10"]),
    "cat_fe_text_emb": load("cat_fe_text_emb"),
    "lgb_fekw_text_emb": load("lgb_fekw_text_emb"),
    "lgb_fe2kw_text": load("lgb_fe2kw_text"),
    "mlp": load("mlp"),
    "et": load("et"),
    "bert_big": bag(["bert_big", "bert_big_s1", "bert_big_s2",
                     "bert_big_s3", "bert_big_s4"]),
    "lgb_rawtext": load("lgb_rawtext"),
    "tabpfn_v1bag": bag(["tabpfn", "tabpfn_ne16", "tabpfn_v1k1",
                         "tabpfn_v1k2", "tabpfn_ne32"]),
    "tabpfn_rich": load("tabpfn_rich"),
    "tabpfn_tobag": bag(["tabpfn_tabonly", "tabpfn_tabonly_k1", "tabpfn_tabonly_k2"]),
    "fusion": load("fusion"),
    "e5tab": load("e5tab"),
    "e5lgb": load("e5lgb"),
    "lgbt_llm": load("lgbt_llm"),
    "catl2_llm": load("catl2_llm"),
    "xgbt_llm": load("xgbt_llm"),
    "tabpfn_llm": load("tabpfn_llm"),
}
names = list(components)
oofs = np.column_stack([components[n][0] for n in names])
tests = np.column_stack([components[n][1] for n in names])
for i, n in enumerate(names):
    print(f"{n:20s} lb-tahmini = {mean_squared_error(y, oofs[:, i], sample_weight=sw):8.3f}")


def loss(w):
    w = np.abs(w) / np.abs(w).sum()
    return mean_squared_error(y, oofs @ w, sample_weight=sw)


best = min((minimize(loss, w0, method="Nelder-Mead",
                     options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-7})
            for w0 in [np.ones(len(names)), *np.eye(len(names))]),
           key=lambda r: r.fun)
w = np.abs(best.x) / np.abs(best.x).sum()
print("\nagirliklar:", {n: round(float(x), 4) for n, x in zip(names, w)})
blend_oof = oofs @ w
blend_test = tests @ w
np.save(OOF / "blend_oof.npy", blend_oof)
np.save(OOF / "blend_test.npy", blend_test)
print(f"blend lb-tahmini (kalibrasyonsuz) = "
      f"{mean_squared_error(y, np.clip(blend_oof, 0, 100), sample_weight=sw):.4f}")

# BERT artik duzeltmesi (bert_res.py ciktisi varsa): en iyi alfa OOF'ta aranir
if (OOF / "bert_res_oof.npy").exists():
    b_oof = np.load(OOF / "bert_res_oof.npy")
    b_test = np.load(OOF / "bert_res_test.npy")
    alphas = np.arange(0.0, 1.01, 0.05)
    scores = [mean_squared_error(y, blend_oof + a * b_oof, sample_weight=sw)
              for a in alphas]
    a_best = float(alphas[int(np.argmin(scores))])
    print(f"bert duzeltmesi: alfa={a_best:.2f} -> {min(scores):.4f}")
    blend_oof = blend_oof + a_best * b_oof
    blend_test = blend_test + a_best * b_test


def calibrate(p_fit, y_fit, yil_fit, p_apply, yil_apply):
    from sklearn.linear_model import HuberRegressor
    out = p_apply.copy()
    for yy in np.unique(yil_apply):
        m_f = yil_fit == yy
        m_a = yil_apply == yy
        if m_f.sum() < 50:
            continue
        h = HuberRegressor(epsilon=1.5).fit(p_fit[m_f].reshape(-1, 1), y_fit[m_f])
        out[m_a] = h.predict(p_apply[m_a].reshape(-1, 1))
    return out


# durust (nested) kalibrasyon kazanci
kf = KFold(5, shuffle=True, random_state=7)
cal = blend_oof.copy()
for itr, iva in kf.split(blend_oof):
    cal[iva] = calibrate(blend_oof[itr], y[itr], yil[itr], blend_oof[iva], yil[iva])
cal = np.clip(cal, 0, 100)
nested = mean_squared_error(y, cal, sample_weight=sw)
print(f"blend + yil kalibrasyonu (nested, durust) = {nested:.4f}")

# final: kalibrasyonu tum OOF'la ogren, test'e uygula
test_cal = np.clip(calibrate(blend_oof, y, yil, blend_test,
                             te["application_year"].to_numpy()), 0, 100)
sub = pd.DataFrame({"student_id": te["student_id"], "career_success_score": test_cal})
assert len(sub) == len(te) and sub["career_success_score"].notna().all()
path = SUBS / f"sub_blendv2_cal_lb{nested:.3f}.csv"
sub.to_csv(path, index=False)
print("yazildi:", path)
