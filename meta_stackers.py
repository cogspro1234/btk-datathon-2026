# Meta-stacker varyantlari (lokal, gece izniyle): TabPFN-stacker'in yaninda
# LGB-meta ve Ridge-meta. Ayni girdi semasi (17 bilesen + yil + 4 ham feature).
# Cikti: oof/lgbstack_*, oof/ridgestack_* + degerlendirme raporu.
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from pathlib import Path

ROOT = Path(__file__).parent
OOF = ROOT / "oof"
tr = pd.read_csv(ROOT / "datathon-2026" / "train.csv", encoding="utf-8-sig")
te = pd.read_csv(ROOT / "datathon-2026" / "test_x.csv", encoding="utf-8-sig")
y = tr["career_success_score"].to_numpy()
te_p = te["application_year"].value_counts(normalize=True)
tr_p = tr["application_year"].value_counts(normalize=True)
sw = tr["application_year"].map(te_p / tr_p).to_numpy(); sw = sw / sw.mean()


def bag(names, suf):
    return np.mean([np.load(OOF / f"{n}_{suf}.npy") for n in names], axis=0)


def comp(suf):
    return {
        "cat_bag": bag(["cat_fekw_text", "cat_fekw_text_s1cpu", "cat_fekw_text_s2cpu",
                        "cat_fekw_text_s3cpu", "cat_fekw_text_s4cpu"], suf),
        "catl2": bag(["cat_l2", "cat_l2_s1", "cat_l2_s2", "cat_l2_s3", "cat_l2_s4"], suf),
        "lgb_bag": bag(["lgb_fekw_text", "lgb_fekw_text_s1", "lgb_fekw_text_s2"], suf),
        "lgbt": bag(["lgb_tuned", "lgb_tuned_s1", "lgb_tuned_s2",
                     "lgb_tuned_s3", "lgb_tuned_s4"], suf),
        "xgbt": bag(["xgb_tuned", "xgb_tuned_s1", "xgb_tuned_s2", "xgb_tuned_s3",
                     "xgb_tuned_s4", "xgb_tuned_s5", "xgb_tuned_s6", "xgb_tuned_s7",
                     "xgb_tuned_s8", "xgb_tuned_s9", "xgb_tuned_s10"], suf),
        "cat_emb": np.load(OOF / f"cat_fe_text_emb_{suf}.npy"),
        "lgb_emb": np.load(OOF / f"lgb_fekw_text_emb_{suf}.npy"),
        "lgb_fe2": np.load(OOF / f"lgb_fe2kw_text_{suf}.npy"),
        "rawtext": np.load(OOF / f"lgb_rawtext_{suf}.npy"),
        "mlp": np.load(OOF / f"mlp_{suf}.npy"),
        "bertbig_bag": bag(["bert_big", "bert_big_s1", "bert_big_s2"], suf),
        "bert_res": np.load(OOF / f"bert_res_{suf}.npy"),
        "tabpfn": np.load(OOF / f"tabpfn_{suf}.npy"),
        "ne16": np.load(OOF / f"tabpfn_ne16_{suf}.npy"),
        "rich": np.load(OOF / f"tabpfn_rich_{suf}.npy"),
        "tabonly": np.load(OOF / f"tabpfn_tabonly_{suf}.npy"),
    }


RAW = ["application_year", "project_quality_score", "technical_interview_score",
       "problem_solving_score", "communication_score"]
Ctr, Cte = comp("oof"), comp("test")
Xtr = np.column_stack(list(Ctr.values()) + [tr[c].fillna(-1).to_numpy() for c in RAW]).astype(np.float32)
Xte = np.column_stack(list(Cte.values()) + [te[c].fillna(-1).to_numpy() for c in RAW]).astype(np.float32)

SEEDS = [7, 21, 101]  # stackbag ile ayni protokol


def run_bag(name, make):
    bag_oof = np.zeros(len(y)); bag_test = np.zeros(len(Xte))
    for s in SEEDS:
        kf = KFold(5, shuffle=True, random_state=s)
        oof = np.zeros(len(y)); test_pred = np.zeros(len(Xte))
        for itr, iva in kf.split(Xtr):
            m = make()
            m.fit(Xtr[itr], y[itr])
            oof[iva] = m.predict(Xtr[iva]); test_pred += m.predict(Xte) / 5
        bag_oof += np.clip(oof, 0, 100) / len(SEEDS)
        bag_test += np.clip(test_pred, 0, 100) / len(SEEDS)
    bag_oof = np.clip(bag_oof, 0, 100); bag_test = np.clip(bag_test, 0, 100)
    np.save(OOF / f"{name}_oof.npy", bag_oof)
    np.save(OOF / f"{name}_test.npy", bag_test)
    print(f"{name}: lb-tahmini {mean_squared_error(y, bag_oof, sample_weight=sw):.3f}", flush=True)
    return bag_oof


lgbstack = run_bag("lgbstack", lambda: lgb.LGBMRegressor(
    n_estimators=2000, learning_rate=0.01, num_leaves=15, max_depth=4,
    min_child_samples=100, colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
    reg_lambda=5.0, verbose=-1))
ridgestack = run_bag("ridgestack", lambda: Ridge(alpha=5.0))

# uc meta-ogrenici bag'i (tabpfn-stackbag + lgbstack + ridgestack) + NM karisimi
s_tab = np.load(OOF / "stackbag_oof.npy")
nm = np.clip(np.load(OOF / "blend_oof.npy") + 0.35 * np.load(OOF / "bert_res_oof.npy"), 0, 100)
best = (None, 1e9)
for a in np.arange(0, 1.01, 0.1):
    for b in np.arange(0, 1.01 - a, 0.1):
        for c in np.arange(0, 1.01 - a - b, 0.1):
            d = 1 - a - b - c
            if d < -1e-9:
                continue
            p = np.clip(a*s_tab + b*lgbstack + c*ridgestack + d*nm, 0, 100)
            m = mean_squared_error(y, p, sample_weight=sw)
            if m < best[1]:
                best = ((round(a, 1), round(b, 1), round(c, 1), round(d, 1)), m)
print(f"\nen iyi karisim (tabpfn-stackbag/lgbstack/ridgestack/NM): {best[0]} -> {best[1]:.3f}")
print("referans: stackbag tek 82.615, sub11 82.563(public), sub12 mix 82.53")
