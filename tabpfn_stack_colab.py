# =====================================================================
# Datathon 2026 - TabPFN META-STACKER (Colab L4)
# Bilesen tahminlerini + yil + birkac ham feature'i girdi alip nihai
# tahmini TabPFN'e ogrettiriyoruz (NM dogrusal blend'in nonlinear alternatifi).
# Kullanim: L4 + tabpfn kurulu + TABPFN_TOKEN ayarli oturum:
#   train.csv, test_x.csv VE stack_inputs.zip yukle ->
#   !unzip -o stack_inputs.zip -d stack  -> sonra bu hucreyi calistir (~15-25 dk)
# Cikti: stack_oof.npy, stack_test.npy
# =====================================================================
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

candidates = [".", "datathon-2026", "/kaggle/input/datathon-2026", "/content"]
base = next(p for p in candidates if os.path.exists(os.path.join(p, "train.csv")))
tr = pd.read_csv(os.path.join(base, "train.csv"), encoding="utf-8-sig")
te = pd.read_csv(os.path.join(base, "test_x.csv"), encoding="utf-8-sig")
y = tr["career_success_score"].to_numpy()
S = "stack"  # unzip klasoru


def bag(names, suf):
    return np.mean([np.load(f"{S}/{n}_{suf}.npy") for n in names], axis=0)


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
        "cat_emb": np.load(f"{S}/cat_fe_text_emb_{suf}.npy"),
        "lgb_emb": np.load(f"{S}/lgb_fekw_text_emb_{suf}.npy"),
        "lgb_fe2": np.load(f"{S}/lgb_fe2kw_text_{suf}.npy"),
        "rawtext": np.load(f"{S}/lgb_rawtext_{suf}.npy"),
        "mlp": np.load(f"{S}/mlp_{suf}.npy"),
        "bert_big": bag(["bert_big", "bert_big_s1", "bert_big_s2"], suf),
        "fusion": np.load(f"{S}/fusion_{suf}.npy"),
        "bert_res": np.load(f"{S}/bert_res_{suf}.npy"),
        "tabpfn": np.load(f"{S}/tabpfn_{suf}.npy"),
        "ne16": np.load(f"{S}/tabpfn_ne16_{suf}.npy"),
        "rich": np.load(f"{S}/tabpfn_rich_{suf}.npy"),
        "tabonly": np.load(f"{S}/tabpfn_tabonly_{suf}.npy"),
    }


Ctr, Cte = comp("oof"), comp("test")
RAW = ["application_year", "project_quality_score", "technical_interview_score",
       "problem_solving_score", "communication_score"]
Xtr = np.column_stack(list(Ctr.values()) + [tr[c].fillna(-1).to_numpy() for c in RAW]).astype(np.float32)
Xte = np.column_stack(list(Cte.values()) + [te[c].fillna(-1).to_numpy() for c in RAW]).astype(np.float32)
print("stacker girdisi:", Xtr.shape)

from tabpfn import TabPFNRegressor

te_p = te["application_year"].value_counts(normalize=True)
tr_p = tr["application_year"].value_counts(normalize=True)
sw = tr["application_year"].map(te_p / tr_p).to_numpy(); sw = sw / sw.mean()

# STACK-BAG: stacker'i 3 farkli fold-seed ile kosup ortala (stacking gurultusunu kirpar)
SEEDS = [7, 21, 101]
bag_oof = np.zeros(len(y))
bag_test = np.zeros(len(Xte))
for s in SEEDS:
    kf = KFold(5, shuffle=True, random_state=s)
    oof = np.zeros(len(y)); test_pred = np.zeros(len(Xte))
    for fold, (itr, iva) in enumerate(kf.split(Xtr)):
        m = TabPFNRegressor(device="cuda", n_estimators=8, ignore_pretraining_limits=True)
        m.fit(Xtr[itr], y[itr])
        oof[iva] = m.predict(Xtr[iva])
        test_pred += m.predict(Xte) / 5
    oof = np.clip(oof, 0, 100)
    print(f"seed {s}: lb-tahmini "
          f"{mean_squared_error(y, oof, sample_weight=sw):.3f}", flush=True)
    bag_oof += oof / len(SEEDS)
    bag_test += np.clip(test_pred, 0, 100) / len(SEEDS)

bag_oof = np.clip(bag_oof, 0, 100)
bag_test = np.clip(bag_test, 0, 100)
np.save("stack_oof.npy", bag_oof)       # geriye uyumluluk (eski isimle)
np.save("stack_test.npy", bag_test)
np.save("stackbag_oof.npy", bag_oof)
np.save("stackbag_test.npy", bag_test)
print("\nSTACK-BAG (3 seed) lb-tahmini:",
      round(mean_squared_error(y, bag_oof, sample_weight=sw), 3))
print("kaydedildi: stackbag_oof.npy, stackbag_test.npy (+ stack_oof/test.npy)")
