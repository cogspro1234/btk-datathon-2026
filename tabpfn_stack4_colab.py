# =====================================================================
# Datathon 2026 - stack4: stack3 + LLM-judge sinyali
# Girdi: bilesen tahminleri (+lgbt_llm) + 45 ham tabular + e5_svd128 + LLM(8)
# Kurulum: drive mount + !pip install -q tabpfn + train/test + stack_inputs.zip
#   !unzip -o stack_inputs.zip -d stack
#   os.environ["TABPFN_TOKEN"]="..." + bu dosya
# e5_train/test.npy Drive'dan; llm_train/test.npy zip icinde (stack/).
# =====================================================================
import os, shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD

OUTDIR = "/content/drive/MyDrive/datathon_night"
def save(name, arr):
    np.save(f"{name}.npy", arr)
    try: shutil.copy(f"{name}.npy", f"{OUTDIR}/{name}.npy")
    except Exception: pass

cands = [".", "datathon-2026", "/kaggle/input/datathon-2026", "/content"]
base = next(p for p in cands if os.path.exists(os.path.join(p, "train.csv")))
tr = pd.read_csv(os.path.join(base, "train.csv"), encoding="utf-8-sig")
te = pd.read_csv(os.path.join(base, "test_x.csv"), encoding="utf-8-sig")
y = tr["career_success_score"].to_numpy()
S = "stack"


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
        "bertbig": bag(["bert_big", "bert_big_s1", "bert_big_s2",
                        "bert_big_s3", "bert_big_s4"], suf),
        "bert_res": np.load(f"{S}/bert_res_{suf}.npy"),
        "tabpfn_v1": bag(["tabpfn", "tabpfn_ne16", "tabpfn_v1k1",
                          "tabpfn_v1k2", "tabpfn_ne32"], suf),
        "rich": np.load(f"{S}/tabpfn_rich_{suf}.npy"),
        "tabonly": bag(["tabpfn_tabonly", "tabpfn_tabonly_k1", "tabpfn_tabonly_k2"], suf),
        "fusion": np.load(f"{S}/fusion_{suf}.npy"),
        "e5tab": np.load(f"{S}/e5tab_{suf}.npy"),
        "e5lgb": np.load(f"{S}/e5lgb_{suf}.npy"),
        "lgbt_llm": np.load(f"{S}/lgbt_llm_{suf}.npy"),  # YENI: LLM-base
    }


CATS = ["department", "university_tier", "target_role", "hobby",
        "preferred_social_media_platform"]
Rtr = tr.drop(columns=["student_id", "career_success_score", "mentor_feedback_text"]).copy()
Rte = te.drop(columns=["student_id", "mentor_feedback_text"]).copy()
for c in CATS:
    cc = pd.Categorical(Rtr[c]).categories
    Rtr[c] = pd.Categorical(Rtr[c], categories=cc).codes
    Rte[c] = pd.Categorical(Rte[c], categories=cc).codes
Rtr = np.nan_to_num(Rtr.to_numpy(np.float32), nan=-1.0)
Rte = np.nan_to_num(Rte.to_numpy(np.float32), nan=-1.0)

# ham e5 embedding -> SVD128 (Drive'dan)
E_tr = np.load(f"{OUTDIR}/e5_train.npy")
E_te = np.load(f"{OUTDIR}/e5_test.npy")
svd = TruncatedSVD(n_components=128, random_state=42)
Etr = svd.fit_transform(E_tr).astype(np.float32)
Ete = svd.transform(E_te).astype(np.float32)

# LLM-judge feature'lari (8) - zip icinde stack/
Ltr = np.load(f"{S}/llm_train.npy").astype(np.float32)
Lte = np.load(f"{S}/llm_test.npy").astype(np.float32)

Ctr, Cte = comp("oof"), comp("test")
Xtr = np.column_stack([np.column_stack(list(Ctr.values())), Rtr, Etr, Ltr]).astype(np.float32)
Xte = np.column_stack([np.column_stack(list(Cte.values())), Rte, Ete, Lte]).astype(np.float32)
print("stack4 girdisi:", Xtr.shape)  # ~20 + 45 + 128 + 8

te_p = te["application_year"].value_counts(normalize=True)
tr_p = tr["application_year"].value_counts(normalize=True)
sw = tr["application_year"].map(te_p / tr_p).to_numpy(); sw = sw / sw.mean()

from tabpfn import TabPFNRegressor
SEEDS = [7, 21, 101]
bag_oof = np.zeros(len(y)); bag_test = np.zeros(len(Xte))
for s in SEEDS:
    kf = KFold(5, shuffle=True, random_state=s)
    oof = np.zeros(len(y)); tp = np.zeros(len(Xte))
    for itr, iva in kf.split(Xtr):
        m = TabPFNRegressor(device="cuda", n_estimators=8, ignore_pretraining_limits=True)
        m.fit(Xtr[itr], y[itr])
        oof[iva] = m.predict(Xtr[iva]); tp += m.predict(Xte) / 5
    oof = np.clip(oof, 0, 100)
    print(f"seed {s}: lb-tahmini {mean_squared_error(y, oof, sample_weight=sw):.3f}", flush=True)
    bag_oof += oof / len(SEEDS); bag_test += np.clip(tp, 0, 100) / len(SEEDS)

bag_oof = np.clip(bag_oof, 0, 100); bag_test = np.clip(bag_test, 0, 100)
save("stack4_oof", bag_oof); save("stack4_test", bag_test)
print("\nstack4 (e5+LLM) lb-tahmini:",
      round(mean_squared_error(y, bag_oof, sample_weight=sw), 3))
print("referans: stack3 (e5, LLM'siz) 82.315")
