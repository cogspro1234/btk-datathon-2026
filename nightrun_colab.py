# =====================================================================
# Datathon 2026 - GECE PAKETI (Colab L4, ~2.5-3 saat)
# 1) TabPFN coklu-seed bag (v1 ve tabonly goruslerinin seed 1-2 + ne32)
# 2) BERT-base-turkish seed 3 ve 4
# Ciktilar HEM lokale HEM Google Drive'a yazilir (kopma sigortasi).
# KURULUM SIRASI:
#   Hucre 1: from google.colab import drive; drive.mount('/content/drive')
#   Hucre 2: !pip install -q tabpfn
#   train.csv + test_x.csv yukle
#   Hucre 3: import os; os.environ["TABPFN_TOKEN"]="SENIN_KEYIN"  + bu dosyanin tamami
# =====================================================================
import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack

OUTDIR = "/content/drive/MyDrive/datathon_night"
os.makedirs(OUTDIR, exist_ok=True)


def save(name, arr):
    np.save(f"{name}.npy", arr)
    shutil.copy(f"{name}.npy", f"{OUTDIR}/{name}.npy")


candidates = [".", "datathon-2026", "/kaggle/input/datathon-2026", "/content"]
base = next(p for p in candidates if os.path.exists(os.path.join(p, "train.csv")))
tr = pd.read_csv(os.path.join(base, "train.csv"), encoding="utf-8-sig")
te = pd.read_csv(os.path.join(base, "test_x.csv"), encoding="utf-8-sig")
y = tr["career_success_score"].to_numpy()
te_p = te["application_year"].value_counts(normalize=True)
tr_p = tr["application_year"].value_counts(normalize=True)
sw = tr["application_year"].map(te_p / tr_p).to_numpy(); sw = sw / sw.mean()

# ---------- BOLUM 1: TabPFN coklu-seed ----------
CATS = ["department", "university_tier", "target_role", "hobby",
        "preferred_social_media_platform"]
Xtab = tr.drop(columns=["student_id", "career_success_score", "mentor_feedback_text"]).copy()
Xtab_te = te.drop(columns=["student_id", "mentor_feedback_text"]).copy()
for c in CATS:
    cats = pd.Categorical(Xtab[c]).categories
    Xtab[c] = pd.Categorical(Xtab[c], categories=cats).codes
    Xtab_te[c] = pd.Categorical(Xtab_te[c], categories=cats).codes

word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                       max_features=50000, sublinear_tf=True)
W = hstack([word.fit_transform(tr["mentor_feedback_text"]),
            char.fit_transform(tr["mentor_feedback_text"])]).tocsr()
Wt = hstack([word.transform(te["mentor_feedback_text"]),
             char.transform(te["mentor_feedback_text"])]).tocsr()
svd = TruncatedSVD(n_components=64, random_state=42)
Ztr, Zte = svd.fit_transform(W), svd.transform(Wt)


def prep(*parts):
    X = np.hstack([p.to_numpy(dtype=np.float32) if hasattr(p, "to_numpy")
                   else p.astype(np.float32) for p in parts])
    return np.nan_to_num(X, nan=-1.0)


V1, V1t = prep(Xtab, Ztr), prep(Xtab_te, Zte)
V2, V2t = prep(Xtab), prep(Xtab_te)

from tabpfn import TabPFNRegressor

jobs = [
    ("tabpfn_v1k1", V1, V1t, 1, 8), ("tabpfn_v1k2", V1, V1t, 2, 8),
    ("tabpfn_tabonly_k1", V2, V2t, 1, 8), ("tabpfn_tabonly_k2", V2, V2t, 2, 8),
    ("tabpfn_ne32", V1, V1t, 42, 32),
]
for name, X, X_te, kseed, ne in jobs:
    kf = KFold(5, shuffle=True, random_state=kseed)
    oof = np.zeros(len(y)); test_pred = np.zeros(len(X_te))
    for itr, iva in kf.split(X):
        m = TabPFNRegressor(device="cuda", n_estimators=ne,
                            ignore_pretraining_limits=True)
        m.fit(X[itr], y[itr])
        oof[iva] = m.predict(X[iva]); test_pred += m.predict(X_te) / 5
    oof = np.clip(oof, 0, 100)
    save(f"{name}_oof", oof)
    save(f"{name}_test", np.clip(test_pred, 0, 100))
    print(f"{name}: lb-tahmini "
          f"{mean_squared_error(y, oof, sample_weight=sw):.3f}", flush=True)

# ---------- BOLUM 2: BERT seed 3-4 ----------
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

MODEL = "dbmdz/bert-base-turkish-cased"
DEV = "cuda"; EPOCHS = 4; BATCH = 32; MAXLEN = 160; LR = 2e-5
tok = AutoTokenizer.from_pretrained(MODEL)
enc_tr = tok(list(tr["mentor_feedback_text"]), truncation=True, padding="max_length",
             max_length=MAXLEN, return_tensors="pt")
enc_te = tok(list(te["mentor_feedback_text"]), truncation=True, padding="max_length",
             max_length=MAXLEN, return_tensors="pt")
ids_tr, am_tr = enc_tr["input_ids"], enc_tr["attention_mask"]
ids_te, am_te = enc_te["input_ids"], enc_te["attention_mask"]
yf = y.astype(np.float32)
mu, sd = yf.mean(), yf.std()
z = (yf - mu) / sd


def bert_predict(model, ids, am):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ids), 128):
            with torch.autocast("cuda"):
                o = model(ids[i:i+128].to(DEV), attention_mask=am[i:i+128].to(DEV))
            out.append(o.logits.squeeze(-1).float().cpu().numpy())
    return np.concatenate(out)


for seed in [3, 4]:
    torch.manual_seed(seed)
    kf = KFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(yf), dtype=np.float32)
    test_pred = np.zeros(len(te), dtype=np.float32)
    for fold, (itr, iva) in enumerate(kf.split(ids_tr)):
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL, num_labels=1, problem_type="regression").to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        dl = DataLoader(TensorDataset(ids_tr[itr], am_tr[itr], torch.tensor(z[itr])),
                        batch_size=BATCH, shuffle=True)
        sch = get_linear_schedule_with_warmup(opt, int(0.1*len(dl)*EPOCHS), len(dl)*EPOCHS)
        scaler = torch.amp.GradScaler("cuda")
        model.train()
        for ep in range(EPOCHS):
            for bids, bam, by in dl:
                opt.zero_grad()
                with torch.autocast("cuda"):
                    out = model(bids.to(DEV), attention_mask=bam.to(DEV)).logits.squeeze(-1)
                    loss = torch.nn.functional.mse_loss(out.float(), by.to(DEV))
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sch.step()
        oof[iva] = bert_predict(model, ids_tr[iva], am_tr[iva]) * sd + mu
        test_pred += (bert_predict(model, ids_te, am_te) * sd + mu) / 5
        print(f"bert s{seed} fold {fold}: "
              f"val mse={mean_squared_error(yf[iva], np.clip(oof[iva],0,100)):.2f}", flush=True)
        model = None
        torch.cuda.empty_cache()
    save(f"bert_big_s{seed}_oof", np.clip(oof, 0, 100))
    save(f"bert_big_s{seed}_test", np.clip(test_pred, 0, 100))
    print(f"bert s{seed} bitti", flush=True)

print("\nGECE PAKETI TAMAM - dosyalar Drive'da: datathon_night/", flush=True)
