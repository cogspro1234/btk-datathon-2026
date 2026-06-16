# =====================================================================
# Datathon 2026 - FUZYON modeli (Colab L4): BERT encoder + tabular feature'lar
# tek agda, ortak regresyon kafasi. Metin+tabular etkilesimini dogrudan ogrenir.
# Kullanim: L4 GPU + csv'ler -> yapistir, calistir (~60-70 dk)
# Cikti: fusion_oof.npy, fusion_test.npy -> oof/ klasorune
# =====================================================================
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

MODEL = "dbmdz/bert-base-turkish-cased"
DEV = "cuda"
EPOCHS = 4
BATCH = 32
MAXLEN = 160
LR = 2e-5

candidates = [".", "datathon-2026", "/kaggle/input/datathon-2026", "/content"]
base = next(p for p in candidates if os.path.exists(os.path.join(p, "train.csv")))
tr = pd.read_csv(os.path.join(base, "train.csv"), encoding="utf-8-sig")
te = pd.read_csv(os.path.join(base, "test_x.csv"), encoding="utf-8-sig")
y = tr["career_success_score"].to_numpy(dtype=np.float32)

CATS = ["department", "university_tier", "target_role", "hobby",
        "preferred_social_media_platform"]
Xtr = tr.drop(columns=["student_id", "career_success_score", "mentor_feedback_text"]).copy()
Xte = te.drop(columns=["student_id", "mentor_feedback_text"]).copy()
for c in CATS:
    cats = pd.Categorical(Xtr[c]).categories
    Xtr[c] = pd.Categorical(Xtr[c], categories=cats).codes
    Xte[c] = pd.Categorical(Xte[c], categories=cats).codes
Xall = pd.concat([Xtr, Xte])
med = Xall.median()
sc = StandardScaler().fit(Xall.fillna(med))
Ttr = sc.transform(Xtr.fillna(med)).astype(np.float32)
Tte = sc.transform(Xte.fillna(med)).astype(np.float32)
NTAB = Ttr.shape[1]

tok = AutoTokenizer.from_pretrained(MODEL)
enc_tr = tok(list(tr["mentor_feedback_text"]), truncation=True, padding="max_length",
             max_length=MAXLEN, return_tensors="pt")
enc_te = tok(list(te["mentor_feedback_text"]), truncation=True, padding="max_length",
             max_length=MAXLEN, return_tensors="pt")
mu, sd = y.mean(), y.std()
z = (y - mu) / sd


class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL)
        self.tab = nn.Sequential(nn.Linear(NTAB, 128), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(128, 128), nn.GELU())
        self.head = nn.Sequential(nn.Linear(768 + 128, 256), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(256, 1))

    def forward(self, ids, am, tab):
        h = self.bert(ids, attention_mask=am).last_hidden_state[:, 0]  # CLS
        t = self.tab(tab)
        return self.head(torch.cat([h, t], dim=1)).squeeze(-1)


def predict(model, ids, am, T):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ids), 128):
            with torch.autocast("cuda"):
                o = model(ids[i:i+128].to(DEV), am[i:i+128].to(DEV),
                          torch.tensor(T[i:i+128]).to(DEV))
            out.append(o.float().cpu().numpy())
    return np.concatenate(out)


kf = KFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(y), dtype=np.float32)
test_pred = np.zeros(len(te), dtype=np.float32)
ids_tr, am_tr = enc_tr["input_ids"], enc_tr["attention_mask"]
ids_te, am_te = enc_te["input_ids"], enc_te["attention_mask"]
for fold, (itr, iva) in enumerate(kf.split(ids_tr)):
    model = Fusion().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    dl = DataLoader(TensorDataset(ids_tr[itr], am_tr[itr],
                                  torch.tensor(Ttr[itr]), torch.tensor(z[itr])),
                    batch_size=BATCH, shuffle=True)
    sch = get_linear_schedule_with_warmup(opt, int(0.1*len(dl)*EPOCHS), len(dl)*EPOCHS)
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    for ep in range(EPOCHS):
        tot = 0.0
        for bids, bam, btab, by in dl:
            opt.zero_grad()
            with torch.autocast("cuda"):
                out = model(bids.to(DEV), bam.to(DEV), btab.to(DEV))
                loss = torch.nn.functional.mse_loss(out.float(), by.to(DEV))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sch.step()
            tot += loss.item() * len(by)
        print(f"fold {fold} ep {ep}: train mse(z)={tot/len(itr):.4f}", flush=True)
        model.train()
    oof[iva] = predict(model, ids_tr[iva], am_tr[iva], Ttr[iva]) * sd + mu
    test_pred += (predict(model, ids_te, am_te, Tte) * sd + mu) / 5
    print(f"fold {fold}: val mse={mean_squared_error(y[iva], np.clip(oof[iva],0,100)):.2f}",
          flush=True)
    del model
    torch.cuda.empty_cache()

oof = np.clip(oof, 0, 100)
test_pred = np.clip(test_pred, 0, 100)
np.save("fusion_oof.npy", oof)
np.save("fusion_test.npy", test_pred)
print("\nesit OOF MSE:", round(mean_squared_error(y, oof), 2))
te_p = te["application_year"].value_counts(normalize=True)
tr_p = tr["application_year"].value_counts(normalize=True)
sw = tr["application_year"].map(te_p / tr_p).to_numpy(); sw = sw / sw.mean()
print("LB-tahmini:", round(mean_squared_error(y, oof, sample_weight=sw), 2))
print("kaydedildi: fusion_oof.npy, fusion_test.npy")
