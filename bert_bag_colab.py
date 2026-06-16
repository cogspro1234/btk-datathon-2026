# =====================================================================
# Datathon 2026 - BERT seed bag (Colab L4): bert_big'in seed 1 ve 2 kosulari
# Kullanim: L4 GPU + csv'ler yuklu -> yapistir, calistir (~50-60 dk)
# Cikti: bert_big_s1_*, bert_big_s2_* npy'leri -> oof/ klasorune
# =====================================================================
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

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

tok = AutoTokenizer.from_pretrained(MODEL)
enc_tr = tok(list(tr["mentor_feedback_text"]), truncation=True, padding="max_length",
             max_length=MAXLEN, return_tensors="pt")
enc_te = tok(list(te["mentor_feedback_text"]), truncation=True, padding="max_length",
             max_length=MAXLEN, return_tensors="pt")
ids_tr, am_tr = enc_tr["input_ids"], enc_tr["attention_mask"]
ids_te, am_te = enc_te["input_ids"], enc_te["attention_mask"]
mu, sd = y.mean(), y.std()
z = (y - mu) / sd


def predict(model, ids, am):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ids), 128):
            with torch.autocast("cuda"):
                o = model(ids[i:i+128].to(DEV), attention_mask=am[i:i+128].to(DEV))
            out.append(o.logits.squeeze(-1).float().cpu().numpy())
    return np.concatenate(out)


for seed in [1, 2]:
    torch.manual_seed(seed)
    kf = KFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float32)
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
        oof[iva] = predict(model, ids_tr[iva], am_tr[iva]) * sd + mu
        test_pred += (predict(model, ids_te, am_te) * sd + mu) / 5
        print(f"seed {seed} fold {fold}: val mse="
              f"{mean_squared_error(y[iva], np.clip(oof[iva],0,100)):.2f}", flush=True)
        del model
        torch.cuda.empty_cache()
    np.save(f"bert_big_s{seed}_oof.npy", np.clip(oof, 0, 100))
    np.save(f"bert_big_s{seed}_test.npy", np.clip(test_pred, 0, 100))
    print(f"seed {seed} bitti: esit={mean_squared_error(y, np.clip(oof,0,100)):.2f}", flush=True)
print("bitti - 4 npy'yi indir")
