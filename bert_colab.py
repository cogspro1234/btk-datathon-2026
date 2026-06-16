# =====================================================================
# Datathon 2026 - Tam boy Turkce BERT (Colab/Kaggle GPU icin)
# Kullanim (Colab): Runtime -> Change runtime type -> T4 GPU
#   1) Bu dosyanin TAMAMINI tek hucreye yapistir
#   2) train.csv ve test_x.csv'yi calisma dizinine yukle
#      (Kaggle Notebook'ta: Add Input -> competition data; yollar otomatik bulunur)
#   3) Calistir (~45-60 dk T4'te). Bitince bert_big_oof.npy ve
#      bert_big_test.npy dosyalarini indir -> projede oof/ klasorune koy
# Cikti: y'yi DOGRUDAN tahmin eden 5-fold OOF + test tahminleri.
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

# --- veri yollarini bul (lokal / Colab / Kaggle) ---
candidates = [".", "datathon-2026", "/kaggle/input/datathon-2026", "/content"]
base = next(p for p in candidates if os.path.exists(os.path.join(p, "train.csv")))
tr = pd.read_csv(os.path.join(base, "train.csv"), encoding="utf-8-sig")
te = pd.read_csv(os.path.join(base, "test_x.csv"), encoding="utf-8-sig")
y = tr["career_success_score"].to_numpy(dtype=np.float32)
print("veri:", tr.shape, te.shape, "| cihaz:", torch.cuda.get_device_name(0))

tok = AutoTokenizer.from_pretrained(MODEL)


def encode(texts):
    e = tok(list(texts), truncation=True, padding="max_length",
            max_length=MAXLEN, return_tensors="pt")
    return e["input_ids"], e["attention_mask"]


ids_tr, am_tr = encode(tr["mentor_feedback_text"])
ids_te, am_te = encode(te["mentor_feedback_text"])

# hedefi standardize et (egitim stabilitesi icin), sonda geri cevrilir
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


kf = KFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(y), dtype=np.float32)
test_pred = np.zeros(len(te), dtype=np.float32)
for fold, (itr, iva) in enumerate(kf.split(ids_tr)):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=1, problem_type="regression").to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    ds = TensorDataset(ids_tr[itr], am_tr[itr], torch.tensor(z[itr]))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)
    total_steps = len(dl) * EPOCHS
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    for ep in range(EPOCHS):
        tot = 0.0
        for bids, bam, by in dl:
            opt.zero_grad()
            with torch.autocast("cuda"):
                out = model(bids.to(DEV), attention_mask=bam.to(DEV)).logits.squeeze(-1)
                loss = torch.nn.functional.mse_loss(out.float(), by.to(DEV))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sch.step()
            tot += loss.item() * len(by)
        print(f"fold {fold} epoch {ep}: train mse(z)={tot/len(itr):.4f}", flush=True)
        model.train()
    oof[iva] = predict(model, ids_tr[iva], am_tr[iva]) * sd + mu
    test_pred += (predict(model, ids_te, am_te) * sd + mu) / 5
    print(f"fold {fold}: val mse={mean_squared_error(y[iva], np.clip(oof[iva],0,100)):.3f} "
          f"corr={np.corrcoef(oof[iva], y[iva])[0,1]:.4f}", flush=True)
    del model
    torch.cuda.empty_cache()

oof = np.clip(oof, 0, 100)
test_pred = np.clip(test_pred, 0, 100)
np.save("bert_big_oof.npy", oof)
np.save("bert_big_test.npy", test_pred)
print("\nOOF MSE (esit agirlik):", round(mean_squared_error(y, oof), 3))
print("kaydedildi: bert_big_oof.npy, bert_big_test.npy -> bunlari indir, oof/ klasorune koy")
