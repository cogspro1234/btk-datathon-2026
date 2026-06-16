# BERT artik-tahmini: Turkce distilbert'i blend'in artigi (y - blend_oof)
# uzerinde fine-tune et. Metinde tabular-otesi sinyal varsa burada cikar.
# Cikti: oof/bert_res_{oof,test}.npy + blend+alfa*bert degerlendirmesi.
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from train import load_data, year_weights, OOF

MODEL = "dbmdz/distilbert-base-turkish-cased"
DEV = "cuda"
EPOCHS = 3
BATCH = 16
MAXLEN = 128

tr, te = load_data()
sw = year_weights(tr, te)
y = tr["career_success_score"].to_numpy()
blend_oof = np.load(OOF / "blend_oof.npy")
blend_test = np.load(OOF / "blend_test.npy")
res = (y - blend_oof).astype(np.float32)
print(f"artik: std={res.std():.2f}")

tok = AutoTokenizer.from_pretrained(MODEL)


def encode(texts):
    enc = tok(list(texts), truncation=True, padding="max_length",
              max_length=MAXLEN, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


ids_tr, am_tr = encode(tr["mentor_feedback_text"])
ids_te, am_te = encode(te["mentor_feedback_text"])


def predict(model, ids, am):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ids), 64):
            o = model(ids[i:i+64].to(DEV), attention_mask=am[i:i+64].to(DEV))
            out.append(o.logits.squeeze(-1).float().cpu().numpy())
    return np.concatenate(out)


kf = KFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(y), dtype=np.float32)
test_pred = np.zeros(len(te), dtype=np.float32)
for fold, (itr, iva) in enumerate(kf.split(ids_tr)):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=1, problem_type="regression").to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    ds = TensorDataset(ids_tr[itr], am_tr[itr], torch.tensor(res[itr]))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)
    model.train()
    for ep in range(EPOCHS):
        tot = 0.0
        for bi, (bids, bam, by) in enumerate(dl):
            opt.zero_grad()
            out = model(bids.to(DEV), attention_mask=bam.to(DEV)).logits.squeeze(-1)
            loss = torch.nn.functional.mse_loss(out, by.to(DEV))
            loss.backward()
            opt.step()
            tot += loss.item() * len(by)
        print(f"fold {fold} epoch {ep}: train mse={tot/len(itr):.3f}", flush=True)
        model.train()
    oof[iva] = predict(model, ids_tr[iva], am_tr[iva])
    test_pred += predict(model, ids_te, am_te) / 5
    va_corr = np.corrcoef(oof[iva], res[iva])[0, 1]
    print(f"fold {fold}: val corr(res, pred)={va_corr:.4f}", flush=True)
    del model
    torch.cuda.empty_cache()

np.save(OOF / "bert_res_oof.npy", oof)
np.save(OOF / "bert_res_test.npy", test_pred)

base = mean_squared_error(y, np.clip(blend_oof, 0, 100), sample_weight=sw)
print(f"\nblend (kalibrasyonsuz): {base:.4f}")
for alpha in [0.25, 0.5, 0.75, 1.0]:
    m = mean_squared_error(y, np.clip(blend_oof + alpha * oof, 0, 100),
                           sample_weight=sw)
    print(f"blend + {alpha:.2f}*bert_res: {m:.4f}")
