# bert_big seed-3, lokal 1050 Ti (gece izniyle). VRAM'e gore batch secer.
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from train import load_data, OOF

MODEL = "dbmdz/bert-base-turkish-cased"
DEV = "cuda"
EPOCHS = 4
MAXLEN = 160
LR = 2e-5
SEED = 3

tr, te = load_data()
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
        for i in range(0, len(ids), 32):
            with torch.autocast("cuda"):
                o = model(ids[i:i+32].to(DEV), attention_mask=am[i:i+32].to(DEV))
            out.append(o.logits.squeeze(-1).float().cpu().numpy())
    return np.concatenate(out)


def make_model():
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=1, problem_type="regression").to(DEV)


def pick_batch():
    for b in [16, 12, 8]:
        try:
            m = make_model()
            opt = torch.optim.AdamW(m.parameters(), lr=LR)
            scaler = torch.amp.GradScaler("cuda")
            for _ in range(2):
                opt.zero_grad()
                with torch.autocast("cuda"):
                    o = m(ids_tr[:b].to(DEV), attention_mask=am_tr[:b].to(DEV)).logits.squeeze(-1)
                    l = torch.nn.functional.mse_loss(o.float(), torch.tensor(z[:b]).to(DEV))
                scaler.scale(l).backward()
                scaler.step(opt)
                scaler.update()
            m = None
            torch.cuda.empty_cache()
            return b
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    return None


BATCH = pick_batch()
print("secilen batch:", BATCH, flush=True)
assert BATCH, "VRAM yetmedi"

torch.manual_seed(SEED)
kf = KFold(5, shuffle=True, random_state=SEED)
oof = np.zeros(len(y), dtype=np.float32)
test_pred = np.zeros(len(te), dtype=np.float32)
for fold, (itr, iva) in enumerate(kf.split(ids_tr)):
    model = make_model()
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
        print(f"fold {fold} ep {ep} ok", flush=True)
    oof[iva] = predict(model, ids_tr[iva], am_tr[iva]) * sd + mu
    test_pred += (predict(model, ids_te, am_te) * sd + mu) / 5
    print(f"fold {fold}: val mse={mean_squared_error(y[iva], np.clip(oof[iva],0,100)):.2f}",
          flush=True)
    model = None
    torch.cuda.empty_cache()

np.save(OOF / "bert_big_s3_oof.npy", np.clip(oof, 0, 100))
np.save(OOF / "bert_big_s3_test.npy", np.clip(test_pred, 0, 100))
print("bert_big_s3 bitti:", round(mean_squared_error(y, np.clip(oof, 0, 100)), 2), flush=True)
