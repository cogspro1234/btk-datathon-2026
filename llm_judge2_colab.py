# =====================================================================
# Datathon 2026 - LLM-judge v2: FEW-SHOT CALIBRATED + Qwen2.5-32B (A100)
# Kaan hoca tuyosu: mentor_feedback'e guclu LLM ile yuklen.
# - Daha guclu model (32B), prompt'a 16 gercek ornek+skor (few-shot kalibrasyon)
# - CHECKPOINT'li (gece internet kesintisi). Kesilirse tekrar calistir.
# Kurulum (Colab A100): !pip install -q -U transformers accelerate bitsandbytes
#   drive mount + train.csv + test_x.csv yukle, calistir.
# A100-40GB -> 32B 4-bit (~20GB) sigar. 80GB varsa USE_4BIT=False yapip bf16.
# Cikti: llm2_train.npy, llm2_test.npy (Drive: datathon_night/)
# =====================================================================
import os, json, re, glob
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTDIR = "/content/drive/MyDrive/datathon_night"
CKDIR = f"{OUTDIR}/llm2_ckpt"
os.makedirs(CKDIR, exist_ok=True)
CHUNK = 500
BATCH = 16
MODEL = "Qwen/Qwen2.5-32B-Instruct"
USE_4BIT = True   # A100-40GB: 4-bit (~20GB). H100-80GB olursa False (bf16).

cands = [".", "datathon-2026", "/kaggle/input/datathon-2026", "/content"]
base = next(p for p in cands if os.path.exists(os.path.join(p, "train.csv")))
tr = pd.read_csv(os.path.join(base, "train.csv"), encoding="utf-8-sig")
te = pd.read_csv(os.path.join(base, "test_x.csv"), encoding="utf-8-sig")
ytr = tr["career_success_score"].to_numpy()

# --- few-shot havuzu: skor dagilimina yayilmis 16 train ornegi (SABIT, leakage-safe) ---
rng = np.random.default_rng(0)
order = np.argsort(ytr)
fs_idx = [int(order[int(q)]) for q in np.linspace(50, len(order)-50, 16)]
fs_set = set(fs_idx)
fewshot = ""
for i in fs_idx:
    t = tr["mentor_feedback_text"].iloc[i]
    s = int(round(ytr[i]))
    fewshot += f"Metin: {t}\nJSON: {{\"genel_potansiyel\": {s}}}\n\n"

# extraction seti: 16 few-shot HARIC tum train + tum test
texts = list(tr["mentor_feedback_text"]) + list(te["mentor_feedback_text"])
ntr = len(tr); N = len(texts)
KEYS = ["genel_potansiyel", "teknik", "iletisim", "liderlik", "proje",
        "ovgu_yogunlugu", "elestiri_yogunlugu", "ust_duzey_ifade"]

SYS = ("Sen kariyer degerlendirme uzmanisin. Mentor degerlendirme metninden "
       "ogrencinin kariyer basari skorunu ve alt boyutlari tahmin et. "
       "Asagida gercek ornekler var; ayni olcegi kullan. SADECE JSON dondur:\n"
       '{"genel_potansiyel":0-100, "teknik":-2..2, "iletisim":-2..2, '
       '"liderlik":-2..2, "proje":-2..2, "ovgu_yogunlugu":0-3, '
       '"elestiri_yogunlugu":0-3, "ust_duzey_ifade":0/1}\n\n'
       "Kalibrasyon ornekleri (gercek skorlar):\n" + fewshot)
PROMPT = "Metin: {t}\nJSON:"


def parse(txt):
    m = re.search(r"\{.*\}", txt, re.S); d = {}
    if m:
        try: d = json.loads(m.group(0))
        except Exception: d = {}
    return [float(d.get(k, np.nan)) if isinstance(d.get(k, None), (int, float)) else np.nan
            for k in KEYS]


done = {int(re.search(r"chunk_(\d+)", f).group(1)) for f in glob.glob(f"{CKDIR}/chunk_*.npy")}
todo = [c for c in range(0, N, CHUNK) if c not in done]
print(f"toplam {N}, {len(done)} chunk bitti, {len(todo)} kaldi | few-shot idx: {fs_idx[:4]}...", flush=True)

if todo:
    tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    if USE_4BIT:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                                     device_map="cuda").eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                     device_map="cuda").eval()

    def gen(bt):
        msgs = [[{"role": "system", "content": SYS}, {"role": "user", "content": PROMPT.format(t=t)}] for t in bt]
        pr = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        enc = tok(pr, return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=110, do_sample=False, pad_token_id=tok.pad_token_id)
        return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    for start in todo:
        ct = texts[start:start+CHUNK]; res = []
        for b in range(0, len(ct), BATCH):
            res.extend(gen(ct[b:b+BATCH]))
        np.save(f"{CKDIR}/chunk_{start}.npy", np.array([parse(t) for t in res], dtype=np.float32))
        print(f"chunk {start} ({start+len(ct)}/{N})", flush=True)

M = np.full((N, len(KEYS)), np.nan, dtype=np.float32)
for f in glob.glob(f"{CKDIR}/chunk_*.npy"):
    s = int(re.search(r"chunk_(\d+)", f).group(1)); a = np.load(f); M[s:s+len(a)] = a
np.save(f"{OUTDIR}/llm2_train.npy", M[:ntr]); np.save(f"{OUTDIR}/llm2_test.npy", M[ntr:])
np.save("llm2_train.npy", M[:ntr]); np.save("llm2_test.npy", M[ntr:])
gp = M[:ntr, 0]; ok = ~np.isnan(gp)
print("TAMAM. NaN:", round(float(np.isnan(M).mean()), 3))
print("few-shot 32B genel_potansiyel<->hedef korelasyon:",
      round(float(np.corrcoef(gp[ok], ytr[ok])[0, 1]), 4), "| (7B few-shot'suz: 0.4629)")
