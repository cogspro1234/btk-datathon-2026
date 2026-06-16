# =====================================================================
# Datathon 2026 - LLM-as-judge (transformers, CHECKPOINT'li)
# vLLM yerine transformers batch generation (Colab CUDA uyumlu).
# Kurulum (Colab L4 GPU): !pip install -q -U transformers accelerate
#   drive mount + train.csv + test_x.csv yukle, calistir.
# Kesilirse SADECE tekrar calistir; biten chunk'lari atlar.
# Cikti: llm_train.npy, llm_test.npy (Drive: datathon_night/)
# =====================================================================
import os, json, re, glob
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTDIR = "/content/drive/MyDrive/datathon_night"
CKDIR = f"{OUTDIR}/llm_ckpt"
os.makedirs(CKDIR, exist_ok=True)
CHUNK = 500
BATCH = 32
MODEL = "Qwen/Qwen2.5-7B-Instruct"

cands = [".", "datathon-2026", "/kaggle/input/datathon-2026", "/content"]
base = next(p for p in cands if os.path.exists(os.path.join(p, "train.csv")))
tr = pd.read_csv(os.path.join(base, "train.csv"), encoding="utf-8-sig")
te = pd.read_csv(os.path.join(base, "test_x.csv"), encoding="utf-8-sig")
texts = list(tr["mentor_feedback_text"]) + list(te["mentor_feedback_text"])
N = len(texts); ntr = len(tr)
KEYS = ["genel_potansiyel","teknik","iletisim","liderlik","proje",
        "ovgu_yogunlugu","elestiri_yogunlugu","ust_duzey_ifade"]

SYS = ("Sen bir teknik ise alim uzmanisin. Sana bir ogrenci hakkinda mentor "
       "degerlendirme metni verilecek. SADECE su JSON'u doldur, baska hicbir sey yazma:\n"
       '{"genel_potansiyel":0-100 tam sayi, "teknik":-2..2, "iletisim":-2..2, '
       '"liderlik":-2..2, "proje":-2..2, "ovgu_yogunlugu":0-3, "elestiri_yogunlugu":0-3, '
       '"ust_duzey_ifade":0/1}\n'
       "genel_potansiyel: metnin ima ettigi kariyer basari skoru tahminin. "
       "teknik/iletisim/liderlik/proje: o alandaki ton (-2 cok zayif, +2 cok guclu, 0 notr/gecmiyor). "
       "ust_duzey_ifade: 'kohortun en iyisi/istisnai/olaganustu' gibi en ust ovgu varsa 1.")
PROMPT = "Mentor degerlendirmesi:\n{t}\n\nJSON:"


def parse(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    d = {}
    if m:
        try: d = json.loads(m.group(0))
        except Exception: d = {}
    out = []
    for k in KEYS:
        v = d.get(k, np.nan)
        out.append(float(v) if isinstance(v, (int, float)) else np.nan)
    return out


done = set()
for f in glob.glob(f"{CKDIR}/chunk_*.npy"):
    done.add(int(re.search(r"chunk_(\d+)", f).group(1)))
todo = [c for c in range(0, N, CHUNK) if c not in done]
print(f"toplam {N} metin, {len(done)} chunk bitmis, {len(todo)} kaldi", flush=True)

if todo:
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="cuda")
    model.eval()

    def gen_batch(batch_texts):
        msgs = [[{"role": "system", "content": SYS},
                 {"role": "user", "content": PROMPT.format(t=t)}] for t in batch_texts]
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in msgs]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=900).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=110, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        return tok.batch_decode(gen, skip_special_tokens=True)

    for start in todo:
        chunk_texts = texts[start:start+CHUNK]
        res = []
        for b in range(0, len(chunk_texts), BATCH):
            res.extend(gen_batch(chunk_texts[b:b+BATCH]))
        arr = np.array([parse(t) for t in res], dtype=np.float32)
        np.save(f"{CKDIR}/chunk_{start}.npy", arr)
        print(f"chunk {start} bitti ({start+len(chunk_texts)}/{N}) nan={np.isnan(arr).mean():.2f}",
              flush=True)

M = np.full((N, len(KEYS)), np.nan, dtype=np.float32)
for f in glob.glob(f"{CKDIR}/chunk_*.npy"):
    s = int(re.search(r"chunk_(\d+)", f).group(1))
    a = np.load(f); M[s:s+len(a)] = a
np.save(f"{OUTDIR}/llm_train.npy", M[:ntr]); np.save(f"{OUTDIR}/llm_test.npy", M[ntr:])
np.save("llm_train.npy", M[:ntr]); np.save("llm_test.npy", M[ntr:])
print("TAMAM. NaN orani:", np.isnan(M).mean())
y = tr["career_success_score"].to_numpy(); gp = M[:ntr, 0]; ok = ~np.isnan(gp)
print("genel_potansiyel <-> hedef korelasyon:", round(float(np.corrcoef(gp[ok], y[ok])[0,1]), 4))
print("ornek (ilk 3):", M[:3].tolist())
