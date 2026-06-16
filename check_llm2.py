import base64, numpy as np, pandas as pd, json, glob, os
# Drive'dan cekilen chunk base64'lerini oof/llm2_b64/ icine .txt koyacagiz;
# bu script onlari decode edip train korelasyonu hesaplar.
# Kullanim: her chunk icin oof/llm2_b64/chunk_<start>.txt (base64 icerik)
d = "oof/llm2_b64"
parts = {}
for f in glob.glob(d + "/chunk_*.txt"):
    s = int(os.path.basename(f)[6:-4])
    b = open(f).read().strip()
    parts[s] = np.frombuffer(base64.b64decode(b), dtype=np.uint8)
if not parts:
    print("oof/llm2_b64/ bos - once base64 .txt'leri koy"); raise SystemExit
tr = pd.read_csv("datathon-2026/train.csv", encoding="utf-8-sig")
y = tr["career_success_score"].to_numpy()
arrs = {}
for s, raw in parts.items():
    open("oof/_tmp.npy", "wb").write(raw.tobytes())
    arrs[s] = np.load("oof/_tmp.npy")
gp_all, y_all = [], []
for s, a in arrs.items():
    if s < len(y):  # train kismi
        n = min(len(a), len(y) - s)
        gp_all.append(a[:n, 0]); y_all.append(y[s:s+n])
gp = np.concatenate(gp_all); yy = np.concatenate(y_all)
ok = ~np.isnan(gp)
print(f"ornek satir: {ok.sum()}")
print(f"few-shot 32B genel_potansiyel<->hedef: {np.corrcoef(gp[ok], yy[ok])[0,1]:.4f}")
print("(referans: 7B few-shot'suz = 0.4629)")
