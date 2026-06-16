# mentor_feedback_text icin cok dilli sentence-transformer embedding'leri
# cikarir ve cache'ler (oof/emb_train.npy, oof/emb_test.npy).
# Bir kez kosulur; train.py --emb bu cache'i kullanir.
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
OOF = ROOT / "oof"
MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

tr = pd.read_csv(ROOT / "datathon-2026" / "train.csv", encoding="utf-8-sig")
te = pd.read_csv(ROOT / "datathon-2026" / "test_x.csv", encoding="utf-8-sig")

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(MODEL, device="cpu")
OOF.mkdir(exist_ok=True)
for name, df in [("train", tr), ("test", te)]:
    emb = model.encode(df["mentor_feedback_text"].tolist(), batch_size=64,
                       show_progress_bar=True, normalize_embeddings=True)
    np.save(OOF / f"emb_{name}.npy", emb)
    print(name, emb.shape)
