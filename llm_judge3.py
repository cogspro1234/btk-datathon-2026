# =====================================================================
# Datathon 2026 - LLM-judge v3: Claude (Sonnet 4.6) as career evaluator
# Metni OKU -> career_success_score (0-100) tahmin et. ZERO-SHOT: hedefi
# gormez -> leakage YOK, fold gerekmez. Fizibilite (Opus, n=120): |corr|=0.7057
# (v2 Qwen-32B baseline ayni satir: 0.6555).
#
# Anahtar: ANTHROPIC_API_KEY env, ya da anthropic_key.txt (proje koku).
# Kurulum: pip install anthropic
# Mod:
#   python llm_judge3.py --probe   -> 120 satir, corr raporu (~$0.01)
#   python llm_judge3.py --full    -> 20k, oof/llm3_train.npy + llm3_test.npy
# Checkpoint'li: cokerse tekrar calistir, kaldigi yerden devam (para iki kez gitmez)
# =====================================================================
import os, sys, re, time, json, argparse
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL = "claude-sonnet-4-6"
BS = 25          # metin / cagri
WORKERS = 8


def get_key():
    k = os.environ.get("ANTHROPIC_API_KEY")
    if k:
        return k.strip()
    for p in ("anthropic_key.txt", "key.txt", "secrets/anthropic_key.txt"):
        if os.path.exists(p):
            return open(p, encoding="utf-8").read().strip()
    sys.exit("ANTHROPIC_API_KEY bulunamadi: env'e koy ya da anthropic_key.txt'e yapistir.")


import anthropic
client = anthropic.Anthropic(api_key=get_key())

SYS = ("Sen uzman bir kariyer degerlendiricisisin. Sana numarali TURKCE mentor "
       "degerlendirme metinleri verilecek. HER metin icin o ogrencinin "
       "career_success_score'unu (0-100 arasi surekli kariyer basari skoru) SADECE "
       "metne dayanarak tahmin et: tonu, ovgu/elestiri dengesini, somut basari/eksik "
       "vurgularini ve kidem ipuclarini dikkate al. Hedef ortalama ~77, std ~15; "
       "tum 0-100 araligini kullan, tahminleri 77 civarina yigma. "
       "Cikti SADECE 'indeks|skor' satirlari olsun (skor ondalikli olabilir), "
       "baska hicbir aciklama yazma.")


def score_batch(items):
    """items: [(global_idx, text), ...] -> {global_idx: score}"""
    body = "\n".join(f"{j}|{' '.join(str(t).split())}" for j, (gi, t) in enumerate(items, 1))
    for attempt in range(4):
        try:
            msg = client.messages.create(
                model=MODEL, max_tokens=BS * 12, system=SYS,
                messages=[{"role": "user", "content": body}],
            )
            out = "".join(b.text for b in msg.content if b.type == "text")
            res = {}
            for ln in out.splitlines():
                m = re.match(r"\s*(\d+)\s*\|\s*([-+]?\d*\.?\d+)", ln)
                if m:
                    li = int(m.group(1))
                    if 1 <= li <= len(items):
                        res[items[li - 1][0]] = max(0.0, min(100.0, float(m.group(2))))
            if len(res) >= int(0.8 * len(items)):
                return res
        except Exception as e:
            sys.stderr.write(f"  batch retry {attempt}: {e}\n")
            time.sleep(2 * (attempt + 1))
    return {}


def run(indices, texts, ckpt):
    os.makedirs(os.path.dirname(ckpt) or ".", exist_ok=True)
    done = {}
    if os.path.exists(ckpt):
        done = {int(k): v for k, v in json.load(open(ckpt)).items()}
    batches = []
    for s in range(0, len(indices), BS):
        chunk = [(int(indices[k]), texts[k]) for k in range(s, min(s + BS, len(indices)))]
        if not all(gi in done for gi, _ in chunk):
            batches.append(chunk)
    print(f"{len(batches)} batch kaldi ({len(done)} ornek bitti), model={MODEL}", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(score_batch, b) for b in batches]
        for n, f in enumerate(as_completed(futs), 1):
            done.update(f.result())
            if n % 20 == 0:
                json.dump({str(k): v for k, v in done.items()}, open(ckpt, "w"))
                print(f"  {n}/{len(batches)}", flush=True)
    json.dump({str(k): v for k, v in done.items()}, open(ckpt, "w"))
    return done


def batch_run(texts, idx, idfile, mapfile):
    """Batch API (%50 ucuz, asenkron). Submit-or-resume + poll + collect.
    Doner: {global_idx: score} ya da None (hala islemde)."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    if os.path.exists(idfile):
        bid = open(idfile).read().strip()
        cmap = json.load(open(mapfile))
        print(f"mevcut batch: {bid} (devam)", flush=True)
    else:
        reqs, cmap = [], {}
        for s in range(0, len(idx), BS):
            gis = [int(idx[k]) for k in range(s, min(s + BS, len(idx)))]
            txs = [texts[k] for k in range(s, min(s + BS, len(idx)))]
            cid = f"b{s}"
            cmap[cid] = gis
            body = "\n".join(f"{j}|{' '.join(str(t).split())}" for j, t in enumerate(txs, 1))
            reqs.append(Request(custom_id=cid, params=MessageCreateParamsNonStreaming(
                model=MODEL, max_tokens=BS * 12, system=SYS,
                messages=[{"role": "user", "content": body}])))
        batch = client.messages.batches.create(requests=reqs)
        bid = batch.id
        open(idfile, "w").write(bid)
        json.dump(cmap, open(mapfile, "w"))
        print(f"batch GONDERILDI: {bid} ({len(reqs)} istek, ~%50 ucuz)", flush=True)
    t0 = time.time()
    while True:
        b = client.messages.batches.retrieve(bid)
        rc = b.request_counts
        print(f"  status={b.processing_status} ok={rc.succeeded} err={rc.errored} "
              f"islemde={rc.processing} ({int(time.time()-t0)}s)", flush=True)
        if b.processing_status == "ended":
            break
        if time.time() - t0 > 3 * 3600:
            print("3 saat doldu -> cikiliyor; sonra --batch ile devam et.", flush=True)
            return None
        time.sleep(60)
    res = {}
    for r in client.messages.batches.results(bid):
        if r.result.type != "succeeded":
            continue
        out = "".join(blk.text for blk in r.result.message.content if blk.type == "text")
        gis = cmap[r.custom_id]
        for ln in out.splitlines():
            m = re.match(r"\s*(\d+)\s*\|\s*([-+]?\d*\.?\d+)", ln)
            if m:
                li = int(m.group(1))
                if 1 <= li <= len(gis):
                    res[gis[li - 1]] = max(0.0, min(100.0, float(m.group(2))))
    return res


def load():
    tr = pd.read_csv("datathon-2026/train.csv", encoding="utf-8-sig")
    te = pd.read_csv("datathon-2026/test_x.csv", encoding="utf-8-sig")
    return tr, te


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--batch", action="store_true")
    args = ap.parse_args()
    from scipy.stats import pearsonr

    if args.probe:
        m = np.load("text_probe_meta.npz")
        idx, y, v2 = m["idx"], m["y"], m["v2"]
        tr, _ = load()
        txt = tr["mentor_feedback_text"].astype(str).values
        done = run(idx, [txt[i] for i in idx], "oof/llm3_probe_ckpt.json")
        pred = np.array([done.get(int(i), np.nan) for i in idx])
        ok = ~np.isnan(pred)
        print(f"\nSonnet judge probe (n={ok.sum()}): |corr|={abs(pearsonr(pred[ok], y[ok])[0]):.4f}")
        print(f"  v2 baseline (ayni satir): {abs(pearsonr(v2[ok], y[ok])[0]):.4f}")
        print(f"  Opus fizibilite referans : 0.7057")

    if args.full:
        tr, te = load()
        txt = list(tr["mentor_feedback_text"].astype(str)) + list(te["mentor_feedback_text"].astype(str))
        idx = np.arange(len(txt))
        done = run(idx, txt, "oof/llm3_ckpt.json")
        arr = np.array([done.get(int(i), np.nan) for i in idx], float)
        ntr = len(tr)
        # NaN kalirsa medyan ile doldur (GBM zaten NaN halleder ama stacker icin temiz tut)
        if np.isnan(arr).any():
            arr = np.where(np.isnan(arr), np.nanmedian(arr), arr)
        np.save("oof/llm3_train.npy", arr[:ntr])
        np.save("oof/llm3_test.npy", arr[ntr:])
        y = tr["career_success_score"].values
        print(f"\nllm3 yazildi (oof/llm3_train.npy, llm3_test.npy). "
              f"train |corr|={abs(pearsonr(arr[:ntr], y)[0]):.4f}")

    if args.batch:
        tr, te = load()
        txt = list(tr["mentor_feedback_text"].astype(str)) + list(te["mentor_feedback_text"].astype(str))
        idx = np.arange(len(txt))
        res = batch_run(txt, idx, "oof/llm3_batch_id.txt", "oof/llm3_batch_map.json")
        if res is None:
            sys.exit(0)
        arr = np.array([res.get(int(i), np.nan) for i in idx], float)
        ntr = len(tr)
        if np.isnan(arr).any():
            arr = np.where(np.isnan(arr), np.nanmedian(arr), arr)
        np.save("oof/llm3_train.npy", arr[:ntr])
        np.save("oof/llm3_test.npy", arr[ntr:])
        y = tr["career_success_score"].values
        print(f"\nllm3 (batch) yazildi. train |corr|={abs(pearsonr(arr[:ntr], y)[0]):.4f}")
