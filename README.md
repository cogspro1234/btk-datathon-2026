# BTK Datathon 2026 — Öğrenci Kariyer Başarısı Tahmini

Bu repo, BTK Datathon 2026 yarışmasındaki çözümümü içerir. Görev: bir öğrencinin
profil + mentor değerlendirme metninden `career_success_score` değerini (0–100 arası
**sürekli** bir skor) tahmin etmek. Değerlendirme metriği: **MSE** (ne kadar düşük o
kadar iyi). Leaderboard %60 public / %40 private; final için 2 submission seçilir.

> Ayrıntılı anlatım, EDA grafikleri ve gerekçeler için ana belge:
> **[`Datathon2026_Cozum.ipynb`](Datathon2026_Cozum.ipynb)** (çıktılarıyla birlikte).

---

## 1. Veri

| Dosya | Boyut | Açıklama |
|---|---|---|
| `train.csv` | 10.000 × 47 | hedef (`career_success_score`) dahil |
| `test_x.csv` | 10.000 × 46 | hedef hariç |

Sütunlar: kimlik (`student_id`), kategorikler (`department`, `university_tier`,
`target_role`, `hobby`, `preferred_social_media_platform`), 9 teknik skor
(coding, problem_solving, data_structures, sql, machine_learning, backend, frontend,
cloud, devops), 4 sosyal skor (communication, teamwork, leadership, presentation),
4 çekirdek skor (project_quality, technical_interview, communication, portfolio),
sayısal sinyaller (başvuru/staj/hackathon/github sayıları, yıllar) ve serbest metin
(`mentor_feedback_text`).

Veri lisans + boyut nedeniyle repoda **yer almaz**; Kaggle yarışma sayfasından
indirilir ve proje kökünde `datathon-2026/train.csv`, `datathon-2026/test_x.csv`
olarak konumlanır (scriptlerin beklediği yol).

---

## 2. Yaklaşımın özü: yıl-ağırlıklı doğrulama

İlk modelimde 5-fold CV ile public leaderboard arasında **~11 MSE** fark gördüm. Sebebi
veride gizli bir dağılım kaymasıydı:

- **Yıl kayması:** test kayıtları son yıllara (2024–2026) eğitim setinden çok daha
  yoğun düşüyor.
- **Gürültü artışı:** hedefin yıllara göre standart sapması büyüyor (2019 ~12.6 →
  2025 ~18.3); yani son yıllar hem daha ağırlıklı hem daha zor.

Çözüm: her eğitim satırını **test/train yıl oranıyla** ağırlıklandıran bir OOF (out-of-fold)
metriği. Bu metrik public skoru **±0.1 hatayla** tahmin etti (örn. 91.12 tahmin /
91.119 gerçek). Bundan sonra **public LB'ye değil, bu metriğe** güvenerek karar verdim.
Yıl-ağırlığı hem skorlamada hem de model eğitiminde `sample_weight` olarak kullanıldı
(`year_weights`, `train.py`).

---

## 3. Pipeline mimarisi (akış)

```
veri (datathon-2026/)
  │
  ├─ feature engineering            train.py: add_fe / add_fe2 / add_keyword_features
  │
  ├─ BASE MODELLER (5-fold OOF, yıl-ağırlıklı, çoğu CPU)
  │     ├─ GBM ailesi               train.py (lgb/xgb/cat) + tune_lgb/tune_xgb/tune_cat
  │     ├─ çeşitlilik               nn_et.py (MLP, ExtraTrees)
  │     └─ tablo+embedding          embed.py (sentence-transformer SVD)
  │
  ├─ METİN KANALI (Colab GPU)
  │     ├─ BERTurk regresör         bert_colab / bert_bag_colab / bert_s3_local / bert_res
  │     ├─ e5 embedding             e5_colab (e5tab=TabPFN, e5lgb=LGBM)
  │     ├─ füzyon                   fusion_colab
  │     └─ LLM-as-judge             llm_judge_colab (Qwen-7B) / llm_judge2_colab (Qwen-32B)
  │
  ├─ TabPFN ailesi (Colab GPU)      tabpfn_colab / tabpfn_v2_colab / nightrun_colab / tabpfn_llm_colab
  │
  └─ STACK / BLEND  →  FINAL
        ├─ sub19 (primary)          tabpfn_stack6_colab.py  (TabPFN meta-stacker)
        └─ sub20 (hedge)            blend_final.py (NM blend + yıl kalibrasyonu) ·0.75 + lgbt_llm2 ·0.25
```

Tüm base modeller OOF tahminlerini `oof/<isim>_oof.npy` (train) ve `oof/<isim>_test.npy`
(test) olarak yazar; stacker/blender bunları okur.

---

## 4. Final teslim (best-of-2)

Kaggle'da iki submission final için işaretlenir. Mimari riskten korunmak için ikisini
**kasıtlı dekorele** seçtim — biri stacker'a bağımlı, diğeri değil:

- **sub19 — primary** · `tabpfn_stack6_colab.py`
  Tüm base bileşenleri + 45 ham tablo özelliği + e5-SVD128 + v2 LLM-judge (8 feature)
  TabPFN'e **stacker girdisi** olarak verir; 3 seed (7/21/101) × 5-fold bagging.
  Nested (dürüst) OOF: **81.81** → public **82.00**.
- **sub20 — hedge** · `blend_final.py` çıktısı (·0.75) + `lgbt_llm2` (·0.25)
  Stacker'sız, tam dekorele sigorta. `blend_final.py`: bileşen bag'leri → Nelder-Mead
  ağırlık optimizasyonu (yıl-ağırlıklı) → yıl-bazlı Huber kalibrasyon.

Not: tüm güçlü modeller birbirine ~0.997 korele (gürültü tabanı). Hedge'in değeri
skorda değil, **mimaride** (stacker private'a aşırı uyarsa diye).

---

## 5. Dosya rehberi

### Çekirdek tablo + base modeller
| Dosya | Rol |
|---|---|
| `train.py` | Feature engineering + 5-fold CV iskeleti. `--model {lgb,xgb,cat}`, `--fe/--fe2/--kw/--text/--emb/--llm/--l2/--yw/--seed` bayraklarıyla tüm GBM base'leri, seed bag'leri ve `_emb`/`_llm` varyantları buradan üretilir. |
| `tune_lgb.py` | LGBM hiperparametre taraması (paired, yıl-ağırlıklı) → `lgb_tuned` |
| `tune_xgb.py` | XGBoost taraması → `xgb_tuned` (11 seed'lik bag) |
| `tune_cat.py` | CatBoost taraması → `cat_tuned` |
| `nn_et.py` | MLP + ExtraTrees (ensemble çeşitliliği) → `mlp`, `et` |
| `embed.py` | Sentence-transformer embedding üretir + cache'ler → `oof/emb_train.npy`, `oof/emb_test.npy` |

### Metin kanalı (NLP — çoğu Colab GPU)
| Dosya | Rol |
|---|---|
| `bert_colab.py` | BERTurk (`dbmdz/bert-base-turkish`) regresörü → `bert_big` |
| `bert_bag_colab.py` | BERTurk seed bag → `bert_big_s1..s4` |
| `bert_s3_local.py` | Aynı modelin lokal s3 koşumu → `bert_big_s3` |
| `bert_res.py` | BERT artık (residual) düzeltmesi; blend'de alfa OOF'ta aranır |
| `e5_colab.py` | `multilingual-e5` embedding + `e5tab` (TabPFN) / `e5lgb` (LGBM) + ham `e5_train/test.npy` |
| `fusion_colab.py` | Tablo + metin füzyon modeli → `fusion` |
| `llm_judge_colab.py` | LLM-as-judge — **Qwen2.5-7B**, 8 boyutlu yapılandırılmış skor (v1, `genel_potansiyel`↔hedef corr ~0.46) → `oof/llm_train.npy` |
| `llm_judge2_colab.py` | LLM-as-judge — **Qwen2.5-32B few-shot kalibrasyonlu** (v2, corr ~0.58) → `oof/llm2_train.npy`; `lgbt_llm2`'nin kaynağı |
| `check_llm2.py` | LLM-judge çıktısının korelasyon/sağlamlık analizi |

### TabPFN ailesi (Colab GPU)
| Dosya | Rol |
|---|---|
| `tabpfn_colab.py` | TabPFN base → `tabpfn` |
| `tabpfn_v2_colab.py` | `tabpfn_rich`, `tabpfn_tabonly`, `tabpfn_ne16` |
| `nightrun_colab.py` | TabPFN seed/k varyantları → `tabpfn_v1k1/k2`, `tabpfn_tabonly_k1/k2`, `tabpfn_ne32` |
| `tabpfn_llm_colab.py` | TabPFN + LLM-judge feature → `tabpfn_llm` |
| `meta_stackers.py` | Meta-stacker yardımcı fonksiyonları |

### Stack / Blend
| Dosya | Rol |
|---|---|
| **`tabpfn_stack6_colab.py`** | **sub19 (primary)** — v2 LLM stacker, `lgbt_llm2` base dahil |
| `tabpfn_stack_colab.py`, `tabpfn_stack2_colab.py`, `tabpfn_stack3_colab.py`, `tabpfn_stack4_colab.py` | Stacker iterasyonları (stack6'nın öncülleri) |
| **`blend_final.py`** | **sub20 (hedge) tabanı** — Nelder-Mead blend + yıl-bazlı Huber kalibrasyon (nested ölçüm) |
| `blend.py` | Erken (stack öncesi) blend |

### EDA ve denenip elenenler
| Dosya | Rol |
|---|---|
| `eda_drift.py` | Yıl kayması + hedef-gürültüsü drift analizi (yaklaşımın temeli) |
| `tabpfn_cfg_colab.py` | TabPFN config denemesi — blende katkı yok → **elendi** |
| `autogluon_colab.py` | AutoGluon — **elendi** |
| `pysr_colab.py` | Sembolik regresyon — **elendi** |
| `ridge_test.py` | Ridge meta-stacker probe |

---

## 6. Sıfırdan reprodüksiyon (aynı veriyle)

> Önkoşul: Kaggle'dan veriyi indir → `datathon-2026/train.csv`, `datathon-2026/test_x.csv`.
> GBM ve tablo modelleri lokal CPU'da koşar; metin/TabPFN modelleri Colab GPU ister
> (kod başlıklarında kurulum notları var). Tüm çıktılar `oof/` altına düşer.

```bash
pip install -r requirements.txt
```

**a) GBM base bag'leri** (yıl-ağırlıklı, CPU — CatBoost'ta GPU kalitesi CPU'dan ~0.8 kötü):
```bash
# CatBoost (5 seed CPU bag)
python train.py --name cat_fekw_text       --model cat --fe --kw --text --yw
python train.py --name cat_fekw_text_s1cpu --model cat --fe --kw --text --yw --seed 1
# ... s2cpu/s3cpu/s4cpu aynı, --seed 2/3/4

# CatBoost l2 (cat_l2 + s1..s4)
python train.py --name cat_l2 --model cat --fe --kw --text --yw --l2 9     # seed bag: --seed 1..4

# LightGBM / fe2
python train.py --name lgb_fekw_text  --model lgb --fe  --kw --text --yw    # + s1,s2
python train.py --name lgb_fe2kw_text --model lgb --fe2 --kw --text --yw

# Embedding'li (önce embed.py ile oof/emb_*.npy üret)
python embed.py
python train.py --name cat_fe_text_emb    --model cat --fe  --text --emb --yw
python train.py --name lgb_fekw_text_emb  --model lgb --fe --kw --text --emb --yw
```

**b) Tuned bag'ler:**
```bash
python tune_lgb.py   # -> oof/lgb_tuned_*.npy  (seed bag: KFold seed'i değiştirilerek)
python tune_xgb.py   # -> oof/xgb_tuned_*.npy  (11 seed)
python tune_cat.py   # -> oof/cat_tuned_*.npy
python nn_et.py      # -> oof/mlp_*.npy, oof/et_*.npy
```

**c) LLM-judge tabanlı base'ler** (Opsiyon A — komutlar):
LLM feature'ları önce Colab'da üretilir (`llm_judge_colab.py` → `oof/llm_train.npy`;
`llm_judge2_colab.py` → `oof/llm2_train.npy`). Sonra `--llm` bayrağıyla GBM base'leri:
```bash
python train.py --name catl2_llm --model cat --fe --kw --text --yw --l2 9 --llm
python train.py --name xgbt_llm  --model xgb --fe --kw --text --yw --llm
python train.py --name lgbt_llm  --model lgb --fe --kw --text --yw --llm   # LLM v1 (oof/llm_*.npy)
```
`lgbt_llm2` = `lgbt_llm` ile **birebir aynı**; tek fark `add_llm_features`'ın
`oof/llm_train.npy` yerine **`oof/llm2_train.npy`** (Qwen-32B, v2) okumasıdır.
`lgb_rawtext` ise LGBM'in ham-metin varyantıdır; mimari `train.py`'nin metin yolundadır.
(Bu varyantların ara `.npy` çıktıları ve yarışma verisi paylaşılmadığından, kod bit-bit
re-run için değil **yaklaşımı tam göstermek** için verilmiştir.)

**d) Metin & TabPFN (Colab GPU):** `bert_colab.py`, `bert_bag_colab.py`,
`bert_s3_local.py`, `bert_res.py`, `e5_colab.py`, `fusion_colab.py`,
`tabpfn_colab.py`, `tabpfn_v2_colab.py`, `nightrun_colab.py`, `tabpfn_llm_colab.py` —
her dosyanın başında Colab kurulum/çalıştırma notu vardır.

**e) Final:**
```bash
python blend_final.py        # -> sub20 tabanı (NM blend + kalibrasyon)
# tabpfn_stack6_colab.py     # -> sub19 (Colab; tüm bileşen .npy'leri stack/ zip'inde)
```

---

## 7. Öne çıkanlar (özet)

- **Yıl-ağırlıklı OOF metriği** — public skoru ±0.1 hatayla tahmin etti; bütün kararların
  dayanağı.
- **Çok-aileli ensemble** — GBM (LGBM/XGB/CatBoost) + MLP/ExtraTrees + BERTurk + e5 +
  TabPFN, üstüne TabPFN meta-stacker.
- **LLM-as-judge** — mentor metnini Qwen2.5'e değerlendirici gibi okutup yapılandırılmış
  skor çıkarma; metin kanalındaki tek gerçek kazanç (v1→v2 ile −1.2 → −2.0 MSE base'de).
- **Disiplin** — kalibrasyonu neden yaptığımı/yapmadığımı nested ölçümle gösterdim;
  ~25 hipotez denedim, ikisini (Sonnet LLM-judge, agresif ağırlık-fit) public'in
  çürüttüğünü gördüm ve seçmeden önce eledim.

---

## 8. Bağımlılıklar

```bash
pip install -r requirements.txt
```

`requirements.txt` lokal çekirdeği kapsar (numpy, pandas, scipy, scikit-learn,
matplotlib, lightgbm, xgboost, catboost, tabpfn). Metin/embedding/LLM scriptleri ayrıca
`transformers`, `accelerate`, `sentence-transformers`, `bitsandbytes` ister; bunlar
ilgili Colab dosyalarının başında belirtilir.
