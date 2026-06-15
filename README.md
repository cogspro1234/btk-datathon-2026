# BTK Datathon 2026 — Öğrenci Kariyer Başarısı Tahmini

Bu repo, BTK Datathon 2026 yarışmasındaki çözümümü içerir. Görev: öğrencilerin
`career_success_score` değerini (0–100, sürekli) tahmin etmek. Metrik: MSE.

## Yaklaşım

Çözümün tek ilkesi: public leaderboard'a değil, kendi yıl-ağırlıklı doğrulama
metriğime güvenmek. İlk denememde CV ile public arasında 11 MSE fark gördüm; sebebi
test verisinin son yıllara kayması ve hedef gürültüsünün yılla artmasıydı. Test
dağılımını yansıtan yıl-ağırlıklı bir OOF metriği kurdum; bu metrik public skoru ±0.1
hatayla tahmin etti ve bütün kararları onunla verdim.

Ayrıntılı anlatım ve grafikler için: **[`Datathon2026_Cozum.ipynb`](Datathon2026_Cozum.ipynb)**

## Öne çıkanlar

- **Yıl-ağırlıklı OOF metriği** — her satırı test/train yıl oranıyla ağırlıklandırma;
  public skoru ±0.1 hatayla tahmin etti (örn. 91.12 tahmin / 91.119 gerçek).
- **Çok-aileli ensemble** — seed-bag LightGBM / XGBoost / CatBoost + BERT-tr / e5 +
  TabPFN, üstüne TabPFN meta-stacker.
- **LLM-as-judge** — mentor metnini bir LLM'e değerlendirici gibi okutup yapılandırılmış
  skor çıkarma; metindeki tek gerçek kazanç.
- **Disiplin** — kalibrasyonu neden yapmadığımı grafiklerle gösterdim; ~25 hipotez
  denedim, ikisinin leakage olduğunu nested validation ile yakaladım.

## Dosyalar

| Dosya | İçerik |
|---|---|
| `Datathon2026_Cozum.ipynb` | Ana çözüm: EDA, yöntem, sonuç (çıktılarıyla) |
| `train.py` | Feature engineering + base modeller |
| `blend_final.py` | Final Nelder-Mead blend + yıl-bazlı kalibrasyon |
| `llm_judge3.py` | LLM-as-judge skorlama |
| `requirements.txt` | Bağımlılıklar |

## Çalıştırma notu

Yarışma verisi (`datathon-2026/`) ve eğitilmiş OOF artifact'leri (`oof/`) lisans ve boyut
nedeniyle repoda yer almaz. Base model eğitimleri ile metin embedding'leri yerel ve
Colab ortamında üretildi; notebook bu çıktıları gömülü hâlde içerir, olduğu gibi
okunabilir.

```bash
pip install -r requirements.txt
```
