# TCMB USD Alış Kuru Zaman Serisi Tahmini (LSTM)

Bu proje, Türkiye Cumhuriyet Merkez Bankası (TCMB) EVDS API üzerinden alınan USD alış kuru verileriyle, LSTM tabanlı bir derin öğrenme modeli kullanarak döviz kuru tahmini yapmayı amaçlamaktadır.

## İçerik

- Veri çekme ve ön işleme (eksik günlerin doldurulması)
- Zaman serisi için pencereleme (windowing)
- LSTM mimarisi ile modelleme
- Hiperparametre optimizasyonu (Optuna)
- Eğitim, test ve ileriye dönük tahmin
- Sonuçların görselleştirilmesi

## Kurulum

1. **Gereksinimler**

   - Python 3.11
   - Gerekli kütüphaneler:
     ```
     python -m venv venv
     pip install --upgrade pip
     pip install -r requirements.txt
     .\venv\Scripts\activate
     ```

2. **TCMB EVDS API Anahtarı**
   - [EVDS](https://evds2.tcmb.gov.tr/) üzerinden ücretsiz API anahtarı alın.
   - Anahtarı `tcmb_evds_api_key.txt` dosyasına kaydedin.

## Kullanım

1. **Veri Çekme**

   - `evds_to_csv.py` dosyasını çalıştırarak 2000-2025 arası USD alış kuru verisi çekilir ve servisten dönen veri sistem tarafından usd_alis_kurlari.csv adında dosya oluşturulur ve kaydedilir.
     ```
     python evds_to_csv.py
     ```

2. **Model Eğitimi ve Tahmin**

   - `main.py` dosyasını çalıştırarak modeli eğitilir:
     Model eğitildikten sonra sistem lstm_model.keras adında dosya oluşturur ve kayıt yapar.
     ```
     python main.py
     ```

3. **Geleceğe Yönelik Tahmin**
   - `predict.py` dosyası ile ileri tarihli tahminler yapabilirsiniz.

## Dosya Açıklamaları

- `evds_to_csv.py` : TCMB EVDS API'den veri çeker ve CSV'ye kaydeder.
- `main.py` : Veriyi işler, modeli eğitir, Optuna ile hiperparametre optimizasyonu yapar ve lstm_model.keras dosyaya modeli kaydeder.
- `predict.py` : Eğitilmiş model ile ileriye dönük tahminler yapar.
- `usd_alis_kurlari.csv` : Çekilen ve doldurulan USD alış kuru verisi.

## Notlar

- Eksik günler bir önceki iş gününün değeriyle doldurulmuştur (forward fill).
- Modelin başarımı RMSE metriği ile değerlendirilmiştir.
- Hiperparametre optimizasyonu için Optuna kullanılmıştır.

---

Her türlü katkı ve öneriye açıktır!
