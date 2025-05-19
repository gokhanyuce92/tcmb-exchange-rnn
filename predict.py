import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

window_size = 30

model = load_model('lstm_model.keras')

# Scaler'ı aynı şekilde yeniden oluştur ve fit et
kur_df = pd.read_csv('usd_alis_kurlari.csv', header=0)
kur = kur_df.iloc[:, 1].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaler.fit(kur)  # Eğitimdekiyle aynı şekilde fit edilmeli

# Son 30 günün verisini hazırla
son_30_gun = kur[-window_size:]

son_tarih = pd.to_datetime(kur_df.iloc[-1, 0], dayfirst=True)
hedef_tarih = pd.to_datetime("20-05-2025", dayfirst=True)
tahmin_gun_sayisi = (hedef_tarih - son_tarih).days

tahminler = []
for i in range(tahmin_gun_sayisi):
    scaled_input = scaler.transform(son_30_gun).reshape(1, window_size, 1)
    tahmin_scaled = model.predict(scaled_input, verbose=0)
    tahmin = scaler.inverse_transform(tahmin_scaled)
    tahminler.append(tahmin[0, 0])
    # Yeni tahmini son 30 günün sonuna ekle, ilkini çıkar
    son_30_gun = np.vstack([son_30_gun[1:], [[tahmin[0, 0]]]])

print(f"20.05.2025 için tahmin edilen kur: {tahminler[-1]}")