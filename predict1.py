import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

window_size = 24

model = load_model('lstm_model.keras')

# Scaler'ı aynı şekilde yeniden oluştur ve fit et
kur_df = pd.read_csv('usd_alis_kurlari.csv', header=0)
kur = kur_df.iloc[:, 1].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaler.fit(kur)  # Eğitimdekiyle aynı şekilde fit edilmeli

# Son window_size günün verisini hazırla
son_size_gun = kur[-window_size:]
son_size_gun_scaled = scaler.transform(son_size_gun).reshape(1, window_size, 1)

tahmin_scaled = model.predict(son_size_gun_scaled, verbose=0)
tahmin = scaler.inverse_transform(tahmin_scaled)

print(f"Bir sonraki gün için tahmin edilen kur: {tahmin[0, 0]}")