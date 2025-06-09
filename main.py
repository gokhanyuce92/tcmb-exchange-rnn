import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
import optuna
import numpy as np
import matplotlib.pyplot as plt
import math
import plotly.graph_objects as go

TRAIN_SIZE = 0.80

def create_dataset(dataset, size=30):
    data_X, data_Y = [], []
    for i in range(len(dataset) - size):
        a = dataset[i:(i + size), 0]
        data_X.append(a.reshape(-1, 1)) 
        data_Y.append(dataset[i + size, 0])
    return np.array(data_X), np.array(data_Y)

def objective(trial):
    window_size = trial.suggest_int('window_size', 8, 15)
    train_X, train_Y = create_dataset(train, window_size)
    test_X, test_Y = create_dataset(test, window_size)

    n_units = trial.suggest_int('n_units', 10, 15)
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)

    model = Sequential([
        LSTM(n_units,
             input_shape=(window_size, 1)
             ),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    model.fit(
        train_X, train_Y,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
        )
    loss = model.evaluate(test_X, test_Y, verbose=0)
    return loss

def predict_and_score(model, X, Y):
    # Şimdi tahminleri 0-1 ile scale edilmiş halinden geri çeviriyoruz.
    pred = scaler.inverse_transform(model.predict(X))
    orig_data = scaler.inverse_transform([Y])
    # Rmse değerlerini ölçüyoruz.
    score = math.sqrt(mean_squared_error(orig_data[0], pred[:, 0]))
    return(score, pred)

# 1. Veri çekme (CSV'den oku)
usd_alis_df = pd.read_csv('usd_alis_kurlari.csv')
print(usd_alis_df.shape)
kur = usd_alis_df['TP_DK_USD_A_YTL'].values.reshape(-1, 1)

# 2. Normalizasyon
scaler = MinMaxScaler()
dataset = scaler.fit_transform(kur)

# 3. Eğitim/test ayır
train_size = int(len(dataset) * TRAIN_SIZE)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print("Gün Sayıları (training set, test set): " + str((len(train), len(test))))

# 4. Model hiperparametre optimizasyonu
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial), n_trials=10)
best_params = study.best_params
print("Best parameters: ", best_params)

# trials = study.trials_dataframe()
# fig = go.Figure()
# fig.add_trace(go.Scatter(
#     x=trials.index,
#     y=trials['value'],
#     mode='lines+markers',
#     name='Validation Loss'
# ))
# fig.update_layout(
#     title='Optuna Trials - Validation Loss',
#     xaxis_title='Trial',
#     yaxis_title='Validation Loss (val_loss)',
#     template='plotly_white'
# )
# fig.show()

final_window_size = best_params['window_size']
# Verisetlerimizi Oluşturalım
train_X, train_Y = create_dataset(train, final_window_size)
test_X, test_Y = create_dataset(test, final_window_size)
print("Original training data shape: ", train_X.shape)

# 5. Final model
tscv = TimeSeriesSplit(n_splits=3)
for fold, (train_idx, val_idx) in enumerate(tscv.split(train_X)):
    print(f"Fold {fold+1}")
    X_train, X_val = train_X[train_idx], train_X[val_idx]
    Y_train, Y_val = train_Y[train_idx], train_Y[val_idx]

    model = Sequential([
        LSTM(best_params['n_units'],
            input_shape=(final_window_size, 1)
            ),
        Dropout(best_params['dropout']),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=best_params['lr']), loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(
        X_train, Y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_val, Y_val),
        callbacks=[early_stop],
        verbose=1
    )

    val_loss = model.evaluate(X_val, Y_val, verbose=0)
    print(f"Validation loss for fold {fold+1}: {val_loss}")

# 6. Tahmin ve görselleştirme
rmse_train, train_predict = predict_and_score(model, train_X, train_Y)
rmse_test, test_predict = predict_and_score(model, test_X, test_Y)

print("Training data score: %.2f RMSE" % rmse_train)
print("Test data score: %.2f RMSE" % rmse_test)
# Training data score: 0.07 RMSE
# Test data score: 0.67 RMSE

model.save('lstm_model.keras')