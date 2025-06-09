from sklearn.model_selection import TimeSeriesSplit
import numpy as np

# Örnek veri (10 elemanlı bir dizi)
data = np.arange(10)

tscv = TimeSeriesSplit(n_splits=3)

for fold, (train_index, test_index) in enumerate(tscv.split(data)):
    print(f"Fold {fold+1}")
    print("Train indices:", train_index)
    print("Test indices:", test_index)
    print("Train data:", data[train_index])
    print("Test data:", data[test_index])
    print("-" * 30)