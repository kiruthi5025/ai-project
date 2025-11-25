import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def create_windows(data, window=30):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window, :-1])
        y.append(data[i+window, -1])
    return np.array(X), np.array(y)

def preprocess_data(csv_path="data.csv", window=30):
    df = pd.read_csv(csv_path)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = create_windows(scaled, window)

    split = int(0.8 * len(X))
    return X[:split], y[:split], X[split:], y[split:], scaler

if __name__ == "__main__":
    preprocess_data()
