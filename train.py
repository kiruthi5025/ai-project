import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import numpy as np

def create_lstm_model(trial, input_shape):
    model = Sequential()
    units = trial.suggest_int("units", 32, 128)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)

    model.add(LSTM(units, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

def objective(trial):
    global X_train, X_val, y_train, y_val
    model = create_lstm_model(trial, (X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=8, batch_size=32, verbose=0)
    pred = model.predict(X_val)
    return mean_squared_error(y_val, pred)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

print("Best Params:", study.best_params)

with open("hyperparam_results.txt", "w") as f:
    f.write(str(study.best_params))
