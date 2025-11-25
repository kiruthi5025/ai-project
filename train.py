from preprocess import preprocess_data
from model_lstm import build_lstm
from model_tcn import build_tcn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_model(model_type="lstm"):
    X_train, y_train, X_test, y_test, _ = preprocess_data()

    input_shape = (X_train.shape[1], X_train.shape[2])

    model = build_lstm(input_shape) if model_type == "lstm" else build_tcn(input_shape)

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)

    print(f"Model: {model_type.upper()} RMSE = {rmse:.4f}, MAE = {mae:.4f}")

    return model

if __name__ == "__main__":
    train_model("lstm")
    train_model("tcn")
