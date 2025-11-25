import shap
import numpy as np
from train import train_model
from preprocess import preprocess_data

def run_shap(model_type="lstm"):
    model = train_model(model_type)
    X_train, y_train, X_test, y_test, _ = preprocess_data()

    explainer = shap.DeepExplainer(model, X_train[:100])
    shap_values = explainer.shap_values(X_test[:5])

    print("SHAP values generated for top 5 predictions.")
    shap.summary_plot(shap_values, X_test[:5])

if __name__ == "__main__":
    run_shap("lstm")
