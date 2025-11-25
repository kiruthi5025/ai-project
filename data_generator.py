import numpy as np
import pandas as pd

def generate_multivariate_timeseries(n_samples=1500, n_features=5):
    time = np.arange(n_samples)

    data = {}
    for i in range(n_features):
        data[f"feature_{i+1}"] = (
            np.sin(0.02 * time + i) +
            np.cos(0.01 * time) +
            np.random.normal(0, 0.1, n_samples)
        )

    df = pd.DataFrame(data)
    df["target"] = (
        df["feature_1"] * 0.4 +
        df["feature_2"] * 0.3 +
        df["feature_3"] * 0.2 +
        np.random.normal(0, 0.05, n_samples)
    )
    return df

if __name__ == "__main__":
    df = generate_multivariate_timeseries()
    df.to_csv("data.csv", index=False)
    print("Data generated and saved as data.csv")
