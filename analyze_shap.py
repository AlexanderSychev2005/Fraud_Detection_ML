from pathlib import Path
from typing import Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import shap

DATA_PROCESSED_DIR = Path("dataset/processed")


def load_data_and_model() -> Tuple[lgb.Booster, pd.DataFrame]:
    """Loading the model and the data for SNAP algorithm"""
    print("The model and data loading")

    model_path = DATA_PROCESSED_DIR / "lgb_model.txt"
    model = lgb.Booster(model_file=model_path)

    df = pd.read_csv(DATA_PROCESSED_DIR / "train_full.csv")

    cols_to_drop = ["id_user", "timestamp_reg", "email", "is_fraud"]
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    cat_cols = [
        "gender",
        "reg_country",
        "traffic_type",
        "card_type",
        "transaction_type",
    ]
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype("category")

    return model, X


def generate_shap_plot() -> None:
    """
    Calculates SHAP values for a sample of the data and generates
    a summary plot for the top 5 features.
    """
    model, X = load_data_and_model()

    print("Calculating the SNAP values")
    X_sample = X.sample(n=10000, random_state=42)  # Take sample of 10,000 lines

    # Explainer initialization
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values

    print("Plotting...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values_to_plot, X_sample, max_display=5, show=False
    )  # Get top 5 features

    plt.tight_layout()
    shap_path = DATA_PROCESSED_DIR / "shap_top5_features.png"
    plt.savefig(shap_path, bbox_inches="tight", dpi=300)
    print(f"Plot is saved into: {shap_path}")


if __name__ == "__main__":
    generate_shap_plot()
