from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree

DATA_PROCESSED_DIR = Path("dataset/processed")


def build_surrogate_tree():
    print("Loading data...")
    df = pd.read_csv(DATA_PROCESSED_DIR / "train_full.csv")

    features = [
        "unique_cards",
        "unique_error_groups",
        "total_geo_mismatch",
        "fail_count",
        "mins_to_first_trans",
    ]

    X = df[features]

    X = X.fillna(0)
    y = df["is_fraud"]

    print("Surrogate tree training...")
    tree_model = DecisionTreeClassifier(
        max_depth=3, class_weight="balanced", random_state=42
    )
    tree_model.fit(X, y)

    print("Plotting tree...")
    plt.figure(figsize=(24, 12))

    plot_tree(
        tree_model,
        feature_names=features,
        class_names=["Good User", "FRAUD"],
        filled=True,
        rounded=True,
        fontsize=14,
        proportion=False,
    )
    tree_path = DATA_PROCESSED_DIR / "surrogate_tree_rules.png"
    plt.savefig(tree_path, bbox_inches="tight", dpi=300)
    print(f"Block if else schema saved in: {tree_path}")


if __name__ == "__main__":
    build_surrogate_tree()
