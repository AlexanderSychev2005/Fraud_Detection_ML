import warnings
from pathlib import Path
from typing import Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")


DATA_PROCESSED_DIR = Path("dataset/processed")


def prepare_data_for_lgb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the dataframe for LightGBM by dropping unused columns and casting categorical features to 'category' dtype.
    """
    # Drop identifiers (id and email) and datetime columns that the model cannot process directly
    cols_to_drop = ["id_user", "timestamp_reg", "email"]
    df_model = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Specify categorical columns for native LightGBM handling
    cat_cols = [
        "gender",
        "reg_country",
        "traffic_type",
        "card_type",
        "transaction_type",
    ]
    for col in cat_cols:
        if col in df_model.columns:
            df_model[col] = df_model[col].astype(
                "category"
            )  # Pandas category type, LightGBM does not need one-hot encoding.

    return df_model


def find_best_threshold(y_true: pd.Series, y_prob: np.array) -> Tuple[float, float]:
    """
    Iterates over possible threshold values to find the one
    that maximizes the F1-score.
    """
    best_thresh = 0.5
    best_f1 = 0.0

    # Check thresholds from 0.1 to 0.9 with a step of 0.02
    thresholds = np.arange(0.1, 0.9, 0.02)
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return best_thresh, best_f1


def train_lightgbm() -> Tuple[lgb.Booster, pd.DataFrame]:
    """
    Trains a LightGBM model using Stratified K-Fold cross-validation
    and generates Out-Of-Fold (OOF) predictions and test predictions.
    """
    print("Loading processed data...")
    train_df = pd.read_csv(DATA_PROCESSED_DIR / "train_full.csv")
    test_df = pd.read_csv(DATA_PROCESSED_DIR / "test_full.csv")

    train_model = prepare_data_for_lgb(train_df)
    test_model = prepare_data_for_lgb(test_df)

    X = train_model.drop(columns=["is_fraud"])
    y = train_model["is_fraud"]
    X_test = test_model.drop(columns=["is_fraud"], errors="ignore")

    # Calculate class imbalance ratio to penalize missed frauds
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    print(f"Weight scale: {scale_pos_weight}")

    # LightGBM settings
    params = {
        "objective": "binary",  # fraud or not fraud
        "metric": "auc",  # ROC-AUC
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "scale_pos_weight": scale_pos_weight,  #
        "random_state": 42,
        "verbose": -1,
    }

    print("Starting Stratified K-Fold Cross-Validation (5 folds)...")
    skf = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    )  # Stratified ensures we'll have 3.78% fraudsters, the model learns on 4 parts and we test it on the 5th part measuring an error.
    # The process repeats 5 times. The predictions on that parts are called Out-Of-Fold (OOF) predictions. During the process, each fold should be the valid one.
    # At the end, we average the results between 5 models. This makes our solution incredibly resilient to emissions.

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
        X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

        # Create LightGBM Datasets
        train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
        val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

        # Setup Early Stopping callback
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False)
        ]  # The model calculates AUC on the test fold after building each tree. If the metric does not improve during 50 steps, stop the training. Protects from overfitting.

        # Train the model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Predict on validation fold
        val_prob = model.predict(X_val_fold)
        oof_preds[val_idx] = val_prob

        # Accumulate predictions on the test set
        test_preds += model.predict(X_test) / skf.n_splits
        print(f"Fold {fold + 1} completed.")

    # Basically, the model returns the probabilities (from 0.0 to 1.0). The default threshold equals to 0.5. However, we have to maximise the F1-score.
    print("\nOptimizing threshold for F1-score...")
    best_thresh, best_f1 = find_best_threshold(y, oof_preds)

    final_preds = (oof_preds >= best_thresh).astype(int)
    print(f"Best Threshold: {best_thresh:.2f}")
    print(f"OOF F1-score: {best_f1:.4f}")
    print(f"Precision: {precision_score(y, final_preds):.4f}")
    print(f"Recall: {recall_score(y, final_preds):.4f}")

    print("\nGenerating results file...")
    results = pd.DataFrame(
        {
            "id_user": test_df["id_user"],
            "is_fraud": (test_preds >= best_thresh).astype(int),
        }
    )

    results_path = DATA_PROCESSED_DIR / "results.csv"
    results.to_csv(results_path, index=False)
    print(f"File saved successfully to: {results_path}")

    return model, X


if __name__ == "__main__":
    model, X_train = train_lightgbm()

    print("Model saving...")
    model_path = DATA_PROCESSED_DIR / "lgb_model.txt"
    model.save_model(model_path)

    print(f"Model is saved into: {model_path}")
