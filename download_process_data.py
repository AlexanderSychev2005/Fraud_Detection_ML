from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

DATA_RAW_DIR = Path("dataset/raw")
DATA_PROCESSED_DIR = Path("dataset/processed")

DATA_PROCESSED_DIR.mkdir(
    parents=True, exist_ok=True
)  # Make sure the folder for the processed data exists


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loading data from csv files"""
    print("Data Loading...")
    train_users = pd.read_csv(
        DATA_RAW_DIR / "train_users.csv", parse_dates=["timestamp_reg"]
    )
    test_users = pd.read_csv(
        DATA_RAW_DIR / "test_users.csv", parse_dates=["timestamp_reg"]
    )

    train_transactions = pd.read_csv(
        DATA_RAW_DIR / "train_transactions.csv", parse_dates=["timestamp_tr"]
    )
    test_transactions = pd.read_csv(
        DATA_RAW_DIR / "test_transactions.csv", parse_dates=["timestamp_tr"]
    )
    print("All data is loaded!")

    return train_users, test_users, train_transactions, test_transactions


def aggregate_transactions(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Performs aggregation by users"""
    print("Transactions aggregation...")

    transactions_df["is_fail"] = (transactions_df["status"] == "fail").astype(int)
    transactions_df["is_success"] = (transactions_df["status"] == "success").astype(int)

    agg_df = (
        transactions_df.groupby("id_user")
        .agg(
            trans_count=("id_user", "count"),
            total_amount=("amount", "sum"),  # Sum of transactions
            mean_amount=("amount", "mean"),
            fail_count=("is_fail", "sum"),
            success_count=("is_success", "sum"),
            unique_cards=("card_mask_hash", "nunique"),  # Different cards
            unique_pay_countries=("payment_country", "nunique"),  # Different countries
            unique_error_groups=("error_group", "nunique"),  # Different errors
        )
        .reset_index()
    )

    agg_df["success_rate"] = agg_df["success_count"] / np.maximum(
        agg_df["trans_count"], 1
    )

    return agg_df


def extract_advanced_features(
    users_df: pd.DataFrame, transactions_df: pd.DataFrame
) -> pd.DataFrame:
    """Extracts complex features requiring both users and transactions data"""

    print("Advanced features extraction...")

    u_df = users_df.copy()
    t_df = transactions_df.copy()

    t_df["timestamp_tr"] = pd.to_datetime(t_df["timestamp_tr"], format="ISO8601")
    u_df["timestamp_reg"] = pd.to_datetime(u_df["timestamp_reg"], format="ISO8601")

    # 1. Time features
    time_agg = (
        t_df.groupby("id_user")
        .agg(
            first_trans_time=("timestamp_tr", "min"),
            last_trans_time=("timestamp_tr", "max"),
        )
        .reset_index()
    )

    time_agg = time_agg.merge(
        u_df[["id_user", "timestamp_reg"]], on="id_user", how="left"
    )
    time_agg["mins_to_first_trans"] = (
        time_agg["first_trans_time"] - time_agg["timestamp_reg"]
    ).dt.total_seconds() / 60.0

    time_agg["activity_duration_mins"] = (
        time_agg["last_trans_time"] - time_agg["first_trans_time"]
    ).dt.total_seconds() / 60.0

    # 2. Geo & Category features (based on the first transaction attempt)
    first_trans = transactions_df.sort_values(
        ["id_user", "timestamp_tr"]
    ).drop_duplicates(subset=["id_user"], keep="first")

    geo_features = first_trans[
        ["id_user", "card_country", "payment_country", "card_type", "transaction_type"]
    ].copy()
    geo_features = geo_features.merge(
        users_df[["id_user", "reg_country"]], on="id_user", how="left"
    )

    geo_features["match_reg_pay"] = (
        geo_features["reg_country"] == geo_features["payment_country"]
    ).astype(int)
    geo_features["match_reg_card"] = (
        geo_features["reg_country"] == geo_features["card_country"]
    ).astype(int)
    geo_features["match_pay_card"] = (
        geo_features["payment_country"] == geo_features["card_country"]
    ).astype(int)
    geo_features["total_geo_mismatch"] = (
        (geo_features["match_reg_pay"] == 0)
        & (geo_features["match_reg_card"] == 0)
        & (geo_features["match_pay_card"] == 0)
    ).astype(int)

    # 3. Combine advanced features
    advanced_features = time_agg[
        ["id_user", "mins_to_first_trans", "activity_duration_mins"]
    ].merge(
        geo_features[
            [
                "id_user",
                "match_reg_pay",
                "match_reg_card",
                "match_pay_card",
                "total_geo_mismatch",
                "card_type",
                "transaction_type",
            ]
        ],
        on="id_user",
        how="left",
    )

    return advanced_features


def main():
    train_users, test_users, train_trans, test_trans = load_data()

    train_transactions_agg = aggregate_transactions(train_trans)
    test_transactions_agg = aggregate_transactions(test_trans)

    train_adv = extract_advanced_features(train_users, train_trans)
    test_adv = extract_advanced_features(test_users, test_trans)

    print("Tables merging...")
    # LEFT JOIN because we don't want to lose the users without transactions
    train_full = train_users.merge(
        train_transactions_agg, on="id_user", how="left"
    ).merge(train_adv, on="id_user", how="left")
    test_full = test_users.merge(test_transactions_agg, on="id_user", how="left").merge(
        test_adv, on="id_user", how="left"
    )

    # Fill NA properly: 0 for numerics, "unknown" for categoricals
    numeric_cols = train_transactions_agg.columns.drop("id_user").tolist() + [
        "mins_to_first_trans",
        "activity_duration_mins",
        "match_reg_pay",
        "match_reg_card",
        "match_pay_card",
        "total_geo_mismatch",
    ]

    train_full[numeric_cols] = train_full[numeric_cols].fillna(0)
    test_full[numeric_cols] = test_full[numeric_cols].fillna(0)

    cat_cols = ["card_type", "transaction_type"]
    train_full[cat_cols] = train_full[cat_cols].fillna("unknown")
    test_full[cat_cols] = test_full[cat_cols].fillna("unknown")

    train_full.to_csv(DATA_PROCESSED_DIR / "train_full.csv", index=False)
    test_full.to_csv(DATA_PROCESSED_DIR / "test_full.csv", index=False)
    print(f"Processed files are finally saved into {DATA_PROCESSED_DIR}")


if __name__ == "__main__":
    main()
