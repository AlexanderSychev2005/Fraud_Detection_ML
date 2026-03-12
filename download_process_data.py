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


def main():
    train_users, test_users, train_trans, test_trans = load_data()

    train_transactions_agg = aggregate_transactions(train_trans)
    test_transactions_agg = aggregate_transactions(test_trans)

    print("Tables merging...")
    # LEFT JOIN because we don't want to lose the users without transactions
    train_full = train_users.merge(train_transactions_agg, on="id_user", how="left")
    test_full = test_users.merge(test_transactions_agg, on="id_user", how="left")

    # Fill the gaps (NA) with 0 for those who didn't have transactions
    cols_to_fill = train_transactions_agg.columns.drop("id_user")
    train_full[cols_to_fill] = train_full[cols_to_fill].fillna(0)
    test_full[cols_to_fill] = test_full[cols_to_fill].fillna(0)

    train_full.to_csv(DATA_PROCESSED_DIR / "train_full.csv", index=False)
    test_full.to_csv(DATA_PROCESSED_DIR / "test_full.csv", index=False)
    print(f"Processed files are finally saved into {DATA_PROCESSED_DIR}")


if __name__ == "__main__":
    main()
