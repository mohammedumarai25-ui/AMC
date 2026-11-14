import pandas as pd
import os
import re


def load_transactions():

    project_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    file_path = os.path.join(
        project_dir, "data", "transactions.csv"
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}"
        )

    # Load CSV
    df = pd.read_csv(
        file_path,
        encoding="utf-8-sig",
        sep=None,
        engine="python"
    )

    # Clean column names
    df.columns = [
        re.sub(
            r"[^a-z0-9]+",
            "_",
            str(col).strip().lower()
        ).strip("_")
        for col in df.columns
    ]

    # Required columns
    required_columns = [
        "transaction_id",
        "sender_account",
        "receiver_account",
        "amount",
        "timestamp",
        "transaction_type",
        "sender_country",
        "receiver_country",
        "is_suspicious",
        "suspicious_pattern"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        print("\nACTUAL CSV COLUMNS:")
        print(df.columns.tolist())

        print("\nMISSING COLUMNS:")
        print(missing_columns)

        raise ValueError(
            "Your CSV does not contain the required AML columns. "
            "Check that you placed the correct transactions.csv file "
            "inside the data folder."
        )

    return df


def inspect_transactions(df):

    print("\n========== AML DATASET INFORMATION ==========")

    print("\nFirst 5 Transactions:")
    print(df.head())

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Transaction IDs:")
    print(df["transaction_id"].duplicated().sum())

    print("\nTransaction Labels:")
    print(df["is_suspicious"].value_counts())

    print("\nSuspicious Patterns:")
    print(df["suspicious_pattern"].value_counts())

    print("\nTotal Transactions:", len(df))

    print("\n========== INSPECTION COMPLETED ==========")


if __name__ == "__main__":

    transactions = load_transactions()

    inspect_transactions(transactions)