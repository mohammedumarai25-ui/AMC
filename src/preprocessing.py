import pandas as pd
import os
from data_loader import load_transactions


def preprocess_transactions(df):

    print("\n========== DATA PREPROCESSING ==========")

    initial_count = len(df)

    # 1. Remove duplicate transactions
    df = df.drop_duplicates(
        subset=["transaction_id"]
    ).copy()

    # 2. Remove rows with missing critical values
    critical_columns = [
        "transaction_id",
        "sender_account",
        "receiver_account",
        "amount",
        "timestamp",
        "is_suspicious"
    ]

    df = df.dropna(
        subset=critical_columns
    ).copy()

    # 3. Clean account IDs
    df["sender_account"] = (
        df["sender_account"].astype(str).str.strip().str.upper()
    )

    df["receiver_account"] = (
        df["receiver_account"].astype(str).str.strip().str.upper()
    )

    # 4. Convert amount into numeric format
    df["amount"] = pd.to_numeric(
        df["amount"], errors="coerce"
    )

    df = df.dropna(subset=["amount"]).copy()

    df = df[df["amount"] > 0].copy()

    # 5. Convert timestamp
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )

    df = df.dropna(subset=["timestamp"]).copy()

    # 6. Clean suspicious labels
    df["is_suspicious"] = pd.to_numeric(
        df["is_suspicious"], errors="coerce"
    )

    df = df.dropna(
        subset=["is_suspicious"]
    ).copy()

    df = df[
        df["is_suspicious"].isin([0, 1])
    ].copy()

    df["is_suspicious"] = (
        df["is_suspicious"].astype(int)
    )

    # 7. Clean text columns
    text_columns = [
        "transaction_type",
        "sender_country",
        "receiver_country",
        "suspicious_pattern"
    ]

    for col in text_columns:
        df[col] = (
            df[col].fillna("UNKNOWN").astype(str).str.strip().str.upper()
        )

    # 8. Sort transactions by time
    df = df.sort_values(
        by="timestamp"
    ).reset_index(drop=True)

    print("\nOriginal Transactions:", initial_count)
    print("Clean Transactions:", len(df))
    print("Removed Transactions:", initial_count - len(df))

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nSuspicious Transactions:")
    print(df["is_suspicious"].value_counts())

    print("\nTransaction Patterns:")
    print(df["suspicious_pattern"].value_counts())

    print("\nPreprocessing Completed!")

    return df


def save_processed_data(df):

    project_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    output_path = os.path.join(
        project_dir,
        "data",
        "processed_transactions.csv"
    )

    df.to_csv(output_path, index=False)

    print("\nProcessed Dataset Saved At:")
    print(output_path)


if __name__ == "__main__":

    transactions = load_transactions()

    processed_data = preprocess_transactions(
        transactions
    )

    save_processed_data(processed_data)