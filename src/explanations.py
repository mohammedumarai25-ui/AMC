
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

PROCESSED_FILE = DATA_DIR / "processed_transactions.csv"
ALERTS_FILE = DATA_DIR / "detected_alerts.csv"
RISK_FILE = DATA_DIR / "risk_scores.csv"
CLUSTER_FILE = DATA_DIR / "account_cluster_map.csv"

OUTPUT_FILE = DATA_DIR / "explainable_cases.csv"


def load_file(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def find_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def generate_explanations():

    transactions = load_file(PROCESSED_FILE)
    alerts = load_file(ALERTS_FILE)
    risks = load_file(RISK_FILE)
    clusters = load_file(CLUSTER_FILE)

    if transactions.empty:
        print("ERROR: Processed transactions file is missing or empty.")
        return

    sender_col = find_column(
        transactions,
        ["sender_account", "sender"]
    )

    receiver_col = find_column(
        transactions,
        ["receiver_account", "receiver"]
    )

    if sender_col is None or receiver_col is None:
        print("ERROR: Sender or receiver column not found.")
        return

    account_transactions = {}

    for _, row in transactions.iterrows():

        sender = str(row[sender_col])
        receiver = str(row[receiver_col])

        account_transactions.setdefault(sender, []).append(row)
        account_transactions.setdefault(receiver, []).append(row)

    risk_account_col = find_column(
        risks,
        ["account_id", "account", "account_number"]
    )

    risk_score_col = find_column(
        risks,
        ["risk_score", "score"]
    )

    risk_level_col = find_column(
        risks,
        ["risk_level", "risk_category"]
    )

    alert_account_col = find_column(
        alerts,
        ["account_id", "account", "sender_account"]
    )

    alert_pattern_col = find_column(
        alerts,
        ["pattern_type", "pattern", "suspicious_pattern"]
    )

    cluster_account_col = find_column(
        clusters,
        ["account_id", "account", "account_number"]
    )

    cluster_id_col = find_column(
        clusters,
        ["cluster_id", "cluster"]
    )

    risk_lookup = {}

    if risk_account_col and not risks.empty:
        for _, row in risks.iterrows():
            risk_lookup[str(row[risk_account_col])] = row

    cluster_lookup = {}

    if cluster_account_col and cluster_id_col and not clusters.empty:
        for _, row in clusters.iterrows():
            cluster_lookup[str(row[cluster_account_col])] = row[cluster_id_col]

    alert_lookup = {}

    if alert_account_col and not alerts.empty:
        for _, row in alerts.iterrows():

            account = str(row[alert_account_col])

            pattern = (
                str(row[alert_pattern_col])
                if alert_pattern_col
                else "Suspicious activity"
            )

            alert_lookup.setdefault(account, []).append(pattern)

    cases = []

    for account, account_rows in account_transactions.items():

        unique_counterparties = set()
        total_amount = 0
        transaction_count = 0

        for row in account_rows:

            sender = str(row[sender_col])
            receiver = str(row[receiver_col])

            if sender == account:
                unique_counterparties.add(receiver)

            if receiver == account:
                unique_counterparties.add(sender)

            try:
                total_amount += float(row["amount"])
            except (ValueError, TypeError, KeyError):
                pass

            transaction_count += 1

        risk_row = risk_lookup.get(account)

        if risk_row is not None:

            risk_score = (
                risk_row[risk_score_col]
                if risk_score_col
                else 0
            )

            risk_level = (
                risk_row[risk_level_col]
                if risk_level_col
                else "UNKNOWN"
            )

        else:
            risk_score = 0
            risk_level = "UNKNOWN"

        patterns = alert_lookup.get(account, [])
        patterns = sorted(set(patterns))

        cluster_id = cluster_lookup.get(account, "UNKNOWN")

        reasons = []

        if patterns:
            reasons.append(
                "Detected patterns: " + ", ".join(patterns)
            )

        if len(unique_counterparties) >= 5:
            reasons.append(
                "Connected to multiple unique counterparties"
            )

        if transaction_count >= 10:
            reasons.append(
                "High transaction activity"
            )

        if not reasons:
            reasons.append(
                "No specific alert reason recorded; review underlying activity if needed"
            )

        explanation = " | ".join(reasons)

        cases.append({
            "account_id": account,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "cluster_id": cluster_id,
            "transaction_count": transaction_count,
            "unique_counterparties": len(unique_counterparties),
            "total_transaction_amount": round(total_amount, 2),
            "detected_patterns": ", ".join(patterns),
            "explanation": explanation
        })

    result = pd.DataFrame(cases)

    result = result.sort_values(
        by="risk_score",
        ascending=False
    )

    result.to_csv(OUTPUT_FILE, index=False)

    print("\nEXPLAINABLE AML CASES GENERATED")
    print("--------------------------------")
    print("Total accounts:", len(result))
    print("Output:", OUTPUT_FILE)

    print("\nTop 10 accounts for review:")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    generate_explanations()
