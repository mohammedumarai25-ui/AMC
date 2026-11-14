import pandas as pd
import networkx as nx
import os


def load_data():

    project_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    file_path = os.path.join(
        project_dir,
        "data",
        "processed_transactions.csv"
    )

    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


def detect_fan_in(df, threshold=4):

    alerts = []

    incoming = df.groupby("receiver_account")["sender_account"].nunique()

    for account, count in incoming.items():

        if count >= threshold:

            alerts.append({
                "pattern": "FAN_IN",
                "account": account,
                "related_accounts": count,
                "evidence": f"{count} unique accounts transferred money to this account"
            })

    return alerts


def detect_fan_out(df, threshold=4):

    alerts = []

    outgoing = df.groupby("sender_account")["receiver_account"].nunique()

    for account, count in outgoing.items():

        if count >= threshold:

            alerts.append({
                "pattern": "FAN_OUT",
                "account": account,
                "related_accounts": count,
                "evidence": f"This account transferred money to {count} unique accounts"
            })

    return alerts


def detect_cycles(df, max_cycles=200):

    alerts = []

    graph = nx.DiGraph()

    for _, row in df.iterrows():

        graph.add_edge(
            row["sender_account"],
            row["receiver_account"]
        )

    cycle_count = 0

    for cycle in nx.simple_cycles(graph):

        if len(cycle) >= 3 and len(cycle) <= 5:

            alerts.append({
                "pattern": "CYCLE",
                "account": cycle[0],
                "related_accounts": len(cycle) - 1,
                "evidence": "Circular transaction path: " + " -> ".join(cycle + [cycle[0]])
            })

            cycle_count += 1

            if cycle_count >= max_cycles:
                break

    return alerts


def detect_rapid_pass_through(df, minutes=30):

    alerts = []

    for account, group in df.groupby("receiver_account"):

        incoming = group[
            ["timestamp", "amount", "transaction_id", "sender_account"]
        ].copy()

        outgoing = df[
            df["sender_account"] == account
        ][
            ["timestamp", "amount", "transaction_id", "receiver_account"]
        ].copy()

        for _, in_tx in incoming.iterrows():

            possible_outgoing = outgoing[
                (outgoing["timestamp"] > in_tx["timestamp"]) &
                (
                    outgoing["timestamp"] <=
                    in_tx["timestamp"] + pd.Timedelta(minutes=minutes)
                )
            ]

            for _, out_tx in possible_outgoing.iterrows():

                amount_ratio = (
                    out_tx["amount"] / in_tx["amount"]
                    if in_tx["amount"] > 0 else 0
                )

                if 0.8 <= amount_ratio <= 1.2:

                    alerts.append({
                        "pattern": "RAPID_PASS_THROUGH",
                        "account": account,
                        "related_accounts": 2,
                        "evidence": (
                            f"Incoming transaction {in_tx['transaction_id']} "
                            f"followed by outgoing transaction {out_tx['transaction_id']} "
                            f"within {minutes} minutes with similar amounts"
                        )
                    })

                    break

    return alerts


def detect_layered_network(df, max_alerts=200):

    alerts = []

    transactions = df.sort_values("timestamp")

    outgoing = {}

    for _, row in transactions.iterrows():

        outgoing.setdefault(row["sender_account"], []).append(row)

    for _, first in transactions.iterrows():

        path = [
            first["sender_account"],
            first["receiver_account"]
        ]

        transaction_ids = [first["transaction_id"]]

        current_account = first["receiver_account"]
        current_time = first["timestamp"]

        for depth in range(3):

            next_transactions = outgoing.get(current_account, [])

            next_tx = None

            for tx in next_transactions:

                if (
                    tx["timestamp"] > current_time and
                    tx["timestamp"] <= current_time + pd.Timedelta(hours=2) and
                    tx["receiver_account"] not in path
                ):
                    next_tx = tx
                    break

            if next_tx is None:
                break

            path.append(next_tx["receiver_account"])
            transaction_ids.append(next_tx["transaction_id"])

            current_account = next_tx["receiver_account"]
            current_time = next_tx["timestamp"]

            if len(transaction_ids) >= 3:

                alerts.append({
                    "pattern": "LAYERED_NETWORK",
                    "account": first["sender_account"],
                    "related_accounts": len(path) - 1,
                    "evidence": (
                        "Transaction chain: " + " -> ".join(path) +
                        " | Transactions: " + ", ".join(transaction_ids)
                    )
                })

                break

        if len(alerts) >= max_alerts:
            break

    return alerts


def detect_all_patterns(df):

    print("\n========== PATTERN DETECTION ==========")

    alerts = []

    alerts.extend(detect_fan_in(df))
    alerts.extend(detect_fan_out(df))
    alerts.extend(detect_cycles(df))
    alerts.extend(detect_rapid_pass_through(df))
    alerts.extend(detect_layered_network(df))

    results = pd.DataFrame(alerts)

    if results.empty:

        print("\nNo suspicious patterns detected.")

    else:

        print("\nDetected Patterns:")
        print(results["pattern"].value_counts())

        print("\nSample Alerts:")
        print(results.head(10).to_string(index=False))

    return results


def save_alerts(results):

    project_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    output_path = os.path.join(
        project_dir,
        "data",
        "detected_alerts.csv"
    )

    results.to_csv(output_path, index=False)

    print("\nAlerts Saved Successfully!")
    print(output_path)


if __name__ == "__main__":

    df = load_data()

    alerts = detect_all_patterns(df)

    save_alerts(alerts)