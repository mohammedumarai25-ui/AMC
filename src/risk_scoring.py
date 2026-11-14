import pandas as pd
import os


def get_project_path():
    return os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )


def load_data():

    project_dir = get_project_path()

    transaction_path = os.path.join(
        project_dir, "data", "processed_transactions.csv"
    )

    alert_path = os.path.join(
        project_dir, "data", "detected_alerts.csv"
    )

    if not os.path.exists(transaction_path):
        raise FileNotFoundError(
            "Run preprocessing.py first."
        )

    if not os.path.exists(alert_path):
        raise FileNotFoundError(
            "Run pattern_detection.py first."
        )

    transactions = pd.read_csv(transaction_path)
    alerts = pd.read_csv(alert_path)

    transactions["timestamp"] = pd.to_datetime(
        transactions["timestamp"]
    )

    return transactions, alerts


def calculate_risk_scores(transactions, alerts):

    accounts = set(transactions["sender_account"])
    accounts.update(transactions["receiver_account"])

    risk_results = []

    for account in accounts:

        sent = transactions[
            transactions["sender_account"] == account
        ]

        received = transactions[
            transactions["receiver_account"] == account
        ]

        account_alerts = (
            alerts[alerts["account"] == account]
            if not alerts.empty
            else pd.DataFrame()
        )

        score = 0
        reasons = []

        # 1. Number of detected pattern alerts
        alert_count = len(account_alerts)

        if alert_count > 0:

            score += min(alert_count * 5, 20)

            reasons.append(
                f"{alert_count} pattern alerts detected"
            )

        # 2. Different suspicious patterns
        pattern_count = (
            account_alerts["pattern"].nunique()
            if not account_alerts.empty
            else 0
        )

        if pattern_count >= 2:

            score += 15

            reasons.append(
                "Multiple different transaction patterns detected"
            )

        # 3. High-risk structural patterns
        high_risk_patterns = {
            "CYCLE": 10,
            "RAPID_PASS_THROUGH": 15,
            "LAYERED_NETWORK": 15
        }

        detected_patterns = (
            set(account_alerts["pattern"])
            if not account_alerts.empty
            else set()
        )

        for pattern, points in high_risk_patterns.items():

            if pattern in detected_patterns:

                score += points

                reasons.append(
                    f"{pattern} pattern detected"
                )

        # 4. Number of unique transaction partners
        counterparties = set(sent["receiver_account"])
        counterparties.update(received["sender_account"])

        unique_partners = len(counterparties)

        if unique_partners >= 10:

            score += 15
            reasons.append(
                "Very high number of unique transaction partners"
            )

        elif unique_partners >= 5:

            score += 10
            reasons.append(
                "High number of unique transaction partners"
            )

        # 5. Cross-border transaction behaviour
        total_transactions = len(sent) + len(received)

        cross_border_sent = (
            sent["sender_country"] != sent["receiver_country"]
        ).sum()

        cross_border_received = (
            received["sender_country"] !=
            received["receiver_country"]
        ).sum()

        cross_border_count = (
            cross_border_sent + cross_border_received
        )

        cross_border_ratio = (
            cross_border_count / total_transactions
            if total_transactions > 0
            else 0
        )

        if total_transactions >= 5 and cross_border_ratio >= 0.5:

            score += 10

            reasons.append(
                "High proportion of cross-border transactions"
            )

        # 6. Transaction activity
        if total_transactions >= 20:

            score += 10

            reasons.append(
                "Very high transaction activity"
            )

        elif total_transactions >= 10:

            score += 5

            reasons.append(
                "High transaction activity"
            )

        # Keep score between 0 and 100
        score = min(score, 100)

        # Risk classification
        if score >= 70:
            risk_level = "HIGH"

        elif score >= 40:
            risk_level = "MEDIUM"

        else:
            risk_level = "LOW"

        total_sent = sent["amount"].sum()
        total_received = received["amount"].sum()

        risk_results.append({

            "account": account,

            "risk_score": score,

            "risk_level": risk_level,

            "total_transactions": total_transactions,

            "total_sent": round(total_sent, 2),

            "total_received": round(total_received, 2),

            "unique_partners": unique_partners,

            "pattern_alerts": alert_count,

            "different_patterns": pattern_count,

            "cross_border_ratio": round(
                cross_border_ratio, 2
            ),

            "reasons": "; ".join(reasons)
            if reasons else "No major risk indicators detected"

        })

    results = pd.DataFrame(risk_results)

    results = results.sort_values(
        by="risk_score",
        ascending=False
    ).reset_index(drop=True)

    return results


def save_risk_scores(results):

    output_path = os.path.join(
        get_project_path(),
        "data",
        "risk_scores.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print("\nRisk Scores Saved:")
    print(output_path)


def display_risk_summary(results):

    print("\n========== RISK SCORING SUMMARY ==========")

    print("\nRisk Level Distribution:")

    print(
        results["risk_level"].value_counts()
    )

    print("\nTop 10 Risky Accounts:")

    print(
        results.head(10).to_string(index=False)
    )

    print("\nRisk Scoring Completed!")


if __name__ == "__main__":

    transactions, alerts = load_data()

    results = calculate_risk_scores(
        transactions,
        alerts
    )

    display_risk_summary(results)

    save_risk_scores(results)