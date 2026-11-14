
import streamlit as st
import pandas as pd
import plotly.express as px
import networkx as nx
from pathlib import Path

# -------------------------------
# PATHS
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

FILES = {
    "transactions": DATA_DIR / "processed_transactions.csv",
    "alerts": DATA_DIR / "detected_alerts.csv",
    "risks": DATA_DIR / "risk_scores.csv",
    "clusters": DATA_DIR / "cluster_summary.csv",
    "accounts": DATA_DIR / "account_cluster_map.csv",
    "predictions": DATA_DIR / "ml_predictions.csv",
    "cases": DATA_DIR / "explainable_cases.csv"
}

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="AML Detection System",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Graph-Based AML Detection System")
st.caption("Synthetic transaction analysis and suspicious network monitoring")

# -------------------------------
# LOAD DATA
# -------------------------------

@st.cache_data
def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


transactions = load_csv(FILES["transactions"])
alerts = load_csv(FILES["alerts"])
risks = load_csv(FILES["risks"])
clusters = load_csv(FILES["clusters"])
accounts = load_csv(FILES["accounts"])
predictions = load_csv(FILES["predictions"])
cases = load_csv(FILES["cases"])


# -------------------------------
# HELPER FUNCTIONS
# -------------------------------

def find_column(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def show_missing_files():

    missing = [
        name for name, path in FILES.items()
        if not path.exists()
    ]

    if missing:
        st.warning(
            "Some output files are missing: "
            + ", ".join(missing)
            + ". Run the corresponding Python scripts first."
        )


show_missing_files()


# -------------------------------
# SIDEBAR
# -------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Dashboard Page",
    [
        "Overview",
        "Transactions",
        "Suspicious Alerts",
        "Account Risk Analysis",
        "Network Clusters",
        "ML Predictions",
        "Explainable Cases"
    ]
)


# -------------------------------
# OVERVIEW
# -------------------------------

if page == "Overview":

    st.header("System Overview")

    total_transactions = len(transactions)

    if not transactions.empty:

        sender_col = find_column(
            transactions,
            ["sender_account"]
        )

        receiver_col = find_column(
            transactions,
            ["receiver_account"]
        )

        total_accounts = 0

        if sender_col and receiver_col:
            total_accounts = len(
                set(transactions[sender_col].astype(str))
                | set(transactions[receiver_col].astype(str))
            )

        suspicious_col = find_column(
            transactions,
            ["is_suspicious"]
        )

        suspicious_count = 0

        if suspicious_col:
            suspicious_count = pd.to_numeric(
                transactions[suspicious_col],
                errors="coerce"
            ).fillna(0).sum()

    else:
        total_accounts = 0
        suspicious_count = 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Transactions", f"{total_transactions:,}")
    col2.metric("Accounts", f"{total_accounts:,}")
    col3.metric("Suspicious Labels", f"{int(suspicious_count):,}")
    col4.metric("Detected Alerts", f"{len(alerts):,}")

    st.divider()

    if not transactions.empty:

        st.subheader("Transaction Amount Distribution")

        if "amount" in transactions.columns:

            amount_data = transactions.copy()

            amount_data["amount"] = pd.to_numeric(
                amount_data["amount"],
                errors="coerce"
            )

            fig = px.histogram(
                amount_data,
                x="amount",
                nbins=30,
                title="Transaction Amount Distribution"
            )

            st.plotly_chart(fig, use_container_width=True)

        if "transaction_type" in transactions.columns:

            st.subheader("Transaction Types")

            type_counts = (
                transactions["transaction_type"]
                .value_counts()
                .reset_index()
            )

            type_counts.columns = [
                "Transaction Type",
                "Count"
            ]

            fig = px.pie(
                type_counts,
                names="Transaction Type",
                values="Count",
                title="Transaction Type Distribution"
            )

            st.plotly_chart(fig, use_container_width=True)


# -------------------------------
# TRANSACTIONS
# -------------------------------

elif page == "Transactions":

    st.header("Transaction Records")

    if transactions.empty:
        st.info("No transaction data available.")

    else:

        filtered = transactions.copy()

        if "transaction_type" in filtered.columns:

            types = filtered["transaction_type"].dropna().unique()

            selected_types = st.multiselect(
                "Filter Transaction Type",
                types,
                default=list(types)
            )

            filtered = filtered[
                filtered["transaction_type"].isin(selected_types)
            ]

        if "amount" in filtered.columns:

            filtered["amount"] = pd.to_numeric(
                filtered["amount"],
                errors="coerce"
            )

            min_amount = float(filtered["amount"].min())
            max_amount = float(filtered["amount"].max())

            if min_amount < max_amount:

                amount_range = st.slider(
                    "Transaction Amount Range",
                    min_amount,
                    max_amount,
                    (min_amount, max_amount)
                )

                filtered = filtered[
                    filtered["amount"].between(*amount_range)
                ]

        st.write("Total matching transactions:", len(filtered))

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Filtered Transactions",
            filtered.to_csv(index=False),
            "filtered_transactions.csv",
            "text/csv"
        )


# -------------------------------
# SUSPICIOUS ALERTS
# -------------------------------

elif page == "Suspicious Alerts":

    st.header("Suspicious Activity Alerts")

    if alerts.empty:
        st.info("No alert data available.")

    else:

        st.metric("Total Alerts", len(alerts))

        pattern_col = find_column(
            alerts,
            ["pattern_type", "pattern", "suspicious_pattern"]
        )

        if pattern_col:

            st.subheader("Alert Pattern Distribution")

            pattern_counts = (
                alerts[pattern_col]
                .value_counts()
                .reset_index()
            )

            pattern_counts.columns = ["Pattern", "Count"]

            fig = px.bar(
                pattern_counts,
                x="Pattern",
                y="Count",
                title="Detected Suspicious Patterns"
            )

            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Alert Records")

        st.dataframe(
            alerts,
            use_container_width=True,
            hide_index=True
        )


# -------------------------------
# ACCOUNT RISK ANALYSIS
# -------------------------------

elif page == "Account Risk Analysis":

    st.header("Account Risk Analysis")

    if risks.empty:
        st.info("Risk score data not available.")

    else:

        risk_col = find_column(
            risks,
            ["risk_level", "risk_category"]
        )

        score_col = find_column(
            risks,
            ["risk_score", "score"]
        )

        if risk_col:

            st.subheader("Risk Level Distribution")

            counts = risks[risk_col].value_counts().reset_index()
            counts.columns = ["Risk Level", "Accounts"]

            fig = px.pie(
                counts,
                names="Risk Level",
                values="Accounts",
                title="Account Risk Distribution"
            )

            st.plotly_chart(fig, use_container_width=True)

        if score_col:

            st.subheader("Highest Risk Accounts")

            top_risks = risks.sort_values(
                score_col,
                ascending=False
            ).head(20)

            st.dataframe(
                top_risks,
                use_container_width=True,
                hide_index=True
            )

        st.download_button(
            "Download Risk Scores",
            risks.to_csv(index=False),
            "risk_scores.csv",
            "text/csv"
        )


# -------------------------------
# NETWORK CLUSTERS
# -------------------------------

elif page == "Network Clusters":

    st.header("Transaction Network Clusters")

    if clusters.empty:
        st.info("Cluster summary not available.")

    else:

        st.subheader("Cluster Summary")

        st.dataframe(
            clusters,
            use_container_width=True,
            hide_index=True
        )

    if not accounts.empty:

        st.subheader("Account-to-Cluster Mapping")

        st.dataframe(
            accounts,
            use_container_width=True,
            hide_index=True
        )

    if not transactions.empty:

        st.subheader("Explore Account Connections")

        sender_col = find_column(
            transactions,
            ["sender_account"]
        )

        receiver_col = find_column(
            transactions,
            ["receiver_account"]
        )

        if sender_col and receiver_col:

            account_list = sorted(
                set(transactions[sender_col].astype(str))
                | set(transactions[receiver_col].astype(str))
            )

            selected_account = st.selectbox(
                "Select an Account",
                account_list
            )

            related = transactions[
                (transactions[sender_col].astype(str) == selected_account)
                | (transactions[receiver_col].astype(str) == selected_account)
            ]

            st.write("Transactions involving this account:")

            st.dataframe(
                related,
                use_container_width=True,
                hide_index=True
            )

            graph = nx.DiGraph()

            for _, row in related.iterrows():

                graph.add_edge(
                    str(row[sender_col]),
                    str(row[receiver_col])
                )

            st.write("Connected accounts:")

            neighbors = list(graph.successors(selected_account))

            predecessors = list(graph.predecessors(selected_account))

            st.write("Outgoing connections:", neighbors)
            st.write("Incoming connections:", predecessors)


# -------------------------------
# ML PREDICTIONS
# -------------------------------

elif page == "ML Predictions":

    st.header("Machine Learning Predictions")

    if predictions.empty:
        st.info("ML predictions are not available. Run ml_model.py first.")

    else:

        prediction_col = find_column(
            predictions,
            ["predicted_suspicious"]
        )

        probability_col = find_column(
            predictions,
            ["suspicious_probability"]
        )

        if prediction_col:

            prediction_counts = (
                predictions[prediction_col]
                .value_counts()
                .reset_index()
            )

            prediction_counts.columns = [
                "Prediction",
                "Count"
            ]

            fig = px.bar(
                prediction_counts,
                x="Prediction",
                y="Count",
                title="ML Prediction Distribution"
            )

            st.plotly_chart(fig, use_container_width=True)

        if probability_col:

            predictions[probability_col] = pd.to_numeric(
                predictions[probability_col],
                errors="coerce"
            )

            st.subheader("Highest Suspicious Probabilities")

            top_predictions = predictions.sort_values(
                probability_col,
                ascending=False
            ).head(20)

            st.dataframe(
                top_predictions,
                use_container_width=True,
                hide_index=True
            )

        st.download_button(
            "Download ML Predictions",
            predictions.to_csv(index=False),
            "ml_predictions.csv",
            "text/csv"
        )


# -------------------------------
# EXPLAINABLE CASES
# -------------------------------

elif page == "Explainable Cases":

    st.header("Explainable AML Cases")

    if cases.empty:
        st.info("Explainable cases are not available.")

    else:

        st.subheader("Account Investigation Records")

        risk_col = find_column(
            cases,
            ["risk_level"]
        )

        filtered_cases = cases.copy()

        if risk_col:

            levels = filtered_cases[risk_col].dropna().unique()

            selected_levels = st.multiselect(
                "Filter Risk Level",
                levels,
                default=list(levels)
            )

            filtered_cases = filtered_cases[
                filtered_cases[risk_col].isin(selected_levels)
            ]

        st.dataframe(
            filtered_cases,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Explainable Cases",
            filtered_cases.to_csv(index=False),
            "explainable_cases.csv",
            "text/csv"
        )


# -------------------------------
# FOOTER
# -------------------------------

st.divider()

st.caption(
    "Academic prototype using synthetic transaction data. "
    "Alerts and risk scores are screening indicators, not proof of financial crime."
)
