import pandas as pd
import networkx as nx
import os


def get_project_path():
    return os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )


def load_graph():

    graph_path = os.path.join(
        get_project_path(),
        "models",
        "transaction_graph.graphml"
    )

    if not os.path.exists(graph_path):
        raise FileNotFoundError(
            "Graph not found. Run graph_builder.py first."
        )

    return nx.read_graphml(graph_path)


def load_alerts():

    alerts_path = os.path.join(
        get_project_path(),
        "data",
        "detected_alerts.csv"
    )

    if not os.path.exists(alerts_path):
        raise FileNotFoundError(
            "Alerts not found. Run pattern_detection.py first."
        )

    return pd.read_csv(alerts_path)


def detect_clusters(G, alerts):

    print("\n========== CLUSTER DETECTION ==========")

    # Convert directed graph into undirected graph
    undirected_graph = G.to_undirected()

    # Find connected account groups
    components = list(
        nx.connected_components(undirected_graph)
    )

    account_cluster_map = []
    cluster_summary = []

    alert_accounts = set(
        alerts["account"].dropna().astype(str)
    ) if not alerts.empty else set()

    for cluster_number, accounts in enumerate(
        components, start=1
    ):

        accounts = set(accounts)

        subgraph = G.subgraph(accounts)

        transaction_count = subgraph.number_of_edges()

        suspicious_accounts = accounts.intersection(
            alert_accounts
        )

        cluster_alerts = (
            alerts[
                alerts["account"].astype(str).isin(accounts)
            ]
            if not alerts.empty
            else pd.DataFrame()
        )

        pattern_counts = (
            cluster_alerts["pattern"].value_counts().to_dict()
            if not cluster_alerts.empty
            else {}
        )

        for account in accounts:

            account_cluster_map.append({
                "account": account,
                "cluster_id": cluster_number,
                "cluster_size": len(accounts),
                "has_pattern_alert": int(
                    account in alert_accounts
                )
            })

        cluster_summary.append({

            "cluster_id": cluster_number,

            "account_count": len(accounts),

            "transaction_count": transaction_count,

            "accounts_with_alerts": len(
                suspicious_accounts
            ),

            "total_pattern_alerts": len(
                cluster_alerts
            ),

            "detected_patterns": ", ".join(
                pattern_counts.keys()
            ) if pattern_counts else "NONE",

            "accounts": ", ".join(
                sorted(accounts)
            )

        })

    account_df = pd.DataFrame(account_cluster_map)

    cluster_df = pd.DataFrame(cluster_summary)

    if not cluster_df.empty:

        cluster_df = cluster_df.sort_values(
            by=[
                "accounts_with_alerts",
                "total_pattern_alerts",
                "transaction_count"
            ],
            ascending=False
        )

    return cluster_df, account_df


def save_results(cluster_df, account_df):

    project_dir = get_project_path()

    cluster_path = os.path.join(
        project_dir,
        "data",
        "cluster_summary.csv"
    )

    account_path = os.path.join(
        project_dir,
        "data",
        "account_cluster_map.csv"
    )

    cluster_df.to_csv(
        cluster_path,
        index=False
    )

    account_df.to_csv(
        account_path,
        index=False
    )

    print("\nCluster Summary Saved:", cluster_path)

    print("Account Mapping Saved:", account_path)


def display_results(cluster_df):

    print("\nTotal Clusters:", len(cluster_df))

    print("\nTop 10 Account Clusters:")

    print(
        cluster_df.head(10).to_string(index=False)
    )


if __name__ == "__main__":

    graph = load_graph()

    alerts = load_alerts()

    cluster_df, account_df = detect_clusters(
        graph,
        alerts
    )

    display_results(cluster_df)

    save_results(
        cluster_df,
        account_df
    )

    print("\nCluster Detection Completed!")