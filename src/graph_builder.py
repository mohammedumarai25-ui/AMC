import pandas as pd
import networkx as nx
import os


def load_processed_data():

    project_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    file_path = os.path.join(
        project_dir,
        "data",
        "processed_transactions.csv"
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            "Processed dataset not found. Run preprocessing.py first."
        )

    df = pd.read_csv(file_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


def build_transaction_graph(df):

    G = nx.MultiDiGraph()

    for _, row in df.iterrows():

        sender = str(row["sender_account"])
        receiver = str(row["receiver_account"])

        G.add_node(sender)
        G.add_node(receiver)

        G.add_edge(
            sender,
            receiver,
            key=str(row["transaction_id"]),
            transaction_id=str(row["transaction_id"]),
            amount=float(row["amount"]),
            timestamp=row["timestamp"].isoformat(),
            transaction_type=str(row["transaction_type"]),
            sender_country=str(row["sender_country"]),
            receiver_country=str(row["receiver_country"]),
            is_suspicious=int(row["is_suspicious"]),
            suspicious_pattern=str(row["suspicious_pattern"])
        )

    return G


def display_graph_information(G):

    print("\n========== TRANSACTION GRAPH ==========")

    print("\nTotal Accounts:", G.number_of_nodes())

    print("Total Transactions:", G.number_of_edges())

    print("Graph Type:", type(G).__name__)

    print(
        "\nNumber of Connected Components:",
        nx.number_weakly_connected_components(G)
    )

    print("\nSample Accounts:")

    print(list(G.nodes())[:10])

    print("\nSample Transactions:")

    for u, v, data in list(G.edges(data=True))[:5]:

        print(
            u,
            " -> ",
            v,
            "| Amount:",
            data["amount"],
            "| Transaction:",
            data["transaction_id"]
        )


def save_graph(G):

    project_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    output_path = os.path.join(
        project_dir,
        "models",
        "transaction_graph.graphml"
    )

    nx.write_graphml(G, output_path)

    print("\nGraph Saved Successfully!")
    print(output_path)


if __name__ == "__main__":

    transactions = load_processed_data()

    graph = build_transaction_graph(transactions)

    display_graph_information(graph)

    save_graph(graph)