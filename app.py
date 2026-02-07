import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import math

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Bellman-Ford Algorithm",
    page_icon="🚦",
    layout="centered"
)

st.title("🚦 Bellman–Ford Algorithm Visualization")
st.caption("DAA | Shortest Path Algorithm with Negative Edge Weights")

st.divider()

# ---------------- INPUT: VERTICES ----------------
st.subheader("🔢 Vertices")
vertices_input = st.text_input("Enter vertices (comma-separated)", "A,B,C")
vertices = [v.strip() for v in vertices_input.split(",") if v.strip()]

# ---------------- INPUT: EDGES ----------------
st.subheader("🔗 Edges")
edge_count = st.number_input("Number of edges", 1, 30, 3)

edges = []
for i in range(edge_count):
    c1, c2, c3 = st.columns(3)
    with c1:
        u = st.selectbox("From", vertices, key=f"u{i}")
    with c2:
        v = st.selectbox("To", vertices, key=f"v{i}")
    with c3:
        w = st.number_input("Weight", value=1, key=f"w{i}")
    edges.append((u, v, w))

edges_df = pd.DataFrame(edges, columns=["From", "To", "Weight"])

# ---------------- SOURCE ----------------
st.subheader("📍 Source Vertex")
source = st.selectbox("Select source", vertices)

st.divider()

# ---------------- BELLMAN-FORD LOGIC ----------------
def bellman_ford(vertices, edges, source):
    dist = {v: math.inf for v in vertices}
    dist[source] = 0

    for _ in range(len(vertices) - 1):
        for u, v, w in edges:
            if dist[u] != math.inf and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    for u, v, w in edges:
        if dist[u] != math.inf and dist[u] + w < dist[v]:
            return None, True

    return dist, False

# ---------------- RUN ----------------
if st.button("🚀 Run Bellman–Ford"):
    st.subheader("📋 Edge List")
    st.dataframe(edges_df, use_container_width=True)

    distances, neg_cycle = bellman_ford(vertices, edges, source)

    if neg_cycle:
        st.error("❌ Negative Weight Cycle Detected")
        st.stop()

    st.success("✅ Shortest paths computed successfully")

    # ---------------- GRAPH VISUALIZATION ----------------
    st.subheader("🧭 Solved Graph (Bellman–Ford Output)")

    G = nx.DiGraph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    # Source-based layout (DAA style)
    pos = {source: (0, 0)}
    other_nodes = [v for v in vertices if v != source]

    y_vals = list(range(len(other_nodes) // 2, -len(other_nodes), -1))
    for i, node in enumerate(other_nodes):
        pos[node] = (3, y_vals[i])

    fig, ax = plt.subplots(figsize=(8, 5))

    # Nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=2400,
        node_color="#4A90E2",
        ax=ax
    )

    # Node labels with distances
    node_labels = {}
    for v in vertices:
        if distances[v] == math.inf:
            node_labels[v] = f"{v}\n∞"
        else:
            node_labels[v] = f"{v}\n{distances[v]}"

    nx.draw_networkx_labels(
        G, pos,
        labels=node_labels,
        font_size=12,
        font_color="white",
        font_weight="bold",
        ax=ax
    )

    # Edges
    nx.draw_networkx_edges(
        G, pos,
        arrowstyle="->",
        arrowsize=25,
        width=2,
        edge_color="black",
        ax=ax
    )

    # Edge weights
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=11,
        ax=ax
    )

    ax.set_title("Final Shortest Path Graph", fontsize=14)
    ax.axis("off")

    st.pyplot(fig)

    # ---------------- SOLUTION EXPLANATION ----------------
    st.subheader("📌 Final Shortest Distances from Source")

    result_df = pd.DataFrame(
        distances.items(),
        columns=["Vertex", "Shortest Distance"]
    )
    st.dataframe(result_df, use_container_width=True)

    st.markdown(
        """
        **Explanation:**  
        The values shown inside each node represent the **shortest distance
        from the source vertex** after applying the Bellman–Ford algorithm.
        """
    )

st.divider()
st.caption("Built with ❤️ using Streamlit | Bellman–Ford Algorithm")
