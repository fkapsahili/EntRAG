import pickle

import matplotlib.pyplot as plt
import networkx as nx


def main(graph_path: str, out_path: str = None, max_nodes: int = 100):
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Debug: print all node names and types for inspection
    print("\nNodes in the graph (id, type, name):")
    for n, d in G.nodes(data=True):
        print(f"  {n}: type={d.get('type')}, name={d.get('name')}")
    print("\nSample edges (source, target, key, type, weight):")
    for u, v, k, d in list(G.edges(keys=True, data=True))[:20]:
        print(f"  {u} -> {v} (key={k}): type={d.get('type')}, weight={d.get('weight')}")
    print("\n--- End of debug info ---\n")

    # Optionally, only plot a subgraph for large graphs
    if G.number_of_nodes() > max_nodes:
        nodes = list(G.nodes)[:max_nodes]
        G = G.subgraph(nodes)
        print(f"Plotting only the first {max_nodes} nodes.")

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=500, font_size=8, edge_color="#cccccc")
    node_labels = {n: G.nodes[n].get("name", n) for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    plt.title("Knowledge Graph Visualization")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
        print(f"Graph image saved to {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize a pickled knowledge graph.")
    parser.add_argument("graph_path", help="Path to the pickled knowledge graph file (e.g., knowledge_graph.gpickle)")
    parser.add_argument("--out", help="Optional: output image file path (e.g., graph.png)")
    parser.add_argument("--max-nodes", type=int, default=100, help="Max nodes to plot (default: 100)")
    args = parser.parse_args()
    main(args.graph_path, args.out, args.max_nodes)
