import os
from corpus import get_corpus
from graph_builder import build_knowledge_graph, visualize_graph
from rag_systems import GraphRAG, FlatRAG
from evaluator import evaluate_responses
import dotenv

def main():
    dotenv.load_dotenv()
    
    print("=== STARTING GRAPHRAG VS FLATRAG PIPELINE ===\n")
    
    # 1. Load Corpus
    print("1. Loading Corpus...")
    corpus = get_corpus()
    print(f"Loaded {len(corpus)} sentences.\n")
    
    # 2. Build Knowledge Graph (Indexing & Construction)
    print("2. Building Knowledge Graph...")
    G = build_knowledge_graph(corpus)
    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.\n")
    
    # 3. Visualize Graph
    print("3. Visualizing Knowledge Graph...")
    visualize_graph(G, "knowledge_graph.png")
    print()
    
    # 4. Initialize RAG Systems
    print("4. Initializing RAG Systems...")
    graph_rag = GraphRAG(G)
    flat_rag = FlatRAG(corpus)
    print("Systems initialized.\n")
    
    # 5. Evaluate
    print("5. Running Benchmark Evaluation...")
    df = evaluate_responses(graph_rag, flat_rag)
    
    print("\n=== PIPELINE COMPLETED ===")

if __name__ == "__main__":
    main()
