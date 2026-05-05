import time
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple
from pydantic import BaseModel, Field
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# We will use the provided custom base URL
client = OpenAI(
    base_url=os.getenv("BASE_URL", "https://api.openai.com/v1"),
    api_key=os.getenv("OPENAI_API_KEY")
)

# Cost tracking variables
TOTAL_TOKENS = 0
TOTAL_TIME = 0.0

class Triple(BaseModel):
    subject: str = Field(description="The main subject entity.")
    relation: str = Field(description="The relationship between subject and object, in uppercase with underscores.")
    object_: str = Field(description="The object entity.")

class TripleList(BaseModel):
    triples: List[Triple]

def track_cost(func):
    """Decorator to track time and token usage."""
    def wrapper(*args, **kwargs):
        global TOTAL_TIME, TOTAL_TOKENS
        start_time = time.time()
        
        # Call the function
        result, tokens_used = func(*args, **kwargs)
        
        end_time = time.time()
        elapsed = end_time - start_time
        TOTAL_TIME += elapsed
        TOTAL_TOKENS += tokens_used
        
        return result
    return wrapper

@track_cost
def extract_triples_from_text(text: str) -> Tuple[List[Tuple[str, str, str]], int]:
    """
    Extracts entities and relations from a given text using LLM.
    Returns a list of triples and the number of tokens used.
    """
    prompt = f"""
    Given the following text, extract all factual relationships and represent them as knowledge graph triples.
    Format the output as JSON with a list of 'triples', where each triple has 'subject', 'relation', and 'object_'.
    The relation should be a concise, uppercase string with underscores (e.g., FOUNDED_BY, DEVELOPED_PRODUCT, IS_CEO_OF, INVESTED_IN).
    
    Text: "{text}"
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful knowledge graph extraction system. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens
        
        data = json.loads(content)
        extracted_triples = []
        for item in data.get("triples", []):
            extracted_triples.append((
                item.get("subject", ""), 
                item.get("relation", ""), 
                item.get("object_", "")
            ))
            
        return extracted_triples, tokens
    except Exception as e:
        print(f"Error extracting from text '{text}': {e}")
        return [], 0

def build_knowledge_graph(corpus: List[str]) -> nx.DiGraph:
    """Builds a NetworkX directed graph from a text corpus."""
    global TOTAL_TIME, TOTAL_TOKENS
    # Reset trackers
    TOTAL_TIME = 0.0
    TOTAL_TOKENS = 0
    
    G = nx.DiGraph()
    all_triples = []
    
    print("Extracting triples from corpus...")
    for i, sentence in enumerate(corpus):
        triples = extract_triples_from_text(sentence)
        all_triples.extend(triples)
        print(f"  Processed sentence {i+1}/{len(corpus)}: found {len(triples)} triples.")
        
    print(f"Extraction complete. Total time: {TOTAL_TIME:.2f}s, Total tokens: {TOTAL_TOKENS}")
    
    for subject, relation, obj in all_triples:
        if subject and relation and obj:
            # Ensure nodes are distinct and clean
            subj_clean = subject.strip()
            obj_clean = obj.strip()
            G.add_node(subj_clean)
            G.add_node(obj_clean)
            G.add_edge(subj_clean, obj_clean, label=relation)
            
    return G

def visualize_graph(G: nx.DiGraph, output_path: str = "knowledge_graph.png"):
    """Visualizes the NetworkX graph and saves it to a file."""
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color="lightblue", alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, edge_color="gray", arrows=True)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", font_family="sans-serif")
    
    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title("Tech Company Knowledge Graph", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Graph visualization saved to {output_path}")
    plt.close()
