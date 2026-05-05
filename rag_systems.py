import os
from dotenv import load_dotenv
load_dotenv()
import networkx as nx
from typing import List, Tuple
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from openai import OpenAI
import chromadb

client = OpenAI(
    base_url=os.getenv("BASE_URL", "https://api.openai.com/v1"),
    api_key=os.getenv("OPENAI_API_KEY")
)

class GraphRAG:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def extract_main_entity(self, question: str) -> str:
        """Extract the main entity from the question using LLM."""
        prompt = f"""
        Extract the main single entity from the following question. 
        Only return the name of the entity, nothing else. If there are multiple, return the most prominent subject.
        Question: "{question}"
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        entity = response.choices[0].message.content.strip()
        return entity

    def get_2hop_context(self, entity: str) -> str:
        """Traverse the graph to get 2-hop neighbors and format as text."""
        if entity not in self.graph:
            # Fallback: try to find an entity that is a substring or vice versa
            for node in self.graph.nodes:
                if entity.lower() in node.lower() or node.lower() in entity.lower():
                    entity = node
                    break
        
        if entity not in self.graph:
            return ""

        context_triples = set()
        
        # 1-hop
        for neighbor in self.graph.successors(entity):
            relation = self.graph.edges[entity, neighbor]['label']
            context_triples.add(f"{entity} {relation} {neighbor}")
            
            # 2-hop
            for second_neighbor in self.graph.successors(neighbor):
                second_relation = self.graph.edges[neighbor, second_neighbor]['label']
                context_triples.add(f"{neighbor} {second_relation} {second_neighbor}")
                
        for neighbor in self.graph.predecessors(entity):
            relation = self.graph.edges[neighbor, entity]['label']
            context_triples.add(f"{neighbor} {relation} {entity}")
            
            # 2-hop
            for second_neighbor in self.graph.predecessors(neighbor):
                second_relation = self.graph.edges[second_neighbor, neighbor]['label']
                context_triples.add(f"{second_neighbor} {second_relation} {neighbor}")

        return ". ".join(list(context_triples))

    def answer(self, question: str) -> str:
        """Answer the question using the extracted graph context."""
        entity = self.extract_main_entity(question)
        context = self.get_2hop_context(entity)
        
        prompt = f"""
        Answer the following question based ONLY on the provided context. If the context does not contain the answer, say "I don't know".
        
        Context: {context}
        
        Question: {question}
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class FlatRAG:
    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve_context(self, question: str, k: int = 3) -> str:
        question_vec = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-k:][::-1]
        
        docs = [self.corpus[i] for i in top_indices]
        return " ".join(docs)

    def answer(self, question: str) -> str:
        context = self.retrieve_context(question)
        prompt = f"""
        Answer the following question based ONLY on the provided context. If the context does not contain the answer, say "I don't know".
        
        Context: {context}
        
        Question: {question}
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
