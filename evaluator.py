import pandas as pd
import os
import time
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from rag_systems import GraphRAG, FlatRAG

client = OpenAI(
    base_url=os.getenv("BASE_URL", "https://api.openai.com/v1"),
    api_key=os.getenv("OPENAI_API_KEY")
)

BENCHMARK_QUESTIONS = [
    "Ai là người sáng lập ra công ty tạo ra ChatGPT?",
    "Công ty nào đã đầu tư vào tổ chức do những cựu nhân viên của OpenAI thành lập?",
    "Sản phẩm nào cạnh tranh với GPT-4 và do ai phát triển?",
    "Elon Musk có liên quan đến những công ty nào?",
    "Công ty sở hữu công cụ tìm kiếm Bing có mối quan hệ gì với tổ chức tạo ra ChatGPT?",
    "Ai là CEO của công ty sản xuất iPhone?",
    "Công ty do Jensen Huang làm CEO sản xuất sản phẩm nào?",
    "Mô hình Llama là do công ty của ai phát triển?",
    "Grok là sản phẩm của công ty nào và ai đã thành lập công ty đó?",
    "Tên CEO của công ty mẹ sở hữu DeepMind là gì?"
]

def evaluate_responses(graph_rag: GraphRAG, flat_rag: FlatRAG):
    results = []
    
    print("Starting evaluation...")
    for i, question in enumerate(BENCHMARK_QUESTIONS):
        print(f"Processing question {i+1}/{len(BENCHMARK_QUESTIONS)}: {question}")
        
        try:
            ans_graph = graph_rag.answer(question)
            ans_flat = flat_rag.answer(question)
            
            # Use LLM to judge which one is better or if Flat RAG hallucinated
            prompt = f"""
            You are an evaluator. I will give you a question and two answers from different AI systems.
            Evaluate if "Flat RAG" hallucinated or failed to answer correctly compared to "Graph RAG".
            Answer with "YES" if Flat RAG hallucinated/failed and Graph RAG was correct. 
            Answer with "NO" if both were equally good, or both failed, or Flat RAG was better.
            
            Question: {question}
            Graph RAG Answer: {ans_graph}
            Flat RAG Answer: {ans_flat}
            """
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            hallucination_flag = response.choices[0].message.content.strip()

            results.append({
                "Question": question,
                "GraphRAG_Answer": ans_graph,
                "FlatRAG_Answer": ans_flat,
                "Flat_RAG_Hallucinated": hallucination_flag
            })
            
            # Save incrementally
            pd.DataFrame(results).to_csv("benchmark_results.csv", index=False, encoding="utf-8-sig")
            time.sleep(1) # Small delay
        except Exception as e:
            print(f"Error at question {i+1}: {e}")
            if "RateLimitError" in str(e) or "429" in str(e):
                print("Rate limit reached. Stopping evaluation.")
                break
        
    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv", index=False, encoding="utf-8-sig")
    print("Evaluation completed. Saved to benchmark_results.csv")
    
    # Print out hallucinations
    hallucinations = df[df["Flat_RAG_Hallucinated"].str.contains("YES", case=False, na=False)]
    print(f"\nFound {len(hallucinations)} cases where Flat RAG hallucinated but GraphRAG was correct.")
    for idx, row in hallucinations.iterrows():
        print(f"- Q: {row['Question']}")
        print(f"  GraphRAG: {row['GraphRAG_Answer']}")
        print(f"  FlatRAG: {row['FlatRAG_Answer']}\n")
    
    return df
