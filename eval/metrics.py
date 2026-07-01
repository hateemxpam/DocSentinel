"""
metrics.py - Computes RAGAs evaluation scores using Groq LLM and HuggingFace Embeddings
========================================================================================
"""

import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    AnswerRelevancy,
    context_precision,
    context_recall
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_groq import ChatGroq

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


def calculate_metrics(results: list[dict]) -> dict:
    """
    Run RAGAs evaluation metrics over the RAG pipeline output.

    Args:
        results: List of dicts containing keys: question, answer, contexts, ground_truth.

    Returns:
        dict: containing faithfulness, answer_relevancy, context_precision, context_recall scores.
    """
    if not results:
        print("[eval/metrics] Warning: Empty results list. Skipping evaluation.")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0
        }

    # 1. Format the data for RAGAs/Datasets
    data_dict = {
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    }
    
    ragas_dataset = Dataset.from_dict(data_dict)

    # 2. Setup Groq Chat Model judge
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment.")

    groq_llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.0,
        api_key=api_key
    )
    wrapped_llm = LangchainLLMWrapper(groq_llm)

    # 3. Setup HuggingFace local embedding wrapper to avoid OpenAI dependency
    hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    wrapped_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    # 4. Configure AnswerRelevancy with strictness=1 to avoid n=3 Groq API constraint
    custom_answer_relevancy = AnswerRelevancy(strictness=1)

    print("\n[eval/metrics] Starting RAGAs evaluation via Groq LLM...")
    try:
        # Run evaluation
        eval_result = evaluate(
            dataset=ragas_dataset,
            metrics=[
                faithfulness,
                custom_answer_relevancy,
                context_precision,
                context_recall
            ],
            llm=wrapped_llm,
            embeddings=wrapped_embeddings
        )
        
        # 5. Extract metrics defensively using the pandas DataFrame representation
        scores = {}
        try:
            df = eval_result.to_pandas()
            for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                if metric_name in df.columns:
                    # Compute mean while filling any NaNs with 0.0
                    scores[metric_name] = float(df[metric_name].fillna(0.0).mean())
                else:
                    scores[metric_name] = 0.0
        except Exception as e:
            print(f"[eval/metrics] Warning: Failed to convert results to pandas DataFrame: {e}")
            # Fallback to direct key access
            for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                val = 0.0
                try:
                    val = eval_result[metric_name]
                    if isinstance(val, list):
                        val = sum(val) / len(val) if val else 0.0
                except Exception:
                    try:
                        if hasattr(eval_result, "scores"):
                            val = eval_result.scores.get(metric_name, 0.0)
                            if isinstance(val, list):
                                val = sum(val) / len(val) if val else 0.0
                    except Exception:
                        pass
                scores[metric_name] = float(val)
        
        print("[eval/metrics] RAGAs evaluation completed successfully.")
        return scores

    except Exception as exc:
        print(f"[eval/metrics] ERROR: RAGAs evaluation failed: {exc}")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0
        }
