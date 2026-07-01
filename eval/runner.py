"""
runner.py - Runs the DocSentinel RAG pipeline over the evaluation dataset
========================================================================
"""

from eval.test_dataset import dataset
from retrieval.retriever import retrieve
from trust.pipeline import run_trust_layer
from generation.pipeline import generate_response


def run_evaluation() -> dict:
    """
    Run evaluation by pushing the test dataset questions through the DocSentinel pipeline.

    Filters out failed, blocked, or out-of-context queries to ensure RAGAs evaluates
    only valid generated responses.

    Returns:
        dict: {
            "results": list[dict] (Dataset elements matching RAGAs expectations),
            "summary": list[dict] (Flat metadata summary for the generation report)
        }
    """
    results = []
    summary = []
    
    total_questions = len(dataset)
    
    print(f"\n--- Running DocSentinel Evaluation over {total_questions} questions ---")
    
    for idx, item in enumerate(dataset, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        print(f"\n[{idx}/{total_questions}] Evaluating: '{question[:50]}...'")
        
        # 1. Run pipeline layers
        chunks = retrieve(question)
        trust = run_trust_layer(chunks)
        response = generate_response(question, trust)
        
        status = response.get("status")
        avg_confidence = response.get("avg_confidence", 0.0)
        answer = response.get("answer")
        
        # 2. Add to flat summary list
        summary.append({
            "index": idx,
            "question": question,
            "status": status,
            "avg_confidence": avg_confidence
        })
        
        # 3. Check for BLOCKED or INSUFFICIENT_CONTEXT and skip RAGAs evaluation if so
        if status in ["BLOCKED", "INSUFFICIENT_CONTEXT"]:
            print(f"  >> Question {idx} skipped - status: {status}")
            continue

        # 4. Collect formatted context chunks
        context_texts = [c.get("chunk_text", "") for c in trust.get("chunks", [])]
        
        # 5. Save results formatted for RAGAs
        results.append({
            "question": question,
            "answer": answer if answer else "No answer generated",
            "contexts": context_texts,
            "ground_truth": ground_truth
        })

    return {
        "results": results,
        "summary": summary
    }
