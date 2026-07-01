"""
report.py - Formats and saves the evaluation report for DocSentinel
===================================================================
"""

import os


def generate_report(scores: dict, results: dict) -> None:
    """
    Generate and print the formatted evaluation report, and save it as eval/report.txt.

    Args:
        scores: Dict of RAGAs metrics (faithfulness, answer_relevancy, etc.).
        results: Dict from runner containing "results" and "summary".
    """
    summary = results.get("summary", [])
    
    # Calculate execution statistics
    total = len(summary)
    successful = sum(1 for q in summary if q["status"] in ["SUCCESS", "HALLUCINATION_RISK"])
    blocked = sum(1 for q in summary if q["status"] == "BLOCKED")
    skipped = sum(1 for q in summary if q["status"] == "INSUFFICIENT_CONTEXT")

    # Compute overall score (average of the 4 RAGAs metrics)
    metrics_list = [
        scores.get("faithfulness", 0.0),
        scores.get("answer_relevancy", 0.0),
        scores.get("context_precision", 0.0),
        scores.get("context_recall", 0.0)
    ]
    overall_score = sum(metrics_list) / len(metrics_list) if metrics_list else 0.0

    # Build the report string
    report_lines = []
    report_lines.append("╔══════════════════════════════════════╗")
    report_lines.append("║     DocSentinel Evaluation Report    ║")
    report_lines.append("╚══════════════════════════════════════╝")
    report_lines.append("")
    report_lines.append("📊 RAGAs Scores:")
    report_lines.append("─────────────────────────────────────")
    report_lines.append(f"Faithfulness:       {scores.get('faithfulness', 0.0):.2f}  (hallucination measure)")
    report_lines.append(f"Answer Relevancy:   {scores.get('answer_relevancy', 0.0):.2f}  (answered right question?)")
    report_lines.append(f"Context Precision:  {scores.get('context_precision', 0.0):.2f}  (right chunks retrieved?)")
    report_lines.append(f"Context Recall:     {scores.get('context_recall', 0.0):.2f}  (all chunks found?)")
    report_lines.append("─────────────────────────────────────")
    report_lines.append(f"Overall Score:      {overall_score:.2f}  (average of all 4)")
    report_lines.append("")
    report_lines.append("📋 Query Results:")
    report_lines.append("─────────────────────────────────────")
    
    for q in summary:
        report_lines.append(
            f"Q{q['index']}: {q['question'][:45]}... "
            f"→ {q['status']} | confidence: {q['avg_confidence']:.2f}"
        )
        
    report_lines.append("─────────────────────────────────────")
    report_lines.append(f"⚠️  Skipped Queries: {skipped}")
    report_lines.append(f"✅  Successful:      {successful}")
    report_lines.append(f"🚫  Blocked:         {blocked}")
    
    report_text = "\n".join(report_lines)

    # Print to console
    print("\n" + report_text + "\n")

    # Save to file (eval/report.txt)
    try:
        report_dir = os.path.dirname(os.path.abspath(__file__))
        report_path = os.path.join(report_dir, "report.txt")
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
            
        print(f"Report saved to: {report_path}")
    except Exception as exc:
        print(f"[eval/report] ERROR: Failed to save report.txt: {exc}")
