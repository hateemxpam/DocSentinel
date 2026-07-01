"""
main.py - Entry point for the DocSentinel evaluation pipeline
=============================================================
"""

import sys
from pathlib import Path

# Add project root to sys.path to enable imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Force standard output & error streams to support UTF-8 on Windows consoles
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass  # Fallback for environments where stdout cannot be reconfigured

from eval.runner import run_evaluation
from eval.metrics import calculate_metrics
from eval.report import generate_report


def main():
    """
    Main entry point for running the evaluation pipeline.
    """
    print("==================================================")
    print("       Starting DocSentinel Evaluation            ")
    print("==================================================")
    
    # 1. Run the RAG pipeline over the test dataset
    results = run_evaluation()
    
    # 2. Check if we have valid results to pass to RAGAs
    ragas_results = results.get("results", [])
    
    if not ragas_results:
        print("\n[eval/main] WARNING: All queries were blocked or skipped by the Trust Layer.")
        print("RAGAs metrics will default to 0.0 (no answers to judge).")
        scores = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0
        }
    else:
        # 3. Calculate metrics using RAGAs
        scores = calculate_metrics(ragas_results)
        
    # 4. Generate the report (prints and saves to eval/report.txt)
    generate_report(scores, results)
    
    print("\nEvaluation complete. Report saved to eval/report.txt")


if __name__ == "__main__":
    main()
