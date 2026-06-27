"""
DocSentinel — Intelligent Policy Compliance Assistant
Main entry point for the application.
"""

from dotenv import load_dotenv
load_dotenv()

from ingestion.pipeline import run as run_ingestion


def main():
    print("=" * 50)
    print("  DocSentinel is initialized successfully! ")
    print("  Intelligent Policy Compliance Assistant")
    print("=" * 50)

    # Run the ingestion pipeline
    run_ingestion()


if __name__ == "__main__":
    main()
