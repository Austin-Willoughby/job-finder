"""
Daily worker script to fetch new jobs and run semantic processing.
"""
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import semantic_pipeline
from job_finder.logging_config import setup_logging, get_logger

logger = get_logger("job_finder.worker")

def run_daily_worker():
    setup_logging()
    logger.info("Starting Daily Job Worker...")
    
    # Configuration for the daily run
    # f_tpr='r172800' is the 48-hour window on LinkedIn
    try:
        semantic_pipeline(
            max_pages=40,        # Scrape up to ~1000 jobs
            use_api=True,        # Use the fast API
            f_tpr='r172800',     # Last 2 days
            scrape_only=False    # Run inference/ranking after scraping
        )
        logger.info("Daily Job Worker completed successfully.")
    except Exception as e:
        logger.error(f"Daily Job Worker failed: {e}")

if __name__ == "__main__":
    run_daily_worker()
