import os
import sys
import pandas as pd
from job_finder.scraper import scrape_linkedin_jobs_api
from job_finder.database import JobDatabase

def main():
    print("Starting API-based LinkedIn discovery test...")
    db = JobDatabase()
    
    # Scrape 5 jobs using the new API method with a different keyword to ensure we find new ones
    df = scrape_linkedin_jobs_api(db=db, keywords="Software Engineer", max_jobs=5)
    
    if not df.empty:
        print(f"\nSuccessfully scraped {len(df)} jobs via API:")
        print(df[['job_id', 'titles', 'companies']].to_string())
    else:
        print("\nNo new jobs found or API scraping failed.")

if __name__ == "__main__":
    main()
