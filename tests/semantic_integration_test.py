"""
Integration test for the Semantic Search Pipeline using real scraped data.
"""
import pandas as pd
from job_finder.semanticmodels import rank_jobs
from job_finder.scraper import scrape_linkedin_jobs, scrape_google_jobs
from job_finder.config import PROFILE_TEXT, CRITERIA
import warnings
warnings.filterwarnings('ignore')

def test_semantic_pipeline():
    print("Testing Semantic Search Pipeline with Real Data...")
    
    # 1. Provide target text
    target_text = f"{PROFILE_TEXT}\n\nKey Requirements:\n{CRITERIA}"
    
    # 2. Scrape real data
    # We'll grab 2 from LinkedIn and 2 from Google as requested
    dfs = []
    
    print("Scraping 2 jobs from LinkedIn...")
    linkedin_df = scrape_linkedin_jobs(max_jobs=2, num_scrolls=0)
    if not linkedin_df.empty:
        linkedin_df['source'] = 'LinkedIn'
        dfs.append(linkedin_df)
    
    print("Scraping jobs from Google...")
    google_url = "https://www.google.com/search?q=data+scientist+jobs+san+francisco&ibp=htl;jobs"
    google_df = scrape_google_jobs(google_url, num_scrolls=0)
    if not google_df.empty:
        google_df['source'] = 'Google'
        # Limit to 2 for the test
        dfs.append(google_df.head(2))
    
    if not dfs:
        print("No jobs were scraped from either source. Test Failed.")
        return
        
    jobs_df = pd.concat(dfs, ignore_index=True)
    print(f"Total jobs collected for test: {len(jobs_df)}")
    
    # 3. Rank jobs using the semantic model
    print("Ranking jobs against your profile using vector similarity...")
    ranked_df = rank_jobs(jobs_df, target_text, score_threshold=0.0)
    
    print("\n--- Semantic Ranking Results ---")
    print(ranked_df[['source', 'titles', 'companies', 'similarity_score']])
    
    print("\nIntegration test finished successfully!")

if __name__ == "__main__":
    test_semantic_pipeline()
