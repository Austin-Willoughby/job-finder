"""
Main entry point for the Job Finder package
"""
import os
import argparse
from typing import Optional
from datetime import datetime
import pandas as pd
import joblib
from nltk.corpus import stopwords

from job_finder.config import KEYWORDS_BINS, COST_PER_1K_TOKENS
from job_finder.features import load_and_preprocess_data, create_corpus, create_bag_of_words
from job_finder.models import evaluate_models, load_and_predict_new_jobs
from job_finder.evaluator import get_desirability
from job_finder.semanticmodels import rank_jobs
from job_finder.scraper import scrape_linkedin_jobs, scrape_linkedin_jobs_api, scrape_google_jobs, get_chrome_driver
from job_finder.database import JobDatabase
from job_finder.scrapers.linkedin import AuthenticatedLinkedInScraper
from job_finder.config import PROFILE_TEXT, CRITERIA, CHROME_USER_DATA_DIR
import job_finder.evaluator as evaluator
import numpy as np

def train_pipeline(data_path: str, synthetic_path: str):
    print("Starting training pipeline...")
    stop_words = set(stopwords.words("english")).union(['said', 'would', 'could', "also", "new"])
    df = load_and_preprocess_data(data_path, synthetic_path, stop_words)
    
    print("Class distribution:")
    print(df['label'].value_counts(normalize=True))
    
    corpus = create_corpus(df['desc'], stop_words)
    titles_corpus = create_corpus(df['titles'], stop_words)
    
    keyword_columns = [
        'keyword_environmental', 'keyword_CV_autonomous_robotics', 
        'keyword_LLM_related', 'keyword_geospatial_r_sensing', 
        'keyword_energy', 'keyword_coding', 'weighted_keywords'
    ]
    keyword_features = df[keyword_columns].values
    
    n_components = 70
    (tfidf, cv_desc, cv_titles, tfidf_transformer_desc,
     tfidf_transformer_title, scaler, pca) = create_bag_of_words(corpus, titles_corpus, stop_words,
                                                                 keyword_features, n_components)
    
    label_array = np.array(df.label)
    tfidf_labelled = tfidf.tocsr()[label_array >= 0, :]
    targets = df[df['label'] >= 0]['label'].values
    
    best_svm_model, best_logistic_regression_model = evaluate_models(tfidf_labelled, targets)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_svm_model, 'models/best_svm_model.pkl')
    joblib.dump(best_logistic_regression_model, 'models/best_logistic_regression_model.pkl')
    
    # Save transformers for prediction pipeline
    joblib.dump({
        'cv_desc': cv_desc,
        'cv_titles': cv_titles,
        'tfidf_transformer_desc': tfidf_transformer_desc,
        'tfidf_transformer_title': tfidf_transformer_title,
        'scaler': scaler,
        'pca': pca
    }, 'models/transformers.pkl')
    
    print("Models and transformers saved to 'models/' directory.")

def predict_pipeline(evaluate: bool = False, num_scrolls_linkedin: int = 40, num_scrolls_google: int = 2):
    print("Starting prediction pipeline...")
    stop_words = set(stopwords.words("english")).union(['said', 'would', 'could', "also", "new"])
    
    try:
        pretrained_model = joblib.load('models/best_logistic_regression_model.pkl')
        transformers = joblib.load('models/transformers.pkl')
    except FileNotFoundError:
        print("Models not found. Please run the training pipeline first.")
        return
        
    job_board_urls = {
        "san_francisco": r"https://www.google.com/search?client=firefox-b-1-d&sca_esv=1ebb0e033accb772&q=data+scientist+san+francisco&prmd=invmsb&sa=X&biw=1760&bih=875&dpr=1.09&jbr=sep:0&ibp=htl;jobs&ved=2ahUKEwj49KjYwJqGAxWHQzABHdcfACMQudcGKAF6BAgiECk#fpstate=tldetail&htivrt=jobs&htidocid=R2A6Q-IkYWZ1jy46AAAAAA%3D%3D",
    }
    
    dfs = []
    for city, url in job_board_urls.items():
        output = load_and_predict_new_jobs(
            job_board_url=url,
            stop_words=stop_words,
            cv_desc=transformers['cv_desc'],
            cv_titles=transformers['cv_titles'],
            tfidf_transformer_desc=transformers['tfidf_transformer_desc'],
            tfidf_transformer_title=transformers['tfidf_transformer_title'],
            scaler=transformers['scaler'],
            pca=transformers['pca'],
            pretrained_model=pretrained_model,
            include_linkedin=True,
            scrape_google=False,
            linkedin_cap=50,
            num_scrolls_linkedin=num_scrolls_linkedin,
            num_scrolls_google=num_scrolls_google
        )
        if not output.empty:
            output['city'] = city
            dfs.append(output)
            
    if not dfs:
        print("No new jobs found.")
        return
        
    all_cities_output_df = pd.concat(dfs, ignore_index=True)
    
    if evaluate:
        print("Evaluating desirability with OpenAI...")
        all_cities_output_df["desirability"] = all_cities_output_df.apply(
            lambda row: get_desirability(
                row.get("companies", ""),
                row.get("titles", ""),
                row.get("location", ""),
                row.get("desc", "")
            ),
            axis=1
        )

        print(f"Total tokens used: {evaluator.total_tokens_used}")
        estimated_cost = (evaluator.total_tokens_used / 1000) * COST_PER_1K_TOKENS
        print(f"Estimated API cost: ${estimated_cost:.6f}")
        
    os.makedirs("data", exist_ok=True)
    timestamp = datetime.now().strftime('%B_%d_%Y_%I-%M-%S_%p')
    file_name = f'data/JobSearchOutput_{timestamp}.csv'
    all_cities_output_df.to_csv(file_name, index=False)
    print(f"Results saved to {file_name}")

def semantic_pipeline(num_scrolls_linkedin: int = 40, num_scrolls_google: int = 2, 
                      scrape_only: bool = False, use_auth: bool = False, 
                      max_pages: int = 5, use_api: bool = False,
                      distance: int = 50, f_tpr: Optional[str] = None):
    print("Starting semantic discovery pipeline...")
    
    # LinkedIn Search URL provided by user for San Jose
    linkedin_target_url = "https://www.linkedin.com/jobs/search?keywords=Data%20Scientist&location=San%20Jose&geoId=106233382&distance=50&f_TPR=r2592000&position=1&pageNum=0"
    
    job_board_urls = {
        "san_jose": r"https://www.google.com/search?q=data+scientist+jobs+san+jose&ibp=htl;jobs",
    }
    
    # Scrape jobs from sources    
    db = JobDatabase()
    
    if use_auth:
        if not CHROME_USER_DATA_DIR:
            print("Error: CHROME_USER_DATA_DIR not set in .env or config.py. Authentication required.")
            db.close()
            return
            
        print("Using authenticated LinkedIn scraper...")
        driver = get_chrome_driver(user_data_dir=CHROME_USER_DATA_DIR)
        auth_scraper = AuthenticatedLinkedInScraper(driver, db=db)
        try:
            auth_scraper.scrape_jobs(linkedin_target_url, max_pages=max_pages)
        finally:
            driver.quit()
    elif use_api:
        print(f"Using LinkedIn API discovery (max jobs: {max_pages * 25}, distance: {distance}, time: {f_tpr or 'any'})...")
        scrape_linkedin_jobs_api(db=db, keywords="Data Scientist", location="San Jose", 
                                 max_jobs=max_pages * 25, distance=distance, f_tpr=f_tpr)
    else:
        print("Using guest LinkedIn scraper (limited to ~70 jobs)...")
        linkedin_jobs_df = scrape_linkedin_jobs(db=db, max_jobs=50, num_scrolls=num_scrolls_linkedin)
    
    # Optional: Scrape google
    # for city, url in job_board_urls.items():
    #     google_jobs_df = scrape_google_jobs(url, num_scrolls=num_scrolls_google)
    #     dfs.append(google_jobs_df)
    
    if scrape_only:
        print("Scrape-only mode. New jobs have been added to the database.")
        db.close()
        return

    # Retrieve jobs from the database that don't have a similarity score yet
    unscored_jobs_df = pd.read_sql_query("SELECT * FROM jobs WHERE similarity_score IS NULL", db.conn)
    
    if unscored_jobs_df.empty:
        print("No unscored jobs in database to process.")
        db.close()
        return
        
    print(f"Found {len(unscored_jobs_df)} unscored jobs in the database.")
    
    # Rank jobs based on profile and criteria
    target_text = f"{PROFILE_TEXT}\n\nKey Requirements:\n{CRITERIA}"
    
    ranked_df = rank_jobs(unscored_jobs_df, target_text, score_threshold=40.0)
    
    if ranked_df.empty:
        print("No jobs matched the semantic criteria above the threshold.")
        db.close()
        return
        
    # Update DB with similarity scores
    cursor = db.conn.cursor()
    for _, row in ranked_df.iterrows():
        cursor.execute("UPDATE jobs SET similarity_score = ? WHERE job_id = ?", (row['similarity_score'], row['job_id']))
    db.conn.commit()
    db.close()
        
    os.makedirs("data", exist_ok=True)
    timestamp = datetime.now().strftime('%B_%d_%Y_%I-%M-%S_%p')
    file_name = f'data/SemanticJobSearchOutput_{timestamp}.csv'
    ranked_df.to_csv(file_name, index=False)
    
    print("\n--- Top Semantic Matches ---")
    print(ranked_df[['titles', 'companies', 'similarity_score']].head(10))
    print(f"\nResults saved to {file_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Job Finder Pipeline")
    parser.add_argument('--train', action='store_true', help="Run the training pipeline")
    parser.add_argument('--predict', action='store_true', help="Run the inference pipeline")
    parser.add_argument('--evaluate', action='store_true', help="Run the OpenAI evaluation during prediction")
    parser.add_argument('--semantic', action='store_true', help="Run the zero-shot semantic discovery pipeline")
    parser.add_argument('--auth', action='store_true', help="Use authenticated LinkedIn scraper")
    parser.add_argument('--api', action='store_true', help="Use fast LinkedIn API discovery")
    parser.add_argument('--max-pages', type=int, default=5, help="Maximum number of pages/requests to scrape (default: 5)")
    parser.add_argument('--scrape-only', action='store_true', help="Scrape jobs into DB without running inference")
    parser.add_argument('--distance', type=int, default=50, help="Search radius in miles for --api (default: 50)")
    parser.add_argument('--f-tpr', type=str, choices=['r86400', 'r604800', 'r2592000'], help="Time Posted Range for --api: day, week, month")
    parser.add_argument('--scrolls-linkedin', type=int, default=40, help="Number of scrolls for LinkedIn (default: 40)")
    parser.add_argument('--scrolls-google', type=int, default=2, help="Number of scrolls for Google (default: 2)")
    parser.add_argument('--data', type=str, default='data/job_labels_196_rows_20250330.csv', help="Path to training data CSV")
    parser.add_argument('--synthetic', type=str, default='data/synthetic_jobs.csv', help="Path to synthetic data CSV")
    
    args = parser.parse_args()
    
    if args.train:
        train_pipeline(args.data, args.synthetic)
    elif args.predict:
        predict_pipeline(args.evaluate, num_scrolls_linkedin=args.scrolls_linkedin, num_scrolls_google=args.scrolls_google)
    elif args.semantic:
        semantic_pipeline(num_scrolls_linkedin=args.scrolls_linkedin, num_scrolls_google=args.scrolls_google, 
                          scrape_only=args.scrape_only, use_auth=args.auth, max_pages=args.max_pages,
                          use_api=args.api, distance=args.distance, f_tpr=args.f_tpr)
    elif args.scrape_only and not args.semantic:
        semantic_pipeline(num_scrolls_linkedin=args.scrolls_linkedin, num_scrolls_google=args.scrolls_google, 
                          scrape_only=True, use_auth=args.auth, max_pages=args.max_pages,
                          use_api=args.api, distance=args.distance, f_tpr=args.f_tpr)
    else:
        parser.print_help()
