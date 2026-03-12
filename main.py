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
from job_finder.semanticmodels import rank_jobs, embed_texts
from job_finder.scraper import scrape_linkedin_jobs, scrape_linkedin_jobs_api, scrape_google_jobs, get_chrome_driver
from job_finder.database import JobDatabase
from job_finder.scrapers.linkedin import AuthenticatedLinkedInScraper
from job_finder.config import PROFILES, CRITERIA, CHROME_USER_DATA_DIR, DEFAULT_LOCATION, DEFAULT_DISTANCE
import job_finder.evaluator as evaluator
import numpy as np
import logging
from job_finder.logging_config import setup_logging, get_logger

logger = get_logger(__name__)

def train_pipeline(data_path: str, synthetic_path: str):
    logger.info("Starting training pipeline...")
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
    logger.info("Starting prediction pipeline...")
    stop_words = set(stopwords.words("english")).union(['said', 'would', 'could', "also", "new"])
    
    try:
        pretrained_model = joblib.load('models/best_logistic_regression_model.pkl')
        transformers = joblib.load('models/transformers.pkl')
    except FileNotFoundError:
        logger.error("Models not found. Please run the training pipeline first.")
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
        logger.info("Evaluating desirability with OpenAI...")
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
                      scrape_only: bool = False, score_only: bool = False,
                      use_auth: bool = False, 
                      max_pages: int = 5, use_api: bool = False,
                      distance: int = DEFAULT_DISTANCE, f_tpr: Optional[str] = 'r2592000',
                      keywords: str = "Data Scientist", location: str = DEFAULT_LOCATION):
    logger.info("Starting semantic discovery pipeline...")
    
    if not score_only:
        # LinkedIn Search URL provided by user for San Jose
        linkedin_target_url = f"https://www.linkedin.com/jobs/search?keywords={keywords.replace(' ', '%20')}&location={location.replace(' ', '%20')}&distance={distance}&f_TPR={f_tpr}"
        logger.info(f"Search URL: {linkedin_target_url}")
        
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
            print(f"Using LinkedIn API discovery for {keywords} in {location} (max jobs: {max_pages * 25}, distance: {distance}, time: {f_tpr or 'any'})...")
            scrape_linkedin_jobs_api(db=db, keywords=keywords, location=location, 
                                     max_jobs=max_pages * 25, distance=distance, f_tpr=f_tpr)
        else:
            print("Using guest LinkedIn scraper (limited to ~70 jobs)...")
            linkedin_jobs_df = scrape_linkedin_jobs(db=db, max_jobs=50, num_scrolls=num_scrolls_linkedin)
        
        if scrape_only:
            logger.info("Scrape-only mode. New jobs have been added to the database.")
            db.close()
            return
        db.close() # Close for now, will reopen or use pd.read_sql
    else:
        logger.info("Score-only mode. Skipping scraping and processing existing database jobs.")

    db = JobDatabase()
    # Or we can re-score all jobs if needed. For now, let's just score unscored ones.
    unscored_jobs_df = pd.read_sql_query("SELECT * FROM jobs WHERE similarity_score IS NULL", db.conn)
    
    if unscored_jobs_df.empty:
        logger.info("No unscored jobs in database to process.")
        db.close()
        return
        
    logger.info(f"Found {len(unscored_jobs_df)} unscored jobs in the database.")
    
    # Pre-calculate embeddings for anchor profiles
    logger.info("Pre-calculating embeddings for anchor profiles...")
    profile_embeddings = {}
    for key, p in PROFILES.items():
        combined_text = f"{p['text']}\n\nKey Requirements:\n{CRITERIA}"
        profile_embeddings[key] = embed_texts([combined_text], is_query=True)
    
    # NEW logic: Calculate relevance for ALL unscored jobs first
    # This ensures we don't re-embed them next time even if they don't meet the threshold
    from job_finder.semanticmodels import calculate_relevance
    scored_df = calculate_relevance(unscored_jobs_df, profile_embeddings)
    
    # Update DB with similarity scores for ALL processed jobs
    logger.info(f"Updating database with scores for {len(scored_df)} jobs in batches...")
    
    # Batch update to avoid N+1 queries while ensuring progress is saved
    batch_size = 500
    cursor = db.conn.cursor()
    
    for i in range(0, len(scored_df), batch_size):
        batch = scored_df.iloc[i:i+batch_size]
        update_data = []
        for _, row in batch.iterrows():
            update_data.append((
                row['similarity_score'],
                row['score_geospatial'],
                row['score_energy'],
                row['score_cv_robotics'],
                row['score_llm_science'],
                row['job_id']
            ))
        
        cursor.executemany("""
            UPDATE jobs 
            SET similarity_score = ?, 
                score_geospatial = ?, 
                score_energy = ?, 
                score_cv_robotics = ?, 
                score_llm_science = ? 
            WHERE job_id = ?
        """, update_data)
        
        db.conn.commit()
        logger.info(f"  > Committed scores for items {i+1} to {min(i+batch_size, len(scored_df))}...")
    
    db.close()

    # Sort and rank jobs for the final report
    ranked_df = scored_df.sort_values(by='similarity_score', ascending=False).reset_index(drop=True)
    
    if ranked_df.empty:
        logger.info(f"Analysis complete. No jobs were found to process.")
        return
        
    os.makedirs("data", exist_ok=True)
    timestamp = datetime.now().strftime('%B_%d_%Y_%I-%M-%S_%p')
    file_name = f'data/SemanticJobSearchOutput_{timestamp}.csv'
    ranked_df.to_csv(file_name, index=False)
    
    logger.info("\n--- Top Semantic Matches (Combined) ---")
    logger.info(ranked_df[['titles', 'companies', 'similarity_score']].head(10))
    
    # Show top 3 for each specific profile
    for key, p in PROFILES.items():
        score_col = f"score_{key}"
        top_p = ranked_df.sort_values(by=score_col, ascending=False).head(3)
        logger.info(f"\n--- Top 3: {p['name']} ---")
        logger.info(top_p[['titles', 'companies', score_col]])

    logger.info(f"\nResults saved to {file_name}")

if __name__ == '__main__':
    setup_logging()
    parser = argparse.ArgumentParser(description="Job Finder Pipeline")
    parser.add_argument('--train', action='store_true', help="Run the training pipeline")
    parser.add_argument('--predict', action='store_true', help="Run the inference pipeline")
    parser.add_argument('--evaluate', action='store_true', help="Run the OpenAI evaluation during prediction")
    parser.add_argument('--semantic', action='store_true', help="Run the zero-shot semantic discovery pipeline")
    parser.add_argument('--auth', action='store_true', help="Use authenticated LinkedIn scraper")
    parser.add_argument('--api', action='store_true', help="Use fast LinkedIn API discovery")
    parser.add_argument('--max-pages', type=int, default=5, help="Maximum number of pages/requests to scrape (default: 5)")
    parser.add_argument('--scrape-only', action='store_true', help="Scrape jobs into DB without running inference")
    parser.add_argument('--score-only', action='store_true', help="Run semantic scoring on DB without scraping new jobs")
    parser.add_argument('--distance', type=int, default=DEFAULT_DISTANCE, help=f"Search radius in miles for --api (default: {DEFAULT_DISTANCE})")
    parser.add_argument('--f-tpr', type=str, choices=['r86400', 'r172800', 'r604800', 'r2592000'], default='r2592000', help="Time Posted Range for --api: day, 2days, week, month (default: r2592000 for month)")
    parser.add_argument('--keywords', type=str, default="Data Scientist", help="Keywords for search (default: 'Data Scientist')")
    parser.add_argument('--location', type=str, default=DEFAULT_LOCATION, help=f"Location for search (default: '{DEFAULT_LOCATION}')")
    parser.add_argument('--scrolls-linkedin', type=int, default=40, help="Number of scrolls for LinkedIn (default: 40)")
    parser.add_argument('--scrolls-google', type=int, default=2, help="Number of scrolls for Google (default: 2)")
    parser.add_argument('--data', type=str, default='data/job_labels_196_rows_20250330.csv', help="Path to training data CSV")
    parser.add_argument('--synthetic', type=str, default='data/synthetic_jobs.csv', help="Path to synthetic data CSV")
    parser.add_argument('--verbose', action='store_true', help="Enable detailed logging in terminal")
    
    args = parser.parse_args()
    
    # Initialize logging with dynamic level
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    if args.train:
        train_pipeline(args.data, args.synthetic)
    elif args.predict:
        predict_pipeline(args.evaluate, num_scrolls_linkedin=args.scrolls_linkedin, num_scrolls_google=args.scrolls_google)
    elif args.semantic:
        semantic_pipeline(num_scrolls_linkedin=args.scrolls_linkedin, num_scrolls_google=args.scrolls_google, 
                          scrape_only=args.scrape_only, score_only=args.score_only,
                          use_auth=args.auth, max_pages=args.max_pages,
                          use_api=args.api, distance=args.distance, f_tpr=args.f_tpr,
                          keywords=args.keywords, location=args.location)
    elif args.score_only:
        semantic_pipeline(score_only=True)
    elif args.scrape_only and not args.semantic:
        semantic_pipeline(scrape_only=True, use_auth=args.auth, max_pages=args.max_pages,
                          use_api=args.api, distance=args.distance, f_tpr=args.f_tpr,
                          keywords=args.keywords, location=args.location)
    else:
        parser.print_help()
