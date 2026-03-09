"""
Integration test for Job Finder Pipeline
"""
import pandas as pd
from job_finder.models import load_and_predict_new_jobs
from job_finder.config import OPENAI_API_KEY
import joblib
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')

def test_pipeline():
    print("Testing Job Finder Pipeline...")
    
    stop_words = set(stopwords.words("english")).union(['said', 'would', 'could', "also", "new"])
    try:
        pretrained_model = joblib.load('models/best_logistic_regression_model.pkl')
        transformers = joblib.load('models/transformers.pkl')
    except FileNotFoundError:
        print("Models not found. Please run the training pipeline first.")
        return

    url = "https://www.google.com/search?client=firefox-b-1-d&sca_esv=1ebb0e033accb772&q=data+scientist+san+francisco&prmd=invmsb&sa=X&biw=1760&bih=875&dpr=1.09&jbr=sep:0&ibp=htl;jobs&ved=2ahUKEwj49KjYwJqGAxWHQzABHdcfACMQudcGKAF6BAgiECk#fpstate=tldetail&htivrt=jobs&htidocid=R2A6Q-IkYWZ1jy46AAAAAA%3D%3D"
    
    print("Scraping 2 jobs from LinkedIn and some from Google...")
    df = load_and_predict_new_jobs(
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
        scrape_google=True,
        linkedin_cap=2,
        count=0,
        num_scrolls_linkedin=0,
        num_scrolls_google=0
    )
    
    if df.empty:
        print("No jobs scraped. Test Failed.")
        return
        
    print(f"Total jobs scraped and classified: {len(df)}")
    print("\nResults Sample:")
    print(df[['source', 'companies', 'titles', 'predictions', 'prediction_probability']].head(4))
    
    print("\nIntegration test finished successfully!")

if __name__ == "__main__":
    test_pipeline()
