"""
Machine learning model training, evaluation, and inference
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from job_finder.config import KEYWORDS_BINS
from job_finder.scraper import scrape_linkedin_jobs, scrape_google_jobs
from job_finder.features import keyword_match, preprocess_and_vectorize_data

def evaluate_models(tfidf_labelled, targets):
    """
    Train and evaluate several classifiers using cross-validation.
    Display evaluation metrics and confusion matrix for the best SVM.
    Returns the trained SVM and Logistic Regression models.
    """
    X_train, X_test, y_train, y_test = train_test_split(tfidf_labelled, targets, random_state=0, stratify=targets)
    
    best_estimators = {
        'SVM': svm.SVC(C=10, coef0=0.0, degree=3, gamma='scale', kernel='sigmoid',
                       random_state=0, probability=True),
        'Logistic Regression': LogisticRegression(C=10, max_iter=1000, penalty='l2',
                                                    solver='saga', random_state=0),
        'XGBoost': XGBClassifier(learning_rate=0.1, max_depth=4, n_estimators=100,
                                 use_label_encoder=False, eval_metric='mlogloss', random_state=0)
    }
    
    models = {
        'SVM': best_estimators['SVM'],
        'Logistic Regression': best_estimators['Logistic Regression'],
        'Decision Tree': DecisionTreeClassifier(random_state=0),
        'Random Forest': RandomForestClassifier(random_state=0),
        'XGBoost': best_estimators['XGBoost']
    }
    
    scoring = ['accuracy', 'recall', 'precision', 'roc_auc']
    metrics_list = ['test_accuracy', 'test_recall', 'test_precision', 'test_roc_auc']
    cv_scores = {metric: [] for metric in metrics_list}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        cv_results = cross_validate(model, tfidf_labelled, targets, cv=5,
                                    scoring=scoring, return_train_score=True)
        for metric in metrics_list:
            cv_scores[metric].append(cv_results[metric].mean())
    
    df_metrics = pd.DataFrame(cv_scores).T
    df_metrics.columns = models.keys()
    ax = df_metrics.plot(kind="bar", figsize=(12, 8))
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9),
                    textcoords='offset points')
    
    plt.legend(models.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.show()
    
    svm_model = best_estimators['SVM']
    svm_model.fit(tfidf_labelled, targets)
    
    logistic_regression_model = best_estimators['Logistic Regression']
    logistic_regression_model.fit(tfidf_labelled, targets)
    
    y_pred = svm_model.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    # plt.show()
    
    return svm_model, logistic_regression_model

def plot_metrics(metrics: dict):
    """
    Plot a bar chart for the provided metrics.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(metrics.keys(), metrics.values())
    for i, v in enumerate(metrics.values()):
        ax.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom')
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Model Evaluation Metrics')
    plt.show()

def load_and_predict_new_jobs(
    job_board_url: str,
    stop_words: set,
    cv_desc: CountVectorizer,
    cv_titles: CountVectorizer,
    tfidf_transformer_desc: TfidfTransformer,
    tfidf_transformer_title: TfidfTransformer,
    scaler: MinMaxScaler,
    pca: PCA,
    pretrained_model,
    include_linkedin: bool = False,   # If True, scrape LinkedIn jobs
    scrape_google: bool = True,         # If True, scrape Google jobs
    count: int = 0,
    linkedin_cap: int = 5,
    num_scrolls_linkedin: int = 40,
    num_scrolls_google: int = 2
) -> pd.DataFrame:
    """
    Scrape new job postings from LinkedIn and/or Google, preprocess them,
    and predict using the pretrained model.
    """
    dfs = []
    
    if count == 0 and include_linkedin:
        linkedin_jobs_df = scrape_linkedin_jobs(max_jobs=linkedin_cap, num_scrolls=num_scrolls_linkedin)
        linkedin_jobs_df['source'] = 'LinkedIn'
        print("LinkedIn data scraped and included.")
        dfs.append(linkedin_jobs_df)
    
    if scrape_google:
        google_jobs_df = scrape_google_jobs(job_board_url, num_scrolls=num_scrolls_google)
        dfs.append(google_jobs_df)
    
    if not dfs:
        return pd.DataFrame()  # No data scraped
    
    jobs_df = pd.concat(dfs, ignore_index=True)
    jobs_df['company_in_description'] = jobs_df.apply(
        lambda row: row['companies'] in row['desc'] if isinstance(row['companies'], str) and isinstance(row['desc'], str) else False, axis=1
    )
    
    # Compute keyword matches for the new job data
    keyword_matches = jobs_df['desc'].apply(lambda x: keyword_match(x, KEYWORDS_BINS))
    for bin_name in KEYWORDS_BINS:
        jobs_df[f'keyword_{bin_name}'] = keyword_matches.apply(lambda x: x[bin_name])
    
    jobs_df['word_count'] = jobs_df['desc'].apply(lambda x: len(str(x).split(" ")))
    jobs_df['total_weighted_keywords'] = (
        jobs_df['keyword_environmental'] + jobs_df['keyword_CV_autonomous_robotics'] +
        jobs_df['keyword_LLM_related'] * 0.7 + jobs_df['keyword_geospatial_r_sensing'] +
        jobs_df['keyword_energy'] * 0.8 + jobs_df['keyword_coding'] * 0.3
    )
    jobs_df['weighted_keywords'] = jobs_df['total_weighted_keywords'] / jobs_df['word_count']
    
    tfidf_new = preprocess_and_vectorize_data(
        jobs_df, cv_desc, cv_titles, tfidf_transformer_desc, tfidf_transformer_title, scaler, pca, stop_words
    )
    
    new_predictions = pretrained_model.predict(tfidf_new)
    jobs_df['predictions'] = new_predictions
    new_predictions_logistic_proba = pretrained_model.predict_proba(tfidf_new)[:, 1]
    jobs_df['prediction_probability'] = new_predictions_logistic_proba
    
    print(jobs_df['predictions'].value_counts())
    return jobs_df
