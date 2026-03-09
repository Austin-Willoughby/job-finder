"""
Feature engineering, text preprocessing, and vectorization
"""
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.sparse import hstack, csr_matrix

from job_finder.config import KEYWORDS_BINS

def keyword_match(text: str, keyword_bins: dict) -> dict:
    """
    Count occurrences of keywords (in bins) within the given text.
    """
    text_tokens = [word.lower() for word in str(text).split()]
    keyword_counts = {bin_name: 0 for bin_name in keyword_bins}
    
    for bin_name, keywords in keyword_bins.items():
        for keyword in keywords:
            keyword_tokens = [word.lower() for word in keyword.split()]
            if all(token in text_tokens for token in keyword_tokens):
                keyword_counts[bin_name] += 1
    return keyword_counts


def preprocess_text(text: str, stop_words: set) -> str:
    """
    Remove HTML, newlines, non-alphabetic characters, stopwords,
    and apply lemmatization.
    """
    text = str(text)
    text = re.sub(r'<.*?>', '', text)
    text = text.replace('\\n', ' ')
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\\s]', '', text)
    tokens = [word for word in text.split() if word not in stop_words]
    lem = WordNetLemmatizer()
    tokens = [lem.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


def create_corpus(text_series: pd.Series, stop_words: set) -> pd.Series:
    """
    Preprocess a Pandas Series of text.
    """
    return text_series.apply(lambda x: preprocess_text(x, stop_words))


def load_and_preprocess_data(filepath: str, synthetic_filepath: str, stop_words: set) -> pd.DataFrame:
    """
    Load job data from a CSV, preprocess text, generate keyword features,
    and combine with synthetic job data.
    """
    df = pd.read_csv(filepath, encoding='latin-1').drop(['Unnamed: 0'], axis=1, errors='ignore').drop_duplicates(
        subset=['titles', 'companies', 'location', 'desc']
    )
    df['desc'] = df['desc'].map(lambda x: str(x).replace("\\n", " "))
    df['titles'] = df['titles'].map(lambda x: str(x).replace("\\n", " "))
    df['label'] = df['true_labels']
    df = df.drop(['location', 'company_in_description', 'predictions', 'true_labels_5', 'true_labels'], axis=1, errors='ignore')
    
    try:
        df_synthetic = pd.read_csv(synthetic_filepath).drop(['Unnamed: 0'], axis=1, errors='ignore')
        df_synthetic['label'] = 1
        df_synthetic.rename(columns={'Title': 'titles', 'Company': 'companies', 'Description': 'desc'}, inplace=True)
        df = pd.concat([df, df_synthetic]).sample(frac=1).reset_index(drop=True)
    except FileNotFoundError:
        print(f"Warning: {synthetic_filepath} not found. Skipping synthetic data.")
        df = df.sample(frac=1).reset_index(drop=True)
        
    df['word_count'] = df['desc'].apply(lambda x: len(str(x).split(" ")))
    
    keyword_matches = df['desc'].apply(lambda x: keyword_match(x, KEYWORDS_BINS))
    for bin_name in KEYWORDS_BINS:
        df[f'keyword_{bin_name}'] = keyword_matches.apply(lambda x: x[bin_name])
    
    df['total_weighted_keywords'] = (
        df['keyword_environmental'] + df['keyword_CV_autonomous_robotics'] +
        df['keyword_LLM_related'] * 0.7 + df['keyword_geospatial_r_sensing'] +
        df['keyword_energy'] * 0.8 + df['keyword_coding'] * 0.3
    )
    df['weighted_keywords'] = df['total_weighted_keywords'] / df['word_count']
    
    return df


def create_bag_of_words(corpus: pd.Series, titles_corpus: pd.Series, stop_words: set,
                        keyword_features, n_components: int = None):
    """
    Create Bag-of-Words and TF-IDF matrices from the corpus and titles.
    Optionally apply PCA for dimensionality reduction.
    """
    cv_desc = CountVectorizer(max_df=0.8, stop_words=list(stop_words),
                              max_features=100000, ngram_range=(1, 2))
    X_desc = cv_desc.fit_transform(corpus)
    
    cv_titles = CountVectorizer(max_df=0.8, stop_words=list(stop_words),
                                max_features=100000, ngram_range=(1, 2))
    X_title = cv_titles.fit_transform(titles_corpus)
    
    tfidf_transformer_desc = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer_title = TfidfTransformer(smooth_idf=True, use_idf=True)
    
    tfidf_transformer_desc.fit(X_desc)
    tfidf_transformer_title.fit(X_title)
    
    tfidf_desc = tfidf_transformer_desc.transform(X_desc)
    tfidf_title = tfidf_transformer_title.transform(X_title)
    
    # Use MinMaxScaler for keyword features
    scaler = MinMaxScaler()
    keyword_features_scaled = scaler.fit_transform(keyword_features)
    keyword_features_sparse = csr_matrix(keyword_features_scaled)
    
    tfidf = hstack((tfidf_desc, tfidf_title, keyword_features_sparse))
    
    pca_obj = None
    if n_components is not None:
        pca_obj = PCA(n_components=n_components)
        tfidf_reduced = pca_obj.fit_transform(tfidf.toarray())
        tfidf = csr_matrix(tfidf_reduced)
    
    return (tfidf, cv_desc, cv_titles, tfidf_transformer_desc,
            tfidf_transformer_title, scaler, pca_obj)


def preprocess_and_vectorize_data(df: pd.DataFrame, cv_desc: CountVectorizer, cv_titles: CountVectorizer,
                                  tfidf_transformer_desc: TfidfTransformer, tfidf_transformer_title: TfidfTransformer,
                                  scaler: MinMaxScaler, pca: PCA, stop_words: set):
    """
    Preprocess the job descriptions and titles, then vectorize and optionally apply PCA.
    """
    corpus = create_corpus(df['desc'], stop_words)
    titles_corpus = create_corpus(df['titles'], stop_words)
    X_desc = cv_desc.transform(corpus)
    X_title = cv_titles.transform(titles_corpus)
    tfidf_desc = tfidf_transformer_desc.transform(X_desc)
    tfidf_title = tfidf_transformer_title.transform(X_title)
    
    keyword_columns = [
        'keyword_environmental', 'keyword_CV_autonomous_robotics', 
        'keyword_LLM_related', 'keyword_geospatial_r_sensing', 
        'keyword_energy', 'keyword_coding', 'weighted_keywords'
    ]
    keyword_features = df[keyword_columns].values
    keyword_features_scaled = scaler.transform(keyword_features)
    keyword_features_sparse = csr_matrix(keyword_features_scaled)
    
    tfidf = hstack((tfidf_desc, tfidf_title, keyword_features_sparse))
    
    if pca is not None:
        tfidf_reduced = pca.transform(tfidf.toarray())
        tfidf = csr_matrix(tfidf_reduced)
    
    return tfidf
