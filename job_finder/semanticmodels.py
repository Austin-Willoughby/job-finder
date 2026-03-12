"""
Semantic modeling using vector embeddings to find jobs matching user criteria.
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from llama_cpp import Llama
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

MODEL_PATH = "models/Qwen3-Embedding-0.6B-Q8_0.gguf"
_model = None

def get_embedding_model() -> Llama:
    """
    Lazy load the llama.cpp model to avoid unnecessary overhead.
    """
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please run the download script first.")
            
        print(f"Loading '{MODEL_PATH}' via Vulkan... (Pooling VRAM + System RAM)")
        
        # n_gpu_layers=-1 offloads all possible layers to the GPU (Vulkan)
        # n_ctx=8192 accommodates long job descriptions
        _model = Llama(
            model_path=MODEL_PATH,
            embedding=True,
            n_gpu_layers=-1,
            n_ctx=8192,
            verbose=False # Set to True if you want to see detailed memory allocation stats
        )
    return _model

def embed_texts(texts: list[str], is_query: bool = False) -> np.ndarray:
    """
    Convert a list of strings into a numpy array of vector embeddings using llama.cpp.
    """
    model = get_embedding_model()
    embeddings = []
    total = len(texts)
    for i, text in enumerate(texts):
        # Progress counter for job descriptions (not for single queries)
        count = i + 1
        if not is_query:
            if count in [1, 5, 10, 20, 50, 100] or count % 500 == 0 or count == total:
                print(f"  > Processing item {count}/{total}...")

        # Format the text according to the specific Qwen Instruct template
        if is_query:
            # The query needs the Instruct prefix for maximum retrieval accuracy
            formatted_text = f"Instruct: Given a query, retrieve relevant documents that answer the query\nQuery: {text}<|endoftext|>"
        else:
            # Documents only need the suffix
            formatted_text = f"{text}<|endoftext|>"
            
        output = model.create_embedding(formatted_text)
        embeddings.append(output['data'][0]['embedding'])
        
    return np.array(embeddings)

def calculate_relevance(jobs_df: pd.DataFrame, profile_embeddings: dict) -> pd.DataFrame:
    """
    Calculates the cosine similarity between multiple pre-embedded profiles
    and all job descriptions in the dataframe.
    profile_embeddings: dict with keys like 'geospatial', 'energy', etc. and values as numpy arrays.
    """
    if jobs_df.empty:
        return jobs_df
        
    df = jobs_df.copy()
    
    # Fill NaN values with empty string to prevent embedding errors
    df['desc'] = df['desc'].fillna('')
    
    print(f"Embedding {len(df)} job descriptions...")
    job_embeddings = embed_texts(df['desc'].tolist(), is_query=False)
    
    for key, target_embedding in profile_embeddings.items():
        print(f"Calculating similarity scores for '{key}'...")
        similarities = cosine_similarity(target_embedding, job_embeddings)
        col_name = f"score_{key}"
        df[col_name] = np.round(similarities[0] * 100, 2)
    
    # Also set a main similarity_score (e.g., the maximum score across all profiles)
    score_cols = [f"score_{key}" for key in profile_embeddings.keys()]
    df['similarity_score'] = df[score_cols].max(axis=1)
    
    return df

def rank_jobs(jobs_df: pd.DataFrame, profiles: dict, score_threshold: float = 0.0) -> pd.DataFrame:
    """
    Filter and rank jobs based on multiple profiles.
    Returns a DataFrame with all scores, sorted by the maximum similarity_score.
    """
    scored_df = calculate_relevance(jobs_df, profiles)
    
    if scored_df.empty:
        return scored_df
        
    # Filter: at least one profile must meet the threshold
    filtered_df = scored_df[scored_df['similarity_score'] >= score_threshold]
    ranked_df = filtered_df.sort_values(by='similarity_score', ascending=False).reset_index(drop=True)
    
    return ranked_df
