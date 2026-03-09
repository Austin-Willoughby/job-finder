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
    
    for text in texts:
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

def calculate_relevance(jobs_df: pd.DataFrame, target_text: str) -> pd.DataFrame:
    """
    Calculates the cosine similarity between the target_text (profile)
    and all job descriptions in the dataframe.
    """
    if jobs_df.empty:
        return jobs_df
        
    df = jobs_df.copy()
    
    # Fill NaN values with empty string to prevent embedding errors
    df['desc'] = df['desc'].fillna('')
    
    print(f"Embedding {len(df)} job descriptions...")
    job_embeddings = embed_texts(df['desc'].tolist(), is_query=False)
    
    print("Embedding target profile/criteria...")
    target_embedding = embed_texts([target_text], is_query=True)
    
    print("Calculating similarity scores...")
    # Calculate cosine distance
    similarities = cosine_similarity(target_embedding, job_embeddings)
    
    df['similarity_score'] = np.round(similarities[0] * 100, 2)
    
    return df

def rank_jobs(jobs_df: pd.DataFrame, target_text: str, score_threshold: float = 40.0) -> pd.DataFrame:
    """
    Filter and rank jobs based on how semantically similar their description
    is to the target text.
    """
    scored_df = calculate_relevance(jobs_df, target_text)
    
    if scored_df.empty:
        return scored_df
        
    # Filter and sort
    filtered_df = scored_df[scored_df['similarity_score'] >= score_threshold]
    ranked_df = filtered_df.sort_values(by='similarity_score', ascending=False).reset_index(drop=True)
    
    return ranked_df
