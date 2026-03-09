import os
import sys
from huggingface_hub import hf_hub_download

def main():
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Official Qwen Repository
    repo_id = "Qwen/Qwen3-Embedding-0.6B-GGUF"
    filename = "Qwen3-Embedding-0.6B-Q8_0.gguf"
    local_dir = "models"
    
    print(f"Downloading {filename} from {repo_id}...")
    try:
        # Download using the Python API, which is more robust than the CLI
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"Success! Model downloaded to: {path}")
        
    except Exception as e:
        print(f"\nError downloading model: {e}")
        print("\nPossible solutions:")
        print("1. Check your internet connection.")
        print("2. Verify the repo name on Hugging Face: https://huggingface.co/models?search=qwen3+embedding+gguf")
        print("3. Ensure you have 'huggingface_hub' installed: pip install huggingface_hub")

if __name__ == "__main__":
    main()
