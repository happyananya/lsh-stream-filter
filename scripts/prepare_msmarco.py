import os
import torch
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def main():
    print("Starting MS MARCO Phase 1 baseline generation...")
    
    # Ensure we use CUDA if available, but prompt user if it fails.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("WARNING: CUDA is not available. Embedding and exact k-NN will be slow.")
        print("Please check your PyTorch installation with CUDA support.")

    # 1. Load Data
    print("Loading microsoft/ms_marco v2.1 from Hugging Face...")
    ds = load_dataset("microsoft/ms_marco", "v2.1")
    
    print("Extracting 1,000,000 unique passages from train split...")
    unique_passages = set()
    for row in ds['train']:
        unique_passages.update(row['passages']['passage_text'])
        if len(unique_passages) >= 1000000:
            break
    
    passages = list(unique_passages)[:1000000]
    passage_ids = list(range(len(passages)))
    print(f"Collected {len(passages)} passages.")
    
    print("Extracting 6,980 queries from validation split...")
    # Select the first 6980 queries for consistency with typical eval sets
    val_subset = ds['validation'].select(range(6980))
    queries = val_subset['query']
    query_ids = val_subset['query_id']
    print(f"Collected {len(queries)} queries.")
    
    # 2. Embed Data
    print(f"Loading SentenceTransformer on {device}...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    
    # Encode passages
    print("Encoding passages...")
    base_embeddings = model.encode(passages, batch_size=512, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    
    # Encode queries
    print("Encoding queries...")
    query_embeddings = model.encode(queries, batch_size=512, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    
    # 3. Compute Ground Truth Exact k-NN
    print("Computing exact top-100 ground truth on GPU using PyTorch...")
    # Base is 1M x 384 = ~1.5 GB. This easily fits in GPU memory.
    
    k = 100
    base_tensor = torch.tensor(base_embeddings, device=device)
    query_tensor = torch.tensor(query_embeddings, device=device)
    
    groundtruth = np.zeros((len(queries), k), dtype=np.int32)
    
    # Chunking queries to prevent OOM
    batch_size_query = 100
    print(f"Computing top-k in batches of {batch_size_query} queries...")
    for i in tqdm(range(0, len(queries), batch_size_query), desc="Exact k-NN"):
        q_batch = query_tensor[i:i+batch_size_query]
        # Inner product since embeddings are normalized
        scores = torch.matmul(q_batch, base_tensor.T)
        topk_scores, topk_indices = torch.topk(scores, k=k, dim=-1)
        groundtruth[i:i+batch_size_query] = topk_indices.cpu().numpy()
        
    # 4. Save to disk
    output_dir = "data/msmarco"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving data to {output_dir}...")
    np.save(os.path.join(output_dir, "base.npy"), base_embeddings)
    np.save(os.path.join(output_dir, "query.npy"), query_embeddings)
    np.save(os.path.join(output_dir, "groundtruth.npy"), groundtruth)
    
    # Also save passage and query IDs for traceability
    np.save(os.path.join(output_dir, "passage_ids.npy"), np.array(passage_ids, dtype=object))
    np.save(os.path.join(output_dir, "query_ids.npy"), np.array(query_ids, dtype=object))
    
    print("Phase 1 complete! Baseline generated successfully.")

if __name__ == "__main__":
    main()
