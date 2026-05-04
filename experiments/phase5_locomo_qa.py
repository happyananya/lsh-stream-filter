"""
Phase 5: LoCoMo LLM Question Answering Benchmark
==================================================
Tests whether the retention policy preserves enough conversational context 
for an LLM (Gemini) to correctly answer questions about earlier sessions.

Pipeline:
  1. Load LoCoMo conversations (multi-session dialogues with QA annotations)
  2. Embed each dialogue turn using sentence-transformers
  3. Stream turns through each retention policy with bounded memory
  4. For each QA question, retrieve top-K context from retained memory
  5. Prompt Gemini to answer; score against ground truth
  6. Compare answer accuracy across policies at different memory budgets
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple

# Add parent dirs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

import faiss
import requests

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'locomo')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'phase5_locomo')
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
OLLAMA_MODEL = 'qwen2.5:7b'
RETRIEVE_K = 5  # top-K chunks to retrieve for RAG context
BUDGET_FRACTIONS = [0.10, 0.25, 0.50, 1.00]

def _get_ollama_base_url():
    """Resolve Ollama URL — handles both native and WSL environments."""
    import subprocess, socket
    
    candidates = ['127.0.0.1', 'localhost']
    
    # 1. Try Windows host IP via hostname.exe (most reliable for WSL)
    try:
        # hostname.exe -I returns all IPs on the Windows host
        res = subprocess.run(['hostname.exe', '-I'], capture_output=True, text=True, timeout=1)
        for ip in res.stdout.split():
            candidates.append(ip.strip())
    except: pass

    # 2. Try the nameserver from resolv.conf
    try:
        with open('/etc/resolv.conf', 'r') as f:
            for line in f:
                if 'nameserver' in line:
                    candidates.append(line.split()[-1].strip())
    except: pass
    
    # 3. Try common WSL host aliases
    candidates.extend(['host.docker.internal', 'gateway.docker.internal'])

    # Try each candidate
    print("  Searching for Ollama...")
    for host in dict.fromkeys(candidates):
        url = f'http://{host}:11434'
        try:
            r = requests.get(f"{url}/api/tags", timeout=0.3)
            if r.ok:
                print(f"  Ollama found at {url}")
                return url
        except:
            continue

    # Fallback
    return 'http://127.0.0.1:11434'

OLLAMA_BASE_URL = _get_ollama_base_url()
OLLAMA_URL = f'{OLLAMA_BASE_URL}/api/generate'



# ─────────────────────────────────────────────────────────────
# Step 1: Download and parse LoCoMo
# ─────────────────────────────────────────────────────────────
def download_locomo():
    """Download locomo10.json from the GitHub repo if not cached."""
    os.makedirs(DATA_DIR, exist_ok=True)
    fpath = os.path.join(DATA_DIR, 'locomo10.json')
    if os.path.exists(fpath):
        print(f"  LoCoMo data already cached at {fpath}")
        return fpath
    
    print("  Downloading LoCoMo dataset from GitHub...")
    import urllib.request
    url = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
    urllib.request.urlretrieve(url, fpath)
    print(f"  Saved to {fpath}")
    return fpath


def parse_locomo(fpath: str) -> List[Dict]:
    """
    Parse LoCoMo conversations into a list of structured conversations.
    Each conversation has:
      - conversation_id
      - turns: list of {speaker, text, session_id, dia_id}
      - qa: list of {question, answer, category, evidence}
    """
    with open(fpath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    conversations = []
    for sample in data:
        conv_id = sample.get('sample_id', 'unknown')
        conv_data = sample.get('conversation', {})
        
        # Extract speaker names
        speaker_a = conv_data.get('speaker_a', 'Speaker A')
        speaker_b = conv_data.get('speaker_b', 'Speaker B')
        
        # Extract all turns across sessions
        turns = []
        session_keys = sorted([k for k in conv_data.keys() if k.startswith('session_') and not k.endswith('_date_time')])
        
        for session_key in session_keys:
            session_id = session_key  # e.g., 'session_1'
            session_turns = conv_data[session_key]
            if isinstance(session_turns, list):
                for turn in session_turns:
                    text = turn.get('text', '')
                    if text.strip():
                        turns.append({
                            'speaker': turn.get('speaker', 'unknown'),
                            'text': text,
                            'session_id': session_id,
                            'dia_id': turn.get('dia_id', ''),
                        })
        
        # Extract QA pairs
        qa_pairs = sample.get('qa', [])
        if isinstance(qa_pairs, dict):
            qa_pairs = [qa_pairs]
        
        # Filter to text-based QA only (skip adversarial/unanswerable for cleaner eval)
        valid_qa = []
        for qa in qa_pairs:
            if isinstance(qa, dict) and qa.get('question') and qa.get('answer'):
                cat = qa.get('category', '')
                if cat != 'adversarial':
                    valid_qa.append({
                        'question': qa['question'],
                        'answer': qa['answer'],
                        'category': cat,
                        'evidence': qa.get('evidence', []),
                    })
        
        if turns and valid_qa:
            conversations.append({
                'conversation_id': conv_id,
                'speaker_a': speaker_a,
                'speaker_b': speaker_b,
                'turns': turns,
                'qa': valid_qa,
            })
    
    return conversations


# ─────────────────────────────────────────────────────────────
# Step 2: Embed conversation turns
# ─────────────────────────────────────────────────────────────
def embed_turns(conversations: List[Dict]) -> List[Dict]:
    """Embed all turns using sentence-transformers. Cache to disk."""
    cache_path = os.path.join(DATA_DIR, 'turn_embeddings.npz')
    
    if os.path.exists(cache_path):
        print("  Loading cached turn embeddings...")
        loaded = np.load(cache_path, allow_pickle=True)
        embeddings = loaded['embeddings']
        turn_texts = loaded['texts'].tolist()
        # Reassign embeddings back to conversations
        idx = 0
        for conv in conversations:
            for turn in conv['turns']:
                turn['embedding'] = embeddings[idx]
                idx += 1
        return conversations
    
    print("  Embedding conversation turns with sentence-transformers...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    all_texts = []
    for conv in conversations:
        for turn in conv['turns']:
            # Include speaker name for better disambiguation
            full_text = f"{turn['speaker']}: {turn['text']}"
            all_texts.append(full_text)
    
    embeddings = model.encode(all_texts, show_progress_bar=True, batch_size=256,
                              normalize_embeddings=True)
    
    # Assign back
    idx = 0
    for conv in conversations:
        for turn in conv['turns']:
            turn['embedding'] = embeddings[idx]
            idx += 1
    
    # Cache
    np.savez(cache_path, embeddings=embeddings, texts=np.array(all_texts, dtype=object))
    print(f"  Cached {len(all_texts)} turn embeddings to {cache_path}")
    
    return conversations


# ─────────────────────────────────────────────────────────────
# Step 3: Stream through retention policy
# ─────────────────────────────────────────────────────────────
def stream_conversation(policy, turns: List[Dict]):
    """Stream turns through a policy in chronological order."""
    for i, turn in enumerate(turns):
        policy.insert(turn['embedding'].astype(np.float32), item_id=i)


# ─────────────────────────────────────────────────────────────
# Step 4: Retrieve context for a question
# ─────────────────────────────────────────────────────────────
def retrieve_context(question_embedding: np.ndarray, 
                     kept_set: np.ndarray, kept_ids: np.ndarray,
                     turns: List[Dict], k: int = 5) -> List[str]:
    """Retrieve top-K turns from the retained set via cosine similarity."""
    if len(kept_set) == 0:
        return []
    
    index = faiss.IndexFlatIP(kept_set.shape[1])
    index.add(kept_set.astype(np.float32))
    _, I = index.search(question_embedding.reshape(1, -1).astype(np.float32), 
                        min(k, len(kept_set)))
    
    context_turns = []
    for local_idx in I[0]:
        if local_idx >= 0:
            turn_idx = kept_ids[local_idx]
            turn = turns[turn_idx]
            context_turns.append(f"[{turn['session_id']}] {turn['speaker']}: {turn['text']}")
    
    return context_turns


def setup_ollama():
    """Verify Ollama is running and the model is available."""
    try:
        resp = requests.get(f'{OLLAMA_BASE_URL}/api/tags', timeout=5)
        resp.raise_for_status()
        models = [m['name'] for m in resp.json().get('models', [])]
        if OLLAMA_MODEL not in models:
            print(f"  WARNING: {OLLAMA_MODEL} not found. Available: {models}")
            print(f"  Run: ollama pull {OLLAMA_MODEL}")
        return True
    except Exception as e:
        msg = f"Ollama not reachable at {OLLAMA_BASE_URL}: {e}"
        if "Connection refused" in str(e):
            msg += ("\n\n  HINT: If you are in WSL, Ollama on Windows needs OLLAMA_HOST set to 0.0.0.0\n"
                    "  Run this in Windows PowerShell and restart Ollama:\n"
                    "  [System.Environment]::SetEnvironmentVariable('OLLAMA_HOST', '0.0.0.0', 'User')")
        raise RuntimeError(msg)


def query_ollama(prompt: str) -> str:
    """Send a prompt to Ollama and return the response text."""
    try:
        resp = requests.post(OLLAMA_URL, json={
            'model': OLLAMA_MODEL,
            'prompt': prompt,
            'stream': False,
        }, timeout=120)
        resp.raise_for_status()
        return resp.json().get('response', '').strip()
    except Exception as e:
        return f"ERROR: {e}"


def ask_llm(question: str, context_chunks: List[str], 
            speaker_a: str, speaker_b: str) -> str:
    """Ask the LLM to answer a question given retrieved context."""
    context_str = "\n".join(context_chunks) if context_chunks else "(No context available)"
    
    prompt = f"""You are evaluating a memory system for a conversational AI agent.
Below are retrieved memory fragments from a long conversation between {speaker_a} and {speaker_b}.

--- Retrieved Memory ---
{context_str}
--- End Memory ---

Based ONLY on the memory above, answer the following question concisely.
If the memory does not contain enough information, say "I don't know."

Question: {question}
Answer:"""
    
    return query_ollama(prompt)


# ─────────────────────────────────────────────────────────────
# Step 6: Score answers
# ─────────────────────────────────────────────────────────────
def score_answer(predicted: str, ground_truth: str) -> float:
    """
    Score a predicted answer against ground truth.
    Uses exact substring match as primary scorer, with LLM-judge as fallback.
    Returns 1.0 for correct, 0.0 for wrong.
    """
    pred_lower = str(predicted).lower().strip()
    gt_lower = str(ground_truth).lower().strip()
    
    # Exact or substring match
    if gt_lower in pred_lower or pred_lower in gt_lower:
        return 1.0
    
    # LLM-judge for fuzzy matching
    judge_prompt = f"""You are judging if a predicted answer is semantically equivalent to the ground truth answer.

Ground truth: {ground_truth}
Predicted: {predicted}

Are they semantically equivalent? Answer only "YES" or "NO"."""
    judge_resp = query_ollama(judge_prompt)
    if 'yes' in judge_resp.lower():
        return 1.0
    
    return 0.0


# ─────────────────────────────────────────────────────────────
# Main experiment loop
# ─────────────────────────────────────────────────────────────
def run_phase5():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("Phase 5: LoCoMo LLM-QA Benchmark")
    print("=" * 60)
    
    # 1. Load data
    print("\n--- Step 1: Loading LoCoMo ---")
    fpath = download_locomo()
    conversations = parse_locomo(fpath)
    print(f"  Loaded {len(conversations)} conversations")
    total_turns = sum(len(c['turns']) for c in conversations)
    total_qa = sum(len(c['qa']) for c in conversations)
    print(f"  Total turns: {total_turns}, Total QA pairs: {total_qa}")
    
    # 2. Embed
    print("\n--- Step 2: Embedding turns ---")
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer(EMBEDDING_MODEL)
    conversations = embed_turns(conversations)
    dim = conversations[0]['turns'][0]['embedding'].shape[0]
    print(f"  Embedding dim: {dim}")
    
    # 3. Setup Ollama
    print("\n--- Step 3: Setting up Ollama ---")
    setup_ollama()
    print(f"  Using model: {OLLAMA_MODEL} (local)")
    
    # 4. Run experiments
    print("\n--- Step 4: Running experiments ---")
    
    from retention_policies import BucketOccupancyRetention
    from baselines import FIFORetention, ReservoirSamplingRetention
    
    all_results = []
    
    for conv in conversations:
        conv_id = conv['conversation_id']
        turns = conv['turns']
        N = len(turns)
        qa_pairs = conv['qa']
        
        print(f"\n  Conversation: {conv_id} ({N} turns, {len(qa_pairs)} questions)")
        
        # Embed questions
        q_texts = [qa['question'] for qa in qa_pairs]
        q_embeddings = st_model.encode(q_texts, normalize_embeddings=True)
        
        for frac in BUDGET_FRACTIONS:
            B = max(int(N * frac), 5)  # minimum 5 items
            
            print(f"\n    Budget: {frac:.0%} ({B}/{N} turns)")
            
            policies = {
                'BucketOccupancy': BucketOccupancyRetention(dim=dim, L=8, K=10, 
                                                             capacity=B, aggregator='median'),
                'FIFO': FIFORetention(dim=dim, capacity=B),
                'Reservoir': ReservoirSamplingRetention(dim=dim, capacity=B),
            }
            
            for policy_name, policy in policies.items():
                # Stream
                stream_conversation(policy, turns)
                kept_set, kept_ids = policy.kept_set()
                
                # Evaluate each QA
                scores = []
                for qi, qa in enumerate(qa_pairs):
                    q_emb = q_embeddings[qi]
                    context = retrieve_context(q_emb, kept_set, kept_ids, turns, k=RETRIEVE_K)
                    
                    predicted = ask_llm(qa['question'], context,
                                       conv['speaker_a'], conv['speaker_b'])
                    
                    score = score_answer(predicted, qa['answer'])
                    scores.append(score)
                
                accuracy = np.mean(scores) if scores else 0.0
                print(f"      {policy_name:20s}: accuracy={accuracy:.2%} ({sum(scores):.0f}/{len(scores)})")
                
                all_results.append({
                    'conversation_id': conv_id,
                    'policy': policy_name,
                    'budget_fraction': frac,
                    'budget_B': B,
                    'n_turns': N,
                    'n_questions': len(qa_pairs),
                    'accuracy': accuracy,
                    'correct': sum(scores),
                    'total': len(scores),
                })
    
    # Save results
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(RESULTS_DIR, 'phase5_locomo_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  Results saved to {csv_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Phase 5 Summary")
    print("=" * 60)
    summary = df.groupby(['policy', 'budget_fraction']).agg(
        mean_accuracy=('accuracy', 'mean'),
        total_correct=('correct', 'sum'),
        total_questions=('total', 'sum'),
    ).reset_index()
    summary['overall_accuracy'] = summary['total_correct'] / summary['total_questions']
    print(summary.to_string(index=False))
    
    return df


if __name__ == "__main__":
    run_phase5()
