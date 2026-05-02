import sys
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch not installed")

try:
    import faiss
    print(f"FAISS: {faiss.__version__ if hasattr(faiss, '__version__') else 'installed'}")
    ngpu = faiss.get_num_gpus()
    print(f"FAISS GPU count: {ngpu}")
except ImportError:
    print("FAISS not installed")
except Exception as e:
    print(f"FAISS GPU check error: {e}")
