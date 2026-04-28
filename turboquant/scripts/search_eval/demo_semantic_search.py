import torch
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer
from turboquant.search import TurboQuantIndex

def run_comparison_demo():
    # 1. Setup
    print("\n" + "="*60)
    print(" TURBOQUANT vs. FULL PRECISION: SEMANTIC SEARCH BATTLE")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device.upper()}")

    # Cache the model locally to avoid re-downloading
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    print(f"Loading model (using cache at {models_dir})...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device, cache_folder=str(models_dir))
    dim = 384
    bits = 4

    # 2. Data: Let's use a larger set to make the timing more interesting
    sentences = [
        "A man is eating a piece of bread.",
        "The girl is carrying a baby.",
        "A man is riding a horse.",
        "A woman is playing violin.",
        "Two men pushed carts through the woods.",
        "A man is riding a white horse on an enclosed ground.",
        "A group of people are dancing in the street.",
        "A chef is preparing a delicious meal.",
        "The astronaut is floating in space.",
        "A programmer is writing code on a laptop.",
    ]

    # 3. Encoding (Standard for both)
    print("\n[1/4] Encoding sentences into vectors...")
    embeddings = model.encode(sentences, convert_to_tensor=True)

    # --- THE OLD WAY (Full Precision) ---
    print("\n[2/4] Setup: 'The Old Way' (Standard FP32 Index)")
    t0 = time.perf_counter()
    # In the old way, "indexing" is just keeping the tensor in memory
    old_memory_bytes = embeddings.element_size() * embeddings.nelement()
    t_old_index = (time.perf_counter() - t0) * 1000

    # --- THE NEW WAY (TurboQuant) ---
    print("[3/4] Setup: 'The New Way' (TurboQuant 4-bit Index)")
    t0 = time.perf_counter()
    index = TurboQuantIndex(dim=dim, bits=bits, device=device)
    index.add(embeddings)
    t_new_index = (time.perf_counter() - t0) * 1000
    new_memory_bytes = (index.memory_usage_mb * 1024 * 1024)

    # --- THE SEARCH BATTLE ---
    query_text = "Someone is playing a musical instrument"
    print(f"\n[4/4] BATTLE: Querying for '{query_text}'")
    query_emb = model.encode([query_text], convert_to_tensor=True)

    # Old Search
    t0 = time.perf_counter()
    scores = torch.matmul(query_emb, embeddings.T)
    old_best_idx = torch.argmax(scores).item()
    t_old_search = (time.perf_counter() - t0) * 1000

    # New Search
    t0 = time.perf_counter()
    distances, indices = index.search(query_emb, k=1)
    new_best_idx = indices[0][0].item()
    t_new_search = (time.perf_counter() - t0) * 1000

    # --- RESULTS ---
    print("\n" + "-"*60)
    print(f"{'METRIC':<20} | {'OLD WAY (FP32)':<18} | {'NEW WAY (TQ)':<15}")
    print("-"*60)
    print(f"{'Memory Usage':<20} | {old_memory_bytes:>10,.0f} bytes  | {new_memory_bytes:>8,.0f} bytes")
    print(f"{'Storage Savings':<20} | {'1.00x (Baseline)':<18} | {old_memory_bytes/new_memory_bytes:>8.2f}x SAVINGS")
    print("-"*60)
    print(f"{'Indexing Time':<20} | {t_old_index:>13.2f} ms | {t_new_index:>10.2f} ms")
    print(f"{'Search Time':<20} | {t_old_search:>13.4f} ms | {t_new_search:>10.4f} ms")
    print("-"*60)
    print(f"{'Top Result':<20} | {sentences[old_best_idx][:25]}... | {sentences[new_best_idx][:25]}...")
    print("-"*60)

    if old_best_idx == new_best_idx:
        print("\nSUCCESS: Both systems found the same correct answer despite the 8x compression!")
    else:
        print("\nNOTE: Results differ slightly due to quantization loss.")

if __name__ == "__main__":
    run_comparison_demo()
