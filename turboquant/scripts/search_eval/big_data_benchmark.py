import torch
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer
from turboquant.search import TurboQuantIndex

def run_big_data_benchmark():
    # 1. Setup paths and model
    root_dir = Path(__file__).resolve().parent.parent
    data_file = root_dir / "sample" / "big.txt"
    models_dir = root_dir / "models"
    
    print("\n" + "="*60)
    print(" SHERLOCK HOLMES BIG DATA BENCHMARK")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device, cache_folder=str(models_dir))
    
    # 2. Load and Chunk the data
    print(f"\n[1/5] Reading {data_file.name}...")
    with open(data_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split by double newline to get paragraphs, filter out very short ones
    all_chunks = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
    
    # Limit to 2000 chunks for a fast but meaningful demo
    num_chunks = 2000 
    chunks = all_chunks[:num_chunks]
    print(f"      Extracted {len(chunks)} paragraphs for the index.")

    # 3. Encoding (The expensive part)
    print(f"\n[2/5] Encoding {len(chunks)} paragraphs into vectors...")
    t0 = time.perf_counter()
    embeddings = model.encode(chunks, convert_to_tensor=True, show_progress_bar=True)
    t_encode = (time.perf_counter() - t0)
    print(f"      Encoding complete in {t_encode:.2f}s")

    # 4. Compare Indexing & Storage
    print("\n[3/5] Building Indices...")
    
    # Old Way
    old_mem = embeddings.element_size() * embeddings.nelement()
    
    # TurboQuant Way
    t0 = time.perf_counter()
    index = TurboQuantIndex(dim=384, bits=4, device=device)
    index.add(embeddings)
    t_new_index = (time.perf_counter() - t0) * 1000
    new_mem = index.memory_usage_mb * 1024 * 1024

    # 5. The Retrieval Battle (Multiple Queries)
    queries = [
        "Who is Sherlock Holmes's companion?",
        "What is Holmes's address in London?",
        "What instrument does Sherlock Holmes play?",
        "A story about a secret organization or club",
        "How does Sherlock Holmes solve crimes?"
    ]
    
    print(f"\n[4/5] Running {len(queries)} Queries...")
    
    total_old_time = 0
    total_new_time = 0
    matches = 0
    
    for q_text in queries:
        q_emb = model.encode([q_text], convert_to_tensor=True)
        
        # Old Search (Full Precision)
        t0 = time.perf_counter()
        scores = torch.matmul(q_emb, embeddings.T)
        old_idx = torch.argmax(scores).item()
        total_old_time += (time.perf_counter() - t0) * 1000
        
        # New Search (TurboQuant)
        t0 = time.perf_counter()
        dist, idxs = index.search(q_emb, k=1)
        new_idx = idxs[0][0].item()
        total_new_time += (time.perf_counter() - t0) * 1000
        
        if old_idx == new_idx:
            matches += 1

    # 6. Final Results
    print("\n" + "="*60)
    print(f" FINAL BENCHMARK RESULTS ({num_chunks} Paragraphs)")
    print("="*60)
    print(f"{'METRIC':<25} | {'OLD WAY (FP32)':<15} | {'NEW WAY (TQ)':<15}")
    print("-"*60)
    print(f"{'Total Memory':<25} | {old_mem/1024:>12.1f} KB | {new_mem/1024:>12.1f} KB")
    print(f"{'Compression Ratio':<25} | {'1.00x':>15} | {old_mem/new_mem:>14.2f}x")
    print(f"{'Avg Search Time':<25} | {total_old_time/len(queries):>12.4f} ms | {total_new_time/len(queries):>12.4f} ms")
    print(f"{'Retrieval Match Rate':<25} | {'100%':>15} | {matches/len(queries)*100:>14.1f}%")
    print("="*60)
    
    # Show one example result
    print(f"\nExample Query: '{queries[2]}'")
    print(f"Found: '{chunks[new_idx][:100]}...'")

if __name__ == "__main__":
    run_big_data_benchmark()
