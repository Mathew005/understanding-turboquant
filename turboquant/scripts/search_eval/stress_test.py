import torch
import time
import random
from pathlib import Path
from sentence_transformers import SentenceTransformer
from turboquant.search import TurboQuantIndex

def run_stress_test():
    root_dir = Path(__file__).resolve().parent.parent
    data_file = root_dir / "sample" / "big.txt"
    models_dir = root_dir / "models"

    print("\n" + "="*60)
    print(" TURBOQUANT STRESS TEST: MULTI-SCALE BENCHMARK")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device, cache_folder=str(models_dir))

    # 1. Load real data
    print(f"\n[1/4] Loading real text from {data_file.name}...")
    with open(data_file, 'r', encoding='utf-8') as f:
        all_chunks = [p.strip() for p in f.read().split('\n\n') if len(p.strip()) > 50]

    base_count = 2000
    print(f"      Encoding {base_count} base vectors...")
    base_embeddings = model.encode(all_chunks[:base_count], convert_to_tensor=True)

    # 2. Generate 50 diverse queries from the actual text
    # Use short snippets from random paragraphs as queries
    print("[2/4] Generating 50 query embeddings...")
    random.seed(42)
    query_texts = [
        # Hand-crafted semantic queries
        "What does Sherlock Holmes smoke?",
        "Who is Holmes's closest friend?",
        "A mysterious letter or message",
        "Someone was murdered or killed",
        "A disguise or hidden identity",
        "A woman asking for help",
        "A robbery or theft",
        "Evidence found at a crime scene",
        "A horse or carriage ride",
        "Someone is lying or deceiving",
        "A wedding or marriage",
        "A foreign country or traveler",
        "Money or financial trouble",
        "A secret door or hidden room",
        "A doctor or medical examination",
        "Someone running or chasing",
        "A newspaper article or advertisement",
        "Footprints or tracks on the ground",
        "A locked room mystery",
        "Poison or poisoning someone",
        "A wealthy nobleman or aristocrat",
        "A servant or butler",
        "A photograph or portrait",
        "A train journey or railway",
        "A dark night or foggy street",
    ]
    # Add 25 more by sampling real sentences from the text
    sampled_indices = random.sample(range(100, base_count), 25)
    for idx in sampled_indices:
        # Take first 60 chars of a random paragraph as a query
        snippet = all_chunks[idx][:60].strip()
        query_texts.append(snippet)

    num_queries = len(query_texts)
    query_embs = model.encode(query_texts, convert_to_tensor=True)
    print(f"      {num_queries} queries ready.")

    # 3. Test at multiple scales
    scales = [100_000, 300_000, 500_000, 1_000_000]
    add_batch_size = 50_000
    k = 10

    print(f"\n[3/4] Running benchmarks at scales: {[f'{s:,}' for s in scales]}")
    print(f"      Device: {device.upper()}")
    print(f"      Queries: {num_queries}, Recall@1 and Recall@{k}")
    print(f"      NOTE: Recall is measured by 'original paragraph match'")
    print(f"            (index % {base_count}), not exact copy match.")

    results = []

    for target_count in scales:
        print(f"\n{'─'*60}")
        print(f"  Scale: {target_count:,} vectors")
        print(f"{'─'*60}")

        repeats = target_count // base_count
        embeddings_cpu = base_embeddings.cpu().repeat(repeats, 1)
        embeddings_cpu += torch.randn_like(embeddings_cpu) * 0.01
        actual_count = embeddings_cpu.shape[0]

        embeddings_device = embeddings_cpu.to(device)
        old_mem = embeddings_device.element_size() * embeddings_device.nelement()

        # Build TurboQuant index in chunks
        try:
            if device == "cuda":
                torch.cuda.empty_cache()

            index = TurboQuantIndex(dim=384, bits=4, device=device)
            t0 = time.perf_counter()
            for start in range(0, actual_count, add_batch_size):
                end = min(start + add_batch_size, actual_count)
                batch = embeddings_cpu[start:end].to(device)
                index.add(batch)
                if device == "cuda":
                    torch.cuda.empty_cache()
            t_index = (time.perf_counter() - t0)
            new_mem = index.memory_usage_mb * 1024 * 1024
        except torch.cuda.OutOfMemoryError:
            print(f"  ⚠ OOM during indexing. Skipping.")
            results.append((target_count, None, None, None, None, None, None, None, None, "OOM"))
            if device == "cuda":
                torch.cuda.empty_cache()
            del embeddings_device, embeddings_cpu
            continue

        # --- SEARCH + ACCURACY ---
        try:
            # Warm up
            torch.matmul(query_embs[:1], embeddings_device.T)
            index.search(query_embs[:1], k=k)

            # Old search timing
            t0 = time.perf_counter()
            for _ in range(5):
                old_scores = torch.matmul(query_embs, embeddings_device.T)
            if device == "cuda":
                torch.cuda.synchronize()
            t_old = (time.perf_counter() - t0) / 5 * 1000

            # Ground truth top-K
            _, old_topk = torch.topk(old_scores, k, dim=-1, largest=True)

            # TurboQuant search timing
            t0 = time.perf_counter()
            for _ in range(5):
                _, new_topk = index.search(query_embs, k=k)
            if device == "cuda":
                torch.cuda.synchronize()
            t_new = (time.perf_counter() - t0) / 5 * 1000

            # --- RECALL BY ORIGINAL PARAGRAPH ---
            # Map every index back to its base paragraph: idx % base_count
            old_topk_base = old_topk % base_count  # (num_queries, k)
            new_topk_base = new_topk % base_count

            # Recall@1: Does TQ's #1 match the same original paragraph?
            r1_hits = (old_topk_base[:, 0] == new_topk_base[:, 0]).sum().item()
            recall_1 = r1_hits / num_queries * 100

            # Recall@K: Of the K original paragraphs in ground truth,
            # how many appear in TQ's top-K (by original paragraph)?
            rk_total = 0
            for q in range(num_queries):
                old_set = set(old_topk_base[q].tolist())
                new_set = set(new_topk_base[q].tolist())
                rk_total += len(old_set & new_set) / len(old_set)
            recall_k = rk_total / num_queries * 100

            ratio = old_mem / new_mem
            results.append((target_count, old_mem, new_mem, ratio, t_old, t_new, t_index, recall_1, recall_k, "OK"))

            print(f"  Memory:    {old_mem/1024/1024:.1f} MB → {new_mem/1024/1024:.1f} MB ({ratio:.1f}x)")
            print(f"  Search:    Old={t_old:.2f}ms  |  TQ={t_new:.2f}ms")
            print(f"  Recall@1:  {recall_1:.1f}%  ({r1_hits}/{num_queries} queries)")
            print(f"  Recall@{k}: {recall_k:.1f}%")
            print(f"  Index:     {t_index:.2f}s")

        except torch.cuda.OutOfMemoryError:
            print(f"  ⚠ OOM during search. Skipping.")
            results.append((target_count, old_mem, new_mem, old_mem/new_mem, None, None, None, None, None, "OOM"))
            if device == "cuda":
                torch.cuda.empty_cache()

        del embeddings_device, embeddings_cpu, index
        if device == "cuda":
            torch.cuda.empty_cache()

    # 4. Final Summary
    print("\n\n" + "="*95)
    print(f" FINAL RESULTS ({num_queries} queries, recall by original paragraph)")
    print("="*95)
    print(f"{'Vectors':>10} | {'Old Mem':>10} | {'TQ Mem':>10} | {'Ratio':>6} | {'Old Search':>11} | {'TQ Search':>11} | {'R@1':>6} | {'R@10':>6} | {'Status'}")
    print("-"*95)
    for r in results:
        count, old_m, new_m, ratio, t_o, t_n, t_i, r1, rk, status = r
        if status == "OK":
            print(f"{count:>10,} | {old_m/1024/1024:>8.1f}MB | {new_m/1024/1024:>8.1f}MB | {ratio:>5.1f}x | {t_o:>9.2f}ms | {t_n:>9.2f}ms | {r1:>5.1f}% | {rk:>5.1f}% | {status}")
        else:
            print(f"{count:>10,} | {'—':>10} | {'—':>10} | {'—':>6} | {'—':>11} | {'—':>11} | {'—':>6} | {'—':>6} | {status}")
    print("="*95)

if __name__ == "__main__":
    run_stress_test()
