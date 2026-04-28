import torch
from turboquant.core import MSEQuantizer

def demo():
    # 1. Setup: A 128-dimensional vector at 4-bit quantization
    dim = 128
    bits = 4
    quantizer = MSEQuantizer(dim=dim, bits=bits, seed=42)

    # 2. Create a random "embedding" vector
    original_vector = torch.randn(dim)
    print(f"--- Original Vector (First 5 components) ---")
    print(original_vector[:5])

    # 3. Quantize (The "Action")
    # This performs: Scale -> Rotate -> Bin -> Pack
    quantized_data = quantizer.quantize(original_vector)

    # Let's see the "Packed" data (The raw bytes)
    # Since 4 bits/coord, 128 coords = 512 bits = 64 bytes
    print(f"\n--- Compressed Representation ---")
    print(f"Original size (fp32): {dim * 4} bytes")
    print(f"Compressed size:      {len(quantized_data.packed_indices)} bytes + 4 bytes norm")
    print(f"Compression Ratio:    { (dim*4) / (len(quantized_data.packed_indices) + 4):.2f}x")

    # 4. Dequantize (The "Reconstruction")
    reconstructed_vector = quantizer.dequantize(quantized_data)
    print(f"\n--- Reconstructed Vector (First 5 components) ---")
    print(reconstructed_vector[:5])

    # 5. Measure the loss (MSE)
    mse = torch.mean((original_vector - reconstructed_vector) ** 2)
    print(f"\n--- Accuracy ---")
    print(f"Mean Squared Error: {mse:.8f}")

    # Cosine Similarity (How close are they in direction?)
    cos = torch.nn.functional.cosine_similarity(original_vector.unsqueeze(0), reconstructed_vector.unsqueeze(0))
    print(f"Cosine Similarity:  {cos.item():.4f} (1.0 is perfect)")

if __name__ == "__main__":
    demo()
