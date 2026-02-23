"""
export.py — Export trained Tiny Llama weights to a flat binary file for C/WASM inference.

Binary format layout:
  1. Header (7 int32 values): dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len
  2. Weights in float32, written in this exact order:
     - tok_embeddings.weight          (vocab_size, dim)
     - For each layer i (0 to n_layers-1):
       - attention_norm.weight        (dim,)
       - wq.weight                    (n_heads * head_dim, dim)
       - wk.weight                    (n_kv_heads * head_dim, dim)
       - wv.weight                    (n_kv_heads * head_dim, dim)
       - wo.weight                    (dim, n_heads * head_dim)
       - ffn_norm.weight              (dim,)
       - w1.weight (gate_proj)        (hidden_dim, dim)
       - w2.weight (down_proj)        (dim, hidden_dim)
       - w3.weight (up_proj)          (hidden_dim, dim)
     - norm.weight (final RMSNorm)    (dim,)
     - output.weight (lm_head)        (vocab_size, dim)
"""

import os
import struct
import torch
import numpy as np

def export_model(checkpoint_path: str, output_path: str):
    """Export a trained PyTorch checkpoint to a flat binary file."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    config = checkpoint["config"]
    state_dict = checkpoint["model"]
    vocab_size = checkpoint["vocab_size"]

    dim = config["dim"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    n_kv_heads = config["n_kv_heads"]
    hidden_dim = config["multiple_of"] * ((int(8 * dim / 3) + config["multiple_of"] - 1) // config["multiple_of"])
    max_seq_len = config["max_seq_len"]

    print(f"Model config:")
    print(f"  dim={dim}, hidden_dim={hidden_dim}, n_layers={n_layers}")
    print(f"  n_heads={n_heads}, n_kv_heads={n_kv_heads}")
    print(f"  vocab_size={vocab_size}, max_seq_len={max_seq_len}")

    # Count total parameters for verification
    total_params = sum(v.numel() for v in state_dict.values())
    print(f"  Total parameters: {total_params:,}")

    with open(output_path, "wb") as f:
        # --- Header: 7 int32 values ---
        header = struct.pack("iiiiiii",
            dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len
        )
        f.write(header)
        print(f"Wrote header: {len(header)} bytes")

        bytes_written = len(header)

        def write_tensor(name, tensor):
            nonlocal bytes_written
            data = tensor.detach().cpu().float().numpy()
            f.write(data.tobytes())
            size = data.nbytes
            bytes_written += size
            print(f"  {name:45s} shape={str(list(tensor.shape)):20s} {size:>10,} bytes")

        # --- Token Embeddings ---
        write_tensor("tok_embeddings.weight", state_dict["tok_embeddings.weight"])

        # --- Transformer Layers ---
        for i in range(n_layers):
            print(f"\n  Layer {i}:")
            write_tensor(f"  layers.{i}.attention_norm.weight", state_dict[f"layers.{i}.attention_norm.weight"])
            write_tensor(f"  layers.{i}.attention.wq.weight",   state_dict[f"layers.{i}.attention.wq.weight"])
            write_tensor(f"  layers.{i}.attention.wk.weight",   state_dict[f"layers.{i}.attention.wk.weight"])
            write_tensor(f"  layers.{i}.attention.wv.weight",   state_dict[f"layers.{i}.attention.wv.weight"])
            write_tensor(f"  layers.{i}.attention.wo.weight",   state_dict[f"layers.{i}.attention.wo.weight"])
            write_tensor(f"  layers.{i}.ffn_norm.weight",       state_dict[f"layers.{i}.ffn_norm.weight"])
            write_tensor(f"  layers.{i}.feed_forward.w1.weight", state_dict[f"layers.{i}.feed_forward.w1.weight"])
            write_tensor(f"  layers.{i}.feed_forward.w2.weight", state_dict[f"layers.{i}.feed_forward.w2.weight"])
            write_tensor(f"  layers.{i}.feed_forward.w3.weight", state_dict[f"layers.{i}.feed_forward.w3.weight"])

        # --- Final RMSNorm ---
        print(f"\n  Final layers:")
        write_tensor("norm.weight", state_dict["norm.weight"])

        # --- Output (LM Head) ---
        write_tensor("output.weight", state_dict["output.weight"])

    file_size = os.path.getsize(output_path)
    print(f"\n{'='*60}")
    print(f"Export complete!")
    print(f"  Output: {output_path}")
    print(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    print(f"  Bytes written: {bytes_written:,}")
    print(f"  Expected: {7*4 + total_params*4:,} bytes (header + {total_params:,} float32 params)")

    if bytes_written == 7 * 4 + total_params * 4:
        print("  ✓ Size verification PASSED")
    else:
        print("  ✗ Size verification FAILED!")

def export_tokenizer(tokenizer_json_path: str, output_path: str):
    """Export the JSON tokenizer to a simple binary format for C.
    
    Binary format:
      - vocab_size (int32)
      - chars: vocab_size bytes, where chars[i] = character for token id i
    """
    import json
    with open(tokenizer_json_path, "r") as f:
        tok_data = json.load(f)
    
    vocab_size = tok_data["vocab_size"]
    char_to_id = tok_data["char_to_id"]
    
    # Build id->char mapping
    chars = ['\x00'] * vocab_size
    for ch, idx in char_to_id.items():
        chars[idx] = ch
    
    with open(output_path, "wb") as f:
        f.write(struct.pack("i", vocab_size))
        for ch in chars:
            f.write(ch.encode('utf-8')[:1])  # Write single byte per char
    
    print(f"Tokenizer binary exported to {output_path}")
    print(f"  vocab_size={vocab_size}, file_size={vocab_size + 4} bytes")


if __name__ == "__main__":
    checkpoint_path = os.path.join(os.path.dirname(__file__), "out", "tiny_llama.pt")
    output_path = os.path.join(os.path.dirname(__file__), "out", "tiny_llama.bin")
    tokenizer_json_path = os.path.join(os.path.dirname(__file__), "out", "tokenizer.json")
    tokenizer_bin_path = os.path.join(os.path.dirname(__file__), "out", "tokenizer.bin")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please run train.py first to train the model.")
    else:
        export_model(checkpoint_path, output_path)
        if os.path.exists(tokenizer_json_path):
            export_tokenizer(tokenizer_json_path, tokenizer_bin_path)
        else:
            print(f"Warning: tokenizer.json not found at {tokenizer_json_path}")
