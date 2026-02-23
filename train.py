"""
train.py — Train a Tiny Llama model on the TinyShakespeare dataset.

This script:
  1. Downloads the TinyShakespeare dataset (~1MB of text)
  2. Builds a character-level tokenizer from it
  3. Trains a small Llama model from scratch
  4. Saves the model checkpoint + tokenizer for later export to C/WASM
"""

import os
import json
import time
import math
import urllib.request
import torch
import torch.nn as nn
from model import LlamaModel, LlamaConfig

# ============================================================================
# Configuration
# ============================================================================
# Training (GPU-optimized)
BATCH_SIZE = 64
SEQ_LEN = 256
MAX_ITERS = 3000
EVAL_INTERVAL = 250
EVAL_ITERS = 50
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"  # Mixed precision for GPU speed

# Model (scaled up for GPU training)
MODEL_CONFIG = {
    "dim": 256,
    "n_layers": 6,
    "n_heads": 8,
    "n_kv_heads": 4,
    "multiple_of": 32,
    "norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "max_seq_len": SEQ_LEN + 1,
}

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_FILE = os.path.join(DATA_DIR, "chess_qa.txt")
OUT_DIR = os.path.join(os.path.dirname(__file__), "out")

# ============================================================================
# Data: Character-Level Tokenizer + Dataset
# ============================================================================

def load_data():
    """Load the local chess QA dataset."""
    if not os.path.exists(DATA_FILE):
        print(f"Error: Dataset not found at {DATA_FILE}")
        print("Run 'python generate_chess_data.py' first.")
        exit(1)
    print(f"Loading data from {DATA_FILE}")

class CharTokenizer:
    """Simple character-level tokenizer."""
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}
        print(f"Tokenizer: vocab_size={self.vocab_size}, chars: {''.join(chars[:50])}...")

    def encode(self, text: str) -> list:
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids: list) -> str:
        return "".join([self.id_to_char[i] for i in ids])

    def save(self, path: str):
        """Save tokenizer mapping to a JSON file."""
        with open(path, "w") as f:
            json.dump({
                "char_to_id": self.char_to_id,
                "id_to_char": {str(k): v for k, v in self.id_to_char.items()},
                "vocab_size": self.vocab_size
            }, f, indent=2)
        print(f"Tokenizer saved to {path}")

def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: str):
    """Generate a random batch of (input, target) pairs."""
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
    return x.to(device), y.to(device)

# ============================================================================
# Training Loop
# ============================================================================

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    """Estimate loss on train and val splits."""
    out = {}
    model.eval()
    for split_name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(data, BATCH_SIZE, SEQ_LEN, DEVICE)
            logits = model(X)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), Y.view(-1)
            )
            losses[k] = loss.item()
        out[split_name] = losses.mean()
    model.train()
    return out

@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200, temperature: float = 0.8):
    """Generate text from a prompt."""
    model.eval()
    ids = tokenizer.encode(prompt)
    context = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    for _ in range(max_new_tokens):
        # Crop context to max_seq_len if needed
        ctx = context if context.size(1) <= SEQ_LEN else context[:, -SEQ_LEN:]
        logits = model(ctx)
        logits = logits[:, -1, :] / temperature
        probs = nn.functional.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        context = torch.cat([context, next_id], dim=1)

    model.train()
    return tokenizer.decode(context[0].tolist())

def main():
    print("=" * 60)
    print("  Tiny Llama Training Script")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Mixed Precision (AMP): {USE_AMP}")

    # --- Data ---
    load_data()
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Dataset size: {len(text):,} characters")

    tokenizer = CharTokenizer(text)

    # Encode the full text
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # Train/Val split (90/10)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

    # --- Model ---
    config = LlamaConfig(vocab_size=tokenizer.vocab_size, **MODEL_CONFIG)
    model = LlamaModel(config).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel instantiated: {total_params / 1e6:.2f}M parameters")
    print(f"  dim={config.dim}, layers={config.n_layers}, heads={config.n_heads}, "
          f"kv_heads={config.n_kv_heads}, hidden_dim={config.hidden_dim}")

    # Note: torch.compile() requires Triton (Linux only). On Windows,
    # we still get great GPU speedups from AMP mixed precision.

    # --- Optimizer & Scaler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    # --- Training ---
    print(f"\nStarting training for {MAX_ITERS} iterations...")
    print("-" * 60)
    best_val_loss = float("inf")
    os.makedirs(OUT_DIR, exist_ok=True)

    t0 = time.time()
    for iter_num in range(MAX_ITERS):
        # Evaluate periodically
        if iter_num % EVAL_INTERVAL == 0 or iter_num == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data)
            elapsed = time.time() - t0
            tokens_per_sec = (iter_num * BATCH_SIZE * SEQ_LEN) / elapsed if elapsed > 0 and iter_num > 0 else 0
            print(f"Step {iter_num:5d} | "
                  f"train loss: {losses['train']:.4f} | "
                  f"val loss: {losses['val']:.4f} | "
                  f"time: {elapsed:.1f}s | "
                  f"tok/s: {tokens_per_sec:,.0f}")

            # Save best checkpoint
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                # Save the underlying model (unwrap compiled if needed)
                raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "config": MODEL_CONFIG,
                    "vocab_size": tokenizer.vocab_size,
                    "iter_num": iter_num,
                    "val_loss": best_val_loss,
                }
                torch.save(checkpoint, os.path.join(OUT_DIR, "tiny_llama.pt"))

            # Generate a sample
            if iter_num > 0:
                sample = generate(model, tokenizer, prompt="Question: How does the knight move?\nAnswer:", max_new_tokens=100)
                print(f"  Sample: {repr(sample[:120])}")

        # Training step with mixed precision
        X, Y = get_batch(train_data, BATCH_SIZE, SEQ_LEN, DEVICE)
        with torch.amp.autocast(device_type=DEVICE, enabled=USE_AMP):
            logits = model(X)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), Y.view(-1)
            )
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    # --- Final ---
    print("=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Save tokenizer
    tokenizer.save(os.path.join(OUT_DIR, "tokenizer.json"))

    # Final generation sample
    print("\n--- Final Generation Sample ---")
    sample = generate(model, tokenizer, prompt="Question: Explain castling.\nAnswer:", max_new_tokens=300)
    print(sample)

if __name__ == "__main__":
    main()
