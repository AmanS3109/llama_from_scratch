# 🦙 Tiny Llama (WASM / C Inference)

This project demonstrates a complete, end-to-end implementation of a **Llama-architecture Large Language Model (LLM)** built entirely from scratch. 

It covers the full AI engineering stack: 
1. **PyTorch Training:** Building and training a 4.5M parameter Llama model on synthetic data.
2. **Custom Binary Export:** Extracting raw tensors into a memory-mapped binary format.
3. **C Inference Engine:** Implementing matrix multiplications and attention mechanisms in pure, dependency-free C.
4. **WebAssembly Deployment:** Compiling the C engine to WASM to run the LLM natively in any web browser without a backend server or GPU.

---

## 🏗️ Architecture

The model implements the core components of modern state-of-the-art LLMs (like Meta's Llama 3):

* **RMSNorm** (Root Mean Square Normalization) instead of LayerNorm for computational efficiency.
* **RoPE** (Rotary Positional Embeddings) for relative sequence positioning.
* **GQA** (Grouped-Query Attention) to drastically reduce KV cache memory footprint.
* **SwiGLU** Feed-Forward Networks for advanced activation.
* **KV Caching** for fast autoregressive generation.

### Model Specs
* **Parameters:** ~4.46 Million
* **Dimensions:** `dim=256`, `hidden_dim=704`
* **Layers:** 6
* **Attention Heads:** 8 Query Heads, 4 KV Heads (GQA)
* **Dataset:** 5,000 synthetic Chess Rules Q&A pairs.

---

## 🚀 How It Works

### Phase 1: Training (`model.py` & `train.py`)
The model is written in pure PyTorch. We trained it on a custom, synthetically generated dataset of Chess Questions and Answers (`generate_chess_data.py`). It uses Mixed Precision (AMP) and the AdamW optimizer to learn the structure of chess rules at the character level.

### Phase 2: Export (`export.py`)
To escape Python's overhead, the trained model weights are flattened and exported into a custom, memory-aligned binary file (`tiny_llama.bin`). The tokenizer mappings are exported to `tokenizer.bin`.

### Phase 3: Pure C Inference (`run.c`)
The inference engine is written in standard C99 with **zero external dependencies** (no cuBLAS, no hardware-specific libraries). It dynamically allocates memory, maps the binary weights, and manually computes the enormous matrix multiplications required for the forward pass. 
* *Performance:* ~488 tokens/sec on an ordinary laptop CPU.

### Phase 4: Edge Deployment (`run_wasm.c` & `web/`)
The C code is compiled into WebAssembly (`llama.wasm`) using Emscripten. The browser fetches the 17MB binary weights and passes them into WASM memory. JavaScript simply calls the exported C functions `_init_model` and `_generate_next`.
* **Result:** A fully functional LLM running natively on the user's device (phone or computer) with zero latency, zero server costs, and total privacy.

---

## 🛠️ Build & Run

### 1. Python Training (Optional)
```bash
# Generate the chess dataset
python generate_chess_data.py

# Train the model (requires PyTorch)
python train.py

# Export weights to .bin
python export.py
```

### 2. Local C Inference
```bash
# Compile
gcc -O3 -o run.exe run.c -lm

# Run with prompt and temperature
./run.exe out/tiny_llama.bin "Question: How does the knight move?\nAnswer:" 0.1 150
```

### 3. WebAssembly (Browser)
```bash
# Compile C to WASM (requires Emscripten SDK)
emcc -O3 run_wasm.c -o web/llama.js -lm \
    -s EXPORTED_FUNCTIONS="['_init_model','_generate_next','_get_logits_ptr','_get_vocab_size','_encode_char','_decode_token','_reset_state','_cleanup','_malloc','_free']" \
    -s EXPORTED_RUNTIME_METHODS="['ccall','cwrap','HEAPU8','HEAPF32']" \
    -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=67108864 -s MODULARIZE=1 \
    -s EXPORT_NAME="Module" -s ENVIRONMENT=web

# Copy model files to the web directory
cp out/tiny_llama.bin web/
cp out/tokenizer.bin web/

# Start a local web server
python -m http.server 8080 --directory web
```
Then open `http://localhost:8080` in your browser.

---


