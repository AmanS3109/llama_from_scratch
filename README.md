# 🔥 LLaMA-Inspired Transformer Components (From Scratch)

This repository contains from-scratch implementations of core components used in large language models like Meta's **LLaMA** — written in pure PyTorch, without relying on any high-level libraries like HuggingFace Transformers.

## ✨ Features

- 🧠 **Custom Tokenizer** using Byte Pair Encoding (BPE)
- 🧮 **Self-Attention Module** with:
  - Multi-head & Grouped-Query Attention (GQA)
  - Rotary Position Embeddings (RoPE)
  - Optional Query/Key L2 Normalization
- ⚡ **Feed-Forward Network (FFN)** with:
  - RMSNorm normalization
  - SwiGLU activation (`SiLU(x) * Linear(x)`)
- 🔎 Educational and minimal design — ideal for learning LLM internals
- ✅ All modules tested with standalone examples

---

## 📁 Project Structure

```
.
├── tokenizer.py          # Custom BPE Tokenizer
├── self_attention.py     # Self-Attention with GQA & RoPE
├── feed_forward.py       # FFN with RMSNorm & SwiGLU
└── README.md             # You're here!
```

---

## 🔧 Requirements

- Python 3.8+
- PyTorch 1.13+ (or any recent version)

Install dependencies:

```bash
pip install torch
```

---

## 🧪 Running the Examples

Each `.py` file is fully self-contained and includes a test snippet.

```bash
# Run tokenizer
python tokenizer.py

# Run attention
python self_attention.py

# Run feed-forward
python feed_forward.py
```

---

## 📚 Concepts Covered

| Module            | Concepts Implemented                                                                 |
|-------------------|----------------------------------------------------------------------------------------|
| `tokenizer.py`     | Byte Pair Encoding (BPE), Vocabulary merging, Token ID mapping                       |
| `self_attention.py`| Multi-head Attention, Grouped-Query Attention (GQA), RoPE, Causal Masking             |
| `feed_forward.py`  | RMSNorm, SwiGLU activation, Intermediate projections, Residual pipeline              |

---

## 🚀 What's Next?

Stay tuned for the next steps:
- ✅ Combine into a `TransformerBlock`
- 🔜 Stack blocks into a full `LLaMA-style LLM`
- 🔜 Add embeddings, causal language modeling, and training loop

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

Inspired by:
- Meta AI's LLaMA and LLaMA-2/3 papers
- nanoGPT, llama.c, and Transformer lens
- Deep dive videos and blogs from Karpathy, Yannic Kilcher, and others

---

## 💬 Author

**Aman Singh**  
_Transforming theory into working ML systems from scratch._

Let's connect: [LinkedIn](https://www.linkedin.com) • [GitHub](https://github.com)
