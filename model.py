import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


# ============================================================================
# Core Building Blocks (self-contained, no global state dependencies)
# These mirror the logic from self_attention.py and feed_forward.py but are
# written as clean, reusable components for the full model.
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in Llama instead of LayerNorm)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return (self.weight * x).type_as(x)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute the complex-valued rotation frequencies for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Positional Embedding to query and key tensors.
    
    Args:
        xq: Query tensor (batch, n_heads, seqlen, head_dim)
        xk: Key tensor (batch, n_kv_heads, seqlen, head_dim)
        freqs_cis: Pre-sliced complex rotation freqs (seqlen, head_dim/2)
    """
    # Reshape to pairs -> view as complex
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Reshape freqs for broadcasting: (1, 1, seqlen, head_dim/2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)
    
    # Apply rotation via complex multiplication
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat K/V heads to match the number of query heads (for GQA)."""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


# ============================================================================
# Model Architecture
# ============================================================================

class LlamaConfig:
    def __init__(self, **kwargs):
        self.dim = kwargs.get('dim', 256)
        self.n_layers = kwargs.get('n_layers', 6)
        self.n_heads = kwargs.get('n_heads', 8)
        self.n_kv_heads = kwargs.get('n_kv_heads', 4)
        self.vocab_size = kwargs.get('vocab_size', 1000)
        self.multiple_of = kwargs.get('multiple_of', 32)
        self.norm_eps = kwargs.get('norm_eps', 1e-5)
        self.rope_theta = kwargs.get('rope_theta', 10000.0)
        self.max_seq_len = kwargs.get('max_seq_len', 512)

        self.head_dim = self.dim // self.n_heads
        ffn_dim = int(8 * self.dim / 3)
        self.hidden_dim = self.multiple_of * ((ffn_dim + self.multiple_of - 1) // self.multiple_of)


class Attention(nn.Module):
    def __init__(self, args: LlamaConfig):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.head_dim

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # GQA: repeat k/v heads
        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)

        # Scaled dot-product attention
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = nn.functional.softmax(scores.float(), dim=-1).type_as(xq)

        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate_proj
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)   # down_proj
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)   # up_proj

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: LlamaConfig):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(dim=args.dim, hidden_dim=args.hidden_dim)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class LlamaModel(nn.Module):
    def __init__(self, params: LlamaConfig):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(i, params) for i in range(params.n_layers)]
        )
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # Precompute RoPE frequencies (not a parameter, just a buffer)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(params.head_dim, params.max_seq_len, params.rope_theta),
            persistent=False
        )

    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, freqs_cis, mask)

        h = self.norm(h)
        output = self.output(h)
        return output

if __name__ == "__main__":
    # Test instantiating the tiny architecture
    config = LlamaConfig(vocab_size=100) # using tiny vocab for testing
    model = LlamaModel(config)
    
    print(f"Tiny Llama Instantiated.")
    print(f"Hidden Dim: {config.dim}, Layers: {config.n_layers}, Heads: {config.n_heads}")
    
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f}M")
    
    # Test a forward pass
    test_tokens = torch.randint(0, 100, (2, 16)) # batch_size=2, seq_len=16
    output_logits = model(test_tokens)
    
    print(f"Input Shape: {test_tokens.shape}")
    print(f"Output Shape (Logits): {output_logits.shape}")
