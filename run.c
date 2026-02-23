/*
 * run.c — Tiny Llama Inference Engine in Pure C
 *
 * This implements the full Llama forward pass (RMSNorm, RoPE, GQA Attention,
 * SwiGLU FFN) in portable C. It reads weights from a flat binary file exported
 * by export.py

 *
 * Usage: run <model.bin> [prompt] [temperature] [max_tokens]
 *
 * Build:
 *   Windows (MSVC):  cl /O2 /fp:fast run.c /Fe:run.exe
 *   GCC/MinGW:       gcc -O3 -o run run.c -lm
 *   Emscripten:      emcc -O3 -o run.js run.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// ============================================================================
// Data Structures
// ============================================================================

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int max_seq_len;
    int head_dim;       // derived: dim / n_heads
    int kv_dim;         // derived: head_dim * n_kv_heads
    int n_kv_groups;    // derived: n_heads / n_kv_heads
} Config;

typedef struct {
    // Token embedding
    float *token_embedding;  // (vocab_size, dim)
    // Per-layer weights
    float **attention_norm;  // (n_layers,) each (dim,)
    float **wq;              // (n_layers,) each (n_heads * head_dim, dim)
    float **wk;              // (n_layers,) each (n_kv_heads * head_dim, dim)
    float **wv;              // (n_layers,) each (n_kv_heads * head_dim, dim)
    float **wo;              // (n_layers,) each (dim, n_heads * head_dim)
    float **ffn_norm;        // (n_layers,) each (dim,)
    float **w1;              // (n_layers,) each (hidden_dim, dim) -- gate
    float **w2;              // (n_layers,) each (dim, hidden_dim) -- down
    float **w3;              // (n_layers,) each (hidden_dim, dim) -- up
    // Final
    float *final_norm;       // (dim,)
    float *output;           // (vocab_size, dim)
} Weights;

typedef struct {
    // Current activations
    float *x;           // (dim,) current token's hidden state
    float *xb;          // (dim,) after RMSNorm
    float *xb2;         // (dim,) after second RMSNorm
    float *hb;          // (hidden_dim,) FFN buffer
    float *hb2;         // (hidden_dim,) FFN buffer 2
    float *q;           // (n_heads * head_dim,)
    float *k;           // (n_kv_heads * head_dim,)
    float *v;           // (n_kv_heads * head_dim,)
    float *att;         // (n_heads, max_seq_len) attention scores
    float *logits;      // (vocab_size,) output logits
    // KV cache
    float *key_cache;   // (n_layers, max_seq_len, n_kv_heads * head_dim)
    float *value_cache; // (n_layers, max_seq_len, n_kv_heads * head_dim)
} RunState;

// ============================================================================
// Tokenizer (Character-level, loaded from tokenizer.bin)
// ============================================================================

typedef struct {
    char *chars;     // Array of characters, indexed by token id
    int vocab_size;
} Tokenizer;

// Load binary tokenizer: vocab_size (int32) + vocab_size raw bytes
int load_tokenizer(Tokenizer *tok, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Error: Cannot open tokenizer: %s\n", path); return 0; }

    if (fread(&tok->vocab_size, sizeof(int), 1, f) != 1) { fclose(f); return 0; }
    tok->chars = (char *)malloc(tok->vocab_size * sizeof(char));
    if (fread(tok->chars, sizeof(char), tok->vocab_size, f) != (size_t)tok->vocab_size) {
        fclose(f); return 0;
    }

    fclose(f);
    printf("Tokenizer loaded: vocab_size=%d\n", tok->vocab_size);
    return 1;
}

int encode_char(Tokenizer *tok, char c) {
    for (int i = 0; i < tok->vocab_size; i++) {
        if (tok->chars[i] == c) return i;
    }
    return 0; // fallback
}

char decode_token(Tokenizer *tok, int id) {
    if (id >= 0 && id < tok->vocab_size) return tok->chars[id];
    return '?';
}

// ============================================================================
// Memory Allocation
// ============================================================================

void malloc_weights(Weights *w, Config *c) {
    w->token_embedding = (float *)malloc(c->vocab_size * c->dim * sizeof(float));
    w->attention_norm = (float **)malloc(c->n_layers * sizeof(float *));
    w->wq = (float **)malloc(c->n_layers * sizeof(float *));
    w->wk = (float **)malloc(c->n_layers * sizeof(float *));
    w->wv = (float **)malloc(c->n_layers * sizeof(float *));
    w->wo = (float **)malloc(c->n_layers * sizeof(float *));
    w->ffn_norm = (float **)malloc(c->n_layers * sizeof(float *));
    w->w1 = (float **)malloc(c->n_layers * sizeof(float *));
    w->w2 = (float **)malloc(c->n_layers * sizeof(float *));
    w->w3 = (float **)malloc(c->n_layers * sizeof(float *));
    for (int i = 0; i < c->n_layers; i++) {
        w->attention_norm[i] = (float *)malloc(c->dim * sizeof(float));
        w->wq[i] = (float *)malloc(c->n_heads * c->head_dim * c->dim * sizeof(float));
        w->wk[i] = (float *)malloc(c->n_kv_heads * c->head_dim * c->dim * sizeof(float));
        w->wv[i] = (float *)malloc(c->n_kv_heads * c->head_dim * c->dim * sizeof(float));
        w->wo[i] = (float *)malloc(c->dim * c->n_heads * c->head_dim * sizeof(float));
        w->ffn_norm[i] = (float *)malloc(c->dim * sizeof(float));
        w->w1[i] = (float *)malloc(c->hidden_dim * c->dim * sizeof(float));
        w->w2[i] = (float *)malloc(c->dim * c->hidden_dim * sizeof(float));
        w->w3[i] = (float *)malloc(c->hidden_dim * c->dim * sizeof(float));
    }
    w->final_norm = (float *)malloc(c->dim * sizeof(float));
    w->output = (float *)malloc(c->vocab_size * c->dim * sizeof(float));
}

void malloc_run_state(RunState *s, Config *c) {
    s->x = (float *)calloc(c->dim, sizeof(float));
    s->xb = (float *)calloc(c->dim, sizeof(float));
    s->xb2 = (float *)calloc(c->dim, sizeof(float));
    s->hb = (float *)calloc(c->hidden_dim, sizeof(float));
    s->hb2 = (float *)calloc(c->hidden_dim, sizeof(float));
    s->q = (float *)calloc(c->n_heads * c->head_dim, sizeof(float));
    s->k = (float *)calloc(c->kv_dim, sizeof(float));
    s->v = (float *)calloc(c->kv_dim, sizeof(float));
    s->att = (float *)calloc(c->n_heads * c->max_seq_len, sizeof(float));
    s->logits = (float *)calloc(c->vocab_size, sizeof(float));
    s->key_cache = (float *)calloc(c->n_layers * c->max_seq_len * c->kv_dim, sizeof(float));
    s->value_cache = (float *)calloc(c->n_layers * c->max_seq_len * c->kv_dim, sizeof(float));
}

// ============================================================================
// Weight Loading
// ============================================================================

int load_weights(Weights *w, Config *c, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Error: Cannot open model: %s\n", path); return 0; }

    // Read header
    int header[7];
    if (fread(header, sizeof(int), 7, f) != 7) { fclose(f); return 0; }
    c->dim = header[0];
    c->hidden_dim = header[1];
    c->n_layers = header[2];
    c->n_heads = header[3];
    c->n_kv_heads = header[4];
    c->vocab_size = header[5];
    c->max_seq_len = header[6];
    c->head_dim = c->dim / c->n_heads;
    c->kv_dim = c->head_dim * c->n_kv_heads;
    c->n_kv_groups = c->n_heads / c->n_kv_heads;

    printf("Model config loaded:\n");
    printf("  dim=%d, hidden_dim=%d, n_layers=%d\n", c->dim, c->hidden_dim, c->n_layers);
    printf("  n_heads=%d, n_kv_heads=%d, head_dim=%d\n", c->n_heads, c->n_kv_heads, c->head_dim);
    printf("  vocab_size=%d, max_seq_len=%d\n", c->vocab_size, c->max_seq_len);

    // Allocate memory
    malloc_weights(w, c);

    // Read weights in the exact order they were exported
    fread(w->token_embedding, sizeof(float), c->vocab_size * c->dim, f);
    for (int i = 0; i < c->n_layers; i++) {
        fread(w->attention_norm[i], sizeof(float), c->dim, f);
        fread(w->wq[i], sizeof(float), c->n_heads * c->head_dim * c->dim, f);
        fread(w->wk[i], sizeof(float), c->n_kv_heads * c->head_dim * c->dim, f);
        fread(w->wv[i], sizeof(float), c->n_kv_heads * c->head_dim * c->dim, f);
        fread(w->wo[i], sizeof(float), c->dim * c->n_heads * c->head_dim, f);
        fread(w->ffn_norm[i], sizeof(float), c->dim, f);
        fread(w->w1[i], sizeof(float), c->hidden_dim * c->dim, f);
        fread(w->w2[i], sizeof(float), c->dim * c->hidden_dim, f);
        fread(w->w3[i], sizeof(float), c->hidden_dim * c->dim, f);
    }
    fread(w->final_norm, sizeof(float), c->dim, f);
    fread(w->output, sizeof(float), c->vocab_size * c->dim, f);

    fclose(f);
    printf("Weights loaded successfully.\n");
    return 1;
}

// ============================================================================
// Math Operations
// ============================================================================

// Matrix-vector multiply: out = W @ x
// W is (rows, cols), x is (cols,), out is (rows,)
void matmul(float *out, float *W, float *x, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        float *row = W + i * cols;
        for (int j = 0; j < cols; j++) {
            sum += row[j] * x[j];
        }
        out[i] = sum;
    }
}

// RMSNorm: out = (x / sqrt(mean(x^2) + eps)) * weight
void rmsnorm(float *out, float *x, float *weight, int size) {
    float ss = 0.0f;
    for (int i = 0; i < size; i++) {
        ss += x[i] * x[i];
    }
    ss = ss / size + 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int i = 0; i < size; i++) {
        out[i] = x[i] * ss * weight[i];
    }
}

// SiLU activation: x * sigmoid(x)
float silu(float x) {
    return x / (1.0f + expf(-x));
}

// Softmax in-place
void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// Apply RoPE rotation to a single head's q or k vector at a given position
void apply_rope(float *vec, int head_dim, int pos) {
    for (int i = 0; i < head_dim; i += 2) {
        float freq = 1.0f / powf(10000.0f, (float)i / (float)head_dim);
        float angle = pos * freq;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);
        float v0 = vec[i];
        float v1 = vec[i + 1];
        vec[i]     = v0 * cos_val - v1 * sin_val;
        vec[i + 1] = v0 * sin_val + v1 * cos_val;
    }
}

// ============================================================================
// Transformer Forward Pass (single token at position `pos`)
// ============================================================================

void forward(Config *c, Weights *w, RunState *s, int token, int pos) {
    int dim = c->dim;
    int hidden_dim = c->hidden_dim;
    int head_dim = c->head_dim;
    int n_heads = c->n_heads;
    int n_kv_heads = c->n_kv_heads;
    int kv_dim = c->kv_dim;
    int n_kv_groups = c->n_kv_groups;

    // Copy token embedding into x
    memcpy(s->x, w->token_embedding + token * dim, dim * sizeof(float));

    // Process each transformer layer
    for (int l = 0; l < c->n_layers; l++) {

        // --- Attention ---

        // 1. RMSNorm before attention
        rmsnorm(s->xb, s->x, w->attention_norm[l], dim);

        // 2. QKV projections
        matmul(s->q, w->wq[l], s->xb, n_heads * head_dim, dim);
        matmul(s->k, w->wk[l], s->xb, n_kv_heads * head_dim, dim);
        matmul(s->v, w->wv[l], s->xb, n_kv_heads * head_dim, dim);

        // 3. Apply RoPE to Q and K
        for (int h = 0; h < n_heads; h++) {
            apply_rope(s->q + h * head_dim, head_dim, pos);
        }
        for (int h = 0; h < n_kv_heads; h++) {
            apply_rope(s->k + h * head_dim, head_dim, pos);
        }

        // 4. Store K,V in cache
        int cache_offset = l * c->max_seq_len * kv_dim + pos * kv_dim;
        memcpy(s->key_cache + cache_offset, s->k, kv_dim * sizeof(float));
        memcpy(s->value_cache + cache_offset, s->v, kv_dim * sizeof(float));

        // 5. Multi-head attention with GQA
        for (int h = 0; h < n_heads; h++) {
            float *q_head = s->q + h * head_dim;
            int kv_head = h / n_kv_groups; // map query head to kv head
            float *att = s->att + h * c->max_seq_len;

            // Compute attention scores for all cached positions
            for (int t = 0; t <= pos; t++) {
                int kv_offset = l * c->max_seq_len * kv_dim + t * kv_dim + kv_head * head_dim;
                float *k_t = s->key_cache + kv_offset;
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += q_head[d] * k_t[d];
                }
                att[t] = score / sqrtf((float)head_dim);
            }

            // Softmax over attention scores
            softmax(att, pos + 1);

            // Weighted sum of values -> write into xb at head position
            float *out_head = s->xb + h * head_dim;
            memset(out_head, 0, head_dim * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                int kv_offset = l * c->max_seq_len * kv_dim + t * kv_dim + kv_head * head_dim;
                float *v_t = s->value_cache + kv_offset;
                float a = att[t];
                for (int d = 0; d < head_dim; d++) {
                    out_head[d] += a * v_t[d];
                }
            }
        }

        // 6. Output projection
        matmul(s->xb2, w->wo[l], s->xb, dim, n_heads * head_dim);

        // 7. Residual connection
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb2[i];
        }

        // --- Feed-Forward (SwiGLU) ---

        // 1. RMSNorm before FFN
        rmsnorm(s->xb, s->x, w->ffn_norm[l], dim);

        // 2. Gate and Up projections
        matmul(s->hb, w->w1[l], s->xb, hidden_dim, dim);  // gate
        matmul(s->hb2, w->w3[l], s->xb, hidden_dim, dim);  // up

        // 3. SiLU(gate) * up
        for (int i = 0; i < hidden_dim; i++) {
            s->hb[i] = silu(s->hb[i]) * s->hb2[i];
        }

        // 4. Down projection
        matmul(s->xb, w->w2[l], s->hb, dim, hidden_dim);

        // 5. Residual connection
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb[i];
        }
    }

    // Final RMSNorm
    rmsnorm(s->x, s->x, w->final_norm, dim);

    // Output logits
    matmul(s->logits, w->output, s->x, c->vocab_size, dim);
}

// ============================================================================
// Sampling
// ============================================================================

int argmax(float *v, int n) {
    int max_i = 0;
    float max_v = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] > max_v) { max_v = v[i]; max_i = i; }
    }
    return max_i;
}

int sample(float *probs, int n) {
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    for (int i = 0; i < n; i++) {
        cumsum += probs[i];
        if (r < cumsum) return i;
    }
    return n - 1;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char *argv[]) {
    // Defaults
    char *model_path = "out/tiny_llama.bin";
    char *tokenizer_path = "out/tokenizer.bin";
    char *prompt = "ROMEO:\n";
    float temperature = 0.8f;
    int max_tokens = 256;

    // Parse args
    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) prompt = argv[2];
    if (argc >= 4) temperature = atof(argv[3]);
    if (argc >= 5) max_tokens = atoi(argv[4]);

    printf("Tiny Llama - C Inference Engine\n");
    printf("================================\n");

    // Load model
    Config config;
    Weights weights;
    if (!load_weights(&weights, &config, model_path)) return 1;

    // Load tokenizer
    Tokenizer tokenizer;
    if (!load_tokenizer(&tokenizer, tokenizer_path)) return 1;

    // Allocate run state
    RunState state;
    malloc_run_state(&state, &config);

    // Encode prompt
    int prompt_len = (int)strlen(prompt);
    int *prompt_tokens = (int *)malloc(prompt_len * sizeof(int));
    for (int i = 0; i < prompt_len; i++) {
        prompt_tokens[i] = encode_char(&tokenizer, prompt[i]);
    }

    printf("\nGenerating with prompt: \"%s\"\n", prompt);
    printf("Temperature: %.2f, Max tokens: %d\n", temperature, max_tokens);
    printf("---\n");

    // Generate
    int token = prompt_tokens[0];
    int pos = 0;
    clock_t start = clock();

    // Print the first prompt character (it's consumed as input but never printed otherwise)
    putchar(decode_token(&tokenizer, token));
    fflush(stdout);

    for (int i = 0; i < max_tokens; i++) {
        // Forward pass
        forward(&config, &weights, &state, token, pos);

        int next_token;
        if (i < prompt_len - 1) {
            // Still processing prompt, force the next prompt token
            next_token = prompt_tokens[i + 1];
        } else {
            // Sample from logits
            if (temperature < 1e-6f) {
                next_token = argmax(state.logits, config.vocab_size);
            } else {
                // Apply temperature
                for (int j = 0; j < config.vocab_size; j++) {
                    state.logits[j] /= temperature;
                }
                softmax(state.logits, config.vocab_size);
                next_token = sample(state.logits, config.vocab_size);
            }
        }

        // Print the token
        char c = decode_token(&tokenizer, next_token);
        putchar(c);
        fflush(stdout);

        token = next_token;
        pos++;

        // Stop on double newline (end of answer in Q&A format)
        if (i >= prompt_len - 1) {
            static char prev_char = 0;
            if (prev_char == '\n' && c == '\n') break;
            prev_char = c;
        }

        if (pos >= config.max_seq_len - 1) break;
    }

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("\n---\n");
    printf("Generated %d tokens in %.2f seconds (%.1f tok/s)\n", pos, elapsed, pos / elapsed);

    // Cleanup
    free(prompt_tokens);
    free(state.x); free(state.xb); free(state.xb2);
    free(state.hb); free(state.hb2);
    free(state.q); free(state.k); free(state.v);
    free(state.att); free(state.logits);
    free(state.key_cache); free(state.value_cache);

    return 0;
}
