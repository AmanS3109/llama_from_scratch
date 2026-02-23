/*
 * run_wasm.c — Tiny Llama WASM Inference Engine
 *
 * This is a WASM-specific version of run.c that exports functions
 * for JavaScript to call. The model weights and tokenizer are loaded
 * via fetch() in JS and passed into WASM memory.
 *
 * Build:
 *   D:\Files\emsdk\upstream\emscripten\emcc -O3 run_wasm.c -o web/llama.js \
 *     -s EXPORTED_FUNCTIONS="['_init_model','_generate_next','_get_logits_ptr','_get_vocab_size','_cleanup']" \
 *     -s EXPORTED_RUNTIME_METHODS="['ccall','cwrap','HEAPU8','HEAPF32']" \
 *     -s ALLOW_MEMORY_GROWTH=1 -s TOTAL_MEMORY=67108864 -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#define EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define EXPORT
#endif

// ============================================================================
// Data Structures (same as run.c)
// ============================================================================

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int max_seq_len;
    int head_dim;
    int kv_dim;
    int n_kv_groups;
} Config;

typedef struct {
    float *token_embedding;
    float **attention_norm;
    float **wq, **wk, **wv, **wo;
    float **ffn_norm;
    float **w1, **w2, **w3;
    float *final_norm;
    float *output;
} Weights;

typedef struct {
    float *x, *xb, *xb2;
    float *hb, *hb2;
    float *q, *k, *v;
    float *att;
    float *logits;
    float *key_cache, *value_cache;
} RunState;

typedef struct {
    char *chars;
    int vocab_size;
} Tokenizer;

// Global state (WASM is single-threaded, so globals are fine)
static Config g_config;
static Weights g_weights;
static RunState g_state;
static Tokenizer g_tokenizer;
static int g_pos = 0;
static int g_initialized = 0;

// ============================================================================
// Math Operations (identical to run.c)
// ============================================================================

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

void rmsnorm(float *out, float *x, float *weight, int size) {
    float ss = 0.0f;
    for (int i = 0; i < size; i++) ss += x[i] * x[i];
    ss = ss / size + 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int i = 0; i < size; i++) out[i] = x[i] * ss * weight[i];
}

float silu(float x) { return x / (1.0f + expf(-x)); }

void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

void apply_rope(float *vec, int head_dim, int pos) {
    for (int i = 0; i < head_dim; i += 2) {
        float freq = 1.0f / powf(10000.0f, (float)i / (float)head_dim);
        float angle = pos * freq;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);
        float v0 = vec[i], v1 = vec[i + 1];
        vec[i]     = v0 * cos_val - v1 * sin_val;
        vec[i + 1] = v0 * sin_val + v1 * cos_val;
    }
}

// ============================================================================
// Forward Pass (identical to run.c)
// ============================================================================

void forward(Config *c, Weights *w, RunState *s, int token, int pos) {
    int dim = c->dim, hidden_dim = c->hidden_dim, head_dim = c->head_dim;
    int n_heads = c->n_heads, n_kv_heads = c->n_kv_heads;
    int kv_dim = c->kv_dim, n_kv_groups = c->n_kv_groups;

    memcpy(s->x, w->token_embedding + token * dim, dim * sizeof(float));

    for (int l = 0; l < c->n_layers; l++) {
        rmsnorm(s->xb, s->x, w->attention_norm[l], dim);
        matmul(s->q, w->wq[l], s->xb, n_heads * head_dim, dim);
        matmul(s->k, w->wk[l], s->xb, n_kv_heads * head_dim, dim);
        matmul(s->v, w->wv[l], s->xb, n_kv_heads * head_dim, dim);

        for (int h = 0; h < n_heads; h++) apply_rope(s->q + h * head_dim, head_dim, pos);
        for (int h = 0; h < n_kv_heads; h++) apply_rope(s->k + h * head_dim, head_dim, pos);

        int cache_offset = l * c->max_seq_len * kv_dim + pos * kv_dim;
        memcpy(s->key_cache + cache_offset, s->k, kv_dim * sizeof(float));
        memcpy(s->value_cache + cache_offset, s->v, kv_dim * sizeof(float));

        for (int h = 0; h < n_heads; h++) {
            float *q_head = s->q + h * head_dim;
            int kv_head = h / n_kv_groups;
            float *att = s->att + h * c->max_seq_len;
            for (int t = 0; t <= pos; t++) {
                float *k_t = s->key_cache + l * c->max_seq_len * kv_dim + t * kv_dim + kv_head * head_dim;
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) score += q_head[d] * k_t[d];
                att[t] = score / sqrtf((float)head_dim);
            }
            softmax(att, pos + 1);
            float *out_head = s->xb + h * head_dim;
            memset(out_head, 0, head_dim * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float *v_t = s->value_cache + l * c->max_seq_len * kv_dim + t * kv_dim + kv_head * head_dim;
                float a = att[t];
                for (int d = 0; d < head_dim; d++) out_head[d] += a * v_t[d];
            }
        }

        matmul(s->xb2, w->wo[l], s->xb, dim, n_heads * head_dim);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];

        rmsnorm(s->xb, s->x, w->ffn_norm[l], dim);
        matmul(s->hb, w->w1[l], s->xb, hidden_dim, dim);
        matmul(s->hb2, w->w3[l], s->xb, hidden_dim, dim);
        for (int i = 0; i < hidden_dim; i++) s->hb[i] = silu(s->hb[i]) * s->hb2[i];
        matmul(s->xb, w->w2[l], s->hb, dim, hidden_dim);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];
    }

    rmsnorm(s->x, s->x, w->final_norm, dim);
    matmul(s->logits, w->output, s->x, c->vocab_size, dim);
}

// ============================================================================
// WASM-Exported Functions
// ============================================================================

EXPORT int init_model(unsigned char *model_data, int model_len,
                      unsigned char *tok_data, int tok_len) {
    // Parse model header
    int *header = (int *)model_data;
    g_config.dim = header[0];
    g_config.hidden_dim = header[1];
    g_config.n_layers = header[2];
    g_config.n_heads = header[3];
    g_config.n_kv_heads = header[4];
    g_config.vocab_size = header[5];
    g_config.max_seq_len = header[6];
    g_config.head_dim = g_config.dim / g_config.n_heads;
    g_config.kv_dim = g_config.head_dim * g_config.n_kv_heads;
    g_config.n_kv_groups = g_config.n_heads / g_config.n_kv_heads;

    Config *c = &g_config;
    Weights *w = &g_weights;
    RunState *s = &g_state;

    // Parse tokenizer
    g_tokenizer.vocab_size = *(int *)tok_data;
    g_tokenizer.chars = (char *)malloc(g_tokenizer.vocab_size);
    memcpy(g_tokenizer.chars, tok_data + 4, g_tokenizer.vocab_size);

    // Allocate weights
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

    // Copy weights from the buffer (skip 7-int header = 28 bytes)
    float *ptr = (float *)(model_data + 7 * sizeof(int));
    memcpy(w->token_embedding, ptr, c->vocab_size * c->dim * sizeof(float));
    ptr += c->vocab_size * c->dim;
    for (int i = 0; i < c->n_layers; i++) {
        int sz;
        sz = c->dim; memcpy(w->attention_norm[i], ptr, sz * sizeof(float)); ptr += sz;
        sz = c->n_heads * c->head_dim * c->dim; memcpy(w->wq[i], ptr, sz * sizeof(float)); ptr += sz;
        sz = c->n_kv_heads * c->head_dim * c->dim; memcpy(w->wk[i], ptr, sz * sizeof(float)); ptr += sz;
        sz = c->n_kv_heads * c->head_dim * c->dim; memcpy(w->wv[i], ptr, sz * sizeof(float)); ptr += sz;
        sz = c->dim * c->n_heads * c->head_dim; memcpy(w->wo[i], ptr, sz * sizeof(float)); ptr += sz;
        sz = c->dim; memcpy(w->ffn_norm[i], ptr, sz * sizeof(float)); ptr += sz;
        sz = c->hidden_dim * c->dim; memcpy(w->w1[i], ptr, sz * sizeof(float)); ptr += sz;
        sz = c->dim * c->hidden_dim; memcpy(w->w2[i], ptr, sz * sizeof(float)); ptr += sz;
        sz = c->hidden_dim * c->dim; memcpy(w->w3[i], ptr, sz * sizeof(float)); ptr += sz;
    }
    memcpy(w->final_norm, ptr, c->dim * sizeof(float)); ptr += c->dim;
    memcpy(w->output, ptr, c->vocab_size * c->dim * sizeof(float));

    // Allocate run state
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

    g_pos = 0;
    g_initialized = 1;
    return c->vocab_size;
}

// Run one forward pass and return the sampled token character
EXPORT int generate_next(int token) {
    if (!g_initialized) return -1;
    if (g_pos >= g_config.max_seq_len - 1) return -1;

    forward(&g_config, &g_weights, &g_state, token, g_pos);
    g_pos++;
    return 0;
}

// Get pointer to logits array (JS reads from WASM memory)
EXPORT float* get_logits_ptr(void) {
    return g_state.logits;
}

EXPORT int get_vocab_size(void) {
    return g_config.vocab_size;
}

// Encode a character to token id
EXPORT int encode_char(int ch) {
    for (int i = 0; i < g_tokenizer.vocab_size; i++) {
        if (g_tokenizer.chars[i] == (char)ch) return i;
    }
    return 0;
}

// Decode token id to character
EXPORT int decode_token(int id) {
    if (id >= 0 && id < g_tokenizer.vocab_size) return (unsigned char)g_tokenizer.chars[id];
    return '?';
}

// Reset the KV cache for a new generation
EXPORT void reset_state(void) {
    if (!g_initialized) return;
    g_pos = 0;
    memset(g_state.key_cache, 0, g_config.n_layers * g_config.max_seq_len * g_config.kv_dim * sizeof(float));
    memset(g_state.value_cache, 0, g_config.n_layers * g_config.max_seq_len * g_config.kv_dim * sizeof(float));
}

EXPORT void cleanup(void) {
    g_initialized = 0;
    // Free all allocated memory (simplified)
}
