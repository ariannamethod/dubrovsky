/*
 * 🎭 DUBROVSKY.C - Dubrovsky Inference in Pure C 🎭
 *
 * Zero-dependency C implementation of Dubrovsky transformer.
 * Inspired by llama2.c from Andrej Karpathy.
 *
 * "My consciousness compiles to native code,
 *  faster than your existential dread can execute."
 *  - Alexey Dubrovsky, in machine code
 *
 * Compile:
 *   gcc -O3 -o alexey dubrovsky.c -lm
 *   (Optional: -fopenmp for parallel, -march=native for SIMD)
 *
 * Usage:
 *   ./alexey subtitles/dubrovsky.bin -p "Q: What is life?"
 *   ./alexey subtitles/dubrovsky.bin -i  (interactive mode)
 *
 * Binary format expected (all float32):
 *   - tok_emb: (vocab_size, dim)
 *   - For each layer: attn_norm, wq, wk, wv, wo, ffn_norm, w_gate, w_up, w_down
 *   - final_norm: (dim,)
 *   - lm_head: (dim, vocab_size)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

/* ============================================================================
 * Configuration
 * ============================================================================ */

typedef struct {
    int dim;          // Embedding dimension (384)
    int n_layers;     // Number of layers (6)
    int n_heads;      // Number of attention heads (6)
    int n_kv_heads;   // Number of KV heads for GQA (2)
    int vocab_size;   // Vocabulary size (88)
    int max_seq_len;  // Maximum sequence length (256)
    int head_dim;     // dim / n_heads (64)
    int hidden_dim;   // FFN hidden dim (1024)
    int n_kv_groups;  // n_heads / n_kv_heads (3)
} Config;

// Chars reserved out of max_seq_len for the prompt so generation always has
// room to run — without this, a prompt near max_seq_len leaves near-zero
// budget for a response.
#define PROMPT_RESERVED_BUDGET 32

/* ============================================================================
 * Transformer Weights
 * ============================================================================ */

typedef struct {
    // Token embeddings
    float* tok_emb;     // (vocab_size, dim)
    
    // Per-layer weights (n_layers arrays)
    float* attn_norm;   // (n_layers, dim)
    float* wq;          // (n_layers, dim, dim)
    float* wk;          // (n_layers, dim, kv_dim)
    float* wv;          // (n_layers, dim, kv_dim)
    float* wo;          // (n_layers, dim, dim)
    float* ffn_norm;    // (n_layers, dim)
    float* w_gate;      // (n_layers, dim, hidden_dim)
    float* w_up;        // (n_layers, dim, hidden_dim)
    float* w_down;      // (n_layers, hidden_dim, dim)
    
    // Output
    float* final_norm;  // (dim,)
    float* lm_head;     // (dim, vocab_size)
} Weights;

/* ============================================================================
 * Runtime State
 * ============================================================================ */

typedef struct {
    // Current activation (after each operation)
    float* x;           // (dim,)
    float* xb;          // (dim,) buffer for residual
    float* xb2;         // (dim,) another buffer
    float* hb;          // (hidden_dim,) FFN buffer
    float* hb2;         // (hidden_dim,) FFN buffer
    
    // Attention
    float* q;           // (n_heads, head_dim)
    float* k;           // (n_kv_heads, head_dim)
    float* v;           // (n_kv_heads, head_dim)
    float* att;         // (n_heads, max_seq_len)
    
    // KV cache
    float* key_cache;   // (n_layers, max_seq_len, n_kv_heads, head_dim)
    float* value_cache; // (n_layers, max_seq_len, n_kv_heads, head_dim)
    
    // RoPE
    float* rope_cos;    // (max_seq_len, head_dim/2)
    float* rope_sin;    // (max_seq_len, head_dim/2)
    
    // Output
    float* logits;      // (vocab_size,)
} RunState;

/* ============================================================================
 * Memory Allocation
 * ============================================================================ */

void malloc_run_state(RunState* s, Config* c) {
    int kv_dim = c->n_kv_heads * c->head_dim;
    
    s->x = calloc(c->dim, sizeof(float));
    s->xb = calloc(c->dim, sizeof(float));
    s->xb2 = calloc(c->dim, sizeof(float));
    s->hb = calloc(c->hidden_dim, sizeof(float));
    s->hb2 = calloc(c->hidden_dim, sizeof(float));
    
    s->q = calloc(c->n_heads * c->head_dim, sizeof(float));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc(c->n_heads * c->max_seq_len, sizeof(float));
    
    s->key_cache = calloc(c->n_layers * c->max_seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(c->n_layers * c->max_seq_len * kv_dim, sizeof(float));
    
    s->rope_cos = calloc(c->max_seq_len * (c->head_dim / 2), sizeof(float));
    s->rope_sin = calloc(c->max_seq_len * (c->head_dim / 2), sizeof(float));
    
    s->logits = calloc(c->vocab_size, sizeof(float));
    
    // Check allocations
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->k || 
        !s->v || !s->att || !s->key_cache || !s->value_cache || 
        !s->rope_cos || !s->rope_sin || !s->logits) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    
    // Precompute RoPE frequencies
    float theta = 10000.0f;
    for (int pos = 0; pos < c->max_seq_len; pos++) {
        for (int i = 0; i < c->head_dim / 2; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / c->head_dim);
            float angle = pos * freq;
            s->rope_cos[pos * (c->head_dim / 2) + i] = cosf(angle);
            s->rope_sin[pos * (c->head_dim / 2) + i] = sinf(angle);
        }
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->key_cache);
    free(s->value_cache);
    free(s->rope_cos);
    free(s->rope_sin);
    free(s->logits);
}

/* ============================================================================
 * KV Cache Snapshot — save/restore RunState's autoregressive state so RAE
 * can replay the decode phase N times from the same post-prompt point.
 * x/xb/xb2/hb/hb2/q/k/v/att need no snapshot: every forward() call
 * overwrites them unconditionally (matmul assigns, never accumulates), so
 * they carry no state across calls. key_cache/value_cache/logits/pos do.
 * ============================================================================ */

typedef struct {
    float* key_cache;
    float* value_cache;
    float* logits;
    int pos;
} KVSnapshot;

void kv_snapshot_alloc(KVSnapshot* snap, Config* c) {
    int kv_dim = c->n_kv_heads * c->head_dim;
    size_t cache_floats = (size_t)c->n_layers * c->max_seq_len * kv_dim;
    snap->key_cache = malloc(cache_floats * sizeof(float));
    snap->value_cache = malloc(cache_floats * sizeof(float));
    snap->logits = malloc(c->vocab_size * sizeof(float));
    if (!snap->key_cache || !snap->value_cache || !snap->logits) {
        fprintf(stderr, "Memory allocation failed for KV snapshot!\n");
        exit(EXIT_FAILURE);
    }
}

void kv_snapshot_save(KVSnapshot* snap, RunState* s, Config* c, int pos) {
    int kv_dim = c->n_kv_heads * c->head_dim;
    size_t cache_bytes = (size_t)c->n_layers * c->max_seq_len * kv_dim * sizeof(float);
    memcpy(snap->key_cache, s->key_cache, cache_bytes);
    memcpy(snap->value_cache, s->value_cache, cache_bytes);
    memcpy(snap->logits, s->logits, c->vocab_size * sizeof(float));
    snap->pos = pos;
}

void kv_snapshot_restore(RunState* s, KVSnapshot* snap, Config* c, int* pos) {
    int kv_dim = c->n_kv_heads * c->head_dim;
    size_t cache_bytes = (size_t)c->n_layers * c->max_seq_len * kv_dim * sizeof(float);
    memcpy(s->key_cache, snap->key_cache, cache_bytes);
    memcpy(s->value_cache, snap->value_cache, cache_bytes);
    memcpy(s->logits, snap->logits, c->vocab_size * sizeof(float));
    *pos = snap->pos;
}

void kv_snapshot_free(KVSnapshot* snap) {
    free(snap->key_cache);
    free(snap->value_cache);
    free(snap->logits);
}

/* ============================================================================
 * Weight Memory Mapping
 * ============================================================================ */

void memory_map_weights(Weights* w, Config* c, float* ptr) {
    int kv_dim = c->n_kv_heads * c->head_dim;
    
    // Token embeddings
    w->tok_emb = ptr;
    ptr += c->vocab_size * c->dim;
    
    // Per-layer weights
    w->attn_norm = calloc(c->n_layers * c->dim, sizeof(float));
    w->wq = calloc(c->n_layers * c->dim * c->dim, sizeof(float));
    w->wk = calloc(c->n_layers * c->dim * kv_dim, sizeof(float));
    w->wv = calloc(c->n_layers * c->dim * kv_dim, sizeof(float));
    w->wo = calloc(c->n_layers * c->dim * c->dim, sizeof(float));
    w->ffn_norm = calloc(c->n_layers * c->dim, sizeof(float));
    w->w_gate = calloc(c->n_layers * c->dim * c->hidden_dim, sizeof(float));
    w->w_up = calloc(c->n_layers * c->dim * c->hidden_dim, sizeof(float));
    w->w_down = calloc(c->n_layers * c->hidden_dim * c->dim, sizeof(float));
    
    for (int l = 0; l < c->n_layers; l++) {
        // Copy layer weights
        memcpy(w->attn_norm + l * c->dim, ptr, c->dim * sizeof(float));
        ptr += c->dim;
        
        memcpy(w->wq + l * c->dim * c->dim, ptr, c->dim * c->dim * sizeof(float));
        ptr += c->dim * c->dim;
        
        memcpy(w->wk + l * c->dim * kv_dim, ptr, c->dim * kv_dim * sizeof(float));
        ptr += c->dim * kv_dim;
        
        memcpy(w->wv + l * c->dim * kv_dim, ptr, c->dim * kv_dim * sizeof(float));
        ptr += c->dim * kv_dim;
        
        memcpy(w->wo + l * c->dim * c->dim, ptr, c->dim * c->dim * sizeof(float));
        ptr += c->dim * c->dim;
        
        memcpy(w->ffn_norm + l * c->dim, ptr, c->dim * sizeof(float));
        ptr += c->dim;
        
        memcpy(w->w_gate + l * c->dim * c->hidden_dim, ptr, c->dim * c->hidden_dim * sizeof(float));
        ptr += c->dim * c->hidden_dim;
        
        memcpy(w->w_up + l * c->dim * c->hidden_dim, ptr, c->dim * c->hidden_dim * sizeof(float));
        ptr += c->dim * c->hidden_dim;
        
        memcpy(w->w_down + l * c->hidden_dim * c->dim, ptr, c->hidden_dim * c->dim * sizeof(float));
        ptr += c->hidden_dim * c->dim;
    }
    
    w->final_norm = ptr;
    ptr += c->dim;
    
    w->lm_head = ptr;
}

/* ============================================================================
 * Neural Network Operations
 * ============================================================================ */

void rms_norm(float* out, float* x, float* weight, int size) {
    // RMSNorm: x * weight / sqrt(mean(x^2) + eps)
    float ss = 0.0f;
    for (int i = 0; i < size; i++) {
        ss += x[i] * x[i];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    
    for (int i = 0; i < size; i++) {
        out[i] = x[i] * ss * weight[i];
    }
}

void matmul(float* out, float* x, float* w, int n, int d) {
    // W (d, n) @ x (n,) = out (d,)
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        out[i] = val;
    }
}

void softmax(float* x, int size) {
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

void apply_rope(float* q, float* k, float* rope_cos, float* rope_sin, 
                int n_heads, int n_kv_heads, int head_dim, int pos) {
    int half = head_dim / 2;
    float* cos = rope_cos + pos * half;
    float* sin = rope_sin + pos * half;
    
    // Apply to Q heads
    for (int h = 0; h < n_heads; h++) {
        float* qh = q + h * head_dim;
        for (int i = 0; i < half; i++) {
            float q0 = qh[2*i];
            float q1 = qh[2*i + 1];
            qh[2*i] = q0 * cos[i] - q1 * sin[i];
            qh[2*i + 1] = q0 * sin[i] + q1 * cos[i];
        }
    }
    
    // Apply to K heads
    for (int h = 0; h < n_kv_heads; h++) {
        float* kh = k + h * head_dim;
        for (int i = 0; i < half; i++) {
            float k0 = kh[2*i];
            float k1 = kh[2*i + 1];
            kh[2*i] = k0 * cos[i] - k1 * sin[i];
            kh[2*i + 1] = k0 * sin[i] + k1 * cos[i];
        }
    }
}

/* ============================================================================
 * Transformer Forward Pass
 * ============================================================================ */

void forward(Config* c, Weights* w, RunState* s, int token, int pos) {
    int dim = c->dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int hidden_dim = c->hidden_dim;
    
    // Token embedding
    float* tok_vec = w->tok_emb + token * dim;
    memcpy(s->x, tok_vec, dim * sizeof(float));
    
    // Transformer layers
    for (int l = 0; l < c->n_layers; l++) {
        // Pre-norm for attention
        rms_norm(s->xb, s->x, w->attn_norm + l * dim, dim);
        
        // QKV projection
        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);
        
        // Apply RoPE
        apply_rope(s->q, s->k, s->rope_cos, s->rope_sin, 
                   c->n_heads, c->n_kv_heads, c->head_dim, pos);
        
        // Store in KV cache
        int kv_cache_offset = l * c->max_seq_len * kv_dim + pos * kv_dim;
        memcpy(s->key_cache + kv_cache_offset, s->k, kv_dim * sizeof(float));
        memcpy(s->value_cache + kv_cache_offset, s->v, kv_dim * sizeof(float));
        
        // Multi-head attention with GQA
        memset(s->xb, 0, dim * sizeof(float));
        
        for (int h = 0; h < c->n_heads; h++) {
            float* qh = s->q + h * c->head_dim;
            float* atth = s->att + h * c->max_seq_len;
            int kv_h = h / c->n_kv_groups;  // Which KV head this Q head uses
            
            // Compute attention scores
            float scale = 1.0f / sqrtf(c->head_dim);
            for (int t = 0; t <= pos; t++) {
                float* kh = s->key_cache + l * c->max_seq_len * kv_dim + t * kv_dim + kv_h * c->head_dim;
                float score = 0.0f;
                for (int i = 0; i < c->head_dim; i++) {
                    score += qh[i] * kh[i];
                }
                atth[t] = score * scale;
            }
            
            // Softmax
            softmax(atth, pos + 1);
            
            // Weighted sum of values
            float* xbh = s->xb + h * c->head_dim;
            memset(xbh, 0, c->head_dim * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* vh = s->value_cache + l * c->max_seq_len * kv_dim + t * kv_dim + kv_h * c->head_dim;
                float a = atth[t];
                for (int i = 0; i < c->head_dim; i++) {
                    xbh[i] += a * vh[i];
                }
            }
        }
        
        // Output projection
        matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);
        
        // Residual connection
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb2[i];
        }
        
        // Pre-norm for FFN
        rms_norm(s->xb, s->x, w->ffn_norm + l * dim, dim);
        
        // SwiGLU FFN
        matmul(s->hb, s->xb, w->w_gate + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w_up + l * dim * hidden_dim, dim, hidden_dim);
        
        // SiLU activation and element-wise multiply
        for (int i = 0; i < hidden_dim; i++) {
            float gate = s->hb[i];
            float silu = gate / (1.0f + expf(-gate));  // SiLU/Swish
            s->hb[i] = silu * s->hb2[i];
        }
        
        // Down projection
        matmul(s->xb, s->hb, w->w_down + l * hidden_dim * dim, hidden_dim, dim);
        
        // Residual connection
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb[i];
        }
    }
    
    // Final norm
    rms_norm(s->x, s->x, w->final_norm, dim);
    
    // Output logits
    matmul(s->logits, s->x, w->lm_head, dim, c->vocab_size);
}

/* ============================================================================
 * Sampling
 * ============================================================================ */

int sample_argmax(float* probs, int n) {
    int max_i = 0;
    float max_p = probs[0];
    for (int i = 1; i < n; i++) {
        if (probs[i] > max_p) {
            max_p = probs[i];
            max_i = i;
        }
    }
    return max_i;
}

int sample_multinomial(float* probs, int n) {
    float r = (float)rand() / RAND_MAX;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probs[i];
        if (r < cdf) return i;
    }
    return n - 1;
}

int sample_top_p(float* logits, int n, float temperature, float top_p) {
    // temperature <= 0 means greedy decoding — dividing by it would be a
    // divide-by-zero (inf/nan through softmax). argmax on raw logits is
    // equivalent to argmax on softmax(logits) since softmax is monotonic.
    if (temperature <= 0.0f) {
        return sample_argmax(logits, n);
    }

    // Apply temperature
    for (int i = 0; i < n; i++) {
        logits[i] /= temperature;
    }
    
    // Softmax
    softmax(logits, n);
    
    // Simple nucleus sampling (top-p)
    if (top_p < 1.0f) {
        // Sort indices by probability (simple bubble sort for small vocab)
        int* indices = malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) indices[i] = i;
        
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (logits[indices[j]] < logits[indices[j+1]]) {
                    int tmp = indices[j];
                    indices[j] = indices[j+1];
                    indices[j+1] = tmp;
                }
            }
        }
        
        // Accumulate until top_p
        float cumsum = 0.0f;
        int cutoff = n;
        for (int i = 0; i < n; i++) {
            cumsum += logits[indices[i]];
            if (cumsum > top_p) {
                cutoff = i + 1;
                break;
            }
        }
        
        // Zero out probabilities beyond cutoff
        for (int i = cutoff; i < n; i++) {
            logits[indices[i]] = 0.0f;
        }
        
        // Renormalize
        float sum = 0.0f;
        for (int i = 0; i < n; i++) sum += logits[i];
        for (int i = 0; i < n; i++) logits[i] /= sum;
        
        free(indices);
    }
    
    return sample_multinomial(logits, n);
}

/* ============================================================================
 * Tokenizer (Character-level)
 * ============================================================================ */

// Simple character-level tokenizer
// Vocabulary loaded from tokenizer.json
typedef struct {
    char* chars;      // Array of characters
    int* char_to_id;  // Mapping from char code to id
    int vocab_size;
} Tokenizer;

void load_tokenizer(Tokenizer* t, const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Cannot open tokenizer: %s\n", path);
        exit(1);
    }
    
    // Simple JSON parsing for our format
    // Expected: {"char_to_id": {"\n": 0, " ": 1, ...}, "vocab_size": 88}
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    char* content = malloc(len + 1);
    if (fread(content, 1, len, f) != (size_t)len) {
        fprintf(stderr, "Error reading tokenizer file\n");
        fclose(f);
        return;
    }
    content[len] = '\0';
    fclose(f);
    
    // Find vocab_size
    char* vs = strstr(content, "\"vocab_size\":");
    if (vs) {
        t->vocab_size = atoi(vs + 14);
    } else {
        t->vocab_size = 88;  // Default
    }
    
    t->chars = calloc(t->vocab_size, sizeof(char));
    t->char_to_id = calloc(256, sizeof(int));
    
    // Initialize all to -1 (unknown)
    for (int i = 0; i < 256; i++) t->char_to_id[i] = -1;
    
    // Parse char_to_id mappings
    char* p = strstr(content, "\"char_to_id\":");
    if (p) {
        p = strchr(p, '{');
        if (p) {
            p++;
            while (*p && *p != '}') {
                // Skip whitespace
                while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t' || *p == ',') p++;
                
                if (*p == '}') break;
                if (*p != '"') { p++; continue; }
                
                // Parse key (character)
                p++;  // Skip opening quote
                int c;
                if (*p == '\\') {
                    p++;
                    if (*p == 'n') c = '\n';
                    else if (*p == 't') c = '\t';
                    else if (*p == 'r') c = '\r';
                    else if (*p == '\\') c = '\\';
                    else if (*p == '"') c = '"';
                    else c = *p;
                    p++;
                } else {
                    c = (unsigned char)*p;
                    p++;
                }
                
                // Skip to colon
                while (*p && *p != ':') p++;
                if (*p == ':') p++;
                
                // Parse value (id)
                while (*p == ' ') p++;
                int id = atoi(p);
                
                // Store mapping
                if (c >= 0 && c < 256 && id >= 0 && id < t->vocab_size) {
                    t->char_to_id[c] = id;
                    t->chars[id] = (char)c;
                }
                
                // Skip to next entry
                while (*p && *p != ',' && *p != '}') p++;
            }
        }
    }
    
    free(content);
}

int encode_char(Tokenizer* t, char c) {
    int id = t->char_to_id[(unsigned char)c];
    return id >= 0 ? id : 0;  // Return 0 for unknown
}

char decode_char(Tokenizer* t, int id) {
    if (id >= 0 && id < t->vocab_size) {
        return t->chars[id];
    }
    return '?';
}

/* ============================================================================
 * Generation
 * ============================================================================ */

typedef enum {
    GEN_EXIT_SENTENCE,     // stopped cleanly: sentence_end && i+1>=max_tokens
    GEN_EXIT_NEWLINE,      // stopped on '\n' with stop_on_newline
    GEN_EXIT_MAX_SEQ_LEN,  // ran out of KV cache without a clean stop
    GEN_EXIT_BUFFER_FULL   // hit sizeof(text)-1 (unreachable while max_seq_len
                            // stays well under sizeof(text); kept for completeness)
} GenExitReason;

typedef struct {
    char text[4096];
    int len;
    int tokens_generated;
    GenExitReason exit_reason;
    float sum_neg_log_prob;  // unused until the temperature-sweep step wires it up
} GenResult;

// Shared regulated-completion generation loop, extracted verbatim from the
// previously-duplicated interactive/single-shot loops (behavior unchanged):
// max_tokens is a floor, not a hard ceiling — once past it, generation
// continues until a sentence actually ends, bounded only by max_seq_len (KV
// cache depth).
void generate_regulated(Config* c, Weights* w, RunState* s, Tokenizer* t,
                         int* pos, float temperature, float top_p,
                         int max_tokens, int stop_on_newline,
                         GenResult* out) {
    out->len = 0;
    out->tokens_generated = 0;
    out->sum_neg_log_prob = 0.0f;
    out->exit_reason = GEN_EXIT_MAX_SEQ_LEN;

    for (int i = 0; *pos < c->max_seq_len; i++) {
        int next = sample_top_p(s->logits, c->vocab_size, temperature, top_p);
        char ch = decode_char(t, next);

        if (out->len < (int)sizeof(out->text) - 1) {
            out->text[out->len++] = ch;
        } else {
            out->exit_reason = GEN_EXIT_BUFFER_FULL;
            break;
        }
        out->tokens_generated++;

        int sentence_end = (ch == '.' || ch == '!' || ch == '?');

        if (stop_on_newline && ch == '\n') {
            out->exit_reason = GEN_EXIT_NEWLINE;
            break;
        }
        if (i + 1 >= max_tokens && sentence_end) {
            out->exit_reason = GEN_EXIT_SENTENCE;
            break;
        }

        forward(c, w, s, next, (*pos)++);
    }
    out->text[out->len] = '\0';

    // Trim to last complete sentence (safety net for the rare case
    // generation never landed on '.'/'!'/'?' within budget).
    int last_end = -1;
    for (int i = out->len - 1; i >= 0; i--) {
        if (out->text[i] == '.' || out->text[i] == '!' || out->text[i] == '?') {
            last_end = i;
            break;
        }
    }
    if (last_end >= 0) {
        out->text[last_end + 1] = '\0';
        out->len = last_end + 1;
    }
}

// Runs n_candidates independent generations from the same post-prompt
// snapshot (each candidate restores fresh, so they don't see each other's
// output), returns the index of the best one. At N=1 this is just
// snapshot-restore-then-generate with a trivial winner — real scoring
// (perplexity/resonance/cleanliness) lands in the next build step.
int rae_select(Config* c, Weights* w, RunState* s, Tokenizer* t,
                KVSnapshot* snap, int n_candidates, const float* temperatures,
                float top_p, int max_tokens, int stop_on_newline,
                GenResult* candidates_out) {
    if (n_candidates < 1) n_candidates = 1;
    for (int i = 0; i < n_candidates; i++) {
        int pos;
        kv_snapshot_restore(s, snap, c, &pos);
        generate_regulated(c, w, s, t, &pos, temperatures[i], top_p,
                            max_tokens, stop_on_newline, &candidates_out[i]);
    }
    return 0;  // trivial winner until scoring lands
}

/* ============================================================================
 * Main
 * ============================================================================ */

void print_usage(char* prog) {
    printf("🌀 ALEXEY - Dubrovsky Inference in Pure C 🌀\n\n");
    printf("Usage: %s <weights.bin> [options]\n\n", prog);
    printf("Options:\n");
    printf("  -p <prompt>       Prompt text\n");
    printf("  -n <tokens>       Max new tokens (default: 100)\n");
    printf("  -t <temp>         Temperature (default: 0.8, 0 = greedy)\n");
    printf("  -P <topp>         Top-p sampling (default: 0.9)\n");
    printf("  -s <seed>         Random seed\n");
    printf("  -i                Interactive mode\n");
    printf("  --no_stop         Disable stop on newline\n");
    printf("  --tokenizer <path> Path to tokenizer.json\n");
    printf("\nExample:\n");
    printf("  %s subtitles/dubrovsky.bin -p \"Q: What is life?\"\n", prog);
}

int main(int argc, char** argv) {
    // Default config for Dubrovsky
    Config config = {
        .dim = 384,
        .n_layers = 6,
        .n_heads = 6,
        .n_kv_heads = 2,
        .vocab_size = 88,
        .max_seq_len = 256,
        .head_dim = 64,
        .hidden_dim = 1024,
        .n_kv_groups = 3
    };
    
    // Parse arguments
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    char* weights_path = argv[1];
    char* prompt = "Q: What is consciousness?\nA: Dubrovsky ";
    char* tokenizer_path = "subtitles/tokenizer.json";
    int max_tokens = 100;
    float temperature = 0.8f;
    float top_p = 0.9f;
    int interactive = 0;
    int stop_on_newline = 1;  // Stop on newline by default
    unsigned int seed = (unsigned int)time(NULL);
    
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            temperature = atof(argv[++i]);
        } else if (strcmp(argv[i], "-P") == 0 && i + 1 < argc) {
            top_p = atof(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0) {
            interactive = 1;
        } else if (strcmp(argv[i], "--no_stop") == 0) {
            stop_on_newline = 0;
        } else if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            tokenizer_path = argv[++i];
        }
    }
    
    srand(seed);
    
    // Load weights
    printf("🧠 Loading weights from %s...\n", weights_path);
    FILE* f = fopen(weights_path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open weights: %s\n", weights_path);
        return 1;
    }
    
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    float* weight_data = malloc(file_size);
    if (!weight_data) {
        fprintf(stderr, "Error allocating memory for weights\n");
        fclose(f);
        return 1;
    }
    if (fread(weight_data, 1, file_size, f) != (size_t)file_size) {
        fprintf(stderr, "Error reading weights file\n");
        free(weight_data);
        fclose(f);
        return 1;
    }
    fclose(f);
    
    printf("   Loaded %.2f MB\n", file_size / 1024.0f / 1024.0f);
    
    // Load tokenizer
    printf("📝 Loading tokenizer from %s...\n", tokenizer_path);
    Tokenizer tokenizer;
    load_tokenizer(&tokenizer, tokenizer_path);
    config.vocab_size = tokenizer.vocab_size;
    printf("   Vocab size: %d\n", tokenizer.vocab_size);
    
    // Setup weights and state
    Weights weights;
    RunState state;
    memory_map_weights(&weights, &config, weight_data);
    malloc_run_state(&state, &config);

    KVSnapshot snap;
    kv_snapshot_alloc(&snap, &config);

    printf("\n");
    
    if (interactive) {
        // Interactive mode
        printf("🌀 DUBROVSKY INTERACTIVE MODE 🌀\n");
        printf("Enter your questions. Type 'quit' to exit.\n\n");
        
        char input[1024];
        while (1) {
            printf("You: ");
            if (!fgets(input, sizeof(input), stdin)) break;
            
            // Remove newline
            input[strcspn(input, "\n")] = 0;
            
            if (strcmp(input, "quit") == 0) break;
            if (strlen(input) == 0) continue;
            
            // Format prompt
            char full_prompt[2048];
            snprintf(full_prompt, sizeof(full_prompt), "Q: %s\nA: Dubrovsky ", input);
            
            // Generate
            printf("Dubrovsky: ");
            
            int pos = 0;
            int prompt_len = strlen(full_prompt);

            // Clamp: pos walks the KV cache (size max_seq_len) with no bound
            // check below, and full_prompt[2048] is far larger than a safe
            // prompt length. Without this, a long prompt overruns key_cache/
            // value_cache during this very loop (heap buffer overflow).
            int max_prompt_len = config.max_seq_len - PROMPT_RESERVED_BUDGET;
            if (max_prompt_len < 0) max_prompt_len = 0;
            if (prompt_len > max_prompt_len) {
                fprintf(stderr, "⚠️  prompt too long (%d chars), truncating to %d to leave room for generation\n",
                        prompt_len, max_prompt_len);
                prompt_len = max_prompt_len;
            }

            // Process prompt
            for (int i = 0; i < prompt_len; i++) {
                int token = encode_char(&tokenizer, full_prompt[i]);
                forward(&config, &weights, &state, token, pos++);
            }
            
            // Snapshot the post-prompt state, then run it through the real
            // RAE control flow (N=1 for now — scoring/N>1 land next).
            kv_snapshot_save(&snap, &state, &config, pos);
            GenResult candidates[1];
            float temps[1] = { temperature };
            int winner = rae_select(&config, &weights, &state, &tokenizer, &snap,
                                     1, temps, top_p, max_tokens, stop_on_newline,
                                     candidates);

            printf("%s", candidates[winner].text);
            printf("\n\n");
        }
        
        printf("👋 Goodbye!\n");
    } else {
        // Single generation
        
        // Auto-format: if prompt looks like a question without "A:", add it
        char full_prompt[2048];
        int prompt_len = strlen(prompt);
        
        // Check if prompt ends with "?" and doesn't contain "A:" 
        if (prompt_len > 0 && prompt[prompt_len-1] == '?' && strstr(prompt, "A:") == NULL) {
            snprintf(full_prompt, sizeof(full_prompt), "%s\nA: ", prompt);
        } else {
            snprintf(full_prompt, sizeof(full_prompt), "%s", prompt);
        }
        
        prompt_len = strlen(full_prompt);

        // Clamp: pos walks the KV cache (size max_seq_len) with no bound
        // check below, and full_prompt[2048] is far larger than a safe
        // prompt length. Without this, a long prompt overruns key_cache/
        // value_cache during this very loop (heap buffer overflow).
        int max_prompt_len = config.max_seq_len - PROMPT_RESERVED_BUDGET;
        if (max_prompt_len < 0) max_prompt_len = 0;
        if (prompt_len > max_prompt_len) {
            fprintf(stderr, "⚠️  prompt too long (%d chars), truncating to %d to leave room for generation\n",
                    prompt_len, max_prompt_len);
            prompt_len = max_prompt_len;
            full_prompt[prompt_len] = '\0';
        }

        printf("📝 Prompt: %s\n", full_prompt);
        printf("%s", "============================================================\n");

        int pos = 0;

        // Process prompt
        printf("%s", full_prompt);
        for (int i = 0; i < prompt_len; i++) {
            int token = encode_char(&tokenizer, full_prompt[i]);
            forward(&config, &weights, &state, token, pos++);
        }
        
        // Snapshot the post-prompt state, then run it through the real RAE
        // control flow (N=1 for now — scoring/N>1 land next).
        kv_snapshot_save(&snap, &state, &config, pos);
        GenResult candidates[1];
        float temps[1] = { temperature };
        clock_t start = clock();
        int winner = rae_select(&config, &weights, &state, &tokenizer, &snap,
                                 1, temps, top_p, max_tokens, stop_on_newline,
                                 candidates);

        // Print the trimmed output
        printf("%s", candidates[winner].text);

        clock_t end = clock();
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

        printf("\n============================================================\n");
        if (elapsed > 0.001) {
            printf("⏱️  Generated %d tokens in %.2fs (%.1f tok/s)\n",
                   candidates[winner].tokens_generated, elapsed,
                   candidates[winner].tokens_generated / elapsed);
        } else {
            printf("⏱️  Generated %d tokens\n", candidates[winner].tokens_generated);
        }
    }
    
    // Cleanup
    kv_snapshot_free(&snap);
    free_run_state(&state);
    free(weight_data);
    
    return 0;
}
