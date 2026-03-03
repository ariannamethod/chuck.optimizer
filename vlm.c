/*
 * micro_vlm.c v5 — Vision-Language Model in pure C
 *
 * Sees images. Speaks words. Zero dependencies beyond libc/libm.
 * Tape-based autograd with arena bump allocator.
 *
 * Architecture:
 *   ViT-style patch tokenization → RoPE → GQA multi-head causal attention →
 *   SwiGLU MLP → RMSNorm → weight-tied lm_head → text
 *
 * v5: Chuck remembers.
 *   - Persistent memory across training runs (chuck.mem, binary append)
 *   - Ψ (psi) = subjectivity signal = memory vs observation
 *   - Formula:
 *       θ_l -= (α × λ_Ψ × λ_l × σ) × m̂/(√v̂ + ε) + η
 *       λ_Ψ   = λ + Ψ_w × (λ_prior - λ)
 *       Ψ_w   = min(0.3, N / (N + 100))  (trust grows with experience)
 *       λ_prior = nearest_neighbor(loss, grad_norm) from chuck.mem
 *   - Lee's Continuum C: chuck.mem IS the memory space ℳ.
 *     Each snapshot is a point. NN lookup is the identity mapping I.
 *     Ψ_w is the belief function B. When Ψ → 0, fixed point s* — Chuck found himself.
 *   - When |Ψ| large: unfamiliar territory. When Ψ → 0: memory matches reality.
 *
 * v4 (preserved):
 *   - GQA (4 Q-heads, 2 KV-heads), 3 layers, 105K params
 *   - Per-layer self-modulation (λ_l), layer freezing
 *   - Self-aware SiLU, RMSNorm, RoPE
 *   - Cross-layer signal flow
 *
 * Build: cc -std=c11 -O2 -march=native -o micro_vlm micro_vlm.c -lm
 * Run:   ./micro_vlm
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

/* ---- Config ---- */
#define IMG_SIZE       8
#define PATCH_SIZE     4
#define PATCHES_SIDE   (IMG_SIZE / PATCH_SIZE)
#define N_PATCHES      (PATCHES_SIDE * PATCHES_SIDE)
#define PATCH_PX       (PATCH_SIZE * PATCH_SIZE)
#define N_VIS          N_PATCHES
#define MAX_TXT        8
#define SEQ_LEN        (N_VIS + MAX_TXT)
#define N_EMBD         48
#define N_HEAD         4
#define N_KV_HEAD      2
#define N_KV_GROUP     (N_HEAD / N_KV_HEAD)
#define HEAD_DIM       (N_EMBD / N_HEAD)
#define KV_DIM         (N_KV_HEAD * HEAD_DIM)
#define N_LAYER        3
#define MLP_DIM        (4 * N_EMBD)
#define VOCAB          17
#define BOS            15
#define EOS            16
#define STEPS          6000
#define LR_MAX         0.003f
#define WARMUP         300
#define CHUCK_B1       0.9f
#define CHUCK_B2       0.999f
#define CHUCK_EPS      1e-8f
#define GRAD_CLIP      1.0f
#define ROPE_BASE      10000.0f
#define TEMP           0.7f
#define TOPK           5
#define CHUCK_WINDOW   16
#define CHUCK_DAMP_LO  0.1f
#define CHUCK_DAMP_HI  2.0f
#define CHUCK_PSI_CAP  0.3f
#define CHUCK_PSI_HALF 100.0f
#define CHUCK_MEM_MAX  10000
#define CHUCK_MEM_FILE "chuck.mem"
#define CHUCK_REC_THR  0.25f
#define CHUCK_REC_CD   50

#define ARENA_SZ       (96 * 1024 * 1024)
#define MAX_ARR        16384
#define MAX_ENT        32768
#define MAX_PAR        128

/* ---- Tape engine ---- */
typedef struct { float *data, *grad; int size, rows, cols; } Arr;
typedef struct { int op, out, in1, in2; float aux; int ai; } Ent;
enum { OP_ADD=1, OP_MUL, OP_SCALE, OP_MATVEC, OP_RMSNORM, OP_SILU,
       OP_CE, OP_EMBED, OP_REDUCE, OP_ATTN, OP_ROPE };

static struct {
    uint8_t *arena; size_t apos, aparam;
    Arr a[MAX_ARR]; int na, npa;
    Ent e[MAX_ENT]; int ne;
    int par[MAX_PAR]; int np;
    float *cm[MAX_PAR], *cv[MAX_PAR]; int cstep;
    int on;
} T; 

static float *aalloc(size_t n) {
    size_t b = n * sizeof(float), al = (T.apos + 15) & ~(size_t)15;
    if (al + b > ARENA_SZ) { fprintf(stderr, "arena OOM\n"); exit(1); }
    T.apos = al + b; float *p = (float*)(T.arena + al); memset(p, 0, b); return p;
}
static void tape_init(void) {
    uint8_t *m = malloc(ARENA_SZ);
    if (!m) { fprintf(stderr, "OOM\n"); exit(1); }
    memset(&T, 0, sizeof(T)); T.arena = m; T.on = 1;
}
static int anew(int sz) {
    int i = T.na++; T.a[i].size = sz; T.a[i].rows = T.a[i].cols = 0;
    T.a[i].data = aalloc(sz); T.a[i].grad = aalloc(sz); return i;
}
static int mnew(int r, int c) { int i = anew(r*c); T.a[i].rows = r; T.a[i].cols = c; return i; }
static void preg(int i) {
    int pi = T.np++; T.par[pi] = i;
    T.cm[pi] = calloc(T.a[i].size, sizeof(float));
    T.cv[pi] = calloc(T.a[i].size, sizeof(float));
}
static void rec(int op, int o, int i1, int i2, float aux, int ai) {
    if (!T.on) return;
    Ent *e = &T.e[T.ne++]; e->op=op; e->out=o; e->in1=i1; e->in2=i2; e->aux=aux; e->ai=ai;
}
static void tape_reset(void) {
    T.apos = T.aparam; T.na = T.npa; T.ne = 0;
    for (int i = 0; i < T.npa; i++) memset(T.a[i].grad, 0, T.a[i].size * sizeof(float));
}

/* ---- RNG (xoshiro256**) ---- */
static uint64_t rng[4];
static uint64_t rnext(void) {
    uint64_t t = rng[1] << 17;
    rng[2] ^= rng[0]; rng[3] ^= rng[1]; rng[1] ^= rng[2]; rng[0] ^= rng[3];
    rng[2] ^= t; rng[3] = (rng[3] << 45) | (rng[3] >> 19);
    uint64_t r = rng[1] * 5; return (r << 7 | r >> 57) * 9;
}
static void rseed(uint64_t s) {
    rng[0]=s; rng[1]=s^0x6a09e667f3bcc908ULL; rng[2]=s^0xbb67ae8584caa73bULL; rng[3]=s^0x3c6ef372fe94f82bULL;
    for (int i = 0; i < 20; i++) rnext();
}
static float ruf(void) { return (float)((rnext()>>11)+1) / (float)(1ULL<<53); }
static float rnf(float mu, float s) {
    double u1 = (double)(((rnext()>>11)+1)) / (double)(1ULL<<53);
    double u2 = (double)(((rnext()>>11)+1)) / (double)(1ULL<<53);
    return mu + s * (float)(sqrt(-2.0*log(u1)) * cos(6.283185307179586*u2));
}
static inline float sigf(float x) { return 1.0f / (1.0f + expf(-x)); }

/* ===========================================================================
 * Chuck Memory — persistent across training runs
 *
 *   chuck.mem: binary append-only file of training snapshots.
 *   Each snapshot: 16 bytes (4 floats).
 *   Nearest-neighbor recall gives λ_prior.
 *   Ψ = λ_prior - λ_current = subjectivity signal.
 *
 *   Lee's Continuum C: chuck.mem is ℳ. NN is identity mapping I.
 *   Ψ_w is belief function B. Fixed point s* when Ψ → 0.
 * =========================================================================== */

typedef struct {
    float loss;           /* where Chuck was */
    float grad_norm;      /* how hard the network was shaking */
    float lambda;         /* what Chuck decided */
    float delta_loss;     /* what happened (negative = improvement) */
} ChuckMem;

static ChuckMem chuck_mem[CHUCK_MEM_MAX];
static int chuck_mem_n = 0;

static void chuck_mem_load(void) {
    FILE *f = fopen(CHUCK_MEM_FILE, "rb");
    if (!f) return;
    chuck_mem_n = (int)fread(chuck_mem, sizeof(ChuckMem), CHUCK_MEM_MAX, f);
    fclose(f);
}

static void chuck_mem_save(ChuckMem *m) {
    FILE *f = fopen(CHUCK_MEM_FILE, "ab");
    if (!f) return;
    fwrite(m, sizeof(ChuckMem), 1, f);
    fclose(f);
    if (chuck_mem_n < CHUCK_MEM_MAX)
        chuck_mem[chuck_mem_n++] = *m;
}

/* Nearest neighbor recall: find most similar past state, return its λ.
 * Distance = normalized (loss, grad_norm) difference.
 * Successful memories (negative delta_loss) get 2x weight. */
static float chuck_mem_recall(float loss, float grad_norm) {
    if (chuck_mem_n == 0) return -1.0f;  /* no memory → no prior */
    float best_dist = 1e9f, best_lambda = -1.0f;
    for (int i = 0; i < chuck_mem_n; i++) {
        float dl = (loss - chuck_mem[i].loss) / (fabsf(loss) + 1e-8f);
        float dg = (grad_norm - chuck_mem[i].grad_norm) / (fabsf(grad_norm) + 1e-8f);
        float dist = dl * dl + dg * dg;
        if (chuck_mem[i].delta_loss < 0) dist *= 0.5f;  /* prefer wins */
        if (dist < best_dist) { best_dist = dist; best_lambda = chuck_mem[i].lambda; }
    }
    return best_lambda;
}

/* ---- Self-Awareness: Eyes ---- */

/* SiLU eye: tracks dead neuron ratio */
static struct { int dead, total; float health; } SiLU_eye;

static void silu_eye_reset(void) { SiLU_eye.dead = 0; SiLU_eye.total = 0; }
static void silu_eye_update(void) {
    if (SiLU_eye.total == 0) { SiLU_eye.health = 1.0f; return; }
    SiLU_eye.health = 1.0f - (float)SiLU_eye.dead / SiLU_eye.total;
    SiLU_eye.dead = 0; SiLU_eye.total = 0;
}

/* RMSNorm eye: tracks normalization scale EMA */
static struct { float scale_ema; int init; } Norm_eye;

/* RoPE eye: tracks frequency band utilization */
static struct { float freq_energy[N_EMBD/2]; int calls; float utilization; } RoPE_eye;

static void rope_eye_reset(void) {
    memset(RoPE_eye.freq_energy, 0, sizeof(RoPE_eye.freq_energy));
    RoPE_eye.calls = 0;
}
static void rope_eye_update(void) {
    if (RoPE_eye.calls == 0) return;
    float max_e = 0;
    for (int b = 0; b < HEAD_DIM/2; b++) {
        RoPE_eye.freq_energy[b] /= RoPE_eye.calls;
        if (RoPE_eye.freq_energy[b] > max_e) max_e = RoPE_eye.freq_energy[b];
    }
    int active = 0;
    for (int b = 0; b < HEAD_DIM/2; b++)
        if (RoPE_eye.freq_energy[b] > max_e * 0.01f) active++;
    RoPE_eye.utilization = (HEAD_DIM/2 > 0) ? (float)active / (HEAD_DIM/2) : 1.0f;
    memset(RoPE_eye.freq_energy, 0, sizeof(RoPE_eye.freq_energy));
    RoPE_eye.calls = 0;
}

/* Cross-layer signal flow */
static float act_mag[N_LAYER];

/* ---- Forward ops (with awareness tracking) ---- */
static int op_add(int xi, int yi) {
    int n = T.a[xi].size, zi = anew(n);
    for (int i = 0; i < n; i++) T.a[zi].data[i] = T.a[xi].data[i] + T.a[yi].data[i];
    rec(OP_ADD,zi,xi,yi,0,0); return zi;
}
static int op_mul(int xi, int yi) {
    int n = T.a[xi].size, zi = anew(n);
    for (int i = 0; i < n; i++) T.a[zi].data[i] = T.a[xi].data[i] * T.a[yi].data[i];
    rec(OP_MUL,zi,xi,yi,0,0); return zi;
}
static int op_scale(int xi, float s) {
    int n = T.a[xi].size, zi = anew(n);
    for (int i = 0; i < n; i++) T.a[zi].data[i] = T.a[xi].data[i] * s;
    rec(OP_SCALE,zi,xi,-1,s,0); return zi;
}
static int op_mv(int Wi, int xi) {
    int r = T.a[Wi].rows, c = T.a[Wi].cols, zi = anew(r);
    for (int i = 0; i < r; i++) { float s = 0; const float *Wr = &T.a[Wi].data[i*c];
        for (int j = 0; j < c; j++) s += Wr[j] * T.a[xi].data[j]; T.a[zi].data[i] = s; }
    rec(OP_MATVEC,zi,Wi,xi,0,0); return zi;
}
static int op_rms(int xi) {
    int n = T.a[xi].size, zi = anew(n); float ms = 0;
    for (int i = 0; i < n; i++) ms += T.a[xi].data[i] * T.a[xi].data[i];
    ms = ms / n + 1e-5f; float sc = 1.0f / sqrtf(ms);
    for (int i = 0; i < n; i++) T.a[zi].data[i] = T.a[xi].data[i] * sc;
    /* Norm eye: track scale */
    if (Norm_eye.init) Norm_eye.scale_ema = 0.99f * Norm_eye.scale_ema + 0.01f * sc;
    else { Norm_eye.scale_ema = sc; Norm_eye.init = 1; }
    rec(OP_RMSNORM,zi,xi,-1,sc,n); return zi;
}
static int op_silu(int xi) {
    int n = T.a[xi].size, zi = anew(n);
    for (int i = 0; i < n; i++) {
        float x = T.a[xi].data[i]; float s = sigf(x);
        T.a[zi].data[i] = x * s;
        /* SiLU eye: track dead zone */
        if (x < -4.0f) SiLU_eye.dead++;
        SiLU_eye.total++;
    }
    rec(OP_SILU,zi,xi,-1,0,0); return zi;
}
static int op_embed(int Wi, int id) {
    int c = T.a[Wi].cols, zi = anew(c);
    memcpy(T.a[zi].data, &T.a[Wi].data[id*c], c * sizeof(float));
    rec(OP_EMBED,zi,Wi,-1,0,id); return zi;
}
static int op_ce(int li, int tgt) {
    int n = T.a[li].size; float mx = T.a[li].data[0];
    for (int i = 1; i < n; i++) if (T.a[li].data[i] > mx) mx = T.a[li].data[i];
    int pi = anew(n); float *p = T.a[pi].data; float s = 0;
    for (int i = 0; i < n; i++) { p[i] = expf(T.a[li].data[i] - mx); s += p[i]; }
    for (int i = 0; i < n; i++) p[i] /= (s + 1e-10f);
    int zi = anew(1); T.a[zi].data[0] = -logf(p[tgt] + 1e-10f);
    rec(OP_CE,zi,li,pi,(float)tgt,n); return zi;
}
static int op_rope(int xi, int pos) {
    int n = T.a[xi].size, zi = anew(n);
    memcpy(T.a[zi].data, T.a[xi].data, n * sizeof(float));
    float *d = T.a[zi].data;
    int n_heads = n / HEAD_DIM;
    for (int h = 0; h < n_heads; h++) {
        for (int i = 0; i < HEAD_DIM; i += 2) {
            float freq = 1.0f / powf(ROPE_BASE, (float)i / (float)HEAD_DIM);
            float ang = pos * freq, c = cosf(ang), s = sinf(ang);
            int idx = h * HEAD_DIM + i;
            float x0 = d[idx], x1 = d[idx+1];
            d[idx] = x0*c - x1*s; d[idx+1] = x0*s + x1*c;
            /* RoPE eye: track frequency energy */
            float energy = d[idx]*d[idx] + d[idx+1]*d[idx+1];
            if (i/2 < N_EMBD/2) RoPE_eye.freq_energy[i/2] += energy;
        }
    }
    RoPE_eye.calls++;
    rec(OP_ROPE,zi,xi,-1,0,pos); return zi;
}
static int op_reduce(int *la, int n) {
    float s = 0; for (int i = 0; i < n; i++) s += T.a[la[i]].data[0];
    int zi = anew(1); T.a[zi].data[0] = s / n;
    int buf = anew(n); for (int i = 0; i < n; i++) ((int*)T.a[buf].data)[i] = la[i];
    rec(OP_REDUCE,zi,buf,-1,0,n); return zi;
}

/* ---- KV cache (GQA: KV_DIM, not N_EMBD) ---- */
static float *kv_k[N_LAYER][SEQ_LEN], *kv_v[N_LAYER][SEQ_LEN];
static int kv_ki[N_LAYER][SEQ_LEN], kv_vi[N_LAYER][SEQ_LEN];
static void kv_clear(void) {
    memset(kv_k,0,sizeof(kv_k)); memset(kv_v,0,sizeof(kv_v));
    memset(kv_ki,0,sizeof(kv_ki)); memset(kv_vi,0,sizeof(kv_vi));
}

/* ---- Backward ---- */
static void backward(int loss) {
    T.a[loss].grad[0] = 1.0f;
    for (int ei = T.ne - 1; ei >= 0; ei--) {
        Ent *e = &T.e[ei];
        Arr *out = &T.a[e->out], *i1 = (e->in1 >= 0) ? &T.a[e->in1] : NULL, *i2 = (e->in2 >= 0) ? &T.a[e->in2] : NULL;
        switch (e->op) {
        case OP_ADD: { int n = out->size;
            for (int i = 0; i < n; i++) { i1->grad[i] += out->grad[i]; i2->grad[i] += out->grad[i]; } break; }
        case OP_MUL: { int n = out->size;
            for (int i = 0; i < n; i++) { i1->grad[i] += out->grad[i]*i2->data[i]; i2->grad[i] += out->grad[i]*i1->data[i]; } break; }
        case OP_SCALE: { int n = out->size; float s = e->aux;
            for (int i = 0; i < n; i++) i1->grad[i] += out->grad[i] * s; break; }
        case OP_MATVEC: { int r = i1->rows, c = i1->cols;
            for (int i = 0; i < r; i++) { float dz = out->grad[i];
                for (int j = 0; j < c; j++) { i1->grad[i*c+j] += dz*i2->data[j]; i2->grad[j] += dz*i1->data[i*c+j]; } } break; }
        case OP_RMSNORM: { int n = e->ai; float sc = e->aux, dot = 0;
            for (int i = 0; i < n; i++) dot += out->grad[i] * out->data[i];
            for (int i = 0; i < n; i++) i1->grad[i] += sc * (out->grad[i] - out->data[i]*dot/n); break; }
        case OP_SILU: { int n = out->size;
            for (int i = 0; i < n; i++) { float sg = sigf(i1->data[i]); i1->grad[i] += out->grad[i]*sg*(1.0f+i1->data[i]*(1.0f-sg)); } break; }
        case OP_CE: { int n = e->ai; int tgt = (int)e->aux; float dl = out->grad[0];
            for (int i = 0; i < n; i++) i1->grad[i] += dl * (i2->data[i] - (i==tgt ? 1.0f : 0.0f)); break; }
        case OP_EMBED: { int id = e->ai, c = i1->cols;
            for (int j = 0; j < c; j++) i1->grad[id*c+j] += out->grad[j]; break; }
        case OP_ROPE: { int n = out->size, pos = e->ai;
            int nh = n / HEAD_DIM;
            for (int h = 0; h < nh; h++) {
                for (int i = 0; i < HEAD_DIM; i += 2) {
                    float freq = 1.0f / powf(ROPE_BASE, (float)i/(float)HEAD_DIM);
                    float ang = pos*freq, c = cosf(ang), s = sinf(ang);
                    int idx = h * HEAD_DIM + i;
                    float g0 = out->grad[idx], g1 = out->grad[idx+1];
                    i1->grad[idx] += g0*c + g1*s; i1->grad[idx+1] += -g0*s + g1*c;
                }
            } break; }
        case OP_ATTN: { /* GQA attention backward */
            int li = (int)e->aux, pos = e->ai;
            float *qd = i1->data, *ag = out->grad, isq = 1.0f / sqrtf((float)HEAD_DIM);
            for (int h = 0; h < N_HEAD; h++) {
                int hs = h * HEAD_DIM;
                int kvh = h / N_KV_GROUP;
                int kvs = kvh * HEAD_DIM;
                float sc[SEQ_LEN], mx = -1e9f;
                for (int t = 0; t <= pos; t++) { float s = 0;
                    for (int d = 0; d < HEAD_DIM; d++) s += qd[hs+d]*kv_k[li][t][kvs+d];
                    sc[t] = s*isq; if (sc[t] > mx) mx = sc[t]; }
                float sm = 0; for (int t = 0; t <= pos; t++) { sc[t] = expf(sc[t]-mx); sm += sc[t]; }
                for (int t = 0; t <= pos; t++) sc[t] /= (sm + 1e-10f);
                float dw[SEQ_LEN];
                for (int t = 0; t <= pos; t++) { dw[t] = 0;
                    for (int d = 0; d < HEAD_DIM; d++) dw[t] += kv_v[li][t][kvs+d]*ag[hs+d]; }
                float dot = 0; for (int t = 0; t <= pos; t++) dot += sc[t]*dw[t];
                for (int t = 0; t <= pos; t++) { float ds = sc[t]*(dw[t]-dot);
                    for (int d = 0; d < HEAD_DIM; d++) {
                        /* grad Q: each Q-head gets its own gradient */
                        i1->grad[hs+d] += ds * kv_k[li][t][kvs+d] * isq;
                        /* grad K: multiple Q-heads accumulate to shared KV-head */
                        T.a[kv_ki[li][t]].grad[kvs+d] += ds * qd[hs+d] * isq;
                        /* grad V: multiple Q-heads accumulate to shared KV-head */
                        T.a[kv_vi[li][t]].grad[kvs+d] += sc[t] * ag[hs+d];
                    } }
            } break; }
        case OP_REDUCE: { int n = e->ai; int *idxs = (int*)i1->data; float dl = out->grad[0]/n;
            for (int i = 0; i < n; i++) T.a[idxs[i]].grad[0] += dl; break; }
        }
    }
}

/* ===========================================================================
 * Chuck v4: Self-Aware Optimizer
 *
 *   θ_l -= (α × λ × λ_l × σ) × m̂/(√v̂ + ε) + η
 *
 *   λ   = global self-modulation (loss trend over 16-step window)
 *   λ_l = per-layer self-modulation (gradient norm trend per layer)
 *   σ   = activation health signal (SiLU alive ratio × norm stability)
 *   η   = stagnation noise (only when globally stuck)
 *   α   = base learning rate from cosine schedule
 *
 *   If λ_l = 0 → layer is frozen. Zero compute waste. Chuck decided it's done.
 *   Adam doesn't know which layers are done. Chuck does.
 * =========================================================================== */

/* Per-layer awareness state */
typedef struct {
    float grad_hist[CHUCK_WINDOW];
    float dampen;
    int frozen;
    int pos, full, stag;
} ChuckLayer;

/* Global awareness state */
static struct {
    float hist[CHUCK_WINDOW];
    float dampen, noise, sigma;
    float loss_ema;         /* EMA-smoothed loss (batch noise filter) */
    float psi;              /* Ψ: subjectivity signal (memory - observation) */
    float psi_w;            /* Ψ weight: trust in memory (0 → 0.3) */
    float rec_lambda;       /* λ at last memory recording */
    float rec_loss;         /* loss at last memory recording */
    int rec_frozen[N_LAYER]; /* frozen state at last recording */
    int rec_cd;             /* cooldown counter (steps since last record) */
    int pos, full, stag;
} Chuck;

static ChuckLayer CL[N_LAYER];

static void chuck_init(void) {
    memset(&Chuck, 0, sizeof(Chuck));
    Chuck.dampen = 1.0f; Chuck.sigma = 1.0f;
    Chuck.rec_lambda = 1.0f; Chuck.rec_loss = 999.0f;
    memset(Chuck.rec_frozen, 0, sizeof(Chuck.rec_frozen));
    Chuck.psi = 0; Chuck.psi_w = 0;
    for (int l = 0; l < N_LAYER; l++) {
        memset(&CL[l], 0, sizeof(ChuckLayer));
        CL[l].dampen = 1.0f;
    }
    Norm_eye.init = 0; Norm_eye.scale_ema = 1.0f;
    SiLU_eye.health = 1.0f;
    RoPE_eye.utilization = 1.0f;
    /* Load persistent memory */
    chuck_mem_load();
    if (chuck_mem_n > 0)
        printf("  chuck: loaded %d memories from %s (Ψ_w=%.2f)\n",
               chuck_mem_n, CHUCK_MEM_FILE,
               fminf(CHUCK_PSI_CAP, (float)chuck_mem_n / ((float)chuck_mem_n + CHUCK_PSI_HALF)));
}

/* Which layer does param pi belong to? -1 = global (patch_proj, wte) */
static int param_layer(int pi) {
    if (pi < 2) return -1;  /* 0=patch_proj, 1=wte */
    return (pi - 2) / 7;     /* 7 params per layer: wq,wk,wv,wo,w1,w3,w2 */
}

static void chuck_step(float lr, float loss) {
    /* ═══ Level 1: Global self-awareness (loss trend) ═══ */
    /* EMA smoothing: filters batch-to-batch noise for mini-batch SGD */
    if (Chuck.loss_ema == 0.0f) Chuck.loss_ema = loss;
    else Chuck.loss_ema = 0.99f * Chuck.loss_ema + 0.01f * loss;
    Chuck.hist[Chuck.pos % CHUCK_WINDOW] = Chuck.loss_ema;
    Chuck.pos++;
    if (Chuck.pos >= CHUCK_WINDOW) Chuck.full = 1;
    if (Chuck.full) {
        int q = CHUCK_WINDOW / 4;
        float recent = 0, old = 0;
        for (int i = 0; i < q; i++) {
            recent += Chuck.hist[(Chuck.pos - 1 - i) % CHUCK_WINDOW];
            old += Chuck.hist[(Chuck.pos - CHUCK_WINDOW + i) % CHUCK_WINDOW];
        }
        recent /= q; old /= q;
        float trend = (recent - old) / (old + 1e-8f);
        if (trend > 0.01f) Chuck.dampen *= 0.95f;        /* loss rising → dampen */
        else if (trend < -0.05f) Chuck.dampen *= 1.05f;   /* loss falling → boost */
        if (fabsf(trend) < 0.001f) {
            Chuck.stag++;
            if (Chuck.stag > 8) { Chuck.noise = 0.001f; Chuck.stag = 0; }
        } else { Chuck.stag = 0; Chuck.noise *= 0.9f; }
        if (Chuck.dampen < CHUCK_DAMP_LO) Chuck.dampen = CHUCK_DAMP_LO;
        if (Chuck.dampen > CHUCK_DAMP_HI) Chuck.dampen = CHUCK_DAMP_HI;
    }

    /* ═══ Level 4: Activation health signal (σ) ═══ */
    silu_eye_update();
    rope_eye_update();
    Chuck.sigma = 1.0f;
    if (SiLU_eye.health < 0.7f) Chuck.sigma *= SiLU_eye.health / 0.7f;
    if (Norm_eye.scale_ema > 5.0f) Chuck.sigma *= 0.9f;
    if (Norm_eye.scale_ema < 0.2f) Chuck.sigma *= 0.9f;

    /* ═══ Level 2: Per-layer self-awareness (grad norm trend) ═══ */
    float layer_gnorm[N_LAYER];
    memset(layer_gnorm, 0, sizeof(layer_gnorm));
    for (int pi = 0; pi < T.np; pi++) {
        int l = param_layer(pi);
        if (l < 0 || l >= N_LAYER) continue;
        Arr *p = &T.a[T.par[pi]];
        float gn = 0;
        for (int i = 0; i < p->size; i++) gn += p->grad[i] * p->grad[i];
        layer_gnorm[l] += gn;
    }
    for (int l = 0; l < N_LAYER; l++) layer_gnorm[l] = sqrtf(layer_gnorm[l]);

    for (int l = 0; l < N_LAYER; l++) {
        if (CL[l].frozen) continue;
        CL[l].grad_hist[CL[l].pos % CHUCK_WINDOW] = layer_gnorm[l];
        CL[l].pos++;
        if (CL[l].pos >= CHUCK_WINDOW) CL[l].full = 1;
        if (CL[l].full) {
            int q = CHUCK_WINDOW / 4;
            float recent = 0, old = 0;
            for (int i = 0; i < q; i++) {
                recent += CL[l].grad_hist[(CL[l].pos - 1 - i) % CHUCK_WINDOW];
                old += CL[l].grad_hist[(CL[l].pos - CHUCK_WINDOW + i) % CHUCK_WINDOW];
            }
            recent /= q; old /= q;
            float trend = (recent - old) / (old + 1e-8f);
            /* grad norm trending up → layer needs more work → boost */
            if (trend > 0.05f) CL[l].dampen *= 1.05f;
            /* grad norm trending down → layer is settling → dampen */
            else if (trend < -0.05f) CL[l].dampen *= 0.95f;
            /* freeze: near-zero gradient norm for extended period */
            if (layer_gnorm[l] < 0.01f) {
                CL[l].stag++;
                if (CL[l].stag > 8) CL[l].frozen = 1;
            } else { CL[l].stag = 0; }
            if (CL[l].dampen < CHUCK_DAMP_LO) CL[l].dampen = CHUCK_DAMP_LO;
            if (CL[l].dampen > CHUCK_DAMP_HI) CL[l].dampen = CHUCK_DAMP_HI;
        }
    }

    /* ═══ Level 5: Cross-layer signal flow ═══ */
    if (act_mag[0] > 1e-8f) {
        float ratio = act_mag[N_LAYER-1] / (act_mag[0] + 1e-8f);
        for (int l = 1; l < N_LAYER; l++) {
            if (CL[l].frozen) continue;
            float depth = (float)l / (N_LAYER - 1);
            if (ratio < 0.3f) CL[l].dampen *= (1.0f + 0.02f * depth);       /* vanishing → boost deep */
            else if (ratio > 3.0f) CL[l].dampen *= (1.0f - 0.02f * depth);  /* exploding → dampen deep */
            if (CL[l].dampen < CHUCK_DAMP_LO) CL[l].dampen = CHUCK_DAMP_LO;
            if (CL[l].dampen > CHUCK_DAMP_HI) CL[l].dampen = CHUCK_DAMP_HI;
        }
    }

    /* ═══ Level 6: Ψ — Subjectivity (memory vs observation) ═══ */
    /*
     *   λ_Ψ   = λ + Ψ_w × (λ_prior - λ)
     *   Ψ_w   = min(0.3, N / (N + 100))
     *   λ_prior = nearest_neighbor(loss, grad_norm) from chuck.mem
     *
     *   When Ψ → 0: memory matches reality. Chuck is home.
     *   When |Ψ| large: unfamiliar territory. Chuck explores.
     */
    float gnorm_sq = 0;
    for (int pi = 0; pi < T.np; pi++) { Arr *p = &T.a[T.par[pi]];
        for (int i = 0; i < p->size; i++) gnorm_sq += p->grad[i] * p->grad[i]; }
    float gnorm = sqrtf(gnorm_sq + 1e-8f);

    Chuck.psi_w = (chuck_mem_n > 0) ?
        fminf(CHUCK_PSI_CAP, (float)chuck_mem_n / ((float)chuck_mem_n + CHUCK_PSI_HALF)) : 0.0f;

    float lambda_psi = Chuck.dampen;  /* default: pure reactive */
    if (chuck_mem_n > 0) {
        float lambda_prior = chuck_mem_recall(loss, gnorm);
        if (lambda_prior > 0) {
            Chuck.psi = lambda_prior - Chuck.dampen;
            lambda_psi = Chuck.dampen + Chuck.psi_w * Chuck.psi;
            if (lambda_psi < CHUCK_DAMP_LO) lambda_psi = CHUCK_DAMP_LO;
            if (lambda_psi > CHUCK_DAMP_HI) lambda_psi = CHUCK_DAMP_HI;
        }
    }

    /* Record memory on regime change — Chuck speaks rarely, but always on point */
    Chuck.rec_cd++;
    if (Chuck.full && Chuck.rec_cd >= CHUCK_REC_CD) {
        float delta_loss = loss - Chuck.rec_loss;
        float lambda_shift = fabsf(Chuck.dampen - Chuck.rec_lambda) / (Chuck.rec_lambda + 1e-8f);
        int regime_change = (lambda_shift > CHUCK_REC_THR);  /* λ shifted >25% */
        for (int l = 0; l < N_LAYER && !regime_change; l++)
            if (CL[l].frozen != Chuck.rec_frozen[l]) regime_change = 1;
        if (regime_change) {
            ChuckMem snap = { loss, gnorm, Chuck.dampen, delta_loss };
            chuck_mem_save(&snap);
            Chuck.rec_lambda = Chuck.dampen;
            Chuck.rec_loss = loss;
            Chuck.rec_cd = 0;
            for (int l = 0; l < N_LAYER; l++) Chuck.rec_frozen[l] = CL[l].frozen;
        }
    }

    /* ═══ Apply parameter updates ═══ */
    T.cstep++;
    float bc1 = 1.0f - powf(CHUCK_B1, (float)T.cstep);
    float bc2 = 1.0f - powf(CHUCK_B2, (float)T.cstep);

    /* Global gradient clipping */
    float clip = (gnorm > GRAD_CLIP) ? GRAD_CLIP / gnorm : 1.0f;

    for (int pi = 0; pi < T.np; pi++) {
        int l = param_layer(pi);
        /* Frozen layer → skip entirely */
        if (l >= 0 && l < N_LAYER && CL[l].frozen) continue;
        float layer_damp = (l >= 0 && l < N_LAYER) ? CL[l].dampen : 1.0f;
        float eff_lr = lr * lambda_psi * layer_damp * Chuck.sigma;

        int idx = T.par[pi]; Arr *p = &T.a[idx];
        float *m = T.cm[pi], *v = T.cv[pi];
        for (int i = 0; i < p->size; i++) { float g = p->grad[i] * clip;
            m[i] = CHUCK_B1*m[i] + (1.0f-CHUCK_B1)*g;
            v[i] = CHUCK_B2*v[i] + (1.0f-CHUCK_B2)*g*g;
            p->data[i] -= eff_lr * (m[i]/bc1) / (sqrtf(v[i]/bc2) + CHUCK_EPS);
            if (Chuck.noise > 0) p->data[i] += Chuck.noise * rnf(0, 1.0f);
        }
    }
}

/* ---- Synthetic digit data ---- */
static const float digit_pat[10][IMG_SIZE*IMG_SIZE] = {
    {0,0,.5f,.8f,.8f,.5f,0,0, 0,.5f,.8f,0,0,.8f,.5f,0, .5f,.8f,0,0,0,0,.8f,.5f, .8f,0,0,0,0,0,0,.8f, .8f,0,0,0,0,0,0,.8f, .5f,.8f,0,0,0,0,.8f,.5f, 0,.5f,.8f,0,0,.8f,.5f,0, 0,0,.5f,.8f,.8f,.5f,0,0},
    {0,0,0,.8f,0,0,0,0, 0,0,.5f,.8f,0,0,0,0, 0,0,0,.8f,0,0,0,0, 0,0,0,.8f,0,0,0,0, 0,0,0,.8f,0,0,0,0, 0,0,0,.8f,0,0,0,0, 0,0,0,.8f,0,0,0,0, 0,0,.5f,.8f,.8f,.5f,0,0},
    {0,.5f,.8f,.8f,.8f,.5f,0,0, 0,0,0,0,0,.8f,0,0, 0,0,0,0,0,.8f,0,0, 0,0,0,.5f,.8f,.5f,0,0, 0,0,.5f,.8f,0,0,0,0, 0,.5f,.8f,0,0,0,0,0, .5f,.8f,0,0,0,0,0,0, .5f,.8f,.8f,.8f,.8f,.8f,.5f,0},
    {.5f,.8f,.8f,.8f,.5f,0,0,0, 0,0,0,0,.8f,0,0,0, 0,0,0,0,.8f,0,0,0, 0,.5f,.8f,.8f,.5f,0,0,0, 0,0,0,0,.8f,0,0,0, 0,0,0,0,.8f,0,0,0, 0,0,0,0,.8f,0,0,0, .5f,.8f,.8f,.8f,.5f,0,0,0},
    {.8f,0,0,0,.8f,0,0,0, .8f,0,0,0,.8f,0,0,0, .8f,0,0,0,.8f,0,0,0, .8f,.8f,.8f,.8f,.8f,.8f,0,0, 0,0,0,0,.8f,0,0,0, 0,0,0,0,.8f,0,0,0, 0,0,0,0,.8f,0,0,0, 0,0,0,0,.8f,0,0,0},
    {.5f,.8f,.8f,.8f,.8f,.5f,0,0, .8f,0,0,0,0,0,0,0, .8f,0,0,0,0,0,0,0, .5f,.8f,.8f,.8f,.5f,0,0,0, 0,0,0,0,.8f,0,0,0, 0,0,0,0,.8f,0,0,0, 0,0,0,0,.8f,0,0,0, .5f,.8f,.8f,.8f,.5f,0,0,0},
    {0,0,.5f,.8f,.8f,.5f,0,0, 0,.5f,.8f,0,0,0,0,0, .5f,.8f,0,0,0,0,0,0, .8f,.5f,.8f,.8f,.5f,0,0,0, .8f,0,0,0,.8f,0,0,0, .8f,0,0,0,.8f,0,0,0, .5f,.8f,0,0,.8f,0,0,0, 0,.5f,.8f,.8f,.5f,0,0,0},
    {.8f,.8f,.8f,.8f,.8f,.8f,0,0, 0,0,0,0,.5f,.8f,0,0, 0,0,0,0,.8f,.5f,0,0, 0,0,0,.5f,.8f,0,0,0, 0,0,0,.8f,.5f,0,0,0, 0,0,.5f,.8f,0,0,0,0, 0,0,.8f,.5f,0,0,0,0, 0,0,.8f,0,0,0,0,0},
    {0,.5f,.8f,.8f,.5f,0,0,0, .8f,0,0,0,.8f,0,0,0, .8f,0,0,0,.8f,0,0,0, 0,.5f,.8f,.8f,.5f,0,0,0, .8f,0,0,0,.8f,0,0,0, .8f,0,0,0,.8f,0,0,0, .8f,0,0,0,.8f,0,0,0, 0,.5f,.8f,.8f,.5f,0,0,0},
    {0,.5f,.8f,.8f,.5f,0,0,0, .8f,0,0,0,.8f,0,0,0, .8f,0,0,0,.8f,0,0,0, 0,.5f,.8f,.8f,.8f,0,0,0, 0,0,0,0,.8f,0,0,0, 0,0,0,0,.8f,0,0,0, 0,0,0,.5f,.8f,0,0,0, 0,.5f,.8f,.8f,.5f,0,0,0},
};
typedef struct { float **imgs; int *labels; int n; } Data;
static Data gen_data(int n) {
    Data d; d.n = n; d.labels = malloc(n * sizeof(int)); d.imgs = malloc(n * sizeof(float*));
    for (int i = 0; i < n; i++) { int l = i % 10; d.labels[i] = l;
        d.imgs[i] = malloc(IMG_SIZE*IMG_SIZE*sizeof(float));
        for (int p = 0; p < IMG_SIZE*IMG_SIZE; p++) { float v = digit_pat[l][p] + rnf(0, 0.07f);
            d.imgs[i][p] = v < 0 ? 0 : v > 1 ? 1 : v; } }
    return d;
}
static const char *names[] = {"zero","one","two","three","four","five","six","seven","eight","nine"};
static const char chars[] = "efghinorstuvwxz";
#define N_CHARS 15
static int c2id(char c) { for (int i = 0; i < N_CHARS; i++) if (chars[i] == c) return i; return -1; }
static char id2c(int i) { return (i == BOS) ? '^' : (i == EOS) ? '$' : (i >= 0 && i < N_CHARS) ? chars[i] : '?'; }

/* ---- Model (GQA: wk/wv use KV_DIM) ---- */
typedef struct {
    int patch_proj, wte;
    struct { int wq, wk, wv, wo, w1, w3, w2; } L[N_LAYER];
} Model;
static Model M;

static int init_w(int r, int c, float s) {
    int i = mnew(r, c);
    for (int j = 0; j < r*c; j++) T.a[i].data[j] = rnf(0, s);
    preg(i); return i;
}
static void init_model(void) {
    M.patch_proj = init_w(N_EMBD, PATCH_PX, 0.1f);        /* param 0 */
    M.wte = init_w(VOCAB, N_EMBD, 0.08f);                  /* param 1 */
    for (int i = 0; i < N_LAYER; i++) {
        float s = 0.08f / sqrtf(2.0f * N_LAYER);
        M.L[i].wq = init_w(N_EMBD, N_EMBD, s);             /* param 2+7i+0 */
        M.L[i].wk = init_w(KV_DIM, N_EMBD, s);             /* param 2+7i+1: GQA! */
        M.L[i].wv = init_w(KV_DIM, N_EMBD, s);             /* param 2+7i+2: GQA! */
        M.L[i].wo = init_w(N_EMBD, N_EMBD, s);             /* param 2+7i+3 */
        M.L[i].w1 = init_w(MLP_DIM, N_EMBD, s);            /* param 2+7i+4 */
        M.L[i].w3 = init_w(MLP_DIM, N_EMBD, s);            /* param 2+7i+5 */
        M.L[i].w2 = init_w(N_EMBD, MLP_DIM, s);            /* param 2+7i+6 */
    }
    T.npa = T.na; T.aparam = T.apos;
}

/* ---- GPT step (one position, GQA attention) ---- */
static int gpt_step(int x, int pos, int layer_track) {
    int h = x;
    for (int li = 0; li < N_LAYER; li++) {
        int res = h; h = op_rms(h);
        int qi = op_mv(M.L[li].wq, h);
        int ki = op_mv(M.L[li].wk, h);  /* KV_DIM output */
        int vi = op_mv(M.L[li].wv, h);  /* KV_DIM output */
        int rqi = op_rope(qi, pos);
        int rki = op_rope(ki, pos);      /* RoPE on KV_DIM */
        kv_k[li][pos] = T.a[rki].data; kv_v[li][pos] = T.a[vi].data;
        kv_ki[li][pos] = rki; kv_vi[li][pos] = vi;

        /* GQA multi-head attention */
        int ao = anew(N_EMBD); float *ad = T.a[ao].data;
        for (int h_ = 0; h_ < N_HEAD; h_++) {
            int hs = h_ * HEAD_DIM;         /* Q offset in N_EMBD */
            int kvh = h_ / N_KV_GROUP;      /* which KV head */
            int kvs = kvh * HEAD_DIM;        /* KV offset in KV_DIM */
            float sc[SEQ_LEN], mx = -1e9f; float *qd = T.a[rqi].data;
            for (int t = 0; t <= pos; t++) { float s = 0;
                for (int d = 0; d < HEAD_DIM; d++) s += qd[hs+d]*kv_k[li][t][kvs+d];
                sc[t] = s / sqrtf((float)HEAD_DIM); if (sc[t] > mx) mx = sc[t]; }
            float sm = 0; for (int t = 0; t <= pos; t++) { sc[t] = expf(sc[t]-mx); sm += sc[t]; }
            for (int t = 0; t <= pos; t++) sc[t] /= (sm + 1e-10f);
            for (int d = 0; d < HEAD_DIM; d++) { float v = 0;
                for (int t = 0; t <= pos; t++) v += sc[t]*kv_v[li][t][kvs+d]; ad[hs+d] = v; }
        }
        rec(OP_ATTN, ao, rqi, -1, (float)li, pos);
        h = op_add(res, op_mv(M.L[li].wo, ao));

        /* Track activation magnitude for cross-layer signal */
        if (layer_track) {
            float rms = 0;
            for (int i = 0; i < N_EMBD; i++) rms += T.a[h].data[i] * T.a[h].data[i];
            act_mag[li] = sqrtf(rms / N_EMBD);
        }

        res = h; h = op_rms(h);
        int gate = op_silu(op_mv(M.L[li].w1, h)), up = op_mv(M.L[li].w3, h);
        h = op_add(res, op_mv(M.L[li].w2, op_mul(gate, up)));
    }
    return op_mv(M.wte, op_rms(h));  /* weight-tied lm_head */
}

/* ---- Vision encoder: ViT-style patch tokenization ---- */
static void encode_vis(float *pix, int *tok) {
    for (int py = 0; py < PATCHES_SIDE; py++)
        for (int px = 0; px < PATCHES_SIDE; px++) {
            int pi = anew(PATCH_PX); float *pd = T.a[pi].data;
            for (int y = 0; y < PATCH_SIZE; y++)
                for (int x = 0; x < PATCH_SIZE; x++)
                    pd[y*PATCH_SIZE + x] = pix[(py*PATCH_SIZE+y)*IMG_SIZE + px*PATCH_SIZE+x];
            tok[py*PATCHES_SIDE + px] = op_mv(M.patch_proj, pi);
        }
}

static float cos_lr(int step, int total) {
    if (step < WARMUP) return LR_MAX * (float)step / WARMUP;
    float p = (float)(step - WARMUP) / (float)(total - WARMUP);
    return LR_MAX * 0.5f * (1.0f + cosf(3.14159265f * p));
}

/* ---- Training ---- */
static void train(Data *data) {
    printf("\n=== TRAINING (%d steps, Chuck v5 — memory + subjectivity) ===\n", STEPS);
    int tp = 0; for (int i = 0; i < T.np; i++) tp += T.a[T.par[i]].size;
    printf("  %d params (%.1fK) | %d layers | GQA %dQ/%dKV | embd=%d | %dx%d patches | RoPE | weight-tied\n\n",
           tp, tp/1000.0f, N_LAYER, N_HEAD, N_KV_HEAD, N_EMBD, PATCHES_SIDE, PATCHES_SIDE);
    float rl = 0; int rn = 0;
    for (int step = 0; step < STEPS; step++) {
        int idx = (int)(rnext() % (uint64_t)data->n); int label = data->labels[idx];
        const char *name = names[label]; int nlen = strlen(name);
        int toks[MAX_TXT+2]; int nt = 0;
        toks[nt++] = BOS; for (int i = 0; i < nlen; i++) toks[nt++] = c2id(name[i]); toks[nt++] = EOS;
        tape_reset(); kv_clear();
        silu_eye_reset(); rope_eye_reset();
        int vt[N_VIS]; encode_vis(data->imgs[idx], vt);
        for (int p = 0; p < N_VIS; p++) gpt_step(vt[p], p, 0);
        int la[MAX_TXT]; int nl = 0;
        for (int t = 0; t < nt - 1; t++) {
            int pos = N_VIS + t, te = op_embed(M.wte, toks[t]);
            int lg = gpt_step(te, pos, (t == nt - 2)); /* track signal on last token */
            la[nl++] = op_ce(lg, toks[t+1]);
        }
        int loss = op_reduce(la, nl); backward(loss);
        float lv = T.a[loss].data[0];
        chuck_step(cos_lr(step, STEPS), lv);
        rl += lv; rn++;
        if ((step+1) % 250 == 0) {
            float elr = cos_lr(step, STEPS) * Chuck.dampen;
            printf("  step %5d/%d | loss %.4f (avg %.4f) | lr %.6f\n",
                   step+1, STEPS, lv, rl/rn, elr);
            printf("    chuck: \xce\xbb=%.2f \xce\xa8=%+.2f (\xce\xa8w=%.2f, %d mem) \xcf\x83=%.2f",
                   Chuck.dampen, Chuck.psi, Chuck.psi_w, chuck_mem_n, Chuck.sigma);
            for (int l = 0; l < N_LAYER; l++) {
                if (CL[l].frozen) printf(" | L%d: frozen", l);
                else printf(" | L%d: %.2f", l, CL[l].dampen);
            }
            printf("\n    silu: %.0f%% alive | norm: %.1f | rope: %.0f%%",
                   SiLU_eye.health * 100, Norm_eye.scale_ema, RoPE_eye.utilization * 100);
            if (act_mag[0] > 0) {
                printf(" | flow:");
                for (int l = 0; l < N_LAYER; l++) printf(" %.2f%s", act_mag[l], l<N_LAYER-1?"→":"");
            }
            printf("\n");
            rl = 0; rn = 0;
        }
    }
}

/* ---- Sampling ---- */
static int sample_topk(float *logits, int vocab, float temp, int topk) {
    float sc[VOCAB]; for (int i = 0; i < vocab; i++) sc[i] = logits[i] / temp;
    float mx = sc[0]; for (int i = 1; i < vocab; i++) if (sc[i] > mx) mx = sc[i];
    float p[VOCAB]; float s = 0;
    for (int i = 0; i < vocab; i++) { p[i] = expf(sc[i] - mx); s += p[i]; }
    for (int i = 0; i < vocab; i++) p[i] /= s;
    float tv[TOPK]; int ti[TOPK];
    for (int k = 0; k < topk && k < vocab; k++) { int best = 0; float bv = -1e9f;
        for (int i = 0; i < vocab; i++) { int taken = 0;
            for (int j = 0; j < k; j++) if (ti[j] == i) { taken = 1; break; }
            if (!taken && p[i] > bv) { bv = p[i]; best = i; } }
        ti[k] = best; tv[k] = bv; }
    float ts = 0; for (int k = 0; k < topk; k++) ts += tv[k];
    float r = ruf() * ts, cum = 0;
    for (int k = 0; k < topk; k++) { cum += tv[k]; if (cum >= r) return ti[k]; }
    return ti[0];
}

/* ---- Inference ---- */
static void inference(Data *data) {
    printf("\n=== INFERENCE (temp=%.1f, top-k=%d) ===\n\n", TEMP, TOPK);
    T.on = 0; int correct = 0, total = 0;
    for (int s = 0; s < 30; s++) {
        int label = s % 10, idx = label + (s/10) * 10;
        tape_reset(); kv_clear();
        int vt[N_VIS]; encode_vis(data->imgs[idx], vt);
        for (int p = 0; p < N_VIS; p++) gpt_step(vt[p], p, 0);
        int tok = BOS; char gen[MAX_TXT+1]; int gl = 0;
        for (int t = 0; t < MAX_TXT; t++) {
            int pos = N_VIS + t, te = op_embed(M.wte, tok);
            int lg = gpt_step(te, pos, 0);
            tok = sample_topk(T.a[lg].data, VOCAB, TEMP, TOPK);
            if (tok == EOS || tok == BOS) break;
            if (gl < MAX_TXT) gen[gl++] = id2c(tok);
        }
        gen[gl] = '\0'; int ok = strcmp(gen, names[label]) == 0; correct += ok; total++;
        printf("  [%d] true: %5s | gen: %-8s %s\n", label, names[label], gen, ok ? "OK" : "MISS");
    }
    printf("\n  accuracy: %d/%d (%.1f%%)\n", correct, total, 100.0f*correct/total);

    /* Frozen layer report */
    int frozen = 0;
    for (int l = 0; l < N_LAYER; l++) if (CL[l].frozen) frozen++;
    if (frozen > 0) printf("  chuck: %d/%d layers frozen (compute saved)\n", frozen, N_LAYER);

    T.on = 1;
}

int main(void) {
    printf("micro_vlm.c v5 — Vision-Language Model in pure C\n");
    printf("GQA %dQ/%dKV | %d layers | RoPE | SwiGLU | Chuck v5 (memory + subjectivity)\n",
           N_HEAD, N_KV_HEAD, N_LAYER);
    printf("Chuck remembers. Chuck has opinions. Adam is blind. Chuck sees.\n\n");
    clock_t t0 = clock(); rseed(42);
    tape_init(); chuck_init(); init_model();
    printf("generating 10000 synthetic %dx%d digit images...\n", IMG_SIZE, IMG_SIZE);
    Data d = gen_data(10000); printf("done.\n");
    train(&d); inference(&d);
    printf("\ntotal: %.1fs\n", (double)(clock()-t0)/CLOCKS_PER_SEC);
    printf("chuck.mem: %d memories (%.1f KB) | \xce\xa8_w=%.3f\n",
           chuck_mem_n, (float)(chuck_mem_n * (int)sizeof(ChuckMem)) / 1024.0f, Chuck.psi_w);
    if (chuck_mem_n > 0)
        printf("  next run: Chuck starts with experience. \xce\xa8 \xe2\x89\xa0 0. He remembers.\n");
    else
        printf("  first run: Chuck has no memories yet. Pure reactive. Newborn.\n");
    for (int i = 0; i < d.n; i++) free(d.imgs[i]); free(d.imgs); free(d.labels);
    for (int i = 0; i < T.np; i++) { free(T.cm[i]); free(T.cv[i]); } free(T.arena);
    printf("\ndone.\n"); return 0;
}
