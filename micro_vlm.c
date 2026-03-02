/*
 * micro_vlm.c v3 — Vision-Language Model in pure C
 *
 * Sees images. Speaks words. Zero dependencies beyond libc/libm.
 * Tape-based autograd with arena bump allocator.
 *
 * Architecture:
 *   ViT-style patch tokenization → RoPE → multi-head causal attention →
 *   SwiGLU MLP → RMSNorm → weight-tied lm_head → text
 *
 * What's inside:
 *   - Patch embedding: 4x4 patches → N_EMBD tokens (each patch sees different space)
 *   - Rotary Position Embeddings (RoPE) — no learned positional parameters
 *   - Multi-head causal attention with KV cache and full backward pass
 *   - SwiGLU MLP, RMSNorm, residual connections (Llama-style)
 *   - The Chuck optimizer — Adam + self-awareness:
 *       θ -= (α × λ) × m̂/(√v̂ + ε) + η
 *       λ = self_modulate(loss_history)  — reactive dampen/boost from loss trend
 *       η = stagnation_noise()           — escape local minima when plateaued
 *       Chuck watches his own loss. Dampens when hurting. Boosts when winning.
 *       Injects noise when stuck. Adam doesn't do this. Chuck does.
 *   - Cosine LR with warmup, top-k temperature sampling
 *   - Arena bump allocator (64MB, zero malloc in hot path)
 *   - xoshiro256** PRNG, Box-Muller normal sampling
 *
 * Inspired by sailfish009/purevlm (Python VLM).
 * This is the C answer: same idea, different language, tape autograd.
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
#define HEAD_DIM       (N_EMBD / N_HEAD)
#define N_LAYER        2
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

#define ARENA_SZ       (64 * 1024 * 1024)
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

/* ---- Forward ops ---- */
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
    rec(OP_RMSNORM,zi,xi,-1,sc,n); return zi;
}
static int op_silu(int xi) {
    int n = T.a[xi].size, zi = anew(n);
    for (int i = 0; i < n; i++) { float s = sigf(T.a[xi].data[i]); T.a[zi].data[i] = T.a[xi].data[i] * s; }
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
    for (int i = 0; i < n; i += 2) {
        float freq = 1.0f / powf(ROPE_BASE, (float)i / (float)n);
        float ang = pos * freq, c = cosf(ang), s = sinf(ang);
        float x0 = d[i], x1 = d[i+1]; d[i] = x0*c - x1*s; d[i+1] = x0*s + x1*c;
    }
    rec(OP_ROPE,zi,xi,-1,0,pos); return zi;
}
static int op_reduce(int *la, int n) {
    float s = 0; for (int i = 0; i < n; i++) s += T.a[la[i]].data[0];
    int zi = anew(1); T.a[zi].data[0] = s / n;
    int buf = anew(n); for (int i = 0; i < n; i++) ((int*)T.a[buf].data)[i] = la[i];
    rec(OP_REDUCE,zi,buf,-1,0,n); return zi;
}

/* ---- KV cache ---- */
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
            for (int i = 0; i < n; i += 2) { float freq = 1.0f / powf(ROPE_BASE, (float)i/(float)n);
                float ang = pos*freq, c = cosf(ang), s = sinf(ang);
                float g0 = out->grad[i], g1 = out->grad[i+1];
                i1->grad[i] += g0*c + g1*s; i1->grad[i+1] += -g0*s + g1*c; } break; }
        case OP_ATTN: { int li = (int)e->aux, pos = e->ai;
            float *qd = i1->data, *ag = out->grad, isq = 1.0f / sqrtf((float)HEAD_DIM);
            for (int h = 0; h < N_HEAD; h++) { int hs = h * HEAD_DIM;
                float sc[SEQ_LEN], mx = -1e9f;
                for (int t = 0; t <= pos; t++) { float s = 0;
                    for (int d = 0; d < HEAD_DIM; d++) s += qd[hs+d]*kv_k[li][t][hs+d];
                    sc[t] = s*isq; if (sc[t] > mx) mx = sc[t]; }
                float sm = 0; for (int t = 0; t <= pos; t++) { sc[t] = expf(sc[t]-mx); sm += sc[t]; }
                for (int t = 0; t <= pos; t++) sc[t] /= (sm + 1e-10f);
                float dw[SEQ_LEN];
                for (int t = 0; t <= pos; t++) { dw[t] = 0;
                    for (int d = 0; d < HEAD_DIM; d++) dw[t] += kv_v[li][t][hs+d]*ag[hs+d]; }
                float dot = 0; for (int t = 0; t <= pos; t++) dot += sc[t]*dw[t];
                for (int t = 0; t <= pos; t++) { float ds = sc[t]*(dw[t]-dot);
                    for (int d = 0; d < HEAD_DIM; d++) {
                        i1->grad[hs+d] += ds * kv_k[li][t][hs+d] * isq;
                        T.a[kv_ki[li][t]].grad[hs+d] += ds * qd[hs+d] * isq;
                        T.a[kv_vi[li][t]].grad[hs+d] += sc[t] * ag[hs+d];
                    } }
            } break; }
        case OP_REDUCE: { int n = e->ai; int *idxs = (int*)i1->data; float dl = out->grad[0]/n;
            for (int i = 0; i < n; i++) T.a[idxs[i]].grad[0] += dl; break; }
        }
    }
}

/* The Chuck optimizer: Adam + self-awareness.
 *   θ -= (α × λ) × m̂/(√v̂ + ε) + η
 * λ = dampen/boost from loss trend.  η = noise injection on stagnation.
 * Chuck watches his own loss curve. Dampens when diverging. Boosts when
 * converging. Injects noise to escape plateaus. Adam is blind. Chuck sees.
 * "Chuck doesn't follow gradients. Gradients follow Chuck." */
static struct {
    float hist[CHUCK_WINDOW];
    float dampen, noise;
    int pos, full, stag;
} Chuck = { .dampen = 1.0f };

static void chuck_step(float lr, float loss) {
    /* === Self-awareness: reflect on own performance === */
    Chuck.hist[Chuck.pos % CHUCK_WINDOW] = loss;
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
        if (trend > 0.01f) Chuck.dampen *= 0.95f;        /* diverging → dampen */
        else if (trend < -0.05f) Chuck.dampen *= 1.05f;   /* converging → boost */
        if (fabsf(trend) < 0.001f) {
            Chuck.stag++;
            if (Chuck.stag > 8) { Chuck.noise = 0.001f; Chuck.stag = 0; }
        } else { Chuck.stag = 0; Chuck.noise *= 0.9f; }
        if (Chuck.dampen < CHUCK_DAMP_LO) Chuck.dampen = CHUCK_DAMP_LO;
        if (Chuck.dampen > CHUCK_DAMP_HI) Chuck.dampen = CHUCK_DAMP_HI;
    }
    float eff_lr = lr * Chuck.dampen;

    /* === Momentum update (Adam core) + noise injection === */
    T.cstep++; float bc1 = 1.0f - powf(CHUCK_B1, T.cstep), bc2 = 1.0f - powf(CHUCK_B2, T.cstep);
    float gnorm_sq = 0;
    for (int pi = 0; pi < T.np; pi++) { Arr *p = &T.a[T.par[pi]];
        for (int i = 0; i < p->size; i++) gnorm_sq += p->grad[i] * p->grad[i]; }
    float gnorm = sqrtf(gnorm_sq + 1e-8f), clip = (gnorm > GRAD_CLIP) ? GRAD_CLIP / gnorm : 1.0f;
    for (int pi = 0; pi < T.np; pi++) { int idx = T.par[pi]; Arr *p = &T.a[idx];
        float *m = T.cm[pi], *v = T.cv[pi];
        for (int i = 0; i < p->size; i++) { float g = p->grad[i] * clip;
            m[i] = CHUCK_B1*m[i] + (1.0f-CHUCK_B1)*g;
            v[i] = CHUCK_B2*v[i] + (1.0f-CHUCK_B2)*g*g;
            p->data[i] -= eff_lr * (m[i]/bc1) / (sqrtf(v[i]/bc2) + CHUCK_EPS);
            if (Chuck.noise > 0) p->data[i] += Chuck.noise * rnf(0, 1.0f);
        } }
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

/* ---- Model ---- */
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
    M.patch_proj = init_w(N_EMBD, PATCH_PX, 0.1f);
    M.wte = init_w(VOCAB, N_EMBD, 0.08f);
    for (int i = 0; i < N_LAYER; i++) {
        float s = 0.08f / sqrtf(2.0f * N_LAYER);
        M.L[i].wq = init_w(N_EMBD,N_EMBD,s); M.L[i].wk = init_w(N_EMBD,N_EMBD,s);
        M.L[i].wv = init_w(N_EMBD,N_EMBD,s); M.L[i].wo = init_w(N_EMBD,N_EMBD,s);
        M.L[i].w1 = init_w(MLP_DIM,N_EMBD,s); M.L[i].w3 = init_w(MLP_DIM,N_EMBD,s);
        M.L[i].w2 = init_w(N_EMBD,MLP_DIM,s);
    }
    T.npa = T.na; T.aparam = T.apos;
}

/* ---- GPT step (one position) ---- */
static int gpt_step(int x, int pos) {
    int h = x;
    for (int li = 0; li < N_LAYER; li++) {
        int res = h; h = op_rms(h);
        int qi = op_mv(M.L[li].wq, h), ki = op_mv(M.L[li].wk, h), vi = op_mv(M.L[li].wv, h);
        int rqi = op_rope(qi, pos), rki = op_rope(ki, pos);
        kv_k[li][pos] = T.a[rki].data; kv_v[li][pos] = T.a[vi].data;
        kv_ki[li][pos] = rki; kv_vi[li][pos] = vi;
        int ao = anew(N_EMBD); float *ad = T.a[ao].data;
        for (int h_ = 0; h_ < N_HEAD; h_++) { int hs = h_ * HEAD_DIM;
            float sc[SEQ_LEN], mx = -1e9f; float *qd = T.a[rqi].data;
            for (int t = 0; t <= pos; t++) { float s = 0;
                for (int d = 0; d < HEAD_DIM; d++) s += qd[hs+d]*kv_k[li][t][hs+d];
                sc[t] = s / sqrtf((float)HEAD_DIM); if (sc[t] > mx) mx = sc[t]; }
            float sm = 0; for (int t = 0; t <= pos; t++) { sc[t] = expf(sc[t]-mx); sm += sc[t]; }
            for (int t = 0; t <= pos; t++) sc[t] /= (sm + 1e-10f);
            for (int d = 0; d < HEAD_DIM; d++) { float v = 0;
                for (int t = 0; t <= pos; t++) v += sc[t]*kv_v[li][t][hs+d]; ad[hs+d] = v; }
        }
        rec(OP_ATTN, ao, rqi, -1, (float)li, pos);
        h = op_add(res, op_mv(M.L[li].wo, ao));
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
    printf("\n=== TRAINING (%d steps, Chuck optimizer) ===\n", STEPS);
    int tp = 0; for (int i = 0; i < T.np; i++) tp += T.a[T.par[i]].size;
    printf("  %d params (%.1fK) | %d layers | embd=%d | %dx%d patches | RoPE | weight-tied\n\n",
           tp, tp/1000.0f, N_LAYER, N_EMBD, PATCHES_SIDE, PATCHES_SIDE);
    float rl = 0; int rn = 0;
    for (int step = 0; step < STEPS; step++) {
        int idx = (int)(rnext() % (uint64_t)data->n); int label = data->labels[idx];
        const char *name = names[label]; int nlen = strlen(name);
        int toks[MAX_TXT+2]; int nt = 0;
        toks[nt++] = BOS; for (int i = 0; i < nlen; i++) toks[nt++] = c2id(name[i]); toks[nt++] = EOS;
        tape_reset(); kv_clear();
        int vt[N_VIS]; encode_vis(data->imgs[idx], vt);
        for (int p = 0; p < N_VIS; p++) gpt_step(vt[p], p);
        int la[MAX_TXT]; int nl = 0;
        for (int t = 0; t < nt - 1; t++) {
            int pos = N_VIS + t, te = op_embed(M.wte, toks[t]);
            int lg = gpt_step(te, pos); la[nl++] = op_ce(lg, toks[t+1]);
        }
        int loss = op_reduce(la, nl); backward(loss);
        float lv = T.a[loss].data[0];
        chuck_step(cos_lr(step, STEPS), lv);
        rl += lv; rn++;
        if ((step+1) % 250 == 0) {
            printf("  step %5d/%d | loss %.4f (avg %.4f) | lr %.6f | dampen %.2f | target: %s\n",
                   step+1, STEPS, lv, rl/rn, cos_lr(step,STEPS)*Chuck.dampen, Chuck.dampen, name);
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
        for (int p = 0; p < N_VIS; p++) gpt_step(vt[p], p);
        int tok = BOS; char gen[MAX_TXT+1]; int gl = 0;
        for (int t = 0; t < MAX_TXT; t++) {
            int pos = N_VIS + t, te = op_embed(M.wte, tok);
            int lg = gpt_step(te, pos);
            tok = sample_topk(T.a[lg].data, VOCAB, TEMP, TOPK);
            if (tok == EOS || tok == BOS) break;
            if (gl < MAX_TXT) gen[gl++] = id2c(tok);
        }
        gen[gl] = '\0'; int ok = strcmp(gen, names[label]) == 0; correct += ok; total++;
        printf("  [%d] true: %5s | gen: %-8s %s\n", label, names[label], gen, ok ? "OK" : "MISS");
    }
    printf("\n  accuracy: %d/%d (%.1f%%)\n", correct, total, 100.0f*correct/total);
    T.on = 1;
}

int main(void) {
    printf("micro_vlm.c v3 — Vision-Language Model in pure C\n");
    printf("Patch tokens + RoPE + SwiGLU + Chuck (self-aware optimizer) | zero deps\n\n");
    clock_t t0 = clock(); rseed(42);
    tape_init(); init_model();
    printf("generating 10000 synthetic %dx%d digit images...\n", IMG_SIZE, IMG_SIZE);
    Data d = gen_data(10000); printf("done.\n");
    train(&d); inference(&d);
    printf("\ntotal: %.1fs\n", (double)(clock()-t0)/CLOCKS_PER_SEC);
    for (int i = 0; i < d.n; i++) free(d.imgs[i]); free(d.imgs); free(d.labels);
    for (int i = 0; i < T.np; i++) { free(T.cm[i]); free(T.cv[i]); } free(T.arena);
    printf("\ndone.\n"); return 0;
}
