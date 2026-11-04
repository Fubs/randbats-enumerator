#include "six.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

constexpr int kThreads = 128;
#ifndef TG_SIX_MINB
#define TG_SIX_MINB 4
#endif
#ifndef TG_SIX_MINB_BIG
#define TG_SIX_MINB_BIG 3
#endif
constexpr int kMaxVary = 16;
constexpr int kMaxSmem = 48 * 1024;

struct df64 {
    float hi, lo;
};

__device__ __forceinline__ df64 quick_two_sum(float a, float b) {
    float s = __fadd_rn(a, b);
    return {s, __fsub_rn(b, __fsub_rn(s, a))};
}

__device__ __forceinline__ df64 two_sum(float a, float b) {
    float s = __fadd_rn(a, b);
    float v = __fsub_rn(s, a);
    return {s, __fadd_rn(__fsub_rn(a, __fsub_rn(s, v)), __fsub_rn(b, v))};
}

__device__ __forceinline__ df64 operator+(df64 a, df64 b) {
    df64 s = two_sum(a.hi, b.hi);
    df64 t = two_sum(a.lo, b.lo);
    s = quick_two_sum(s.hi, __fadd_rn(s.lo, t.hi));
    return quick_two_sum(s.hi, __fadd_rn(s.lo, t.lo));
}

__device__ __forceinline__ df64 operator-(df64 a) { return {-a.hi, -a.lo}; }

__device__ __forceinline__ df64 operator*(df64 a, df64 b) {
    float p = __fmul_rn(a.hi, b.hi);
    float e = fmaf(a.hi, b.hi, -p);
    e = fmaf(a.hi, b.lo, e);
    e = fmaf(a.lo, b.hi, e);
    return quick_two_sum(p, e);
}

__device__ __forceinline__ df64 times(df64 a, int k) { return a * df64{static_cast<float>(k), 0.f}; }

__device__ __forceinline__ df64 div(df64 a, df64 b) {
    const float r = __frcp_rn(b.hi);
    const float q1 = __fmul_rn(a.hi, r);
    df64 rem = a + -(df64{q1, 0.f} * b);
    const float q2 = __fmul_rn(rem.hi, r);
    rem = rem + -(df64{q2, 0.f} * b);
    const float q3 = __fmul_rn(rem.hi, r);
    return quick_two_sum(q1, q2) + df64{q3, 0.f};
}

__device__ __forceinline__ df64 word(uint64_t w) {
    return {__uint_as_float(static_cast<unsigned>(w)), __uint_as_float(static_cast<unsigned>(w >> 32))};
}

__device__ __forceinline__ df64 ld(const uint64_t* w, int64_t i) { return word(__ldg(w + i)); }

__device__ __forceinline__ df64 sm(const float2* s, int i) {
    const float2 v = s[i];
    return {v.x, v.y};
}

__device__ __forceinline__ float2 f2(df64 v) { return make_float2(v.hi, v.lo); }

__device__ __forceinline__ df64 pw(df64 e, int k) {
    if (k == 0) {
        return {1.f, 0.f};
    }
    df64 v = e;
    for (int i = 1; i < k; i++) {
        v = v * e;
    }
    return v;
}

__device__ __forceinline__ df64 role(int r, df64 bb, df64 ep) {
    return r == 0 ? df64{0.f, 0.f} : times(bb * pw(ep, r - 1), r);
}

struct Lazy {
    float hi = 0.f, lo = 0.f;
    __device__ __forceinline__ void add(df64 x) {
        float s = __fadd_rn(hi, x.hi);
        float v = __fsub_rn(s, hi);
        float t = __fadd_rn(__fsub_rn(hi, __fsub_rn(s, v)), __fsub_rn(x.hi, v));
        hi = s;
        lo = __fadd_rn(lo, __fadd_rn(t, x.lo));
    }
    __device__ __forceinline__ void norm() {
        df64 s = quick_two_sum(hi, lo);
        hi = s.hi;
        lo = s.lo;
    }
    __device__ __forceinline__ df64 get() const { return quick_two_sum(hi, lo); }
};

struct Dev {
    const TgSixPick* picks;
    const int4* blocks;
    const uint64_t* words;
    const uint64_t* mus;
    const uint16_t* shorts;
    const TgSixBase* bases;
    double* out;
    int64_t nwords, nmus, nshorts, nbases, nout;
};

__device__ __forceinline__ double fin(df64 v, double scale) { return (static_cast<double>(v.hi) + static_cast<double>(v.lo)) * scale; }

#ifdef TG_SIX_BOUNDS
#define TG_CHK(c)                                                                                                   \
    do {                                                                                                            \
        if (!(c)) {                                                                                                 \
            printf("tg_six bound failed: %s; pick %d np %d nvq %d nslot %d npat %d nkeys %d nrow %d class %d z0 %d " \
                   "zl %d nz %d\n",                                                                                 \
                   #c, bk.x, p.np, p.nvq, p.nslot, p.npat, p.nkeys, p.nrow, p.class_rows, z0, zl, p.nz);            \
            __trap();                                                                                               \
        }                                                                                                           \
    } while (0)
#else
#define TG_CHK(c) \
    do {          \
    } while (0)
#endif

constexpr int kRowTile = 16;
constexpr int kMinRowTile = 4;

constexpr int kMaxParts = 4;

__host__ __device__ inline int row_parts(const TgSixPick& p, int zl) {
    const int n = kThreads / (p.nkeys * zl);
    return n < 1 ? 1 : n > kMaxParts ? kMaxParts : n;
}

__host__ __device__ inline int smem_bytes(const TgSixPick& p, int zl, int tile) {
    const int nvs = p.nvq + p.ncv;
    const int rows = p.class_rows ? 0 : tile * (nvs * zl * 2 + p.nvq);
    const int pairs = p.nslot * zl * 4 + p.np * 2 + zl + row_parts(p, zl) * p.nkeys * zl * (1 + p.nvq) + rows;
    const int cols = p.class_rows ? 0 : nvs * zl;
    return 8 * pairs + 4 * (cols + p.np) + 2 * (p.npat * p.np + p.nkeys * p.nvq + p.npat);
}

template <int NP, int MV>
constexpr int kMinBlocks = MV > 8 ? 2 : NP >= 32 ? TG_SIX_MINB_BIG : TG_SIX_MINB;

template <int NP, int MV, bool CLASS>
__global__ void __launch_bounds__(kThreads, kMinBlocks<NP, MV>) nodes(Dev d, const int32_t* ids) {
    const int4 bk = d.blocks[ids[blockIdx.x]];
    const TgSixPick p = d.picks[bk.x];
    const int z0 = bk.y, zl = bk.z, tile = bk.w;
    const int np = p.np, nvq = p.nvq, nrow = p.nrow, nslot = p.nslot, npat = p.npat, nkeys = p.nkeys, ny = p.ny,
              nxy = p.nx * p.ny;
    const uint64_t* W = d.words;
    const int nvs = nvq + p.ncv, ns1 = 1 + nvq, nkz = nkeys * zl, parts = row_parts(p, zl), ncol = nvs * zl;
    extern __shared__ float2 shm[];
    float2* tab = shm;
    float2* r5 = tab + nslot * zl * 4;
    float2* zs = r5 + np * 2;
    float2* sums = zs + zl;
    float2* ft = sums + parts * nkz * ns1;
    float2* jt = ft + (CLASS ? 0 : tile * ncol * 2);
    uint32_t* cols = reinterpret_cast<uint32_t*>(jt + (CLASS ? 0 : tile * nvq));
    int* rmax = reinterpret_cast<int*>(cols + (CLASS ? 0 : ncol));
    uint16_t* ps = reinterpret_cast<uint16_t*>(rmax + np);
    uint16_t* ks = ps + npat * np;
    uint16_t* pk = ks + nkeys * nvq;
    const int tid = static_cast<int>(threadIdx.x);
#ifdef TG_SIX_BOUNDS
    {
        unsigned dyn;
        asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn));
        const int64_t nxy64 = nxy;
        TG_CHK(np <= NP && nvq <= MV && nvq <= np && npat * zl <= kThreads && nkeys <= npat && nslot >= np);
        TG_CHK(z0 >= 0 && z0 + zl <= p.nz && p.nz <= p.nzp);
        TG_CHK(tile >= kMinRowTile && tile <= kRowTile);
        TG_CHK(smem_bytes(p, zl, tile) <= static_cast<int>(dyn));
        TG_CHK(reinterpret_cast<char*>(pk + npat) <= reinterpret_cast<char*>(shm) + dyn);
        TG_CHK(p.pslot >= 0 && p.pslot + npat * np <= d.nshorts);
        TG_CHK(p.keys >= 0 && p.keys + nkeys * nvq + npat <= d.nshorts);
        TG_CHK(p.cq >= 0 && p.cq + (nslot - np) <= d.nshorts);
        TG_CHK(p.zv >= 0 && p.zv + p.nz <= d.nwords);
        TG_CHK(p.bases >= 0 && p.bases + np <= d.nbases);
        TG_CHK(p.cb >= 0 && p.cb + 2 * static_cast<int64_t>(p.nx) * np + ny <= d.nwords);
        TG_CHK(p.wn >= 0 && p.wn + nxy64 <= d.nwords);
        TG_CHK(p.wm >= 0 && p.wm + static_cast<int64_t>(p.nx) * nrow <= d.nwords);
        TG_CHK(p.q5 >= 0 && p.q5 + 2 * nslot <= d.nwords);
        TG_CHK(p.mu >= 0 && p.mu + nrow <= d.nmus);
        TG_CHK(CLASS || (p.lin >= 0 && p.lin + 2 * static_cast<int64_t>(nrow) * nvq <= d.nwords));
        TG_CHK(p.out >= 0 && p.out + static_cast<int64_t>(npat) * p.nzp * (1 + np) <= d.nout);
        for (int i = tid; i < npat * np; i += kThreads) {
            const int s = d.shorts[p.pslot + i];
            TG_CHK((s & TG_SIX_SLOT) < nslot && (!(s & TG_SIX_J5) || (s & TG_SIX_SLOT) < np));
        }
        for (int i = tid; i < nkeys * nvq; i += kThreads) {
            const int s = d.shorts[p.keys + i];
            const int ss = s & TG_SIX_SLOT;
            TG_CHK(ss < nslot && (ss < nvq || (ss >= np && ss < np + p.ncv)));
        }
        for (int i = tid; i < npat; i += kThreads) {
            TG_CHK(d.shorts[p.keys + nkeys * nvq + i] < nkeys);
        }
        for (int c = tid; c < nslot - np; c += kThreads) {
            const int q = d.shorts[p.cq + c];
            TG_CHK(q < np && (c >= p.ncv || q < nvq));
        }
        for (int q = tid; q < np; q += kThreads) {
            TG_CHK(d.bases[p.bases + q].cpu >= 0 && d.bases[p.bases + q].cpu < np && d.bases[p.bases + q].rmax >= 0);
        }
        for (int r = tid; r < nrow; r += kThreads) {
            const uint64_t m = d.mus[p.mu + r];
            TG_CHK(nvq == 16 || (m >> (4 * nvq)) == 0);
        }
    }
#endif
    for (int i = tid; i < npat * np; i += kThreads) {
        ps[i] = d.shorts[p.pslot + i];
    }
    for (int i = tid; i < nkeys * nvq + npat; i += kThreads) {
        ks[i] = d.shorts[p.keys + i];
    }
    for (int i = tid; i < zl; i += kThreads) {
        zs[i] = f2(ld(W, p.zv + z0 + i));
    }
    for (int q = tid; q < np; q += kThreads) {
        rmax[q] = d.bases[p.bases + q].rmax;
    }
    if constexpr (!CLASS) {
        for (int j = tid; j < ncol; j += kThreads) {
            const int vsi = j / zl, l = j % zl;
            const int s = vsi < nvq ? vsi : np + vsi - nvq;
            const int v = vsi < nvq ? vsi : d.shorts[p.cq + vsi - nvq];
            cols[j] = static_cast<uint32_t>(s) | static_cast<uint32_t>(v) << 12 | static_cast<uint32_t>(l) << 16;
        }
    }

    const bool active = tid < npat * zl;
    const int pi = active ? tid / zl : 0, lane = active ? tid % zl : 0;
    const uint16_t* mine = ps + pi * np;
    const bool ract = tid < nkz * parts;
    const int part = ract ? tid / nkz : 0, rt = ract ? tid % nkz : 0, rl = rt % zl;

    const int ts0 = tid / zl, tl0 = tid % zl, tds = kThreads / zl, tdl = kThreads % zl;
    const int fr0 = ncol ? tid / ncol : 0, fc0 = ncol ? tid % ncol : 0, fdr = ncol ? kThreads / ncol : 0,
              fdc = ncol ? kThreads % ncol : 0;
    __syncthreads();

    uint32_t vs[MV];
    int vx[MV];
#pragma unroll
    for (int v = 0; v < MV; v++) {
        vs[v] = v < nvq ? ks[(rt / zl) * nvq + v] : 0;
        const int ss = static_cast<int>(vs[v] & TG_SIX_SLOT);
        vx[v] = (ss < np ? ss : nvq + ss - np) * zl + rl;
    }

    Lazy acc[NP + 1];
    for (int n = 0; n < nxy; n++) {
        const int ix = n / ny;

        const int64_t cb = p.cb + 2 * static_cast<int64_t>(ix) * np;
        const df64 yv = ld(W, p.cb + 2 * static_cast<int64_t>(p.nx) * np + (n - ix * ny));
        __syncthreads();
        for (int s = ts0, l = tl0; s < nslot;) {
            const int q = s < np ? s : d.shorts[p.cq + s - np];
            const df64 cc = ld(W, cb + 2 * q), bb = ld(W, cb + 2 * q + 1) * yv;
            const df64 q5 = ld(W, p.q5 + 2 * s), om = ld(W, p.q5 + 2 * s + 1);
            const df64 bz = (bb * q5) * sm(zs, l);
            const df64 e = (cc + bb * om) + bz;
            const df64 e2 = e * e;
            float2* t = tab + 4 * (s * zl + l);
            t[0] = f2(e);
            t[1] = f2(e2);
            t[2] = f2(e2 * e);
            t[3] = f2(div(bz, e));
            s += tds;
            l += tdl;
            if (l >= zl) {
                l -= zl;
                s++;
            }
        }
        for (int q = tid; q < np; q += kThreads) {
            const df64 cc = ld(W, cb + 2 * q), bb = ld(W, cb + 2 * q + 1) * yv;
            r5[2 * q] = f2(bb);
            r5[2 * q + 1] = f2(cc + bb);
        }
        __syncthreads();

        const auto at = [&](uint32_t s, int l, int k) { return sm(tab, (static_cast<int>(s & TG_SIX_SLOT) * zl + l) * 4 + k); };

        Lazy s0;
        Lazy sv[MV];
        const uint64_t* wm = W + p.wm + static_cast<int64_t>(ix) * nrow;
        if constexpr (CLASS) {
            for (int r = part; ract && r < nrow; r += parts) {
                const df64 w = word(__ldg(wm + r));
                if (w.hi == 0.f) {
                    continue;
                }

                const uint64_t mus = __ldg(d.mus + p.mu + r);
                df64 pr = w;
#pragma unroll
                for (int v = 0; v < MV; v++) {
                    if (v < nvq) {
                        const int m = static_cast<int>((mus >> (4 * v)) & 15);
                        const uint32_t s = vs[v];
                        if (s & TG_SIX_J5) {
                            pr = pr * role(m, sm(r5, 2 * v), sm(r5, 2 * v + 1));
                        } else if (m > 0) {
                            pr = pr * at(s, rl, m - 1);
                        }
                    }
                }
                s0.add(pr);
#pragma unroll
                for (int v = 0; v < MV; v++) {
                    if (v < nvq && (vs[v] & TG_SIX_LIVE)) {
                        const int m = static_cast<int>((mus >> (4 * v)) & 15);
                        if (m == 1) {
                            sv[v].add(pr);
                        } else if (m == 2) {
                            sv[v].add(df64{2.f * pr.hi, 2.f * pr.lo});
                        } else if (m == 3) {
                            sv[v].add(times(pr, 3));
                        }
                    }
                }
            }
        } else {

            for (int r0 = 0; r0 < nrow; r0 += tile) {
                const int tr = min(tile, nrow - r0);
                if (r0 > 0) {
                    __syncthreads();
                }
                for (int rr = fr0, j = fc0; ncol && rr < tr;) {
                    const uint32_t c = cols[j];
                    const int s = static_cast<int>(c & 0xfff), v = static_cast<int>((c >> 12) & 15), l = static_cast<int>(c >> 16);
                    const int r = r0 + rr;
                    const int m = static_cast<int>((__ldg(d.mus + p.mu + r) >> (4 * v)) & 15);
                    const int64_t li = p.lin + 2 * (static_cast<int64_t>(r) * nvq + v);
                    const df64 al = ld(W, li), be = ld(W, li + 1);
                    const df64 e = sm(tab, (s * zl + l) * 4);
                    const df64 lv = al + be * e;
                    float2* f = ft + 2 * (rr * ncol + j);
                    f[0] = f2(pw(lv, m));
                    f[1] = f2(m == 0 ? df64{0.f, 0.f} : div(times(be * e, m), lv));
                    rr += fdr;
                    j += fdc;
                    if (j >= ncol) {
                        j -= ncol;
                        rr++;
                    }
                }
                for (int i = tid; i < tr * nvq; i += kThreads) {
                    const int rr = i / nvq, v = i % nvq, r = r0 + rr;
                    const int m = static_cast<int>((__ldg(d.mus + p.mu + r) >> (4 * v)) & 15);
                    const int64_t li = p.lin + 2 * (static_cast<int64_t>(r) * nvq + v);
                    const df64 al = ld(W, li), be = ld(W, li + 1);
                    jt[i] = f2(m == 0 ? df64{0.f, 0.f} : times((be * sm(r5, 2 * v)) * pw(al + be * sm(r5, 2 * v + 1), m - 1), m));
                }
                __syncthreads();
                for (int rr = part; ract && rr < tr; rr += parts) {
                    const df64 w = word(__ldg(wm + r0 + rr));
                    if (w.hi == 0.f) {
                        continue;
                    }
                    df64 pr = w;
                    df64 g[MV];
#pragma unroll
                    for (int v = 0; v < MV; v++) {
                        if (v < nvq) {
                            if (vs[v] & TG_SIX_J5) {
                                pr = pr * sm(jt, rr * nvq + v);
                                g[v] = df64{0.f, 0.f};
                            } else {
                                const float2* f = ft + 2 * (rr * ncol + vx[v]);
                                pr = pr * sm(f, 0);
                                g[v] = sm(f, 1);
                            }
                        }
                    }
                    s0.add(pr);
#pragma unroll
                    for (int v = 0; v < MV; v++) {
                        if (v < nvq && (vs[v] & TG_SIX_LIVE)) {
                            sv[v].add(pr * g[v]);
                        }
                    }
                }
            }
        }
        if (ract) {
            float2* out = sums + (part * nkz + rt) * ns1;
            out[0] = f2(s0.get());
#pragma unroll
            for (int v = 0; v < MV; v++) {
                if (v < nvq) {
                    out[1 + v] = f2(sv[v].get());
                }
            }
        }
        __syncthreads();
        if (!active) {
            continue;
        }
        df64 fix = ld(W, p.wn + n);
#pragma unroll
        for (int q = 0; q < NP; q++) {
            if (q < np && !(q < MV && q < nvq)) {
                const uint32_t s = mine[q];
                const int r = rmax[q];
                if (s & TG_SIX_J5) {
                    fix = fix * role(r, sm(r5, 2 * q), sm(r5, 2 * q + 1));
                } else if (r > 0) {
                    fix = fix * at(s, lane, r - 1);
                }
            }
        }

        const float2* key = sums + (pk[pi] * zl + lane) * ns1;
        const auto keysum = [&](int k) {
            df64 v = sm(key, k);
            for (int pt = 1; pt < parts; pt++) {
                v = v + sm(key + pt * nkz * ns1, k);
            }
            return v;
        };
        const df64 t0 = fix * keysum(0);
        acc[0].add(t0);

#pragma unroll
        for (int q = 0; q < NP; q++) {
            if (q < np) {
                const uint32_t s = mine[q];
                if (s & TG_SIX_LIVE) {
                    const df64 rho = at(s, lane, 3);
                    if (q < MV && q < nvq) {
                        acc[1 + q].add(fix * rho * keysum(1 + q));
                    } else {
                        acc[1 + q].add(times(t0 * rho, rmax[q]));
                    }
                }
            }
        }
        if ((n & 15) == 15) {
#pragma unroll
            for (int q = 0; q <= NP; q++) {
                acc[q].norm();
            }
        }
    }
    if (!active) {
        return;
    }
    const int nzp = p.nzp, iz = z0 + lane;
    d.out[p.out + static_cast<int64_t>(pi) * nzp + iz] = fin(acc[0].get(), p.scale);
    double* mom = d.out + p.out + static_cast<int64_t>(npat) * nzp + static_cast<int64_t>(pi) * np * nzp + iz;
#pragma unroll
    for (int q = 0; q < NP; q++) {
        if (q < np) {
            mom[static_cast<int64_t>(d.bases[p.bases + q].cpu) * nzp] = fin(acc[1 + q].get(), p.scale);
        }
    }
}

thread_local std::string g_error;

bool ok(cudaError_t e, const char* what) {
    if (e == cudaSuccess) {
        return true;
    }
    g_error = std::string(what) + ": " + cudaGetErrorString(e);
    return false;
}

struct Slot {
    char* host = nullptr;
    size_t cap = 0;
    cudaEvent_t done = nullptr;
    bool busy = false;
    size_t out_bytes = 0;
};

constexpr int kInFlight = 256;

constexpr size_t kMinStaging = 32 << 20;

Slot g_slots[kInFlight];

struct Gate {
    std::mutex m;
    std::condition_variable cv;
    std::vector<int> free;
    Gate() {
        for (int i = kInFlight - 1; i >= 0; i--) {
            free.push_back(i);
        }
    }
};

Gate g_gate;

int admit() {
    std::unique_lock<std::mutex> lock(g_gate.m);
    g_gate.cv.wait(lock, [] { return !g_gate.free.empty(); });
    const int i = g_gate.free.back();
    g_gate.free.pop_back();
    return i;
}

void leave(int i) {
    {
        std::lock_guard<std::mutex> lock(g_gate.m);
        g_gate.free.push_back(i);
    }
    g_gate.cv.notify_one();
}

std::atomic<int> g_pending{0};

void CUDART_CB on_done(void*) { g_pending.fetch_sub(1, std::memory_order_relaxed); }

bool reserve(Slot& slot, size_t host_bytes) {
    static std::once_flag pool;
    std::call_once(pool, [] {
        cudaMemPool_t mp;
        if (cudaDeviceGetDefaultMemPool(&mp, 0) == cudaSuccess) {
            uint64_t keep = ~0ull;
            cudaMemPoolSetAttribute(mp, cudaMemPoolAttrReleaseThreshold, &keep);
        }
    });

    if (slot.done == nullptr && !ok(cudaEventCreateWithFlags(&slot.done, cudaEventBlockingSync | cudaEventDisableTiming), "cudaEventCreate")) {
        return false;
    }
    if (host_bytes > slot.cap) {

        if (slot.host != nullptr) {
            cudaFreeHost(slot.host);
        }
        size_t cap = std::max({host_bytes + host_bytes / 2, slot.cap * 2, kMinStaging});
        if (!ok(cudaMallocHost(reinterpret_cast<void**>(&slot.host), cap), "cudaMallocHost")) {
            slot.host = nullptr;
            slot.cap = 0;
            return false;
        }
        slot.cap = cap;
    }
    return true;
}

Slot* slot_of(int64_t ticket) {
    if (ticket < 1 || ticket > kInFlight || !g_slots[ticket - 1].busy) {
        g_error = "no such call in flight";
        return nullptr;
    }
    return &g_slots[ticket - 1];
}

size_t align_up(size_t n) { return (n + 255) & ~static_cast<size_t>(255); }

template <int NP, int MV, bool CLASS>
bool launch(const Dev& d, const int32_t* ids, int n, int smem, cudaStream_t s) {
    if (n == 0) {
        return true;
    }

    static const bool carveout = ok(cudaFuncSetAttribute(nodes<NP, MV, CLASS>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared), "carveout");
    if (!carveout) {
        return false;
    }
    nodes<NP, MV, CLASS><<<n, kThreads, smem, s>>>(d, ids);
    return ok(cudaGetLastError(), "nodes");
}

constexpr int kKernels = 10;

int kernel_of(const TgSixPick& p) {
    const int b = p.np <= 8 ? 0 : p.nvq <= 8 ? (p.np <= 16 ? 1 : 2) : (p.np <= 16 ? 3 : 4);
    return b + (p.class_rows ? 0 : 5);
}

template <bool CLASS>
bool launch_all(const Dev& d, const int32_t* const* ids, const int* n, const int* smem, cudaStream_t s) {
    return launch<8, 8, CLASS>(d, ids[0], n[0], smem[0], s) && launch<16, 8, CLASS>(d, ids[1], n[1], smem[1], s) &&
           launch<32, 8, CLASS>(d, ids[2], n[2], smem[2], s) && launch<16, 16, CLASS>(d, ids[3], n[3], smem[3], s) &&
           launch<32, 16, CLASS>(d, ids[4], n[4], smem[4], s);
}

int smem_target() {
    static const int t = [] {
        const char* v = std::getenv("TG_SIX_SMEM_KB");
        return v ? std::atoi(v) << 10 : 24 << 10;
    }();
    return t;
}

}

extern "C" const char* tg_six_error(void) { return g_error.c_str(); }

extern "C" int32_t tg_six_pending(void) { return g_pending.load(std::memory_order_relaxed); }

extern "C" int32_t tg_six_slots(void) { return kInFlight; }

extern "C" int32_t tg_six_ready(int64_t ticket) {
    Slot* slot = slot_of(ticket);
    if (slot == nullptr) {
        return -1;
    }
    const cudaError_t e = cudaEventQuery(slot->done);
    if (e == cudaErrorNotReady) {
        return 0;
    }
    return ok(e, "kernels") ? 1 : -1;
}

extern "C" int32_t tg_six_wait(int64_t ticket, uint64_t* out, int64_t nout) {
    Slot* slot = slot_of(ticket);
    if (slot == nullptr) {
        return 1;
    }

    struct Back {
        Slot* slot;
        ~Back() {
            slot->busy = false;
            leave(static_cast<int>(slot - g_slots));
        }
    } back{slot};
    if (!ok(cudaEventSynchronize(slot->done), "kernels")) {
        return 1;
    }
    if (static_cast<size_t>(nout) * sizeof(uint64_t) != slot->out_bytes) {
        g_error = "outputs of another length than submitted";
        return 1;
    }
    std::memcpy(out, slot->host, slot->out_bytes);
    return 0;
}

extern "C" const uint64_t* tg_six_view(int64_t ticket, int64_t nout) {
    Slot* slot = slot_of(ticket);
    if (slot == nullptr || !ok(cudaEventSynchronize(slot->done), "kernels")) {
        return nullptr;
    }
    if (static_cast<size_t>(nout) * sizeof(uint64_t) != slot->out_bytes) {
        g_error = "outputs of another length than submitted";
        return nullptr;
    }
    return reinterpret_cast<const uint64_t*>(slot->host);
}

extern "C" int32_t tg_six_release(int64_t ticket) {
    Slot* slot = slot_of(ticket);
    if (slot == nullptr) {
        return 1;
    }
    slot->busy = false;
    leave(static_cast<int>(slot - g_slots));
    return 0;
}

extern "C" int32_t tg_six(const TgSixPick* picks, int32_t npicks, const uint64_t* words, int64_t nwords,
                          const uint64_t* mus, int64_t nmus, const uint16_t* shorts, int64_t nshorts,
                          const TgSixBase* bases, int64_t nbases, uint64_t* out, int64_t nout) {
    if (npicks == 0) {
        return 0;
    }
    const int64_t ticket = tg_six_submit(picks, npicks, words, nwords, mus, nmus, shorts, nshorts, bases, nbases, nout);
    return ticket == 0 ? 1 : tg_six_wait(ticket, out, nout);
}

extern "C" int64_t tg_six_submit(const TgSixPick* picks, int32_t npicks, const uint64_t* words, int64_t nwords,
                                 const uint64_t* mus, int64_t nmus, const uint16_t* shorts, int64_t nshorts,
                                 const TgSixBase* bases, int64_t nbases, int64_t nout) {

    std::vector<int4> blocks;
    std::vector<int32_t> ids[kKernels];
    int smem[kKernels] = {};
    const int target = smem_target();
    for (int32_t i = 0; i < npicks; i++) {
        const TgSixPick& p = picks[i];
        if (p.np > 32 || p.nvq > kMaxVary || p.npat > kThreads) {
            g_error = "a pick past the kernel's bounds";
            return 0;
        }
        const int per = kThreads / p.npat;
        const int nblk = (p.nz + per - 1) / per;
        const int zl = (p.nz + nblk - 1) / nblk;

        const auto need = [&](int tile) { return std::max(smem_bytes(p, zl, tile), smem_bytes(p, p.nz - (nblk - 1) * zl, tile)); };
        int tile = kRowTile;
        while (!p.class_rows && tile > kMinRowTile && need(tile) > target) {
            tile /= 2;
        }
        const int bytes = need(tile);
        if (bytes > kMaxSmem) {
            g_error = "a pick's node table past shared memory";
            return 0;
        }
        const int c = kernel_of(p);
        smem[c] = std::max(smem[c], bytes);
        for (int z0 = 0; z0 < p.nz; z0 += zl) {
            ids[c].push_back(static_cast<int32_t>(blocks.size()));
            blocks.push_back(make_int4(i, z0, std::min(zl, p.nz - z0), tile));
        }
    }
    struct Part {
        const void* src;
        size_t bytes;
        size_t at;
    };
    constexpr int kParts = 6 + kKernels;
    Part in[kParts] = {
        {picks, sizeof(TgSixPick) * static_cast<size_t>(npicks), 0},
        {blocks.data(), sizeof(int4) * blocks.size(), 0},
        {words, sizeof(uint64_t) * static_cast<size_t>(nwords), 0},
        {mus, sizeof(uint64_t) * static_cast<size_t>(nmus), 0},
        {shorts, sizeof(uint16_t) * static_cast<size_t>(nshorts), 0},
        {bases, sizeof(TgSixBase) * static_cast<size_t>(nbases), 0},
    };
    for (int c = 0; c < kKernels; c++) {
        in[6 + c] = {ids[c].data(), sizeof(int32_t) * ids[c].size(), 0};
    }
    size_t at = 0;
    for (Part& p : in) {
        p.at = at;
        at = align_up(at + p.bytes);
    }
    const size_t in_bytes = at;
    const size_t out_at = at;
    const size_t out_bytes = sizeof(uint64_t) * static_cast<size_t>(nout);
    at = align_up(at + out_bytes);

    const int si = admit();

    struct Hold {
        int si;
        bool held = true;
        ~Hold() {
            if (held) {
                cudaStreamSynchronize(cudaStreamPerThread);
                leave(si);
            }
        }
    } hold{si};
    Slot& slot = g_slots[si];
    cudaStream_t s = cudaStreamPerThread;

    if (!reserve(slot, std::max(in_bytes, out_bytes))) {
        return 0;
    }
    for (const Part& p : in) {
        if (p.bytes) {
            std::memcpy(slot.host + p.at, p.src, p.bytes);
        }
    }
    char* dv = nullptr;
    if (!ok(cudaMallocAsync(reinterpret_cast<void**>(&dv), at, s), "cudaMallocAsync")) {
        return 0;
    }

    struct Release {
        char* p;
        cudaStream_t s;
        ~Release() { cudaFreeAsync(p, s); }
    } release{dv, s};
    if (!ok(cudaMemcpyAsync(dv, slot.host, in_bytes, cudaMemcpyHostToDevice, s), "upload") ||
        !ok(cudaMemsetAsync(dv + out_at, 0, out_bytes, s), "memset")) {
        return 0;
    }
    Dev d;
    d.picks = reinterpret_cast<const TgSixPick*>(dv + in[0].at);
    d.blocks = reinterpret_cast<const int4*>(dv + in[1].at);
    d.words = reinterpret_cast<const uint64_t*>(dv + in[2].at);
    d.mus = reinterpret_cast<const uint64_t*>(dv + in[3].at);
    d.shorts = reinterpret_cast<const uint16_t*>(dv + in[4].at);
    d.bases = reinterpret_cast<const TgSixBase*>(dv + in[5].at);
    d.out = reinterpret_cast<double*>(dv + out_at);
    d.nwords = nwords;
    d.nmus = nmus;
    d.nshorts = nshorts;
    d.nbases = nbases;
    d.nout = nout;
    const int32_t* ids_at[kKernels];
    int n_of[kKernels];
    for (int c = 0; c < kKernels; c++) {
        ids_at[c] = reinterpret_cast<const int32_t*>(dv + in[6 + c].at);
        n_of[c] = static_cast<int>(ids[c].size());
    }
    if (!launch_all<true>(d, ids_at, n_of, smem, s) || !launch_all<false>(d, ids_at + 5, n_of + 5, smem + 5, s) ||
        !ok(cudaMemcpyAsync(slot.host, dv + out_at, out_bytes, cudaMemcpyDeviceToHost, s), "download") ||
        !ok(cudaEventRecord(slot.done, s), "cudaEventRecord")) {
        return 0;
    }
    g_pending.fetch_add(1, std::memory_order_relaxed);
    if (!ok(cudaLaunchHostFunc(s, on_done, nullptr), "cudaLaunchHostFunc")) {
        g_pending.fetch_sub(1, std::memory_order_relaxed);
        return 0;
    }
    hold.held = false;
    slot.busy = true;
    slot.out_bytes = out_bytes;
    return static_cast<int64_t>(si) + 1;
}
