#include "slot5.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <condition_variable>
#include <mutex>
#include <string>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kNodes = 32;
constexpr int kMaxVariant = 16;
constexpr int kMaxRandom = 24;

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

__device__ __forceinline__ df64 operator*(df64 a, df64 b) {
    float p = __fmul_rn(a.hi, b.hi);
    float e = fmaf(a.hi, b.hi, -p);
    e = fmaf(a.hi, b.lo, e);
    e = fmaf(a.lo, b.hi, e);
    return quick_two_sum(p, e);
}

__device__ __forceinline__ df64 split(double x) {
    float hi = __double2float_rn(x);
    return {hi, __double2float_rn(x - static_cast<double>(hi))};
}

__device__ __forceinline__ df64 word(uint64_t w) {
    return {__uint_as_float(static_cast<unsigned>(w)), __uint_as_float(static_cast<unsigned>(w >> 32))};
}

__device__ __forceinline__ df64 shfl_xor(df64 v, int m) {
    return {__shfl_xor_sync(~0u, v.hi, m), __shfl_xor_sync(~0u, v.lo, m)};
}

__device__ __forceinline__ df64 shfl(df64 v, int lane) {
    return {__shfl_sync(~0u, v.hi, lane), __shfl_sync(~0u, v.lo, lane)};
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

    __device__ __forceinline__ void fma(df64 a, df64 b) {
        float p = __fmul_rn(a.hi, b.hi);
        float e = fmaf(a.hi, b.hi, -p);
        e = fmaf(a.hi, b.lo, e);
        e = fmaf(a.lo, b.hi, e);
        float s = __fadd_rn(hi, p);
        float v = __fsub_rn(s, hi);
        float t = __fadd_rn(__fsub_rn(hi, __fsub_rn(s, v)), __fsub_rn(p, v));
        hi = s;
        lo = __fadd_rn(lo, __fadd_rn(t, e));
    }
    __device__ __forceinline__ void norm() {
        df64 s = quick_two_sum(hi, lo);
        hi = s.hi;
        lo = s.lo;
    }
    __device__ __forceinline__ df64 get() const { return quick_two_sum(hi, lo); }
};

__device__ __forceinline__ int alive(uint64_t lo, uint64_t hi, int i) {
    return static_cast<int>((i < 32 ? lo >> (2 * i) : hi >> (2 * (i - 32))) & 3);
}

__device__ __forceinline__ double pow_int(double e, int n) {
    double v = 1.0;
    for (int i = 0; i < n; i++) {
        v *= e;
    }
    return v;
}

struct Pick {
    TgPick p;
    int64_t consts;
    int32_t level0;
    int32_t cells;
    int32_t unit0, nunits;
};

struct Level {
    TgLevel l;
    int64_t tables;
    int32_t nchunks;
    int32_t pick;
};

constexpr int kW4 = 5;

struct Unit {
    int32_t pick, first;
    int32_t k0, level;
    int64_t partial;
};

struct Dev {
    const TgSituation* sits;
    const Pick* picks;
    const Level* levels;
    const TgBase* bases;
    const TgRandom* randoms;
    const TgCorr* corrs;
    const double* nodes;
    const uint64_t* keys;
    const uint64_t* dense;
    const uint8_t* present;
    const Unit* units;
    df64* tables;
    df64* partial;
    uint8_t* flags;
    double* rows_hi;
    double* rows_lo;
    uint8_t* touched;
};

__host__ __device__ inline int64_t chunk_words(const TgPick& p) {
    return static_cast<int64_t>(kNodes) * (1 + 4 * p.nvariant + p.nrandom);
}

__global__ void tables(Dev d) {
    const Level lv = d.levels[blockIdx.x];
    const Pick pk = d.picks[lv.pick];
    const TgPick& p = pk.p;
    const int nx = lv.l.nx, ny = lv.l.ny, np = nx * ny, nv = p.nvariant, nr = p.nrandom;
    const int64_t cw = chunk_words(p);
    const double* xs = d.nodes + lv.l.nodes;
    const double* wx = xs + nx;
    const double* ys = wx + nx;
    const double* wy = ys + ny;
    const TgBase* common = d.bases + p.bases;
    const TgBase* variant = common + p.ncommon;
    const TgRandom* random = d.randoms + p.randoms;

    for (int q = threadIdx.x; q < lv.nchunks * kNodes; q += blockDim.x) {
        df64* ch = d.tables + lv.tables + (q / kNodes) * cw;
        df64* tv = ch + kNodes;
        df64* tr = tv + 4 * nv * kNodes;
        const int j = q % kNodes;

        if (q >= np) {
            ch[j] = {0.f, 0.f};
            for (int k = 0; k < nv; k++) {
                for (int r = 0; r < 4; r++) {
                    tv[(k * kNodes + j) * 4 + r] = {0.f, 0.f};
                }
            }
            for (int k = 0; k < nr; k++) {
                tr[k * kNodes + j] = {0.f, 0.f};
            }
            continue;
        }
        const double x = xs[q / ny], y = ys[q % ny];
        double v = wx[q / ny] * wy[q % ny];
        for (int k = 0; k < p.ncommon; k++) {
            const TgBase b = common[k];
            const double e = b.a + b.b * x + b.c * x * y;
            v *= b.first ? b.s * x * pow_int(e, b.r - 1) : pow_int(e, b.r);
        }
        ch[j] = split(v);
        for (int k = 0; k < nv; k++) {
            const TgBase b = variant[k];
            const double e = b.a + b.b * x + b.c * x * y;
            double pw = 1.0;
            for (int r = 0; r < 4; r++) {
                double t = !b.first ? pw : r == 0 ? 0.0 : b.s * x * pow_int(e, r - 1);
                tv[(k * kNodes + j) * 4 + r] = split(r <= b.r ? t : 0.0);
                pw *= e;
            }
        }
        for (int k = 0; k < nr; k++) {
            const TgRandom b = random[k];
            tr[k * kNodes + j] = split(b.c * x * y / (b.a + b.b * x + b.c * x * y));
        }
    }
    if (static_cast<int>(blockIdx.x) != pk.level0) {
        return;
    }
    df64* consts = d.tables + pk.consts;
    for (int k = threadIdx.x; k < kW4 + 2 * p.ncorr; k += blockDim.x) {
        double v;
        if (k < kW4) {
            v = static_cast<double>(k < 4 ? static_cast<uint32_t>(k) : p.num) / static_cast<double>(p.den);
        } else {
            const TgCorr q = d.corrs[p.corrs + (k - kW4) / 2];
            v = (k - kW4) % 2 == 0 ? q.inv_size : -q.omega;
        }
        consts[k] = split(v);
    }
}

constexpr int blocks_per_sm(int nr) { return nr <= 8 ? 4 : nr <= 16 ? 3 : 2; }

template <int NR>
__global__ void __launch_bounds__(kThreads, blocks_per_sm(NR)) classes(Dev d, const int32_t* ids) {
    extern __shared__ __align__(16) df64 sm[];
    const Unit u = d.units[ids[blockIdx.x]];
    const Pick pk = d.picks[u.pick];
    const TgPick& p = pk.p;
    const TgSituation st = d.sits[p.situation];
    const int k0 = u.k0, nv = p.nvariant, nr = min(NR, p.nrandom - k0), nc = st.nc, cells = pk.cells;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int c = u.first + threadIdx.x;
    const bool valid = c < st.nclasses;
    uint64_t klo = 0, khi = 0;
    if (valid) {
        klo = d.keys[st.keys + 2 * c];
        khi = d.keys[st.keys + 2 * c + 1];
    }
    const TgBase* variant = d.bases + p.bases + p.ncommon;
    uint32_t rpack = 0;
    for (int k = 0; k < nv; k++) {
        rpack |= static_cast<uint32_t>(alive(klo, khi, variant[k].i)) << (2 * k);
    }

    const uint32_t lead = __shfl_sync(~0u, rpack, 0);
    rpack = valid ? rpack : lead;

    uint32_t uni = 0;
    for (int k = 0; k < nv; k++) {
        const uint32_t r = (rpack >> (2 * k)) & 3;
        uni |= static_cast<uint32_t>(__all_sync(~0u, r == __shfl_sync(~0u, r, 0))) << k;
    }
    const uint32_t varying = ((1u << nv) - 1) & ~uni;

    Lazy i0;
    Lazy mom[NR];
    const Level lv = d.levels[u.level];
    const int64_t cw = chunk_words(p);
    const df64* src = d.tables + lv.tables;
    const df64* tv = sm + kNodes;
    const df64* tr = tv + (4 * nv + k0) * kNodes;
    for (int ch = 0; ch < lv.nchunks; ch++) {
        __syncthreads();
        for (int64_t w = threadIdx.x; w < cw; w += kThreads) {
            sm[w] = src[ch * cw + w];
        }
        __syncthreads();
        df64 shared = sm[lane];
        for (uint32_t m = uni; m; m &= m - 1) {
            const int k = __ffs(m) - 1;
            shared = shared * tv[(k * kNodes + lane) * 4 + ((rpack >> (2 * k)) & 3)];
        }
#pragma unroll 1
        for (int q = 0; q < kNodes; q += 4) {
            df64 v[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                v[j] = shfl(shared, q + j);
            }
            for (uint32_t m = varying; m; m &= m - 1) {
                const int k = __ffs(m) - 1;
                const df64* t = tv + (k * kNodes + q) * 4 + ((rpack >> (2 * k)) & 3);
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    v[j] = v[j] * t[4 * j];
                }
            }
#pragma unroll
            for (int j = 0; j < 4; j++) {
                i0.add(v[j]);
            }
#pragma unroll
            for (int k = 0; k < NR; k++) {
                if (k < nr) {
                    const float4 r01 = *reinterpret_cast<const float4*>(tr + k * kNodes + q);
                    const float4 r23 = *reinterpret_cast<const float4*>(tr + k * kNodes + q + 2);
                    mom[k].fma(v[0], df64{r01.x, r01.y});
                    mom[k].fma(v[1], df64{r01.z, r01.w});
                    mom[k].fma(v[2], df64{r23.x, r23.y});
                    mom[k].fma(v[3], df64{r23.z, r23.w});
                }
            }
        }
        i0.norm();
#pragma unroll
        for (int k = 0; k < NR; k++) {
            mom[k].norm();
        }
    }

    const df64* consts = d.tables + pk.consts;
    const TgRandom* random = d.randoms + p.randoms;
    const TgCorr* corr = d.corrs + p.corrs;
    const int num4 = !valid ? 0 : p.base4_at >= 0 ? alive(klo, khi, p.base4_at) : 4;
    const df64 w4 = consts[num4];
    const df64 inv = w4 * i0.get();
    df64 m[NR];
#pragma unroll
    for (int k = 0; k < NR; k++) {
        m[k] = k < nr ? df64{static_cast<float>(alive(klo, khi, random[k0 + k].i)), 0.f} * mom[k].get() : df64{0.f, 0.f};
    }
    const bool live = valid && num4 != 0 && (p.base4_at >= 0 || p.num != 0);
    const uint64_t* dense = d.dense + st.dense + static_cast<int64_t>(valid ? c : 0) * nc;
    const uint8_t* present = d.present + st.present + static_cast<int64_t>(valid ? c : 0) * nc;

    __syncthreads();
    df64* wpart = sm;
    uint8_t* wflag = reinterpret_cast<uint8_t*>(sm + kWarps * cells);
    df64 coef{0.f, 0.f};
    bool on = false;
    for (int g = 0; g < cells; g += 8) {
        df64 v[8];
        uint32_t fl = 0;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int e = g + i;
            v[i] = {0.f, 0.f};
            if (e < cells) {
                const int r = e / nc, dd = e - r * nc;
                if (dd == 0 || i == 0) {
                    if (r == 0) {
                        coef = inv;
                        on = k0 == 0;
                    } else {
                        const TgCorr q = corr[r - 1];
                        const int a = alive(klo, khi, q.i3);
                        const df64 inv_size = consts[kW4 + 2 * (r - 1)], neg_omega = consts[kW4 + 2 * (r - 1) + 1];
                        df64 per{0.f, 0.f};
                        on = q.moment >= 0 ? q.moment >= k0 && q.moment < k0 + NR : k0 == 0;
                        if (on && q.moment != -1) {
                            df64 mk{0.f, 0.f};
#pragma unroll
                            for (int k = 0; k < NR; k++) {
                                if (k0 + k == q.moment) {
                                    mk = m[k];
                                }
                            }
                            per = w4 * mk * inv_size;
                        } else if (on && q.full && a == q.mu) {
                            on = false;
                        } else if (on) {
                            per = df64{static_cast<float>(a), 0.f} * inv_size * inv;
                        }
                        coef = on ? per + neg_omega * inv : df64{0.f, 0.f};
                    }
                }
                if (live && on && present[dd]) {
                    v[i] = word(dense[dd]) * coef;
                    fl |= 1u << i;
                }
            }
        }
        uint32_t any = 0;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            any |= static_cast<uint32_t>(__any_sync(~0u, (fl >> i) & 1)) << i;
        }

#pragma unroll
        for (int i = 0; i < 4; i++) {
            const bool up = lane & 16;
            const df64 send = up ? v[i] : v[i + 4];
            v[i] = (up ? v[i + 4] : v[i]) + shfl_xor(send, 16);
        }
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const bool up = lane & 8;
            const df64 send = up ? v[i] : v[i + 2];
            v[i] = (up ? v[i + 2] : v[i]) + shfl_xor(send, 8);
        }
        {
            const bool up = lane & 4;
            const df64 send = up ? v[0] : v[1];
            v[0] = (up ? v[1] : v[0]) + shfl_xor(send, 4);
        }
        v[0] = v[0] + shfl_xor(v[0], 2);
        v[0] = v[0] + shfl_xor(v[0], 1);
        const int j = (lane >> 2) & 7;
        if ((lane & 3) == 0 && g + j < cells) {
            wpart[warp * cells + g + j] = v[0];
            wflag[warp * cells + g + j] = (any >> j) & 1;
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < cells; e += kThreads) {
        df64 s = wpart[e];
        uint8_t f = wflag[e];
        for (int w = 1; w < kWarps; w++) {
            s = s + wpart[w * cells + e];
            f |= wflag[w * cells + e];
        }
        d.partial[u.partial + e] = s;
        d.flags[u.partial + e] = f;
    }
}

__global__ void reduce(Dev d) {
    const Pick pk = d.picks[blockIdx.x];
    for (int e = threadIdx.x; e < pk.cells; e += blockDim.x) {
        df64 s{0.f, 0.f};
        uint8_t f = 0;
        for (int k = 0; k < pk.nunits; k++) {
            const int64_t at = d.units[pk.unit0 + k].partial + e;
            s = s + d.partial[at];
            f |= d.flags[at];
        }
        d.rows_hi[pk.p.rows + e] = static_cast<double>(s.hi) + static_cast<double>(s.lo);
        d.rows_lo[pk.p.rows + e] = 0.0;
        d.touched[pk.p.rows + e] = f;
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

struct Buffers {
    char* host = nullptr;
    size_t host_cap = 0;
};

thread_local Buffers g_buf;

constexpr int kInFlight = 16;

struct Gate {
    std::mutex m;
    std::condition_variable cv;
    int free = kInFlight;
};

Gate g_gate;

struct Admit {
    Admit() {
        std::unique_lock<std::mutex> lock(g_gate.m);
        g_gate.cv.wait(lock, [] { return g_gate.free > 0; });
        g_gate.free--;
    }
    ~Admit() {
        {
            std::lock_guard<std::mutex> lock(g_gate.m);
            g_gate.free++;
        }
        g_gate.cv.notify_one();
    }
};

bool reserve(size_t host_bytes) {
    static std::once_flag pool;
    std::call_once(pool, [] {
        cudaMemPool_t mp;
        if (cudaDeviceGetDefaultMemPool(&mp, 0) == cudaSuccess) {
            uint64_t keep = ~0ull;
            cudaMemPoolSetAttribute(mp, cudaMemPoolAttrReleaseThreshold, &keep);
        }
    });
    if (host_bytes > g_buf.host_cap) {
        size_t cap = std::max(host_bytes, g_buf.host_cap * 2);
        if (!ok(cudaMallocHost(reinterpret_cast<void**>(&g_buf.host), cap), "cudaMallocHost")) {
            g_buf.host = nullptr;
            g_buf.host_cap = 0;
            return false;
        }
        g_buf.host_cap = cap;
    }
    return true;
}

size_t align_up(size_t n) { return (n + 255) & ~static_cast<size_t>(255); }

template <int NR>
bool launch(const Dev& d, const int32_t* ids, int n, size_t smem, cudaStream_t s) {
    if (n == 0) {
        return true;
    }
    static std::once_flag attr;
    std::call_once(attr, [] {
        cudaFuncSetAttribute(classes<NR>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);
    });
    classes<NR><<<n, kThreads, smem, s>>>(d, ids);
    return ok(cudaGetLastError(), "classes");
}

}

extern "C" const char* tg_error(void) { return g_error.c_str(); }

extern "C" int32_t tg_slot5(const TgSituation* sits, int32_t nsits, const TgPick* picks, int32_t npicks,
                            const TgBase* bases, int64_t nbases, const TgRandom* randoms, int64_t nrandoms,
                            const TgCorr* corrs, int64_t ncorrs, const TgLevel* levels, int64_t nlevels,
                            const uint8_t* block_level, int64_t nblock_level, const double* nodes, int64_t nnodes,
                            const uint64_t* keys, int64_t nkeys, const uint64_t* dense, int64_t ndense,
                            const uint8_t* present, int64_t npresent, double* rows_hi, double* rows_lo,
                            uint8_t* touched, int64_t nrows) {
    if (npicks == 0) {
        return 0;
    }

    std::vector<Pick> pk(npicks);
    std::vector<Level> lv(nlevels);
    std::vector<Unit> units;
    std::vector<int32_t> ids[3];
    int64_t table_words = 0, partial_words = 0;
    size_t smem[3] = {0, 0, 0};
    for (int32_t i = 0; i < npicks; i++) {
        const TgPick& p = picks[i];
        const TgSituation& st = sits[p.situation];
        if (p.nvariant > kMaxVariant) {
            g_error = "a pick has " + std::to_string(p.nvariant) + " variant bases; the kernel takes " +
                      std::to_string(kMaxVariant);
            return 1;
        }
        Pick& q = pk[i];
        q.p = p;
        q.level0 = p.levels;
        q.cells = (1 + p.ncorr) * st.nc;
        q.consts = table_words;
        table_words += kW4 + 2 * p.ncorr;
        for (int32_t j = p.levels; j < p.levels + p.nlevels; j++) {
            lv[j].l = levels[j];
            lv[j].pick = i;
            lv[j].nchunks = (levels[j].nx * levels[j].ny + kNodes - 1) / kNodes;
            lv[j].tables = table_words;
            table_words += lv[j].nchunks * chunk_words(p);
        }
        q.unit0 = static_cast<int32_t>(units.size());
        q.nunits = (st.nclasses + kThreads - 1) / kThreads;
        const int b = p.nrandom <= 8 ? 0 : p.nrandom <= 16 ? 1 : 2;
        const size_t need = std::max(sizeof(df64) * chunk_words(p),
                                     (sizeof(df64) + 1) * kWarps * static_cast<size_t>(q.cells));
        smem[b] = std::max(smem[b], need);
        const int slices = std::max(1, (p.nrandom + kMaxRandom - 1) / kMaxRandom);
        q.nunits *= slices;
        for (int32_t k = 0; k < q.nunits / slices; k++) {
            const int32_t level = p.levels + block_level[p.blocks + k];
            for (int sl = 0; sl < slices; sl++) {
                ids[b].push_back(static_cast<int32_t>(units.size()));
                units.push_back({i, k * kThreads, sl * kMaxRandom, level, partial_words});
                partial_words += q.cells;
            }
        }
    }
    (void)nblock_level;
    for (size_t s : smem) {
        if (s > 64 * 1024) {
            g_error = "a pick needs " + std::to_string(s) + " bytes of shared memory";
            return 1;
        }
    }

    struct Part {
        const void* src;
        size_t bytes;
        size_t at;
    };
    Part in[] = {
        {sits, sizeof(TgSituation) * nsits, 0},
        {pk.data(), sizeof(Pick) * npicks, 0},
        {bases, sizeof(TgBase) * nbases, 0},
        {randoms, sizeof(TgRandom) * nrandoms, 0},
        {corrs, sizeof(TgCorr) * ncorrs, 0},
        {nodes, sizeof(double) * nnodes, 0},
        {keys, sizeof(uint64_t) * nkeys, 0},
        {dense, sizeof(uint64_t) * ndense, 0},
        {present, static_cast<size_t>(npresent), 0},
        {units.data(), sizeof(Unit) * units.size(), 0},
        {ids[0].data(), sizeof(int32_t) * ids[0].size(), 0},
        {ids[1].data(), sizeof(int32_t) * ids[1].size(), 0},
        {ids[2].data(), sizeof(int32_t) * ids[2].size(), 0},
        {lv.data(), sizeof(Level) * nlevels, 0},
    };
    size_t at = 0;
    for (Part& p : in) {
        p.at = at;
        at = align_up(at + p.bytes);
    }
    const size_t in_bytes = at;
    const size_t hi_at = at;
    at = align_up(at + sizeof(double) * nrows);
    const size_t lo_at = at;
    at = align_up(at + sizeof(double) * nrows);
    const size_t touched_at = at;
    at = align_up(at + nrows);
    const size_t out_bytes = at - hi_at;
    const size_t tables_at = at;
    at = align_up(at + sizeof(df64) * table_words);
    const size_t partial_at = at;
    at = align_up(at + sizeof(df64) * partial_words);
    const size_t flags_at = at;
    at += partial_words;

    cudaStream_t s = cudaStreamPerThread;
    if (!reserve(std::max(in_bytes, out_bytes))) {
        return 1;
    }
    for (const Part& p : in) {
        if (p.bytes) {
            std::memcpy(g_buf.host + p.at, p.src, p.bytes);
        }
    }
    Admit admit;
    char* dv = nullptr;
    if (!ok(cudaMallocAsync(reinterpret_cast<void**>(&dv), at, s), "cudaMallocAsync")) {
        return 1;
    }

    struct Release {
        char* p;
        cudaStream_t s;
        ~Release() { cudaFreeAsync(p, s); }
    } release{dv, s};
    if (!ok(cudaMemcpyAsync(dv, g_buf.host, in_bytes, cudaMemcpyHostToDevice, s), "upload")) {
        return 1;
    }
    Dev d;
    d.sits = reinterpret_cast<const TgSituation*>(dv + in[0].at);
    d.picks = reinterpret_cast<const Pick*>(dv + in[1].at);
    d.bases = reinterpret_cast<const TgBase*>(dv + in[2].at);
    d.randoms = reinterpret_cast<const TgRandom*>(dv + in[3].at);
    d.corrs = reinterpret_cast<const TgCorr*>(dv + in[4].at);
    d.nodes = reinterpret_cast<const double*>(dv + in[5].at);
    d.keys = reinterpret_cast<const uint64_t*>(dv + in[6].at);
    d.dense = reinterpret_cast<const uint64_t*>(dv + in[7].at);
    d.present = reinterpret_cast<const uint8_t*>(dv + in[8].at);
    d.units = reinterpret_cast<const Unit*>(dv + in[9].at);
    d.levels = reinterpret_cast<const Level*>(dv + in[13].at);
    d.tables = reinterpret_cast<df64*>(dv + tables_at);
    d.partial = reinterpret_cast<df64*>(dv + partial_at);
    d.flags = reinterpret_cast<uint8_t*>(dv + flags_at);
    d.rows_hi = reinterpret_cast<double*>(dv + hi_at);
    d.rows_lo = reinterpret_cast<double*>(dv + lo_at);
    d.touched = reinterpret_cast<uint8_t*>(dv + touched_at);

    tables<<<static_cast<unsigned>(nlevels), 128, 0, s>>>(d);
    if (!ok(cudaGetLastError(), "tables") ||
        !launch<8>(d, reinterpret_cast<const int32_t*>(dv + in[10].at), static_cast<int>(ids[0].size()), smem[0], s) ||
        !launch<16>(d, reinterpret_cast<const int32_t*>(dv + in[11].at), static_cast<int>(ids[1].size()), smem[1], s) ||
        !launch<24>(d, reinterpret_cast<const int32_t*>(dv + in[12].at), static_cast<int>(ids[2].size()), smem[2], s)) {
        return 1;
    }
    reduce<<<npicks, 64, 0, s>>>(d);
    if (!ok(cudaGetLastError(), "reduce") ||
        !ok(cudaMemcpyAsync(g_buf.host, dv + hi_at, out_bytes, cudaMemcpyDeviceToHost, s), "download") ||
        !ok(cudaStreamSynchronize(s), "kernels")) {
        return 1;
    }
    std::memcpy(rows_hi, g_buf.host, sizeof(double) * nrows);
    std::memcpy(rows_lo, g_buf.host + (lo_at - hi_at), sizeof(double) * nrows);
    std::memcpy(touched, g_buf.host + (touched_at - hi_at), nrows);
    return 0;
}
