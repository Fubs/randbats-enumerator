#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t nclasses;
    int32_t nl3;

    int32_t nc;
    int32_t pad;
    int64_t keys;
    int64_t dense;
    int64_t present;
} TgSituation;

typedef struct {
    int32_t i;
    int32_t r;
    int32_t first;
    int32_t pad;
    double a, b, c, s;
} TgBase;

typedef struct {
    int32_t i;
    int32_t pad;
    double a, b, c;
} TgRandom;

typedef struct {
    int32_t i3;
    int32_t moment;
    int32_t full;
    int32_t mu;
    double inv_size;
    double omega;
} TgCorr;

typedef struct {
    int32_t nx, ny;
    int64_t nodes;
} TgLevel;

typedef struct {
    int32_t situation;
    int32_t base4_at;
    int32_t bases, ncommon, nvariant;
    int32_t randoms, nrandom;
    int32_t corrs, ncorr;
    uint32_t num, den;
    int32_t levels, nlevels;
    int32_t pad;
    int64_t blocks;
    int64_t rows;
} TgPick;

int32_t tg_slot5(const TgSituation* sits, int32_t nsits, const TgPick* picks, int32_t npicks,
                 const TgBase* bases, int64_t nbases, const TgRandom* randoms, int64_t nrandoms,
                 const TgCorr* corrs, int64_t ncorrs, const TgLevel* levels, int64_t nlevels,
                 const uint8_t* block_level, int64_t nblock_level, const double* nodes, int64_t nnodes,
                 const uint64_t* keys, int64_t nkeys, const uint64_t* dense, int64_t ndense,
                 const uint8_t* present, int64_t npresent, double* rows_hi, double* rows_lo,
                 uint8_t* touched, int64_t nrows);

const char* tg_error(void);

#ifdef __cplusplus
}
#endif
