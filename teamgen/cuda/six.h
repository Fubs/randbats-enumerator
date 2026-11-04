#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t rmax;
    int32_t cpu;
} TgSixBase;

#define TG_SIX_SLOT 0x0fff
#define TG_SIX_LIVE 0x1000
#define TG_SIX_J5 0x2000

typedef struct {
    int32_t nx, ny, nz, nzp;
    int32_t np, nvq, nrow, nslot;
    int32_t npat, class_rows, ncv, nkeys;
    int64_t cq;
    int64_t keys;
    int64_t cb;
    int64_t wn;
    int64_t wm;
    int64_t zv;
    int64_t q5;
    int64_t lin;
    int64_t mu;
    int64_t bases;
    int64_t pslot;
    int64_t out;
    double scale;
} TgSixPick;

int32_t tg_six(const TgSixPick* picks, int32_t npicks, const uint64_t* words, int64_t nwords, const uint64_t* mus,
               int64_t nmus, const uint16_t* shorts, int64_t nshorts, const TgSixBase* bases, int64_t nbases,
               uint64_t* out, int64_t nout);

int64_t tg_six_submit(const TgSixPick* picks, int32_t npicks, const uint64_t* words, int64_t nwords,
                      const uint64_t* mus, int64_t nmus, const uint16_t* shorts, int64_t nshorts,
                      const TgSixBase* bases, int64_t nbases, int64_t nout);

int32_t tg_six_ready(int64_t ticket);

int32_t tg_six_wait(int64_t ticket, uint64_t* out, int64_t nout);

const uint64_t* tg_six_view(int64_t ticket, int64_t nout);

int32_t tg_six_release(int64_t ticket);

int32_t tg_six_pending(void);

int32_t tg_six_slots(void);

const char* tg_six_error(void);

#ifdef __cplusplus
}
#endif
