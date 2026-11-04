//! Slot 5's class sums on the GPU (`cuda/slot5.cu`, ABI in `cuda/slot5.h`):
//! a thread's batch of picks, each laid out from its situation and its
//! quadrature plan, and run in one call that blocks the thread.
use std::ffi::{CStr, c_char};
use super::{Num, Pick5, Quad, QuadPlan, Rows, Situation, Tables};
#[repr(C)]
#[derive(Clone, Copy)]
struct TgSituation {
    nclasses: i32,
    nl3: i32,
    nc: i32,
    pad: i32,
    keys: i64,
    dense: i64,
    present: i64,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct TgBase {
    i: i32,
    r: i32,
    first: i32,
    pad: i32,
    a: f64,
    b: f64,
    c: f64,
    s: f64,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct TgRandom {
    i: i32,
    pad: i32,
    a: f64,
    b: f64,
    c: f64,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct TgCorr {
    i3: i32,
    moment: i32,
    full: i32,
    mu: i32,
    inv_size: f64,
    omega: f64,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct TgLevel {
    nx: i32,
    ny: i32,
    nodes: i64,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct TgPick {
    situation: i32,
    base4_at: i32,
    bases: i32,
    ncommon: i32,
    nvariant: i32,
    randoms: i32,
    nrandom: i32,
    corrs: i32,
    ncorr: i32,
    num: u32,
    den: u32,
    levels: i32,
    nlevels: i32,
    pad: i32,
    blocks: i64,
    rows: i64,
}
unsafe extern "C" {
    fn tg_slot5(
        sits: *const TgSituation,
        nsits: i32,
        picks: *const TgPick,
        npicks: i32,
        bases: *const TgBase,
        nbases: i64,
        randoms: *const TgRandom,
        nrandoms: i64,
        corrs: *const TgCorr,
        ncorrs: i64,
        levels: *const TgLevel,
        nlevels: i64,
        block_level: *const u8,
        nblock_level: i64,
        nodes: *const f64,
        nnodes: i64,
        keys: *const u64,
        nkeys: i64,
        dense: *const u64,
        ndense: i64,
        present: *const u8,
        npresent: i64,
        rows_hi: *mut f64,
        rows_lo: *mut f64,
        touched: *mut u8,
        nrows: i64,
    ) -> i32;
    fn tg_error() -> *const c_char;
}
/// Classes a kernel block takes (`kThreads` in cuda/slot5.cu): each block of
/// a pick integrates under its own rule, sized to its classes' degree.
const BLOCK: usize = 256;
/// `TG_SHAPE_LOG=<path>`: a line per pick as it runs — classes, codes, rows,
/// nodes at the pick's degree and summed over its blocks' own, variant and
/// random bases, and its batch's pick count.
static SHAPE_LOG: std::sync::OnceLock<
    Option<std::sync::Mutex<std::io::BufWriter<std::fs::File>>>,
> = std::sync::OnceLock::new();
static SKIP_KERNEL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
/// One thread's picks awaiting the GPU; [`Batch::run`] fills their rows.
#[derive(Default)]
pub(crate) struct Batch {
    sits: Vec<TgSituation>,
    picks: Vec<TgPick>,
    bases: Vec<TgBase>,
    randoms: Vec<TgRandom>,
    corrs: Vec<TgCorr>,
    levels: Vec<TgLevel>,
    block_level: Vec<u8>,
    nodes: Vec<f64>,
    keys: Vec<u64>,
    dense: Vec<u64>,
    present: Vec<u8>,
    /// Each situation's blocks' most alive on each size-3 live base,
    /// `[block][base]`, from `bmax_at[situation]`.
    bmax: Vec<u8>,
    bmax_at: Vec<usize>,
    /// The device's work arrays for the picks so far, in 8-byte words: each
    /// block's partial cells and each level's node tables.
    work_words: usize,
    hi: Vec<f64>,
    lo: Vec<f64>,
    touched: Vec<u8>,
}
impl Batch {
    pub fn is_empty(&self) -> bool {
        self.picks.is_empty()
    }
    /// The device memory a run of the batch takes, in 8-byte words: when to
    /// flush.
    pub fn words(&self) -> usize {
        self.keys.len() + self.dense.len() + self.present.len() / 8 + 3 * self.hi.len()
            + self.work_words
    }
    /// `s`'s classes and contexts, once for all its picks; each context as the
    /// kernel's f32 pair. The classes go in key order, so a warp's neighbours
    /// share most bases' alive counts and the kernel multiplies those once,
    /// and a block's classes are mostly alike in degree.
    #[inline(never)]
    pub fn situation<N: Num>(&mut self, s: &Situation<N>) -> i32 {
        let (nc, nl3) = (s.nc(), s.live3.len());
        let sit = TgSituation {
            nclasses: s.keys.len() as i32,
            nl3: nl3 as i32,
            nc: nc as i32,
            pad: 0,
            keys: self.keys.len() as i64,
            dense: self.dense.len() as i64,
            present: self.present.len() as i64,
        };
        let mut order: Vec<u32> = (0..s.keys.len() as u32).collect();
        order.sort_unstable_by_key(|&c| s.keys[c as usize]);
        for &c in &order {
            let (c, k) = (c as usize, s.keys[c as usize]);
            self.keys.push(k as u64);
            self.keys.push((k >> 64) as u64);
            self.dense
                .extend(
                    s
                        .dense[c * nc..][..nc]
                        .iter()
                        .map(|&x| {
                            let x = N::to_f64(x);
                            let hi = x as f32;
                            let lo = (x - f64::from(hi)) as f32;
                            u64::from(hi.to_bits()) | u64::from(lo.to_bits()) << 32
                        }),
                );
            self.present.extend(s.present[c * nc..][..nc].iter().map(|&b| u8::from(b)));
        }
        if std::env::var_os("TG_SHAPE_LOG").is_some() {
            static SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(
                0,
            );
            static RANK1: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(
                0,
            );
            let row = |c: usize| -> Vec<f64> {
                let v: Vec<f64> = (0..nc)
                    .map(|d| {
                        if s.present[c * nc + d] {
                            N::to_f64(s.dense[c * nc + d])
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let t: f64 = v.iter().sum();
                v.iter().map(|x| x / t).collect()
            };
            let r0 = row(0);
            let rank1 = (1..s.keys.len())
                .all(|c| {
                    row(c)
                        .iter()
                        .zip(&r0)
                        .all(|(a, b)| {
                            (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1e-300)
                        })
                });
            let seen = SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            let r1 = RANK1
                .fetch_add(u64::from(rank1), std::sync::atomic::Ordering::Relaxed)
                + u64::from(rank1);
            if seen % 2000 == 0 {
                eprintln!("context rank-1: {r1} of {seen} situations");
            }
        }
        self.bmax_at.push(self.bmax.len());
        for blk in order.chunks(BLOCK) {
            let at = self.bmax.len();
            self.bmax.resize(at + nl3, 0);
            for &c in blk {
                for i in 0..nl3 {
                    self.bmax[at + i] = self.bmax[at + i].max(s.r(c as usize, i));
                }
            }
        }
        self.sits.push(sit);
        self.sits.len() as i32 - 1
    }
    /// Pick `p` of situation `sit` under `plan`; returns where its rows start.
    /// Each block of its classes takes the smallest rule exact for them: the
    /// nodes for its own degree, from `quad`'s rules.
    #[inline(never)]
    pub fn pick<N: Num>(
        &mut self,
        t: &Tables,
        sit: i32,
        s: &Situation<N>,
        p: &Pick5<N>,
        plan: &QuadPlan,
        quad: &mut Quad,
    ) -> i64 {
        debug_assert_eq!(plan.base4_at, p.at4_base);
        let base = |&(i, a, b, c, s, r): &(usize, f64, f64, f64, f64, u8)| TgBase {
            i: i as i32,
            r: i32::from(r),
            first: i32::from(Some(i) == plan.base4_at),
            pad: 0,
            a,
            b,
            c,
            s,
        };
        let pick = TgPick {
            situation: sit,
            base4_at: plan.base4_at.map_or(-1, |i| i as i32),
            bases: self.bases.len() as i32,
            ncommon: plan.common.len() as i32,
            nvariant: plan.variant.len() as i32,
            randoms: self.randoms.len() as i32,
            nrandom: plan.random.len() as i32,
            corrs: self.corrs.len() as i32,
            ncorr: p.corr.len() as i32,
            num: p.num,
            den: p.den,
            levels: self.levels.len() as i32,
            nlevels: 0,
            pad: 0,
            blocks: self.block_level.len() as i64,
            rows: self.hi.len() as i64,
        };
        self.bases.extend(plan.common.iter().chain(&plan.variant).map(base));
        self.randoms
            .extend(
                plan
                    .random
                    .iter()
                    .map(|&(i, _, a, b, c)| TgRandom {
                        i: i as i32,
                        pad: 0,
                        a,
                        b,
                        c,
                    }),
            );
        self.corrs
            .extend(
                p
                    .corr
                    .iter()
                    .map(|&(j, i3, size, omega, full)| TgCorr {
                        i3: i3 as i32,
                        moment: if p.random4[j] {
                            plan.random
                                .iter()
                                .position(|r| r.1 == j)
                                .map_or(-2, |k| k as i32)
                        } else {
                            -1
                        },
                        full: i32::from(full),
                        mu: i32::from(t.bases[p.live4[j].base as usize].mu),
                        inv_size: 1.0 / size as f64,
                        omega: N::to_f64(omega),
                    }),
            );
        let nl3 = s.live3.len();
        let bmax = self.bmax_at[sit as usize];
        let mut sizes: Vec<(usize, usize)> = Vec::new();
        for blk in 0..s.keys.len().div_ceil(BLOCK) {
            let row = &self.bmax[bmax + blk * nl3..][..nl3];
            let (mut dx, mut dy) = (0usize, 0usize);
            for &(i, _, _, c, _, _) in plan.common.iter().chain(&plan.variant) {
                dx += usize::from(row[i]);
                if c != 0.0 {
                    dy += usize::from(row[i]);
                }
            }
            let size = (dx / 2 + 1, dy / 2 + 1);
            let level = sizes
                .iter()
                .position(|&z| z == size)
                .unwrap_or_else(|| {
                    sizes.push(size);
                    sizes.len() - 1
                });
            self.block_level.push(level as u8);
        }
        for &(nx, ny) in &sizes {
            self.levels
                .push(TgLevel {
                    nx: nx as i32,
                    ny: ny as i32,
                    nodes: self.nodes.len() as i64,
                });
            let (xs, wx) = quad.rule(plan.f3, nx);
            self.nodes.extend_from_slice(xs);
            self.nodes.extend_from_slice(wx);
            let (ys, wy) = quad.rule(plan.f4, ny);
            self.nodes.extend_from_slice(ys);
            self.nodes.extend_from_slice(wy);
        }
        let n = (1 + p.corr.len()) * s.nc();
        let slices = plan.random.len().div_ceil(24).max(1);
        self.work_words += s.keys.len().div_ceil(BLOCK) * slices * n * 9 / 8;
        let chunk = 32 * (1 + 4 * plan.variant.len() + plan.random.len());
        self.work_words
            += sizes
                .iter()
                .map(|&(nx, ny)| (nx * ny).div_ceil(32) * chunk)
                .sum::<usize>();
        self.hi.resize(self.hi.len() + n, 0.0);
        self.picks
            .push(TgPick {
                nlevels: sizes.len() as i32,
                ..pick
            });
        pick.rows
    }
    pub fn run(&mut self) {
        if let Some(log) = SHAPE_LOG
            .get_or_init(|| {
                std::env::var_os("TG_SHAPE_LOG")
                    .map(|p| std::sync::Mutex::new(
                        std::io::BufWriter::new(std::fs::File::create(p).unwrap()),
                    ))
            })
        {
            use std::io::Write;
            static RUNS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(
                0,
            );
            let run = RUNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut f = log.lock().unwrap();
            for p in &self.picks {
                let s = &self.sits[p.situation as usize];
                let payload = {
                    use std::hash::{Hash, Hasher};
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    let bits = |v: f64| v.to_bits();
                    for b in &self
                        .bases[p.bases as usize..][..(p.ncommon + p.nvariant) as usize]
                    {
                        (b.i, b.r, b.first, bits(b.a), bits(b.b), bits(b.c), bits(b.s))
                            .hash(&mut h);
                    }
                    p.ncommon.hash(&mut h);
                    for r in &self.randoms[p.randoms as usize..][..p.nrandom as usize] {
                        (r.i, bits(r.a), bits(r.b), bits(r.c)).hash(&mut h);
                    }
                    for q in &self.corrs[p.corrs as usize..][..p.ncorr as usize] {
                        (q.i3, q.moment, q.full, q.mu, bits(q.inv_size), bits(q.omega))
                            .hash(&mut h);
                    }
                    for l in &self.levels[p.levels as usize..][..p.nlevels as usize] {
                        (l.nx, l.ny, bits(self.nodes[l.nodes as usize])).hash(&mut h);
                    }
                    (p.base4_at, p.num, p.den).hash(&mut h);
                    h.finish()
                };
                let levels = &self.levels[p.levels as usize..][..p.nlevels as usize];
                let blocks = &self
                    .block_level[p.blocks
                    as usize..][..(s.nclasses as usize).div_ceil(BLOCK)];
                let full = levels.iter().map(|l| l.nx * l.ny).max().unwrap();
                let own: i64 = blocks
                    .iter()
                    .enumerate()
                    .map(|(b, &l)| {
                        i64::from(levels[l as usize].nx * levels[l as usize].ny)
                            * (s.nclasses as usize - b * BLOCK).min(BLOCK) as i64
                    })
                    .sum();
                writeln!(
                    f, "{} {} {} {} {} {} {} {} {} {}", s.nclasses, s.nc, 1 + p.ncorr,
                    full, own, p.nvariant, p.nrandom, self.picks.len(), run << 20 |
                    u64::from(p.situation as u32), payload
                )
                    .unwrap();
            }
            f.flush().unwrap();
        }
        let n = self.hi.len();
        self.lo.resize(n, 0.0);
        self.touched.resize(n, 0);
        if *SKIP_KERNEL.get_or_init(|| std::env::var_os("TG_SKIP_KERNEL").is_some()) {
            self.touched.fill(1);
            return;
        }
        let rc = unsafe {
            tg_slot5(
                self.sits.as_ptr(),
                self.sits.len() as i32,
                self.picks.as_ptr(),
                self.picks.len() as i32,
                self.bases.as_ptr(),
                self.bases.len() as i64,
                self.randoms.as_ptr(),
                self.randoms.len() as i64,
                self.corrs.as_ptr(),
                self.corrs.len() as i64,
                self.levels.as_ptr(),
                self.levels.len() as i64,
                self.block_level.as_ptr(),
                self.block_level.len() as i64,
                self.nodes.as_ptr(),
                self.nodes.len() as i64,
                self.keys.as_ptr(),
                self.keys.len() as i64,
                self.dense.as_ptr(),
                self.dense.len() as i64,
                self.present.as_ptr(),
                self.present.len() as i64,
                self.hi.as_mut_ptr(),
                self.lo.as_mut_ptr(),
                self.touched.as_mut_ptr(),
                n as i64,
            )
        };
        if rc != 0 {
            panic!(
                "tg_slot5: {}", unsafe { CStr::from_ptr(tg_error()) } .to_string_lossy()
            );
        }
    }
    /// A run pick's rows, from where [`Batch::pick`] put them.
    #[inline(never)]
    pub fn rows<N: Num>(&self, at: i64, len: usize) -> Rows<N> {
        let at = at as usize;
        let acc = (at..at + len)
            .map(|i| {
                let mut a = N::Acc::default();
                a += N::of_f64(self.hi[i]);
                a += N::of_f64(self.lo[i]);
                a
            })
            .collect();
        Rows {
            acc,
            touched: self.touched[at..at + len].iter().map(|&b| b != 0).collect(),
        }
    }
    pub fn clear(&mut self) {
        self.sits.clear();
        self.picks.clear();
        self.bases.clear();
        self.randoms.clear();
        self.corrs.clear();
        self.levels.clear();
        self.block_level.clear();
        self.nodes.clear();
        self.keys.clear();
        self.dense.clear();
        self.present.clear();
        self.bmax.clear();
        self.bmax_at.clear();
        self.work_words = 0;
        self.hi.clear();
        self.lo.clear();
        self.touched.clear();
    }
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct TgSixBase {
    pub rmax: i32,
    pub cpu: i32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct TgSixPick {
    pub nx: i32,
    pub ny: i32,
    pub nz: i32,
    pub nzp: i32,
    pub np: i32,
    pub nvq: i32,
    pub nrow: i32,
    pub nslot: i32,
    pub npat: i32,
    pub class_rows: i32,
    pub ncv: i32,
    pub nkeys: i32,
    pub cq: i64,
    pub keys: i64,
    pub cb: i64,
    pub wn: i64,
    pub wm: i64,
    pub zv: i64,
    pub q5: i64,
    pub lin: i64,
    pub mu: i64,
    pub bases: i64,
    pub pslot: i64,
    pub out: i64,
    pub scale: f64,
}
unsafe extern "C" {
    fn tg_six(
        picks: *const TgSixPick,
        npicks: i32,
        words: *const u64,
        nwords: i64,
        mus: *const u64,
        nmus: i64,
        shorts: *const u16,
        nshorts: i64,
        bases: *const TgSixBase,
        nbases: i64,
        out: *mut u64,
        nout: i64,
    ) -> i32;
    fn tg_six_submit(
        picks: *const TgSixPick,
        npicks: i32,
        words: *const u64,
        nwords: i64,
        mus: *const u64,
        nmus: i64,
        shorts: *const u16,
        nshorts: i64,
        bases: *const TgSixBase,
        nbases: i64,
        nout: i64,
    ) -> i64;
    fn tg_six_ready(ticket: i64) -> i32;
    fn tg_six_wait(ticket: i64, out: *mut u64, nout: i64) -> i32;
    fn tg_six_view(ticket: i64, nout: i64) -> *const u64;
    fn tg_six_release(ticket: i64) -> i32;
    fn tg_six_pending() -> i32;
    fn tg_six_slots() -> i32;
    fn tg_six_error() -> *const c_char;
}
/// Slot-6 batches the GPU has not finished, every thread's.
pub(crate) fn six_pending() -> usize {
    unsafe { tg_six_pending() }.max(0) as usize
}
/// Slot-6 batches that may be in flight at once, every thread's.
pub(crate) fn six_slots() -> usize {
    unsafe { tg_six_slots() }.max(0) as usize
}
fn six_failed(what: &str) -> ! {
    panic!("{what}: {}", unsafe { CStr::from_ptr(tg_six_error()) } .to_string_lossy());
}
/// One thread's slot-6 picks awaiting the GPU node loop (`cuda/six.cu`, ABI
/// in `cuda/six.h`); [`SixBatch::run`] fills their A and Mom.
#[derive(Default)]
pub(crate) struct SixBatch {
    pub picks: Vec<TgSixPick>,
    /// Reals as df64 pairs: [`SixBatch::real`].
    pub words: Vec<u64>,
    pub mus: Vec<u64>,
    pub shorts: Vec<u16>,
    pub bases: Vec<TgSixBase>,
    nout: usize,
    out: Vec<u64>,
    /// The submitted call's ticket, 0 when none is in flight.
    ticket: i64,
    /// Once waited on, where its outputs are in the ticket's pinned staging
    /// (an address, 0 for none), until [`SixBatch::clear`] gives it back.
    view: usize,
}
impl SixBatch {
    /// `x` as the kernel's f32 pair.
    #[inline]
    pub fn real(x: f64) -> u64 {
        let hi = x as f32;
        let lo = (x - f64::from(hi)) as f32;
        u64::from(hi.to_bits()) | u64::from(lo.to_bits()) << 32
    }
    pub fn len(&self) -> usize {
        self.picks.len()
    }
    /// Bytes the batch moves each way, about: when to run it.
    pub fn bytes(&self) -> usize {
        8 * (self.words.len() + self.mus.len() + self.nout) + 2 * self.shorts.len()
    }
    /// Room for a pick's outputs, `len` pairs; returns where they start.
    pub fn alloc_out(&mut self, len: usize) -> i64 {
        let at = self.nout;
        self.nout += len;
        at as i64
    }
    pub fn run(&mut self) {
        self.out.clear();
        self.out.resize(self.nout, 0);
        let rc = unsafe {
            tg_six(
                self.picks.as_ptr(),
                self.picks.len() as i32,
                self.words.as_ptr(),
                self.words.len() as i64,
                self.mus.as_ptr(),
                self.mus.len() as i64,
                self.shorts.as_ptr(),
                self.shorts.len() as i64,
                self.bases.as_ptr(),
                self.bases.len() as i64,
                self.out.as_mut_ptr(),
                self.nout as i64,
            )
        };
        if rc != 0 {
            six_failed("tg_six");
        }
    }
    /// [`SixBatch::run`] without waiting: the GPU takes the batch, and
    /// [`SixBatch::wait`] collects its outputs. A batch of no picks has
    /// nothing to submit.
    pub fn submit(&mut self) {
        assert_eq!(self.ticket, 0, "a batch submitted twice");
        if self.picks.is_empty() {
            return;
        }
        self.ticket = unsafe {
            tg_six_submit(
                self.picks.as_ptr(),
                self.picks.len() as i32,
                self.words.as_ptr(),
                self.words.len() as i64,
                self.mus.as_ptr(),
                self.mus.len() as i64,
                self.shorts.as_ptr(),
                self.shorts.len() as i64,
                self.bases.as_ptr(),
                self.bases.len() as i64,
                self.nout as i64,
            )
        };
        if self.ticket == 0 {
            six_failed("tg_six_submit");
        }
    }
    /// Whether a submitted batch's outputs are ready (or there were none).
    pub fn ready(&self) -> bool {
        if self.ticket == 0 {
            return true;
        }
        match unsafe { tg_six_ready(self.ticket) } {
            1 => true,
            0 => false,
            _ => six_failed("tg_six_ready"),
        }
    }
    /// The submitted batch's outputs, waited for and read where they landed:
    /// a copy out cost more than the deposits' reading them (a batch's 23 MB
    /// zeroed, copied and read again, cold).
    pub fn wait(&mut self) {
        if self.ticket == 0 {
            return;
        }
        let p = unsafe { tg_six_view(self.ticket, self.nout as i64) };
        if p.is_null() {
            six_failed("tg_six_view");
        }
        self.view = p as usize;
    }
    /// Pick `i`'s A and Mom, `na` and `nm` long, where they landed: the
    /// kernel writes them as the deposits read them, so nothing converts.
    pub fn outputs(&self, i: usize, na: usize, nm: usize) -> (&[f64], &[f64]) {
        let base = if self.view != 0 {
            self.view as *const f64
        } else {
            assert_eq!(self.out.len(), self.nout, "outputs of a batch not run");
            self.out.as_ptr().cast::<f64>()
        };
        let all = unsafe { std::slice::from_raw_parts(base, self.nout) };
        let at = self.picks[i].out as usize;
        (&all[at..at + na], &all[at + na..at + na + nm])
    }
    /// Pick `i`'s outputs, `len` pairs, toward L2: they land in DRAM.
    pub fn prefetch(&self, i: usize, len: usize) {
        if self.view == 0 {
            return;
        }
        let p = (self.view as *const u64)
            .wrapping_add(self.picks[i].out as usize)
            .cast::<i8>();
        for o in (0..8 * len).step_by(64) {
            unsafe {
                std::arch::x86_64::_mm_prefetch::<
                    { std::arch::x86_64::_MM_HINT_T1 },
                >(p.wrapping_add(o))
            };
        }
    }
    pub fn clear(&mut self) {
        if self.view != 0 {
            if unsafe { tg_six_release(self.ticket) } != 0 {
                six_failed("tg_six_release");
            }
            (self.view, self.ticket) = (0, 0);
        }
        self.picks.clear();
        self.words.clear();
        self.mus.clear();
        self.shorts.clear();
        self.bases.clear();
        self.nout = 0;
    }
}
