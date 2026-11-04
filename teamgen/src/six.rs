//! Slot 6 (MATH.md §7), one fourth pick at a time, its fifth picks by
//! pattern.
//!
//! A fourth pick's fifth picks change the integrand in two ways. A base that
//! steps 3 and 4 leave whole and the same in every class (W) is x^r·y^r
//! times a z-factor (1 − q₅ + q₅z)^r: a polynomial R̃(z) per fifth pick, no
//! work at the (x, y) nodes. A base that steps 3 or 4 split, or whose count
//! varies over the classes (P), is a factor at every (x, y, z) node; the
//! fifth pick changes few of them, and the distinct ways it does — patterns —
//! are few. So the node loop runs once per pick, each pattern a ratio on the
//! base integrand at every node, and each fifth pick is a dot over the z
//! nodes of its pattern's arrays with its own R̃ and pool size.
//!
//! Admissibility is a word: a context's filled caps as 64 bits, a species'
//! as the bits that would refuse it, so one AND decides it, and a fifth pick's
//! context is the fourth's with the caps that pick fills.
use super::*;
/// What every pick reads, built once: per species, the bits that refuse it
/// and whether it is a Zoroark (refused at the sixth pick).
pub(super) struct Bits {
    sp: Vec<u64>,
    zor: Vec<bool>,
}
const B_FIRE_GATE: u64 = 1 << 54;
const B_FD: u64 = 1 << 55;
const B_L100: u64 = 1 << 56;
const B_TB: u64 = 1 << 63;
const FIRE_BIT: u32 = 1 << FIRE;
impl Bits {
    pub(super) fn new(t: &Tables) -> Self {
        let sp = t
            .sp
            .iter()
            .map(|s| {
                let c = u64::from(s.compat);
                u64::from(s.types) | u64::from(s.weak) << 18 | u64::from(s.dweak) << 36
                    | if s.fire_gate { B_FIRE_GATE } else { 0 }
                    | if s.fd { B_FD } else { 0 } | if s.l100 { B_L100 } else { 0 }
                    | (c & u64::from(G_CLONE_A | G_CLONE_B | G_WEB | G_SCREEN)) << 57
                    | u64::from(s.compat & G_TOXICROAK != 0) << 61
                    | u64::from(s.compat & G_SUN != 0) << 62
                    | if s.tera_blocked { B_TB } else { 0 }
            })
            .collect();
        Bits {
            sp,
            zor: t.sp.iter().map(|s| s.zoroark).collect(),
        }
    }
    /// `ctx.adm(s)` for a size-5 context given as bits.
    #[inline]
    fn adm5(&self, ctx: u64, s: u16) -> bool {
        self.sp[s as usize] & ctx == 0 && !self.zor[s as usize]
    }
}
/// [`table_pass`] for slot-6 signatures (size 5, lead placed): there a
/// species is admissible exactly when its refusal bits miss the context and
/// it is no Zoroark, so its credit ω·U depends on the signature only through
/// that mask. Each signature's U goes once into every admissible distinct
/// mask's dense row, and each species takes ω times its mask's row at the
/// end: a credit per signature and mask, not per signature, species and
/// code.
pub(super) fn table_pass6<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    b: &Bits,
    table: Sharded<Ctx, KDeco<N>>,
    threads: usize,
    weights: &Mutex<Weights<N>>,
) {
    let timed = std::env::var_os("TG_TIME_TABLE").is_some();
    let t0 = Instant::now();
    let mut laps: Vec<(&str, f64)> = Vec::new();
    let mut lap = |name: &'static str| {
        if timed {
            laps.push((name, t0.elapsed().as_secs_f64()));
        }
    };
    let sigs = table.into_vec();
    lap("into_vec");
    if sigs.is_empty() {
        return;
    }
    let mut masks: Vec<u64> = (0..t.sp.len())
        .filter(|&s| !b.zor[s])
        .map(|s| b.sp[s])
        .collect();
    masks.sort_unstable();
    masks.dedup();
    let seen: Vec<Vec<bool>> = std::thread::scope(|scope| {
        let handles: Vec<_> = sigs
            .chunks(sigs.len().div_ceil(threads.max(1)))
            .map(|part| {
                scope
                    .spawn(move || {
                        let mut seen = vec![false; 1 << 16];
                        for (sig, u) in part {
                            let tb = if sig.tb { TB_BIT } else { 0 };
                            for e in u {
                                seen[usize::from(e.0 | tb)] = true;
                            }
                        }
                        seen
                    })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let codes: Vec<u16> = (0..=u16::MAX)
        .filter(|&c| seen.iter().any(|s| s[usize::from(c)]))
        .collect();
    let mut code_ix = vec![u32::MAX; 1 << 16];
    for (i, &c) in codes.iter().enumerate() {
        code_ix[usize::from(c)] = i as u32;
    }
    let (nm, nc) = (masks.len(), codes.len());
    lap("masks and codes");
    let chunk = sigs.len().div_ceil(threads.max(1));
    if N::QUADRATURE {
        let total = table_rows_f64::<N>(&sigs, &masks, &code_ix, nc, chunk);
        lap("rows and merge");
        table_credit::<N>(t, ar, b, &masks, &codes, &total, weights);
        lap("credit");
        if timed {
            println!(
                "table_pass6: {} signatures, {nm} masks × {nc} codes; {}", sigs.len(),
                laps.iter().map(| (n, s) | format!("{n} {s:.2}s")).collect::< Vec < _ >>
                ().join(", ")
            );
        }
        std::thread::spawn(move || drop(sigs));
        return;
    }
    let rows: Vec<Vec<N::Acc>> = std::thread::scope(|scope| {
        let handles: Vec<_> = sigs
            .chunks(chunk)
            .map(|part| {
                let (masks, code_ix) = (&masks, &code_ix);
                scope
                    .spawn(move || {
                        let mut acc = vec![N::Acc::default(); nm * nc];
                        let mut at: Vec<(usize, N)> = Vec::new();
                        for (sig, u) in part {
                            assert!(
                                sig.n == 5 && sig.lead,
                                "table_pass6 on a signature that is not a sixth pick's"
                            );
                            let bits = ctx_bits(sig);
                            let tb = if sig.tb { TB_BIT } else { 0 };
                            at.clear();
                            at.extend(
                                u
                                    .iter()
                                    .map(|&(c, x)| (
                                        code_ix[usize::from(c | tb)] as usize,
                                        N::get(x),
                                    )),
                            );
                            for (m, &mask) in masks.iter().enumerate() {
                                if mask & bits == 0 {
                                    let row = &mut acc[m * nc..(m + 1) * nc];
                                    for &(i, x) in &at {
                                        row[i] += x;
                                    }
                                }
                            }
                        }
                        acc
                    })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    lap("rows");
    let mut total = vec![N::Acc::default(); nm * nc];
    for part in rows {
        for (a, x) in total.iter_mut().zip(part) {
            *a += x;
        }
    }
    lap("merge");
    table_credit::<N>(t, ar, b, &masks, &codes, &total, weights);
    lap("credit");
    if timed {
        println!(
            "table_pass6: {} signatures, {nm} masks × {nc} codes; {}", sigs.len(), laps
            .iter().map(| (n, s) | format!("{n} {s:.2}s")).collect::< Vec < _ >> ()
            .join(", ")
        );
    }
}
/// Each species its mask's row of `total` (mask-major), times ω.
fn table_credit<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    b: &Bits,
    masks: &[u64],
    codes: &[u16],
    total: &[N::Acc],
    weights: &Mutex<Weights<N>>,
) {
    let nc = codes.len();
    let mut w = weights.lock().unwrap();
    for s in 0..t.sp.len() {
        if b.zor[s] {
            continue;
        }
        let m = masks.binary_search(&b.sp[s]).unwrap();
        for (i, &code) in codes.iter().enumerate() {
            let x = N::get(total[m * nc + i]);
            if x.to_f64() != 0.0 {
                credit(&mut w, s as u16, 6, false, code, ar.omega[s] * x);
            }
        }
    }
}
/// [`table_pass6`]'s rows in floats, mask-major: a thread's accumulator is
/// code-major, each code's row over the masks in f64x8 lanes, so an entry
/// goes into every admissible mask's row in one sequential sweep — not one
/// scattered add a mask into a mask-major table of tens of MB. Neumaier
/// sums in each lane (a masked-out entry adds an exact zero).
fn table_rows_f64<N: Num>(
    sigs: &[(Ctx, KDeco<N>)],
    masks: &[u64],
    code_ix: &[u32],
    nc: usize,
    chunk: usize,
) -> Vec<N::Acc> {
    use crate::simd::cmp::{SimdPartialEq, SimdPartialOrd};
    use crate::simd::num::SimdFloat;
    use crate::simd::{Mask, Select, u64x8};
    let nm = masks.len();
    let nmv = nm.div_ceil(8);
    let mv: Vec<u64x8> = (0..nmv)
        .map(|v| u64x8::from_array(
            std::array::from_fn(|l| masks.get(v * 8 + l).copied().unwrap_or(u64::MAX)),
        ))
        .collect();
    let valid: Vec<Mask<i64, 8>> = (0..nmv)
        .map(|v| Mask::from_array(std::array::from_fn(|l| v * 8 + l < nm)))
        .collect();
    let parts: Vec<(Vec<f64x8>, Vec<f64x8>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = sigs
            .chunks(chunk)
            .map(|part| {
                let (mv, valid) = (&mv, &valid);
                scope
                    .spawn(move || {
                        let zero = f64x8::splat(0.0);
                        let (mut s, mut c) = (
                            vec![zero; nc * nmv],
                            vec![zero; nc * nmv],
                        );
                        let mut adm: Vec<Mask<i64, 8>> = vec![Mask::splat(false); nmv];
                        for (sig, u) in part {
                            assert!(
                                sig.n == 5 && sig.lead,
                                "table_pass6 on a signature that is not a sixth pick's"
                            );
                            let bits = u64x8::splat(ctx_bits(sig));
                            for v in 0..nmv {
                                adm[v] = (mv[v] & bits).simd_eq(u64x8::splat(0)).cast()
                                    & valid[v];
                            }
                            let tb = if sig.tb { TB_BIT } else { 0 };
                            for &(code, x) in u {
                                let at = code_ix[usize::from(code | tb)] as usize * nmv;
                                let xv = f64x8::splat(N::get(x).to_f64());
                                for v in 0..nmv {
                                    let xm = adm[v].select(xv, zero);
                                    let s0 = s[at + v];
                                    let t = s0 + xm;
                                    c[at + v]
                                        += s0
                                            .abs()
                                            .simd_ge(xm.abs())
                                            .select((s0 - t) + xm, (xm - t) + s0);
                                    s[at + v] = t;
                                }
                            }
                        }
                        (s, c)
                    })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut total = vec![N::Acc::default(); nm * nc];
    for (s, c) in &parts {
        for i in 0..nc {
            for v in 0..nmv {
                let (sv, cv) = (s[i * nmv + v].to_array(), c[i * nmv + v].to_array());
                for l in 0..8 {
                    let m = v * 8 + l;
                    if m < nm && (sv[l] != 0.0 || cv[l] != 0.0) {
                        let acc = &mut total[m * nc + i];
                        *acc += N::of_f64(sv[l]);
                        *acc += N::of_f64(cv[l]);
                    }
                }
            }
        }
    }
    total
}
fn ctx_bits(c: &Ctx) -> u64 {
    u64::from(c.types_full) | u64::from(c.weak_full) << 18
        | u64::from(c.dweak_full) << 36 | if c.fire_full { B_FIRE_GATE } else { 0 }
        | if c.fd_full { B_FD } else { 0 } | if c.l100_full { B_L100 } else { 0 }
        | (u64::from(c.compat) & 15) << 57 | u64::from(c.compat & G_SUN != 0) << 61
        | u64::from(c.compat & G_TOXICROAK != 0) << 62 | if c.tb { B_TB } else { 0 }
}
/// The size-5 context (lead placed) whose bits are `b`.
fn ctx_of_bits(b: u64) -> Ctx {
    let compat = ((b >> 57) & 15) as u8 | if b >> 61 & 1 != 0 { G_SUN } else { 0 }
        | if b >> 62 & 1 != 0 { G_TOXICROAK } else { 0 };
    Ctx {
        n: 5,
        lead: true,
        tb: b & B_TB != 0,
        types_full: (b & 0x3FFFF) as u32,
        weak_full: ((b >> 18) & 0x3FFFF) as u32,
        dweak_full: ((b >> 36) & 0x3FFFF) as u32,
        fd_full: b & B_FD != 0,
        l100_full: b & B_L100 != 0,
        fire_full: b & B_FIRE_GATE != 0,
        compat,
    }
}
/// A team's caps with the counts one more member can tip: the types held
/// once, the weaknesses held twice, the Fire count, the Freeze-Dry count.
struct Caps {
    bits: u64,
    t1: u32,
    w2: u32,
    fire: u8,
    fd: u8,
}
impl Caps {
    fn of_key(t: &Tables, k: &Key) -> Caps {
        let (mut tc, mut wc) = ([0u8; 18], [0u8; 18]);
        let mut fd = 0u8;
        for &c in k.members() {
            let s = &t.sp[t.classes[c as usize].members[0] as usize];
            for i in 0..18 {
                tc[i] += u8::from(s.types >> i & 1 != 0);
                wc[i] += u8::from(s.weak >> i & 1 != 0);
            }
            fd += u8::from(s.fd);
        }
        wc[FIRE] += k.fire;
        let mask = |c: &[u8; 18], n: u8| {
            (0..18).filter(|&i| c[i] == n).fold(0u32, |m, i| m | 1 << i)
        };
        Caps {
            bits: ctx_bits(&Ctx::of_key(t, k)),
            t1: mask(&tc, 1),
            w2: mask(&wc, 2) & !FIRE_BIT,
            fire: wc[FIRE],
            fd,
        }
    }
    /// The context bits once `s` joins, its set adding `fire` to the Fire
    /// count and leaving Tera Blast `tb`.
    #[inline]
    fn add(&self, s: &Sp, fire: u8, tb: bool) -> u64 {
        let mut b = self.bits | u64::from(s.types & self.t1)
            | u64::from(s.weak & self.w2) << 18 | u64::from(s.dweak) << 36;
        if self.fire + u8::from(s.weak & FIRE_BIT != 0) + fire >= 3 {
            b |= u64::from(FIRE_BIT) << 18 | B_FIRE_GATE;
        }
        if self.fd + u8::from(s.fd) >= 4 {
            b |= B_FD;
        }
        if s.l100 {
            b |= B_L100;
        }
        b
            |= (u64::from(s.compat) & 15) << 57 | u64::from(s.compat & G_SUN != 0) << 61
                | u64::from(s.compat & G_TOXICROAK != 0) << 62;
        if tb {
            b |= B_TB;
        }
        b
    }
}
/// One fifth pick on one outcome of its set.
#[derive(Clone, Copy)]
struct Sig {
    ctx: u64,
    o5: u16,
    /// Its routing of the fourth pick's codes: a range of `Six::maps`, or
    /// none for the identity.
    map: (u32, u32),
    tb: bool,
    /// The Fire count its set adds.
    fire: u8,
    picked: u16,
    class: u16,
    /// Its base's live4 index, when a multi-forme base.
    j5: u8,
    num: u32,
    den: u32,
    pattern: u16,
    m0: u32,
    coef: f64,
    /// Range of `Six::wch` (W bases it changes) and the pick's m5 row.
    wch: (u32, u32),
    group: u32,
}
#[derive(Clone, Copy)]
struct Live4 {
    i3: usize,
    mask4: u32,
    n4: u32,
    base: u16,
    /// Whether a P base (steps 3–4 split it, or its count varies).
    p: bool,
    /// W: its count; P: its index among the P bases.
    r: u8,
    pi: u8,
    /// Formes of its base admissible at the sixth pick, Zoroark aside.
    nzor: u32,
    /// Every refusal bit of its live formes.
    orb: u64,
}
#[derive(Clone, Copy)]
struct PBase {
    j: u8,
    i3: usize,
    q3: f64,
    q4: f64,
}
/// A factor in x alone: a base the fourth pick took (role 4) or left with
/// nothing admissible, split at step 3 or with a varying count.
#[derive(Clone, Copy)]
struct XBase {
    i3: usize,
    q3: f64,
    role4: bool,
}
/// One pattern: the P bases a fifth pick changes, with their new masks
/// (0: refused), and the P base it takes, if any.
#[derive(Clone, PartialEq)]
struct Pattern {
    changed: Vec<(u8, u32)>,
    j5: u8,
}
/// One context group of a pick's fifth picks: its range of `order`, its row
/// of `m5`, what its plain picks (no multi-forme base, no set routing)
/// share — W bases changed (`wch`) and refused or changed (`wex`), the
/// pattern — and, over all its picks, the pool sizes' range and the largest
/// z-degree besides z^{m0}: its quadrature rule.
#[derive(Clone, Copy)]
struct Group {
    lo: u32,
    hi: u32,
    row: u32,
    wch: (u32, u32),
    wex: (u32, u32),
    pattern: u16,
    m_min: u32,
    m_max: u32,
    deg: u32,
}
/// A pick past sections A–D: its fifth picks by context, and what its node
/// loop and deposits read of its rows, bases and rules.
#[derive(Default)]
pub(super) struct Prep {
    sigs: Vec<Sig>,
    maps: Vec<(u8, u16, f64)>,
    wch: Vec<(u8, f64)>,
    wex: Vec<u8>,
    m5: Vec<u32>,
    patterns: Vec<Pattern>,
    order: Vec<u32>,
    groups: Vec<Group>,
    nclass: usize,
    nrow: usize,
    nv: usize,
    vary: Vec<bool>,
    vix: Vec<u8>,
    rcf: Vec<[f64; 4]>,
    rlin: Vec<(f64, f64, u8)>,
    class_rows: bool,
    kc: Vec<u16>,
    wt: Vec<f64>,
    live_row: Vec<bool>,
    rdeg: Vec<u8>,
    xb: Vec<XBase>,
    pb: Vec<PBase>,
    l4: Vec<Live4>,
    rmax_p: Vec<u8>,
    den4: f64,
    members4: Vec<u16>,
    nz: usize,
    zv: Vec<f64>,
    bary: Vec<f64>,
    xs: Vec<f64>,
    wx: Vec<f64>,
    ys: Vec<f64>,
    wy: Vec<f64>,
    /// Whether the rows' code weights are one profile scaled: `prof`, and
    /// each row's scale `srow`.
    single: bool,
    prof: Vec<f64>,
    srow: Vec<f64>,
    /// Else, when the rows' code weights span few profiles (every
    /// multi-row pick's, two, measured): `nb` orthonormal profiles `pbas`
    /// [nb][nk] and each row's coordinates `ub` [nrow][nb] — the node loop
    /// runs `nb` code rows, not one a code.
    nb: usize,
    pbas: Vec<f64>,
    ub: Vec<f64>,
}
impl Prep {
    /// Code rows the node loop carries.
    pub(super) fn nkk(&self) -> usize {
        if self.single { 1 } else { self.kc.len() }
    }
    /// Code rows the node loop runs: `nkk`, or the basis rows (`nb`) when
    /// the rows' code weights span fewer.
    fn nkn(&self) -> usize {
        if self.single { 1 } else if self.nb > 0 { self.nb } else { self.kc.len() }
    }
    /// Row `ri`'s weight in the node loop's code row `k`.
    #[inline]
    fn row_weight(&self, ri: usize, k: usize) -> f64 {
        if self.single {
            self.srow[ri]
        } else if self.nb > 0 {
            self.ub[ri * self.nb + k]
        } else {
            self.wt[ri * self.kc.len() + k]
        }
    }
    fn nzp(&self) -> usize {
        self.nz.div_ceil(8) * 8
    }
    /// A's and Mom's lengths, as the node loop lays them out.
    #[cfg(feature = "gpu")]
    pub(super) fn out_lens(&self) -> (usize, usize) {
        let w = self.patterns.len() * self.nkk() * self.nzp();
        (w, w * self.pb.len())
    }
}
/// A thread's buffers, kept across picks, and its situation-level sums.
#[derive(Default)]
pub(super) struct Six {
    /// The CPU path's pick, kept for its allocations.
    prep: Prep,
    a: Vec<f64>,
    mom: Vec<f64>,
    ta: Vec<f64>,
    tm: Vec<f64>,
    cs: Codes,
    sums: Sums,
    pcell: Pcell,
    pcell1: Pcell1,
    f1: F1,
    /// A group's deposits summed by pick code, for its members' clause; the
    /// codes it touched, and a mark for each.
    gu: Vec<f64>,
    gtouch: Vec<usize>,
    /// Per pick code index, its raw code, set on the group's first touch.
    gcode: Vec<u16>,
    gmark: Vec<bool>,
    /// Gauss–Jacobi rules with their interpolation weights from the
    /// Chebyshev points, by (m_min, nodes, points).
    lcache: Lcache,
    /// The item's table deposits.
    pub(super) table: FxMap<u64, Vec<(u16, f64)>>,
    /// [`add_route`]'s index of a routing's entries (1-based), and the slots
    /// in use.
    rhash: Vec<u32>,
    rused: Vec<usize>,
    /// A pick's node table, kept for its allocation; likewise the node
    /// loop's pattern integrands and shared moment columns.
    tab: Vec<f64>,
    gp: Vec<f64x8>,
    gb: Vec<f64x8>,
    /// A pick's set kernels ([`kern_of`]), gathered ahead of section C.
    kos_at: Vec<(usize, usize)>,
}
/// Final codes (Tera Blast bit included): the situation's indices, and a
/// pick's own, local to it, with the pick's U and W exceptions by local
/// code (`exc` code-major, `nl4` bases a code).
#[derive(Default)]
struct Codes {
    cix: Vec<u16>,
    codes: Vec<u16>,
    loc: Vec<u16>,
    lcodes: Vec<u16>,
    nl4: usize,
    u_all: Vec<f64>,
    exc: Vec<f64>,
}
/// The situation's sums, dense: per species and per simple class, a row of
/// `cap` code indices (doubled as the situation's codes grow), and the rows
/// touched.
#[derive(Default)]
struct Sums {
    cap: usize,
    acc: Vec<f64>,
    touched: Vec<bool>,
    tlist: Vec<u16>,
    cls: Vec<f64>,
    ctouched: Vec<bool>,
    clist: Vec<u16>,
}
impl Sums {
    fn ready(&mut self, species: usize, classes: usize) {
        if self.cap == 0 {
            self.cap = 64;
            self.acc = vec![0.0; species * self.cap];
            self.touched = vec![false; species];
            self.cls = vec![0.0; classes * self.cap];
            self.ctouched = vec![false; classes];
        }
    }
    /// Room for code index `at`, rows moved to the wider stride.
    #[cold]
    fn grow(&mut self, at: usize) {
        let old = self.cap;
        let mut cap = old;
        while cap <= at {
            cap *= 2;
        }
        for (buf, n) in [
            (&mut self.acc, self.touched.len()),
            (&mut self.cls, self.ctouched.len()),
        ] {
            let mut wide = vec![0.0; n * cap];
            for r in 0..n {
                wide[r * cap..r * cap + old]
                    .copy_from_slice(&buf[r * old..(r + 1) * old]);
            }
            *buf = wide;
        }
        self.cap = cap;
    }
    #[inline]
    fn add(&mut self, f: u16, at: u16, x: f64) {
        if usize::from(at) >= self.cap {
            self.grow(usize::from(at));
        }
        if !self.touched[f as usize] {
            self.touched[f as usize] = true;
            self.tlist.push(f);
        }
        self.acc[f as usize * self.cap + usize::from(at)] += x;
    }
    /// Species `f`'s row, wide enough for code index `top`, marked touched.
    #[inline]
    fn row_mut(&mut self, f: u16, top: u16) -> &mut [f64] {
        if usize::from(top) >= self.cap {
            self.grow(usize::from(top));
        }
        if !self.touched[f as usize] {
            self.touched[f as usize] = true;
            self.tlist.push(f);
        }
        &mut self.acc[f as usize * self.cap..(f as usize + 1) * self.cap]
    }
    #[inline]
    fn add_class(&mut self, c: u16, at: u16, x: f64) {
        if usize::from(at) >= self.cap {
            self.grow(usize::from(at));
        }
        if !self.ctouched[c as usize] {
            self.ctouched[c as usize] = true;
            self.clist.push(c);
        }
        self.cls[c as usize * self.cap + usize::from(at)] += x;
    }
}
impl Six {
    /// `TG_MEM_STATS`: what the thread's caches hold, roughly in MB.
    pub(super) fn mem_stats(&self) -> String {
        let lc: usize = self.lcache.values().map(|r| 8 * r.len() + 64).sum();
        format!(
            "lcache {} entries {:.0} MB, sums {:.0} MB, table {}", self.lcache.len(), lc
            as f64 / 1e6, 8.0 * (self.sums.acc.len() + self.sums.cls.len()) as f64 / 1e6,
            self.table.len()
        )
    }
    fn ready(&mut self, t: &Tables) {
        if self.cs.cix.is_empty() {
            self.cs.cix = vec![0; 1 << 14];
        }
        self.sums.ready(t.sp.len(), t.classes.len());
    }
    /// The situation's sums into `w`, at the sixth slot.
    pub(super) fn flush<N: Num>(&mut self, t: &Tables, w: &mut Weights<N>) {
        let omega = |f: u16| {
            let b = &t.bases[t.sp[f as usize].base as usize];
            f64::from(b.mu) / f64::from(b.phi)
        };
        let (cs, sums) = (&mut self.cs, &mut self.sums);
        let (cap, nc) = (sums.cap, cs.codes.len());
        if fine() {
            let nz = sums
                .tlist
                .iter()
                .map(|&f| {
                    sums
                        .acc[f as usize * cap..f as usize * cap + nc]
                        .iter()
                        .filter(|&&v| v != 0.0)
                        .count()
                })
                .sum::<usize>();
            let lines = sums
                .tlist
                .iter()
                .map(|&f| {
                    sums
                        .acc[f as usize * cap..f as usize * cap + nc]
                        .chunks(8)
                        .filter(|c| c.iter().any(|&v| v != 0.0))
                        .count()
                })
                .sum::<usize>();
            for (i, v) in [1, sums.tlist.len(), nc, nz, lines].into_iter().enumerate() {
                DENS[i].fetch_add(v as u64, Ordering::Relaxed);
            }
            stat_flush();
        }
        for (i, v) in [1, sums.tlist.len(), sums.clist.len(), nc, cap]
            .into_iter()
            .enumerate()
        {
            FLSTAT[i].fetch_add(v as u64, Ordering::Relaxed);
        }
        stat_flush();
        for &f in &sums.tlist {
            for at in 0..nc {
                let x = std::mem::take(&mut sums.acc[f as usize * cap + at]);
                if x != 0.0 {
                    credit(w, f, 6, false, cs.codes[at], N::of_f64(x));
                }
            }
            sums.touched[f as usize] = false;
        }
        for &c in &sums.clist {
            let members = &t.classes[c as usize].members;
            for at in 0..nc {
                let x = std::mem::take(&mut sums.cls[c as usize * cap + at]);
                if x != 0.0 {
                    for &m in members {
                        credit(
                            w,
                            m,
                            6,
                            false,
                            cs.codes[at],
                            N::of_f64(-omega(m) * x / members.len() as f64),
                        );
                    }
                }
            }
            sums.ctouched[c as usize] = false;
        }
        sums.tlist.clear();
        sums.clist.clear();
        for &code in &cs.codes {
            cs.cix[code as usize] = 0;
        }
        cs.codes.clear();
        cs.loc.clear();
        cs.lcodes.clear();
    }
    /// The item's table deposits into `table6`.
    pub(super) fn flush_table<N: Num>(&mut self, table6: &Sharded<Ctx, KDeco<N>>) {
        for (bits, deco) in self.table.drain() {
            let ctx = ctx_of_bits(bits);
            let mut shard = table6.shard(fxhash(&ctx)).lock().unwrap();
            let entry = shard.entry(ctx).or_default();
            for (code, x) in deco {
                kdeco_add::<N>(entry, code, N::of_f64(x));
            }
        }
    }
}
impl Codes {
    /// The situation's index for final code `code`, and the pick's.
    #[inline]
    fn code(&mut self, code: u16) -> (u16, usize) {
        let mut at = self.cix[code as usize];
        if at == 0 {
            self.codes.push(code);
            at = self.codes.len() as u16;
            self.cix[code as usize] = at;
            self.loc.push(0);
        }
        let at = at - 1;
        let mut l = self.loc[at as usize];
        if l == 0 {
            self.lcodes.push(at);
            l = self.lcodes.len() as u16;
            self.loc[at as usize] = l;
            self.u_all.push(0.0);
            self.exc.resize(self.exc.len() + self.nl4, 0.0);
        }
        (at, usize::from(l) - 1)
    }
    /// A new pick's code space, `nl4` live bases.
    fn new_pick(&mut self, nl4: usize) {
        for &at in &self.lcodes {
            self.loc[at as usize] = 0;
        }
        self.lcodes.clear();
        self.u_all.clear();
        self.exc.clear();
        self.nl4 = nl4;
    }
}
#[inline]
fn ratio(a: u32, b: u32) -> f64 {
    f64::from(a.count_ones()) / f64::from(b.count_ones())
}
use std::cell::Cell;
use crate::simd::{StdFloat, f64x8};
#[inline(always)]
fn lanes(s: &[f64]) -> impl Iterator<Item = f64x8> + '_ {
    s.chunks_exact(8).map(f64x8::from_slice)
}
#[inline(always)]
fn mul_into(d: &mut [f64], s: &[f64]) {
    for (d, s) in d.chunks_exact_mut(8).zip(lanes(s)) {
        (f64x8::from_slice(d) * s).copy_to_slice(d);
    }
}
#[inline(always)]
fn div_into(d: &mut [f64], s: &[f64]) {
    for (d, s) in d.chunks_exact_mut(8).zip(lanes(s)) {
        (f64x8::from_slice(d) / s).copy_to_slice(d);
    }
}
#[inline(always)]
fn scale_into(d: &mut [f64], s: f64) {
    let s = f64x8::splat(s);
    for d in d.chunks_exact_mut(8) {
        (f64x8::from_slice(d) * s).copy_to_slice(d);
    }
}
/// d += a·b
#[inline(always)]
fn fma_into(d: &mut [f64], a: &[f64], b: &[f64]) {
    for ((d, a), b) in d.chunks_exact_mut(8).zip(lanes(a)).zip(lanes(b)) {
        a.mul_add(b, f64x8::from_slice(d)).copy_to_slice(d);
    }
}
/// d += a·b·c·s
#[inline(always)]
fn fma3_into(d: &mut [f64], a: &[f64], b: &[f64], c: &[f64], s: f64) {
    let s = f64x8::splat(s);
    for (((d, a), b), c) in d
        .chunks_exact_mut(8)
        .zip(lanes(a))
        .zip(lanes(b))
        .zip(lanes(c))
    {
        (a * b * s).mul_add(c, f64x8::from_slice(d)).copy_to_slice(d);
    }
}
/// d += a·s
#[inline(always)]
fn axpy_into(d: &mut [f64], a: &[f64], s: f64) {
    let s = f64x8::splat(s);
    for (d, a) in d.chunks_exact_mut(8).zip(lanes(a)) {
        a.mul_add(s, f64x8::from_slice(d)).copy_to_slice(d);
    }
}
/// Σ_i p_i·g_{b,i} for the `B` rows of `g` (row-minor, `p.len()` long)
/// at `gis`: `p` read once for all `B`, the sums in registers.
#[inline(always)]
fn gsums<const B: usize>(p: &[f64x8], g: &[f64x8], gis: &[usize], out: &mut [f64x8]) {
    let na = p.len();
    let rows: [&[f64x8]; B] = std::array::from_fn(|b| {
        &g[gis[b] * na..(gis[b] + 1) * na]
    });
    let mut acc = [f64x8::splat(0.0); B];
    for i in 0..na {
        let pv = unsafe { *p.get_unchecked(i) };
        for b in 0..B {
            acc[b] = pv.mul_add(unsafe { *rows[b].get_unchecked(i) }, acc[b]);
        }
    }
    out[..B].copy_from_slice(&acc);
}
/// Σ_i p_i and Σ_i p_i·g_{b,i} for the `B` rows of `g` at `gis`, with
/// p_i = `pf(i)` made on the pass, in registers.
#[inline(always)]
fn psums<const B: usize, F: Fn(usize) -> f64x8>(
    na: usize,
    pf: &F,
    g: &[f64x8],
    gis: &[usize],
    out: &mut [f64x8],
) -> f64x8 {
    let rows: [&[f64x8]; B] = std::array::from_fn(|b| {
        &g[gis[b] * na..(gis[b] + 1) * na]
    });
    let mut acc = [f64x8::splat(0.0); B];
    let mut s = f64x8::splat(0.0);
    for i in 0..na {
        let p = pf(i);
        s += p;
        for b in 0..B {
            acc[b] = p.mul_add(unsafe { *rows[b].get_unchecked(i) }, acc[b]);
        }
    }
    out[..B].copy_from_slice(&acc);
    s
}
/// [`psums`] over any number of rows, eight at a time; Σ_i p_i returned.
fn psum_many<F: Fn(usize) -> f64x8>(
    na: usize,
    pf: &F,
    g: &[f64x8],
    gis: &[usize],
    out: &mut Vec<f64x8>,
) -> f64x8 {
    out.clear();
    out.resize(gis.len(), f64x8::splat(0.0));
    let mut sum = None;
    let mut blocks = gis.chunks(8).zip(out.chunks_mut(8));
    if gis.is_empty() {
        return psums::<0, F>(na, pf, g, gis, &mut []);
    }
    for (gc, oc) in &mut blocks {
        let s = match gc.len() {
            1 => psums::<1, F>(na, pf, g, gc, oc),
            2 => psums::<2, F>(na, pf, g, gc, oc),
            3 => psums::<3, F>(na, pf, g, gc, oc),
            4 => psums::<4, F>(na, pf, g, gc, oc),
            5 => psums::<5, F>(na, pf, g, gc, oc),
            6 => psums::<6, F>(na, pf, g, gc, oc),
            7 => psums::<7, F>(na, pf, g, gc, oc),
            _ => psums::<8, F>(na, pf, g, gc, oc),
        };
        sum.get_or_insert(s);
    }
    sum.unwrap()
}
/// x^μ for a count μ ≤ 3.
#[inline(always)]
fn pow_mu(x: f64x8, mu: u8) -> f64x8 {
    match mu {
        0 => f64x8::splat(1.0),
        1 => x,
        2 => x * x,
        _ => x * x * x,
    }
}
/// Slots of [`add_route`]'s index, a power of two; past half full a routing
/// merges by a linear scan.
const RHASH: usize = 8192;
/// Routing entries up to which a deposit credits its bases in one pass.
const MK_PASS: usize = 24;
/// Rows a tile of the one-code-row node loop.
const ROW_TILE: usize = 32;
/// [`gsums`] over any number of rows, eight at a time.
fn gsum_many(p: &[f64x8], g: &[f64x8], gis: &[usize], out: &mut Vec<f64x8>) {
    out.clear();
    out.resize(gis.len(), f64x8::splat(0.0));
    for (gc, oc) in gis.chunks(8).zip(out.chunks_mut(8)) {
        match gc.len() {
            1 => gsums::<1>(p, g, gc, oc),
            2 => gsums::<2>(p, g, gc, oc),
            3 => gsums::<3>(p, g, gc, oc),
            4 => gsums::<4>(p, g, gc, oc),
            5 => gsums::<5>(p, g, gc, oc),
            6 => gsums::<6>(p, g, gc, oc),
            7 => gsums::<7>(p, g, gc, oc),
            _ => gsums::<8>(p, g, gc, oc),
        }
    }
}
#[inline(always)]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    use crate::simd::num::SimdFloat;
    let mut acc = f64x8::splat(0.0);
    for (a, b) in lanes(a).zip(lanes(b)) {
        acc = a.mul_add(b, acc);
    }
    acc.reduce_sum()
}
/// `TG_CHECK_SIX=1`: every fifth pick's context, pool and masks against
/// `Ctx::of_key` and `options_into` on its team.
static CHECK: OnceLock<bool> = OnceLock::new();
/// `TG_PICK_LOG=PATH`: a line per pick on the CPU's path — µs, classes,
/// rows, patterns, P bases, xy nodes, z points, fifth picks, the node loop's µs.
pub(super) static PICK_LOG: OnceLock<Mutex<std::io::BufWriter<std::fs::File>>> = OnceLock::new();
pub(super) fn open_pick_log() {
    if let Some(path) = std::env::var_os("TG_PICK_LOG") {
        let f = std::fs::File::create(path).unwrap();
        PICK_LOG
            .set(Mutex::new(std::io::BufWriter::new(f)))
            .unwrap_or_else(|_| unreachable!());
    }
}
pub(super) fn close_pick_log() {
    if let Some(log) = PICK_LOG.get() {
        use std::io::Write;
        log.lock().unwrap().flush().unwrap();
    }
}
/// A run-wide counter, bumped in a per-thread cell and summed into the
/// shared one when a pick ends, not a shared atomic bumped from every
/// thread a few hundred times a pick.
#[derive(Clone, Copy)]
pub(super) struct Stat(usize);
const NSTAT: usize = 84;
static STAT_SUM: [AtomicU64; NSTAT] = [const { AtomicU64::new(0) }; NSTAT];
thread_local! {
    static STAT_TL : [Cell < u64 >; NSTAT] = const { [const { Cell::new(0) }; NSTAT] };
}
impl Stat {
    fn fetch_add(self, v: u64, _: Ordering) {
        STAT_TL.with(|s| s[self.0].set(s[self.0].get() + v));
    }
    pub(super) fn load(self, o: Ordering) -> u64 {
        STAT_SUM[self.0].load(o)
    }
    pub(super) fn store(self, v: u64, o: Ordering) {
        STAT_SUM[self.0].store(v, o);
    }
}
fn stat_flush() {
    STAT_TL
        .with(|s| {
            for (c, sum) in s.iter().zip(&STAT_SUM) {
                let v = c.take();
                if v != 0 {
                    sum.fetch_add(v, Ordering::Relaxed);
                }
            }
        });
}
const fn stats<const N: usize>(at: usize) -> [Stat; N] {
    let mut a = [Stat(0); N];
    let mut i = 0;
    while i < N {
        a[i] = Stat(at + i);
        i += 1;
    }
    a
}
/// Nanoseconds in each section of [`pick`] (A–G), and picks, fifth picks,
/// patterns, xy nodes × z nodes, summed over the run; then picks and
/// nanoseconds by class count (1, 2–4, 5–16, 17–64, 65+).
pub(super) const SECTIONS: [Stat; 24] = stats(0);
fn bucket(classes: usize) -> usize {
    match classes {
        0..=1 => 0,
        2..=4 => 1,
        5..=16 => 2,
        17..=64 => 3,
        _ => 4,
    }
}
/// E's row loop and pattern loop, apart (ns).
const ESPLIT: [Stat; 2] = stats(24);
/// F's work: groups, cores, deposits, the fifth picks' own member clauses;
/// distinct mask rows among the groups.
const FCOUNT: [Stat; 5] = stats(26);
/// F's time (ns): group setup, cores, deposits, members' clauses, the
/// group's pick loop in all.
const FTIME: [Stat; 5] = stats(31);
/// Rows into the node loop, before and after merging alike ones; moment
/// entries the pattern loop visits, those of changed bases, pattern visits.
const ROWSTAT: [Stat; 5] = stats(36);
/// Situation flushes: count, species rows, codes, nonzero entries, nonzero
/// eight-entry lines.
const DENS: [Stat; 5] = stats(46);
/// Situation flushes: count, species rows, class rows, codes, row width.
const FLSTAT: [Stat; 5] = stats(41);
/// The GPU path: nanoseconds threads waited on batches, the picks those
/// ran, nanoseconds submitting them and packing them, and the picks left
/// to the CPU's node loop.
const GPUSTAT: [Stat; 5] = stats(51);
/// F's shape, under `TG_SIX_TIMES`: single-row picks, their codes (nk),
/// plain runs, routed sub-runs, routed fifth picks with a coefficient, their
/// map entries, and pick-local codes at the pick's end; the picks counted.
const FSHAPE: [Stat; 8] = stats(56);
/// D's time (ns), under `TG_SIX_TIMES`: the order and the fixed pool, the
/// groups' rows and shares, their fifth picks, the rules and profile after.
const DTIME: [Stat; 4] = stats(64);
/// `deposit1`'s pick end (ns), under `TG_SIX_TIMES`: the identity's
/// scalars expanded and the members' clause by pattern.
const FEND: [Stat; 3] = stats(75);
/// Multi-row picks, their codes, the rank of their rows' code weights,
/// their rows (`TG_SIX_TIMES`).
const RANK: [Stat; 4] = stats(80);
/// The GPU path's picks left to the CPU's node loop — several code rows,
/// past the kernel's bounds, the GPU's backlog — then each kind's node-loop
/// nanoseconds.
const CPUWHY: [Stat; 6] = stats(69);
/// Nanoseconds into a [`Stat`] until dropped, under `TG_SIX_TIMES` only:
/// E's and F's parts read the clock over a thousand times a pick.
struct Lap(Stat, Option<Instant>);
impl Lap {
    fn new(s: Stat) -> Lap {
        Lap(s, fine().then(Instant::now))
    }
}
/// `TG_SIX_TIMES`: the fine timers and the costlier counters on.
fn fine() -> bool {
    static FINE: OnceLock<bool> = OnceLock::new();
    *FINE.get_or_init(|| std::env::var_os("TG_SIX_TIMES").is_some())
}
impl Drop for Lap {
    fn drop(&mut self) {
        if let Some(t) = self.1 {
            self.0.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }
}
/// A pick's section timer: when it started, and the last lap.
pub(super) struct Clock {
    start: Instant,
    last: Instant,
}
impl Clock {
    pub(super) fn new() -> Clock {
        let now = Instant::now();
        Clock { start: now, last: now }
    }
    fn lap(&mut self, i: usize) {
        let now = Instant::now();
        SECTIONS[i].fetch_add((now - self.last).as_nanos() as u64, Ordering::Relaxed);
        self.last = now;
    }
}
pub(super) fn report() {
    let v: Vec<u64> = SECTIONS.iter().map(|a| a.load(Ordering::Relaxed)).collect();
    let total: u64 = v[..7].iter().sum();
    let picks = v[7].max(1) as f64;
    println!(
        "six::pick: {} picks, {} by rank-one terms, {} multi-row ones with rank-one code weights; E's node tables {:.1}, rows {:.1}, patterns {:.1} µs a pick",
        v[7], v[11], v[23], v[22] as f64 / picks / 1e3, ESPLIT[0].load(Ordering::Relaxed)
        as f64 / picks / 1e3, ESPLIT[1].load(Ordering::Relaxed) as f64 / picks / 1e3
    );
    println!(
        "six::pick sections (µs a pick): {}; {:.0} fifth picks, {:.1} patterns, {:.0} nodes a pick",
        ["A", "B", "C", "D", "E", "F", "G"].iter().enumerate().map(| (i, n) |
        format!("{n} {:.1}", v[i] as f64 / picks / 1e3)).collect::< Vec < _ >> ()
        .join(", ") + & format!(", total {:.1}", total as f64 / picks / 1e3), v[8] as f64
        / picks, v[9] as f64 / picks, v[10] as f64 / picks
    );
    let f: Vec<f64> = FCOUNT
        .iter()
        .map(|a| a.load(Ordering::Relaxed) as f64 / picks)
        .collect();
    println!(
        "six::pick F a pick: {:.1} groups ({:.1} distinct mask rows), {:.1} cores, {:.1} deposits, {:.1} own-member clauses",
        f[0], f[4], f[1], f[2], f[3]
    );
    let fs: Vec<f64> = FSHAPE.iter().map(|a| a.load(Ordering::Relaxed) as f64).collect();
    if fs[7] > 0.0 {
        println!(
            "six::pick F shape a pick: {:.1}% single-row, {:.1} codes, {:.1} plain runs, {:.1} routed sub-runs, {:.1} routed fifth picks with {:.1} map entries each, {:.1} pick-local codes",
            100.0 * fs[0] / fs[7], fs[1] / fs[7], fs[2] / fs[7], fs[3] / fs[7], fs[4] /
            fs[7], fs[5] / fs[4].max(1.0), fs[6] / fs[7]
        );
    }
    let ft: Vec<f64> = FTIME
        .iter()
        .map(|a| a.load(Ordering::Relaxed) as f64 / picks / 1e3)
        .collect();
    println!(
        "six::pick F µs a pick: setup {:.1}, cores {:.1}, deposits {:.1}, group ends {:.1}, pick loops {:.1}, pick end {:.1}",
        ft[0], ft[1], ft[2], ft[3], ft[4], FEND[0].load(Ordering::Relaxed) as f64 / picks
        / 1e3
    );
    let (gen_ns, gen_n) = (
        FEND[1].load(Ordering::Relaxed) as f64,
        FEND[2].load(Ordering::Relaxed),
    );
    println!(
        "six::pick F's general path (several code rows): {gen_n} picks, {:.0} thread-s, {:.0} µs a pick",
        gen_ns * 1e-9, gen_ns / gen_n.max(1) as f64 / 1e3
    );
    let rk: Vec<f64> = RANK.iter().map(|a| a.load(Ordering::Relaxed) as f64).collect();
    if rk[0] > 0.0 {
        println!(
            "six::pick multi-row picks: {:.0}, {:.2} codes, rank {:.2}, {:.1} rows each",
            rk[0], rk[1] / rk[0], rk[2] / rk[0], rk[3] / rk[0]
        );
    }
    let dt: Vec<f64> = DTIME
        .iter()
        .map(|a| a.load(Ordering::Relaxed) as f64 / picks / 1e3)
        .collect();
    println!(
        "six::pick D µs a pick: order and fixed pool {:.1}, groups' rows {:.1}, their fifth picks {:.1}, rules and profile {:.1}",
        dt[0], dt[1], dt[2], dt[3]
    );
    println!(
        "six::pick rows a pick: {:.1}, {:.1} after merging alike ones", ROWSTAT[0]
        .load(Ordering::Relaxed) as f64 / picks, ROWSTAT[1].load(Ordering::Relaxed) as
        f64 / picks
    );
    let fl = FLSTAT[0].load(Ordering::Relaxed).max(1) as f64;
    println!(
        "six flushes: {:.0}; per flush {:.0} species rows, {:.0} class rows, {:.0} codes, width {:.0}",
        fl, FLSTAT[1].load(Ordering::Relaxed) as f64 / fl, FLSTAT[2]
        .load(Ordering::Relaxed) as f64 / fl, FLSTAT[3].load(Ordering::Relaxed) as f64 /
        fl, FLSTAT[4].load(Ordering::Relaxed) as f64 / fl
    );
    let fl = DENS[0].load(Ordering::Relaxed).max(1) as f64;
    println!(
        "six flushes {:.0}: per flush {:.0} species rows × {:.0} codes, {:.0} nonzero, {:.0} nonzero lines",
        fl, DENS[1].load(Ordering::Relaxed) as f64 / fl, DENS[2].load(Ordering::Relaxed)
        as f64 / fl, DENS[3].load(Ordering::Relaxed) as f64 / fl, DENS[4]
        .load(Ordering::Relaxed) as f64 / fl
    );
    println!(
        "six::pick pattern loop a pick: {:.0} pattern visits, {:.0} moment entries ({:.0} of changed bases)",
        ROWSTAT[4].load(Ordering::Relaxed) as f64 / picks, ROWSTAT[2]
        .load(Ordering::Relaxed) as f64 / picks, ROWSTAT[3].load(Ordering::Relaxed) as
        f64 / picks
    );
    let all: u64 = v[17..22].iter().sum();
    println!(
        "six::pick by classes (picks, share of time, µs a pick): {}", ["1", "2-4",
        "5-16", "17-64", "65+"].iter().enumerate().map(| (i, n) |
        format!("{n}: {} {:.1}% {:.0}", v[12 + i], 100.0 * v[17 + i] as f64 / all.max(1)
        as f64, v[17 + i] as f64 / v[12 + i].max(1) as f64 / 1e3)).collect::< Vec < _ >>
        ().join(", ")
    );
    let gpu_picks = GPUSTAT[1].load(Ordering::Relaxed);
    if gpu_picks > 0 {
        let per = |i: usize| {
            GPUSTAT[i].load(Ordering::Relaxed) as f64 / gpu_picks as f64 / 1e3
        };
        println!(
            "six GPU node loop: {gpu_picks} picks ({:.1}% of all; {} on the CPU's), threads waited {:.1}, submitting {:.1}, packing {:.1} µs a GPU pick",
            100.0 * gpu_picks as f64 / picks, GPUSTAT[4].load(Ordering::Relaxed), per(0),
            per(2), per(3)
        );
        let w: Vec<u64> = CPUWHY.iter().map(|a| a.load(Ordering::Relaxed)).collect();
        println!(
            "six CPU node loops: {} with several code rows ({:.0} thread-s), {} past the kernel's bounds ({:.0}), {} for the backlog ({:.0})",
            w[0], w[3] as f64 * 1e-9, w[1], w[4] as f64 * 1e-9, w[2], w[5] as f64 * 1e-9
        );
    }
    #[cfg(feature = "gpu")]
    {
        let g = GPU_GAP.lock().unwrap();
        if g.2 > 0 {
            println!(
                "six GPU check over {} arrays, {} entries: worst relative gap {:.2e} ({})",
                g.2, g.3, g.0, g.1
            );
        }
    }
}
/// Slot 6 from one fourth pick of a situation — [`slot6_pick`]'s contract —
/// over its classes, or over `terms`, its rank-one form, when given.
#[allow(clippy::too_many_arguments)]
pub(super) fn pick<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    b: &Bits,
    s: &Situation<N>,
    p: &Pick5<N>,
    terms: Option<&[CpTerm]>,
    sc: &mut Scratch,
    x: &mut Six,
    quad: &mut Quad,
) {
    let mut clock = Clock::new();
    let bk = bucket(s.keys.len());
    struct Done(Instant, usize, [u32; 7], u64);
    impl Drop for Done {
        fn drop(&mut self) {
            let ns = self.0.elapsed().as_nanos() as u64;
            SECTIONS[17 + self.1].fetch_add(ns, Ordering::Relaxed);
            stat_flush();
            if let Some(log) = PICK_LOG.get() {
                use std::io::Write;
                let s = self.2;
                let _ = writeln!(
                    log.lock().unwrap(), "{} {} {} {} {} {} {} {} {}", ns / 1000, s[0],
                    s[1], s[2], s[3], s[4], s[5], s[6], self.3 / 1000
                );
            }
        }
    }
    let mut _done = Done(clock.start, bk, [s.keys.len() as u32, 0, 0, 0, 0, 0, 0], 0);
    let mut pp = std::mem::take(&mut x.prep);
    if prepare(t, ar, b, s, p, terms, sc, x, quad, &mut pp, &mut clock) {
        _done.2 = pp.shape();
        let te = Instant::now();
        node_loop(&pp, x, &mut clock);
        _done.3 = te.elapsed().as_nanos() as u64;
        let (a, mom) = (std::mem::take(&mut x.a), std::mem::take(&mut x.mom));
        deposit(t, b, &pp, &a, &mom, x, quad, &mut clock);
        (x.a, x.mom) = (a, mom);
    }
    x.prep = pp;
}
/// Sections A–D of [`pick`] into `pp`, its rows, bases, fifth picks and
/// rules; false when it has nothing to deposit.
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    b: &Bits,
    s: &Situation<N>,
    p: &Pick5<N>,
    terms: Option<&[CpTerm]>,
    sc: &mut Scratch,
    six: &mut Six,
    quad: &mut Quad,
    pp: &mut Prep,
    clock: &mut Clock,
) -> bool {
    six.ready(t);
    SECTIONS[7].fetch_add(1, Ordering::Relaxed);
    SECTIONS[12 + bucket(s.keys.len())].fetch_add(1, Ordering::Relaxed);
    let check = *CHECK.get_or_init(|| std::env::var_os("TG_CHECK_SIX").is_some());
    if terms.is_some() {
        SECTIONS[11].fetch_add(1, Ordering::Relaxed);
    }
    let nclass = s.keys.len();
    let tb4 = p.c4.tb;
    let tb_code = |tb: bool| if tb { TB_BIT } else { 0 };
    let nl3 = s.live3.len();
    let mut vary = vec![false; nl3];
    match terms {
        Some(ts) => {
            for &(i3, _) in &ts[0].coef {
                vary[i3] = true;
            }
        }
        None => {
            for i in 0..nl3 {
                vary[i] = s.rmin[i] < s.rmax[i];
            }
        }
    }
    let mut vix = vec![u8::MAX; nl3];
    let mut nv = 0usize;
    for i in 0..nl3 {
        if vary[i] {
            vix[i] = nv as u8;
            nv += 1;
        }
    }
    let base4_vary = p.at4_base.is_some_and(|i4| vary[i4]);
    let w4_fixed = match p.at4_base {
        _ if base4_vary => 1.0,
        Some(i4) => f64::from(s.rmax[i4]) / f64::from(p.den),
        None => f64::from(p.num) / f64::from(p.den),
    };
    let nrow = terms.map_or(nclass, |ts| ts.len());
    let mut rcf = vec![[0.0f64; 4]; nrow * nv];
    let mut rlin = vec![
        (0.0f64, 1.0f64, 0u8); if terms.is_some() { nrow * nv } else { 0 }
    ];
    let decos: Vec<Deco<N>> = match terms {
        Some(ts) => {
            let mut out: Vec<Deco<N>> = Vec::with_capacity(ts.len());
            let mut last: Option<(&Vec<(u16, f64)>, Deco<N>)> = None;
            for (ri, term) in ts.iter().enumerate() {
                assert_eq!(
                    term.coef.len(), nv, "rank-one terms with differing random bases"
                );
                for (&(i3, c), &l) in term.coef.iter().zip(&term.lin) {
                    rcf[ri * nv + usize::from(vix[i3])] = c;
                    rlin[ri * nv + usize::from(vix[i3])] = l;
                }
                let past = match &last {
                    Some((d, past)) if *d == &term.deco => past.clone(),
                    _ => {
                        let d: Deco<N> = term
                            .deco
                            .iter()
                            .map(|&(c, x)| (c, N::of_f64(x)))
                            .collect();
                        let past = kernel_part(
                            t,
                            ar,
                            &s.c3,
                            p.picked4,
                            p.lead_pick4,
                            p.flagged4,
                            tb4,
                            p.c4.fire,
                            d,
                        );
                        last = Some((&term.deco, past.clone()));
                        past
                    }
                };
                out.push(
                    past
                        .into_iter()
                        .map(|(c, x)| (c, N::of_f64(x.to_f64() * term.w)))
                        .collect(),
                );
            }
            out
        }
        None => {
            let nc = s.nc();
            (0..nclass)
                .map(|c| {
                    for i in 0..nl3 {
                        if vary[i] {
                            rcf[c * nv + usize::from(vix[i])][usize::from(s.r(c, i))] = 1.0;
                        }
                    }
                    let d: Deco<N> = (0..nc)
                        .filter(|&d| s.present[c * nc + d])
                        .map(|d| (s.codes[d], s.dense[c * nc + d]))
                        .collect();
                    kernel_part(
                        t,
                        ar,
                        &s.c3,
                        p.picked4,
                        p.lead_pick4,
                        p.flagged4,
                        tb4,
                        p.c4.fire,
                        d,
                    )
                })
                .collect()
        }
    };
    let mut kc: Vec<u16> = decos.iter().flat_map(|d| d.iter().map(|e| e.0)).collect();
    kc.sort_unstable();
    kc.dedup();
    let nk = kc.len();
    if nk == 0 {
        return false;
    }
    let mut wt = vec![0.0f64; nrow * nk];
    let mut live_row = vec![false; nrow];
    for (ri, d) in decos.iter().enumerate() {
        for &(code, v) in d {
            let k = kc.binary_search(&code).unwrap();
            wt[ri * nk + k] = w4_fixed * v.to_f64();
            live_row[ri] |= wt[ri * nk + k] != 0.0;
        }
    }
    let mut rdeg: Vec<u8> = s.rmax.clone();
    for i in 0..nl3 {
        if vary[i] {
            let v = usize::from(vix[i]);
            rdeg[i] = (0..4)
                .filter(|&r| (0..nrow).any(|ri| rcf[ri * nv + v][r] != 0.0))
                .max()
                .unwrap_or(0) as u8;
        }
    }
    clock.lap(0);
    let live4 = &p.live4;
    let nl4 = live4.len();
    let mut at4_of = [u8::MAX; 64];
    for (j, l) in live4.iter().enumerate() {
        at4_of[usize::from(l.mi)] = j as u8;
    }
    let mut f3 = s.f3;
    let mut f4 = p.f4;
    let (mut dx, mut dy) = (0usize, 0usize);
    let mut xb: Vec<XBase> = Vec::new();
    let mut pb: Vec<PBase> = Vec::new();
    let mut l4: Vec<Live4> = Vec::with_capacity(nl4);
    for l in live4.iter() {
        let i3 = usize::from(s.at3_of[usize::from(l.mi)]);
        let base = &t.bases[l.base as usize];
        let nzor = base
            .formes
            .iter()
            .enumerate()
            .filter(|&(_, &f)| !b.zor[f as usize])
            .fold(0u32, |m, (i, _)| m | 1 << i);
        let orb = base
            .formes
            .iter()
            .enumerate()
            .filter(|&(i, _)| l.mask >> i & 1 != 0)
            .fold(0u64, |o, (_, &f)| o | b.sp[f as usize]);
        l4.push(Live4 {
            i3,
            mask4: l.mask,
            n4: l.mask.count_ones(),
            base: l.base,
            p: false,
            r: 0,
            pi: 0,
            nzor,
            orb,
        });
    }
    for (i, l3) in s.live3.iter().enumerate() {
        let rmax = rdeg[i];
        let q3 = ratio(l3.mask, s.mask2[usize::from(l3.mi)]);
        let varying = vary[i];
        let j = at4_of[usize::from(l3.mi)];
        if l3.base == p.base4 || j == u8::MAX {
            if rmax == 0 {
                continue;
            }
            if !varying && q3 == 1.0 {
                f3 += u32::from(rmax);
            } else {
                xb.push(XBase {
                    i3: i,
                    q3,
                    role4: l3.base == p.base4,
                });
                dx += usize::from(rmax);
            }
            continue;
        }
        let j = usize::from(j);
        let q4 = ratio(live4[j].mask, l3.mask);
        if !varying && (rmax == 0 || (q3 == 1.0 && q4 == 1.0)) {
            l4[j].r = rmax;
            f3 += u32::from(rmax);
            f4 += u32::from(rmax);
        } else {
            l4[j].p = true;
            l4[j].pi = pb.len() as u8;
            pb.push(PBase { j: j as u8, i3: i, q3, q4 });
            dx += usize::from(rmax);
            dy += usize::from(rmax);
        }
    }
    let np = pb.len();
    let rmax_p: Vec<u8> = pb.iter().map(|q| rdeg[q.i3]).collect();
    let den4 = f64::from(p.den);
    clock.lap(1);
    options(t, &p.c4, &p.ctx4, &p.av4, sc);
    let caps = Caps::of_key(t, &p.c4);
    pp.sigs.clear();
    pp.maps.clear();
    let mut outs: Vec<(bool, u8, u32)> = Vec::new();
    let mut flat: Vec<(u8, u8, u16, f64)> = Vec::new();
    let mut kos_at = std::mem::take(&mut six.kos_at);
    kos_at.clear();
    for o5 in &sc.opts {
        let (_, picked5, lead_pick5, _) = pick_parts(t, &p.c4, o5.pick);
        if !(matches!(o5.pick, Pick::Forme(_)) && t.sp[picked5 as usize].flags) {
            continue;
        }
        assert!(! lead_pick5, "a fifth pick after a lead");
        for &code in &kc {
            let kos = kern_of(t, ar, picked5, code | tb_code(tb4));
            prefetch_bytes(kos.as_ptr().cast(), std::mem::size_of_val(kos));
            kos_at.push((kos.as_ptr() as usize, kos.len()));
        }
    }
    let mut ki = 0;
    for (oi, o5) in sc.opts.iter().enumerate() {
        let (class5, picked5, _, base5) = pick_parts(t, &p.c4, o5.pick);
        let sp5 = &t.sp[picked5 as usize];
        let flagged5 = matches!(o5.pick, Pick::Forme(_)) && sp5.flags;
        let j5 = if base5 != NONE && t.bases[base5 as usize].mi != u8::MAX {
            at4_of[usize::from(t.bases[base5 as usize].mi)]
        } else {
            u8::MAX
        };
        let sig = |ctx: u64, map: (u32, u32), tb: bool, fire: u8| Sig {
            ctx,
            o5: oi as u16,
            map,
            tb,
            fire,
            picked: picked5,
            class: class5,
            j5,
            num: o5.num,
            den: o5.den,
            pattern: 0,
            m0: 0,
            coef: 0.0,
            wch: (0, 0),
            group: 0,
        };
        if !flagged5 {
            pp.sigs.push(sig(caps.add(sp5, 0, tb4), (u32::MAX, 0), tb4, 0));
            continue;
        }
        outs.clear();
        flat.clear();
        for k in 0..kc.len() {
            let (at0, len) = kos_at[ki];
            ki += 1;
            let kos = unsafe {
                std::slice::from_raw_parts(at0 as *const (u16, bool, u8, f64), len)
            };
            for &(kcode, ktb, kfire, kp) in kos {
                let key = (tb4 || ktb, kfire);
                let at = match outs.iter().position(|o| (o.0, o.1) == key) {
                    Some(at) => at,
                    None => {
                        outs.push((key.0, key.1, 0));
                        outs.len() - 1
                    }
                };
                flat.push((at as u8, k as u8, kcode, kp));
            }
        }
        for (oi, o) in outs.iter().enumerate() {
            let lo = pp.maps.len() as u32;
            pp.maps
                .extend(
                    flat
                        .iter()
                        .filter(|e| usize::from(e.0) == oi)
                        .map(|e| (e.1, e.2, e.3)),
                );
            let hi = pp.maps.len() as u32;
            pp.sigs.push(sig(caps.add(sp5, o.1, o.0), (lo, hi), o.0, o.1));
        }
    }
    six.kos_at = kos_at;
    clock.lap(2);
    let d_lap = Lap::new(DTIME[0]);
    let ns = pp.sigs.len();
    pp.order.clear();
    pp.order.extend(0..ns as u32);
    {
        let sigs = &pp.sigs;
        pp.order.sort_unstable_by_key(|&i| sigs[i as usize].ctx);
    }
    let (mut fixed_bits, mut fixed_w): (Vec<u64>, Vec<u64>) = (Vec::new(), Vec::new());
    for &c in &t.simple_classes {
        let m0 = t.classes[c as usize].members[0];
        if p.ctx4.adm(&t.sp[m0 as usize]) && !b.zor[m0 as usize] {
            fixed_bits.push(b.sp[m0 as usize]);
            fixed_w.push(t.classes[c as usize].members.len() as u64);
        }
    }
    for &bs in &t.single_bases {
        let f = t.bases[bs as usize].formes[0];
        if p.ctx4.adm(&t.sp[f as usize]) && !b.zor[f as usize] {
            fixed_bits.push(b.sp[f as usize]);
            fixed_w.push(u64::from(t.bases[bs as usize].mu));
        }
    }
    let member = |c: u16| -> (u16, u32) {
        let class = &t.classes[c as usize];
        let rep = class.members[0];
        if class.simple {
            (rep, 1)
        } else {
            let base = &t.bases[t.sp[rep as usize].base as usize];
            (rep, if base.mi == u8::MAX { u32::from(base.mu) } else { 0 })
        }
    };
    let members4: Vec<(u16, u32)> = p.c4.members().iter().map(|&c| member(c)).collect();
    let mu4: Vec<u32> = l4
        .iter()
        .map(|l| u32::from(t.bases[l.base as usize].mu))
        .collect();
    pp.m5.clear();
    pp.wch.clear();
    pp.wex.clear();
    pp.groups.clear();
    pp.patterns.clear();
    let intern = |patterns: &mut Vec<Pattern>, changed: &[(u8, u32)], j5: u8| -> u16 {
        match patterns.iter().position(|q| q.j5 == j5 && q.changed == changed) {
            Some(pi) => pi as u16,
            None => {
                patterns
                    .push(Pattern {
                        changed: changed.to_vec(),
                        j5,
                    });
                (patterns.len() - 1) as u16
            }
        }
    };
    let mut changed_g: Vec<(u8, u32)> = Vec::new();
    let mut changed_j: Vec<(u8, u32)> = Vec::new();
    let mut jpat: Vec<(u8, u16, usize)> = Vec::new();
    let dz_of = |pat: &Pattern| -> usize {
        (0..np)
            .filter(|&q| {
                q as u8 != pat.j5
                    && pat
                        .changed
                        .iter()
                        .find(|c| usize::from(c.0) == q)
                        .is_none_or(|c| c.1 != 0)
            })
            .map(|q| usize::from(rmax_p[q]))
            .sum()
    };
    let mut lo = 0;
    let mut m_lo = u32::MAX;
    let mut degz = 0usize;
    let m5_base: Vec<u32> = l4.iter().map(|l| l.mask4 & l.nzor).collect();
    let mut fz: Vec<(u64, u8, u32)> = Vec::new();
    for (j, l) in l4.iter().enumerate() {
        for (i, &f) in t.bases[l.base as usize].formes.iter().enumerate() {
            if m5_base[j] >> i & 1 != 0 {
                fz.push((b.sp[f as usize], j as u8, 1 << i));
            }
        }
    }
    let always: Vec<u8> = (0..nl4)
        .filter(|&j| m5_base[j] != l4[j].mask4)
        .map(|j| j as u8)
        .collect();
    let mu_base: u32 = m5_base
        .iter()
        .zip(&mu4)
        .filter(|e| *e.0 != 0)
        .map(|e| *e.1)
        .sum();
    let delta_base: u32 = l4.iter().filter(|l| !l.p).map(|l| u32::from(l.r)).sum();
    let mut hit_j: Vec<u8> = Vec::new();
    drop(d_lap);
    while lo < ns {
        let rows_lap = Lap::new(DTIME[1]);
        let ctx5 = pp.sigs[pp.order[lo] as usize].ctx;
        let mut hi = lo;
        while hi < ns && pp.sigs[pp.order[hi] as usize].ctx == ctx5 {
            hi += 1;
        }
        let row = pp.m5.len();
        pp.m5.extend_from_slice(&m5_base);
        hit_j.clear();
        for &(s, j, bit) in &fz {
            if s & ctx5 != 0 {
                pp.m5[row + usize::from(j)] &= !bit;
                if hit_j.last() != Some(&j) {
                    hit_j.push(j);
                }
            }
        }
        let mut mu_all = mu_base;
        for &j in &hit_j {
            if pp.m5[row + usize::from(j)] == 0 {
                mu_all -= mu4[usize::from(j)];
            }
        }
        let f5raw = fixed_bits
            .iter()
            .zip(&fixed_w)
            .map(|(&s, &w)| if s & ctx5 == 0 { w } else { 0 })
            .sum::<u64>() as u32;
        let f5c4: u32 = members4
            .iter()
            .filter(|&&(rep, take)| take > 0 && b.adm5(ctx5, rep))
            .map(|e| e.1)
            .sum();
        let wlo = pp.wch.len() as u32;
        let xlo = pp.wex.len() as u32;
        let (mut delta_g, mut dr_g) = (delta_base, 0usize);
        changed_g.clear();
        let (mut ia, mut ih) = (0, 0);
        loop {
            let j = match (always.get(ia), hit_j.get(ih)) {
                (Some(&a), Some(&h)) if a == h => {
                    ia += 1;
                    ih += 1;
                    a
                }
                (Some(&a), Some(&h)) if a < h => {
                    ia += 1;
                    a
                }
                (_, Some(&h)) => {
                    ih += 1;
                    h
                }
                (Some(&a), None) => {
                    ia += 1;
                    a
                }
                (None, None) => break,
            };
            let l = &l4[usize::from(j)];
            let m5 = pp.m5[row + usize::from(j)];
            if l.p {
                changed_g.push((l.pi, m5));
            } else {
                delta_g -= u32::from(l.r);
                pp.wex.push(j);
                if m5 != 0 && l.r > 0 {
                    pp.wch.push((j, f64::from(m5.count_ones()) / f64::from(l.n4)));
                    dr_g += usize::from(l.r);
                }
            }
        }
        if check {
            let (mut mu, mut delta, mut dr, mut wex) = (0, 0, 0, Vec::new());
            for (j, l) in l4.iter().enumerate() {
                let mut m = l.mask4 & l.nzor;
                for (i, &f) in t.bases[l.base as usize].formes.iter().enumerate() {
                    if m >> i & 1 != 0 && b.sp[f as usize] & ctx5 != 0 {
                        m &= !(1 << i);
                    }
                }
                assert_eq!(pp.m5[row + j], m, "a group's mask row");
                mu += if m != 0 { mu4[j] } else { 0 };
                if !l.p && m == l.mask4 {
                    delta += u32::from(l.r);
                } else if !l.p {
                    wex.push(j as u8);
                    dr += if m != 0 { usize::from(l.r) } else { 0 };
                }
            }
            assert_eq!(
                (mu, delta, dr, & wex[..]), (mu_all, delta_g, dr_g, & pp.wex[xlo as usize
                ..]), "a group's row sums"
            );
        }
        let whi = pp.wch.len() as u32;
        let pat_g = intern(&mut pp.patterns, &changed_g, u8::MAX);
        let dz_g = dz_of(&pp.patterns[usize::from(pat_g)]);
        jpat.clear();
        pp.groups
            .push(Group {
                lo: lo as u32,
                hi: hi as u32,
                row: row as u32,
                wch: (wlo, whi),
                wex: (xlo, pp.wex.len() as u32),
                pattern: pat_g,
                m_min: u32::MAX,
                m_max: 0,
                deg: 0,
            });
        drop(rows_lap);
        let _picks_lap = Lap::new(DTIME[2]);
        for &oi in &pp.order[lo..hi] {
            let sg = pp.sigs[oi as usize];
            let (rep5, take5) = member(sg.class);
            let f5 = f5raw - f5c4
                - if take5 > 0 && b.adm5(ctx5, rep5) { take5 } else { 0 };
            let j5 = sg.j5;
            let mut mu5 = mu_all;
            if j5 != u8::MAX && pp.m5[row + usize::from(j5)] != 0 {
                mu5 -= u32::from(t.bases[l4[usize::from(j5)].base as usize].mu);
            }
            let num5 = match j5 {
                u8::MAX => sg.num,
                j5 if !l4[usize::from(j5)].p => u32::from(l4[usize::from(j5)].r),
                _ => 1,
            };
            let sigm = &mut pp.sigs[oi as usize];
            sigm.group = (pp.groups.len() - 1) as u32;
            if f5 + mu5 == 0 || num5 == 0 {
                sigm.coef = 0.0;
                continue;
            }
            let (mut delta, mut dr, mut wch, mut pi, mut dz) = (
                delta_g,
                dr_g,
                (wlo, whi),
                pat_g,
                dz_g,
            );
            if j5 != u8::MAX {
                let l = &l4[usize::from(j5)];
                let m5 = pp.m5[row + usize::from(j5)];
                if l.p {
                    match jpat.iter().find(|e| e.0 == l.pi) {
                        Some(&(_, p, d)) => (pi, dz) = (p, d),
                        None => {
                            changed_j.clear();
                            changed_j
                                .extend(changed_g.iter().copied().filter(|c| c.0 != l.pi));
                            pi = intern(&mut pp.patterns, &changed_j, l.pi);
                            dz = dz_of(&pp.patterns[usize::from(pi)]);
                            jpat.push((l.pi, pi, dz));
                        }
                    }
                } else if m5 == l.mask4 {
                    delta -= u32::from(l.r);
                } else if m5 != 0 && l.r > 0 {
                    let at = pp.wch.len() as u32;
                    for e in wlo..whi {
                        let w = pp.wch[e as usize];
                        if w.0 != j5 {
                            pp.wch.push(w);
                        }
                    }
                    wch = (at, pp.wch.len() as u32);
                    dr -= usize::from(l.r);
                }
            }
            let sigm = &mut pp.sigs[oi as usize];
            sigm.pattern = pi;
            sigm.m0 = f5 + delta;
            sigm.coef = f64::from(num5) / f64::from(sg.den);
            sigm.wch = wch;
            let m0 = sigm.m0;
            m_lo = m_lo.min(m0);
            degz = degz.max(dz);
            let gr = pp.groups.last_mut().unwrap();
            gr.m_min = gr.m_min.min(m0);
            gr.m_max = gr.m_max.max(m0);
            gr.deg = gr.deg.max((dz + dr) as u32);
            if check {
                check_sig(t, p, &sg, ctx5, f5, &pp.m5[row..row + nl4]);
            }
        }
        lo = hi;
    }
    if m_lo == u32::MAX {
        return false;
    }
    if fine() {
        let mut rows: Vec<&[u32]> = pp
            .groups
            .iter()
            .map(|g| &pp.m5[g.row as usize..g.row as usize + nl4])
            .collect();
        rows.sort_unstable();
        rows.dedup();
        FCOUNT[4].fetch_add(rows.len() as u64, Ordering::Relaxed);
    }
    assert!(m_lo >= 1, "a sixth pick from an empty fixed pool");
    let end_lap = Lap::new(DTIME[3]);
    let nz = degz + 1;
    let nzp = nz.div_ceil(8) * 8;
    let mut zv = vec![0.5f64; nzp];
    let mut bary = vec![0.0f64; nz];
    for c in 0..nz {
        let th = std::f64::consts::PI * (2 * c + 1) as f64 / (2 * nz) as f64;
        zv[c] = 0.5 - 0.5 * th.cos();
        bary[c] = if c % 2 == 0 { 1.0 } else { -1.0 } * th.sin();
    }
    let (xs, wx) = quad.rule(f3, dx / 2 + 1).clone();
    let (ys, wy) = quad.rule(f4, dy / 2 + 1).clone();
    let mut prof = vec![0.0f64; nk];
    let mut srow = vec![0.0f64; nrow];
    let single = {
        let norm = |ri: usize| {
            wt[ri * nk..(ri + 1) * nk].iter().fold(0.0f64, |m, &v| m.max(v.abs()))
        };
        let r0 = (0..nrow).max_by(|&a, &b| norm(a).total_cmp(&norm(b))).unwrap();
        let km = (0..nk)
            .max_by(|&a, &b| wt[r0 * nk + a].abs().total_cmp(&wt[r0 * nk + b].abs()))
            .unwrap();
        prof.copy_from_slice(&wt[r0 * nk..(r0 + 1) * nk]);
        let pm = prof[km];
        if pm == 0.0 {
            false
        } else {
            (0..nrow)
                .all(|ri| {
                    let s = wt[ri * nk + km] / pm;
                    srow[ri] = s;
                    let tol = 1e-14 * norm(ri);
                    (0..nk).all(|k| (wt[ri * nk + k] - s * prof[k]).abs() <= tol)
                })
        }
    };
    if nrow > 1 {
        SECTIONS[if single { 23 } else { 16 + 7 }]
            .fetch_add(u64::from(single), Ordering::Relaxed);
    }
    let (mut nb, mut pbas, mut ub) = (
        0usize,
        std::mem::take(&mut pp.pbas),
        std::mem::take(&mut pp.ub),
    );
    pbas.clear();
    ub.clear();
    if !single && nk > 1 {
        let row = |ri: usize| &wt[ri * nk..(ri + 1) * nk];
        let dotk = |a: &[f64], b: &[f64]| {
            a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>()
        };
        let mut c = vec![0.0f64; nk];
        for ri in 0..nrow {
            c.copy_from_slice(row(ri));
            let n0 = dotk(&c, &c).sqrt();
            if n0 == 0.0 {
                continue;
            }
            for _ in 0..2 {
                for b in 0..pbas.len() / nk {
                    let d = dotk(&c, &pbas[b * nk..(b + 1) * nk]);
                    for (x, &y) in c.iter_mut().zip(&pbas[b * nk..(b + 1) * nk]) {
                        *x -= d * y;
                    }
                }
            }
            let n1 = dotk(&c, &c).sqrt();
            if n1 > 1e-9 * n0 {
                pbas.extend(c.iter().map(|&v| v / n1));
            }
        }
        let nbas = pbas.len() / nk;
        if nbas < nk {
            ub.extend(
                (0..nrow)
                    .flat_map(|ri| (0..nbas).map(move |b| (ri, b)))
                    .map(|(ri, b)| dotk(row(ri), &pbas[b * nk..(b + 1) * nk])),
            );
            let rebuilt = (0..nrow)
                .all(|ri| {
                    let size = row(ri).iter().fold(0.0f64, |m, &v| m.max(v.abs()));
                    (0..nk)
                        .all(|k| {
                            ((0..nbas)
                                .map(|b| ub[ri * nbas + b] * pbas[b * nk + k])
                                .sum::<f64>() - row(ri)[k])
                                .abs() <= 1e-14 * size
                        })
                });
            if rebuilt {
                nb = nbas;
            }
        }
        if fine() {
            RANK[0].fetch_add(1, Ordering::Relaxed);
            RANK[1].fetch_add(nk as u64, Ordering::Relaxed);
            RANK[2].fetch_add(nb as u64, Ordering::Relaxed);
            RANK[3].fetch_add(nrow as u64, Ordering::Relaxed);
        }
    }
    drop(end_lap);
    clock.lap(3);
    if let Some(ms) = *SLOW
        .get_or_init(|| std::env::var("TG_SIX_SLOW").ok().map(|v| v.parse().unwrap()))
    {
        let t_d = clock.start.elapsed().as_millis();
        if t_d >= ms {
            eprintln!(
                "slow pick setup {t_d} ms: classes {nclass}, fifth picks {ns}, groups {}, patterns {}, P bases {np}, x-only {}, codes {nk}, nodes {}x{}x{nz}, m_lo {m_lo}",
                pp.groups.len(), pp.patterns.len(), xb.len(), xs.len(), ys.len()
            );
        }
    }
    SECTIONS[8].fetch_add(ns as u64, Ordering::Relaxed);
    SECTIONS[9].fetch_add(pp.patterns.len() as u64, Ordering::Relaxed);
    SECTIONS[10].fetch_add((xs.len() * ys.len() * nz) as u64, Ordering::Relaxed);
    pp.nclass = nclass;
    pp.nrow = nrow;
    pp.nv = nv;
    pp.vary = vary;
    pp.vix = vix;
    pp.rcf = rcf;
    pp.rlin = rlin;
    pp.class_rows = terms.is_none();
    pp.kc = kc;
    pp.wt = wt;
    pp.live_row = live_row;
    pp.rdeg = rdeg;
    pp.xb = xb;
    pp.pb = pb;
    pp.l4 = l4;
    pp.rmax_p = rmax_p;
    pp.den4 = den4;
    pp.members4.clear();
    pp.members4.extend_from_slice(p.c4.members());
    pp.nz = nz;
    pp.zv = zv;
    pp.bary = bary;
    (pp.xs, pp.wx, pp.ys, pp.wy) = (xs, wx, ys, wy);
    pp.single = single;
    pp.prof = prof;
    pp.srow = srow;
    (pp.nb, pp.pbas, pp.ub) = (nb, pbas, ub);
    true
}
/// `TG_SIX_SLOW=MS`: picks whose setup or node loop passes MS milliseconds.
static SLOW: OnceLock<Option<u128>> = OnceLock::new();
impl Prep {
    /// Row `ri`'s x-only factors at `xx`: role 4's r/den when it varies.
    #[inline(always)]
    fn x_scale(&self, ri: usize, xx: f64) -> f64 {
        if !self.live_row[ri] {
            return 0.0;
        }
        let mut xsc = 1.0;
        for xq in &self.xb {
            let e = (1.0 - xq.q3) + xq.q3 * xx;
            let at_r = |r: usize| -> f64 {
                if xq.role4 {
                    if r == 0 {
                        0.0
                    } else {
                        let w = if self.vary[xq.i3] {
                            r as f64 / self.den4
                        } else {
                            1.0
                        };
                        w * xq.q3 * xx * e.powi(r as i32 - 1)
                    }
                } else {
                    e.powi(r as i32)
                }
            };
            xsc
                *= if self.vary[xq.i3] {
                    let c = &self.rcf[ri * self.nv + usize::from(self.vix[xq.i3])];
                    (0..4)
                        .map(|r| if c[r] == 0.0 { 0.0 } else { c[r] * at_r(r) })
                        .sum::<f64>()
                } else {
                    at_r(usize::from(self.rdeg[xq.i3]))
                };
        }
        xsc
    }
    /// The pick log's shape columns: classes, rows, patterns, P bases, xy
    /// nodes, z points, fifth picks.
    fn shape(&self) -> [u32; 7] {
        [
            self.nclass,
            self.nrow,
            self.patterns.len(),
            self.pb.len(),
            self.xs.len() * self.ys.len(),
            self.nz,
            self.sigs.len(),
        ]
            .map(|v| v as u32)
    }
}
/// The GPU node loop's bounds (`cuda/six.cu`): the P bases a pick
/// multiplies, and the varying ones among them.
#[cfg(feature = "gpu")]
const GPU_NP: usize = 32;
#[cfg(feature = "gpu")]
const GPU_NVQ: usize = 16;
/// A block's threads and shared memory.
#[cfg(feature = "gpu")]
const GPU_THREADS: usize = 128;
#[cfg(feature = "gpu")]
const GPU_SMEM: usize = 48 << 10;
/// Rank-one rows a block tabulates at once at the least (`kMinRowTile`):
/// tg_six shrinks a pick's tile to fit, so it fits if this does.
#[cfg(feature = "gpu")]
const GPU_ROW_TILE: usize = 4;
/// `pp`'s node loop laid out in `g` for the GPU, as [`node_loop`] computes
/// it; its index there, or None for the CPU's loop — several code rows, past
/// the kernel's bounds, or no weight at all.
#[cfg(feature = "gpu")]
fn gpu_pack(pp: &Prep, g: &mut gpu::SixBatch) -> Option<usize> {
    use gpu::SixBatch as B;
    const LIVE: u16 = 0x1000;
    const J5: u16 = 0x2000;
    let np = pp.pb.len();
    if np > GPU_NP {
        return None;
    }
    let fixed = |q: usize| !pp.vary[pp.pb[q].i3];
    let vq: Vec<usize> = (0..np).filter(|&q| !fixed(q)).collect();
    let nvq = vq.len();
    if nvq > GPU_NVQ {
        return None;
    }
    let order: Vec<usize> = vq
        .iter()
        .copied()
        .chain((0..np).filter(|&q| fixed(q)))
        .collect();
    let mut dev_of = [0u16; GPU_NP];
    for (d, &q) in order.iter().enumerate() {
        dev_of[q] = d as u16;
    }
    let (nx, ny, nz, nzp, npat, nv) = (
        pp.xs.len(),
        pp.ys.len(),
        pp.nz,
        pp.nzp(),
        pp.patterns.len(),
        pp.nv,
    );
    if npat > GPU_THREADS {
        return None;
    }
    let mut chg: Vec<(usize, u32)> = pp
        .patterns
        .iter()
        .flat_map(|pat| pat.changed.iter().map(|&(q, m5)| (usize::from(q), m5)))
        .collect();
    chg.sort_unstable_by_key(|&(q, m5)| (fixed(q), q, m5));
    chg.dedup();
    let ncv = chg.iter().filter(|&&(q, _)| !fixed(q)).count();
    let mut slots: Vec<u16> = Vec::with_capacity(npat * np);
    for pat in &pp.patterns {
        for &q in &order {
            let weighs = !fixed(q) || pp.rmax_p[q] > 0;
            slots
                .push(
                    if usize::from(pat.j5) == q {
                        J5 | dev_of[q]
                    } else if let Some(&(_, m5)) = pat
                        .changed
                        .iter()
                        .find(|c| usize::from(c.0) == q)
                    {
                        let c = chg
                            .binary_search_by_key(
                                &(fixed(q), q, m5),
                                |&(q, m5)| (fixed(q), q, m5),
                            )
                            .unwrap();
                        (np + c) as u16 | if m5 != 0 && weighs { LIVE } else { 0 }
                    } else {
                        dev_of[q] | if weighs { LIVE } else { 0 }
                    },
                );
        }
    }
    let mut keys: Vec<&[u16]> = Vec::new();
    let pkey: Vec<u16> = (0..npat)
        .map(|pi| {
            let v = &slots[pi * np..pi * np + nvq];
            (keys
                .iter()
                .position(|k| *k == v)
                .unwrap_or_else(|| {
                    keys.push(v);
                    keys.len() - 1
                })) as u16
        })
        .collect();
    let nkeys = keys.len();
    {
        let bytes = |zl: usize| {
            let parts = (GPU_THREADS / (nkeys * zl)).clamp(1, 4);
            let ncol = if pp.class_rows { 0 } else { (nvq + ncv) * zl };
            let rows = if pp.class_rows { 0 } else { GPU_ROW_TILE * (ncol * 2 + nvq) };
            let words = (np + chg.len()) * zl * 4 + np * 2 + zl
                + parts * nkeys * zl * (1 + nvq) + rows;
            8 * words + 4 * (ncol + np) + 2 * (npat * np + nkeys * nvq + npat)
        };
        let nblk = nz.div_ceil(GPU_THREADS / npat);
        let zl = nz.div_ceil(nblk);
        if bytes(zl).max(bytes(nz - (nblk - 1) * zl)) > GPU_SMEM {
            return None;
        }
    }
    let coef = |ri: usize, q: usize| &pp.rcf[ri * nv + usize::from(pp.vix[pp.pb[q].i3])];
    let sig_cmp = |a: usize, b: usize| {
        vq.iter()
            .flat_map(|&q| coef(a, q).iter().zip(coef(b, q)))
            .map(|(x, y)| x.to_bits().cmp(&y.to_bits()))
            .find(|o| o.is_ne())
            .unwrap_or(std::cmp::Ordering::Equal)
    };
    let mut by_sig: Vec<usize> = (0..pp.nrow).collect();
    by_sig.sort_by(|&a, &b| sig_cmp(a, b));
    let (mut mrow, mut mrep) = (vec![0usize; pp.nrow], Vec::new());
    for (i, &ri) in by_sig.iter().enumerate() {
        if i == 0 || sig_cmp(by_sig[i - 1], ri).is_ne() {
            mrep.push(ri);
        }
        mrow[ri] = mrep.len() - 1;
    }
    let nrow = mrep.len();
    let nkk = pp.nkn();
    let mut wm = vec![0.0f64; nkk * nx * nrow];
    for k4 in 0..nkk {
        for (ix, &xx) in pp.xs.iter().enumerate() {
            for ri in 0..pp.nrow {
                wm[(k4 * nx + ix) * nrow + mrow[ri]]
                    += pp.x_scale(ri, xx) * pp.row_weight(ri, k4);
            }
        }
    }
    let scales: Vec<f64> = wm
        .chunks(nx * nrow)
        .map(|w| {
            let wmax = w.iter().fold(0.0f64, |m, &w| m.max(w.abs()));
            if wmax == 0.0 { 1.0 } else { 2f64.powi(wmax.log2().floor() as i32) }
        })
        .collect();
    if wm.iter().all(|&w| w == 0.0) {
        return None;
    }
    let keys_at = g.shorts.len() as i64;
    for k in &keys {
        g.shorts.extend_from_slice(k);
    }
    g.shorts.extend_from_slice(&pkey);
    let pslot = g.shorts.len() as i64;
    g.shorts.extend_from_slice(&slots);
    let cq = g.shorts.len() as i64;
    g.shorts.extend(chg.iter().map(|&(q, _)| dev_of[q]));
    let cb = g.words.len() as i64;
    for &xx in &pp.xs {
        for &q in &order {
            let pbq = &pp.pb[q];
            g.words.push(B::real((1.0 - pbq.q3) + pbq.q3 * xx * (1.0 - pbq.q4)));
            g.words.push(B::real(pbq.q3 * pbq.q4 * xx));
        }
    }
    g.words.extend(pp.ys.iter().map(|&yy| B::real(yy)));
    let wn = g.words.len() as i64;
    for &vx in &pp.wx {
        g.words.extend(pp.wy.iter().map(|&vy| B::real(vx * vy)));
    }
    let wmo = g.words.len() as i64;
    for (w, &scale) in wm.chunks(nx * nrow).zip(&scales) {
        g.words.extend(w.iter().map(|&w| B::real(w / scale)));
    }
    let zv = g.words.len() as i64;
    g.words.extend(pp.zv[..nz].iter().map(|&z| B::real(z)));
    let q5 = g.words.len() as i64;
    for _ in 0..np {
        g.words.extend([B::real(1.0), B::real(0.0)]);
    }
    for &(q, m5) in &chg {
        let q5 = f64::from(m5.count_ones())
            / f64::from(pp.l4[usize::from(pp.pb[q].j)].n4);
        g.words.extend([B::real(q5), B::real(1.0 - q5)]);
    }
    let (lin, mu) = (g.words.len() as i64, g.mus.len() as i64);
    for &ri in &mrep {
        let mut m = 0u64;
        for (v, &q) in vq.iter().enumerate() {
            let k = if pp.class_rows {
                let c = coef(ri, q);
                (0..4).find(|&r| c[r] != 0.0).unwrap_or(0) as u64
            } else {
                let (al, be, mu) = pp.rlin[ri * nv + usize::from(pp.vix[pp.pb[q].i3])];
                g.words.extend([B::real(al), B::real(be)]);
                u64::from(mu)
            };
            m |= k << (4 * v);
        }
        g.mus.push(m);
    }
    let bases = g.bases.len() as i64;
    g.bases
        .extend(
            order
                .iter()
                .map(|&q| gpu::TgSixBase {
                    rmax: if fixed(q) { i32::from(pp.rmax_p[q]) } else { 0 },
                    cpu: q as i32,
                }),
        );
    let first = g.picks.len();
    for (k4, &scale) in scales.iter().enumerate() {
        let out = g.alloc_out(npat * nzp * (1 + np));
        g.picks
            .push(gpu::TgSixPick {
                nx: nx as i32,
                ny: ny as i32,
                nz: nz as i32,
                nzp: nzp as i32,
                np: np as i32,
                nvq: nvq as i32,
                nrow: nrow as i32,
                nslot: (np + chg.len()) as i32,
                npat: npat as i32,
                class_rows: i32::from(pp.class_rows),
                ncv: ncv as i32,
                nkeys: nkeys as i32,
                cq,
                keys: keys_at,
                cb,
                wn,
                wm: wmo + (k4 * nx * nrow) as i64,
                zv,
                q5,
                lin,
                mu,
                bases,
                pslot,
                out,
                scale,
                ..Default::default()
            });
    }
    Some(first)
}
/// Section E of [`pick`] on the CPU: `pp`'s node loop into `x.a` and
/// `x.mom`.
fn node_loop(pp: &Prep, x: &mut Six, clock: &mut Clock) {
    let Prep {
        nrow,
        nv,
        class_rows,
        nz,
        ref vary,
        ref vix,
        ref rcf,
        ref rlin,
        ref kc,
        ref pb,
        ref l4,
        ref rmax_p,
        ref zv,
        ref xs,
        ref wx,
        ref ys,
        ref wy,
        ..
    } = *pp;
    let (nk, nkk, nzp, np) = (kc.len(), pp.nkn(), pp.nzp(), pb.len());
    let npat = pp.patterns.len();
    x.a.clear();
    x.a.resize(npat * nkk * nzp, 0.0);
    x.mom.clear();
    x.mom.resize(npat * np * nkk * nzp, 0.0);
    let mut q5p = vec![1.0f64; npat * np];
    let mut live_p = vec![true; npat * np];
    for (pi, pat) in pp.patterns.iter().enumerate() {
        for &(q, m5) in &pat.changed {
            let l = &l4[usize::from(pb[usize::from(q)].j)];
            q5p[pi * np + usize::from(q)] = f64::from(m5.count_ones()) / f64::from(l.n4);
            live_p[pi * np + usize::from(q)] = m5 != 0;
        }
        if pat.j5 != u8::MAX {
            live_p[pi * np + usize::from(pat.j5)] = false;
        }
    }
    let mut chg: Vec<(usize, u32, f64)> = Vec::new();
    let mut pat_ch: Vec<Vec<usize>> = Vec::with_capacity(npat);
    for pat in pp.patterns.iter() {
        let mut ids = Vec::with_capacity(pat.changed.len());
        for &(q, m5) in &pat.changed {
            let q = usize::from(q);
            let id = match chg.iter().position(|c| c.0 == q && c.1 == m5) {
                Some(i) => i,
                None => {
                    let l = &l4[usize::from(pb[q].j)];
                    chg.push((q, m5, f64::from(m5.count_ones()) / f64::from(l.n4)));
                    chg.len() - 1
                }
            };
            ids.push(id);
        }
        pat_ch.push(ids);
    }
    let nchg = chg.len();
    let nxy = xs.len() * ys.len();
    let stride = (2 * np + 3 * nchg) * nzp;
    let c_e = |c: usize| (2 * np + 3 * c) * nzp;
    let c_rl = |c: usize| (2 * np + 3 * c + 1) * nzp;
    let c_rho = |c: usize| (2 * np + 3 * c + 2) * nzp;
    let mut tab = std::mem::take(&mut x.tab);
    if tab.len() < stride {
        tab.resize(stride, 0.0);
    }
    let mut inv0 = vec![f64x8::splat(0.0); np * (nzp / 8)];
    let mut node_w = vec![0.0f64; nxy];
    let mut node_cb = vec![(0.0f64, 0.0f64); nxy * np];
    for (ix, (&xx, &vx)) in xs.iter().zip(wx).enumerate() {
        for (iy, (&yy, &vy)) in ys.iter().zip(wy).enumerate() {
            let n = ix * ys.len() + iy;
            node_w[n] = vx * vy;
            for (q, pbq) in pb.iter().enumerate() {
                let cc = (1.0 - pbq.q3) + pbq.q3 * xx * (1.0 - pbq.q4);
                let bb = pbq.q3 * pbq.q4 * xx * yy;
                node_cb[n * np + q] = (cc, bb);
            }
        }
    }
    {
        let now = Instant::now();
        SECTIONS[22].fetch_add((now - clock.last).as_nanos() as u64, Ordering::Relaxed);
    }
    let vq: Vec<usize> = (0..np).filter(|&q| vary[pb[q].i3]).collect();
    let mut vq_of = vec![usize::MAX; np];
    for (v, &q) in vq.iter().enumerate() {
        vq_of[q] = v;
    }
    let nvq = vq.len();
    let touch: Vec<bool> = pp
        .patterns
        .iter()
        .map(|pat| {
            pat.changed.iter().any(|c| vq_of[usize::from(c.0)] != usize::MAX)
                || (pat.j5 != u8::MAX && vq_of[usize::from(pat.j5)] != usize::MAX)
        })
        .collect();
    let mut touch_at = vec![usize::MAX; npat];
    let mut keys: Vec<usize> = Vec::new();
    {
        let vkey = |pi: usize| -> (Vec<(usize, u32)>, u8) {
            let pat = &pp.patterns[pi];
            let ch: Vec<(usize, u32)> = pat
                .changed
                .iter()
                .filter(|c| vq_of[usize::from(c.0)] != usize::MAX)
                .map(|c| (usize::from(c.0), c.1))
                .collect();
            let j5v = if pat.j5 != u8::MAX && vq_of[usize::from(pat.j5)] != usize::MAX {
                pat.j5
            } else {
                u8::MAX
            };
            (ch, j5v)
        };
        for pi in 0..npat {
            if !touch[pi] {
                continue;
            }
            let kp = vkey(pi);
            touch_at[pi] = match keys.iter().position(|&k| vkey(k) == kp) {
                Some(at) => at,
                None => {
                    keys.push(pi);
                    keys.len() - 1
                }
            };
        }
    }
    let ntouch = keys.len();
    let mut cslot = vec![usize::MAX; nchg];
    let mut cv: Vec<usize> = Vec::new();
    for (c, &(q, ..)) in chg.iter().enumerate() {
        if vq_of[q] != usize::MAX {
            cslot[c] = cv.len();
            cv.push(c);
        }
    }
    let mut pat_v = vec![usize::MAX; npat * nvq.max(1)];
    for pi in 0..npat {
        for &c in &pat_ch[pi] {
            let v = vq_of[chg[c].0];
            if v != usize::MAX {
                pat_v[pi * nvq + v] = c;
            }
        }
    }
    let ncv = cv.len();
    let mut key_ch: Vec<usize> = Vec::new();
    let mut kstart = vec![0usize; keys.len() + 1];
    let (mut key_goff, mut key_gi): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
    let mut gstart = vec![0usize; keys.len() + 1];
    let mut key_j5v: Vec<Option<usize>> = vec![None; keys.len()];
    for (key, &pi) in keys.iter().enumerate() {
        let pat = &pp.patterns[pi];
        for v in 0..nvq {
            let c = pat_v[pi * nvq + v];
            if c != usize::MAX {
                key_ch.push(cslot[c]);
            }
            if live_p[pi * np + vq[v]] {
                key_goff.push((1 + v) * nkk * nzp);
                key_gi.push(if c == usize::MAX { v } else { nvq + cslot[c] });
            }
        }
        kstart[key + 1] = key_ch.len();
        gstart[key + 1] = key_gi.len();
        if pat.j5 != u8::MAX && vq_of[usize::from(pat.j5)] != usize::MAX {
            key_j5v[key] = Some(vq_of[usize::from(pat.j5)]);
        }
    }
    let mut korder: Vec<usize> = (0..keys.len()).collect();
    korder
        .sort_by(|&a, &b| {
            key_ch[kstart[a]..kstart[a + 1]].cmp(&key_ch[kstart[b]..kstart[b + 1]])
        });
    let klcp: Vec<usize> = (0..korder.len())
        .map(|oi| {
            if oi == 0 {
                return 0;
            }
            let (a, b) = (korder[oi - 1], korder[oi]);
            key_ch[kstart[a]..kstart[a + 1]]
                .iter()
                .zip(&key_ch[kstart[b]..kstart[b + 1]])
                .take_while(|(x, y)| x == y)
                .count()
        })
        .collect();
    let kdepth = (0..keys.len()).map(|k| kstart[k + 1] - kstart[k]).max().unwrap_or(0);
    let mut lvl: Vec<f64x8> = Vec::new();
    let mut f0c = vec![f64x8::splat(1.0); nvq];
    let mut if0c = vec![f64x8::splat(1.0); nvq];
    let vci: Vec<usize> = vq.iter().map(|&q| usize::from(vix[pb[q].i3])).collect();
    let v_gi: Vec<usize> = (0..nvq).collect();
    let cvv: Vec<usize> = cv.iter().map(|&c| vq_of[chg[c].0]).collect();
    let cvo: Vec<usize> = cv
        .iter()
        .map(|&c| if class_rows { c_rl(c) } else { c_e(c) })
        .collect();
    let mut rlino: Vec<(f64, f64, u8)> = Vec::new();
    let mut gout: Vec<f64x8> = Vec::new();
    let (mut act, mut rrs, mut s5): (Vec<usize>, Vec<u8>, Vec<f64>) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let (
        mut rv,
        mut rc,
        mut rg,
        mut ri0,
        mut pbuf,
    ): (Vec<f64x8>, Vec<f64x8>, Vec<f64x8>, Vec<f64x8>, Vec<f64x8>) = Default::default();
    let row_r = |ri: usize, v: usize| -> usize {
        let c = &rcf[ri * nv + usize::from(vix[pb[vq[v]].i3])];
        (0..4).find(|&r| c[r] != 0.0).unwrap_or(0)
    };
    let mut mlist: Vec<(usize, f64, usize, usize)> = Vec::new();
    let mut mstart = vec![0usize; npat + 1];
    let mut mlist_c: Vec<(usize, f64, usize, usize)> = Vec::new();
    let mut mcstart = vec![0usize; npat + 1];
    let mut gvalid = vec![false; npat * np];
    for pi in 0..npat {
        let mut ch = pat_ch[pi].iter().peekable();
        for q in 0..np {
            let rho_at = match ch.peek() {
                Some(&&c) if chg[c].0 == q => {
                    ch.next();
                    c_rho(c)
                }
                _ => (np + q) * nzp,
            };
            if !live_p[pi * np + q] {
                continue;
            }
            let r = if vq_of[q] == usize::MAX { f64::from(rmax_p[q]) } else { 1.0 };
            if r == 0.0 {
                continue;
            }
            let hrow = if vq_of[q] == usize::MAX { 0 } else { 1 + vq_of[q] };
            mlist.push((rho_at, r, hrow, (pi * np + q) * nkk * nzp));
            if rho_at == (np + q) * nzp {
                gvalid[pi * np + q] = true;
            } else {
                mlist_c.push((rho_at, r, hrow, (pi * np + q) * nkk * nzp));
            }
        }
        mstart[pi + 1] = mlist.len();
        mcstart[pi + 1] = mlist_c.len();
    }
    let gemm = nkk == 1;
    let pu: Vec<usize> = if gemm {
        (0..npat).filter(|&pi| !touch[pi]).collect()
    } else {
        Vec::new()
    };
    let mut pu_ix = vec![usize::MAX; npat];
    for (u, &pi) in pu.iter().enumerate() {
        pu_ix[pi] = u;
    }
    let (npu4, nb4, nck) = (pu.len().div_ceil(4) * 4, (np + 1).div_ceil(4) * 4, nzp / 8);
    let nxy_all = xs.len() * ys.len();
    let mut gp = std::mem::take(&mut x.gp);
    let mut gb = std::mem::take(&mut x.gb);
    if gemm {
        let z8 = f64x8::splat(0.0);
        if gp.len() < nck * npu4 * nxy_all {
            gp.resize(nck * npu4 * nxy_all, z8);
        }
        if gb.len() < nck * nb4 * nxy_all {
            gb.resize(nck * nb4 * nxy_all, z8);
        }
        for cn in 0..nck * nxy_all {
            gp[cn * npu4 + pu.len()..(cn + 1) * npu4].fill(z8);
            gb[cn * nb4 + np + 1..(cn + 1) * nb4].fill(z8);
        }
    }
    let bq: Vec<(f64, usize)> = (0..np)
        .map(|q| {
            if vq_of[q] == usize::MAX {
                (f64::from(rmax_p[q]), 0)
            } else {
                (1.0, 1 + vq_of[q])
            }
        })
        .collect();
    let mut fixch: Vec<(usize, u8)> = Vec::new();
    let mut fstart = vec![0usize; npat + 1];
    let mut role5f: Vec<Option<(usize, u8)>> = vec![None; npat];
    let (mut fc_slot, mut fc_list) = (vec![usize::MAX; nchg], Vec::<(usize, u8)>::new());
    let mut fixs: Vec<usize> = Vec::new();
    let (mut r5_slot, mut r5_list) = (vec![usize::MAX; np], Vec::<(usize, u8)>::new());
    let mut r5s: Vec<usize> = vec![usize::MAX; npat];
    for (pi, pat) in pp.patterns.iter().enumerate() {
        for &c in &pat_ch[pi] {
            let q = chg[c].0;
            if vq_of[q] == usize::MAX && rmax_p[q] > 0 {
                fixch.push((c_rl(c), rmax_p[q]));
                if fc_slot[c] == usize::MAX {
                    fc_slot[c] = fc_list.len();
                    fc_list.push((c_rl(c), rmax_p[q]));
                }
                fixs.push(fc_slot[c]);
            }
        }
        fstart[pi + 1] = fixch.len();
        if pat.j5 != u8::MAX && vq_of[usize::from(pat.j5)] == usize::MAX {
            let q = usize::from(pat.j5);
            role5f[pi] = Some((q, rmax_p[q]));
            if r5_slot[q] == usize::MAX {
                r5_slot[q] = r5_list.len();
                r5_list.push((q, rmax_p[q]));
            }
            r5s[pi] = r5_slot[q];
        }
    }
    {
        let visits = (xs.len() * ys.len() * (nzp / 8)) as u64;
        ROWSTAT[2].fetch_add(visits * mlist.len() as u64, Ordering::Relaxed);
        ROWSTAT[3].fetch_add(visits * mlist_c.len() as u64, Ordering::Relaxed);
        ROWSTAT[4].fetch_add(visits * npat as u64, Ordering::Relaxed);
    }
    let mut fcv = vec![f64x8::splat(1.0); fc_list.len()];
    let mut r5v = vec![f64x8::splat(1.0); r5_list.len()];
    let ny = ys.len();
    let blk = nkk * nzp;
    let mut hh = vec![0.0f64; (1 + nvq) * blk];
    let mut tt = vec![0.0f64; ntouch * (1 + nvq) * blk];
    let mut pv = vec![0.0f64; nzp];
    let mut fixd = vec![0.0f64; nzp];
    let need_inv = !class_rows || key_j5v.iter().any(Option::is_some);
    let mut wrow = vec![0.0f64; nrow * nkk];
    let psig = |ri: usize| -> Vec<u64> {
        vq.iter()
            .flat_map(|&q| rcf[ri * nv + usize::from(vix[pb[q].i3])].map(f64::to_bits))
            .collect()
    };
    let mut by_sig: Vec<usize> = (0..nrow).collect();
    by_sig.sort_by_cached_key(|&ri| psig(ri));
    let mut mrow = vec![0usize; nrow];
    let mut mrep: Vec<usize> = Vec::new();
    let mut last_sig: Option<Vec<u64>> = None;
    for &ri in &by_sig {
        let sig = psig(ri);
        if last_sig.as_ref() != Some(&sig) {
            mrep.push(ri);
            last_sig = Some(sig);
        }
        mrow[ri] = mrep.len() - 1;
    }
    let nmrow = mrep.len();
    let mut wm = vec![0.0f64; nmrow * nkk];
    ROWSTAT[0].fetch_add(nrow as u64, Ordering::Relaxed);
    ROWSTAT[1].fetch_add(nmrow as u64, Ordering::Relaxed);
    for ix in 0..xs.len() {
        let xx = xs[ix];
        for ri in 0..nrow {
            let xsc = pp.x_scale(ri, xx);
            for k4 in 0..nkk {
                wrow[ri * nkk + k4] = xsc * pp.row_weight(ri, k4);
            }
        }
        wm.fill(0.0);
        for ri in 0..nrow {
            let mi = mrow[ri];
            for k4 in 0..nkk {
                wm[mi * nkk + k4] += wrow[ri * nkk + k4];
            }
        }
        for n in ix * ny..(ix + 1) * ny {
            {
                let sp = f64x8::splat;
                let nck = nzp / 8;
                for q in 0..np {
                    let (cc, bb) = node_cb[n * np + q];
                    for kc in (0..nzp).step_by(8) {
                        let z = f64x8::from_slice(&zv[kc..kc + 8]);
                        let e = sp(bb).mul_add(z, sp(cc));
                        let ie = sp(1.0) / e;
                        inv0[q * nck + kc / 8] = ie;
                        e.copy_to_slice(&mut tab[q * nzp + kc..q * nzp + kc + 8]);
                        (sp(bb) * z * ie)
                            .copy_to_slice(
                                &mut tab[(np + q) * nzp + kc..(np + q) * nzp + kc + 8],
                            );
                    }
                }
                for (c, &(q, _, q5)) in chg.iter().enumerate() {
                    let (cc, bb) = node_cb[n * np + q];
                    let (a0, b0) = (cc + bb * (1.0 - q5), bb * q5);
                    for kc in (0..nzp).step_by(8) {
                        let z = f64x8::from_slice(&zv[kc..kc + 8]);
                        let e = sp(b0).mul_add(z, sp(a0));
                        e.copy_to_slice(&mut tab[c_e(c) + kc..c_e(c) + kc + 8]);
                        (e * inv0[q * nck + kc / 8])
                            .copy_to_slice(&mut tab[c_rl(c) + kc..c_rl(c) + kc + 8]);
                        (sp(b0) * z / e)
                            .copy_to_slice(&mut tab[c_rho(c) + kc..c_rho(c) + kc + 8]);
                    }
                }
            }
            let row = &tab[..stride];
            fixd.fill(node_w[n]);
            for q in 0..np {
                if vq_of[q] == usize::MAX {
                    let e = &row[q * nzp..(q + 1) * nzp];
                    for _ in 0..rmax_p[q] {
                        mul_into(&mut fixd, e);
                    }
                }
            }
            let rows_lap = Lap::new(ESPLIT[0]);
            hh.fill(0.0);
            tt.fill(0.0);
            act.clear();
            act.extend(
                (0..nmrow)
                    .filter(|&mi| wm[mi * nkk..(mi + 1) * nkk].iter().any(|&v| v != 0.0)),
            );
            let na = act.len();
            rrs.clear();
            s5.clear();
            s5.resize(ntouch * na, 0.0);
            rlino.clear();
            for (ai, &mi) in act.iter().enumerate() {
                let ri = mrep[mi];
                if class_rows {
                    rrs.extend((0..nvq).map(|v| row_r(ri, v) as u8));
                } else {
                    rlino.extend(vci.iter().map(|&c| rlin[ri * nv + c]));
                }
                for key in 0..ntouch {
                    if let Some(v5) = key_j5v[key] {
                        let (cc, bb) = node_cb[n * np + vq[v5]];
                        let ep = cc + bb;
                        let pw = |x: f64, k: u8| match k {
                            0 => 1.0,
                            1 => x,
                            2 => x * x,
                            _ => x.powi(i32::from(k)),
                        };
                        s5[key * na + ai] = if class_rows {
                            let r = rrs[ai * nvq + v5];
                            if r == 0 { 0.0 } else { f64::from(r) * bb * pw(ep, r - 1) }
                        } else {
                            let (al, be, mu) = rlino[ai * nvq + v5];
                            if mu == 0 {
                                0.0
                            } else {
                                f64::from(mu) * be * bb * pw(be.mul_add(ep, al), mu - 1)
                            }
                        };
                    }
                }
            }
            let tile = if nkk == 1 { ROW_TILE.min(na.max(1)) } else { na.max(1) };
            let z8 = f64x8::splat(0.0);
            for (buf, len) in [
                (&mut rv, tile),
                (&mut rc, ncv * tile),
                (&mut rg, (nvq + ncv) * tile),
                (&mut ri0, if need_inv { nvq * tile } else { 0 }),
                (&mut pbuf, tile),
            ] {
                if buf.len() < len {
                    buf.resize(len, z8);
                }
            }
            if lvl.len() < (kdepth + 1) * tile {
                lvl.resize((kdepth + 1) * tile, z8);
            }
            let kblk = (1 + nvq) * blk;
            for (kc, t0) in (0..nzp)
                .step_by(8)
                .flat_map(|kc| (0..na).step_by(tile).map(move |t0| (kc, t0)))
            {
                let tn = tile.min(na - t0);
                let lane = |buf: &[f64], at: usize| {
                    debug_assert!(at + kc + 8 <= buf.len());
                    unsafe { f64x8::from_slice(buf.get_unchecked(at + kc..at + kc + 8)) }
                };
                let bump = |buf: &mut [f64], at: usize, add: f64x8| {
                    debug_assert!(at + kc + 8 <= buf.len());
                    unsafe {
                        let d = buf.get_unchecked_mut(at + kc..at + kc + 8);
                        (f64x8::from_slice(d) + add).copy_to_slice(d);
                    }
                };
                for (i, &mi) in act[t0..t0 + tn].iter().enumerate() {
                    let ai = t0 + i;
                    let rr = |v: usize| rrs[ai * nvq + v];
                    let mut vbc = f64x8::splat(1.0);
                    let sp = f64x8::splat;
                    if class_rows {
                        for v in 0..nvq {
                            let e = lane(row, vq[v] * nzp);
                            let mut f = sp(1.0);
                            for _ in 0..rr(v) {
                                f *= e;
                            }
                            f0c[v] = f;
                            rg[v * tn + i] = sp(f64::from(rr(v)));
                            vbc *= f;
                        }
                        if need_inv {
                            for v in 0..nvq {
                                let iv = sp(1.0) / f0c[v];
                                if0c[v] = iv;
                                ri0[v * tn + i] = iv;
                            }
                        }
                        for (s, (&v, &off)) in cvv.iter().zip(&cvo).enumerate() {
                            let rl = lane(row, off);
                            let mut f = sp(1.0);
                            for _ in 0..rr(v) {
                                f *= rl;
                            }
                            unsafe {
                                *rc.get_unchecked_mut(s * tn + i) = f;
                                *rg.get_unchecked_mut((nvq + s) * tn + i) = sp(
                                    f64::from(rr(v)),
                                );
                            }
                        }
                    } else {
                        let lin = |v: usize| unsafe {
                            *rlino.get_unchecked(ai * nvq + v)
                        };
                        for v in 0..nvq {
                            let (al, be, mu) = lin(v);
                            let e = lane(row, vq[v] * nzp);
                            let l = sp(be).mul_add(e, sp(al));
                            let il = sp(1.0) / l;
                            let f = pow_mu(l, mu);
                            if0c[v] = il;
                            rg[v * tn + i] = sp(f64::from(mu) * be) * e * il;
                            vbc *= f;
                            if need_inv {
                                ri0[v * tn + i] = pow_mu(il, mu);
                            }
                        }
                        for (s, (&v, &off)) in cvv.iter().zip(&cvo).enumerate() {
                            let (al, be, mu) = lin(v);
                            let e = lane(row, off);
                            let l = sp(be).mul_add(e, sp(al));
                            let il = sp(1.0) / l;
                            unsafe {
                                *rc.get_unchecked_mut(s * tn + i) = pow_mu(l * if0c[v], mu);
                                *rg.get_unchecked_mut((nvq + s) * tn + i) = sp(
                                    f64::from(mu) * be,
                                ) * e * il;
                            }
                        }
                    }
                    rv[i] = if nkk == 1 { vbc * f64x8::splat(wm[mi]) } else { vbc };
                }
                let wsum = |p: &[f64x8], k4: usize| -> f64x8 {
                    if nkk == 1 {
                        return p[..tn].iter().fold(z8, |a, &v| a + v);
                    }
                    let mut acc = z8;
                    for i in 0..tn {
                        acc = unsafe { *p.get_unchecked(i) }
                            .mul_add(f64x8::splat(wm[act[t0 + i] * nkk + k4]), acc);
                    }
                    acc
                };
                if nkk == 1 {
                    bump(&mut hh, 0, wsum(&rv, 0));
                    gsum_many(&rv[..tn], &rg, &v_gi, &mut gout);
                    for v in 0..nvq {
                        bump(&mut hh, (1 + v) * nzp, gout[v]);
                    }
                } else {
                    for k4 in 0..nkk {
                        let pfk = |i: usize| {
                            rv[i] * f64x8::splat(wm[act[t0 + i] * nkk + k4])
                        };
                        let s = psum_many(tn, &pfk, &rg, &v_gi, &mut gout);
                        bump(&mut hh, k4 * nzp, s);
                        for v in 0..nvq {
                            bump(&mut hh, ((1 + v) * nkk + k4) * nzp, gout[v]);
                        }
                    }
                }
                let mut valid = 0usize;
                for (oi, &key) in korder.iter().enumerate() {
                    let ks = &key_ch[kstart[key]..kstart[key + 1]];
                    let top = ks.len().saturating_sub(1);
                    for d in klcp[oi].min(valid)..top {
                        let (lo, hi) = lvl.split_at_mut((d + 1) * tn);
                        let src: &[f64x8] = if d == 0 {
                            &rv[..tn]
                        } else {
                            &lo[d * tn..]
                        };
                        let r = &rc[ks[d] * tn..(ks[d] + 1) * tn];
                        for ((o, &a), &b) in hi[..tn].iter_mut().zip(&src[..tn]).zip(r) {
                            *o = a * b;
                        }
                    }
                    valid = top;
                    let base: &[f64x8] = if top == 0 {
                        &rv[..tn]
                    } else {
                        &lvl[top * tn..(top + 1) * tn]
                    };
                    let last: &[f64x8] = match ks.last() {
                        Some(&s) => &rc[s * tn..(s + 1) * tn],
                        None => &[],
                    };
                    let j5 = key_j5v[key];
                    let pf = |i: usize| -> f64x8 {
                        let mut p = unsafe { *base.get_unchecked(i) };
                        if !last.is_empty() {
                            p *= unsafe { *last.get_unchecked(i) };
                        }
                        match j5 {
                            Some(v5) => {
                                p
                                    * f64x8::splat(unsafe {
                                        *s5.get_unchecked(key * na + t0 + i)
                                    }) * unsafe { *ri0.get_unchecked(v5 * tn + i) }
                            }
                            None => p,
                        }
                    };
                    let tb0 = key * kblk;
                    let (goff, gis) = (
                        &key_goff[gstart[key]..gstart[key + 1]],
                        &key_gi[gstart[key]..gstart[key + 1]],
                    );
                    if nkk == 1 {
                        let s = psum_many(tn, &pf, &rg, gis, &mut gout);
                        bump(&mut tt, tb0, s);
                        for (&off, &s) in goff.iter().zip(&gout) {
                            bump(&mut tt, tb0 + off, s);
                        }
                        continue;
                    }
                    for i in 0..tn {
                        pbuf[i] = pf(i);
                    }
                    for k4 in 0..nkk {
                        let pfk = |i: usize| {
                            pbuf[i] * f64x8::splat(wm[act[t0 + i] * nkk + k4])
                        };
                        let at = tb0 + k4 * nzp;
                        let s = psum_many(tn, &pfk, &rg, gis, &mut gout);
                        bump(&mut tt, at, s);
                        for (&off, &s) in goff.iter().zip(&gout) {
                            bump(&mut tt, at + off, s);
                        }
                    }
                }
            }
            drop(rows_lap);
            let _pat_lap = Lap::new(ESPLIT[1]);
            if nkk == 1 {
                let (a_buf, m_buf): (&mut [f64], &mut [f64]) = (&mut x.a, &mut x.mom);
                debug_assert!(
                    a_buf.len() >= npat * nzp && m_buf.len() >= npat * np * nzp
                );
                for kc in (0..nzp).step_by(8) {
                    let lane = |buf: &[f64], at: usize| {
                        debug_assert!(at + kc + 8 <= buf.len());
                        unsafe {
                            f64x8::from_slice(buf.get_unchecked(at + kc..at + kc + 8))
                        }
                    };
                    let fx = lane(&fixd, 0);
                    let ck = kc / 8;
                    if gemm {
                        for (q, &(r, hrow)) in bq.iter().enumerate() {
                            gb[(ck * nxy_all + n) * nb4 + q] = lane(row, (np + q) * nzp)
                                * lane(&hh, hrow * nzp) * f64x8::splat(r);
                        }
                        gb[(ck * nxy_all + n) * nb4 + np] = lane(&hh, 0);
                    }
                    for (v, &(rl_at, r)) in fcv.iter_mut().zip(&fc_list) {
                        let rl = lane(row, rl_at);
                        let mut f = rl;
                        for _ in 1..r {
                            f *= rl;
                        }
                        *v = f;
                    }
                    for (v, &(q, r)) in r5v.iter_mut().zip(&r5_list) {
                        *v = if r == 0 {
                            f64x8::splat(0.0)
                        } else {
                            let (cc, bb) = node_cb[n * np + q];
                            let s = f64::from(r) * bb * (cc + bb).powi(i32::from(r) - 1);
                            let e = lane(row, q * nzp);
                            let mut er = e;
                            for _ in 1..r {
                                er *= e;
                            }
                            f64x8::splat(s) / er
                        };
                    }
                    for pi in 0..npat {
                        let mut pvv = fx;
                        for &s in &fixs[fstart[pi]..fstart[pi + 1]] {
                            pvv *= unsafe { *fcv.get_unchecked(s) };
                        }
                        if r5s[pi] != usize::MAX {
                            pvv *= r5v[r5s[pi]];
                        }
                        if gemm && !touch[pi] {
                            gp[(ck * nxy_all + n) * npu4 + pu_ix[pi]] = pvv;
                            for &(rho_at, r, hrow, mom_at) in &mlist_c[mcstart[pi]..mcstart[pi
                                + 1]]
                            {
                                let m_at = mom_at + kc;
                                let t = pvv * lane(row, rho_at) * f64x8::splat(r);
                                debug_assert!(m_at + 8 <= m_buf.len());
                                unsafe {
                                    let m = m_buf.get_unchecked_mut(m_at..m_at + 8);
                                    t.mul_add(lane(&hh, hrow * nzp), f64x8::from_slice(m))
                                        .copy_to_slice(m);
                                }
                            }
                            continue;
                        }
                        let (src, sat): (&[f64], usize) = if touch[pi] {
                            (&tt, touch_at[pi] * (1 + nvq) * blk)
                        } else {
                            (&hh, 0)
                        };
                        let a_at = pi * nzp + kc;
                        unsafe {
                            let a = a_buf.get_unchecked_mut(a_at..a_at + 8);
                            pvv.mul_add(lane(src, sat), f64x8::from_slice(a))
                                .copy_to_slice(a);
                        }
                        for &(rho_at, r, hrow, mom_at) in &mlist[mstart[pi]..mstart[pi
                            + 1]]
                        {
                            let m_at = mom_at + kc;
                            let t = pvv * lane(row, rho_at) * f64x8::splat(r);
                            debug_assert!(m_at + 8 <= m_buf.len());
                            unsafe {
                                let m = m_buf.get_unchecked_mut(m_at..m_at + 8);
                                t.mul_add(lane(src, sat + hrow * nzp), f64x8::from_slice(m))
                                    .copy_to_slice(m);
                            }
                        }
                    }
                }
                continue;
            }
            for (pi, pat) in pp.patterns.iter().enumerate() {
                pv.copy_from_slice(&fixd);
                for &c in &pat_ch[pi] {
                    let q = chg[c].0;
                    if vq_of[q] == usize::MAX {
                        for _ in 0..rmax_p[q] {
                            mul_into(&mut pv, &row[c_rl(c)..c_rl(c) + nzp]);
                        }
                    }
                }
                if pat.j5 != u8::MAX && vq_of[usize::from(pat.j5)] == usize::MAX {
                    let q = usize::from(pat.j5);
                    let r = rmax_p[q];
                    if r == 0 {
                        continue;
                    }
                    let (cc, bb) = node_cb[n * np + q];
                    let s = f64::from(r) * bb * (cc + bb).powi(i32::from(r) - 1);
                    let e = &row[q * nzp..(q + 1) * nzp];
                    for _ in 0..r {
                        div_into(&mut pv, e);
                    }
                    scale_into(&mut pv, s);
                }
                let src = if touch[pi] {
                    &tt[touch_at[pi] * (1 + nvq)
                        * blk..(touch_at[pi] + 1) * (1 + nvq) * blk]
                } else {
                    &hh[..]
                };
                for k4 in 0..nkk {
                    fma_into(
                        &mut x.a[(pi * nkk + k4) * nzp..(pi * nkk + k4 + 1) * nzp],
                        &pv,
                        &src[k4 * nzp..(k4 + 1) * nzp],
                    );
                }
                for &(rho_at, r, hrow, mom_at) in &mlist[mstart[pi]..mstart[pi + 1]] {
                    let rho = &row[rho_at..rho_at + nzp];
                    for k4 in 0..nkk {
                        let m = &mut x.mom[mom_at + k4 * nzp..mom_at + (k4 + 1) * nzp];
                        let h = &src[(hrow * nkk + k4)
                            * nzp..(hrow * nkk + k4 + 1) * nzp];
                        fma3_into(m, &pv, rho, h, r);
                    }
                }
            }
        }
    }
    if gemm && !pu.is_empty() {
        let _gemm_lap = Lap::new(ESPLIT[1]);
        let (a_buf, m_buf): (&mut [f64], &mut [f64]) = (&mut x.a, &mut x.mom);
        for ck in 0..nck {
            let (gpc, gbc) = (
                &gp[ck * nxy_all * npu4..(ck + 1) * nxy_all * npu4],
                &gb[ck * nxy_all * nb4..(ck + 1) * nxy_all * nb4],
            );
            for p4 in (0..npu4).step_by(4) {
                for q4 in (0..nb4).step_by(4) {
                    let mut acc = [[f64x8::splat(0.0); 4]; 4];
                    for nn in 0..nxy_all {
                        let pv: [f64x8; 4] = std::array::from_fn(|i| unsafe {
                            *gpc.get_unchecked(nn * npu4 + p4 + i)
                        });
                        let bv: [f64x8; 4] = std::array::from_fn(|j| unsafe {
                            *gbc.get_unchecked(nn * nb4 + q4 + j)
                        });
                        for i in 0..4 {
                            for j in 0..4 {
                                acc[i][j] = pv[i].mul_add(bv[j], acc[i][j]);
                            }
                        }
                    }
                    for i in 0..4 {
                        let Some(&pi) = pu.get(p4 + i) else { break };
                        for j in 0..4 {
                            let q = q4 + j;
                            let at = if q < np && gvalid[pi * np + q] {
                                &mut m_buf[(pi * np + q) * nzp
                                    + ck * 8..(pi * np + q) * nzp + ck * 8 + 8]
                            } else if q == np {
                                &mut a_buf[pi * nzp + ck * 8..pi * nzp + ck * 8 + 8]
                            } else {
                                continue;
                            };
                            (f64x8::from_slice(at) + acc[i][j]).copy_to_slice(at);
                        }
                    }
                }
            }
        }
    }
    x.gp = gp;
    x.gb = gb;
    clock.lap(4);
    if let Some(ms) = *SLOW.get().unwrap() {
        let t_e = clock.start.elapsed().as_millis();
        if t_e >= ms {
            eprintln!(
                "slow pick node loop: {t_e} ms total so far; classes {}, patterns {npat}, P bases {np}, codes {nk}, nodes {}x{}x{nz}",
                pp.nclass, xs.len(), ys.len()
            );
        }
    }
    x.tab = tab;
    if pp.nb > 0 {
        let (ta, tm) = (std::mem::take(&mut x.a), std::mem::take(&mut x.mom));
        pp.expand(&ta, &tm, &mut x.a, &mut x.mom);
        (x.ta, x.tm) = (ta, tm);
    }
}
impl Prep {
    /// The node loop's arrays over the basis rows (`nb`) into the codes'
    /// rows: a code's array is Σ_b pbas[b][code] · the basis row b's, for A
    /// [pattern][code][z] and Mom [pattern][base][code][z].
    fn expand(&self, at: &[f64], mt: &[f64], a: &mut Vec<f64>, mom: &mut Vec<f64>) {
        let (nb, nk, nzp, np, npat) = (
            self.nb,
            self.kc.len(),
            self.nzp(),
            self.pb.len(),
            self.patterns.len(),
        );
        let spread = |src: &[f64], dst: &mut Vec<f64>, nslab: usize| {
            dst.clear();
            dst.resize(nslab * nk * nzp, 0.0);
            for s in 0..nslab {
                for k in 0..nk {
                    let out = &mut dst[(s * nk + k) * nzp..(s * nk + k + 1) * nzp];
                    for b in 0..nb {
                        axpy_into(
                            out,
                            &src[(s * nb + b) * nzp..(s * nb + b + 1) * nzp],
                            self.pbas[b * nk + k],
                        );
                    }
                }
            }
        };
        spread(at, a, npat);
        spread(mt, mom, npat * np);
    }
}
/// Sections F and G of [`pick`]: `pp`'s deposits, from its node loop's `a`
/// and `mom`.
#[allow(clippy::too_many_arguments)]
pub(super) fn deposit(
    t: &Tables,
    b: &Bits,
    pp: &Prep,
    a: &[f64],
    mom: &[f64],
    x: &mut Six,
    quad: &mut Quad,
    clock: &mut Clock,
) {
    static NO_F: OnceLock<bool> = OnceLock::new();
    if *NO_F.get_or_init(|| std::env::var_os("TG_SIX_NO_F").is_some()) {
        return;
    }
    if pp.single && !OLD_F.load(Ordering::Relaxed) {
        return deposit1(t, b, pp, a, mom, x, quad, clock);
    }
    let _general = Lap::new(FEND[1]);
    FEND[2].fetch_add(1, Ordering::Relaxed);
    let Prep { single, ref kc, ref pb, ref l4, ref prof, .. } = *pp;
    let (nk, nkk, nzp, np, nl4) = (kc.len(), pp.nkk(), pp.nzp(), pb.len(), l4.len());
    let pk = Pk {
        t,
        b,
        pp,
        mom,
        l4,
        pb,
        prof,
        kc,
        members4: &pp.members4,
        nk,
        nkk,
        nzp,
        single,
    };
    let mut bufs = Bufs {
        u4: vec![0.0; nk],
        mom4: vec![0.0; nk],
        omega: vec![0.0; nzp],
        fin: Vec::new(),
        ufin: Vec::new(),
        route: Vec::new(),
        mk: Vec::new(),
    };
    x.cs.new_pick(nl4);
    let mut psi: Vec<f64> = Vec::new();
    let mut rt: Vec<f64> = Vec::new();
    let mut zpow: Vec<f64> = Vec::new();
    let mut lcache = std::mem::take(&mut x.lcache);
    let mut wfb: Vec<f64> = Vec::new();
    let mut sel: Vec<(bool, u8, u32, u32)> = Vec::new();
    let mut cbuf: Vec<f64> = Vec::new();
    let mut at_cache: Vec<(usize, Vec<f64>)> = Vec::new();
    let mut at_pool: Vec<Vec<f64>> = Vec::new();
    let (mut omp_keys, mut omp): (Vec<(usize, bool)>, Vec<f64>) = (
        Vec::new(),
        Vec::new(),
    );
    for gi in 0..pp.groups.len() {
        let g = pp.groups[gi];
        if g.m_min == u32::MAX {
            continue;
        }
        let row = g.row as usize;
        let s0 = pp.sigs[pp.order[g.lo as usize] as usize];
        FCOUNT[0].fetch_add(1, Ordering::Relaxed);
        let setup_lap = Lap::new(FTIME[0]);
        let (ng, rule) = group_rule(&mut lcache, quad, pp, g, 0);
        let (tt, wtt, lmat) = (&rule[..ng], &rule[ng..2 * ng], &rule[2 * ng..]);
        group_z(pp, g, tt, &mut zpow, &mut wfb);
        let gz = Gz { t: tt, lmat, wf: &wfb };
        let identity = Route {
            map: (u32::MAX, 0),
            tb: s0.tb,
            ctx: s0.ctx,
        };
        group_sel(pp, g, &mut sel);
        at_pool.extend(at_cache.drain(..).map(|e| e.1));
        drop(setup_lap);
        let _loop_lap = Lap::new(FTIME[4]);
        let mut i = 0;
        while i < sel.len() {
            let (routed, j5, _, si0) = sel[i];
            let sg0 = pp.sigs[si0 as usize];
            wz_rt(&mut rt, wtt, tt, l4, &pp.wch[sg0.wch.0 as usize..sg0.wch.1 as usize]);
            let pi = usize::from(sg0.pattern);
            let ati = at_of(&mut at_cache, &mut at_pool, a, lmat, pi, nkk, ng, nzp);
            let at_t = std::mem::take(&mut at_cache[ati].1);
            let mut end = i;
            while end < sel.len() && sel[end].0 == routed && sel[end].1 == j5 {
                end += 1;
            }
            if fine() {
                if routed {
                    for e in &sel[i..end] {
                        let sg = pp.sigs[e.3 as usize];
                        FSHAPE[4].fetch_add(1, Ordering::Relaxed);
                        FSHAPE[5]
                            .fetch_add(
                                u64::from(sg.map.1 - sg.map.0),
                                Ordering::Relaxed,
                            );
                    }
                    let mut sh = u32::MAX;
                    for e in &sel[i..end] {
                        if e.2 != sh {
                            sh = e.2;
                            FSHAPE[3].fetch_add(1, Ordering::Relaxed);
                        }
                    }
                } else {
                    FSHAPE[2].fetch_add(1, Ordering::Relaxed);
                }
            }
            if !routed {
                psi.clear();
                psi.extend(
                    (0..ng)
                        .map(|k| {
                            rt[k]
                                * sel[i..end]
                                    .iter()
                                    .map(|e| {
                                        pp.sigs[e.3 as usize].coef * zpow[e.2 as usize * ng + k]
                                    })
                                    .sum::<f64>()
                        }),
                );
                core(&pk, &gz, &mut bufs, &psi, &at_t, j5, pi, row, g, &mut cbuf, false);
                apply(&pk, x, &mut bufs, &cbuf, 1.0, identity, j5, row, g, None, false);
                gz.omega(&psi, &mut bufs.omega);
                let ki = match omp_keys.iter().position(|&k| k == (pi, identity.tb)) {
                    Some(ki) => ki,
                    None => {
                        omp_keys.push((pi, identity.tb));
                        omp.resize(omp.len() + nzp, 0.0);
                        omp_keys.len() - 1
                    }
                };
                axpy_into(&mut omp[ki * nzp..(ki + 1) * nzp], &bufs.omega, 1.0);
                let mut a = i;
                while a < end {
                    let sh = sel[a].2;
                    psi.clear();
                    psi.extend((0..ng).map(|k| rt[k] * zpow[sh as usize * ng + k]));
                    for (k4, u) in bufs.u4.iter_mut().enumerate() {
                        let row_k = if single { 0 } else { k4 };
                        let v: f64 = at_t[row_k * ng..(row_k + 1) * ng]
                            .iter()
                            .zip(&psi)
                            .map(|(a, b)| a * b)
                            .sum();
                        *u = if single { v * prof[k4] } else { v };
                    }
                    route_into(
                        &pk,
                        &bufs.u4,
                        identity,
                        &pp.maps,
                        &mut x.cs,
                        &mut bufs.ufin,
                    );
                    while a < end && sel[a].2 == sh {
                        let sg = pp.sigs[sel[a].3 as usize];
                        clause_scaled(
                            &pk,
                            &mut x.sums,
                            &bufs.ufin,
                            identity.ctx,
                            sg.class,
                            1.0,
                            sg.coef,
                        );
                        a += 1;
                    }
                }
            } else {
                let mut a = i;
                while a < end {
                    let sh = sel[a].2;
                    psi.clear();
                    psi.extend((0..ng).map(|k| rt[k] * zpow[sh as usize * ng + k]));
                    core(
                        &pk,
                        &gz,
                        &mut bufs,
                        &psi,
                        &at_t,
                        j5,
                        pi,
                        row,
                        g,
                        &mut cbuf,
                        true,
                    );
                    let a0 = a;
                    let mut rte = std::mem::take(&mut bufs.route);
                    rte.clear();
                    while a < end && sel[a].2 == sh {
                        let sg = pp.sigs[sel[a].3 as usize];
                        add_route(
                            &pk,
                            x,
                            sg.coef,
                            Route {
                                map: sg.map,
                                tb: sg.tb,
                                ctx: sg.ctx,
                            },
                            &mut rte,
                        );
                        a += 1;
                    }
                    apply_routed(
                        &pk,
                        x,
                        &mut bufs,
                        &cbuf,
                        &rte,
                        s0.ctx,
                        j5,
                        row,
                        g,
                        None,
                        true,
                    );
                    for b in a0..a {
                        let sg = pp.sigs[sel[b].3 as usize];
                        let tb = if sg.tb { TB_BIT } else { 0 };
                        bufs.ufin.clear();
                        for e in sg.map.0..sg.map.1 {
                            let (k4, code, pr) = pp.maps[e as usize];
                            let val = cbuf[usize::from(k4)] * pr * sg.coef;
                            if val != 0.0 {
                                let (at, l) = x.cs.code(code | tb);
                                bufs.ufin.push((at, l, code, val));
                            }
                        }
                        clause(&pk, &mut x.sums, &bufs.ufin, s0.ctx, sg.class, 1.0);
                    }
                    bufs.route = rte;
                }
            }
            at_cache[ati].1 = at_t;
            i = end;
        }
        let _mc_lap = Lap::new(FTIME[3]);
        group_sums(x, pp, &mut bufs.ufin, s0.ctx, g, nl4);
        clause_members(&pk, &mut x.sums, &bufs.ufin, s0.ctx);
    }
    x.lcache = lcache;
    let omega_of = |f: u16| {
        let bs = &t.bases[t.sp[f as usize].base as usize];
        f64::from(bs.mu) / f64::from(bs.phi)
    };
    for (ki, &(pi, tb)) in omp_keys.iter().enumerate() {
        let om = &omp[ki * nzp..(ki + 1) * nzp];
        let r = Route {
            map: (u32::MAX, 0),
            tb,
            ctx: 0,
        };
        rows(&pk, a, pi * nkk, om, &mut bufs.u4);
        route_into(&pk, &bufs.u4, r, &pp.maps, &mut x.cs, &mut bufs.ufin);
        let pat = &pp.patterns[pi];
        for q in 0..np {
            if q as u8 == pat.j5 {
                continue;
            }
            let j = usize::from(pb[q].j);
            let m5 = pat
                .changed
                .iter()
                .find(|c| usize::from(c.0) == q)
                .map_or(l4[j].mask4, |c| c.1);
            if m5 == 0 {
                continue;
            }
            rows(&pk, mom, (pi * np + q) * nkk, om, &mut bufs.mom4);
            route_into(&pk, &bufs.mom4, r, &pp.maps, &mut x.cs, &mut bufs.fin);
            x.pcell
                .credit(
                    t,
                    &mut x.sums,
                    j as u8,
                    m5,
                    l4[j].base,
                    &bufs.fin,
                    &bufs.ufin,
                    &omega_of,
                );
        }
    }
    if fine() {
        for (i, v) in [
            u64::from(single),
            nk as u64,
            0,
            0,
            0,
            0,
            x.cs.lcodes.len() as u64,
            1,
        ]
            .into_iter()
            .enumerate()
        {
            if v != 0 {
                FSHAPE[i].fetch_add(v, Ordering::Relaxed);
            }
        }
    }
    clock.lap(5);
    fixed_w_bases(t, &l4, &x.cs, &mut x.sums, &mut x.pcell.vals);
    x.pcell.flush(t, &mut x.sums, &x.cs.lcodes);
    clock.lap(6);
}
/// `TG_SIX_OLD_F=1` (set in `main`): [`deposit`]'s general path for every
/// pick, single-row ones too.
pub(super) static OLD_F: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(
    false,
);
/// [`deposit`] for a single-row pick. Every vector it deposits is a scalar
/// times one of a few code vectors fixed by the Prep: the profile at the
/// fourth pick's codes (the identity routing, `F1::id`), or a routed fifth
/// pick's map folded through it (`F1::rm`). So the identity's deposits add
/// as scalars per target — Pcell row, W exception, own-member clause — and
/// expand once a pick; the routed ones go through their folded maps; the
/// members' clause comes once per admissibility pattern of the fourth
/// pick's members, on its groups' deposits summed; and the pick's codes are
/// numbered once, up front.
#[allow(clippy::too_many_arguments)]
fn deposit1(
    t: &Tables,
    b: &Bits,
    pp: &Prep,
    a: &[f64],
    mom: &[f64],
    x: &mut Six,
    quad: &mut Quad,
    clock: &mut Clock,
) {
    let Prep { ref kc, ref pb, ref l4, ref prof, .. } = *pp;
    let (nk, nzp, np, nl4) = (kc.len(), pp.nzp(), pb.len(), l4.len());
    let pk = Pk {
        t,
        b,
        pp,
        mom,
        l4,
        pb,
        prof,
        kc,
        members4: &pp.members4,
        nk,
        nkk: 1,
        nzp,
        single: true,
    };
    x.cs.new_pick(nl4);
    let mut f1 = std::mem::take(&mut x.f1);
    f1.number(pp, &mut x.cs);
    let n = x.cs.lcodes.len();
    x.pcell1.reset(n);
    f1.dg.reset(n);
    f1.rs.reset(n);
    f1.exc.clear();
    f1.exc.resize(nl4, 0.0);
    let (ncl, nsp) = (t.classes.len(), t.sp.len());
    if f1.own_c.len() != ncl || f1.own_s.len() != nsp {
        f1.own_c = vec![0.0; ncl];
        f1.own_s = vec![0.0; nsp];
    }
    let mut lcache = std::mem::take(&mut x.lcache);
    let Scratch1 {
        mut psi,
        mut rt,
        mut zpow,
        mut wfb,
        mut sel,
        mut cv,
        mut w,
        mut at_cache,
        mut at_pool,
        mut ufin,
        mut mk,
        mut vis,
        mut omp_keys,
        mut omp,
        mut known,
        mut gslot,
    } = std::mem::take(&mut f1.scr);
    w.clear();
    w.resize(nzp, 0.0);
    omp_keys.clear();
    omp.clear();
    known.clear();
    known
        .extend(
            pp
                .groups
                .iter()
                .map(|&g| {
                    if g.m_min == u32::MAX {
                        0
                    } else {
                        rule_slot(pp, g).map_or(0, |s| s.load(Ordering::Acquire))
                    }
                }),
        );
    for (&p, &g) in known.iter().zip(&pp.groups) {
        if p != 0 {
            let ng = group_ng(g);
            prefetch_bytes(p as *const u8, 8 * (2 * ng + ng * nzp));
        }
    }
    for gi in 0..pp.groups.len() {
        let g = pp.groups[gi];
        if g.m_min == u32::MAX {
            continue;
        }
        let row = g.row as usize;
        let s0 = pp.sigs[pp.order[g.lo as usize] as usize];
        let ctx = s0.ctx;
        FCOUNT[0].fetch_add(1, Ordering::Relaxed);
        let setup_lap = Lap::new(FTIME[0]);
        let (ng, rule) = group_rule(&mut lcache, quad, pp, g, known[gi]);
        let (tt, wtt, lmat) = (&rule[..ng], &rule[ng..2 * ng], &rule[2 * ng..]);
        group_z(pp, g, tt, &mut zpow, &mut wfb);
        let gz = Gz { t: tt, lmat, wf: &wfb };
        group_sel(pp, g, &mut sel);
        at_pool.extend(at_cache.drain(..).map(|e| e.1));
        drop(setup_lap);
        let loop_lap = Lap::new(FTIME[4]);
        let m5row = &pp.m5[row..row + nl4];
        gslot.clear();
        gslot.resize(nl4, u32::MAX);
        let wex = &pp.wex[g.wex.0 as usize..g.wex.1 as usize];
        let mut dg_id = 0.0;
        let mut i = 0;
        while i < sel.len() {
            let (routed, j5, _, si0) = sel[i];
            let sg0 = pp.sigs[si0 as usize];
            wz_rt(&mut rt, wtt, tt, l4, &pp.wch[sg0.wch.0 as usize..sg0.wch.1 as usize]);
            let pi = usize::from(sg0.pattern);
            let ati = at_of(&mut at_cache, &mut at_pool, a, lmat, pi, 1, ng, nzp);
            let at_t = std::mem::take(&mut at_cache[ati].1);
            let mut end = i;
            while end < sel.len() && sel[end].0 == routed && sel[end].1 == j5 {
                end += 1;
            }
            let j5_w = j5 != u8::MAX && !l4[usize::from(j5)].p && !wex.contains(&j5);
            vis.clear();
            vis.extend(w_visits(&pk, g, j5));
            mk.clear();
            for (vi, &j) in vis.iter().enumerate() {
                if j as u8 != j5 && m5row[j] != 0 {
                    if gslot[j] == u32::MAX {
                        gslot[j] = x.pcell1.slot(t, j as u8, m5row[j], l4[j].base)
                            as u32;
                    }
                    mk.push((gslot[j] as usize, 1 + vi));
                }
            }
            if routed {
                for q in 0..np {
                    let j = usize::from(pb[q].j);
                    if j as u8 != j5 && m5row[j] != 0 {
                        if gslot[j] == u32::MAX {
                            gslot[j] = x.pcell1.slot(t, j as u8, m5row[j], l4[j].base)
                                as u32;
                        }
                        mk.push((gslot[j] as usize, 1 + vis.len() + q));
                    }
                }
            }
            if !routed {
                psi.clear();
                psi.extend(
                    (0..ng)
                        .map(|k| {
                            rt[k]
                                * sel[i..end]
                                    .iter()
                                    .map(|e| {
                                        pp.sigs[e.3 as usize].coef * zpow[e.2 as usize * ng + k]
                                    })
                                    .sum::<f64>()
                        }),
                );
                core1(&pk, &gz, &psi, &at_t, &vis, j5, pi, row, false, &mut w, &mut cv);
                let u = cv[0];
                for &(s, off) in &mk {
                    x.pcell1.add_id(s, cv[off], u);
                }
                if j5_w {
                    f1.exc[usize::from(j5)] += u;
                }
                dg_id += u;
                gz.omega(&psi, &mut w);
                let ki = match omp_keys.iter().position(|&k| k == pi) {
                    Some(ki) => ki,
                    None => {
                        omp_keys.push(pi);
                        omp.resize(omp.len() + nzp, 0.0);
                        omp_keys.len() - 1
                    }
                };
                axpy_into(&mut omp[ki * nzp..(ki + 1) * nzp], &w, 1.0);
                let mut e = i;
                while e < end {
                    let sh = sel[e].2 as usize;
                    let v: f64 = (0..ng)
                        .map(|k| at_t[k] * rt[k] * zpow[sh * ng + k])
                        .sum();
                    while e < end && sel[e].2 as usize == sh {
                        let sg = pp.sigs[sel[e].3 as usize];
                        f1.own(t, b, ctx, sg.class, sg.coef * v);
                        e += 1;
                    }
                }
            } else {
                let mut e = i;
                while e < end {
                    let sh = sel[e].2 as usize;
                    psi.clear();
                    psi.extend((0..ng).map(|k| rt[k] * zpow[sh * ng + k]));
                    core1(
                        &pk,
                        &gz,
                        &psi,
                        &at_t,
                        &vis,
                        j5,
                        pi,
                        row,
                        true,
                        &mut w,
                        &mut cv,
                    );
                    let u = cv[0];
                    let e0 = e;
                    while e < end && sel[e].2 as usize == sh {
                        let si = sel[e].3 as usize;
                        let coef = pp.sigs[si].coef;
                        let (lo, hi) = f1.rr[si];
                        for &(_, li, _, v) in &f1.rm[lo as usize..hi as usize] {
                            f1.rs.add(li, coef * v);
                        }
                        e += 1;
                    }
                    let _dlap = Lap::new(FTIME[2]);
                    FCOUNT[2].fetch_add(1, Ordering::Relaxed);
                    for &li in &f1.rs.list {
                        let li = li as usize;
                        let uv = u * f1.rs.v[li];
                        f1.dg.add(li, uv);
                        if j5_w {
                            x.cs.exc[li * nl4 + usize::from(j5)] += uv;
                        }
                    }
                    for &(s, off) in &mk {
                        let c = x.pcell1.weight(s, cv[off], u);
                        let (row, hit) = x.pcell1.row(s);
                        for &li in &f1.rs.list {
                            let li = li as usize;
                            row[li] += c * f1.rs.v[li];
                            hit[li / 64] |= 1 << (li % 64);
                        }
                    }
                    for &li in &f1.rs.list {
                        f1.rs.v[li as usize] = 0.0;
                        f1.rs.on[li as usize] = false;
                    }
                    f1.rs.list.clear();
                    for b5 in e0..e {
                        let sg = pp.sigs[sel[b5].3 as usize];
                        let (lo, hi) = f1.rr[sel[b5].3 as usize];
                        let s = u * sg.coef;
                        ufin.clear();
                        ufin.extend(
                            f1
                                .rm[lo as usize..hi as usize]
                                .iter()
                                .map(|&(at, li, code, v)| (at, li, code, s * v)),
                        );
                        clause(&pk, &mut x.sums, &ufin, ctx, sg.class, 1.0);
                    }
                }
            }
            at_cache[ati].1 = at_t;
            i = end;
        }
        drop(loop_lap);
        let _mc_lap = Lap::new(FTIME[3]);
        if dg_id != 0.0 {
            for &(_, li, _, pf) in &f1.id {
                f1.dg.add(li, dg_id * pf);
            }
        }
        let entry = x.table.entry(ctx).or_default();
        let tb = if s0.tb { TB_BIT } else { 0 };
        for e in entry.iter_mut() {
            let at = x.cs.cix[usize::from(e.0 | tb)];
            if at == 0 {
                continue;
            }
            let l = x.cs.loc[usize::from(at) - 1];
            if l != 0 && f1.dg.on[usize::from(l) - 1] {
                let li = usize::from(l) - 1;
                e.1 += f1.dg.v[li];
                f1.dg.on[li] = false;
            }
        }
        let mp = f1.pattern(t, b, &pp.members4, ctx, n);
        for &li in &f1.dg.list {
            let li = li as usize;
            let v = std::mem::take(&mut f1.dg.v[li]);
            if std::mem::take(&mut f1.dg.on[li]) && v != 0.0 {
                entry.push((f1.raw[li], v));
            }
            if v == 0.0 {
                continue;
            }
            x.cs.u_all[li] += v;
            for &j in wex {
                x.cs.exc[li * nl4 + usize::from(j)] += v;
            }
            f1.mpat[mp].2[li] += v;
        }
        f1.dg.list.clear();
    }
    x.lcache = lcache;
    for (ki, &pi) in omp_keys.iter().enumerate() {
        let om = &omp[ki * nzp..(ki + 1) * nzp];
        let u = dot(&a[pi * nzp..(pi + 1) * nzp], om);
        let pat = &pp.patterns[pi];
        for q in 0..np {
            if q as u8 == pat.j5 {
                continue;
            }
            let j = usize::from(pb[q].j);
            let m5 = pat
                .changed
                .iter()
                .find(|c| usize::from(c.0) == q)
                .map_or(l4[j].mask4, |c| c.1);
            if m5 == 0 {
                continue;
            }
            let p = dot(&mom[(pi * np + q) * nzp..(pi * np + q + 1) * nzp], om);
            let s = x.pcell1.slot(t, j as u8, m5, l4[j].base);
            x.pcell1.add_id(s, p, u);
        }
    }
    let _mc_lap = Lap::new(FEND[0]);
    for (j, e) in f1.exc.iter().enumerate() {
        if *e != 0.0 {
            for &(_, li, _, pf) in &f1.id {
                x.cs.exc[li * nl4 + j] += e * pf;
            }
        }
    }
    for &c in &f1.own_ct {
        let v = std::mem::take(&mut f1.own_c[usize::from(c)]);
        for &(at, _, _, pf) in &f1.id {
            x.sums.add_class(c, at, v * pf);
        }
    }
    f1.own_ct.clear();
    for &f in &f1.own_st {
        let v = std::mem::take(&mut f1.own_s[usize::from(f)]);
        for &(at, _, _, pf) in &f1.id {
            x.sums.add(f, at, v * pf);
        }
    }
    f1.own_st.clear();
    for (_, ctx, vals) in &mut f1.mpat {
        ufin.clear();
        for (li, v) in vals.iter_mut().enumerate() {
            let v = std::mem::take(v);
            if v != 0.0 {
                ufin.push((x.cs.lcodes[li], li, 0, v));
            }
        }
        clause_members(&pk, &mut x.sums, &ufin, *ctx);
    }
    f1.pool.extend(f1.mpat.drain(..).map(|e| e.2));
    drop(_mc_lap);
    if fine() {
        for (i, v) in [1, nk as u64, 0, 0, 0, 0, n as u64, 1].into_iter().enumerate() {
            if v != 0 {
                FSHAPE[i].fetch_add(v, Ordering::Relaxed);
            }
        }
    }
    f1.scr = Scratch1 {
        psi,
        rt,
        zpow,
        wfb,
        sel,
        cv,
        w,
        at_cache,
        at_pool,
        ufin,
        mk,
        vis,
        omp_keys,
        omp,
        known,
        gslot,
    };
    x.f1 = f1;
    clock.lap(5);
    fixed_w_bases(t, l4, &x.cs, &mut x.sums, &mut x.pcell.vals);
    x.pcell1.flush(t, &mut x.sums, &x.cs.lcodes, &x.f1.id);
    clock.lap(6);
}
/// [`Pcell`] for [`deposit1`]: per (live4 base, its mask at the sixth pick),
/// one row over the pick's codes of moment/|mask| − ω·inverse, the two
/// combined as they come — the flush's arithmetic is linear — and the
/// identity's credits as one scalar, which the profile expands at the end.
#[derive(Default)]
struct Pcell1 {
    keys: Vec<(u8, u32, u16)>,
    /// Per key, 1/|mask| and ω.
    coef: Vec<(f64, f64)>,
    rows: Vec<f64>,
    /// Per key, the pick codes its row has touched, a bit each: the flush
    /// visits those, not the whole row once a forme.
    hit: Vec<u64>,
    idv: Vec<f64>,
    by_j: Vec<Vec<(u32, usize)>>,
    n: usize,
    /// Words of `hit` a key.
    nw: usize,
}
impl Pcell1 {
    /// A pick of `n` codes.
    fn reset(&mut self, n: usize) {
        for &(j, ..) in &self.keys {
            self.by_j[usize::from(j)].clear();
        }
        self.keys.clear();
        self.coef.clear();
        self.rows.clear();
        self.hit.clear();
        self.idv.clear();
        self.n = n;
        self.nw = n.div_ceil(64);
    }
    /// The row of base `j` under mask `m5`.
    #[inline]
    fn slot(&mut self, t: &Tables, j: u8, m5: u32, base: u16) -> usize {
        let ju = usize::from(j);
        if self.by_j.len() <= ju {
            self.by_j.resize(ju + 1, Vec::new());
        }
        if let Some(e) = self.by_j[ju].iter().find(|e| e.0 == m5) {
            return e.1;
        }
        let bs = &t.bases[base as usize];
        self.keys.push((j, m5, base));
        self.coef
            .push((
                1.0 / f64::from(m5.count_ones()),
                f64::from(bs.mu) / f64::from(bs.phi),
            ));
        self.rows.resize(self.rows.len() + self.n, 0.0);
        self.hit.resize(self.hit.len() + self.nw, 0);
        self.idv.push(0.0);
        self.by_j[ju].push((m5, self.keys.len() - 1));
        self.keys.len() - 1
    }
    /// Row `s`'s weight for a (moment, inverse) credit.
    #[inline]
    fn weight(&self, s: usize, mom: f64, inv: f64) -> f64 {
        let (size, omega) = self.coef[s];
        mom * size - omega * inv
    }
    #[inline]
    fn add_id(&mut self, s: usize, mom: f64, inv: f64) {
        self.idv[s] += self.weight(s, mom, inv);
    }
    /// Row `s` and its touched bits.
    #[inline]
    fn row(&mut self, s: usize) -> (&mut [f64], &mut [u64]) {
        (
            &mut self.rows[s * self.n..(s + 1) * self.n],
            &mut self.hit[s * self.nw..(s + 1) * self.nw],
        )
    }
    /// The pick's credits onto the formes, the identity's (`id`) expanded
    /// first.
    fn flush(
        &mut self,
        t: &Tables,
        sums: &mut Sums,
        lcodes: &[u16],
        id: &[(u16, usize, u16, f64)],
    ) {
        let (n, nw) = (self.n, self.nw);
        let top_at = lcodes.iter().copied().max().unwrap_or(0);
        for (s, &(_, m5, base)) in self.keys.iter().enumerate() {
            let row = &mut self.rows[s * n..(s + 1) * n];
            let hit = &mut self.hit[s * nw..(s + 1) * nw];
            let v = self.idv[s];
            if v != 0.0 {
                for &(_, li, _, pf) in id {
                    row[li] += v * pf;
                    hit[li / 64] |= 1 << (li % 64);
                }
            }
            for (fi, &f) in t.bases[base as usize].formes.iter().enumerate() {
                if m5 >> fi & 1 == 0 {
                    continue;
                }
                let srow = sums.row_mut(f, top_at);
                for (wi, &word) in hit.iter().enumerate() {
                    let mut bits = word;
                    while bits != 0 {
                        let li = wi * 64 + bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        srow[usize::from(lcodes[li])] += row[li];
                    }
                }
            }
        }
    }
}
/// [`core`] for a single-row pick, as scalars into `out`: u, then a moment
/// per base of `vis`, its [`w_visits`] (zero where it has none), then,
/// `with_p`, one per P base (zero where it is not live). `w` is scratch for
/// the z-point weights.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn core1(
    pk: &Pk,
    gz: &Gz,
    psi: &[f64],
    at_t: &[f64],
    vis: &[usize],
    j5: u8,
    pi: usize,
    row: usize,
    with_p: bool,
    w: &mut [f64],
    out: &mut Vec<f64>,
) {
    FCOUNT[1].fetch_add(1, Ordering::Relaxed);
    let _lap = Lap::new(FTIME[1]);
    let ng = psi.len();
    let m5row = &pk.pp.m5[row..row + pk.l4.len()];
    out.clear();
    out.push(at_t[..ng].iter().zip(psi).map(|(a, b)| a * b).sum());
    for (vi, &j) in vis.iter().enumerate() {
        let l = &pk.l4[j];
        out.push(
            if j as u8 == j5 || m5row[j] == 0 || l.r == 0 {
                0.0
            } else {
                at_t[..ng]
                    .iter()
                    .zip(psi)
                    .zip(&gz.wf[vi * ng..(vi + 1) * ng])
                    .map(|((a, b), c)| a * b * c)
                    .sum()
            },
        );
    }
    if !with_p {
        return;
    }
    gz.omega(psi, w);
    let (np, nzp) = (pk.pb.len(), pk.nzp);
    for q in 0..np {
        let j = usize::from(pk.pb[q].j);
        out.push(
            if j as u8 == j5 || m5row[j] == 0 {
                0.0
            } else {
                dot(&pk.mom[(pi * np + q) * nzp..(pi * np + q + 1) * nzp], w)
            },
        );
    }
}
/// A vector over a pick's codes, with the entries it touched.
#[derive(Default)]
struct Dense {
    v: Vec<f64>,
    on: Vec<bool>,
    list: Vec<u32>,
}
impl Dense {
    fn reset(&mut self, n: usize) {
        self.v.clear();
        self.v.resize(n, 0.0);
        self.on.clear();
        self.on.resize(n, false);
        self.list.clear();
    }
    #[inline]
    fn add(&mut self, i: usize, x: f64) {
        if !self.on[i] {
            self.on[i] = true;
            self.list.push(i as u32);
        }
        self.v[i] += x;
    }
}
/// [`deposit1`]'s pick state, kept for its allocations.
#[derive(Default)]
pub(super) struct F1 {
    /// The identity routing's codes with a profile weight: (situation
    /// index, pick index, raw code, weight).
    id: Vec<(u16, usize, u16, f64)>,
    /// Each routed fifth pick's map folded through the profile by code, as
    /// `id`; fifth pick `si`'s at `rr[si]`.
    rm: Vec<(u16, usize, u16, f64)>,
    rr: Vec<(u32, u32)>,
    /// Per pick code, its raw code; folding scratch.
    raw: Vec<u16>,
    fold: Vec<u32>,
    /// A group's routed deposits summed, and a sub-run's routings.
    dg: Dense,
    rs: Dense,
    /// The identity's own-member clause scalars by class and by species,
    /// with those touched; its W exceptions by live4 base.
    own_c: Vec<f64>,
    own_ct: Vec<u16>,
    own_s: Vec<f64>,
    own_st: Vec<u16>,
    exc: Vec<f64>,
    /// The members' clause by admissibility pattern: (pattern, a context
    /// with it, its groups' deposits summed by pick code); spare rows.
    mpat: Vec<(u64, u64, Vec<f64>)>,
    pool: Vec<Vec<f64>>,
    scr: Scratch1,
}
/// [`deposit1`]'s scratch, kept for its allocations: ψ, R̃'s weights, z
/// powers, W factors, the group's picks, core scalars, z-point weights,
/// patterns' A at the nodes (and spares), a routing's entries, credited
/// bases, W visits, P credits by pattern, the groups' known rules, the
/// group's Pcell rows by live4 base.
#[derive(Default)]
struct Scratch1 {
    psi: Vec<f64>,
    rt: Vec<f64>,
    zpow: Vec<f64>,
    wfb: Vec<f64>,
    sel: Vec<(bool, u8, u32, u32)>,
    cv: Vec<f64>,
    w: Vec<f64>,
    at_cache: Vec<(usize, Vec<f64>)>,
    at_pool: Vec<Vec<f64>>,
    ufin: Vec<(u16, usize, u16, f64)>,
    mk: Vec<(usize, usize)>,
    vis: Vec<usize>,
    omp_keys: Vec<usize>,
    omp: Vec<f64>,
    known: Vec<usize>,
    gslot: Vec<u32>,
}
impl F1 {
    /// The pick's codes numbered: the identity's, then each routed fifth
    /// pick's, its map folded.
    fn number(&mut self, pp: &Prep, cs: &mut Codes) {
        self.id.clear();
        self.rm.clear();
        self.rr.clear();
        self.rr.resize(pp.sigs.len(), (0, 0));
        self.raw.clear();
        let push_raw = |raw: &mut Vec<u16>, l: usize, code: u16| {
            if raw.len() <= l {
                raw.resize(l + 1, 0);
            }
            raw[l] = code;
        };
        let tb_id = pp
            .sigs
            .iter()
            .find(|s| s.map.0 == u32::MAX && s.coef != 0.0)
            .map(|s| s.tb);
        if let Some(tb) = tb_id {
            let tb = if tb { TB_BIT } else { 0 };
            for (k, &code) in pp.kc.iter().enumerate() {
                if pp.prof[k] != 0.0 {
                    let (at, l) = cs.code(code | tb);
                    push_raw(&mut self.raw, l, code);
                    self.id.push((at, l, code, pp.prof[k]));
                }
            }
        }
        for (si, sg) in pp.sigs.iter().enumerate() {
            if sg.coef == 0.0 {
                continue;
            }
            if sg.map.0 == u32::MAX {
                assert_eq!(
                    Some(sg.tb), tb_id,
                    "plain fifth picks with differing Tera Blast bits"
                );
                continue;
            }
            let tb = if sg.tb { TB_BIT } else { 0 };
            let lo = self.rm.len();
            for &(k4, code, pr) in &pp.maps[sg.map.0 as usize..sg.map.1 as usize] {
                let v = pp.prof[usize::from(k4)] * pr;
                if v == 0.0 {
                    continue;
                }
                let (at, l) = cs.code(code | tb);
                push_raw(&mut self.raw, l, code);
                if self.fold.len() <= l {
                    self.fold.resize(l + 1, 0);
                }
                match self.fold[l] {
                    0 => {
                        self.rm.push((at, l, code, v));
                        self.fold[l] = (self.rm.len() - lo) as u32;
                    }
                    p => self.rm[lo + p as usize - 1].3 += v,
                }
            }
            for e in &self.rm[lo..] {
                self.fold[e.1] = 0;
            }
            self.rr[si] = (lo as u32, self.rm.len() as u32);
        }
    }
    /// [`clause_scaled`] for the identity, `x` times the profile: a class or
    /// its admissible formes take it as a scalar.
    #[inline]
    fn own(&mut self, t: &Tables, b: &Bits, ctx: u64, c: u16, x: f64) {
        FCOUNT[3].fetch_add(1, Ordering::Relaxed);
        let class = &t.classes[c as usize];
        if class.simple {
            if b.adm5(ctx, class.members[0]) {
                let o = &mut self.own_c[usize::from(c)];
                if *o == 0.0 {
                    self.own_ct.push(c);
                }
                *o += x;
            }
            return;
        }
        let bs = &t.bases[t.sp[class.members[0] as usize].base as usize];
        let omega = f64::from(bs.mu) / f64::from(bs.phi);
        for &f in &bs.formes {
            if b.adm5(ctx, f) {
                let o = &mut self.own_s[usize::from(f)];
                if *o == 0.0 {
                    self.own_st.push(f);
                }
                *o -= omega * x;
            }
        }
    }
    /// The members' pattern of context `ctx`, its index in `mpat`: which of
    /// the fourth pick's members (a simple class whole, a base by forme) the
    /// species clause finds admissible there.
    fn pattern(
        &mut self,
        t: &Tables,
        b: &Bits,
        members4: &[u16],
        ctx: u64,
        n: usize,
    ) -> usize {
        let (mut bits, mut at) = (0u64, 0);
        for (i, &c) in members4.iter().enumerate() {
            if i > 0 && members4[i - 1] == c {
                continue;
            }
            let class = &t.classes[c as usize];
            if class.simple {
                bits |= u64::from(b.adm5(ctx, class.members[0])) << at;
                at += 1;
            } else {
                for &f in &t.bases[t.sp[class.members[0] as usize].base as usize].formes
                {
                    bits |= u64::from(b.adm5(ctx, f)) << at;
                    at += 1;
                }
            }
        }
        assert!(at <= 64, "more than 64 member formes");
        if let Some(i) = self.mpat.iter().position(|e| e.0 == bits) {
            return i;
        }
        let mut v = self.pool.pop().unwrap_or_default();
        v.clear();
        v.resize(n, 0.0);
        self.mpat.push((bits, ctx, v));
        self.mpat.len() - 1
    }
}
type Lcache = FxMap<(u32, u16, u16), Box<[f64]>>;
/// Rules by (m_min, ng, nz) under these bounds, every thread's: each made
/// once and kept for the run, where each worker's own (tens of MB a worker)
/// were read from DRAM. A slot holds its rule's address, 0 until made.
const RULE_M: usize = 512;
const RULE_G: usize = 64;
const RULE_Z: usize = 128;
static RULES: OnceLock<Box<[AtomicUsize]>> = OnceLock::new();
/// Context codes: 12 flags and a spikes count to 2.
const KERN_CODES: usize = 1 << 14;
static KERNS: OnceLock<Box<[AtomicU64]>> = OnceLock::new();
/// Species `s`'s set kernel at full code `full`, not a lead, as section C
/// reads it — (code after, Tera Blast, Fire added, probability) — from a
/// table every thread shares: each made once and kept for the run, where
/// each worker's own copy (~150k entries, tens of MB) was read from DRAM.
/// A slot holds the outcomes' address and, from bit 48, their count; 0
/// until made.
fn kern_of<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    s: u16,
    full: u16,
) -> &'static [(u16, bool, u8, f64)] {
    let slots = KERNS
        .get_or_init(|| {
            (0..t.sp.len() * KERN_CODES).map(|_| AtomicU64::new(0)).collect()
        });
    let slot = &slots[usize::from(full) * t.sp.len() + usize::from(s)];
    let mut v = slot.load(Ordering::Acquire);
    if v == 0 {
        let made: Box<[(u16, bool, u8, f64)]> = ar
            .kernel(t, s, false, full)
            .iter()
            .map(|ko| (ko.code, ko.tb, ko.fire, ko.p.to_f64()))
            .collect();
        let len = made.len();
        assert!(len < 1 << 16, "a set kernel of {len} outcomes");
        let p = Box::leak(made).as_mut_ptr();
        let packed = p as u64 | (len as u64) << 48;
        v = match slot.compare_exchange(0, packed, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => packed,
            Err(won) => {
                drop(unsafe {
                    Box::from_raw(std::ptr::slice_from_raw_parts_mut(p, len))
                });
                won
            }
        };
    }
    unsafe {
        std::slice::from_raw_parts(
            (v & ((1 << 48) - 1)) as *const (u16, bool, u8, f64),
            (v >> 48) as usize,
        )
    }
}
/// Group `g`'s node count: its rule's.
fn group_ng(g: Group) -> usize {
    ((g.m_max - g.m_min) as usize + g.deg as usize) / 2 + 1
}
/// Group `g`'s slot in the shared rule table, when within its bounds.
fn rule_slot(pp: &Prep, g: Group) -> Option<&'static AtomicUsize> {
    let (m, ng, nz) = (g.m_min as usize, group_ng(g), pp.nz);
    (m < RULE_M && ng < RULE_G && nz < RULE_Z)
        .then(|| {
            let slots = RULES
                .get_or_init(|| {
                    (0..RULE_M * RULE_G * RULE_Z).map(|_| AtomicUsize::new(0)).collect()
                });
            &slots[(m * RULE_G + ng) * RULE_Z + nz]
        })
}
/// Prefetches `n` bytes from `p` toward L2.
#[inline]
fn prefetch_bytes(p: *const u8, n: usize) {
    for o in (0..n).step_by(64) {
        unsafe {
            std::arch::x86_64::_mm_prefetch::<
                { std::arch::x86_64::_MM_HINT_T1 },
            >(p.add(o).cast::<i8>())
        };
    }
}
/// Group `g`'s node count and its rule — the Gauss–Jacobi nodes t_k and
/// weights, weight z^{m_min−1}, then ℓ_c(t_k), the interpolating polynomial's
/// weights at each node, `nzp` a node — from the shared table (at `known`,
/// when already read from it), or past its bounds `lcache`, made if missing.
fn group_rule<'a>(
    lcache: &'a mut Lcache,
    quad: &mut Quad,
    pp: &Prep,
    g: Group,
    known: usize,
) -> (usize, &'a [f64]) {
    let (nz, nzp) = (pp.nz, pp.nzp());
    let ng = group_ng(g);
    if known != 0 {
        return (
            ng,
            unsafe { std::slice::from_raw_parts(known as *const f64, 2 * ng + ng * nzp) },
        );
    }
    let make = |quad: &mut Quad| -> Box<[f64]> {
        let (tt, wtt) = quad.rule(g.m_min, ng);
        let mut r = vec![0.0f64; 2 * ng + ng * nzp];
        r[..ng].copy_from_slice(tt);
        r[ng..2 * ng].copy_from_slice(wtt);
        for (k, &tk) in tt.iter().enumerate() {
            let row_l = &mut r[2 * ng + k * nzp..2 * ng + k * nzp + nz];
            match pp.zv[..nz].iter().position(|&z| z == tk) {
                Some(c) => row_l[c] = 1.0,
                None => {
                    let mut den = 0.0;
                    for c in 0..nz {
                        row_l[c] = pp.bary[c] / (tk - pp.zv[c]);
                        den += row_l[c];
                    }
                    for v in row_l.iter_mut() {
                        *v /= den;
                    }
                }
            }
        }
        r.into_boxed_slice()
    };
    if let Some(slot) = rule_slot(pp, g) {
        let mut p = slot.load(Ordering::Acquire);
        if p == 0 {
            let r = Box::leak(make(quad)).as_mut_ptr() as usize;
            p = match slot.compare_exchange(0, r, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => r,
                Err(won) => {
                    drop(unsafe {
                        Box::from_raw(
                            std::ptr::slice_from_raw_parts_mut(
                                r as *mut f64,
                                2 * ng + ng * nzp,
                            ),
                        )
                    });
                    won
                }
            };
        }
        return (
            ng,
            unsafe { std::slice::from_raw_parts(p as *const f64, 2 * ng + ng * nzp) },
        );
    }
    let lkey = (g.m_min, ng as u16, nz as u16);
    if !lcache.contains_key(&lkey) {
        if lcache.len() > 20_000 {
            lcache.clear();
        }
        let r = make(quad);
        lcache.insert(lkey, r);
    }
    (ng, &lcache[&lkey])
}
/// Group `g`'s powers of its nodes `tt`, `zpow[s · ng + k]` = t_k^s up to its
/// pool sizes' span, and [`Gz::wf`] into `wfb`.
fn group_z(pp: &Prep, g: Group, tt: &[f64], zpow: &mut Vec<f64>, wfb: &mut Vec<f64>) {
    let (ng, span, row) = (tt.len(), (g.m_max - g.m_min) as usize, g.row as usize);
    zpow.clear();
    zpow.resize((span + 1) * ng, 0.0);
    for k in 0..ng {
        let mut v = 1.0;
        for s in 0..=span {
            zpow[s * ng + k] = v;
            v *= tt[k];
        }
    }
    wfb.clear();
    for &j in &pp.wex[g.wex.0 as usize..g.wex.1 as usize] {
        let l = &pp.l4[usize::from(j)];
        let m5 = pp.m5[row + usize::from(j)];
        if m5 == 0 || l.r == 0 {
            wfb.extend(std::iter::repeat_n(0.0, ng));
        } else {
            let q5 = f64::from(m5.count_ones()) / f64::from(l.n4);
            let rr = f64::from(l.r) * q5;
            wfb.extend(tt.iter().map(|&z| rr * z / ((1.0 - q5) + q5 * z)));
        }
    }
}
/// The group's picks by how they deposit, as (routed, base taken, pool size
/// past m_min, sig): those whose set leaves the codes alone, by the base
/// they take (none for most), add into one ψ each; the rest share a core by
/// (base, pool size) and route one by one.
fn group_sel(pp: &Prep, g: Group, sel: &mut Vec<(bool, u8, u32, u32)>) {
    sel.clear();
    for oi in g.lo..g.hi {
        let si = pp.order[oi as usize] as usize;
        let sg = pp.sigs[si];
        if sg.coef != 0.0 {
            sel.push((sg.map.0 != u32::MAX, sg.j5, sg.m0 - g.m_min, si as u32));
        }
    }
    sel.sort_unstable_by_key(|e| (e.0, e.1, e.2));
}
/// w_k · R̃(t_k) over a W-change range.
fn wz_rt(out: &mut Vec<f64>, wtt: &[f64], tt: &[f64], l4: &[Live4], wch: &[(u8, f64)]) {
    out.clear();
    out.extend_from_slice(wtt);
    for &(j, q5) in wch {
        let r = l4[usize::from(j)].r;
        for (o, &tk) in out.iter_mut().zip(tt) {
            let e = (1.0 - q5) + q5 * tk;
            *o
                *= match r {
                    0 => 1.0,
                    1 => e,
                    2 => e * e,
                    3 => e * e * e,
                    _ => e.powi(i32::from(r)),
                };
        }
    }
}
/// Pattern `pi`'s A at the group's nodes, code row by code row: its index in
/// `at_cache`, read through `lmat` on first use.
#[allow(clippy::too_many_arguments)]
fn at_of(
    at_cache: &mut Vec<(usize, Vec<f64>)>,
    at_pool: &mut Vec<Vec<f64>>,
    a: &[f64],
    lmat: &[f64],
    pi: usize,
    nkk: usize,
    ng: usize,
    nzp: usize,
) -> usize {
    if let Some(at) = at_cache.iter().position(|e| e.0 == pi) {
        return at;
    }
    let mut v = at_pool.pop().unwrap_or_default();
    v.clear();
    v.resize(nkk * ng, 0.0);
    for k4 in 0..nkk {
        let arow = &a[(pi * nkk + k4) * nzp..(pi * nkk + k4 + 1) * nzp];
        for k in 0..ng {
            v[k4 * ng + k] = dot(&lmat[k * nzp..(k + 1) * nzp], arow);
        }
    }
    at_cache.push((pi, v));
    at_cache.len() - 1
}
/// What a pick's deposits read.
struct Pk<'a> {
    t: &'a Tables,
    b: &'a Bits,
    pp: &'a Prep,
    /// The node loop's Mom.
    mom: &'a [f64],
    l4: &'a [Live4],
    pb: &'a [PBase],
    prof: &'a [f64],
    kc: &'a [u16],
    members4: &'a [u16],
    nk: usize,
    nkk: usize,
    nzp: usize,
    single: bool,
}
/// A group's z rule: its Gauss–Jacobi nodes, the interpolation weights that
/// read the node loop's z points there, and per W base the group changes
/// (in `wex` order), its z-factor's z∂/∂z over the factor, r·q₅z/(1 − q₅ + q₅z),
/// at the nodes (zero where it has no moment).
struct Gz<'a> {
    t: &'a [f64],
    lmat: &'a [f64],
    wf: &'a [f64],
}
impl Gz<'_> {
    /// ω over the z points such that ⟨array, ω⟩ = Σ_k ψ_k · array(t_k).
    #[inline]
    fn omega(&self, psi: &[f64], out: &mut [f64]) {
        let nzp = out.len();
        out.fill(0.0);
        for (k, &p) in psi.iter().enumerate() {
            if p != 0.0 {
                axpy_into(out, &self.lmat[k * nzp..(k + 1) * nzp], p);
            }
        }
    }
}
/// A deposit's buffers: (situation index, pick index, raw code, value)
/// lists, and the pick's U and W exceptions by its code index.
struct Bufs {
    u4: Vec<f64>,
    mom4: Vec<f64>,
    omega: Vec<f64>,
    fin: Vec<(u16, usize, u16, f64)>,
    ufin: Vec<(u16, usize, u16, f64)>,
    route: Vec<(u8, u16, usize, u16, f64)>,
    mk: Vec<(usize, usize, u32, u16)>,
}
/// Where a deposit's codes go: a set's routing (a range of `Six::maps`, or
/// none), the Tera Blast flag, the context.
#[derive(Clone, Copy)]
struct Route {
    map: (u32, u32),
    tb: bool,
    ctx: u64,
}
/// A code row's dot with ψ, the lone class's weights applied.
#[inline]
fn rows(pk: &Pk, arr: &[f64], first: usize, psi: &[f64], out: &mut [f64]) {
    let nzp = pk.nzp;
    if pk.single {
        let v = dot(&arr[first * nzp..(first + 1) * nzp], psi);
        for (o, &w) in out.iter_mut().zip(&pk.prof[..pk.nk]) {
            *o = w * v;
        }
    } else {
        for (k4, o) in out.iter_mut().enumerate() {
            *o = dot(&arr[(first + k4) * nzp..(first + k4 + 1) * nzp], psi);
        }
    }
}
/// A fourth-pick-code vector to final codes.
#[inline(never)]
fn route_into(
    pk: &Pk,
    v: &[f64],
    r: Route,
    maps: &[(u8, u16, f64)],
    cs: &mut Codes,
    out: &mut Vec<(u16, usize, u16, f64)>,
) {
    let tb = if r.tb { TB_BIT } else { 0 };
    out.clear();
    if r.map.0 == u32::MAX {
        for (k4, &val) in v.iter().enumerate() {
            if val != 0.0 {
                let (at, l) = cs.code(pk.kc[k4] | tb);
                out.push((at, l, pk.kc[k4], val));
            }
        }
    } else {
        for &(k4, code, pr) in &maps[r.map.0 as usize..r.map.1 as usize] {
            let val = v[usize::from(k4)] * pr;
            if val != 0.0 {
                let (at, l) = cs.code(code | tb);
                out.push((at, l, code, val));
            }
        }
    }
}
/// The species clause on `ufin` for one class of the team, used `used`
/// times: it cannot come again. A simple class's members share ω·used/n of
/// it (at the flush); a base's admissible formes take ω each.
#[inline(never)]
fn clause(
    pk: &Pk,
    sums: &mut Sums,
    ufin: &[(u16, usize, u16, f64)],
    ctx: u64,
    c: u16,
    used: f64,
) {
    clause_scaled(pk, sums, ufin, ctx, c, used, 1.0);
}
/// [`clause`] on `ufin` times `scale`.
#[allow(clippy::too_many_arguments)]
fn clause_scaled(
    pk: &Pk,
    sums: &mut Sums,
    ufin: &[(u16, usize, u16, f64)],
    ctx: u64,
    c: u16,
    used: f64,
    scale: f64,
) {
    FCOUNT[3].fetch_add(1, Ordering::Relaxed);
    let (t, b) = (pk.t, pk.b);
    let class = &t.classes[c as usize];
    if class.simple {
        if b.adm5(ctx, class.members[0]) {
            for &(at, _, _, val) in ufin {
                sums.add_class(c, at, used * scale * val);
            }
        }
        return;
    }
    let bs = &t.bases[t.sp[class.members[0] as usize].base as usize];
    let omega = f64::from(bs.mu) / f64::from(bs.phi);
    for &f in &bs.formes {
        if b.adm5(ctx, f) {
            for &(at, _, _, val) in ufin {
                sums.add(f, at, -omega * scale * val);
            }
        }
    }
}
/// [`clause`] for the fourth pick's members, each class at its count.
fn clause_members(pk: &Pk, sums: &mut Sums, ufin: &[(u16, usize, u16, f64)], ctx: u64) {
    let m = pk.members4;
    for (i, &c) in m.iter().enumerate() {
        if i > 0 && m[i - 1] == c {
            continue;
        }
        let used = m.iter().filter(|&&x| x == c).count() as f64;
        clause(pk, sums, ufin, ctx, c, used);
    }
}
/// The W bases a deposit visits: those the group changes or refuses, then
/// the one the pick takes when the group leaves it.
fn w_visits<'a>(pk: &Pk<'a>, g: Group, j5: u8) -> impl Iterator<Item = usize> + 'a {
    let wex = &pk.pp.wex[g.wex.0 as usize..g.wex.1 as usize];
    let j5_w = j5 != u8::MAX && !pk.l4[usize::from(j5)].p && !wex.contains(&j5);
    wex.iter().map(|&j| usize::from(j)).chain(j5_w.then_some(usize::from(j5)))
}
/// The fourth-pick-code vectors behind ψ (z weights at the group's nodes)
/// into `core`: u, then a moment per [`w_visits`] base (zero where it has
/// none), then, `with_p`, one per P base (zero where it is not live).
/// `at_t` holds the pattern's A read at the group's nodes, code row by code
/// row.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn core(
    pk: &Pk,
    gz: &Gz,
    bufs: &mut Bufs,
    psi: &[f64],
    at_t: &[f64],
    j5: u8,
    pi: usize,
    row: usize,
    g: Group,
    core: &mut Vec<f64>,
    with_p: bool,
) {
    FCOUNT[1].fetch_add(1, Ordering::Relaxed);
    let _lap = Lap::new(FTIME[1]);
    let (pp, nk, nkk) = (pk.pp, pk.nk, pk.nkk);
    let ng = psi.len();
    let nvis = w_visits(pk, g, j5).count();
    let np = pk.pb.len();
    core.clear();
    core.resize(nk * (1 + nvis + if with_p { np } else { 0 }), 0.0);
    let at_dot = |wf: Option<&[f64]>, out: &mut [f64]| {
        let wt = |row: &[f64]| -> f64 {
            match wf {
                None => row.iter().zip(psi).map(|(a, b)| a * b).sum(),
                Some(f) => row.iter().zip(psi).zip(f).map(|((a, b), c)| a * b * c).sum(),
            }
        };
        if pk.single {
            let v = wt(&at_t[..ng]);
            for (o, &w) in out.iter_mut().zip(&pk.prof[..nk]) {
                *o = w * v;
            }
        } else {
            for (k4, o) in out.iter_mut().enumerate() {
                *o = wt(&at_t[k4 * ng..(k4 + 1) * ng]);
            }
        }
    };
    at_dot(None, &mut core[..nk]);
    let nl4 = pk.l4.len();
    let m5row = &pp.m5[row..row + nl4];
    for (vi, j) in w_visits(pk, g, j5).enumerate() {
        let at = nk * (1 + vi);
        let l = &pk.l4[j];
        let m5 = m5row[j];
        if j as u8 == j5 || m5 == 0 || l.r == 0 {
            continue;
        }
        at_dot(Some(&gz.wf[vi * ng..(vi + 1) * ng]), &mut core[at..at + nk]);
    }
    if !with_p {
        return;
    }
    let mut w = std::mem::take(&mut bufs.omega);
    gz.omega(psi, &mut w);
    for q in 0..np {
        let at = nk * (1 + nvis + q);
        let j = usize::from(pk.pb[q].j);
        if j as u8 == j5 || m5row[j] == 0 {
            continue;
        }
        rows(pk, pk.mom, (pi * np + q) * nkk, &w, &mut core[at..at + nk]);
    }
    bufs.omega = w;
}
/// The deposits of `core` times `coef`, routed by `r`: the table entry, U
/// and the W exceptions, the changed W and (`with_p`) live P bases'
/// credits, and the species clause — the fourth pick's members always, the
/// fifth's when `class5`.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn apply(
    pk: &Pk,
    x: &mut Six,
    bufs: &mut Bufs,
    core: &[f64],
    coef: f64,
    r: Route,
    j5: u8,
    row: usize,
    g: Group,
    class5: Option<u16>,
    with_p: bool,
) {
    let mut rte = std::mem::take(&mut bufs.route);
    rte.clear();
    add_route(pk, x, coef, r, &mut rte);
    apply_routed(pk, x, bufs, core, &rte, r.ctx, j5, row, g, class5, with_p);
    bufs.route = rte;
}
/// `r`'s routing times `coef` into `rte`: (code row, situation index, pick
/// index, raw code, weight), alike entries summed — so picks sharing a core
/// deposit once through their routings' sum.
fn add_route(
    pk: &Pk,
    x: &mut Six,
    coef: f64,
    r: Route,
    rte: &mut Vec<(u8, u16, usize, u16, f64)>,
) {
    let pp = pk.pp;
    let tb = if r.tb { TB_BIT } else { 0 };
    if r.map.0 == u32::MAX && rte.is_empty() {
        for k4 in 0..pk.nk {
            let (at, l) = x.cs.code(pk.kc[k4] | tb);
            rte.push((k4 as u8, at, l, pk.kc[k4], coef));
        }
        return;
    }
    if x.rhash.len() != RHASH {
        x.rhash = vec![0; RHASH];
    }
    if rte.is_empty() {
        for &h in &x.rused {
            x.rhash[h] = 0;
        }
        x.rused.clear();
    }
    let mut push = |k4: u8, code: u16, w: f64, x: &mut Six| {
        let (at, l) = x.cs.code(code | tb);
        if rte.len() >= RHASH / 2 {
            match rte.iter_mut().find(|e| e.0 == k4 && e.1 == at) {
                Some(e) => e.4 += w,
                None => rte.push((k4, at, l, code, w)),
            }
            return;
        }
        let mut h = (u32::from(at).wrapping_mul(0x9e37_79b1)
            ^ u32::from(k4).wrapping_mul(0x85eb_ca6b)) as usize
            >> (32 - RHASH.trailing_zeros()) & (RHASH - 1);
        loop {
            match x.rhash[h] {
                0 => {
                    rte.push((k4, at, l, code, w));
                    x.rhash[h] = rte.len() as u32;
                    x.rused.push(h);
                    break;
                }
                s => {
                    let e = &mut rte[s as usize - 1];
                    if e.0 == k4 && e.1 == at {
                        e.4 += w;
                        break;
                    }
                }
            }
            h = (h + 1) & (RHASH - 1);
        }
    };
    if r.map.0 == u32::MAX {
        for k4 in 0..pk.nk {
            push(k4 as u8, pk.kc[k4], coef, x);
        }
    } else {
        for e in r.map.0..r.map.1 {
            let (k4, code, pr) = pp.maps[e as usize];
            push(k4, code, pr * coef, x);
        }
    }
}
/// G. W bases fixed by a fifth pick: (r/|mask| − ω) of every deposit it did
/// not change — a base's values made once, then each forme's row taken once.
#[inline(never)]
fn fixed_w_bases(
    t: &Tables,
    l4: &[Live4],
    cs: &Codes,
    sums: &mut Sums,
    vals: &mut Vec<f64>,
) {
    let (n, nl4) = (cs.lcodes.len(), cs.nl4);
    let top_at = cs.lcodes.iter().copied().max().unwrap_or(0);
    for (j, l) in l4.iter().enumerate() {
        if l.p {
            continue;
        }
        vals.clear();
        vals.extend((0..n).map(|li| cs.u_all[li] - cs.exc[li * nl4 + j]));
        if vals.iter().all(|&v| v == 0.0) {
            continue;
        }
        let size = 1.0 / f64::from(l.n4);
        for (fi, &f) in t.bases[l.base as usize].formes.iter().enumerate() {
            if l.mask4 >> fi & 1 == 0 {
                continue;
            }
            let bs = &t.bases[t.sp[f as usize].base as usize];
            let coeff = f64::from(l.r) * size - f64::from(bs.mu) / f64::from(bs.phi);
            if coeff == 0.0 {
                continue;
            }
            let srow = sums.row_mut(f, top_at);
            for (&v, &at) in vals.iter().zip(&cs.lcodes) {
                if v != 0.0 {
                    srow[usize::from(at)] += coeff * v;
                }
            }
        }
    }
}
/// A group's deposits summed, into `ufin` for the members' clause, the table
/// entry, U and the group's W exceptions.
#[inline(never)]
fn group_sums(
    x: &mut Six,
    pp: &Prep,
    ufin: &mut Vec<(u16, usize, u16, f64)>,
    ctx: u64,
    g: Group,
    nl4: usize,
) {
    ufin.clear();
    let entry = x.table.entry(ctx).or_default();
    let wex = &pp.wex[g.wex.0 as usize..g.wex.1 as usize];
    for &li in &x.gtouch {
        x.gmark[li] = false;
        let v = std::mem::take(&mut x.gu[li]);
        if v != 0.0 {
            ufin.push((x.cs.lcodes[li], li, 0, v));
            let code = x.gcode[li];
            match entry.iter_mut().find(|e| e.0 == code) {
                Some(e) => e.1 += v,
                None => entry.push((code, v)),
            }
            x.cs.u_all[li] += v;
            for &j in wex {
                x.cs.exc[li * nl4 + usize::from(j)] += v;
            }
        }
    }
    x.gtouch.clear();
}
/// [`apply`] through a routing table built by [`add_route`].
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn apply_routed(
    pk: &Pk,
    x: &mut Six,
    bufs: &mut Bufs,
    core: &[f64],
    rte: &[(u8, u16, usize, u16, f64)],
    ctx: u64,
    j5: u8,
    row: usize,
    g: Group,
    class5: Option<u16>,
    with_p: bool,
) {
    FCOUNT[2].fetch_add(1, Ordering::Relaxed);
    let _lap = Lap::new(FTIME[2]);
    let (t, pp, nk) = (pk.t, pk.pp, pk.nk);
    let omega = |f: u16| {
        let bs = &t.bases[t.sp[f as usize].base as usize];
        f64::from(bs.mu) / f64::from(bs.phi)
    };
    let route = |v: &[f64], out: &mut Vec<(u16, usize, u16, f64)>| {
        out.clear();
        for &(k4, at, l, code, w) in rte {
            let val = v[usize::from(k4)] * w;
            if val != 0.0 {
                out.push((at, l, code, val));
            }
        }
    };
    route(&core[..nk], &mut bufs.ufin);
    let nl4 = pk.l4.len();
    let mut mk = std::mem::take(&mut bufs.mk);
    mk.clear();
    let (w0, w1) = (g.wex.0 as usize, g.wex.1 as usize);
    let j5_w = j5 != u8::MAX && !pk.l4[usize::from(j5)].p
        && !pp.wex[w0..w1].contains(&j5);
    let mut at = nk;
    for vi in 0..w1 - w0 + usize::from(j5_w) {
        let j = usize::from(if w0 + vi < w1 { pp.wex[w0 + vi] } else { j5 });
        if w0 + vi >= w1 {
            for &(_, li, _, val) in &bufs.ufin {
                x.cs.exc[li * nl4 + j] += val;
            }
        }
        let m5 = pp.m5[row + j];
        let here = at;
        at += nk;
        if j as u8 == j5 || m5 == 0 {
            continue;
        }
        mk.push((x.pcell.slot(j as u8, m5, pk.l4[j].base), here, m5, pk.l4[j].base));
    }
    for q in 0..if with_p { pk.pb.len() } else { 0 } {
        let j = usize::from(pk.pb[q].j);
        let m5 = pp.m5[row + j];
        let here = at;
        at += nk;
        if j as u8 == j5 || m5 == 0 {
            continue;
        }
        mk.push((x.pcell.slot(j as u8, m5, pk.l4[j].base), here, m5, pk.l4[j].base));
    }
    if rte.len() > MK_PASS {
        for &(s, off, m5, base) in &mk {
            route(&core[off..off + nk], &mut bufs.fin);
            x.pcell
                .credit_row(t, &mut x.sums, s, m5, base, &bufs.fin, &bufs.ufin, &omega);
        }
    } else if !mk.is_empty() {
        for &(k4, sat, li, _, w) in rte {
            let k4 = usize::from(k4);
            let u = core[k4] * w;
            if li < PCW {
                for &(s, off, ..) in &mk {
                    let mom = core[off + k4] * w;
                    if mom != 0.0 || u != 0.0 {
                        x.pcell.add(s, li, mom, u);
                    }
                }
            } else {
                for &(_, off, m5, base) in &mk {
                    let mom = core[off + k4] * w;
                    if mom == 0.0 && u == 0.0 {
                        continue;
                    }
                    let bs = &t.bases[base as usize];
                    let v = mom / f64::from(m5.count_ones())
                        - f64::from(bs.mu) / f64::from(bs.phi) * u;
                    for (fi, &f) in bs.formes.iter().enumerate() {
                        if m5 >> fi & 1 != 0 {
                            x.sums.add(f, sat, v);
                        }
                    }
                }
            }
        }
    }
    bufs.mk = mk;
    for &(_, li, code, val) in &bufs.ufin {
        if x.gu.len() <= li {
            x.gu.resize(li + 1, 0.0);
            x.gmark.resize(li + 1, false);
            x.gcode.resize(li + 1, 0);
        }
        if !x.gmark[li] {
            x.gmark[li] = true;
            x.gtouch.push(li);
            x.gcode[li] = code;
        }
        x.gu[li] += val;
    }
    if let Some(c) = class5 {
        clause(pk, &mut x.sums, &bufs.ufin, ctx, c, 1.0);
    }
}
/// Slots of [`Pcell`] rows: pick code indices it keeps dense.
const PCW: usize = 64;
/// A pick's base credits before they reach the formes: per (live4 base,
/// its mask at the sixth pick), per pick code index, the moment and the
/// inverse summed — so a base's formes are credited once a pick.
#[derive(Default)]
struct Pcell {
    keys: Vec<(u8, u32, u16)>,
    acc: Vec<f64>,
    /// Per key, one past the highest dense slot credited.
    top: Vec<usize>,
    /// Per live4 base, its keys: (mask, key index).
    by_j: Vec<Vec<(u32, usize)>>,
    /// The flush's values of a key, by pick code.
    vals: Vec<f64>,
}
impl Pcell {
    /// The row of base `j` under mask `m5`.
    #[inline]
    fn slot(&mut self, j: u8, m5: u32, base: u16) -> usize {
        let ju = usize::from(j);
        if self.by_j.len() <= ju {
            self.by_j.resize(ju + 1, Vec::new());
        }
        match self.by_j[ju].iter().find(|e| e.0 == m5) {
            Some(e) => e.1,
            None => {
                self.keys.push((j, m5, base));
                self.acc.resize(self.acc.len() + 2 * PCW, 0.0);
                self.top.push(0);
                self.by_j[ju].push((m5, self.keys.len() - 1));
                self.keys.len() - 1
            }
        }
    }
    /// One (moment, inverse) credit to row `s` at a dense pick code.
    #[inline]
    fn add(&mut self, s: usize, li: usize, mom: f64, inv: f64) {
        let row = &mut self.acc[s * 2 * PCW..(s + 1) * 2 * PCW];
        row[li] += mom;
        row[PCW + li] += inv;
        let top = &mut self.top[s];
        *top = (*top).max(li + 1);
    }
    /// Base `j`'s (moment, inverse) credits under mask `m5`, by pick code;
    /// codes past the dense width go straight to the formes.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn credit(
        &mut self,
        t: &Tables,
        sums: &mut Sums,
        j: u8,
        m5: u32,
        base: u16,
        mom: &[(u16, usize, u16, f64)],
        inv: &[(u16, usize, u16, f64)],
        omega: &dyn Fn(u16) -> f64,
    ) {
        let s = self.slot(j, m5, base);
        self.credit_row(t, sums, s, m5, base, mom, inv, omega);
    }
    /// [`Pcell::credit`] to row `s`.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn credit_row(
        &mut self,
        t: &Tables,
        sums: &mut Sums,
        s: usize,
        m5: u32,
        base: u16,
        mom: &[(u16, usize, u16, f64)],
        inv: &[(u16, usize, u16, f64)],
        omega: &dyn Fn(u16) -> f64,
    ) {
        let row = &mut self.acc[s * 2 * PCW..(s + 1) * 2 * PCW];
        let top = &mut self.top[s];
        let size = 1.0 / f64::from(m5.count_ones());
        for &(sat, li, _, val) in mom {
            if li < PCW {
                row[li] += val;
                *top = (*top).max(li + 1);
            } else {
                for (fi, &f) in t.bases[base as usize].formes.iter().enumerate() {
                    if m5 >> fi & 1 != 0 {
                        sums.add(f, sat, val * size);
                    }
                }
            }
        }
        for &(sat, li, _, val) in inv {
            if li < PCW {
                row[PCW + li] += val;
                *top = (*top).max(li + 1);
            } else {
                for (fi, &f) in t.bases[base as usize].formes.iter().enumerate() {
                    if m5 >> fi & 1 != 0 {
                        sums.add(f, sat, -omega(f) * val);
                    }
                }
            }
        }
    }
    /// The pick's credits onto the formes: moment/|mask| − ω·inverse.
    fn flush(&mut self, t: &Tables, sums: &mut Sums, lcodes: &[u16]) {
        for (s, &(_, m5, base)) in self.keys.iter().enumerate() {
            let n = self.top[s].min(lcodes.len());
            let row = &self.acc[s * 2 * PCW..(s + 1) * 2 * PCW];
            let size = 1.0 / f64::from(m5.count_ones());
            let bs = &t.bases[base as usize];
            let omega = f64::from(bs.mu) / f64::from(bs.phi);
            self.vals.clear();
            self.vals.extend((0..n).map(|li| row[li] * size - omega * row[PCW + li]));
            let top_at = lcodes[..n].iter().copied().max().unwrap_or(0);
            for (fi, &f) in bs.formes.iter().enumerate() {
                if m5 >> fi & 1 == 0 {
                    continue;
                }
                let srow = sums.row_mut(f, top_at);
                for (&v, &at) in self.vals.iter().zip(&lcodes[..n]) {
                    if v != 0.0 {
                        srow[usize::from(at)] += v;
                    }
                }
            }
        }
        for &(j, ..) in &self.keys {
            self.by_j[usize::from(j)].clear();
        }
        self.keys.clear();
        self.acc.clear();
        self.top.clear();
    }
}
/// [`pick`]'s fifth pick `sg` against the team it makes.
#[allow(clippy::too_many_arguments)]
fn check_sig<N: Num>(
    t: &Tables,
    p: &Pick5<N>,
    sg: &Sig,
    ctx5: u64,
    f5: u32,
    m5: &[u32],
) {
    let mut c5 = p.c4.with(sg.class);
    c5.lead = true;
    c5.tb = sg.tb;
    c5.fire = p.c4.fire + sg.fire;
    let want = Ctx::of_key(t, &c5);
    assert!(ctx_of_bits(ctx5) == want, "context bits");
    let av = Avail::of_key(t, &c5);
    let mut live = Vec::new();
    let total = options_into(t, &c5, &want, &av, &mut live, None);
    let f5_ref = total - live.iter().map(|l| u32::from(l.alive)).sum::<u32>();
    assert_eq!(f5, f5_ref, "fixed pool");
    for (j, l) in p.live4.iter().enumerate() {
        let m = live.iter().find(|x| x.mi == l.mi).map_or(0, |x| x.mask);
        let mine = if j as u8 == sg.j5 { 0 } else { m5[j] };
        assert_eq!(
            mine, m, "mask of {}", t.sp[t.bases[l.base as usize].formes[0] as usize].name
        );
    }
}
/// A thread's prepared picks awaiting the GPU node loop, deposited in their
/// order — each situation's flush and each item's table in its place — once
/// it has run. The worker goes on preparing while its batches are on the
/// GPU, and waits only with more than [`depth`] in flight.
#[cfg(feature = "gpu")]
#[derive(Default)]
pub(super) struct Queue {
    batch: gpu::SixBatch,
    items: Vec<Queued>,
    /// The queued picks' allocations, counted against [`queue_bytes`].
    held: usize,
    /// Submitted batches, oldest first, with their picks and markers.
    flight: std::collections::VecDeque<(gpu::SixBatch, Vec<Queued>)>,
    /// Spent batches, kept for their allocations.
    spare: Vec<(gpu::SixBatch, Vec<Queued>)>,
    pool: Vec<Box<Prep>>,
    /// A multi-row pick's A and Mom, gathered from its rows' kernel picks
    /// (`ta`, `tm`), then over its codes (`ma`, `mm`) from basis rows.
    ta: Vec<f64>,
    tm: Vec<f64>,
    ma: Vec<f64>,
    mm: Vec<f64>,
}
#[cfg(feature = "gpu")]
enum Queued {
    /// A prepared pick and its index in the batch, or None for the CPU's
    /// node loop.
    Pick(Box<Prep>, Option<usize>),
    Flush,
    /// The end of an item: its table deposits into the pass's table.
    Table,
}
impl Prep {
    /// What [`deposit`] reads of the Prep, toward L2.
    fn prefetch(&self) {
        fn lines<T>(v: &[T]) {
            let (p, n) = (v.as_ptr().cast::<i8>(), std::mem::size_of_val(v));
            for o in (0..n).step_by(64) {
                unsafe {
                    std::arch::x86_64::_mm_prefetch::<
                        { std::arch::x86_64::_MM_HINT_T1 },
                    >(p.add(o))
                };
            }
        }
        lines(&self.sigs);
        lines(&self.maps);
        lines(&self.order);
        lines(&self.groups);
        lines(&self.m5);
        lines(&self.wch);
        lines(&self.wex);
        lines(&self.patterns);
        for p in &self.patterns {
            lines(&p.changed);
        }
        lines(&self.l4);
        lines(&self.pb);
        lines(&self.kc);
        lines(&self.prof);
        lines(&self.members4);
        lines(&self.zv);
        lines(&self.bary);
    }
}
#[cfg(feature = "gpu")]
impl Prep {
    /// The pick's allocations, about: what a queue of them holds.
    fn heap_bytes(&self) -> usize {
        size_of::<Sig>() * self.sigs.capacity()
            + 16 * (self.maps.capacity() + self.wch.capacity()) + 4 * self.m5.capacity()
            + size_of::<Pattern>() * self.patterns.capacity()
            + self.patterns.iter().map(|p| 8 * p.changed.capacity()).sum::<usize>()
            + size_of::<Group>() * self.groups.capacity() + 32 * self.rcf.capacity()
            + 8
                * (self.wt.capacity() + self.zv.capacity() + self.prof.capacity()
                    + self.srow.capacity())
    }
}
#[cfg(feature = "gpu")]
impl Queue {
    /// `TG_MEM_STATS`: the pooled picks' allocations, roughly in MB.
    pub(super) fn mem_stats(&self) -> String {
        let pooled: usize = self.pool.iter().map(|p| p.heap_bytes()).sum();
        format!(
            "pool {} picks {:.0} MB, batch words {}", self.pool.len(), pooled as f64 /
            1e6, self.batch.words.capacity()
        )
    }
}
/// Prepared picks a queue keeps for reuse, not all of an item's thousands.
#[cfg(feature = "gpu")]
const POOL: usize = 64;
/// Bytes a queue's batch reaches before it goes to the GPU; `TG_SIX_QMB`
/// in MB.
#[cfg(feature = "gpu")]
fn queue_bytes() -> usize {
    static B: OnceLock<usize> = OnceLock::new();
    *B
        .get_or_init(|| {
            std::env::var("TG_SIX_QMB").ok().map_or(24, |v| v.parse().unwrap()) << 20
        })
}
/// The GPU's unfinished batches at which a pick's node loop goes to the CPU
/// instead; `TG_SIX_ROUTE`. It matters at an epoch's end, where the heavy
/// items leave the GPU behind: too high, and workers wait on it.
#[cfg(feature = "gpu")]
fn route_at() -> usize {
    static R: OnceLock<usize> = OnceLock::new();
    *R
        .get_or_init(|| {
            std::env::var("TG_SIX_ROUTE").ok().map_or(24, |v| v.parse().unwrap())
        })
}
/// The same for a pick of several code rows, a kernel pick a row: the GPU
/// takes it only while its backlog is under `TG_SIX_MULTI`.
#[cfg(feature = "gpu")]
fn multi_at() -> usize {
    static R: OnceLock<usize> = OnceLock::new();
    *R
        .get_or_init(|| {
            std::env::var("TG_SIX_MULTI").ok().map_or(8, |v| v.parse().unwrap())
        })
}
/// Batches a worker keeps in flight before it waits on the oldest;
/// `TG_SIX_DEPTH`.
#[cfg(feature = "gpu")]
pub(super) fn depth() -> usize {
    static D: OnceLock<usize> = OnceLock::new();
    *D
        .get_or_init(|| {
            std::env::var("TG_SIX_DEPTH").ok().map_or(2, |v| v.parse().unwrap())
        })
}
/// `--gpu` and not `TG_SIX_CPU`: slot 6's node loops on the GPU.
#[cfg(feature = "gpu")]
fn on_gpu() -> bool {
    static CPU: OnceLock<bool> = OnceLock::new();
    super::GPU.load(Ordering::Relaxed)
        && !*CPU.get_or_init(|| std::env::var_os("TG_SIX_CPU").is_some())
}
/// `TG_SIX_GPU_CHECK=1`: every GPU pick's A and Mom against the CPU's loop.
#[cfg(feature = "gpu")]
fn gpu_check() -> bool {
    static C: OnceLock<bool> = OnceLock::new();
    *C.get_or_init(|| std::env::var_os("TG_SIX_GPU_CHECK").is_some())
}
/// The worst relative gap the check saw — entries below 10⁻³⁰ of their
/// array's largest aside — with where, and the picks and entries compared.
#[cfg(feature = "gpu")]
static GPU_GAP: Mutex<(f64, String, u64, u64)> = Mutex::new((0.0, String::new(), 0, 0));
#[cfg(feature = "gpu")]
fn gpu_compare(cpu: &[f64], gpu: &[f64], what: &str, pp: &Prep) {
    let top = cpu.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    let mut g = GPU_GAP.lock().unwrap();
    g.2 += 1;
    let (nz, nzp) = (pp.nz, pp.nzp());
    for (i, (&c, &d)) in cpu.iter().zip(gpu).enumerate() {
        let size = c.abs().max(d.abs());
        if i % nzp >= nz || size <= 1e-30 * top {
            continue;
        }
        g.3 += 1;
        let rel = (c - d).abs() / size;
        if rel > g.0 || rel.is_nan() {
            g.0 = rel;
            g.1 = format!("{what}[{i}] cpu {c:e} gpu {d:e}; pick {:?}", pp.shape());
        }
    }
}
/// Workers waiting on a batch now, for `TG_TIMELINE`.
pub(super) static WAITING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(
    0,
);
/// The filling batch onto the GPU, a fresh one in its place.
#[cfg(feature = "gpu")]
fn submit(qu: &mut Queue) {
    let (mut batch, mut items) = qu.spare.pop().unwrap_or_default();
    std::mem::swap(&mut batch, &mut qu.batch);
    std::mem::swap(&mut items, &mut qu.items);
    GPUSTAT[1].fetch_add(batch.len() as u64, Ordering::Relaxed);
    let t0 = Instant::now();
    batch.submit();
    GPUSTAT[2].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    qu.flight.push_back((batch, items));
    qu.held = 0;
}
/// The batches in flight deposited, oldest first: every one the GPU has
/// finished, and the oldest waited for while more than `keep` remain.
#[cfg(feature = "gpu")]
fn settle<N: Num>(
    qu: &mut Queue,
    keep: usize,
    t: &Tables,
    b: &Bits,
    six: &mut Six,
    quad: &mut Quad,
    fold: &mut Weights<N>,
    table6: &Sharded<Ctx, KDeco<N>>,
) {
    while let Some((batch, _)) = qu.flight.front() {
        if qu.flight.len() <= keep && !batch.ready() {
            break;
        }
        let (mut batch, mut items) = qu.flight.pop_front().unwrap();
        let t0 = Instant::now();
        WAITING.fetch_add(1, Ordering::Relaxed);
        batch.wait();
        WAITING.fetch_sub(1, Ordering::Relaxed);
        GPUSTAT[0].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        deposit_batch::<N>(qu, &batch, &mut items, t, b, six, quad, fold, table6);
        batch.clear();
        qu.spare.push((batch, items));
    }
}
/// A worker's [`item`]s finished: on the GPU path the filling batch goes
/// too and every deposit is made.
pub(super) fn finish<N: Num>(
    t: &Tables,
    b: &Bits,
    l: &mut Local<N>,
    table6: &Sharded<Ctx, KDeco<N>>,
) {
    #[cfg(feature = "gpu")]
    {
        let Local { fold, quad, six, sixq, .. } = l;
        if !sixq.items.is_empty() {
            submit(sixq);
        }
        settle::<N>(sixq, 0, t, b, six, quad, fold, table6);
    }
    #[cfg(not(feature = "gpu"))]
    let _ = (t, b, l, table6);
}
/// A finished batch's picks deposited and its markers run, in order.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn deposit_batch<N: Num>(
    qu: &mut Queue,
    batch: &gpu::SixBatch,
    items: &mut Vec<Queued>,
    t: &Tables,
    b: &Bits,
    six: &mut Six,
    quad: &mut Quad,
    fold: &mut Weights<N>,
    table6: &Sharded<Ctx, KDeco<N>>,
) {
    static PREFETCH: OnceLock<bool> = OnceLock::new();
    let prefetch = *PREFETCH
        .get_or_init(|| std::env::var("TG_SIX_PREFETCH").map_or(true, |v| v != "0"));
    let mut it = items.drain(..).peekable();
    while let Some(item) = it.next() {
        match item {
            Queued::Flush => six.flush::<N>(t, fold),
            Queued::Table => six.flush_table::<N>(table6),
            Queued::Pick(pp, at) => {
                if prefetch && let Some(Queued::Pick(next, at)) = it.peek() {
                    next.prefetch();
                    if let Some(j) = *at {
                        let (na, nm) = next.out_lens();
                        batch.prefetch(j, na + nm);
                    }
                }
                let mut clock = Clock::new();
                match at {
                    Some(i) => {
                        let (na, nm) = pp.out_lens();
                        let (a, mom) = if pp.nkk() == 1 {
                            batch.outputs(i, na, nm)
                        } else {
                            let (nkn, nzp, np, npat) = (
                                pp.nkn(),
                                pp.nzp(),
                                pp.pb.len(),
                                pp.patterns.len(),
                            );
                            qu.ta.clear();
                            qu.ta.resize(npat * nkn * nzp, 0.0);
                            qu.tm.clear();
                            qu.tm.resize(npat * np * nkn * nzp, 0.0);
                            for k4 in 0..nkn {
                                let (ak, mk) = batch
                                    .outputs(i + k4, npat * nzp, npat * np * nzp);
                                for pi in 0..npat {
                                    qu.ta[(pi * nkn + k4) * nzp..(pi * nkn + k4 + 1) * nzp]
                                        .copy_from_slice(&ak[pi * nzp..(pi + 1) * nzp]);
                                    for q in 0..np {
                                        let (to, from) = (
                                            ((pi * np + q) * nkn + k4) * nzp,
                                            (pi * np + q) * nzp,
                                        );
                                        qu.tm[to..to + nzp].copy_from_slice(&mk[from..from + nzp]);
                                    }
                                }
                            }
                            if pp.nb > 0 {
                                pp.expand(&qu.ta, &qu.tm, &mut qu.ma, &mut qu.mm);
                                (qu.ma.as_slice(), qu.mm.as_slice())
                            } else {
                                (qu.ta.as_slice(), qu.tm.as_slice())
                            }
                        };
                        if gpu_check() {
                            node_loop(&pp, six, &mut clock);
                            gpu_compare(&six.a, a, "A", &pp);
                            gpu_compare(&six.mom, mom, "Mom", &pp);
                        }
                        deposit(t, b, &pp, a, mom, six, quad, &mut clock);
                    }
                    None => {
                        GPUSTAT[4].fetch_add(1, Ordering::Relaxed);
                        let t0 = Instant::now();
                        node_loop(&pp, six, &mut clock);
                        let why = if pp.nkk() != 1 {
                            0
                        } else if pp.pb.len() > GPU_NP || pp.patterns.len() > GPU_THREADS
                        {
                            1
                        } else {
                            2
                        };
                        CPUWHY[why].fetch_add(1, Ordering::Relaxed);
                        CPUWHY[3 + why]
                            .fetch_add(
                                t0.elapsed().as_nanos() as u64,
                                Ordering::Relaxed,
                            );
                        let (a, mom) = (
                            std::mem::take(&mut six.a),
                            std::mem::take(&mut six.mom),
                        );
                        deposit(t, b, &pp, &a, &mom, six, quad, &mut clock);
                        (six.a, six.mom) = (a, mom);
                    }
                }
                if qu.pool.len() < POOL {
                    qu.pool.push(pp);
                }
            }
        }
    }
    stat_flush();
}
/// Slot 6 for one (label group, third pick), [`pick`] per fourth pick.
pub(super) fn item<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    b: &Bits,
    group: &[(Key, Deco<N>)],
    o3: usize,
    l: &mut Local<N>,
    table6: &Sharded<Ctx, KDeco<N>>,
) {
    #[cfg(feature = "gpu")]
    let gpu = on_gpu();
    #[cfg(feature = "gpu")]
    let Local { fold, sc, sc3, sc4, fold_sc, quad, six, sixq, .. } = l;
    #[cfg(not(feature = "gpu"))]
    let Local { fold, sc, sc3, sc4, fold_sc, quad, six, .. } = l;
    let mut rep = group[0].0;
    rep.dead = [0; 2];
    let ctx2 = Ctx::of_key(t, &rep);
    let av2 = Avail::of_key(t, &rep);
    let total2 = options(t, &rep, &ctx2, &av2, sc);
    let f2 = total2 - sc.live.iter().map(|l| u32::from(l.alive)).sum::<u32>();
    let mut mask2 = [0u32; 64];
    for l in &sc.live {
        mask2[usize::from(l.mi)] = l.mask;
    }
    let alive_of = |k: &Key, base: u16| -> u8 {
        let bs = &t.bases[base as usize];
        bs.mu - k.dead_of(bs.mi)
    };
    let multi = |base: u16| base != NONE && t.bases[base as usize].mi != u8::MAX;
    let o = sc.opts[o3];
    let (class3, picked3, lead_pick3, base3) = pick_parts(t, &rep, o.pick);
    let flagged3 = matches!(o.pick, Pick::Forme(_)) && t.sp[picked3 as usize].flags;
    let mut outcomes3: Vec<(bool, u8, Vec<(usize, Deco<N>)>)> = Vec::new();
    let mut split = Vec::new();
    for (v, (k, deco)) in group.iter().enumerate() {
        let num = if multi(base3) { u32::from(alive_of(k, base3)) } else { o.num };
        if num == 0 {
            continue;
        }
        let m3 = f2
            + sc.live.iter().map(|l| u32::from(alive_of(k, l.base))).sum::<u32>();
        let pr = N::ratio(u64::from(num), u64::from(o.den) * u64::from(m3));
        split.clear();
        if flagged3 {
            kernel_split(t, ar, &rep, picked3, lead_pick3, deco, &mut split);
        } else {
            split.push((rep.tb, rep.fire, deco.clone()));
        }
        for (tb, fire, d) in split.drain(..) {
            let d: Deco<N> = d.into_iter().map(|(c, x)| (c, x * pr)).collect();
            match outcomes3.iter_mut().find(|e| e.0 == tb && e.1 == fire) {
                Some(e) => e.2.push((v, d)),
                None => outcomes3.push((tb, fire, vec![(v, d)])),
            }
        }
    }
    let paths = if group.len() > 4 || std::env::var_os("TG_FORCE_CP").is_some() {
        Some(cp_paths(t, ar, &rep, &mut Scratch::default()))
    } else {
        None
    };
    for (tb3, fire3, variants) in outcomes3 {
        let mut c3 = rep.with(class3);
        c3.lead = rep.lead || lead_pick3;
        c3.tb = tb3;
        c3.fire = fire3;
        let s = Situation::new(t, c3, mask2, group, &variants, sc3);
        let terms = paths
            .as_ref()
            .map(|paths| cp_terms(
                t,
                ar,
                &rep,
                paths,
                (picked3, lead_pick3, flagged3, base3, o.num, o.den),
                (tb3, fire3),
                &s.live3,
                total2,
                quad,
            ))
            .inspect(|terms| {
                if std::env::var_os("TG_CP_DEBUG").is_some() {
                    eprintln!(
                        "cp: group {} paths {} terms {} classes {}", group.len(), paths
                        .as_ref().map_or(0, | p | p.len()), terms.len(), s.keys.len()
                    );
                }
            })
            .filter(|terms| {
                static MODE: OnceLock<u8> = OnceLock::new();
                let mode = *MODE
                    .get_or_init(|| {
                        u8::from(std::env::var_os("TG_NO_CP").is_some())
                            + 2 * u8::from(std::env::var_os("TG_FORCE_CP").is_some())
                    });
                static RATIO: OnceLock<f64> = OnceLock::new();
                let ratio = *RATIO
                    .get_or_init(|| {
                        std::env::var("TG_CP_RATIO")
                            .ok()
                            .map_or(1.0, |v| v.parse().unwrap())
                    });
                !terms.is_empty() && mode != 1
                    && (mode == 2 || (terms.len() as f64) < ratio * s.keys.len() as f64)
            });
        for o4 in &sc3.opts {
            let (class4, picked4, lead_pick4, base4) = pick_parts(t, &s.c3, o4.pick);
            if !(s.c3.lead || lead_pick4) {
                continue;
            }
            let flagged4 = matches!(o4.pick, Pick::Forme(_))
                && t.sp[picked4 as usize].flags;
            let mut outcomes4: Vec<(bool, u8)> = Vec::new();
            if flagged4 {
                let mut out = Vec::new();
                kernel_split(t, ar, &s.c3, picked4, lead_pick4, &s.union, &mut out);
                outcomes4.extend(out.iter().map(|o| (o.0, o.1)));
            } else {
                outcomes4.push((s.c3.tb, s.c3.fire));
            }
            for (tb4, fire4) in outcomes4 {
                let mut c4 = s.c3.with(class4);
                c4.lead = true;
                c4.tb = tb4;
                c4.fire = fire4;
                let Some(p) = Pick5::new(
                    t,
                    ar,
                    &s,
                    c4,
                    o4,
                    picked4,
                    lead_pick4,
                    flagged4,
                    base4,
                    sc4,
                ) else { continue };
                #[cfg(feature = "gpu")]
                if gpu {
                    let mut pp = sixq.pool.pop().unwrap_or_default();
                    if prepare(
                        t,
                        ar,
                        b,
                        &s,
                        &p,
                        terms.as_deref(),
                        fold_sc,
                        six,
                        quad,
                        &mut pp,
                        &mut Clock::new(),
                    ) {
                        let tp = Instant::now();
                        let pending = gpu::six_pending();
                        let at = if pending >= route_at()
                            || (pp.nkk() > 1 && pending >= multi_at())
                        {
                            None
                        } else {
                            gpu_pack(&pp, &mut sixq.batch)
                        };
                        GPUSTAT[3]
                            .fetch_add(
                                tp.elapsed().as_nanos() as u64,
                                Ordering::Relaxed,
                            );
                        sixq.held += pp.heap_bytes();
                        sixq.items.push(Queued::Pick(pp, at));
                        if sixq.batch.bytes() + sixq.held >= queue_bytes() {
                            submit(sixq);
                            settle::<N>(sixq, depth(), t, b, six, quad, fold, table6);
                        }
                    } else {
                        sixq.pool.push(pp);
                    }
                    continue;
                }
                pick(t, ar, b, &s, &p, terms.as_deref(), fold_sc, six, quad);
            }
        }
        #[cfg(feature = "gpu")]
        if gpu {
            sixq.items.push(Queued::Flush);
            continue;
        }
        six.flush::<N>(t, fold);
    }
    #[cfg(feature = "gpu")]
    if gpu {
        sixq.items.push(Queued::Table);
        settle::<N>(sixq, depth(), t, b, six, quad, fold, table6);
        return;
    }
    six.flush_table::<N>(table6);
}
