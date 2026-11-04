//! `exact_team_weights`: the randbats team loop's weights, computed rather
//! than sampled. For every species, the probability that PS's `randomTeam`
//! accepts it, as the lead or not, in each party slot, under each
//! `teamDetails` context its set is rolled under. MATH.md derives what this
//! computes, and the lemmas cited below are its.
//!
//! ```text
//! cargo run --release [--features gpu] --bin exact_team_weights -- \
//!     [--slots | --max-size 1..3] [--exact BATCH] [--mem-gb 8] [--threads N] \
//!     [--self-check 2000] [--check-mc TEAMS] [--check-analytic STATES] \
//!     [--slot5 [--slot6]] [--gpu] [--limit STATES] [--partial-stats TEAMS] \
//!     [--out exact-team-weights.json]
//! cargo run --release --bin exact_team_weights -- \
//!     --mc-dump TEAMS [--mc-rb K] [--mc-seed S] --out COUNTS.json
//! ```
//!
//! Weights are by party slot, as PS returns the team: the lead `unshift`ed
//! to slot 1, every other member after the ones drawn before it. `--slots`
//! computes slots 1–4 exactly, `--slot5` adds slot 5 and `--slot6` slot 6
//! (`six.rs`), `--gpu` running their heavy loops on the GPU; the rows are
//! for the `slots` renderer (`src/bin/slots/`). `--max-size k` holds every
//! state through size k instead (slots 2..=k complete), which stops at
//! size 3 on memory; `--check-analytic` holds the integrated slot-4 fold to
//! the enumerated one.
//!
//! The arithmetic is f64, or with `--exact BATCH` residues modulo the
//! BATCH-th 32 primes below 2^63: every quantity the walk forms is a
//! rational with small factors, so the same walk computes either. Each
//! exact rational is rebuilt from enough batches by the Chinese remainder
//! theorem and rational reconstruction (MATH.md §5); this crate writes the
//! residues ([`write_residues`]) but does not rebuild them.
//!
//! The process is a random greedy walk over class multisets (Lemma 4), with
//! the flags a team has set carried beside each state as a distribution
//! (Lemma 5) rather than as part of it. A pool entry's forme label is fixed
//! when it is drawn and independent of the draw order, so the labels can be
//! drawn in advance; then every entry is admissible or not, and the next
//! acceptance is uniform over the entries whose label is admissible
//! (Corollary 3): a pick has probability weight / M. What a state must
//! remember is which labels are dead: when a team shrinks a multi-forme
//! base's admissible formes from F to F′ ≠ ∅, each of its live labels
//! survives with probability |F′|/|F| — a binomial split, and the only
//! branching besides the pick itself.
//!
//! Those dead labels make the states past size 2 too many to hold: a class
//! multiset's states multiply by ∏(r+1) over its partly admissible bases
//! (MATH.md, Lemma 6 and §4). So sizes 1–2 are held, and slot 4 —
//! the pick after a size-2 state's next one — is folded without the size-3
//! states it would pass through: that step's label splits are integrated,
//! E[1/M] and E[A/M] over independent binomials as one-variable polynomial
//! integrals, instead of enumerated. A fold credits a pick through a table
//! keyed by the team's admissibility signature — every species the
//! signature admits takes ω·E[1/M] of it — with exact corrections for the
//! species clause and the multi-forme bases. Slots 5 and 6 integrate the
//! splits of two and three steps jointly (MATH.md §§6–7).
#![allow(dead_code)]
mod simd;
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::ops::{Add, AddAssign, Mul, Sub};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::Instant;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use randbats_gen9::generated::ability_id::AbilityId;
use randbats_gen9::generated::move_id::MoveId;
use randbats_gen9::generated::randbats_data::{RANDBATS_BY_SPECIES, RoleId};
use randbats_gen9::generated::species_id::SpeciesId;
use randbats_gen9::randbats_gen::{
    RandomSet, TeamDetails, TeamGenState, enumerate_random_set,
    enumerate_random_set_exact, formes_of, never_leads, random_set, weighted_base_pool,
};
use randbats_gen9::species::SpeciesTypes;
use randbats_gen9::types::{Effectiveness, Type};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
#[cfg(feature = "gpu")]
mod gpu;
mod six;
const TEAM: u8 = 6;
const NONE: u16 = u16::MAX;
/// `teamDetails.teraBlast` in a context code. The other flags take bits
/// 0–10, Spikes (capped at 2: the generator reads only `>= 2`) bits 12–13.
const TB_BIT: u16 = 1 << 11;
const ALL18: [Type; 18] = [
    Type::Normal,
    Type::Fire,
    Type::Water,
    Type::Electric,
    Type::Grass,
    Type::Ice,
    Type::Fighting,
    Type::Poison,
    Type::Ground,
    Type::Flying,
    Type::Psychic,
    Type::Bug,
    Type::Rock,
    Type::Ghost,
    Type::Dragon,
    Type::Dark,
    Type::Steel,
    Type::Fairy,
];
const FIRE: usize = 1;
const G_CLONE_A: u8 = 1;
const G_CLONE_B: u8 = 2;
const G_WEB: u8 = 4;
const G_SCREEN: u8 = 8;
const G_TOXICROAK: u8 = 16;
const G_SUN: u8 = 32;
#[derive(Default)]
struct Fx(u64);
impl Fx {
    #[inline]
    fn add(&mut self, x: u64) {
        self.0 = (self.0.rotate_left(5) ^ x).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}
impl Hasher for Fx {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.add(u64::from(b));
        }
    }
    fn write_u8(&mut self, i: u8) {
        self.add(u64::from(i));
    }
    fn write_u16(&mut self, i: u16) {
        self.add(u64::from(i));
    }
    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
}
type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;
fn fxhash<K: Hash>(k: &K) -> u64 {
    let mut h = Fx::default();
    k.hash(&mut h);
    let x = h.finish();
    (x ^ (x >> 29)).wrapping_mul(0xbf58_476d_1ce4_e5b9) ^ (x >> 32)
}
struct Sharded<K, V> {
    shards: Vec<Mutex<FxMap<K, V>>>,
}
impl<K: Hash + Eq, V> Sharded<K, V> {
    fn new(n: usize) -> Self {
        Self {
            shards: (0..n).map(|_| Mutex::new(FxMap::default())).collect(),
        }
    }
    fn shard(&self, h: u64) -> &Mutex<FxMap<K, V>> {
        &self.shards[((h >> 40) as usize) % self.shards.len()]
    }
    fn len(&self) -> usize {
        self.shards.iter().map(|s| s.lock().unwrap().len()).sum()
    }
    fn into_vec(self) -> Vec<(K, V)> {
        self.shards.into_iter().flat_map(|s| s.into_inner().unwrap()).collect()
    }
}
/// The arithmetic the walk runs in: f64 for the float files, residues modulo
/// a batch of primes for the exact ones. Everything the walk forms is a
/// rational with small factors — w/M, a split's binomial, ω = μ/φ, a
/// polynomial integral, a set's branch probabilities — so both run it
/// unchanged.
trait Num: Copy + Default + Send + Sync + Add<
        Output = Self,
    > + Sub<Output = Self> + Mul<Output = Self> + 'static {
    /// A sum of many values: compensated, for floats.
    type Acc: Copy + Default + Send + Sync + AddAssign<Self> + AddAssign<Self::Acc>;
    fn ratio(n: u64, d: u64) -> Self;
    fn get(acc: Self::Acc) -> Self;
    /// One set table, each set's probability in this arithmetic.
    fn table(
        species: SpeciesId,
        details: &TeamDetails,
        lead: bool,
    ) -> Vec<(RandomSet, Self)>;
    /// Whether `a` and `b` agree: to rounding for floats, exactly for residues.
    fn same(a: Self, b: Self) -> bool;
    fn show(self) -> String;
    /// The complete slots' weights, for the renderer.
    fn write(t: &Tables, weights: &Weights<Self>, slots: &[u8], path: &str);
    /// The weights against sampled teams, where that means anything.
    fn check_mc(
        t: &Tables,
        weights: &Weights<Self>,
        slots: &[u8],
        teams: u64,
        threads: usize,
    );
    /// Whether slot 5 may integrate by quadrature, whose nodes are
    /// irrational: floats only, and then `of_f64` is the identity.
    const QUADRATURE: bool = false;
    fn of_f64(_: f64) -> Self {
        unreachable!("quadrature is for floats")
    }
    fn to_f64(self) -> f64 {
        unreachable!("quadrature is for floats")
    }
}
/// A Neumaier-compensated sum: a weight cell or a signature's context adds
/// up millions of terms.
#[derive(Default, Clone, Copy)]
struct Sum {
    s: f64,
    c: f64,
}
impl AddAssign<f64> for Sum {
    fn add_assign(&mut self, x: f64) {
        let t = self.s + x;
        self.c
            += if self.s.abs() >= x.abs() { (self.s - t) + x } else { (x - t) + self.s };
        self.s = t;
    }
}
impl AddAssign<Sum> for Sum {
    fn add_assign(&mut self, o: Sum) {
        *self += o.s;
        *self += o.c;
    }
}
impl Num for f64 {
    type Acc = Sum;
    const QUADRATURE: bool = true;
    fn of_f64(x: f64) -> f64 {
        x
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn ratio(n: u64, d: u64) -> f64 {
        n as f64 / d as f64
    }
    fn get(acc: Sum) -> f64 {
        acc.s + acc.c
    }
    fn table(
        species: SpeciesId,
        details: &TeamDetails,
        lead: bool,
    ) -> Vec<(RandomSet, f64)> {
        enumerate_random_set(species, details, lead)
            .into_iter()
            .map(|o| (o.set, o.prob))
            .collect()
    }
    fn same(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-10 * a.abs().max(b.abs()) || (a - b).abs() < 1e-22
    }
    fn show(self) -> String {
        format!("{self:.15}")
    }
    fn write(t: &Tables, weights: &Weights<f64>, slots: &[u8], path: &str) {
        write_rows(t, weights, slots, path);
    }
    fn check_mc(
        t: &Tables,
        weights: &Weights<f64>,
        slots: &[u8],
        teams: u64,
        threads: usize,
    ) {
        check_mc(t, weights, slots, teams, threads);
    }
}
/// Residues modulo `LANES` primes below 2^63, in Montgomery form: one batch
/// of the exact computation.
const LANES: usize = 32;
/// Small integers and their inverses are tabled up to here: every
/// denominator the walk forms is below it, and nearly every numerator.
const SMALL: u64 = 1 << 14;
#[derive(Clone, Copy, PartialEq, Eq)]
struct Fp([u64; LANES]);
impl Default for Fp {
    fn default() -> Self {
        Fp([0; LANES])
    }
}
#[derive(Clone, Copy)]
struct Prime {
    p: u64,
    /// −p⁻¹ mod 2^64.
    pinv: u64,
    /// 2^128 mod p.
    r2: u64,
}
struct Field {
    primes: [Prime; LANES],
    small: Vec<Fp>,
    inv: Vec<Fp>,
}
static FIELD: OnceLock<Field> = OnceLock::new();
fn field() -> &'static Field {
    FIELD.get().expect("Field::init before any residue")
}
#[inline(always)]
fn mont_mul(a: u64, b: u64, pr: &Prime) -> u64 {
    let t = u128::from(a) * u128::from(b);
    let m = (t as u64).wrapping_mul(pr.pinv);
    let u = ((t + u128::from(m) * u128::from(pr.p)) >> 64) as u64;
    if u >= pr.p { u - pr.p } else { u }
}
fn mulmod(a: u64, b: u64, m: u64) -> u64 {
    (u128::from(a) * u128::from(b) % u128::from(m)) as u64
}
fn powmod(mut a: u64, mut e: u64, m: u64) -> u64 {
    let mut r = 1;
    while e > 0 {
        if e & 1 == 1 {
            r = mulmod(r, a, m);
        }
        a = mulmod(a, a, m);
        e >>= 1;
    }
    r
}
/// Miller–Rabin with the first twelve primes as bases: deterministic below
/// 3.3·10^24.
fn is_prime(n: u64) -> bool {
    const BASES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    if n < 2 || BASES.contains(&n) {
        return BASES.contains(&n);
    }
    if BASES.iter().any(|&b| n % b == 0) {
        return false;
    }
    let (mut d, mut s) = (n - 1, 0);
    while d % 2 == 0 {
        d /= 2;
        s += 1;
    }
    BASES
        .iter()
        .all(|&a| {
            let mut x = powmod(a, d, n);
            if x == 1 || x == n - 1 {
                return true;
            }
            for _ in 1..s {
                x = mulmod(x, x, n);
                if x == n - 1 {
                    return true;
                }
            }
            false
        })
}
impl Field {
    /// Batch `batch`: the primes below 2^63, counted down, from the
    /// `batch·LANES`-th on.
    fn init(batch: usize) {
        let mut ps = Vec::new();
        let mut n = (1u64 << 63) - 1;
        while ps.len() < (batch + 1) * LANES {
            if is_prime(n) {
                ps.push(n);
            }
            n -= 2;
        }
        let primes: [Prime; LANES] = std::array::from_fn(|i| {
            let p = ps[batch * LANES + i];
            let mut inv = p;
            for _ in 0..5 {
                inv = inv.wrapping_mul(2u64.wrapping_sub(p.wrapping_mul(inv)));
            }
            let r = ((1u128 << 64) % u128::from(p)) as u64;
            Prime {
                p,
                pinv: inv.wrapping_neg(),
                r2: mulmod(r, r, p),
            }
        });
        let small: Vec<Fp> = (0..SMALL)
            .map(|n| Fp(std::array::from_fn(|i| mont_mul(n, primes[i].r2, &primes[i]))))
            .collect();
        let inv: Vec<Fp> = (0..SMALL)
            .map(|n| {
                if n == 0 {
                    Fp::default()
                } else {
                    Fp(
                        std::array::from_fn(|i| mont_pow(
                            small[n as usize].0[i],
                            primes[i].p - 2,
                            &primes[i],
                        )),
                    )
                }
            })
            .collect();
        assert!(FIELD.set(Field { primes, small, inv }).is_ok(), "one batch a run");
    }
}
fn mont_pow(mut a: u64, mut e: u64, pr: &Prime) -> u64 {
    let mut r = mont_mul(1, pr.r2, pr);
    while e > 0 {
        if e & 1 == 1 {
            r = mont_mul(r, a, pr);
        }
        a = mont_mul(a, a, pr);
        e >>= 1;
    }
    r
}
impl Fp {
    fn of_int(x: u64) -> Fp {
        let f = field();
        if x < SMALL {
            return f.small[x as usize];
        }
        Fp(
            std::array::from_fn(|i| mont_mul(
                x % f.primes[i].p,
                f.primes[i].r2,
                &f.primes[i],
            )),
        )
    }
    fn inverse_of_int(x: u64) -> Fp {
        let f = field();
        if x < SMALL {
            return f.inv[x as usize];
        }
        let m = Fp::of_int(x);
        Fp(std::array::from_fn(|i| mont_pow(m.0[i], f.primes[i].p - 2, &f.primes[i])))
    }
    /// Each lane out of Montgomery form.
    fn canonical(self) -> [u64; LANES] {
        let f = field();
        std::array::from_fn(|i| mont_mul(self.0[i], 1, &f.primes[i]))
    }
}
impl Add for Fp {
    type Output = Fp;
    #[inline]
    fn add(self, o: Fp) -> Fp {
        let f = field();
        Fp(
            std::array::from_fn(|i| {
                let s = self.0[i] + o.0[i];
                if s >= f.primes[i].p { s - f.primes[i].p } else { s }
            }),
        )
    }
}
impl Sub for Fp {
    type Output = Fp;
    #[inline]
    fn sub(self, o: Fp) -> Fp {
        let f = field();
        Fp(
            std::array::from_fn(|i| {
                if self.0[i] >= o.0[i] {
                    self.0[i] - o.0[i]
                } else {
                    self.0[i] + f.primes[i].p - o.0[i]
                }
            }),
        )
    }
}
impl Mul for Fp {
    type Output = Fp;
    #[inline]
    fn mul(self, o: Fp) -> Fp {
        let f = field();
        Fp(std::array::from_fn(|i| mont_mul(self.0[i], o.0[i], &f.primes[i])))
    }
}
impl AddAssign for Fp {
    #[inline]
    fn add_assign(&mut self, o: Fp) {
        *self = *self + o;
    }
}
impl Num for Fp {
    type Acc = Fp;
    fn ratio(n: u64, d: u64) -> Fp {
        Fp::of_int(n) * Fp::inverse_of_int(d)
    }
    fn get(acc: Fp) -> Fp {
        acc
    }
    fn table(
        species: SpeciesId,
        details: &TeamDetails,
        lead: bool,
    ) -> Vec<(RandomSet, Fp)> {
        enumerate_random_set_exact(
            species,
            details,
            lead,
            |factors| {
                factors
                    .fold(
                        Fp::ratio(1, 1),
                        |p, (n, d)| p * Fp::ratio(u64::from(n), u64::from(d)),
                    )
            },
            |a, b| *a = *a + b,
        )
    }
    fn same(a: Fp, b: Fp) -> bool {
        a == b
    }
    fn show(self) -> String {
        if self == Fp::ratio(1, 1) {
            "1 exactly".into()
        } else {
            format!(
                "≠ 1 (residue {} mod {})", self.canonical() [0], field().primes[0].p
            )
        }
    }
    fn write(t: &Tables, weights: &Weights<Fp>, slots: &[u8], path: &str) {
        write_residues(t, weights, slots, path);
    }
    fn check_mc(_: &Tables, _: &Weights<Fp>, _: &[u8], _: u64, _: usize) {
        println!(
            "--check-mc skipped: residues cannot be compared with counts (run it on the float weights)"
        );
    }
}
struct Sp {
    id: SpeciesId,
    name: &'static str,
    base: u16,
    types: u32,
    weak: u32,
    dweak: u32,
    fd: bool,
    l100: bool,
    fire_neutral: bool,
    /// Fire-neutral with Dry Skin / Fluffy among its dex abilities: the Fire
    /// weakness gate counts it (`admits_composition`).
    fire_gate: bool,
    compat: u8,
    tera_blocked: bool,
    zoroark: bool,
    no_lead: bool,
    /// Can its set change `teamDetails` or the Fire count. A superset is
    /// safe; a species left out would carry no flags it should.
    flags: bool,
    class: u16,
}
struct Base {
    formes: Vec<u16>,
    mu: u8,
    phi: u8,
    simple: bool,
    /// Its index among the multi-forme bases (its slot in `Key::dead`).
    mi: u8,
}
struct Class {
    members: Vec<u16>,
    simple: bool,
}
struct Tables {
    sp: Vec<Sp>,
    bases: Vec<Base>,
    classes: Vec<Class>,
    simple_classes: Vec<u16>,
    /// Single-forme bases kept apart from the simple classes.
    single_bases: Vec<u16>,
    multi: Vec<u16>,
}
fn ps_eff(att: Type, t: SpeciesTypes) -> i32 {
    let s = |d: Type| match att.effectiveness_against(d) {
        Effectiveness::SuperEffective => 1,
        Effectiveness::Resistant => -1,
        _ => 0,
    };
    s(t.0) + t.1.map_or(0, s)
}
fn type_bit(t: Type) -> u32 {
    1 << ALL18.iter().position(|&x| x == t).expect("a standard type")
}
const FLAG_MOVES: &[MoveId] = &[
    MoveId::Raindance,
    MoveId::Sunnyday,
    MoveId::Snowscape,
    MoveId::Chillyreception,
    MoveId::Healbell,
    MoveId::Spikes,
    MoveId::Ceaselessedge,
    MoveId::Toxicspikes,
    MoveId::Stealthrock,
    MoveId::Stoneaxe,
    MoveId::Stickyweb,
    MoveId::Defog,
    MoveId::Rapidspin,
    MoveId::Mortalspin,
    MoveId::Auroraveil,
    MoveId::Reflect,
    MoveId::Lightscreen,
];
const FLAG_ABILITIES: &[AbilityId] = &[
    AbilityId::Drizzle,
    AbilityId::Drought,
    AbilityId::Orichalcumpulse,
    AbilityId::Sandstream,
    AbilityId::Snowwarning,
    AbilityId::Toxicdebris,
    AbilityId::Dryskin,
    AbilityId::Fluffy,
];
fn build_tables() -> Tables {
    use SpeciesId as S;
    let group = |s: SpeciesId| -> u8 {
        let mut g = 0;
        if matches!(s, S::Blissey | S::Chansey) {
            g |= G_CLONE_A;
        }
        if matches!(s, S::Illumise | S::Volbeat) {
            g |= G_CLONE_B;
        }
        if matches!(
            s, S::Ariados | S::Smeargle | S::Masquerain | S::Kricketune | S::Leavanny |
            S::Galvantula | S::Vikavolt | S::Ribombee | S::Araquanid | S::Spidops
        ) {
            g |= G_WEB;
        }
        if matches!(s, S::Meowstic | S::Grimmsnarl | S::Ninetalesalola | S::Abomasnow) {
            g |= G_SCREEN;
        }
        if s == S::Toxicroak {
            g |= G_TOXICROAK;
        }
        if matches!(s, S::Ninetales | S::Torkoal | S::Groudon | S::Koraidon) {
            g |= G_SUN;
        }
        g
    };
    let mut bases: Vec<Base> = Vec::new();
    let mut base_of: FxMap<SpeciesId, u16> = FxMap::default();
    for (base, mu) in randbats_gen9::randbats_gen::weighted_base_runs() {
        let formes = formes_of(base);
        if formes.is_empty() {
            continue;
        }
        for &f in formes {
            base_of.insert(f, bases.len() as u16);
        }
        bases
            .push(Base {
                formes: Vec::new(),
                mu: mu as u8,
                phi: formes.len() as u8,
                simple: false,
                mi: u8::MAX,
            });
    }
    let mut sp: Vec<Sp> = Vec::new();
    for e in RANDBATS_BY_SPECIES.iter().flatten() {
        let s = e.species;
        let Some(&base) = base_of.get(&s) else { continue };
        let types = s.data().types;
        let (mut weak, mut dweak) = (0u32, 0u32);
        for &t in &ALL18 {
            let x = ps_eff(t, types);
            if x > 0 {
                weak |= type_bit(t);
            }
            if x > 1 {
                dweak |= type_bit(t);
            }
        }
        let ice = ps_eff(Type::Ice, types);
        let water = types.0 == Type::Water || types.1 == Some(Type::Water);
        let abilities = s.data().abilities;
        let dex: Vec<AbilityId> = [
            Some(abilities.slot_1),
            abilities.slot_2,
            abilities.hidden,
            abilities.special,
        ]
            .into_iter()
            .flatten()
            .collect();
        let fire_neutral = ps_eff(Type::Fire, types) == 0;
        let tera_blocked = matches!(
            s, S::Ogerpon | S::Ogerponhearthflame | S::Terapagos
        );
        let flags = tera_blocked
            || e
                .roles
                .iter()
                .any(|r| {
                    r.id == RoleId::TeraBlastUser
                        || r.moves.iter().any(|m| FLAG_MOVES.contains(&m.id))
                        || r.abilities.iter().any(|a| FLAG_ABILITIES.contains(&a.id))
                });
        sp.push(Sp {
            id: s,
            name: s.data().name,
            base,
            types: type_bit(types.0) | types.1.map_or(0, type_bit),
            weak,
            dweak,
            fd: ice > 0 || (ice > -2 && water),
            l100: e.level == 100,
            fire_neutral,
            fire_gate: fire_neutral
                && dex
                    .iter()
                    .any(|a| matches!(a, AbilityId::Dryskin | AbilityId::Fluffy)),
            compat: group(s),
            tera_blocked,
            zoroark: s.data().base_species == S::Zoroark,
            no_lead: never_leads(s),
            flags,
            class: 0,
        });
    }
    let idx: FxMap<SpeciesId, u16> = sp
        .iter()
        .enumerate()
        .map(|(i, s)| (s.id, i as u16))
        .collect();
    for (base, _) in randbats_gen9::randbats_gen::weighted_base_runs() {
        if let Some(&b) = formes_of(base).first().and_then(|f| base_of.get(f)) {
            bases[b as usize].formes = formes_of(base).iter().map(|f| idx[f]).collect();
        }
    }
    let mut classes: Vec<Class> = Vec::new();
    let mut simple_key: FxMap<(u32, bool), u16> = FxMap::default();
    for i in 0..sp.len() {
        let s = &sp[i];
        let b = &bases[s.base as usize];
        let simple = b.phi == 1 && s.compat == 0 && !s.tera_blocked && !s.zoroark
            && !s.no_lead && !s.fire_gate && !s.flags;
        let c = if simple {
            *simple_key
                .entry((s.types, s.l100))
                .or_insert_with(|| {
                    classes
                        .push(Class {
                            members: Vec::new(),
                            simple: true,
                        });
                    (classes.len() - 1) as u16
                })
        } else {
            classes
                .push(Class {
                    members: Vec::new(),
                    simple: false,
                });
            (classes.len() - 1) as u16
        };
        classes[c as usize].members.push(i as u16);
        sp[i].class = c;
    }
    let mut multi = Vec::new();
    for (i, b) in bases.iter_mut().enumerate() {
        b.simple = b.formes.len() == 1
            && classes[sp[b.formes[0] as usize].class as usize].simple;
        b.mi = u8::MAX;
        if b.formes.len() > 1 {
            assert!(
                b.formes.len() <= 32 && b.mu <= 3,
                "a forme mask is a u32 and a dead count two bits"
            );
            b.mi = multi.len() as u8;
            multi.push(i as u16);
        }
    }
    assert!(
        multi.len() <= 64, "{} multi-forme bases: `Key::dead` holds 64", multi.len()
    );
    let simple_classes = (0..classes.len() as u16)
        .filter(|&c| classes[c as usize].simple)
        .collect();
    let single_bases = (0..bases.len() as u16)
        .filter(|&b| !bases[b as usize].simple && bases[b as usize].formes.len() == 1)
        .collect();
    Tables {
        sp,
        bases,
        classes,
        simple_classes,
        single_bases,
        multi,
    }
}
fn details_of(code: u16) -> TeamDetails {
    let bit = |i: u16| code & (1 << i) != 0;
    TeamDetails {
        rain: bit(0),
        sun: bit(1),
        sand: bit(2),
        snow: bit(3),
        stealth_rock: bit(4),
        toxic_spikes: bit(5),
        sticky_web: bit(6),
        rapid_spin: bit(7),
        defog: bit(8),
        screens: bit(9),
        status_cure: bit(10),
        tera_blast: bit(11),
        spikes: ((code >> 12) & 3) as u8,
    }
}
fn code_of(d: &TeamDetails) -> u16 {
    let flags = [
        d.rain,
        d.sun,
        d.sand,
        d.snow,
        d.stealth_rock,
        d.toxic_spikes,
        d.sticky_web,
        d.rapid_spin,
        d.defog,
        d.screens,
        d.status_cure,
        d.tera_blast,
    ];
    let mut code = 0u16;
    for (i, &f) in flags.iter().enumerate() {
        code |= u16::from(f) << i;
    }
    code | (u16::from(d.spikes.min(2)) << 12)
}
/// A context code's flag bits, in order.
const CONTEXT_FLAGS: [&str; 12] = [
    "rain",
    "sun",
    "sand",
    "snow",
    "stealthRock",
    "toxicSpikes",
    "stickyWeb",
    "rapidSpin",
    "defog",
    "screens",
    "statusCure",
    "teraBlast",
];
fn code_label(code: u16) -> String {
    let mut parts: Vec<String> = (0..12)
        .filter(|&i| code & (1 << i) != 0)
        .map(|i| CONTEXT_FLAGS[i].to_string())
        .collect();
    match (code >> 12) & 3 {
        0 => {}
        n => parts.push(format!("spikes{n}")),
    }
    if parts.is_empty() { "none".into() } else { parts.join("+") }
}
/// One outcome of a member's set: the context after it (its `teamDetails`
/// flag stripped), the flag itself, and whether it adds a Fire weakness.
#[derive(Clone, Copy)]
struct KOut<N> {
    code: u16,
    tb: bool,
    fire: u8,
    p: N,
}
/// What the walk needs in its arithmetic: each species' ω = μ/φ, its share
/// of a uniform pool draw, and its set tables folded to the context they
/// leave behind.
struct Arith<N: Num> {
    omega: Vec<N>,
    kernels: Vec<RwLock<FxMap<(bool, u16), Arc<Vec<KOut<N>>>>>>,
    misses: AtomicU64,
}
impl<N: Num> Arith<N> {
    fn new(t: &Tables) -> Self {
        Arith {
            omega: t
                .sp
                .iter()
                .map(|s| N::ratio(
                    u64::from(t.bases[s.base as usize].mu),
                    u64::from(t.bases[s.base as usize].phi),
                ))
                .collect(),
            kernels: (0..t.sp.len()).map(|_| RwLock::new(FxMap::default())).collect(),
            misses: AtomicU64::new(0),
        }
    }
    fn kernel(&self, t: &Tables, s: u16, lead: bool, full: u16) -> Arc<Vec<KOut<N>>> {
        let cell = &self.kernels[s as usize];
        if let Some(k) = cell.read().unwrap().get(&(lead, full)) {
            return k.clone();
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        let sp = &t.sp[s as usize];
        let details = details_of(full);
        let mut out: Vec<KOut<N>> = Vec::new();
        for (set, p) in N::table(sp.id, &details, lead) {
            let mut g = TeamGenState::default();
            g.record(&set);
            let mut after = details.clone();
            after.merge(&g.team_details);
            let code = code_of(&after);
            let fire = u8::from(
                matches!(set.ability, AbilityId::Dryskin | AbilityId::Fluffy)
                    && sp.fire_neutral,
            );
            let (tb, code) = (code & TB_BIT != 0, code & !TB_BIT);
            match out.iter_mut().find(|k| k.code == code && k.tb == tb && k.fire == fire)
            {
                Some(k) => k.p = k.p + p,
                None => out.push(KOut { code, tb, fire, p }),
            }
        }
        let k = Arc::new(out);
        cell.write().unwrap().insert((lead, full), k.clone());
        k
    }
}
/// A partial team as Lemma 6 needs it: its classes, whether the lead is
/// placed, the Tera Blast flag (it gates Ogerpon and Terapagos), the Fire
/// weaknesses Dry Skin / Fluffy added, and, per partly admissible
/// multi-forme base, how many of its entries carry a label already dead
/// (two bits a base, by `Base::mi`; zero once the base is out of play).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Key {
    m: [u16; 5],
    n: u8,
    lead: bool,
    tb: bool,
    fire: u8,
    dead: [u64; 2],
}
impl Key {
    fn empty() -> Self {
        Key {
            m: [NONE; 5],
            n: 0,
            lead: false,
            tb: false,
            fire: 0,
            dead: [0; 2],
        }
    }
    fn dead_of(&self, mi: u8) -> u8 {
        ((self.dead[usize::from(mi / 32)] >> (2 * (mi % 32))) & 3) as u8
    }
    fn set_dead(&mut self, mi: u8, n: u8) {
        let (word, shift) = (usize::from(mi / 32), 2 * (mi % 32));
        self.dead[word] = (self.dead[word] & !(3 << shift)) | (u64::from(n) << shift);
    }
    fn dead_bases(&self) -> u32 {
        self.dead
            .iter()
            .map(|&w| ((w | (w >> 1)) & 0x5555_5555_5555_5555).count_ones())
            .sum()
    }
    fn with(&self, class: u16) -> Key {
        let mut k = *self;
        let mut i = k.n as usize;
        while i > 0 && k.m[i - 1] > class {
            k.m[i] = k.m[i - 1];
            i -= 1;
        }
        k.m[i] = class;
        k.n += 1;
        k
    }
    fn members(&self) -> &[u16] {
        &self.m[..self.n as usize]
    }
}
type Deco<N> = Vec<(u16, N)>;
type KDeco<N> = Vec<(u16, <N as Num>::Acc)>;
#[inline]
fn deco_add<N: Num>(d: &mut Deco<N>, code: u16, w: N) {
    match d.iter_mut().find(|e| e.0 == code) {
        Some(e) => e.1 = e.1 + w,
        None => d.push((code, w)),
    }
}
fn kdeco_add<N: Num>(d: &mut KDeco<N>, code: u16, w: N) {
    match d.iter_mut().find(|e| e.0 == code) {
        Some(e) => e.1 += w,
        None => {
            let mut s = N::Acc::default();
            s += w;
            d.push((code, s));
        }
    }
}
/// The caps a partial team has filled, and the rules that read its size:
/// everything admissibility reads except the species clause. At size 5 it
/// is the sixth pick's signature.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Ctx {
    n: u8,
    lead: bool,
    tb: bool,
    types_full: u32,
    weak_full: u32,
    dweak_full: u32,
    fd_full: bool,
    l100_full: bool,
    fire_full: bool,
    compat: u8,
}
impl Ctx {
    fn new(
        t: &Tables,
        members: impl Iterator<Item = u16>,
        n: u8,
        lead: bool,
        tb: bool,
        fire: u8,
    ) -> Ctx {
        let (mut tc, mut wc, mut dc) = ([0u8; 18], [0u8; 18], [0u8; 18]);
        let (mut fd, mut l100, mut compat) = (0u8, 0u8, 0u8);
        for s in members {
            let s = &t.sp[s as usize];
            for i in 0..18 {
                let bit = 1u32 << i;
                tc[i] += u8::from(s.types & bit != 0);
                wc[i] += u8::from(s.weak & bit != 0);
                dc[i] += u8::from(s.dweak & bit != 0);
            }
            fd += u8::from(s.fd);
            l100 += u8::from(s.l100);
            compat |= s.compat;
        }
        wc[FIRE] += fire;
        let mask = |c: &[u8; 18], cap: u8| {
            (0..18).filter(|&i| c[i] >= cap).fold(0u32, |m, i| m | 1 << i)
        };
        Ctx {
            n,
            lead,
            tb,
            types_full: mask(&tc, 2),
            weak_full: mask(&wc, 3),
            dweak_full: mask(&dc, 1),
            fd_full: fd >= 4,
            l100_full: l100 >= 1,
            fire_full: wc[FIRE] >= 3,
            compat,
        }
    }
    fn of_key(t: &Tables, k: &Key) -> Ctx {
        let reps = k.members().iter().map(|&c| t.classes[c as usize].members[0]);
        Ctx::new(t, reps, k.n, k.lead, k.tb, k.fire)
    }
    /// Admissible but for the species clause (`admits` plus the loop's
    /// no-lead rule).
    #[inline]
    fn adm(&self, s: &Sp) -> bool {
        if s.types & self.types_full != 0 || s.weak & self.weak_full != 0
            || s.dweak & self.dweak_full != 0
        {
            return false;
        }
        if (self.fire_full && s.fire_gate) || (s.fd && self.fd_full)
            || (s.l100 && self.l100_full)
        {
            return false;
        }
        if s.compat & self.compat & (G_CLONE_A | G_CLONE_B | G_WEB | G_SCREEN) != 0
            || (s.compat & G_TOXICROAK != 0 && self.compat & G_SUN != 0)
            || (s.compat & G_SUN != 0 && self.compat & G_TOXICROAK != 0)
        {
            return false;
        }
        if s.tera_blocked && self.tb {
            return false;
        }
        if s.zoroark && self.n + 1 >= TEAM {
            return false;
        }
        !(s.no_lead && !self.lead && self.n + 1 == TEAM)
    }
}
/// The species clause and what is left of each base, for one key.
#[derive(Clone, Copy)]
struct Avail {
    used_bases: [u16; 5],
    class_counts: [(u16, u8); 5],
}
impl Avail {
    fn of_key(t: &Tables, k: &Key) -> Avail {
        let mut a = Avail {
            used_bases: [NONE; 5],
            class_counts: [(NONE, 0); 5],
        };
        for (i, &c) in k.members().iter().enumerate() {
            let class = &t.classes[c as usize];
            if class.simple {
                match a.class_counts.iter_mut().find(|e| e.0 == c) {
                    Some(e) => e.1 += 1,
                    None => {
                        *a.class_counts.iter_mut().find(|e| e.0 == NONE).unwrap() = (
                            c,
                            1,
                        );
                    }
                }
            } else {
                a.used_bases[i] = t.sp[class.members[0] as usize].base;
            }
        }
        a
    }
    fn base_used(&self, b: u16) -> bool {
        self.used_bases.contains(&b)
    }
    fn class_used(&self, c: u16) -> u8 {
        self.class_counts.iter().find(|e| e.0 == c).map_or(0, |e| e.1)
    }
}
#[derive(Clone, Copy)]
enum Pick {
    Class(u16),
    Forme(u16),
}
/// A pick and its weight `num/den`: a simple class's unused members, a
/// single-forme base's entries, or a multi-forme base's live entries over
/// its admissible formes.
#[derive(Clone, Copy)]
struct Opt {
    pick: Pick,
    num: u32,
    den: u32,
}
/// An unused multi-forme base with a forme still admissible: which formes
/// (a mask over `Base::formes`), and how many of its entries' labels are
/// still alive, each uniform over those formes.
#[derive(Clone, Copy)]
struct Live {
    base: u16,
    mi: u8,
    mask: u32,
    alive: u8,
}
#[derive(Default)]
struct Scratch {
    opts: Vec<Opt>,
    live: Vec<Live>,
}
fn forme_mask(t: &Tables, ctx: &Ctx, b: u16) -> u32 {
    let formes = &t.bases[b as usize].formes;
    (0..formes.len())
        .filter(|&i| ctx.adm(&t.sp[formes[i] as usize]))
        .fold(0, |m, i| m | 1 << i)
}
/// Every admissible pick from `k` with its weight, and the total M — the
/// live entries, a whole number. With the labels drawn in advance, the next
/// acceptance is uniform over the live entries (Corollary 3): a pick's
/// probability is its weight over M.
fn options(t: &Tables, k: &Key, ctx: &Ctx, av: &Avail, sc: &mut Scratch) -> u32 {
    options_into(t, k, ctx, av, &mut sc.live, Some(&mut sc.opts))
}
/// [`options`] into `live` and, when it is asked for, `opts`.
fn options_into(
    t: &Tables,
    k: &Key,
    ctx: &Ctx,
    av: &Avail,
    live: &mut Vec<Live>,
    mut opts: Option<&mut Vec<Opt>>,
) -> u32 {
    if let Some(o) = opts.as_deref_mut() {
        o.clear();
    }
    live.clear();
    let mut total = 0u32;
    for &c in &t.simple_classes {
        let class = &t.classes[c as usize];
        let left = class.members.len() as u8 - av.class_used(c);
        if left > 0 && ctx.adm(&t.sp[class.members[0] as usize]) {
            total += u32::from(left);
            if let Some(o) = opts.as_deref_mut() {
                o.push(Opt {
                    pick: Pick::Class(c),
                    num: u32::from(left),
                    den: 1,
                });
            }
        }
    }
    for &b in &t.single_bases {
        let base = &t.bases[b as usize];
        if !av.base_used(b) && ctx.adm(&t.sp[base.formes[0] as usize]) {
            total += u32::from(base.mu);
            if let Some(o) = opts.as_deref_mut() {
                o.push(Opt {
                    pick: Pick::Forme(base.formes[0]),
                    num: u32::from(base.mu),
                    den: 1,
                });
            }
        }
    }
    for &b in &t.multi {
        if av.base_used(b) {
            continue;
        }
        let base = &t.bases[b as usize];
        let mask = forme_mask(t, ctx, b);
        if mask == 0 {
            continue;
        }
        let alive = base.mu - k.dead_of(base.mi);
        live.push(Live {
            base: b,
            mi: base.mi,
            mask,
            alive,
        });
        if alive == 0 {
            continue;
        }
        total += u32::from(alive);
        if let Some(o) = opts.as_deref_mut() {
            for (i, &f) in base.formes.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    o.push(Opt {
                        pick: Pick::Forme(f),
                        num: u32::from(alive),
                        den: mask.count_ones(),
                    });
                }
            }
        }
    }
    total
}
fn binom(n: u8, k: u8) -> u64 {
    (0..u64::from(k)).fold(1, |acc, i| acc * (u64::from(n) - i) / (i + 1))
}
/// The children of `child` (the pick already added): each live base's
/// alive labels split by whether they survive the new team, `C(r,j) a^{r−j}
/// (b−a)^j / b^r` with a of its b admissible formes still admissible; a
/// base no longer in play forgotten. `f(key, probability)`.
fn split_children<N: Num>(
    t: &Tables,
    child: Key,
    picked_base: u16,
    live: &[Live],
    f: &mut impl FnMut(Key, N),
) {
    let ctx = Ctx::of_key(t, &child);
    let mut c = child;
    let mut splits = [(0u8, 0u8, 0u8, 0u64, 0u64); 64];
    let mut n = 0;
    for l in live {
        if l.base == picked_base {
            c.set_dead(l.mi, 0);
            continue;
        }
        let mask = forme_mask(t, &ctx, l.base);
        if mask == 0 {
            c.set_dead(l.mi, 0);
        } else if mask != l.mask && l.alive > 0 {
            splits[n] = (
                l.mi,
                c.dead_of(l.mi),
                l.alive,
                u64::from(mask.count_ones()),
                u64::from(l.mask.count_ones()),
            );
            n += 1;
        }
    }
    let splits = &splits[..n];
    let mut js = [0u8; 64];
    loop {
        let mut key = c;
        let (mut num, mut den) = (1u64, 1u64);
        for (i, &(mi, dead, r, a, b)) in splits.iter().enumerate() {
            let j = js[i];
            num *= binom(r, j) * a.pow(u32::from(r - j)) * (b - a).pow(u32::from(j));
            den *= b.pow(u32::from(r));
            key.set_dead(mi, dead + j);
        }
        f(key, N::ratio(num, den));
        let mut i = 0;
        loop {
            if i == n {
                return;
            }
            if js[i] < splits[i].2 {
                js[i] += 1;
                break;
            }
            js[i] = 0;
            i += 1;
        }
    }
}
/// The weights, keyed by (species, party slot, lead, full context code).
type Weights<N> = FxMap<(u16, u8, bool, u16), <N as Num>::Acc>;
#[inline]
fn credit<N: Num>(w: &mut Weights<N>, s: u16, slot: u8, lead: bool, code: u16, x: N) {
    *w.entry((s, slot, lead, code)).or_default() += x;
}
/// The party slot a pick lands in after `n` members: PS `unshift`s the lead
/// and appends every other member, so slot 1 is the lead and the rest keep
/// generation order.
fn slot_of(n: u8, lead_placed: bool, lead_pick: bool) -> u8 {
    if lead_pick { 1 } else if lead_placed { n + 1 } else { n + 2 }
}
/// One step out of a state: the pick's class and base (NONE for a simple
/// class), the lead and Tera Blast flags and Fire count after it, its
/// probability, and the contexts after it (not yet scaled by `p`).
struct Step<'a, N> {
    class: u16,
    base: u16,
    lead: bool,
    tb: bool,
    fire: u8,
    p: N,
    deco: &'a Deco<N>,
    /// The pick's weight `num/den` and the state's total M: `p` is their
    /// ratio when the state's live counts are fixed.
    num: u32,
    den: u32,
    total: u32,
}
impl<N: Num> Step<'_, N> {
    /// The team after the step, its labels not yet split.
    fn child(&self, k: &Key) -> Key {
        let mut c = k.with(self.class);
        c.lead = self.lead;
        c.tb = self.tb;
        c.fire = self.fire;
        c
    }
}
/// `step`'s children, each with its probability folded into its contexts.
fn emit_children<N: Num>(
    t: &Tables,
    k: &Key,
    step: &Step<N>,
    live: &[Live],
    emit: &mut impl FnMut(Key, Deco<N>),
) -> u64 {
    let mut n = 0;
    split_children(
        t,
        step.child(k),
        step.base,
        live,
        &mut |ck, q: N| {
            let pq = step.p * q;
            emit(ck, step.deco.iter().map(|&(c, x)| (c, x * pq)).collect());
            n += 1;
        },
    );
    n
}
/// A flagged pick's own set out of `k`: `deco` through its kernel, by the
/// Tera Blast flag and Fire count after it, appended to `out`.
fn kernel_split<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    k: &Key,
    picked: u16,
    lead_pick: bool,
    deco: &Deco<N>,
    out: &mut Vec<(bool, u8, Deco<N>)>,
) {
    let tb_code = if k.tb { TB_BIT } else { 0 };
    for &(code, x) in deco {
        for ko in ar.kernel(t, picked, lead_pick, code | tb_code).iter() {
            let tb = k.tb || ko.tb;
            let fire = k.fire + ko.fire;
            let slot = match out.iter().position(|e| e.0 == tb && e.1 == fire) {
                Some(i) => i,
                None => {
                    out.push((tb, fire, Vec::new()));
                    out.len() - 1
                }
            };
            deco_add(&mut out[slot].2, ko.code, x * ko.p);
        }
    }
}
/// Every step out of (`k`, `deco`), handed to `on_step` with `k`'s live
/// bases, and the accepted species' weights credited by party slot.
fn walk<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    k: &Key,
    deco: &Deco<N>,
    w: Option<&mut Weights<N>>,
    sc: &mut Scratch,
    on_step: &mut impl FnMut(Step<N>, &[Live]),
) {
    let ctx = Ctx::of_key(t, k);
    let av = Avail::of_key(t, k);
    let total = u64::from(options(t, k, &ctx, &av, sc));
    let Scratch { opts, live } = sc;
    let tb_code = if k.tb { TB_BIT } else { 0 };
    let mut w = w;
    for o in opts.iter() {
        let p = N::ratio(u64::from(o.num), u64::from(o.den) * total);
        let (class, picked, lead_pick, base) = match o.pick {
            Pick::Class(c) => (c, t.classes[c as usize].members[0], !k.lead, NONE),
            Pick::Forme(f) => {
                let s = &t.sp[f as usize];
                (s.class, f, !k.lead && !s.no_lead, s.base)
            }
        };
        let lead_after = k.lead || lead_pick;
        let s = &t.sp[picked as usize];
        let flagged = matches!(o.pick, Pick::Forme(_)) && s.flags;
        let mut variants: Vec<(bool, u8, Deco<N>)> = Vec::new();
        if flagged {
            kernel_split(t, ar, k, picked, lead_pick, deco, &mut variants);
        }
        let step = |tb: bool, fire: u8, deco| Step {
            class,
            base,
            lead: lead_after,
            tb,
            fire,
            p,
            deco,
            num: o.num,
            den: o.den,
            total: total as u32,
        };
        if flagged {
            for (tb, fire, d) in &variants {
                on_step(step(*tb, *fire, d), live);
            }
        } else {
            on_step(step(k.tb, k.fire, deco), live);
        }
        if let Some(w) = w.as_deref_mut() {
            let slot = slot_of(k.n, k.lead, lead_pick);
            match o.pick {
                Pick::Class(c) => {
                    let members = &t.classes[c as usize].members;
                    let share = N::ratio(
                        u64::from(o.num),
                        u64::from(o.den) * total * members.len() as u64,
                    );
                    for &m in members {
                        for &(code, x) in deco {
                            credit(w, m, slot, lead_pick, code | tb_code, x * share);
                        }
                    }
                }
                Pick::Forme(f) => {
                    for &(code, x) in deco {
                        credit(w, f, slot, lead_pick, code | tb_code, x * p);
                    }
                }
            }
        }
    }
}
/// For M = f + Σ_b A_b, the A_b ~ Bin(r_b, a_b/b_b) independent: E[1/M]
/// = ∫₀¹ x^{f−1} ∏_b (1 − q_b + q_b x)^{r_b} dx, returned, and in `out[b]`
/// E[A_b/M] = ∫₀¹ x^{f−1} · r_b q_b x (1 − q_b + q_b x)^{r_b−1} ∏_{b′≠b} … dx
/// (1/M = ∫₀¹ x^{M−1} dx, and E[x^A] = (1 − q + qx)^r). Every coefficient
/// is positive, so nothing cancels.
fn binomial_moments<N: Num>(f: u32, factors: &[(u8, u32, u32)], out: &mut Vec<N>) -> N {
    out.clear();
    if factors.is_empty() {
        return N::ratio(1, u64::from(f));
    }
    assert!(f >= 1, "a split that can kill every live entry would end the team");
    let zero = N::default();
    let poly = |skip: Option<usize>| {
        let mut p = vec![N::ratio(1, 1)];
        for (i, &(r, a, b)) in factors.iter().enumerate() {
            let (q, nq) = (
                N::ratio(u64::from(a), u64::from(b)),
                N::ratio(u64::from(b - a), u64::from(b)),
            );
            let times = if skip == Some(i) { r - 1 } else { r };
            for _ in 0..times {
                let mut next = vec![zero; p.len() + 1];
                for (j, &c) in p.iter().enumerate() {
                    next[j] = next[j] + c * nq;
                    next[j + 1] = next[j + 1] + c * q;
                }
                p = next;
            }
            if skip == Some(i) {
                p.insert(0, zero);
                let rq = N::ratio(u64::from(r) * u64::from(a), u64::from(b));
                for c in &mut p {
                    *c = *c * rq;
                }
            }
        }
        p.iter()
            .enumerate()
            .fold(zero, |acc, (j, &c)| acc + c * N::ratio(1, u64::from(f) + j as u64))
    };
    let inv = poly(None);
    out.extend((0..factors.len()).map(|b| poly(Some(b))));
    let one = out.iter().fold(N::ratio(u64::from(f), 1) * inv, |acc, &m| acc + m);
    assert!(N::same(one, N::ratio(1, 1)), "binomial moments do not sum to 1");
    inv
}
/// The next pick from `k`, folded: a species with weight ω takes ω·E[1/M]
/// of `deco`, credited through `table[signature]` to every species the
/// signature admits, and the exceptions are credited directly — the species
/// clause, and multi-forme bases whose live entries are not μ shared over
/// all φ formes. `sc` holds `options` on `k` (its `total`); `split` lists
/// the live bases whose labels this step split, with a of their b
/// admissible formes still admissible — their live counts are Bin(alive,
/// a/b) and enter through `binomial_moments` instead of being enumerated.
fn fold_next<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    k: &Key,
    ctx: &Ctx,
    av: &Avail,
    total: u32,
    sc: &Scratch,
    split: &[(u16, u32, u32)],
    deco: &Deco<N>,
    table: &Sharded<Ctx, KDeco<N>>,
    w: &mut Weights<N>,
) {
    if total == 0 {
        return;
    }
    let mut fixed = total;
    let mut factors: Vec<(u8, u32, u32)> = Vec::new();
    let mut factor_of = [usize::MAX; 64];
    for (i, l) in sc.live.iter().enumerate() {
        if let Some(&(_, a, b)) = split.iter().find(|s| s.0 == l.base) && l.alive > 0 {
            fixed -= u32::from(l.alive);
            factor_of[i] = factors.len();
            factors.push((l.alive, a, b));
        }
    }
    let mut moments = Vec::new();
    let inv: N = binomial_moments(fixed, &factors, &mut moments);
    let random: Vec<(usize, N)> = (0..sc.live.len())
        .filter(|&i| factor_of[i] != usize::MAX)
        .map(|i| (i, moments[factor_of[i]]))
        .collect();
    fold_credit(t, ar, k, ctx, av, sc, inv, &random, deco, table, w);
}
/// The credits of the next pick from `k`, given what the path carries
/// times E[1/M] (`inv`) and, for each live base whose live count is random,
/// times E[A/M] (`random`: its index in `sc.live`, the value) — the rest of
/// [`fold_next`]'s contract.
fn fold_credit<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    k: &Key,
    ctx: &Ctx,
    av: &Avail,
    sc: &Scratch,
    inv: N,
    random: &[(usize, N)],
    deco: &Deco<N>,
    table: &Sharded<Ctx, KDeco<N>>,
    w: &mut Weights<N>,
) {
    let tb_code = if k.tb { TB_BIT } else { 0 };
    let u: Deco<N> = deco.iter().map(|&(c, x)| (c, x * inv)).collect();
    credit_signature(t, ar, k, ctx, av, &u, table, w);
    for (i, l) in sc.live.iter().enumerate() {
        let base = &t.bases[l.base as usize];
        let full = (1u32 << base.formes.len()) - 1;
        let moment = random.iter().find(|r| r.0 == i).map(|r| r.1);
        if moment.is_none() && l.mask == full && l.alive == base.mu {
            continue;
        }
        let per_forme = match moment {
            Some(m) => m * N::ratio(1, u64::from(l.mask.count_ones())),
            None => N::ratio(u64::from(l.alive), u64::from(l.mask.count_ones())) * inv,
        };
        for (j, &f) in base.formes.iter().enumerate() {
            if l.mask & (1 << j) != 0 {
                let (sl, lead) = pick_slot(k, &t.sp[f as usize]);
                let coefficient = per_forme - ar.omega[f as usize] * inv;
                for &(code, x) in deco {
                    credit(w, f, sl, lead, code | tb_code, coefficient * x);
                }
            }
        }
    }
}
/// The party slot of picking `s` next from `k`, and whether it leads.
fn pick_slot(k: &Key, s: &Sp) -> (u8, bool) {
    let lead_pick = !k.lead && !s.no_lead;
    (slot_of(k.n, k.lead, lead_pick), lead_pick)
}
/// The next pick from `k` through its signature table — `u` is the
/// contexts times E[w/M] — and the species clause's correction to it: a
/// member's base (and a simple class's used share) cannot come again.
fn credit_signature<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    k: &Key,
    ctx: &Ctx,
    av: &Avail,
    u: &Deco<N>,
    table: &Sharded<Ctx, KDeco<N>>,
    w: &mut Weights<N>,
) {
    let zero = N::default();
    let tb_code = if k.tb { TB_BIT } else { 0 };
    {
        let mut shard = table.shard(fxhash(ctx)).lock().unwrap();
        let entry = shard.entry(*ctx).or_default();
        for &(c, x) in u {
            kdeco_add::<N>(entry, c, x);
        }
    }
    let mut seen_classes: [u16; 5] = [NONE; 5];
    for (i, &c) in k.members().iter().enumerate() {
        let class = &t.classes[c as usize];
        if class.simple {
            if seen_classes.contains(&c) {
                continue;
            }
            seen_classes[i] = c;
            if !ctx.adm(&t.sp[class.members[0] as usize]) {
                continue;
            }
            let used = N::ratio(u64::from(av.class_used(c)), class.members.len() as u64);
            for &m in &class.members {
                let (sl, lead) = pick_slot(k, &t.sp[m as usize]);
                let om = ar.omega[m as usize] * used;
                for &(code, x) in u {
                    credit(w, m, sl, lead, code | tb_code, zero - om * x);
                }
            }
        } else {
            let base = t.sp[class.members[0] as usize].base;
            for &f in &t.bases[base as usize].formes {
                let s = &t.sp[f as usize];
                if ctx.adm(s) {
                    let (sl, lead) = pick_slot(k, s);
                    for &(code, x) in u {
                        credit(
                            w,
                            f,
                            sl,
                            lead,
                            code | tb_code,
                            zero - ar.omega[f as usize] * x,
                        );
                    }
                }
            }
        }
    }
}
/// [`fold_next`] for the pick after `step`, with `step`'s label splits
/// integrated: the team after it, and each live base whose admissible
/// formes `step` shrank, with how many of how many survived.
fn fold_after<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    k: &Key,
    step: &Step<N>,
    live: &[Live],
    sc: &mut Scratch,
    table: &Sharded<Ctx, KDeco<N>>,
    w: &mut Weights<N>,
) {
    let child = step.child(k);
    let ctx = Ctx::of_key(t, &child);
    let av = Avail::of_key(t, &child);
    let total = options(t, &child, &ctx, &av, sc);
    let mut split: Vec<(u16, u32, u32)> = Vec::new();
    for l in &sc.live {
        let parent = live
            .iter()
            .find(|p| p.base == l.base)
            .expect("admissibility only shrinks");
        if parent.mask != l.mask {
            split.push((l.base, l.mask.count_ones(), parent.mask.count_ones()));
        }
    }
    let deco: Deco<N> = step.deco.iter().map(|&(c, x)| (c, x * step.p)).collect();
    fold_next(t, ar, &child, &ctx, &av, total, sc, &split, &deco, table, w);
}
/// A polynomial of at most three terms (i, j, coefficient): one entry's
/// generating function in x and y, or its derivative.
#[derive(Clone, Copy)]
struct Terms<N> {
    n: u8,
    t: [(u8, u8, N); 3],
}
impl<N: Num> Terms<N> {
    fn new(t: &[(u8, u8, N)]) -> Self {
        let mut out = Terms {
            n: t.len() as u8,
            t: [(0, 0, N::default()); 3],
        };
        out.t[..t.len()].copy_from_slice(t);
        out
    }
    fn terms(&self) -> &[(u8, u8, N)] {
        &self.t[..self.n as usize]
    }
}
/// The grids of the slot-5 polynomial passes, reused pick after pick.
#[derive(Default)]
struct PolyBuf<N> {
    data: Vec<N>,
}
static SHAPE: (AtomicU64, AtomicU64, AtomicU64, AtomicU64, AtomicU64) = (
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
);
/// Slot 5 from one size-3 situation — `step3` out of `k2`, its label splits
/// integrated: every fourth pick, and the fifth folded through `table5`
/// (`fold5_pick`).
fn fold5_after<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    k2: &Key,
    step3: &Step<N>,
    live2: &[Live],
    sc3: &mut Scratch,
    sc4: &mut Scratch,
    polys: &mut PolyBuf<N>,
    table5: &Sharded<Ctx, KDeco<N>>,
    w: &mut Weights<N>,
) {
    let child3 = step3.child(k2);
    let d3: Deco<N> = step3.deco.iter().map(|&(c, x)| (c, x * step3.p)).collect();
    let mut mask2 = [0u32; 64];
    for l in live2 {
        mask2[usize::from(l.mi)] = l.mask;
    }
    walk(
        t,
        ar,
        &child3,
        &d3,
        None,
        sc3,
        &mut |step4, live3| {
            fold5_pick(t, ar, &child3, &step4, &mask2, live3, sc4, polys, table5, w);
        },
    );
}
/// The fifth pick after `step4`, with the label splits of steps 3 and 4
/// integrated jointly. With 1/M₃ = ∫x^{M₃−1}dx and 1/M₄ = ∫y^{M₄−1}dy, each
/// size-3 live base contributes a factor E[x^{A₃} y^{A₄}] per entry:
///
/// - split at step 3, unchanged at 4: (1 − q₃) + q₃xy
/// - split at both: (1 − q₃) + q₃x((1 − q₄) + q₄y)
/// - split at step 3, then out of play (killed, or `step4`'s own base):
///   (1 − q₃) + q₃x
/// - fixed at step 3, split at 4: x · ((1 − q₄) + q₄y), the x into f₃
///
/// and a fixed base's entries go into f₃ (and f₄ while still live). The
/// fourth pick's weight w₄ is constant, or A₃/|F₃| for a forme of a base
/// split at step 3 — x∂/∂x on one of its entries. Then V = E[w₄/(M₃M₄)] =
/// ∫∫x^{f₃−1}y^{f₄−1}∏ is what every species of ω takes ω·V of, and a base
/// whose A₄ is random takes E[w₄A₄/(M₃M₄)] (y∂/∂y on one of its entries)
/// over its formes. Every pick checks f₄·V + Σ E[w₄A₄/(M₃M₄)] = E[w₄/M₃].
fn fold5_pick<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    child3: &Key,
    step4: &Step<N>,
    mask2: &[u32; 64],
    live3: &[Live],
    sc4: &mut Scratch,
    polys: &mut PolyBuf<N>,
    table5: &Sharded<Ctx, KDeco<N>>,
    w: &mut Weights<N>,
) {
    let child4 = step4.child(child3);
    if !child4.lead {
        return;
    }
    let ctx4 = Ctx::of_key(t, &child4);
    let av4 = Avail::of_key(t, &child4);
    let total4 = options_into(t, &child4, &ctx4, &av4, &mut sc4.live, None);
    if total4 == 0 {
        return;
    }
    let f3 = step4.total - live3.iter().map(|l| u32::from(l.alive)).sum::<u32>();
    let f4 = total4 - sc4.live.iter().map(|l| u32::from(l.alive)).sum::<u32>();
    let mut moments = Vec::new();
    let i0 = integrate5(
        mask2,
        live3,
        step4.base,
        &sc4.live,
        f3,
        f4,
        polys,
        &mut moments,
    );
    let c4 = N::ratio(u64::from(step4.num), u64::from(step4.den));
    for m in &mut moments {
        m.1 = c4 * m.1;
    }
    fold_credit(
        t,
        ar,
        &child4,
        &ctx4,
        &av4,
        sc4,
        c4 * i0,
        &moments,
        step4.deco,
        table5,
        w,
    );
}
/// The integrals behind one fifth pick, from a size-3 situation and its
/// fourth pick (of base `base4`, NONE for a simple class): `live3` are the
/// size-3 live bases with their alive counts before step 3's splits, `mask2`
/// their masks before step 3 by multi index, `live4` the size-4 live bases
/// (only their masks are read), and `f3`, `f4` the fixed parts of the two
/// pools, without any live base. Returns E[w₄/(M₃M₄)] over the fourth pick's
/// `num/den` (which carries w₄'s constant part), and pushes each random base's
/// E[w₄A₄/(M₃M₄)] over the same onto `moments`, by its index in `live4`. See
/// [`fold5_pick`] for the generating functions.
fn integrate5<N: Num>(
    mask2: &[u32; 64],
    live3: &[Live],
    base4: u16,
    live4: &[Live],
    mut f3: u32,
    mut f4: u32,
    polys: &mut PolyBuf<N>,
    moments: &mut Vec<(usize, N)>,
) -> N {
    let mut at4_of = [u8::MAX; 64];
    for (i, l) in live4.iter().enumerate() {
        at4_of[usize::from(l.mi)] = i as u8;
    }
    let mut qx = [N::default(); 65];
    qx[0] = N::ratio(1, 1);
    let mut dq = 0usize;
    let mut times_x = |a0: N, a1: N| {
        qx[dq + 1] = qx[dq] * a1;
        for i in (1..=dq).rev() {
            qx[i] = qx[i] * a0 + qx[i - 1] * a1;
        }
        qx[0] = qx[0] * a0;
        dq += 1;
    };
    let mut random = [(0usize, 0u8, Terms::new(&[]), Terms::new(&[])); 64];
    let (mut nr, mut degree) = (0, 0usize);
    let split_of = |l3: &Live| {
        let m2 = mask2[usize::from(l3.mi)];
        let (a, b) = (u64::from(l3.mask.count_ones()), u64::from(m2.count_ones()));
        (l3.mask != m2).then(|| (N::ratio(a, b), N::ratio(b - a, b)))
    };
    for l3 in live3 {
        let alive = l3.alive;
        if alive == 0 {
            continue;
        }
        let split3 = split_of(l3);
        if l3.base == base4 {
            if let Some((q3, nq3)) = split3 {
                for _ in 1..alive {
                    times_x(nq3, q3);
                }
                times_x(N::default(), q3);
            } else {
                f3 += u32::from(alive);
            }
            continue;
        }
        let at4 = at4_of[usize::from(l3.mi)];
        let m4 = if at4 == u8::MAX { 0 } else { live4[usize::from(at4)].mask };
        if m4 == 0 {
            if let Some((q3, nq3)) = split3 {
                for _ in 0..alive {
                    times_x(nq3, q3);
                }
            } else {
                f3 += u32::from(alive);
            }
        } else if m4 == l3.mask {
            if let Some((q3, nq3)) = split3 {
                random[nr] = (
                    usize::from(at4),
                    alive,
                    Terms::new(&[(0, 0, nq3), (1, 1, q3)]),
                    Terms::new(&[(1, 1, q3)]),
                );
                nr += 1;
                degree += usize::from(alive);
            } else {
                f3 += u32::from(alive);
                f4 += u32::from(alive);
            }
        } else {
            let (a, b) = (u64::from(m4.count_ones()), u64::from(l3.mask.count_ones()));
            let (q4, nq4) = (N::ratio(a, b), N::ratio(b - a, b));
            random[nr] = match split3 {
                Some((q3, nq3)) => {
                    (
                        usize::from(at4),
                        alive,
                        Terms::new(&[(0, 0, nq3), (1, 0, q3 * nq4), (1, 1, q3 * q4)]),
                        Terms::new(&[(1, 1, q3 * q4)]),
                    )
                }
                None => {
                    f3 += u32::from(alive);
                    (
                        usize::from(at4),
                        alive,
                        Terms::new(&[(0, 0, nq4), (0, 1, q4)]),
                        Terms::new(&[(0, 1, q4)]),
                    )
                }
            };
            nr += 1;
            degree += usize::from(alive);
        }
    }
    let _ = &mut times_x;
    SHAPE.0.fetch_add(1, Ordering::Relaxed);
    SHAPE.1.fetch_add(nr as u64, Ordering::Relaxed);
    SHAPE.2.fetch_add(degree as u64, Ordering::Relaxed);
    SHAPE.3.fetch_add(dq as u64, Ordering::Relaxed);
    SHAPE.4.fetch_max(degree as u64, Ordering::Relaxed);
    assert!(f3 >= 1 && f4 >= 1, "a pool that can run dry before the fifth pick");
    assert!(degree < 64 && dq < 64, "{degree} random and {dq} fixed entries");
    let (mut h, mut iy) = ([N::default(); 65], [N::default(); 65]);
    for k in 0..=degree {
        h[k] = (0..=dq)
            .fold(
                N::default(),
                |acc, l| acc + qx[l] * N::ratio(1, u64::from(f3) + (k + l) as u64),
            );
        iy[k] = N::ratio(1, u64::from(f4) + k as u64);
    }
    let degree_of = |f: &Terms<N>, axis: usize| {
        f
            .terms()
            .iter()
            .map(|t| usize::from(if axis == 0 { t.0 } else { t.1 }))
            .max()
            .unwrap_or(0)
    };
    let (gx, gy) = random[..nr]
        .iter()
        .fold(
            (0, 0),
            |(x, y), (_, r, f, _)| (
                x + usize::from(*r) * degree_of(f, 0),
                y + usize::from(*r) * degree_of(f, 1),
            ),
        );
    let (row, size) = (gy + 1, (gx + 1) * (gy + 1));
    let zero = N::default();
    polys.data.clear();
    polys.data.resize(size * (nr + 2), zero);
    let (snaps, work) = polys.data.split_at_mut(size * nr);
    let (mut cur, mut next) = work.split_at_mut(size);
    let mul = |src: &[N], dst: &mut [N], f: &Terms<N>| {
        dst.fill(zero);
        for i in 0..=gx {
            for j in 0..=gy {
                let x = src[i * row + j];
                for &(di, dj, b) in f.terms() {
                    let (a, c) = (i + usize::from(di), j + usize::from(dj));
                    if a <= gx && c <= gy {
                        dst[a * row + c] = dst[a * row + c] + x * b;
                    }
                }
            }
        }
    };
    let adjoint = |src: &[N], dst: &mut [N], f: &Terms<N>| {
        for i in 0..=gx {
            for j in 0..=gy {
                let mut s = zero;
                for &(di, dj, b) in f.terms() {
                    let (a, c) = (i + usize::from(di), j + usize::from(dj));
                    if a <= gx && c <= gy {
                        s = s + src[a * row + c] * b;
                    }
                }
                dst[i * row + j] = s;
            }
        }
    };
    cur[0] = N::ratio(1, 1);
    for (k, &(_, r, f, _)) in random[..nr].iter().enumerate() {
        snaps[k * size..(k + 1) * size].copy_from_slice(cur);
        for _ in 0..r {
            mul(cur, next, &f);
            std::mem::swap(&mut cur, &mut next);
        }
    }
    let (mut i0, mut ix0) = (zero, zero);
    for i in 0..=gx {
        for j in 0..=gy {
            i0 = i0 + cur[i * row + j] * h[i] * iy[j];
            ix0 = ix0 + cur[i * row + j] * h[i];
        }
    }
    for i in 0..=gx {
        for j in 0..=gy {
            cur[i * row + j] = h[i] * iy[j];
        }
    }
    let first = moments.len();
    moments.resize(first + nr, (0, zero));
    let mut check = N::ratio(u64::from(f4), 1) * i0;
    for k in (0..nr).rev() {
        let (at4, r, f, d) = random[k];
        for _ in 1..r {
            adjoint(cur, next, &f);
            std::mem::swap(&mut cur, &mut next);
        }
        let snap = &snaps[k * size..(k + 1) * size];
        let mut ib = zero;
        for i in 0..=gx {
            for j in 0..=gy {
                for &(di, dj, b) in d.terms() {
                    let (a, c) = (i + usize::from(di), j + usize::from(dj));
                    if a <= gx && c <= gy {
                        ib = ib + snap[i * row + j] * b * cur[a * row + c];
                    }
                }
            }
        }
        let ib = ib * N::ratio(u64::from(r), 1);
        check = check + ib;
        moments[first + k] = (at4, ib);
        adjoint(cur, next, &f);
        std::mem::swap(&mut cur, &mut next);
    }
    assert!(N::same(check, ix0), "slot-5 moments do not sum to E[w4/M3]");
    i0
}
/// Gauss–Jacobi nodes and weights on [0, 1] for the weight x^{f−1}, exact
/// for polynomials of degree below 2n: Golub–Welsch, the Jacobi matrix's
/// eigenvalues by implicit QL, the weights 1/f times the squared first
/// components of its eigenvectors.
fn gauss_jacobi(f: u32, n: usize) -> (Vec<f64>, Vec<f64>) {
    let beta = f64::from(f) - 1.0;
    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n];
    for k in 0..n {
        let (kf, s) = (k as f64, 2.0 * k as f64 + beta);
        d[k] = if k == 0 {
            1.0 / (beta + 2.0)
        } else {
            (2.0 * kf * (kf + beta + 1.0) + beta) / (s * (s + 2.0))
        };
        if k + 1 < n {
            let k1 = k as f64 + 1.0;
            let s1 = 2.0 * k1 + beta;
            e[k] = (4.0 * k1 * k1 * (k1 + beta) * (k1 + beta)
                / (s1 * s1 * (s1 + 1.0) * (s1 - 1.0)))
                .sqrt() / 2.0;
        }
    }
    let (a, b) = (d.clone(), e.clone());
    let mut z = vec![0.0; n];
    z[0] = 1.0;
    for l in 0..n {
        let mut iter = 0;
        loop {
            let mut m = l;
            while m + 1 < n {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() + dd == dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iter += 1;
            assert!(iter < 60, "QL did not converge for f {f}, n {n}");
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = g.hypot(1.0);
            g = d[m] - d[l] + e[l] / (g + r.copysign(g));
            let (mut s, mut c, mut p) = (1.0, 1.0, 0.0);
            let mut i = m;
            let mut deflated = false;
            while i > l {
                i -= 1;
                let mut ff = s * e[i];
                let b = c * e[i];
                r = ff.hypot(g);
                e[i + 1] = r;
                if r == 0.0 {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    deflated = true;
                    break;
                }
                s = ff / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
                ff = z[i + 1];
                z[i + 1] = s * z[i] + c * ff;
                z[i] = c * z[i] - s * ff;
            }
            if deflated {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
    let _ = z;
    for u in &mut d {
        for _ in 0..3 {
            let (mut p0, mut p1, mut d0, mut d1) = (0.0, 1.0, 0.0, 0.0);
            for j in 0..n {
                let bb = if j == 0 { 0.0 } else { b[j - 1] * b[j - 1] };
                let p2 = (*u - a[j]) * p1 - bb * p0;
                let d2 = p1 + (*u - a[j]) * d1 - bb * d0;
                (p0, p1, d0, d1) = (p1, p2, d1, d2);
            }
            *u -= p1 / d1;
        }
    }
    let w = d
        .iter()
        .map(|&u| {
            let (mut p0, mut p1) = (0.0, f64::from(f).sqrt());
            let mut s = p1 * p1;
            for j in 0..n - 1 {
                let bb = if j == 0 { 0.0 } else { b[j - 1] };
                let p2 = ((u - a[j]) * p1 - bb * p0) / b[j];
                (p0, p1) = (p1, p2);
                s += p1 * p1;
            }
            1.0 / s
        })
        .collect();
    (d.iter().map(|u| 1.0 - u).collect(), w)
}
/// The quadrature rules [`quad5`] asks for, kept per thread.
#[derive(Default)]
struct Quad {
    rules: FxMap<(u32, usize), (Vec<f64>, Vec<f64>)>,
    common: Vec<f64>,
    tables: Vec<f64>,
    ratios: Vec<f64>,
    val: Vec<f64>,
}
impl Quad {
    fn rule(&mut self, f: u32, n: usize) -> &(Vec<f64>, Vec<f64>) {
        self.rules.entry((f, n)).or_insert_with(|| gauss_jacobi(f, n))
    }
}
/// [`integrate5`] for every class of a (size-3 situation, fourth pick) at
/// once, by Gauss–Jacobi quadrature in x and y, exact for the polynomial
/// integrand. Each entry of the size-3 live base i carries the factor
/// E_i = (1 − q₃) + q₃(1 − q₄)x + q₃q₄xy (q₃, q₄ the shares of its formes
/// steps 3 and 4 leave admissible), and the fourth pick's base one entry's
/// x∂/∂x instead, q₃x. A base whose alive count r is the same in every class
/// folds in once — a monomial x^r or (xy)^r into the pools' fixed parts,
/// anything else into the common node weights — and the rest are tables
/// E_i^r at the nodes, r ≤ 3, so a class costs one product of table rows and
/// a weighted sum per random base, at y∂/∂y E_i / E_i. `rmin`/`rmax` bound
/// each live3 base's alive count over `keys`; out: `i0s[c]` and
/// `moms[c · live4.len() + j]` as [`integrate5`] returns them.
fn quad5(
    q: &mut Quad,
    plan: &QuadPlan,
    keys: &[u128],
    nl4: usize,
    i0s: &mut [f64],
    moms: &mut [f64],
) {
    let r_of = |key: u128, i: usize| ((key >> (2 * i)) & 3) as usize;
    let QuadPlan { common, variant, random, base4_at, xs, wx, ys, wy, .. } = plan;
    let base4_at = *base4_at;
    let (nx, ny) = (xs.len(), ys.len());
    let pp = (nx * ny).div_ceil(8) * 8;
    let factor = |a: f64, b: f64, c: f64, p: usize| {
        let (x, y) = (xs[p / ny], ys[p % ny]);
        a + b * x + c * x * y
    };
    let first = |s: f64, p: usize| s * xs[p / ny];
    q.common.clear();
    q.common.resize(pp, 0.0);
    for p in 0..nx * ny {
        let mut v = wx[p / ny] * wy[p % ny];
        for &(i, a, b, c, s, r) in common {
            let e = factor(a, b, c, p);
            if Some(i) == base4_at {
                v *= first(s, p) * e.powi(i32::from(r) - 1);
            } else {
                v *= e.powi(i32::from(r));
            }
        }
        q.common[p] = v;
    }
    q.tables.clear();
    q.tables.resize(variant.len() * 4 * pp, 0.0);
    for (k, &(i, a, b, c, s, rmax)) in variant.iter().enumerate() {
        for p in 0..nx * ny {
            let e = factor(a, b, c, p);
            let mut pow = 1.0;
            for r in 0..=usize::from(rmax) {
                q.tables[(k * 4 + r) * pp + p] = if Some(i) != base4_at {
                    pow
                } else if r == 0 {
                    0.0
                } else {
                    first(s, p) * e.powi(r as i32 - 1)
                };
                pow *= e;
            }
        }
    }
    q.ratios.clear();
    q.ratios.resize(random.len() * pp, 0.0);
    for (k, &(_, _, a, b, c)) in random.iter().enumerate() {
        for p in 0..nx * ny {
            let (x, y) = (xs[p / ny], ys[p % ny]);
            q.ratios[k * pp + p] = c * x * y / (a + b * x + c * x * y);
        }
    }
    q.val.resize(pp, 0.0);
    for (ci, &key) in keys.iter().enumerate() {
        if base4_at.is_some_and(|i| r_of(key, i) == 0) {
            i0s[ci] = 0.0;
            continue;
        }
        let val = &mut q.val[..pp];
        val.copy_from_slice(&q.common);
        for (k, &(i, ..)) in variant.iter().enumerate() {
            let t = &q.tables[(k * 4 + r_of(key, i)) * pp..][..pp];
            for (v, t) in val.iter_mut().zip(t) {
                *v *= t;
            }
        }
        i0s[ci] = lanes_sum(val);
        for (k, &(i, j, ..)) in random.iter().enumerate() {
            moms[ci * nl4 + j] = r_of(key, i) as f64
                * lanes_dot(val, &q.ratios[k * pp..][..pp]);
        }
    }
    QSHAPE.0.fetch_add(1, Ordering::Relaxed);
    QSHAPE.1.fetch_add(keys.len() as u64, Ordering::Relaxed);
    QSHAPE.2.fetch_add((pp * keys.len()) as u64, Ordering::Relaxed);
    QSHAPE.3.fetch_add((variant.len() * keys.len()) as u64, Ordering::Relaxed);
    QSHAPE.4.fetch_add((random.len() * keys.len()) as u64, Ordering::Relaxed);
}
/// What [`quad5`] makes of one pick before any class: which bases fold into
/// the pools, which multiply every class alike (`common`, with their one
/// alive count) and which vary (`variant`, with the most any class has) —
/// each as (live3 index, a, b, c, the first entry's x coefficient: NaN off
/// the fourth pick's base, alive count) — the random bases as (live3 index,
/// live4 index, a, b, c), the weights' exponents past the folded monomials,
/// and the nodes.
struct QuadPlan {
    common: Vec<(usize, f64, f64, f64, f64, u8)>,
    variant: Vec<(usize, f64, f64, f64, f64, u8)>,
    random: Vec<(usize, usize, f64, f64, f64)>,
    base4_at: Option<usize>,
    f3: u32,
    f4: u32,
    xs: Vec<f64>,
    wx: Vec<f64>,
    ys: Vec<f64>,
    wy: Vec<f64>,
}
#[inline(never)]
fn quad_plan(
    q: &mut Quad,
    mask2: &[u32; 64],
    live3: &[Live],
    rmin: &[u8],
    rmax: &[u8],
    base4: u16,
    live4: &[Live],
    mut f3: u32,
    mut f4: u32,
) -> QuadPlan {
    let mut at4_of = [u8::MAX; 64];
    for (j, l) in live4.iter().enumerate() {
        at4_of[usize::from(l.mi)] = j as u8;
    }
    let mut common = Vec::new();
    let mut variant = Vec::new();
    let mut random = Vec::new();
    let (mut dx, mut dy) = (0usize, 0usize);
    let base4_at = live3.iter().position(|l| l.base == base4);
    for (i, l3) in live3.iter().enumerate() {
        if rmax[i] == 0 {
            continue;
        }
        let (n3, d3) = (
            u64::from(l3.mask.count_ones()),
            u64::from(mask2[usize::from(l3.mi)].count_ones()),
        );
        let at4 = at4_of[usize::from(l3.mi)];
        let n4 = if at4 == u8::MAX {
            0
        } else {
            u64::from(live4[usize::from(at4)].mask.count_ones())
        };
        let (a, b, c) = (
            f64::ratio(d3 - n3, d3),
            f64::ratio(n3 - n4, d3),
            f64::ratio(n4, d3),
        );
        if n4 > 0 && (a != 0.0 || b != 0.0) {
            random.push((i, usize::from(at4), a, b, c));
        }
        let monomial = a == 0.0 && (b == 0.0 || c == 0.0);
        if rmin[i] == rmax[i] && monomial {
            f3 += u32::from(rmax[i]);
            if c != 0.0 {
                f4 += u32::from(rmax[i]);
            }
            continue;
        }
        dx += usize::from(rmax[i]);
        if c != 0.0 {
            dy += usize::from(rmax[i]);
        }
        let s = if Some(i) == base4_at { b } else { f64::NAN };
        if rmin[i] == rmax[i] {
            common.push((i, a, b, c, s, rmax[i]));
        } else {
            variant.push((i, a, b, c, s, rmax[i]));
        }
    }
    let (nx, ny) = (dx / 2 + 1, dy / 2 + 1);
    let (xs, wx) = q.rule(f3, nx).clone();
    let (ys, wy) = q.rule(f4, ny).clone();
    QuadPlan {
        common,
        variant,
        random,
        base4_at,
        f3,
        f4,
        xs,
        wx,
        ys,
        wy,
    }
}
static QSHAPE: (AtomicU64, AtomicU64, AtomicU64, AtomicU64, AtomicU64) = (
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
);
/// Σ a over a multiple of 8, in 8 independent lanes: positive terms, so the
/// reassociation costs nothing and lets the adds vectorize.
#[inline]
fn lanes_sum(a: &[f64]) -> f64 {
    let mut acc = [0.0f64; 8];
    for c in a.chunks_exact(8) {
        for l in 0..8 {
            acc[l] += c[l];
        }
    }
    acc.iter().sum()
}
/// Σ a·b, as [`lanes_sum`].
#[inline]
fn lanes_dot(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = [0.0f64; 8];
    for (ca, cb) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        for l in 0..8 {
            acc[l] += ca[l] * cb[l];
        }
    }
    acc.iter().sum()
}
/// [`gauss_jacobi`] against the moments it must reproduce, ∫x^{f−1}x^j =
/// 1/(f + j) for j < 2n, over the pools and degrees slot 5 meets.
fn check_quadrature() {
    for (fs, ns) in [
        (&[1u32, 2, 3, 7][..], 1..=40),
        (&[40, 150, 300, 450, 600, 900][..], 1..=40),
        (&[40, 150, 300, 450, 600, 900][..], 1..=16),
    ] {
        let mut worst: (f64, u32, usize, usize) = (0.0, 0, 0, 0);
        for &f in fs {
            for n in ns.clone() {
                let (x, w) = gauss_jacobi(f, n);
                for j in 0..2 * n {
                    let exact = 1.0 / f64::from(f + j as u32);
                    let mut got = Sum::default();
                    for (x, w) in x.iter().zip(&w) {
                        got += w * (0..j).fold(1.0, |p, _| p * x);
                    }
                    let rel = ((f64::get(got) - exact) / exact).abs();
                    if rel > worst.0 {
                        worst = (rel, f, n, j);
                    }
                }
            }
        }
        println!(
            "quadrature over f in {fs:?}, n up to {}: worst relative moment error {:.1e} (f {}, n {}, degree {})",
            ns.end(), worst.0, worst.1, worst.2, worst.3
        );
    }
}
/// `level`'s label-free groups: runs of states that share members, lead,
/// Tera Blast flag and Fire count (`level` sorted by them), differing only
/// in their dead labels.
fn label_groups<N>(level: &[(Key, Deco<N>)]) -> Vec<(u32, u32)> {
    let same = |a: &Key, b: &Key| {
        a.m == b.m && a.lead == b.lead && a.tb == b.tb && a.fire == b.fire
    };
    let mut groups = Vec::new();
    let mut lo = 0;
    for i in 1..=level.len() {
        if i == level.len() || !same(&level[i].0, &level[lo].0) {
            groups.push((lo as u32, i as u32));
            lo = i;
        }
    }
    groups
}
/// One pick's class, the species that stands for it, whether it leads,
/// and its base (NONE for a simple class).
fn pick_parts(t: &Tables, k: &Key, pick: Pick) -> (u16, u16, bool, u16) {
    match pick {
        Pick::Class(c) => (c, t.classes[c as usize].members[0], !k.lead, NONE),
        Pick::Forme(f) => {
            let s = &t.sp[f as usize];
            (s.class, f, !k.lead && !s.no_lead, s.base)
        }
    }
}
fn to_deco<N: Num>(d: &KDeco<N>) -> Deco<N> {
    d.iter().map(|&(c, x)| (c, N::get(x))).collect()
}
/// `deco` past a pick's set: through its kernel when it is `flagged`, the
/// part landing on (`tb`, `fire`).
#[inline(never)]
fn kernel_part<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    k: &Key,
    picked: u16,
    lead_pick: bool,
    flagged: bool,
    tb: bool,
    fire: u8,
    deco: Deco<N>,
) -> Deco<N> {
    if !flagged {
        return deco;
    }
    let mut out = Vec::new();
    kernel_split(t, ar, k, picked, lead_pick, &deco, &mut out);
    out.into_iter().find(|o| o.0 == tb && o.1 == fire).map_or_else(Vec::new, |o| o.2)
}
/// Slot 5, group-major. A label group's states differ only in their dead
/// labels, so one (group, third pick) item takes each fourth pick's options,
/// kernel, signature deposit and corrections once, on contexts summed over
/// the group's variants, each weighted by its own integrals. Variants whose
/// alive counts agree on every size-3 live base have the same integrals, so
/// they are summed first, into classes.
fn slot5_item<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    group: &[(Key, Deco<N>)],
    o3: usize,
    l: &mut Local<N>,
    table5: &Sharded<Ctx, KDeco<N>>,
    grid: bool,
    profile: bool,
) {
    let Local { fold, sc, sc3, sc4, polys, quad, .. } = l;
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
        let b = &t.bases[base as usize];
        b.mu - k.dead_of(b.mi)
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
        let p = N::ratio(u64::from(num), u64::from(o.den) * u64::from(m3));
        split.clear();
        if flagged3 {
            kernel_split(t, ar, &rep, picked3, lead_pick3, deco, &mut split);
        } else {
            split.push((rep.tb, rep.fire, deco.clone()));
        }
        for (tb, fire, d) in split.drain(..) {
            let d: Deco<N> = d.into_iter().map(|(c, x)| (c, x * p)).collect();
            match outcomes3.iter_mut().find(|e| e.0 == tb && e.1 == fire) {
                Some(e) => e.2.push((v, d)),
                None => outcomes3.push((tb, fire, vec![(v, d)])),
            }
        }
    }
    let mut sx = ClassScratch::<N>::default();
    #[cfg(feature = "gpu")]
    let gpu = GPU.load(Ordering::Relaxed) && N::QUADRATURE && !grid;
    #[cfg(feature = "gpu")]
    let (mut held, mut deferred): (Vec<Situation<N>>, Vec<(usize, Pick5<N>, i64)>) = (
        Vec::new(),
        Vec::new(),
    );
    let cp_paths_here = if CHECK_CP_ROWS.load(Ordering::Relaxed) && N::QUADRATURE {
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
        let cp_terms_here = cp_paths_here
            .as_ref()
            .map(|paths| {
                cp_terms(
                    t,
                    ar,
                    &rep,
                    paths,
                    (picked3, lead_pick3, flagged3, base3, o.num, o.den),
                    (tb3, fire3),
                    &s.live3,
                    total2,
                    quad,
                )
            });
        let mut prof: Option<Profiles<N>> = None;
        let mut sets4: std::collections::HashSet<u64> = Default::default();
        #[cfg(feature = "gpu")]
        let mut sit_id: Option<i32> = None;
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
                if COUNT_CHANGES.load(Ordering::Relaxed) {
                    count_changes(t, &s, &p, sc, &mut sets4);
                }
                if let Some(log) = SLOT6_STATS.get() {
                    slot6_stats(t, &s, &p, sc, log);
                }
                let unchanged = profile && p.at4_base.is_none()
                    && s
                        .live3
                        .iter()
                        .enumerate()
                        .all(|(i, l3)| {
                            s.rmax[i] == 0
                                || p
                                    .live4
                                    .iter()
                                    .any(|l4| l4.mi == l3.mi && l4.mask == l3.mask)
                        });
                #[cfg(feature = "gpu")]
                if !unchanged && gpu {
                    let plan = quad_plan(
                        quad,
                        &s.mask2,
                        &s.live3,
                        &s.rmin,
                        &s.rmax,
                        p.base4,
                        &p.live4,
                        s.f3,
                        p.f4,
                    );
                    let si = *sit_id.get_or_insert_with(|| l.batch.situation(&s));
                    let at = l.batch.pick(t, si, &s, &p, &plan, quad);
                    deferred.push((held.len(), p, at));
                    if l.batch.words() > GPU_FLUSH_WORDS {
                        flush_gpu(
                            t,
                            ar,
                            &mut l.batch,
                            &mut held,
                            Some(&s),
                            &mut deferred,
                            table5,
                            fold,
                        );
                        sit_id = None;
                    }
                    continue;
                }
                let mut rows = p.rows(&s);
                if unchanged {
                    let pr = prof
                        .get_or_insert_with(|| profiles5(
                            t,
                            ar,
                            &s.mask2,
                            &s.live3,
                            &s.keys,
                            &s.dense,
                            &s.present,
                            s.nc(),
                            &s.rmin,
                            &s.rmax,
                        ));
                    profile_rows(&s, &p, pr, &mut rows);
                } else {
                    class_rows(t, &s, &p, grid, quad, polys, &mut sx, &mut rows);
                    if let Some(terms) = &cp_terms_here {
                        let mut cp = p.rows(&s);
                        cp_rows(t, &s, &p, terms, quad, &mut cp);
                        CP_CHECK.record(&rows, &cp, s.keys.len(), terms.len());
                    }
                }
                finish_pick(t, ar, &s, &p, &rows, table5, fold);
            }
        }
        if COUNT_CHANGES.load(Ordering::Relaxed) {
            CHANGES.situation(0, sets4.len());
        }
        #[cfg(feature = "gpu")]
        if sit_id.is_some() {
            held.push(s);
        }
    }
    #[cfg(feature = "gpu")]
    if gpu {
        flush_gpu(t, ar, &mut l.batch, &mut held, None, &mut deferred, table5, fold);
    }
}
/// How much device memory a thread's batch takes, in 8-byte words, before
/// it runs mid-item: 64 MB, against at most 16 calls in flight.
#[cfg(feature = "gpu")]
const GPU_FLUSH_WORDS: usize = 1 << 23;
/// Runs a thread's batch and finishes its picks: those of the held
/// situations, and those of `current`, the one still being walked.
#[cfg(feature = "gpu")]
fn flush_gpu<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    batch: &mut gpu::Batch,
    held: &mut Vec<Situation<N>>,
    current: Option<&Situation<N>>,
    deferred: &mut Vec<(usize, Pick5<N>, i64)>,
    table5: &Sharded<Ctx, KDeco<N>>,
    fold: &mut Weights<N>,
) {
    if batch.is_empty() {
        return;
    }
    batch.run();
    for (si, p, at) in deferred.drain(..) {
        let s = held
            .get(si)
            .or(current)
            .expect("a deferred pick's situation is held or current");
        let rows = batch.rows::<N>(at, (1 + p.corr.len()) * s.nc());
        finish_pick(t, ar, s, &p, &rows, table5, fold);
    }
    held.clear();
    batch.clear();
}
/// One order of a label group's two members, its labels unsplit: what the
/// rank-one form takes from steps 1 and 2 (`--check-cp` holds the identity
/// to the held states).
struct CpPath {
    /// The contexts after step 2, times p₁/den₂, and num₂ when fixed.
    deco: Vec<(u16, f64)>,
    /// M₁'s fixed part: every entry but the random bases'.
    f1: u32,
    /// The multi-forme bases whose count at size 2 is random: (base, q₁, q₂,
    /// μ), an entry surviving step 1 with q₁ and then step 2 with q₂.
    random: Vec<(u16, f64, f64, u8)>,
    /// The second pick's base when its count after step 1 is random (q₁, μ):
    /// its weight is that count, t₁∂/∂t₁.
    second: Option<(f64, u8)>,
}
/// The paths to `rep`'s label group from the empty team.
fn cp_paths<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    rep: &Key,
    sc: &mut Scratch,
) -> Vec<CpPath> {
    let k0 = Key::empty();
    let deco0: Deco<N> = vec![(0, N::ratio(1, 1))];
    let mut steps1: Vec<(Key, Deco<N>)> = Vec::new();
    let mut live0: Vec<Live> = Vec::new();
    walk(
        t,
        ar,
        &k0,
        &deco0,
        None,
        sc,
        &mut |step, live| {
            if live0.is_empty() {
                live0 = live.to_vec();
            }
            if rep.members().contains(&step.class) {
                steps1
                    .push((
                        step.child(&k0),
                        step.deco.iter().map(|&(c, x)| (c, x * step.p)).collect(),
                    ));
            }
        },
    );
    let mut out = Vec::new();
    for (child1, deco1) in &steps1 {
        let at = rep
            .members()
            .iter()
            .position(|&c| c == child1.members()[0])
            .expect("a member");
        let second = rep.members()[1 - at];
        let mut steps2: Vec<(Key, u16, u32, u32, Deco<N>, Vec<Live>, u32)> = Vec::new();
        walk(
            t,
            ar,
            child1,
            deco1,
            None,
            sc,
            &mut |step, live| {
                if step.class == second {
                    steps2
                        .push((
                            step.child(child1),
                            step.base,
                            step.num,
                            step.den,
                            step.deco.clone(),
                            live.to_vec(),
                            step.total,
                        ));
                }
            },
        );
        for (child2, base2, num2, den2, deco2, live1, total1) in steps2 {
            if (child2.m, child2.lead, child2.tb, child2.fire)
                != (rep.m, rep.lead, rep.tb, rep.fire)
            {
                continue;
            }
            let ctx2 = Ctx::of_key(t, &child2);
            let (mut f1, mut weight, mut random, mut second_random) = (
                total1,
                1.0 / f64::from(den2),
                Vec::new(),
                None,
            );
            for l in &live1 {
                let mask0 = live0
                    .iter()
                    .find(|l0| l0.base == l.base)
                    .expect("admissibility only shrinks")
                    .mask;
                let q1 = f64::from(l.mask.count_ones()) / f64::from(mask0.count_ones());
                if l.base == base2 {
                    if q1 < 1.0 {
                        second_random = Some((q1, l.alive));
                        f1 -= u32::from(l.alive);
                    } else {
                        weight *= f64::from(num2);
                    }
                    continue;
                }
                let q2 = f64::from(forme_mask(t, &ctx2, l.base).count_ones())
                    / f64::from(l.mask.count_ones());
                if q1 == 1.0 && (q2 == 1.0 || q2 == 0.0) {
                    continue;
                }
                random.push((l.base, q1, q2, l.alive));
                f1 -= u32::from(l.alive);
            }
            if !live1.iter().any(|l| l.base == base2) {
                weight *= f64::from(num2);
            }
            out.push(CpPath {
                deco: deco2.iter().map(|&(c, x)| (c, N::to_f64(x) * weight)).collect(),
                f1,
                random,
                second: second_random,
            });
        }
    }
    out
}
/// One term of a situation's class sum in rank-one form, for one path and
/// one (t₁, t₂) node: Σ_c X(c)·∏_b T_b[r_b(c)] = Σ_terms deco·w·∏_b
/// Σ_r coef_b[r]·T_b[r], over the situation's live bases whose count at size
/// 2 is random (by size-3 live index); the rest are fixed, as in a class.
struct CpTerm {
    deco: Vec<(u16, f64)>,
    w: f64,
    coef: Vec<(usize, [f64; 4])>,
    /// Each coefficient set as (α + βG)^μ, its (α, β, μ), in `coef`'s order.
    lin: Vec<(f64, f64, u8)>,
}
/// A situation's terms: the paths through the third pick, whose 1/M₂ is
/// ∫t₂^{M₂−1} and whose weight, for a multi-forme pick, its base's count at
/// size 2 (t₂∂/∂t₂). `total2` is M₂ with every live entry; `num3`/`den3` the
/// pick's weight at the label-free state.
#[allow(clippy::too_many_arguments)]
fn cp_terms<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    rep: &Key,
    paths: &[CpPath],
    pick3: (u16, bool, bool, u16, u32, u32),
    outcome: (bool, u8),
    live3: &[Live],
    total2: u32,
    quad: &mut Quad,
) -> Vec<CpTerm> {
    let (picked3, lead_pick3, flagged3, base3, num3, den3) = pick3;
    let mut out = Vec::new();
    let mut split = Vec::new();
    for path in paths {
        let deco: Deco<N> = path.deco.iter().map(|&(c, x)| (c, N::of_f64(x))).collect();
        split.clear();
        if flagged3 {
            kernel_split(t, ar, rep, picked3, lead_pick3, &deco, &mut split);
        } else {
            split.push((rep.tb, rep.fire, deco));
        }
        let Some((_, _, deco3)) = split.iter().find(|e| (e.0, e.1) == outcome) else {
            continue
        };
        let deco3: Vec<(u16, f64)> = deco3
            .iter()
            .map(|&(c, x)| (c, N::to_f64(x)))
            .collect();
        let f2 = total2
            - path
                .random
                .iter()
                .filter(|r| r.2 > 0.0)
                .map(|r| u32::from(r.3))
                .sum::<u32>();
        let d1 = path.random.iter().map(|r| u32::from(r.3)).sum::<u32>()
            + path.second.map_or(0, |s| u32::from(s.1));
        let d2 = path
            .random
            .iter()
            .filter(|r| r.2 > 0.0)
            .map(|r| u32::from(r.3))
            .sum::<u32>();
        let (x1, w1) = quad.rule(path.f1, (d1 / 2 + 1) as usize).clone();
        let (x2, w2) = quad.rule(f2, (d2 / 2 + 1) as usize).clone();
        let third_random = path.random.iter().find(|r| r.0 == base3).copied();
        let base_w = if third_random.is_some() { 1.0 } else { f64::from(num3) }
            / f64::from(den3);
        for (&t1, &v1) in x1.iter().zip(&w1) {
            for (&t2, &v2) in x2.iter().zip(&w2) {
                let mut w = v1 * v2 * base_w;
                if let Some((q1, mu)) = path.second {
                    w
                        *= f64::from(mu) * q1 * t1
                            * ((1.0 - q1) + q1 * t1).powi(i32::from(mu) - 1);
                }
                let (mut coef, mut lin) = (Vec::new(), Vec::new());
                for &(b, q1, q2, mu) in &path.random {
                    let (alpha, beta) = (
                        (1.0 - q1) + q1 * t1 * (1.0 - q2),
                        q1 * q2 * t1 * t2,
                    );
                    if b == base3 {
                        w
                            *= f64::from(mu) * beta
                                * (alpha + beta).powi(i32::from(mu) - 1);
                    } else if let Some(i3) = live3.iter().position(|l| l.base == b) {
                        let mut c = [0.0; 4];
                        for (r, cr) in c.iter_mut().enumerate().take(usize::from(mu) + 1)
                        {
                            *cr = binom(mu, r as u8) as f64
                                * alpha.powi(i32::from(mu) - r as i32)
                                * beta.powi(r as i32);
                        }
                        coef.push((i3, c));
                        lin.push((alpha, beta, mu));
                    } else {
                        w *= (alpha + beta).powi(i32::from(mu));
                    }
                }
                out.push(CpTerm {
                    deco: deco3.clone(),
                    w,
                    coef,
                    lin,
                });
            }
        }
    }
    out
}
/// [`class_rows`] from a situation's rank-one terms instead of its classes:
/// each class's ∏_b T_b[r_b] becomes, per term, ∏_b Σ_r coef_b[r]·T_b[r],
/// and a row that weights a class by a base's count (the fourth pick's own
/// base, a moment, a fixed correction) weights that base's coefficients
/// instead. A reference, not fast: every row recomputes its products.
fn cp_rows<N: Num>(
    t: &Tables,
    s: &Situation<N>,
    p: &Pick5<N>,
    terms: &[CpTerm],
    quad: &mut Quad,
    rows: &mut Rows<N>,
) {
    let nc = s.nc();
    let plan = quad_plan(
        quad,
        &s.mask2,
        &s.live3,
        &s.rmin,
        &s.rmax,
        p.base4,
        &p.live4,
        s.f3,
        p.f4,
    );
    quad5(quad, &plan, &[], p.live4.len(), &mut [], &mut []);
    let np = plan.xs.len() * plan.ys.len();
    let pp = np.div_ceil(8) * 8;
    let code_at = |code: u16| s.codes.binary_search(&code).expect("a situation code");
    let fixed_r = |i3: usize| s.rmin[i3];
    let eval = |
        rowmods: &[(usize, &dyn Fn(u8) -> f64)],
        functional: Option<usize>,
        term: &CpTerm,
    | -> f64 {
        let mut scale = 1.0;
        for &(i3, f) in rowmods {
            if !term.coef.iter().any(|c| c.0 == i3) {
                scale *= f(fixed_r(i3));
            }
        }
        if scale == 0.0 {
            return 0.0;
        }
        let mut sum = 0.0;
        for node in 0..np {
            let mut v = quad.common[node];
            for (k, &(i, ..)) in plan.variant.iter().enumerate() {
                let c = term
                    .coef
                    .iter()
                    .find(|c| c.0 == i)
                    .map(|c| c.1)
                    .expect("a varying base is random at size 2");
                let mut phi = 0.0;
                for (r, &cr) in c.iter().enumerate() {
                    let m = rowmods
                        .iter()
                        .fold(
                            1.0,
                            |m, &(i3, f)| if i3 == i { m * f(r as u8) } else { m },
                        );
                    phi += cr * m * quad.tables[(k * 4 + r) * pp + node];
                }
                v *= phi;
            }
            if let Some(k) = functional {
                v *= quad.ratios[k * pp + node];
            }
            sum += v;
        }
        sum * scale * term.w
    };
    let w4 = |r: u8| f64::from(r) / f64::from(p.den);
    let base4: Vec<(usize, &dyn Fn(u8) -> f64)> = match p.at4_base {
        Some(i4) => vec![(i4, & w4)],
        None => Vec::new(),
    };
    let c4 = if p.at4_base.is_some() {
        1.0
    } else {
        f64::from(p.num) / f64::from(p.den)
    };
    for term in terms {
        let inv = c4 * eval(&base4, None, term);
        for &(code, x) in &term.deco {
            let d = code_at(code);
            rows.acc[d] += N::of_f64(x * inv);
            rows.touched[d] = true;
        }
        for (k, &(j, i3, size, omega, full)) in p.corr.iter().enumerate() {
            let omega = N::to_f64(omega);
            let mu = t.bases[p.live4[j].base as usize].mu;
            let value = if p.random4[j] {
                let alive = |r: u8| f64::from(r) / size as f64;
                let mut mods = base4.clone();
                mods.push((i3, &alive));
                let moment = plan.random.iter().position(|r| r.1 == j);
                let m = moment.map_or(0.0, |kk| c4 * eval(&mods, Some(kk), term));
                m - omega * inv
            } else {
                let phi = move |r: u8| {
                    if full && r == mu {
                        0.0
                    } else {
                        f64::from(r) / size as f64 - omega
                    }
                };
                if !term.coef.iter().any(|c| c.0 == i3) && full && fixed_r(i3) == mu {
                    continue;
                }
                let mut mods = base4.clone();
                mods.push((i3, &phi));
                c4 * eval(&mods, None, term)
            };
            let row = (1 + k) * nc;
            for &(code, x) in &term.deco {
                let d = code_at(code);
                rows.acc[row + d] += N::of_f64(x * value);
                rows.touched[row + d] = true;
            }
        }
    }
}
static CHECK_CP_ROWS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(
    false,
);
/// `--check-cp-rows`: the worst gap between a pick's rows from its classes
/// and from the rank-one terms, over every cell either touches.
struct CpCheck {
    state: Mutex<(f64, String, u64, u64, u64)>,
}
static CP_CHECK: CpCheck = CpCheck {
    state: Mutex::new((0.0, String::new(), 0, 0, 0)),
};
impl CpCheck {
    fn record<N: Num>(&self, a: &Rows<N>, b: &Rows<N>, classes: usize, terms: usize) {
        let mut st = self.state.lock().unwrap();
        st.2 += 1;
        st.3 += classes as u64;
        st.4 += terms as u64;
        for i in 0..a.acc.len() {
            let (x, y) = (N::to_f64(N::get(a.acc[i])), N::to_f64(N::get(b.acc[i])));
            let scale = x.abs().max(y.abs());
            if scale < 1e-30 && a.touched[i] == b.touched[i] {
                continue;
            }
            let rel = if a.touched[i] != b.touched[i] {
                f64::INFINITY
            } else {
                (x - y).abs() / scale
            };
            if rel > st.0 {
                st.0 = rel;
                st.1 = format!(
                    "cell {i}: classes {x:.6e} ({}), rank-one {y:.6e} ({})", a
                    .touched[i], b.touched[i]
                );
            }
        }
    }
    fn report(&self) {
        let st = self.state.lock().unwrap();
        println!(
            "check-cp-rows over {} picks ({:.1} classes, {:.1} terms a pick): worst relative gap {:.2e} ({})",
            st.2, st.3 as f64 / st.2.max(1) as f64, st.4 as f64 / st.2.max(1) as f64, st
            .0, st.1
        );
    }
}
/// The pre-kernel sums of one pick over its situation's classes, dense in
/// the situation's codes: row 0 the E[w₄/(M₃M₄)]-weighted contexts, row
/// 1 + k the k-th correctable base's per-forme coefficient-weighted ones;
/// `touched` where any class added anything.
struct Rows<N: Num> {
    acc: Vec<N::Acc>,
    touched: Vec<bool>,
}
/// One size-3 situation of a slot-5 item — a (label group, third pick,
/// flags) outcome: its team, live bases and classes, the group's variants
/// whose alive counts agree on every size-3 live base, their contexts
/// summed and laid dense over the codes any of them carries.
struct Situation<N: Num> {
    c3: Key,
    live3: Vec<Live>,
    mask2: [u32; 64],
    at3_of: [u8; 64],
    f3: u32,
    keys: Vec<u128>,
    codes: Vec<u16>,
    dense: Vec<N>,
    present: Vec<bool>,
    rmin: Vec<u8>,
    rmax: Vec<u8>,
    union: Deco<N>,
}
impl<N: Num> Situation<N> {
    /// `c3`'s situation from the group `variants` reaching it, each with its
    /// contexts past the third pick; leaves `c3`'s options in `sc3`.
    #[inline(never)]
    fn new(
        t: &Tables,
        c3: Key,
        mask2: [u32; 64],
        group: &[(Key, Deco<N>)],
        variants: &[(usize, Deco<N>)],
        sc3: &mut Scratch,
    ) -> Self {
        let ctx3 = Ctx::of_key(t, &c3);
        let av3 = Avail::of_key(t, &c3);
        let total3 = options(t, &c3, &ctx3, &av3, sc3);
        let live3 = sc3.live.clone();
        let f3 = total3 - live3.iter().map(|l| u32::from(l.alive)).sum::<u32>();
        assert!(live3.len() <= 64, "{} live bases", live3.len());
        let mut at3_of = [u8::MAX; 64];
        for (i, l) in live3.iter().enumerate() {
            at3_of[usize::from(l.mi)] = i as u8;
        }
        let alive_of = |k: &Key, base: u16| -> u8 {
            let b = &t.bases[base as usize];
            b.mu - k.dead_of(b.mi)
        };
        let mut index: FxMap<u128, usize> = FxMap::default();
        let mut classes: Vec<(u128, KDeco<N>)> = Vec::new();
        for (v, d) in variants {
            let k = &group[*v].0;
            let key = live3
                .iter()
                .enumerate()
                .fold(
                    0u128,
                    |acc, (i, l)| acc | u128::from(alive_of(k, l.base)) << (2 * i),
                );
            let c = *index
                .entry(key)
                .or_insert_with(|| {
                    classes.push((key, Vec::new()));
                    classes.len() - 1
                });
            for &(code, x) in d {
                kdeco_add::<N>(&mut classes[c].1, code, x);
            }
        }
        let classes: Vec<(u128, Deco<N>)> = classes
            .iter()
            .map(|(key, d)| (*key, to_deco(d)))
            .collect();
        let keys: Vec<u128> = classes.iter().map(|c| c.0).collect();
        let mut codes: Vec<u16> = classes
            .iter()
            .flat_map(|c| c.1.iter().map(|e| e.0))
            .collect();
        codes.sort_unstable();
        codes.dedup();
        let nc = codes.len();
        let mut dense = vec![N::default(); classes.len() * nc];
        let mut present = vec![false; classes.len() * nc];
        let mut union: Deco<N> = Vec::new();
        for (ci, (_, d)) in classes.iter().enumerate() {
            for &(code, x) in d {
                let at = ci * nc + codes.binary_search(&code).unwrap();
                dense[at] = x;
                present[at] = true;
                deco_add(&mut union, code, x);
            }
        }
        let mut s = Situation {
            c3,
            live3,
            mask2,
            at3_of,
            f3,
            keys,
            codes,
            dense,
            present,
            rmin: Vec::new(),
            rmax: Vec::new(),
            union,
        };
        s.rmin = (0..s.live3.len())
            .map(|i| (0..s.keys.len()).map(|c| s.r(c, i)).min().unwrap_or(0))
            .collect();
        s.rmax = (0..s.live3.len())
            .map(|i| (0..s.keys.len()).map(|c| s.r(c, i)).max().unwrap_or(0))
            .collect();
        s
    }
    fn nc(&self) -> usize {
        self.codes.len()
    }
    /// Class `c`'s alive count on live3 base `i`.
    fn r(&self, c: usize, i: usize) -> u8 {
        ((self.keys[c] >> (2 * i)) & 3) as u8
    }
}
/// One fourth pick (and the flags it lands on) of a situation: its team,
/// its pool's fixed part and live bases, which of those have random counts,
/// and the corrections its classes can owe — (live4 index, live3 index,
/// |mask|, ω, whether the mask is full).
struct Pick5<N: Num> {
    c4: Key,
    ctx4: Ctx,
    av4: Avail,
    picked4: u16,
    lead_pick4: bool,
    flagged4: bool,
    base4: u16,
    at4_base: Option<usize>,
    num: u32,
    den: u32,
    live4: Vec<Live>,
    f4: u32,
    random4: Vec<bool>,
    corr: Vec<(usize, usize, u64, N, bool)>,
}
impl<N: Num> Pick5<N> {
    #[inline(never)]
    fn new(
        t: &Tables,
        ar: &Arith<N>,
        s: &Situation<N>,
        c4: Key,
        o4: &Opt,
        picked4: u16,
        lead_pick4: bool,
        flagged4: bool,
        base4: u16,
        sc4: &mut Scratch,
    ) -> Option<Self> {
        let ctx4 = Ctx::of_key(t, &c4);
        let av4 = Avail::of_key(t, &c4);
        let total4 = options_into(t, &c4, &ctx4, &av4, &mut sc4.live, None);
        if total4 == 0 {
            return None;
        }
        let live4 = sc4.live.clone();
        let f4 = total4 - live4.iter().map(|l| u32::from(l.alive)).sum::<u32>();
        let random4: Vec<bool> = live4
            .iter()
            .map(|l4| {
                let l3 = &s.live3[usize::from(s.at3_of[usize::from(l4.mi)])];
                l3.mask != s.mask2[usize::from(l3.mi)] || l4.mask != l3.mask
            })
            .collect();
        let corr = live4
            .iter()
            .enumerate()
            .filter_map(|(j, l4)| {
                let base = &t.bases[l4.base as usize];
                let i3 = usize::from(s.at3_of[usize::from(l4.mi)]);
                let full = l4.mask == (1u32 << base.formes.len()) - 1;
                (random4[j] || !full || s.rmin[i3] < base.mu)
                    .then(|| (
                        j,
                        i3,
                        u64::from(l4.mask.count_ones()),
                        ar.omega[base.formes[0] as usize],
                        full,
                    ))
            })
            .collect();
        let multi = base4 != NONE && t.bases[base4 as usize].mi != u8::MAX;
        let at4_base = multi
            .then(|| usize::from(s.at3_of[usize::from(t.bases[base4 as usize].mi)]));
        Some(Pick5 {
            c4,
            ctx4,
            av4,
            picked4,
            lead_pick4,
            flagged4,
            base4,
            at4_base,
            num: o4.num,
            den: o4.den,
            live4,
            f4,
            random4,
            corr,
        })
    }
    fn rows(&self, s: &Situation<N>) -> Rows<N> {
        let n = (1 + self.corr.len()) * s.nc();
        Rows {
            acc: vec![< N::Acc >::default(); n],
            touched: vec![false; n],
        }
    }
}
/// The per-class buffers of [`class_rows`], kept across picks.
#[derive(Default)]
struct ClassScratch<N> {
    i0s: Vec<N>,
    moms: Vec<N>,
    qi0: Vec<f64>,
    qmoms: Vec<f64>,
    moments: Vec<(usize, N)>,
    live3_c: Vec<Live>,
}
/// [`Rows`] from the situation's profiles: an unchanged pick contracts
/// them with its weights 1/((f₃+k)(f₄+k)).
#[inline(never)]
fn profile_rows<N: Num>(
    s: &Situation<N>,
    p: &Pick5<N>,
    pr: &Profiles<N>,
    rows: &mut Rows<N>,
) {
    let nc = s.nc();
    let c = N::ratio(u64::from(p.num), u64::from(p.den));
    let w: Vec<N> = (0..=pr.degree as u64)
        .map(|k| N::ratio(1, (u64::from(s.f3) + k) * (u64::from(p.f4) + k)) * c)
        .collect();
    let mut contract = |r: usize, sums: &[N::Acc], codes_in: &[bool]| {
        for d in 0..nc {
            if codes_in[d] {
                let v = w
                    .iter()
                    .enumerate()
                    .fold(
                        N::default(),
                        |acc, (k, &wk)| acc + wk * N::get(sums[k * nc + d]),
                    );
                rows.acc[r * nc + d] += v;
                rows.touched[r * nc + d] = true;
            }
        }
    };
    contract(0, &pr.a, &pr.touched);
    for (k, &(_, i3, ..)) in p.corr.iter().enumerate() {
        let (sums, codes_in) = pr
            .rows[i3]
            .as_ref()
            .expect("an unchanged pick's corrections were all profiled");
        contract(1 + k, sums, codes_in);
    }
    PSHAPE.0.fetch_add(1, Ordering::Relaxed);
    PSHAPE.1.fetch_add(s.keys.len() as u64, Ordering::Relaxed);
}
/// [`Rows`] class by class: each class's integrals (quadrature, or the
/// grid), then its share of every row.
fn class_rows<N: Num>(
    t: &Tables,
    s: &Situation<N>,
    p: &Pick5<N>,
    grid: bool,
    quad: &mut Quad,
    polys: &mut PolyBuf<N>,
    sx: &mut ClassScratch<N>,
    rows: &mut Rows<N>,
) {
    let (nc, nl4, classes) = (s.nc(), p.live4.len(), s.keys.len());
    let ClassScratch { i0s, moms, qi0, qmoms, moments, live3_c } = sx;
    i0s.clear();
    i0s.resize(classes, N::default());
    moms.clear();
    moms.resize(classes * nl4, N::default());
    if N::QUADRATURE && !grid {
        qi0.clear();
        qi0.resize(classes, 0.0);
        qmoms.clear();
        qmoms.resize(classes * nl4, 0.0);
        let plan = quad_plan(
            quad,
            &s.mask2,
            &s.live3,
            &s.rmin,
            &s.rmax,
            p.base4,
            &p.live4,
            s.f3,
            p.f4,
        );
        quad5(quad, &plan, &s.keys, nl4, qi0, qmoms);
        for (x, &q) in i0s.iter_mut().zip(qi0.iter()) {
            *x = N::of_f64(q);
        }
        for (x, &q) in moms.iter_mut().zip(qmoms.iter()) {
            *x = N::of_f64(q);
        }
    } else {
        for c in 0..classes {
            if p.at4_base.is_some_and(|i| s.r(c, i) == 0) {
                continue;
            }
            live3_c.clear();
            live3_c
                .extend(
                    s
                        .live3
                        .iter()
                        .enumerate()
                        .map(|(i, l)| Live { alive: s.r(c, i), ..*l }),
                );
            moments.clear();
            i0s[c] = integrate5(
                &s.mask2,
                live3_c,
                p.base4,
                &p.live4,
                s.f3,
                p.f4,
                polys,
                moments,
            );
            for &(j, m) in moments.iter() {
                moms[c * nl4 + j] = m;
            }
        }
    }
    for c in 0..classes {
        let num4 = p.at4_base.map_or(p.num, |i| u32::from(s.r(c, i)));
        if num4 == 0 {
            continue;
        }
        let w4 = N::ratio(u64::from(num4), u64::from(p.den));
        let inv = w4 * i0s[c];
        let x = &s.dense[c * nc..][..nc];
        let present = &s.present[c * nc..][..nc];
        for d in 0..nc {
            if present[d] {
                rows.acc[d] += x[d] * inv;
                rows.touched[d] = true;
            }
        }
        for (k, &(j, i3, size, omega, full)) in p.corr.iter().enumerate() {
            let alive = s.r(c, i3);
            let per_forme = if p.random4[j] {
                w4 * moms[c * nl4 + j] * N::ratio(1, size)
            } else if full && alive == t.bases[p.live4[j].base as usize].mu {
                continue;
            } else {
                N::ratio(u64::from(alive), size) * inv
            };
            let coefficient = per_forme - omega * inv;
            let row = (1 + k) * nc;
            for d in 0..nc {
                if present[d] {
                    rows.acc[row + d] += x[d] * coefficient;
                    rows.touched[row + d] = true;
                }
            }
        }
    }
}
/// A pick's rows past its own set, into the signature table and the
/// corrections.
#[inline(never)]
fn finish_pick<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    s: &Situation<N>,
    p: &Pick5<N>,
    rows: &Rows<N>,
    table5: &Sharded<Ctx, KDeco<N>>,
    fold: &mut Weights<N>,
) {
    let nc = s.nc();
    let row_deco = |r: usize| -> Deco<N> {
        (0..nc)
            .filter(|&d| rows.touched[r * nc + d])
            .map(|d| (s.codes[d], N::get(rows.acc[r * nc + d])))
            .collect()
    };
    let w0 = row_deco(0);
    if w0.is_empty() {
        return;
    }
    let past = |d: Deco<N>| kernel_part(
        t,
        ar,
        &s.c3,
        p.picked4,
        p.lead_pick4,
        p.flagged4,
        p.c4.tb,
        p.c4.fire,
        d,
    );
    let u = past(w0);
    credit_signature(t, ar, &p.c4, &p.ctx4, &p.av4, &u, table5, fold);
    let tb_code = if p.c4.tb { TB_BIT } else { 0 };
    for (k, &(j, ..)) in p.corr.iter().enumerate() {
        let wk = row_deco(1 + k);
        if wk.is_empty() {
            continue;
        }
        let deco = past(wk);
        let l4 = &p.live4[j];
        let base = &t.bases[l4.base as usize];
        for (i, &f) in base.formes.iter().enumerate() {
            if l4.mask & (1 << i) != 0 {
                let (sl, lead) = pick_slot(&p.c4, &t.sp[f as usize]);
                for &(code, x) in &deco {
                    credit(fold, f, sl, lead, code | tb_code, x);
                }
            }
        }
    }
}
/// `--count-changes` for one pick: the live bases with entries it changes,
/// and those each fifth pick from it changes.
fn count_changes<N: Num>(
    t: &Tables,
    s: &Situation<N>,
    p: &Pick5<N>,
    sc: &mut Scratch,
    sets4: &mut std::collections::HashSet<u64>,
) {
    let mut b4 = 0u64;
    for (i, l3) in s.live3.iter().enumerate() {
        if s.rmax[i] > 0
            && !p.live4.iter().any(|l4| l4.mi == l3.mi && l4.mask == l3.mask)
        {
            b4 |= 1 << i;
        }
    }
    CHANGES.record(0, b4, s.keys.len());
    sets4.insert(b4);
    let mut sets5: std::collections::HashSet<u64> = Default::default();
    options(t, &p.c4, &p.ctx4, &p.av4, sc);
    for o5 in &sc.opts {
        let (class5, _, _, base5) = pick_parts(t, &p.c4, o5.pick);
        let mut c5 = p.c4.with(class5);
        c5.lead = true;
        let ctx5 = Ctx::of_key(t, &c5);
        let mut b5 = 0u64;
        for (j, l4) in p.live4.iter().enumerate() {
            let i3 = usize::from(s.at3_of[usize::from(l4.mi)]);
            if s.rmax[i3] > 0
                && (l4.base == base5 || forme_mask(t, &ctx5, l4.base) != l4.mask)
            {
                b5 |= 1 << j;
            }
        }
        CHANGES.record(1, b5, s.keys.len());
        sets5.insert(b5);
    }
    CHANGES.situation(1, sets5.len());
}
/// The class sums behind every fourth pick that leaves each live base with
/// entries its formes (and is no live base's forme). Such a pick has q₄ = 1
/// everywhere, so each entry's factor is (1 − q₃) + q₃·xy and the integrand
/// x^{f₃−1}y^{f₄−1}P_c(xy): E[w₄/(M₃M₄)] = Σ_k p_{c,k}/((f₃+k)(f₄+k)), each
/// moment the same with its base's factor u∂/∂u'd. So the sums over the
/// classes of p_{c,k}·contexts_c — and of each correction coefficient's —
/// depend on the size-3 situation alone, and a pick contracts them with
/// 1/((f₃+k)(f₄+k)): `a[k · nc + d]`, `rows[i]` the same for live3 base i
/// (None where no class owes it anything), `touched[d]` per code.
struct Profiles<N: Num> {
    degree: usize,
    a: Vec<N::Acc>,
    touched: Vec<bool>,
    rows: Vec<Option<(Vec<N::Acc>, Vec<bool>)>>,
}
/// `p / (a + q·u)` in floats, for a factor that divides `p`: from the top
/// when q ≥ a and from the bottom otherwise, so no step multiplies an error
/// by more than 1.
fn div_linear<N: Num>(p: &[N], a: N, q: N, out: &mut Vec<N>) {
    let d = p.len() - 1;
    out.clear();
    out.resize(d, N::default());
    let (af, qf) = (a.to_f64(), q.to_f64());
    if qf == 0.0 {
        return;
    }
    if qf >= af {
        let inv = N::of_f64(1.0 / qf);
        out[d - 1] = p[d] * inv;
        for k in (1..d).rev() {
            out[k - 1] = (p[k] - a * out[k]) * inv;
        }
    } else {
        let inv = N::of_f64(1.0 / af);
        out[0] = p[0] * inv;
        for k in 1..d {
            out[k] = (p[k] - q * out[k - 1]) * inv;
        }
    }
}
#[inline(never)]
fn profiles5<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    mask2: &[u32; 64],
    live3: &[Live],
    keys: &[u128],
    dense: &[N],
    present: &[bool],
    nc: usize,
    rmin: &[u8],
    rmax: &[u8],
) -> Profiles<N> {
    let nl3 = live3.len();
    let degree: usize = rmax.iter().map(|&r| usize::from(r)).sum();
    let width = (degree + 1) * nc;
    let r_of = |key: u128, i: usize| ((key >> (2 * i)) & 3) as usize;
    let bases: Vec<(N, N, bool, u64, N, bool, u8)> = live3
        .iter()
        .map(|l| {
            let b = &t.bases[l.base as usize];
            let (n3, d3) = (
                u64::from(l.mask.count_ones()),
                u64::from(mask2[usize::from(l.mi)].count_ones()),
            );
            let full = l.mask == (1u32 << b.formes.len()) - 1;
            (
                N::ratio(d3 - n3, d3),
                N::ratio(n3, d3),
                n3 != d3,
                n3,
                ar.omega[b.formes[0] as usize],
                full,
                b.mu,
            )
        })
        .collect();
    let owes = |i: usize| {
        let (_, _, split, _, _, full, mu) = bases[i];
        split || !full || rmin[i] < mu
    };
    let mut p = Profiles {
        degree,
        a: vec![< N::Acc >::default(); width],
        touched: (0..nc).map(|d| (0..keys.len()).any(|c| present[c * nc + d])).collect(),
        rows: (0..nl3)
            .map(|i| {
                owes(i).then(|| (vec![< N::Acc >::default(); width], vec![false; nc]))
            })
            .collect(),
    };
    fn times_linear<N: Num>(poly: &mut Vec<N>, a: N, q: N, r: usize) {
        for _ in 0..r {
            poly.push(N::default());
            for k in (1..poly.len()).rev() {
                poly[k] = poly[k] * a + poly[k - 1] * q;
            }
            poly[0] = poly[0] * a;
        }
    }
    let mut pre: Vec<Vec<N>> = vec![Vec::new(); nl3 + 1];
    let mut suf: Vec<Vec<N>> = vec![Vec::new(); nl3 + 1];
    let mut m: Vec<N> = Vec::new();
    let mut coef: Vec<N> = Vec::new();
    for (ci, &key) in keys.iter().enumerate() {
        if N::QUADRATURE {
            pre[nl3].clear();
            pre[nl3].push(N::ratio(1, 1));
            for i in 0..nl3 {
                times_linear(&mut pre[nl3], bases[i].0, bases[i].1, r_of(key, i));
            }
        } else {
            pre[0].clear();
            pre[0].push(N::ratio(1, 1));
            for i in 0..nl3 {
                let mut next = pre[i].clone();
                times_linear(&mut next, bases[i].0, bases[i].1, r_of(key, i));
                pre[i + 1] = next;
            }
            suf[nl3].clear();
            suf[nl3].push(N::ratio(1, 1));
            for i in (0..nl3).rev() {
                let mut next = suf[i + 1].clone();
                times_linear(&mut next, bases[i].0, bases[i].1, r_of(key, i));
                suf[i] = next;
            }
        }
        let poly = &pre[nl3];
        let x = &dense[ci * nc..][..nc];
        let add = |acc: &mut [N::Acc], touched: Option<&mut [bool]>, coef: &[N]| {
            for (k, &ck) in coef.iter().enumerate() {
                for d in 0..nc {
                    if present[ci * nc + d] {
                        acc[k * nc + d] += ck * x[d];
                    }
                }
            }
            if let Some(touched) = touched {
                for d in 0..nc {
                    touched[d] |= present[ci * nc + d];
                }
            }
        };
        add(&mut p.a, None, poly);
        for i in 0..nl3 {
            let Some((row, touched)) = p.rows[i].as_mut() else { continue };
            let (a, q, split, size, omega, full, mu) = bases[i];
            let r = r_of(key, i);
            coef.clear();
            if split {
                m.clear();
                if r > 0 {
                    if N::QUADRATURE {
                        div_linear(poly, a, q, &mut m);
                    } else {
                        m.extend_from_slice(&pre[i]);
                        times_linear(&mut m, a, q, r - 1);
                        let mut prod = vec![
                            N::default(); m.len() + suf[i + 1].len() - 1
                        ];
                        for (j, &mj) in m.iter().enumerate() {
                            for (l, &sl) in suf[i + 1].iter().enumerate() {
                                prod[j + l] = prod[j + l] + mj * sl;
                            }
                        }
                        m = prod;
                    }
                    m.insert(0, N::default());
                    let rq = N::ratio(r as u64, 1) * q;
                    for v in &mut m {
                        *v = *v * rq;
                    }
                }
                let inv_size = N::ratio(1, size);
                coef.extend(
                    (0..poly.len())
                        .map(|k| {
                            m.get(k).copied().unwrap_or_default() * inv_size
                                - omega * poly[k]
                        }),
                );
            } else if full && r == usize::from(mu) {
                continue;
            } else {
                let per = N::ratio(r as u64, size) - omega;
                coef.extend(poly.iter().map(|&v| per * v));
            }
            add(row, Some(touched), &coef);
        }
    }
    p
}
static PSHAPE: (AtomicU64, AtomicU64) = (AtomicU64::new(0), AtomicU64::new(0));
/// `--count-changes`: how many live bases with entries a pick changes (its
/// own base included) — [0] the fourth pick against the size-3 live bases,
/// [1] the fifth against the size-4 ones — by picks and by the class
/// evaluations they stand for; and the distinct changed sets per situation.
struct ChangeStats {
    picks: [[AtomicU64; 17]; 2],
    classes: [[AtomicU64; 17]; 2],
    distinct: [AtomicU64; 2],
    situations: [AtomicU64; 2],
}
static CHANGES: ChangeStats = ChangeStats {
    picks: [const { [const { AtomicU64::new(0) }; 17] }; 2],
    classes: [const { [const { AtomicU64::new(0) }; 17] }; 2],
    distinct: [const { AtomicU64::new(0) }; 2],
    situations: [const { AtomicU64::new(0) }; 2],
};
static COUNT_CHANGES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(
    false,
);
/// `--slot6-stats PATH`'s companion, `PATH.split`: a line per fourth pick
/// about how its fifth picks split the live bases.
static SPLIT_LOG: OnceLock<Mutex<std::io::BufWriter<std::fs::File>>> = OnceLock::new();
/// `--slot6-stats PATH`: a line per fourth pick about the fifth picks after
/// it, for pricing slot 6 — see [`slot6_stats`].
static SLOT6_STATS: OnceLock<Mutex<std::io::BufWriter<std::fs::File>>> = OnceLock::new();
/// One fourth pick's fifth picks: how many there are (flag outcomes apart),
/// how many distinct integrands they make — each size-4 live base's mask
/// after step 5 and the pick's own base, then also with f₅ — how many
/// distinct contexts after five picks (slot 6's signature keys), and the
/// largest z-degree: the alive counts, at most the situation's, on the bases
/// still live after step 5. Line: classes, picks, signatures, signatures
/// with f₅, contexts, z-degree, size-3 live bases, x and y degree.
fn slot6_stats<N: Num>(
    t: &Tables,
    s: &Situation<N>,
    p: &Pick5<N>,
    sc: &mut Scratch,
    log: &Mutex<std::io::BufWriter<std::fs::File>>,
) {
    use std::collections::HashSet;
    options(t, &p.c4, &p.ctx4, &p.av4, sc);
    let opts = sc.opts.clone();
    let live4 = sc.live.clone();
    let rmax_of = |l: &Live| s.rmax[usize::from(s.at3_of[usize::from(l.mi)])];
    let (mut picks, mut dz_max) = (0usize, 0u32);
    let (
        mut sigs,
        mut sigs_f5,
        mut ctxs,
    ): (HashSet<(Vec<u32>, u16)>, HashSet<(Vec<u32>, u16, u32)>, HashSet<Ctx>) = Default::default();
    let mut sc5 = Scratch::default();
    for o5 in &opts {
        let (class5, _, _, base5) = pick_parts(t, &p.c4, o5.pick);
        let mut c5 = p.c4.with(class5);
        c5.lead = true;
        let ctx5 = Ctx::of_key(t, &c5);
        let masks: Vec<u32> = live4
            .iter()
            .map(|l| if l.base == base5 { 0 } else { forme_mask(t, &ctx5, l.base) })
            .collect();
        let dz: u32 = live4
            .iter()
            .zip(&masks)
            .filter(|&(_, &m)| m != 0)
            .map(|(l, _)| u32::from(rmax_of(l)))
            .sum();
        dz_max = dz_max.max(dz);
        let total5 = options(t, &c5, &ctx5, &Avail::of_key(t, &c5), &mut sc5);
        let f5 = total5 - sc5.live.iter().map(|l| u32::from(l.alive)).sum::<u32>();
        let own = if base5 != NONE && t.bases[base5 as usize].mi != u8::MAX {
            base5
        } else {
            NONE
        };
        sigs_f5.insert((masks.clone(), own, f5));
        sigs.insert((masks, own));
        ctxs.insert(ctx5);
        picks += 1;
    }
    let dx: u32 = s.rmax.iter().map(|&r| u32::from(r)).sum();
    let ratio = |a: u32, b: u32| {
        f64::from(a.count_ones()) / f64::from(b.count_ones().max(1))
    };
    let mut q34: Vec<(usize, f64, f64, u8, bool)> = Vec::new();
    for (i, l3) in s.live3.iter().enumerate() {
        if s.rmax[i] == 0 || l3.base == p.base4 {
            continue;
        }
        let q3 = ratio(l3.mask, s.mask2[usize::from(l3.mi)]);
        let at4 = live4.iter().position(|l| l.mi == l3.mi);
        let q4 = at4.map_or(0.0, |j| ratio(live4[j].mask, l3.mask));
        q34.push((at4.unwrap_or(usize::MAX), q3, q4, s.rmax[i], s.rmin[i] < s.rmax[i]));
    }
    let (mut sum_changed, mut sum_changed_var, mut dxy, mut dz_sum, mut dz2) = (
        0usize,
        0usize,
        (0u32, 0u32),
        0u32,
        0u32,
    );
    let mut var_sigs: HashSet<Vec<u32>> = Default::default();
    for (masks, _) in &sigs {
        let (mut dxx, mut dyy, mut dzz) = (0u32, 0u32, 0u32);
        let mut var_sig = Vec::new();
        for &(j, q3, q4, r, varies) in &q34 {
            let q5 = if j == usize::MAX || q4 == 0.0 {
                0.0
            } else {
                ratio(masks[j], live4[j].mask)
            };
            if q5 < 1.0 && j != usize::MAX && q4 > 0.0 {
                sum_changed += 1;
                if varies {
                    sum_changed_var += 1;
                }
            }
            if varies {
                var_sig.push(if j == usize::MAX { 0 } else { masks[j] });
            }
            if q3 < 1.0 || q4 < 1.0 || q5 < 1.0 || varies {
                dxx += u32::from(r);
                if q4 > 0.0 {
                    dyy += u32::from(r);
                }
                if q5 > 0.0 {
                    dzz += u32::from(r);
                }
            }
        }
        var_sigs.insert(var_sig);
        dxy = (dxy.0.max(dxx), dxy.1.max(dyy));
        dz_sum += dzz;
        dz2 = dz2.max(dzz);
    }
    let ns = sigs.len().max(1);
    let variants = (0..s.live3.len()).filter(|&i| s.rmin[i] < s.rmax[i]).count();
    let split: Vec<&(usize, f64, f64, u8, bool)> = q34
        .iter()
        .filter(|b| b.1 < 1.0 || b.2 < 1.0 || b.4)
        .collect();
    let sx: u32 = split.iter().map(|b| u32::from(b.3)).sum();
    let sy: u32 = split.iter().filter(|b| b.2 > 0.0).map(|b| u32::from(b.3)).sum();
    let sz: u32 = split
        .iter()
        .filter(|b| b.2 > 0.0 && b.0 != usize::MAX)
        .map(|b| u32::from(b.3))
        .sum();
    let mut patterns: HashSet<Vec<u32>> = Default::default();
    let mut changed_split = 0usize;
    for (masks, _) in &sigs {
        let pat: Vec<u32> = split
            .iter()
            .map(|b| if b.0 == usize::MAX { 0 } else { masks[b.0] })
            .collect();
        changed_split
            += split
                .iter()
                .filter(|b| {
                    b.0 != usize::MAX && b.2 > 0.0 && masks[b.0] != live4[b.0].mask
                })
                .count();
        patterns.insert(pat);
    }
    writeln!(
        SPLIT_LOG.get().expect("--slot6-stats opens the split log").lock().unwrap(),
        "{} {} {sx} {sy} {sz} {:.2} {} {}", s.keys.len(), split.len(), changed_split as
        f64 / ns as f64, patterns.len(), ns
    )
        .unwrap();
    use std::io::Write;
    writeln!(
        log.lock().unwrap(),
        "{} {picks} {} {} {} {dz_max} {} {dx} {} {} {} {} {:.2} {:.2} {} {}", s.keys
        .len(), sigs.len(), sigs_f5.len(), ctxs.len(), s.live3.len(), variants, dxy.0,
        dxy.1, dz2, dz_sum as f64 / ns as f64, sum_changed as f64 / ns as f64,
        sum_changed_var, var_sigs.len()
    )
        .unwrap();
}
impl ChangeStats {
    fn record(&self, level: usize, changed: u64, classes: usize) {
        let n = (changed.count_ones() as usize).min(16);
        self.picks[level][n].fetch_add(1, Ordering::Relaxed);
        self.classes[level][n].fetch_add(classes as u64, Ordering::Relaxed);
    }
    fn situation(&self, level: usize, sets: usize) {
        self.distinct[level].fetch_add(sets as u64, Ordering::Relaxed);
        self.situations[level].fetch_add(1, Ordering::Relaxed);
    }
    fn report(&self) {
        for (level, name) in ["fourth", "fifth"].iter().enumerate() {
            let classes: Vec<u64> = self
                .classes[level]
                .iter()
                .map(|a| a.load(Ordering::Relaxed))
                .collect();
            let picks: Vec<u64> = self
                .picks[level]
                .iter()
                .map(|a| a.load(Ordering::Relaxed))
                .collect();
            let (tc, tp) = (
                classes.iter().sum::<u64>().max(1) as f64,
                picks.iter().sum::<u64>().max(1) as f64,
            );
            let hist: Vec<String> = (0..17)
                .filter(|&n| picks[n] > 0)
                .map(|n| {
                    format!(
                        "{n}: {:.1}% ({:.1}%)", 100.0 * classes[n] as f64 / tc, 100.0 *
                        picks[n] as f64 / tp
                    )
                })
                .collect();
            println!(
                "{name} picks, bases changed — share of class evaluations (of picks): {}; {:.1} distinct changed sets a situation",
                hist.join(", "), self.distinct[level].load(Ordering::Relaxed) as f64 /
                self.situations[level].load(Ordering::Relaxed).max(1) as f64
            );
        }
    }
}
/// [`slot5_item`] over every (label group, third pick) of `level`, sorted
/// by its label-free key; the heaviest items first. Returns each group's
/// thread-seconds.
fn slot5_groups<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    level: &[(Key, Deco<N>)],
    threads: usize,
    pb: &ProgressBar,
    table5: &Sharded<Ctx, KDeco<N>>,
    weights: &Mutex<Weights<N>>,
    grid: bool,
    profile: bool,
) -> Vec<f64> {
    let groups = label_groups(level);
    let mut sc = Scratch::default();
    let mut items: Vec<(u32, u16)> = Vec::new();
    for (g, &(lo, _)) in groups.iter().enumerate() {
        let mut rep = level[lo as usize].0;
        rep.dead = [0; 2];
        options(t, &rep, &Ctx::of_key(t, &rep), &Avail::of_key(t, &rep), &mut sc);
        items.extend((0..sc.opts.len()).map(|i| (g as u32, i as u16)));
    }
    items
        .sort_by_key(|&(g, _)| std::cmp::Reverse(
            groups[g as usize].1 - groups[g as usize].0,
        ));
    pb.set_length(items.len() as u64);
    let nanos: Vec<AtomicU64> = groups.iter().map(|_| AtomicU64::new(0)).collect();
    parallel::<
        _,
        N,
    >(
        &items,
        threads,
        pb,
        weights,
        |&(g, o3), l| {
            let t0 = Instant::now();
            let (lo, hi) = groups[g as usize];
            slot5_item(
                t,
                ar,
                &level[lo as usize..hi as usize],
                usize::from(o3),
                l,
                table5,
                grid,
                profile,
            );
            nanos[g as usize]
                .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        },
    );
    nanos.iter().map(|n| n.load(Ordering::Relaxed) as f64 * 1e-9).collect()
}
/// `--count-six K`: over every K-th item of the full pass (as `--time-six`),
/// the fourth picks and their rows: classes, and the rank-one terms that
/// could stand for them — each sum times K estimates the pass's.
fn count_six<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    level: &[(Key, Deco<N>)],
    every: usize,
    threads: usize,
) {
    let groups = label_groups(level);
    let mut sc = Scratch::default();
    let mut items: Vec<(u32, u16)> = Vec::new();
    for (g, &(lo, _)) in groups.iter().enumerate() {
        let mut rep = level[lo as usize].0;
        rep.dead = [0; 2];
        options(t, &rep, &Ctx::of_key(t, &rep), &Avail::of_key(t, &rep), &mut sc);
        items.extend((0..sc.opts.len()).map(|i| (g as u32, i as u16)));
    }
    items
        .sort_by_key(|&(g, _)| std::cmp::Reverse(
            groups[g as usize].1 - groups[g as usize].0,
        ));
    let items: Vec<(u32, u16)> = items
        .into_iter()
        .skip(every / 2)
        .step_by(every)
        .collect();
    let sums: Vec<[AtomicU64; 5]> = (0..7)
        .map(|_| std::array::from_fn(|_| AtomicU64::new(0)))
        .collect();
    let cursor = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope
                .spawn(|| {
                    let mut quad = Quad::default();
                    let (mut sc, mut sc3, mut sc4) = (
                        Scratch::default(),
                        Scratch::default(),
                        Scratch::default(),
                    );
                    loop {
                        let i = cursor.fetch_add(1, Ordering::Relaxed);
                        if i >= items.len() {
                            break;
                        }
                        let (g, o3) = items[i];
                        let (lo, hi) = groups[g as usize];
                        let group = &level[lo as usize..hi as usize];
                        let dec = ((hi - lo) as f64).log10().floor() as usize;
                        let mut rep = group[0].0;
                        rep.dead = [0; 2];
                        let ctx2 = Ctx::of_key(t, &rep);
                        let av2 = Avail::of_key(t, &rep);
                        let total2 = options(t, &rep, &ctx2, &av2, &mut sc);
                        let mut mask2 = [0u32; 64];
                        for l in &sc.live {
                            mask2[usize::from(l.mi)] = l.mask;
                        }
                        let o = sc.opts[usize::from(o3)];
                        let (class3, picked3, lead_pick3, base3) = pick_parts(
                            t,
                            &rep,
                            o.pick,
                        );
                        let flagged3 = matches!(o.pick, Pick::Forme(_))
                            && t.sp[picked3 as usize].flags;
                        let paths = cp_paths(t, ar, &rep, &mut Scratch::default());
                        let mut split = Vec::new();
                        let mut outcomes3: Vec<(bool, u8, Vec<(usize, Deco<N>)>)> = Vec::new();
                        for (v, (_, deco)) in group.iter().enumerate() {
                            split.clear();
                            if flagged3 {
                                kernel_split(
                                    t,
                                    ar,
                                    &rep,
                                    picked3,
                                    lead_pick3,
                                    deco,
                                    &mut split,
                                );
                            } else {
                                split.push((rep.tb, rep.fire, deco.clone()));
                            }
                            for (tb, fire, d) in split.drain(..) {
                                match outcomes3
                                    .iter_mut()
                                    .find(|e| e.0 == tb && e.1 == fire)
                                {
                                    Some(e) => e.2.push((v, d)),
                                    None => outcomes3.push((tb, fire, vec![(v, d)])),
                                }
                            }
                        }
                        for (tb3, fire3, variants) in outcomes3 {
                            let mut c3 = rep.with(class3);
                            c3.lead = rep.lead || lead_pick3;
                            c3.tb = tb3;
                            c3.fire = fire3;
                            let s = Situation::new(
                                t,
                                c3,
                                mask2,
                                group,
                                &variants,
                                &mut sc3,
                            );
                            let terms = cp_terms(
                                    t,
                                    ar,
                                    &rep,
                                    &paths,
                                    (picked3, lead_pick3, flagged3, base3, o.num, o.den),
                                    (tb3, fire3),
                                    &s.live3,
                                    total2,
                                    &mut quad,
                                )
                                .len();
                            let mut picks = 0u64;
                            let mut pcl = 0u64;
                            for o4 in &sc3.opts {
                                let (class4, picked4, lead_pick4, base4) = pick_parts(
                                    t,
                                    &s.c3,
                                    o4.pick,
                                );
                                if !(s.c3.lead || lead_pick4) {
                                    continue;
                                }
                                let mut c4 = s.c3.with(class4);
                                c4.lead = true;
                                let flagged4 = matches!(o4.pick, Pick::Forme(_))
                                    && t.sp[picked4 as usize].flags;
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
                                    &mut sc4,
                                ) else { continue };
                                picks += 1;
                                let mut seen: std::collections::HashSet<Vec<u8>> = Default::default();
                                for c in 0..s.keys.len() {
                                    seen.insert(
                                        p
                                            .live4
                                            .iter()
                                            .map(|l| s.r(c, usize::from(s.at3_of[usize::from(l.mi)])))
                                            .collect(),
                                    );
                                }
                                pcl += seen.len() as u64;
                            }
                            let classes = s.keys.len() as u64;
                            let e = &sums[dec.min(6)];
                            e[0].fetch_add(picks, Ordering::Relaxed);
                            e[1].fetch_add(picks * classes, Ordering::Relaxed);
                            e[2].fetch_add(picks * terms as u64, Ordering::Relaxed);
                            e[3]
                                .fetch_add(
                                    picks * classes.min(terms as u64),
                                    Ordering::Relaxed,
                                );
                            e[4].fetch_add(pcl, Ordering::Relaxed);
                        }
                    }
                });
        }
    });
    let k = every as f64;
    for (dec, e) in sums.iter().enumerate() {
        let v: Vec<f64> = e
            .iter()
            .map(|a| a.load(Ordering::Relaxed) as f64 * k)
            .collect();
        if v[0] > 0.0 {
            println!(
                "groups of 1e{dec}+ states: {:.3e} picks; rows a pick: classes {:.1}, terms {:.1}, min {:.1}, classes alike on size-4 live bases {:.1}",
                v[0], v[1] / v[0], v[2] / v[0], v[3] / v[0], v[4] / v[0]
            );
        }
    }
}
/// `--time-six K`: [`six::item`] on every K-th (label group, third pick)
/// item of the full pass, the items in size order so the sample is
/// stratified; the thread-seconds times K estimate the pass.
fn time_six<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    level: &[(Key, Deco<N>)],
    every: usize,
    threads: usize,
    mp: &MultiProgress,
) {
    let bits = six::Bits::new(t);
    let groups = label_groups(level);
    let mut sc = Scratch::default();
    let mut items: Vec<(u32, u16)> = Vec::new();
    for (g, &(lo, _)) in groups.iter().enumerate() {
        let mut rep = level[lo as usize].0;
        rep.dead = [0; 2];
        options(t, &rep, &Ctx::of_key(t, &rep), &Avail::of_key(t, &rep), &mut sc);
        items.extend((0..sc.opts.len()).map(|i| (g as u32, i as u16)));
    }
    let total = items.len();
    items
        .sort_by_key(|&(g, _)| std::cmp::Reverse(
            groups[g as usize].1 - groups[g as usize].0,
        ));
    let block = std::env::var("TG_SIX_BLOCK")
        .ok()
        .map_or(1, |v| v.parse::<usize>().unwrap().max(1));
    let mut items: Vec<(u32, u16)> = items
        .chunks(block)
        .skip(every / 2)
        .step_by(every)
        .flatten()
        .copied()
        .collect();
    if let Some(v) = std::env::var_os("TG_SIX_SIZES") {
        let v = v.into_string().unwrap();
        let (lo, hi) = v.split_once(',').unwrap();
        let (lo, hi): (u32, u32) = (lo.parse().unwrap(), hi.parse().unwrap());
        items
            .retain(|&(g, _)| {
                (lo..=hi).contains(&(groups[g as usize].1 - groups[g as usize].0))
            });
    }
    let table6: Sharded<Ctx, KDeco<N>> = Sharded::new(4096);
    let w = Mutex::new(Weights::<N>::default());
    let size = |&(g, _): &(u32, u16)| u64::from(
        groups[g as usize].1 - groups[g as usize].0,
    );
    if std::env::var_os("TG_TIME_WARM").is_some()
        && std::env::var_os("TG_TIME_CHECK").is_none()
    {
        let pb = bar(mp, items.len() as u64, "warm six");
        parallel_weighted::<
            _,
            N,
        >(
            &items,
            threads,
            &pb,
            &w,
            size,
            None,
            |&(g, o3), l| {
                let (lo, hi) = groups[g as usize];
                six::item(
                    t,
                    ar,
                    &bits,
                    &level[lo as usize..hi as usize],
                    usize::from(o3),
                    l,
                    &table6,
                );
            },
            |l| six::finish(t, &bits, l, &table6),
        );
        for a in &six::SECTIONS {
            a.store(0, Ordering::Relaxed);
        }
    }
    let pb = bar(mp, items.len() as u64, "time six");
    six::open_pick_log();
    let nanos = AtomicU64::new(0);
    let t0 = Instant::now();
    let stop = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|scope| {
        if std::env::var_os("TG_TIMELINE").is_some() {
            scope.spawn(|| timeline(threads, &stop));
        }
        parallel_weighted::<
            _,
            N,
        >(
            &items,
            threads,
            &pb,
            &w,
            size,
            None,
            |&(g, o3), l| {
                let t1 = Instant::now();
                let (lo, hi) = groups[g as usize];
                six::item(
                    t,
                    ar,
                    &bits,
                    &level[lo as usize..hi as usize],
                    usize::from(o3),
                    l,
                    &table6,
                );
                nanos.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);
            },
            |l| {
                let t1 = Instant::now();
                six::finish(t, &bits, l, &table6);
                nanos.fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);
            },
        );
        stop.store(true, Ordering::Relaxed);
    });
    let wall = t0.elapsed().as_secs_f64();
    let spent = nanos.load(Ordering::Relaxed) as f64 * 1e-9;
    if let Some(mode) = std::env::var_os("TG_TIME_CHECK") {
        println!("time-six: {} items in {wall:.0}s", items.len());
        six::table_pass6::<N>(t, ar, &bits, table6, threads, &w);
        let (wb, tb) = (Mutex::new(Weights::<N>::default()), Sharded::new(4096));
        let other_f = mode == "f";
        if other_f {
            six::OLD_F.fetch_xor(true, Ordering::Relaxed);
        }
        #[cfg(feature = "gpu")]
        let gpu = GPU.swap(other_f && GPU.load(Ordering::Relaxed), Ordering::Relaxed);
        let t1 = Instant::now();
        parallel_weighted::<
            _,
            N,
        >(
            &items,
            threads,
            &bar(mp, items.len() as u64, "check six"),
            &wb,
            size,
            None,
            |&(g, o3), l| {
                let (lo, hi) = groups[g as usize];
                six::item(
                    t,
                    ar,
                    &bits,
                    &level[lo as usize..hi as usize],
                    usize::from(o3),
                    l,
                    &tb,
                );
            },
            |l| six::finish(t, &bits, l, &tb),
        );
        #[cfg(feature = "gpu")] GPU.store(gpu, Ordering::Relaxed);
        six::table_pass6::<N>(t, ar, &bits, tb, threads, &wb);
        println!(
            "time-six check, {} in {:.0}s: {}", if other_f { "the other F" } else {
            "the CPU's path" }, t1.elapsed().as_secs_f64(), compare_slot::< N > (t, & w
            .into_inner().unwrap(), & wb.into_inner().unwrap(), 6)
        );
        return;
    }
    if let Ok(v) = std::env::var("TG_TIME_TABLE") {
        let n = table6.len();
        let entries: usize = table6
            .shards
            .iter()
            .map(|s| s.lock().unwrap().values().map(Vec::len).sum::<usize>())
            .sum();
        println!("time-six table: {n} signatures, {entries} (signature, code) entries");
        if v == "6" {
            let wb = Mutex::new(Weights::<N>::default());
            let t1 = Instant::now();
            six::table_pass6(t, ar, &bits, table6, threads, &wb);
            println!("time-six table pass by masks {:.1}s", t1.elapsed().as_secs_f64());
            return;
        }
        if v == "2" {
            let sigs = table6.into_vec();
            let (ta, tb): (Sharded<Ctx, KDeco<N>>, Sharded<Ctx, KDeco<N>>) = (
                Sharded::new(4096),
                Sharded::new(4096),
            );
            for (sig, u) in sigs {
                ta.shard(fxhash(&sig)).lock().unwrap().insert(sig, u.clone());
                tb.shard(fxhash(&sig)).lock().unwrap().insert(sig, u);
            }
            let (wa, wb) = (
                Mutex::new(Weights::<N>::default()),
                Mutex::new(Weights::<N>::default()),
            );
            let t1 = Instant::now();
            table_pass(t, ar, ta, threads, &bar(mp, 0, "table"), &wa);
            let t2 = Instant::now();
            six::table_pass6(t, ar, &bits, tb, threads, &wb);
            println!(
                "time-six table pass {:.1}s, by masks {:.1}s: {}", (t2 - t1)
                .as_secs_f64(), t2.elapsed().as_secs_f64(), compare_slot::< N > (t, & wa
                .into_inner().unwrap(), & wb.into_inner().unwrap(), 6)
            );
        }
    }
    six::close_pick_log();
    six::report();
    println!(
        "time-six: {} of {total} items in {wall:.0}s ({spent:.0} thread-s); the full pass: {:.0} thread-s, {:.1} h at this run's {:.1} threads' worth",
        items.len(), spent * every as f64, spent * every as f64 / (spent / wall) /
        3600.0, spent / wall
    );
}
/// [`six::item`] over every (label group, third pick) of `level`, in
/// `ck.epochs` stratified epochs — epoch e the e-th of every epochs blocks of
/// 64 items in the heavy-first order, so each costs about the same and its
/// signature table fits in memory. An epoch's signatures are credited when
/// it ends (the whole pass's would not fit), and the weights checkpointed;
/// the pass starts after `ck.done` epochs. Returns each group's
/// thread-seconds.
fn six_epochs<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    level: &[(Key, Deco<N>)],
    threads: usize,
    mp: &MultiProgress,
    weights: &Mutex<Weights<N>>,
    ck: &mut Ckpt,
) -> Vec<f64> {
    let bits = six::Bits::new(t);
    let groups = label_groups(level);
    let mut sc = Scratch::default();
    let mut items: Vec<(u32, u16)> = Vec::new();
    for (g, &(lo, _)) in groups.iter().enumerate() {
        let mut rep = level[lo as usize].0;
        rep.dead = [0; 2];
        options(t, &rep, &Ctx::of_key(t, &rep), &Avail::of_key(t, &rep), &mut sc);
        items.extend((0..sc.opts.len()).map(|i| (g as u32, i as u16)));
    }
    items
        .sort_by_key(|&(g, _)| std::cmp::Reverse(
            groups[g as usize].1 - groups[g as usize].0,
        ));
    let size = |&(g, _): &(u32, u16)| u64::from(
        groups[g as usize].1 - groups[g as usize].0,
    );
    let blocks = items.len().div_ceil(64);
    let epochs = (ck.epochs as usize).min(blocks.max(1));
    ck.epochs = epochs as u32;
    let epoch_items = |e: usize| -> Vec<(u32, u16)> {
        (e..blocks)
            .step_by(epochs)
            .flat_map(|b| items[b * 64..((b + 1) * 64).min(items.len())].iter().copied())
            .collect()
    };
    let nanos: Vec<AtomicU64> = groups.iter().map(|_| AtomicU64::new(0)).collect();
    let progress = Progress::new(mp, "slot 6", epochs, ck.done as usize);
    let cur_done = AtomicU64::new(0);
    let cur_total = AtomicU64::new(1);
    let finished = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| progress.watch(&cur_done, &cur_total, &finished));
        let mut writing: Option<std::thread::ScopedJoinHandle<()>> = None;
        for e in ck.done as usize..epochs {
            let t0 = Instant::now();
            let list = epoch_items(e);
            cur_done.store(0, Ordering::Relaxed);
            cur_total
                .store(list.iter().map(size).sum::<u64>().max(1), Ordering::Relaxed);
            let table6: Sharded<Ctx, KDeco<N>> = Sharded::new(16384);
            parallel_weighted::<
                _,
                N,
            >(
                &list,
                threads,
                &ProgressBar::hidden(),
                weights,
                size,
                Some(&cur_done),
                |&(g, o3), l| {
                    let t1 = Instant::now();
                    let (lo, hi) = groups[g as usize];
                    six::item(
                        t,
                        ar,
                        &bits,
                        &level[lo as usize..hi as usize],
                        usize::from(o3),
                        l,
                        &table6,
                    );
                    nanos[g as usize]
                        .fetch_add(t1.elapsed().as_nanos() as u64, Ordering::Relaxed);
                },
                |l| six::finish(t, &bits, l, &table6),
            );
            let sigs = table6.len();
            let t1 = Instant::now();
            six::table_pass6::<N>(t, ar, &bits, table6, threads, weights);
            let credit = t1.elapsed().as_secs_f64();
            ck.done = e as u32 + 1;
            let t2 = Instant::now();
            if let Some(h) = writing.take() {
                h.join().unwrap();
            }
            if let Some((path, bytes)) = ck.snapshot::<N>(weights) {
                writing = Some(scope.spawn(move || Ckpt::write(&path, &bytes)));
            }
            progress.epoch_done(e + 1);
            progress
                .println(
                    &format!(
                        "slot 6 epoch {}/{epochs}: {} items in {:.0}s, {sigs} signatures credited in {credit:.1}s, checkpoint {:.1}s",
                        e + 1, list.len(), t0.elapsed().as_secs_f64(), t2.elapsed()
                        .as_secs_f64()
                    ),
                );
        }
        if let Some(h) = writing.take() {
            h.join().unwrap();
        }
        finished.store(true, Ordering::Relaxed);
    });
    progress.finish();
    six::report();
    nanos.iter().map(|n| n.load(Ordering::Relaxed) as f64 * 1e-9).collect()
}
/// A long pass's progress in stages of about equal cost (epochs): a bar on a
/// terminal, a line on the log, and the ETA from the rate over the last 30
/// minutes — the run's early hours, or a pause, say little about its end.
struct Progress {
    pb: ProgressBar,
    name: String,
    stages: usize,
    done: AtomicUsize,
    /// (seconds since the start, fraction done), the window's samples.
    samples: Mutex<std::collections::VecDeque<(f64, f64)>>,
    started: Instant,
}
impl Progress {
    /// The window the rate is measured over.
    const WINDOW: f64 = 1800.0;
    fn new(mp: &MultiProgress, name: &str, stages: usize, done: usize) -> Progress {
        let pb = mp.add(ProgressBar::new(1_000_000));
        pb.set_style(
            ProgressStyle::with_template(
                    "{prefix:>18} [{elapsed_precise}] {bar:36.cyan/blue} {msg}",
                )
                .unwrap()
                .progress_chars("=> "),
        );
        pb.set_prefix(name.to_string());
        Progress {
            pb,
            name: name.into(),
            stages,
            done: AtomicUsize::new(done),
            samples: Mutex::new(Default::default()),
            started: Instant::now(),
        }
    }
    fn epoch_done(&self, done: usize) {
        self.done.store(done, Ordering::Relaxed);
    }
    /// A line on the log (stdout), above the bar.
    fn println(&self, line: &str) {
        self.pb.suspend(|| println!("{line}"));
    }
    /// The fraction done, the current stage `cur` of `total` weight in.
    fn fraction(&self, cur: f64) -> f64 {
        (self.done.load(Ordering::Relaxed) as f64 + cur.min(1.0)) / self.stages as f64
    }
    /// Ticks until `finished`: the bar every two seconds, a status line on
    /// the log every ten minutes, and on stderr every minute when there is
    /// no terminal to draw the bar on.
    fn watch(
        &self,
        cur_done: &AtomicU64,
        cur_total: &AtomicU64,
        finished: &std::sync::atomic::AtomicBool,
    ) {
        let (mut last_log, mut last_err) = (0.0f64, 0.0f64);
        while !finished.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_secs(2));
            let now = self.started.elapsed().as_secs_f64();
            let frac = self
                .fraction(
                    cur_done.load(Ordering::Relaxed) as f64
                        / cur_total.load(Ordering::Relaxed) as f64,
                );
            let status = self.status(now, frac);
            self.pb.set_position((frac * 1e6) as u64);
            self.pb.set_message(status.clone());
            if now - last_log >= 600.0 {
                last_log = now;
                self.println(&format!("{}: {status}", self.name));
            } else if self.pb.is_hidden() && now - last_err >= 60.0 {
                last_err = now;
                eprintln!("{}: {status}", self.name);
            }
        }
    }
    /// "37.42% (epoch 383/1024) · 6.10%/h over the last 30 min · ETA 10h 16m (Thu 08:45)".
    fn status(&self, now: f64, frac: f64) -> String {
        let mut s = self.samples.lock().unwrap();
        s.push_back((now, frac));
        while s.len() > 2 && s[1].0 <= now - Self::WINDOW {
            s.pop_front();
        }
        let (t0, f0) = s[0];
        let window = now - t0;
        let rate = if window > 0.0 { (frac - f0) / window } else { 0.0 };
        let stage = (self.done.load(Ordering::Relaxed) + 1).min(self.stages);
        let mut line = format!("{:.2}% (epoch {stage}/{})", 100.0 * frac, self.stages);
        if rate > 0.0 && window >= 60.0 {
            let eta = (1.0 - frac) / rate;
            line
                += &format!(
                    " · {:.2}%/h over the last {:.0} min · ETA {}", 100.0 * rate *
                    3600.0, window / 60.0, hms(eta)
                );
            if let Some(at) = clock_in(eta) {
                line += &format!(" ({at})");
            }
        }
        line
    }
    fn finish(&self) {
        self.pb
            .finish_with_message(
                format!("done in {}", hms(self.started.elapsed().as_secs_f64())),
            );
    }
}
/// "2d 14h 03m".
fn hms(secs: f64) -> String {
    let m = (secs / 60.0).round() as u64;
    let (d, h, m) = (m / 1440, m / 60 % 24, m % 60);
    if d > 0 { format!("{d}d {h}h {m:02}m") } else { format!("{h}h {m:02}m") }
}
/// The local wall-clock time `secs` from now, as `date` prints it; cached a
/// minute, as it is asked for every tick.
fn clock_in(secs: f64) -> Option<String> {
    static LAST: Mutex<Option<(Instant, String)>> = Mutex::new(None);
    let mut last = LAST.lock().unwrap();
    if let Some((at, s)) = last.as_ref().filter(|(at, _)| at.elapsed().as_secs() < 60) {
        let _ = at;
        return Some(s.clone());
    }
    let when = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs() + secs as u64;
    let out = std::process::Command::new("date")
        .arg(format!("-d@{when}"))
        .arg("+%a %H:%M")
        .output()
        .ok()?;
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    *last = Some((Instant::now(), s.clone()));
    Some(s)
}
/// A long run's resume point, `<out>.ckpt`: the weights credited so far and
/// how far they reach — every pass before slot 6's (`stage` 1), then slot
/// 6's epochs in order (`done`). Written whole after each, to a temporary
/// file renamed over the last (kept as `.prev`); a rerun of the same
/// command loads it and picks up after it. `fingerprint` is the arguments
/// that decide the weights: a checkpoint of other ones is refused.
struct Ckpt {
    path: Option<std::path::PathBuf>,
    fingerprint: String,
    epochs: u32,
    stage: u8,
    done: u32,
}
const CKPT_MAGIC: &[u8; 8] = b"TGCKPT01";
impl Ckpt {
    /// `TG_SIX_EPOCHS`: the slot-6 pass's epochs (default 1024).
    fn new(args: &Args) -> Ckpt {
        let epochs = std::env::var("TG_SIX_EPOCHS")
            .ok()
            .map_or(1024, |v| v.parse().unwrap());
        let fingerprint = format!(
            "slot5={} slot6={} limit={} limit_groups={} sample_groups={} grid={} no_profile={} epochs={epochs}",
            args.slot5, args.slot6, args.limit, args.limit_groups, args.sample_groups,
            args.grid, args.no_profile
        );
        let path = (!args.out.is_empty()).then(|| format!("{}.ckpt", args.out).into());
        Ckpt {
            path,
            fingerprint,
            epochs,
            stage: 0,
            done: 0,
        }
    }
    /// The checkpoint at `path`, if any, into `weights`: floats only. A
    /// damaged or missing one falls back to the one before (`.ckpt.prev`);
    /// one of other arguments is refused.
    fn load<N: Num>(&mut self, weights: &Mutex<Weights<N>>) {
        let Some(path) = self.path.clone() else { return };
        let prev = path.with_extension("ckpt.prev");
        let mut guard = weights.lock().unwrap();
        let Some(w) = (&mut *guard as &mut dyn std::any::Any)
            .downcast_mut::<Weights<f64>>() else {
            if path.exists() {
                println!(
                    "checkpoint {}: not loaded, the exact walk does not checkpoint", path
                    .display()
                );
            }
            return;
        };
        let mut seen = false;
        for at in [&path, &prev] {
            if !at.exists() {
                continue;
            }
            seen = true;
            match self.read(at) {
                Ok((stage, epochs, done, entries)) => {
                    w.clear();
                    w.extend(entries);
                    (self.stage, self.epochs, self.done) = (stage, epochs, done);
                    println!(
                        "checkpoint {}: resuming at stage {stage}, slot 6 epoch {done}/{epochs}, {} weights",
                        at.display(), w.len()
                    );
                    return;
                }
                Err(e) => {
                    let bad = at
                        .with_extension(
                            if *at == path { "ckpt.bad" } else { "ckpt.prev.bad" },
                        );
                    let _ = std::fs::rename(at, &bad);
                    println!(
                        "checkpoint {}: unreadable ({e}), moved to {}; trying the one before",
                        at.display(), bad.display()
                    );
                }
            }
        }
        if seen {
            println!(
                "checkpoint {}: NO READABLE CHECKPOINT, starting over", path.display()
            );
        }
    }
    /// A checkpoint file's (stage, epochs, epochs done, weights), checked
    /// for length; a fingerprint of other arguments panics.
    #[allow(clippy::type_complexity)]
    fn read(
        &self,
        path: &std::path::Path,
    ) -> Result<(u8, u32, u32, Vec<((u16, u8, bool, u16), Sum)>), String> {
        let buf = std::fs::read(path).map_err(|e| e.to_string())?;
        let mut at = 0usize;
        let mut take = |n: usize| -> Result<&[u8], String> {
            let s = buf
                .get(at..at + n)
                .ok_or_else(|| format!("truncated at byte {at} of {}", buf.len()))?;
            at += n;
            Ok(s)
        };
        if take(8)? != CKPT_MAGIC {
            return Err("not a checkpoint".into());
        }
        let fl = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
        let fp = String::from_utf8(take(fl)?.to_vec()).map_err(|e| e.to_string())?;
        assert!(
            fp == self.fingerprint,
            "{}: a checkpoint of other arguments ({fp}, these are {}); move it aside to start over",
            path.display(), self.fingerprint
        );
        let stage = take(1)?[0];
        let epochs = u32::from_le_bytes(take(4)?.try_into().unwrap());
        let done = u32::from_le_bytes(take(4)?.try_into().unwrap());
        let n = u64::from_le_bytes(take(8)?.try_into().unwrap()) as usize;
        let mut entries = Vec::with_capacity(n.min(1 << 26));
        for _ in 0..n {
            let e = take(22)?;
            let key = (
                u16::from_le_bytes([e[0], e[1]]),
                e[2],
                e[3] != 0,
                u16::from_le_bytes([e[4], e[5]]),
            );
            let s = f64::from_le_bytes(e[6..14].try_into().unwrap());
            let c = f64::from_le_bytes(e[14..22].try_into().unwrap());
            entries.push((key, Sum { s, c }));
        }
        if at != buf.len() {
            return Err(format!("{} bytes past the weights", buf.len() - at));
        }
        Ok((stage, epochs, done, entries))
    }
    /// The weights and this point, so that a crash or a power cut at any
    /// moment leaves a whole checkpoint on disk: written and synced aside,
    /// the last one kept by a hard link as `.ckpt.prev`, the new one renamed
    /// over it (one atomic step), the directory synced.
    fn save<N: Num>(&self, weights: &Mutex<Weights<N>>) {
        if let Some((path, bytes)) = self.snapshot::<N>(weights) {
            Ckpt::write(&path, &bytes);
        }
    }
    /// The checkpoint's bytes and path, taken under the weights' lock; what
    /// [`Ckpt::write`] puts on disk.
    fn snapshot<N: Num>(
        &self,
        weights: &Mutex<Weights<N>>,
    ) -> Option<(std::path::PathBuf, Vec<u8>)> {
        let path = self.path.clone()?;
        let w = weights.lock().unwrap();
        let w = (&*w as &dyn std::any::Any).downcast_ref::<Weights<f64>>()?;
        let mut out = Vec::with_capacity(64 + self.fingerprint.len() + 32 * w.len());
        out.extend_from_slice(CKPT_MAGIC);
        out.extend_from_slice(&(self.fingerprint.len() as u32).to_le_bytes());
        out.extend_from_slice(self.fingerprint.as_bytes());
        out.push(self.stage);
        out.extend_from_slice(&self.epochs.to_le_bytes());
        out.extend_from_slice(&self.done.to_le_bytes());
        out.extend_from_slice(&(w.len() as u64).to_le_bytes());
        for (&(s, slot, lead, code), x) in w {
            out.extend_from_slice(&s.to_le_bytes());
            out.extend_from_slice(&[slot, u8::from(lead)]);
            out.extend_from_slice(&code.to_le_bytes());
            out.extend_from_slice(&x.s.to_le_bytes());
            out.extend_from_slice(&x.c.to_le_bytes());
        }
        Some((path, out))
    }
    /// A snapshot onto disk at `path`, as [`Ckpt::save`] describes.
    fn write(path: &std::path::Path, bytes: &[u8]) {
        use std::io::Write;
        let tmp = path.with_extension("ckpt.tmp");
        let mut out = std::fs::File::create(&tmp).unwrap();
        out.write_all(bytes).unwrap();
        out.sync_all().unwrap();
        if path.exists() {
            let link = path.with_extension("ckpt.prev.tmp");
            let _ = std::fs::remove_file(&link);
            std::fs::hard_link(path, &link).unwrap();
            std::fs::rename(&link, path.with_extension("ckpt.prev")).unwrap();
        }
        std::fs::rename(&tmp, path).unwrap();
        let dir = path
            .parent()
            .filter(|d| !d.as_os_str().is_empty())
            .unwrap_or(std::path::Path::new("."));
        std::fs::File::open(dir).and_then(|d| d.sync_all()).unwrap();
    }
}
/// A signature table's credit: every species the signature admits takes ω
/// of its contexts, in the slot its pick lands in.
fn table_pass<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    table: Sharded<Ctx, KDeco<N>>,
    threads: usize,
    pb: &ProgressBar,
    weights: &Mutex<Weights<N>>,
) {
    let sigs = table.into_vec();
    pb.set_length(sigs.len() as u64);
    parallel::<
        _,
        N,
    >(
        &sigs,
        threads,
        pb,
        weights,
        |(sig, u), l| {
            let tb_code = if sig.tb { TB_BIT } else { 0 };
            for (i, s) in t.sp.iter().enumerate() {
                if sig.adm(s) {
                    let lead_pick = !sig.lead && !s.no_lead;
                    let slot = slot_of(sig.n, sig.lead, lead_pick);
                    for &(code, x) in u {
                        credit(
                            &mut l.w,
                            i as u16,
                            slot,
                            lead_pick,
                            code | tb_code,
                            ar.omega[i] * N::get(x),
                        );
                    }
                }
            }
        },
    );
}
/// The port's loop, draw for draw as `generate_team`, recording each
/// member's (species, party slot, lead, context); with `check`, every
/// species' fast admissibility is held to `TeamGenState::admits` before
/// every draw.
fn sampled_team(
    t: &Tables,
    idx: &FxMap<SpeciesId, u16>,
    rng: &mut StdRng,
    check: bool,
    out: &mut FxMap<(u16, u8, bool, u16), u64>,
) -> u64 {
    let mut pool = weighted_base_pool().to_vec();
    let mut state = TeamGenState::default();
    let mut members: Vec<u16> = Vec::new();
    let (mut lead_placed, mut fire) = (false, 0u8);
    let mut mismatches = 0;
    while !pool.is_empty() && members.len() < usize::from(TEAM) {
        let n = members.len();
        if check {
            let ctx = Ctx::new(
                t,
                members.iter().copied(),
                n as u8,
                lead_placed,
                state.team_details.tera_blast,
                fire,
            );
            for (i, s) in t.sp.iter().enumerate() {
                let fast = ctx.adm(s)
                    && !members.iter().any(|&m| t.sp[m as usize].base == s.base);
                let slow = state.admits(s.id, n, usize::from(TEAM))
                    && !(never_leads(s.id) && !lead_placed
                        && n + 1 == usize::from(TEAM));
                if fast != slow {
                    if mismatches < 5 {
                        eprintln!(
                            "admissibility mismatch: {} at size {n} (fast {fast}, port {slow})",
                            t.sp[i].name
                        );
                    }
                    mismatches += 1;
                }
            }
        }
        let base = pool.swap_remove(rng.random_range(0..pool.len()));
        let formes = formes_of(base);
        if formes.is_empty() {
            continue;
        }
        let species = formes[rng.random_range(0..formes.len())];
        if !state.admits(species, n, usize::from(TEAM)) {
            continue;
        }
        let no_lead = never_leads(species);
        if !lead_placed && no_lead && n + 1 == usize::from(TEAM) {
            continue;
        }
        let lead = !lead_placed && !no_lead;
        let code = code_of(&state.team_details);
        let set = random_set(rng, species, &state.team_details, lead, false);
        let s = idx[&species];
        *out.entry((s, slot_of(n as u8, lead_placed, lead), lead, code)).or_default()
            += 1;
        members.push(s);
        lead_placed |= lead;
        if members.len() == usize::from(TEAM) {
            break;
        }
        state.record(&set);
        fire
            += u8::from(
                matches!(set.ability, AbilityId::Dryskin | AbilityId::Fluffy)
                    && t.sp[s as usize].fire_neutral,
            );
    }
    mismatches
}
/// [`sampled_team`] to its first `k − 1` members, then the `k`-th pick
/// Rao–Blackwellized: instead of drawing it, its distribution given the
/// team so far and one draw of every remaining entry's forme label, added
/// to `out` at its (species, party slot, lead, context). Each remaining
/// entry is drawn at most once, so its label can be drawn in advance; the
/// pick is then uniform over the entries whose label is admissible
/// (MATH.md, Corollary 3). A team whose pool admits nothing more adds
/// nothing, as the loop ends it short.
fn sampled_rb(
    idx: &FxMap<SpeciesId, u16>,
    rng: &mut StdRng,
    k: usize,
    out: &mut FxMap<(u16, u8, bool, u16), f64>,
) {
    let team = usize::from(TEAM);
    let mut pool = weighted_base_pool().to_vec();
    let mut state = TeamGenState::default();
    let (mut members, mut lead_placed) = (0usize, false);
    let admits = |state: &TeamGenState, species: SpeciesId, n: usize, lead_placed: bool| {
        state.admits(species, n, team)
            && !(!lead_placed && never_leads(species) && n + 1 == team)
    };
    while !pool.is_empty() && members + 1 < k {
        let base = pool.swap_remove(rng.random_range(0..pool.len()));
        let formes = formes_of(base);
        if formes.is_empty() {
            continue;
        }
        let species = formes[rng.random_range(0..formes.len())];
        if !admits(&state, species, members, lead_placed) {
            continue;
        }
        let lead = !lead_placed && !never_leads(species);
        let set = random_set(rng, species, &state.team_details, lead, false);
        members += 1;
        lead_placed |= lead;
        state.record(&set);
    }
    if members + 1 < k {
        return;
    }
    let code = code_of(&state.team_details);
    let mut tally: Vec<(SpeciesId, u32)> = Vec::new();
    let mut m = 0u32;
    for &base in &pool {
        let formes = formes_of(base);
        if formes.is_empty() {
            continue;
        }
        let species = formes[rng.random_range(0..formes.len())];
        if !admits(&state, species, members, lead_placed) {
            continue;
        }
        m += 1;
        match tally.iter_mut().find(|e| e.0 == species) {
            Some(e) => e.1 += 1,
            None => tally.push((species, 1)),
        }
    }
    for (species, c) in tally {
        let lead = !lead_placed && !never_leads(species);
        *out
            .entry((
                idx[&species],
                slot_of(members as u8, lead_placed, lead),
                lead,
                code,
            ))
            .or_default() += f64::from(c) / f64::from(m);
    }
}
/// [`sampled_rb`] over `teams` teams, on the streams of `seed`.
fn sample_rb(
    t: &Tables,
    teams: u64,
    threads: usize,
    seed: u64,
    k: usize,
) -> FxMap<(u16, u8, bool, u16), f64> {
    let idx: FxMap<SpeciesId, u16> = t
        .sp
        .iter()
        .enumerate()
        .map(|(i, s)| (s.id, i as u16))
        .collect();
    let total = Mutex::new(FxMap::default());
    std::thread::scope(|scope| {
        for th in 0..threads {
            let (idx, total) = (&idx, &total);
            scope
                .spawn(move || {
                    let mut out = FxMap::default();
                    let mut rng = StdRng::seed_from_u64(
                        0x5EED_0B ^ th as u64 ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                            ^ (k as u64) << 56,
                    );
                    for _ in 0..teams / threads as u64 {
                        sampled_rb(idx, &mut rng, k, &mut out);
                    }
                    let mut g = total.lock().unwrap();
                    for (key, v) in out {
                        *g.entry(key).or_default() += v;
                    }
                });
        }
    });
    total.into_inner().unwrap()
}
/// Along `teams` sampled teams: at each size, how many multi-forme bases are
/// partly admissible, the live entries they hold, and ∏(r+1) — the factor
/// by which their states multiply a team's.
fn partial_stats(t: &Tables, teams: u64) {
    let idx: FxMap<SpeciesId, u16> = t
        .sp
        .iter()
        .enumerate()
        .map(|(i, s)| (s.id, i as u16))
        .collect();
    let mut rng = StdRng::seed_from_u64(7);
    let mut acc = [(0f64, 0f64, 0f64, 0f64, 0u32); 6];
    let mut prods: Vec<Vec<f64>> = vec![Vec::new(); 6];
    let mus: Vec<(u8, u8)> = t
        .multi
        .iter()
        .map(|&b| (t.bases[b as usize].mu, t.bases[b as usize].phi))
        .collect();
    println!(
        "{} multi-forme bases holding {} entries; μ = 1/2/3: {}/{}/{}", mus.len(), mus
        .iter().map(| m | u32::from(m.0)).sum::< u32 > (), mus.iter().filter(| m | m.0 ==
        1).count(), mus.iter().filter(| m | m.0 == 2).count(), mus.iter().filter(| m | m
        .0 == 3).count()
    );
    for _ in 0..teams {
        let mut pool = weighted_base_pool().to_vec();
        let mut state = TeamGenState::default();
        let mut members: Vec<u16> = Vec::new();
        let (mut lead_placed, mut fire) = (false, 0u8);
        let mut burned: FxMap<u16, u8> = FxMap::default();
        while !pool.is_empty() && members.len() < usize::from(TEAM) {
            let n = members.len();
            let ctx = Ctx::new(
                t,
                members.iter().copied(),
                n as u8,
                lead_placed,
                state.team_details.tera_blast,
                fire,
            );
            let (mut f, mut r, mut prod) = (0u32, 0u32, 1f64);
            for &b in &t.multi {
                if members.iter().any(|&m| t.sp[m as usize].base == b) {
                    continue;
                }
                let base = &t.bases[b as usize];
                let mask = forme_mask(t, &ctx, b);
                let live = base.mu - burned.get(&b).copied().unwrap_or(0);
                if mask != 0 && mask != (1 << base.formes.len()) - 1 && live > 0 {
                    f += 1;
                    r += u32::from(live);
                    prod *= f64::from(live + 1);
                }
            }
            let a = &mut acc[n];
            a.0 += f64::from(f);
            a.1 += f64::from(r);
            a.2 += prod;
            a.3 = a.3.max(prod);
            a.4 += 1;
            prods[n].push(prod);
            let i = rng.random_range(0..pool.len());
            let base = pool.swap_remove(i);
            let formes = formes_of(base);
            if formes.is_empty() {
                continue;
            }
            let species = formes[rng.random_range(0..formes.len())];
            let no_lead = never_leads(species);
            if !state.admits(species, n, usize::from(TEAM))
                || (!lead_placed && no_lead && n + 1 == usize::from(TEAM))
            {
                if formes.len() > 1 {
                    *burned.entry(t.sp[idx[&species] as usize].base).or_default() += 1;
                }
                continue;
            }
            let lead = !lead_placed && !no_lead;
            let set = random_set(&mut rng, species, &state.team_details, lead, false);
            let s = idx[&species];
            members.push(s);
            lead_placed |= lead;
            state.record(&set);
            fire
                += u8::from(
                    matches!(set.ability, AbilityId::Dryskin | AbilityId::Fluffy)
                        && t.sp[s as usize].fire_neutral,
                );
        }
    }
    for (n, a) in acc.iter().enumerate() {
        let c = f64::from(a.4.max(1));
        let p = &mut prods[n];
        p.sort_by(f64::total_cmp);
        let q = |x: f64| {
            p.get(((p.len() as f64 - 1.0) * x) as usize).copied().unwrap_or(0.0)
        };
        println!(
            "size {n}: partly admissible multi-forme bases {:.2}, their live entries {:.2}; ∏(r+1) mean {:.1}, \
             median {:.0}, p90 {:.0}, p99 {:.0}, max {:.0}",
            a.0 / c, a.1 / c, a.2 / c, q(0.5), q(0.9), q(0.99), a.3
        );
    }
}
fn sample(
    t: &Tables,
    teams: u64,
    check: bool,
    threads: usize,
) -> (FxMap<(u16, u8, bool, u16), u64>, u64) {
    sample_seeded(t, teams, check, threads, 0)
}
/// [`sample`] from the streams of `seed`; seed 0 is [`sample`]'s own.
fn sample_seeded(
    t: &Tables,
    teams: u64,
    check: bool,
    threads: usize,
    seed: u64,
) -> (FxMap<(u16, u8, bool, u16), u64>, u64) {
    let idx: FxMap<SpeciesId, u16> = t
        .sp
        .iter()
        .enumerate()
        .map(|(i, s)| (s.id, i as u16))
        .collect();
    let total = Mutex::new((FxMap::default(), 0u64));
    std::thread::scope(|scope| {
        for th in 0..threads {
            let (idx, total) = (&idx, &total);
            scope
                .spawn(move || {
                    let mut out = FxMap::default();
                    let mut bad = 0;
                    let mut rng = StdRng::seed_from_u64(
                        0xC0DE ^ th as u64 ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15),
                    );
                    for _ in 0..teams / threads as u64 {
                        bad += sampled_team(t, idx, &mut rng, check, &mut out);
                    }
                    let mut g = total.lock().unwrap();
                    for (k, v) in out {
                        *g.0.entry(k).or_default() += v;
                    }
                    g.1 += bad;
                });
        }
    });
    total.into_inner().unwrap()
}
struct Args {
    slots: bool,
    max_size: u8,
    exact: Option<usize>,
    mem_gb: f64,
    threads: usize,
    self_check: u64,
    check_mc: u64,
    check_analytic: usize,
    slot5: bool,
    check_analytic5: usize,
    check_analytic6: usize,
    check_groups6: usize,
    check_six: usize,
    time_six: usize,
    count_six: usize,
    slot6: bool,
    group_stats: bool,
    cp_stats: bool,
    check_cp: usize,
    sample_groups: usize,
    check_groups: bool,
    limit_groups: usize,
    check_quadrature: bool,
    grid: bool,
    no_profile: bool,
    limit: usize,
    out: String,
    partial_stats: u64,
    mc_dump: u64,
    mc_seed: u64,
    mc_rb: usize,
    gpu: bool,
}
fn parse_args() -> Args {
    let mut a = Args {
        slots: false,
        max_size: 2,
        exact: None,
        mem_gb: 8.0,
        threads: std::thread::available_parallelism().map_or(8, |n| n.get()),
        self_check: 2000,
        check_mc: 0,
        check_analytic: 0,
        slot5: false,
        check_analytic5: 0,
        check_analytic6: 0,
        check_groups6: 0,
        check_six: 0,
        time_six: 0,
        count_six: 0,
        slot6: false,
        group_stats: false,
        cp_stats: false,
        check_cp: 0,
        sample_groups: 0,
        check_groups: false,
        limit_groups: 0,
        check_quadrature: false,
        grid: false,
        no_profile: false,
        limit: 0,
        out: "exact-team-weights.json".into(),
        partial_stats: 0,
        mc_dump: 0,
        mc_seed: 0,
        mc_rb: 0,
        gpu: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--slots" => a.slots = true,
            "--max-size" => a.max_size = val().parse().unwrap(),
            "--exact" => a.exact = Some(val().parse().unwrap()),
            "--mem-gb" => a.mem_gb = val().parse().unwrap(),
            "--threads" => a.threads = val().parse().unwrap(),
            "--self-check" => a.self_check = val().parse().unwrap(),
            "--check-mc" => a.check_mc = val().parse().unwrap(),
            "--check-analytic" => a.check_analytic = val().parse().unwrap(),
            "--slot5" => a.slot5 = true,
            "--group-stats" => a.group_stats = true,
            "--cp-stats" => a.cp_stats = true,
            "--check-cp" => a.check_cp = val().parse().unwrap(),
            "--sample-groups" => a.sample_groups = val().parse().unwrap(),
            "--check-cp-rows" => CHECK_CP_ROWS.store(true, Ordering::Relaxed),
            "--check-groups" => a.check_groups = true,
            "--limit-groups" => a.limit_groups = val().parse().unwrap(),
            "--check-quadrature" => a.check_quadrature = true,
            "--grid" => a.grid = true,
            "--no-profile" => a.no_profile = true,
            "--count-changes" => COUNT_CHANGES.store(true, Ordering::Relaxed),
            "--slot6-stats" => {
                let path = val();
                let f = std::fs::File::create(&path).unwrap();
                SLOT6_STATS
                    .set(Mutex::new(std::io::BufWriter::new(f)))
                    .unwrap_or_else(|_| unreachable!());
                let split = std::fs::File::create(format!("{path}.split")).unwrap();
                SPLIT_LOG
                    .set(Mutex::new(std::io::BufWriter::new(split)))
                    .unwrap_or_else(|_| unreachable!());
            }
            "--check-analytic5" => a.check_analytic5 = val().parse().unwrap(),
            "--check-analytic6" => a.check_analytic6 = val().parse().unwrap(),
            "--check-groups6" => a.check_groups6 = val().parse().unwrap(),
            "--check-six" => a.check_six = val().parse().unwrap(),
            "--time-six" => a.time_six = val().parse().unwrap(),
            "--count-six" => a.count_six = val().parse().unwrap(),
            "--slot6" => a.slot6 = true,
            "--limit" => a.limit = val().parse().unwrap(),
            "--out" => a.out = val(),
            "--partial-stats" => a.partial_stats = val().parse().unwrap(),
            "--mc-dump" => a.mc_dump = val().parse().unwrap(),
            "--mc-seed" => a.mc_seed = val().parse().unwrap(),
            "--mc-rb" => a.mc_rb = val().parse().unwrap(),
            "--gpu" => a.gpu = true,
            other => panic!("unknown flag {other}"),
        }
    }
    assert!(
        (1..= 3).contains(& a.max_size),
        "--max-size takes 1..=3: the states past it do not fit"
    );
    a
}
fn bar(mp: &MultiProgress, len: u64, prefix: &str) -> ProgressBar {
    let pb = mp.add(ProgressBar::new(len));
    pb.set_style(
        ProgressStyle::with_template(
                "{prefix:>18} [{elapsed_precise}] {bar:36.cyan/blue} {human_pos}/{human_len} {per_sec:>12} eta {eta_precise} {msg}",
            )
            .unwrap()
            .progress_chars("=> "),
    );
    pb.set_prefix(prefix.to_string());
    pb
}
/// A thread's own weights and scratch: the walk's, and a fold's beside it.
struct Local<N: Num> {
    w: Weights<N>,
    fold: Weights<N>,
    sc: Scratch,
    fold_sc: Scratch,
    /// The slot-5 fold's: the size-3 walk's, the size-4 options', and its
    /// polynomials.
    sc3: Scratch,
    sc4: Scratch,
    polys: PolyBuf<N>,
    quad: Quad,
    six: six::Six,
    #[cfg(feature = "gpu")]
    batch: gpu::Batch,
    #[cfg(feature = "gpu")]
    sixq: six::Queue,
}
impl<N: Num> Default for Local<N> {
    fn default() -> Self {
        Local {
            w: Weights::<N>::default(),
            fold: Weights::<N>::default(),
            sc: Scratch::default(),
            fold_sc: Scratch::default(),
            sc3: Scratch::default(),
            sc4: Scratch::default(),
            polys: PolyBuf::default(),
            quad: Quad::default(),
            six: six::Six::default(),
            #[cfg(feature = "gpu")]
            batch: gpu::Batch::default(),
            #[cfg(feature = "gpu")]
            sixq: six::Queue::default(),
        }
    }
}
/// `<out>.pause`: while it exists, [`parallel`]'s workers idle.
static PAUSE: OnceLock<std::path::PathBuf> = OnceLock::new();
/// Run `f` over `items` on `threads` threads in chunks, each thread with
/// its own [`Local`], whose weights merge into `w` at the end.
fn parallel<T: Sync, N: Num>(
    items: &[T],
    threads: usize,
    pb: &ProgressBar,
    w: &Mutex<Weights<N>>,
    f: impl Fn(&T, &mut Local<N>) + Sync,
) {
    static CHUNK: OnceLock<Option<usize>> = OnceLock::new();
    let chunk = CHUNK
        .get_or_init(|| std::env::var("TG_CHUNK").ok().map(|v| v.parse().unwrap()))
        .unwrap_or_else(|| (items.len() / (threads * 16)).clamp(1, 64));
    let bounds: Vec<usize> = (0..items.len())
        .step_by(chunk)
        .chain([items.len()])
        .collect();
    parallel_chunks(items, &bounds, threads, Ends::InOrder, pb, w, f, &|_| {});
}
/// [`parallel`] over items of very unequal cost, `weight` an item's guess:
/// a chunk takes up to 64 items and about 1/(64 · threads) of the total
/// weight, so the heaviest items — taken first — go out one by one and the
/// pass does not end on a few threads still holding a heavy chunk each.
/// `finish` runs on each worker's state after its last item.
#[allow(clippy::too_many_arguments)]
fn parallel_weighted<T: Sync, N: Num>(
    items: &[T],
    threads: usize,
    pb: &ProgressBar,
    w: &Mutex<Weights<N>>,
    weight: impl Fn(&T) -> u64 + Sync,
    done: Option<&AtomicU64>,
    f: impl Fn(&T, &mut Local<N>) + Sync,
    finish: impl Fn(&mut Local<N>) + Sync,
) {
    if std::env::var_os("TG_CHUNK").is_some() {
        let chunk: usize = std::env::var("TG_CHUNK").unwrap().parse().unwrap();
        let bounds: Vec<usize> = (0..items.len())
            .step_by(chunk)
            .chain([items.len()])
            .collect();
        return parallel_chunks(
            items,
            &bounds,
            threads,
            Ends::InOrder,
            pb,
            w,
            f,
            &finish,
        );
    }
    let total: u64 = items.iter().map(&weight).sum();
    let budget = (total / (64 * threads as u64)).max(1);
    let mut bounds = vec![0];
    let mut sum = 0;
    for (i, item) in items.iter().enumerate() {
        if i > *bounds.last().unwrap()
            && (i - bounds.last().unwrap() >= 64 || sum >= budget)
        {
            bounds.push(i);
            sum = 0;
        }
        sum += weight(item);
    }
    bounds.push(items.len());
    let f = |item: &T, l: &mut Local<N>| {
        f(item, l);
        if let Some(d) = done {
            d.fetch_add(weight(item), Ordering::Relaxed);
        }
    };
    static HEAVY: OnceLock<Option<usize>> = OnceLock::new();
    let heavy = *HEAVY
        .get_or_init(|| std::env::var("TG_HEAVY").ok().map(|v| v.parse().unwrap()));
    #[cfg(feature = "gpu")]
    let backlog = GPU
        .load(Ordering::Relaxed)
        .then(|| {
            std::env::var("TG_SIX_BACKLOG").ok().map_or(16, |v| v.parse().unwrap())
        });
    #[cfg(not(feature = "gpu"))]
    let backlog = None;
    let ends = match (heavy, backlog) {
        (Some(0), _) => Ends::InOrder,
        (Some(k), _) => Ends::Split(k),
        (None, Some(b)) => Ends::Backlog(b),
        (None, None) => Ends::Split((threads / 2).max(1)),
    };
    parallel_chunks(items, &bounds, threads, ends, pb, w, f, &finish);
}
/// Workers on a front (heavy) chunk now, for `TG_TIMELINE`.
static HEAVY_NOW: AtomicUsize = AtomicUsize::new(0);
/// `TG_TIMELINE=1`: every half second until `stop`, the process's CPU use
/// (of `threads`), the GPU's unfinished batches, workers waiting on one,
/// and workers on heavy chunks; printed in 2 s bins at the end (the load
/// swings faster than 5 s bins show).
fn timeline(threads: usize, stop: &std::sync::atomic::AtomicBool) {
    let ticks = || -> u64 {
        let s = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
        let rest = s.rsplit_once(')').map_or("", |r| r.1);
        let f: Vec<&str> = rest.split_whitespace().collect();
        f.get(11).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0)
            + f.get(12).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0)
    };
    let t0 = Instant::now();
    let (mut last_t, mut last_k) = (0.0f64, ticks());
    const BIN: f64 = 2.0;
    let mut bins: Vec<[f64; 5]> = Vec::new();
    while !stop.load(Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(500));
        let (t, k) = (t0.elapsed().as_secs_f64(), ticks());
        let busy = (k - last_k) as f64 / 100.0 / (t - last_t);
        (last_t, last_k) = (t, k);
        #[cfg(feature = "gpu")]
        let pending = gpu::six_pending() as f64;
        #[cfg(not(feature = "gpu"))]
        let pending = 0.0;
        let b = (t / BIN) as usize;
        if bins.len() <= b {
            bins.resize(b + 1, [0.0; 5]);
        }
        let row = [
            1.0,
            busy,
            pending,
            six::WAITING.load(Ordering::Relaxed) as f64,
            HEAVY_NOW.load(Ordering::Relaxed) as f64,
        ];
        for (a, v) in bins[b].iter_mut().zip(row) {
            *a += v;
        }
    }
    println!(
        "timeline ({BIN} s bins: CPU threads busy of {threads}, GPU batches unfinished, workers waiting, workers on heavy chunks):"
    );
    for (i, b) in bins.iter().enumerate().filter(|(_, b)| b[0] > 0.0) {
        println!(
            "  {:4}s  cpu {:5.1}  backlog {:5.1}  waiting {:5.1}  heavy {:5.1}", i as f64
            * BIN, b[1] / b[0], b[2] / b[0], b[3] / b[0], b[4] / b[0]
        );
    }
}
/// How [`parallel_chunks`] hands out its chunks: in order; at most K
/// workers at once from the front, the rest from the back; or from the
/// front while the GPU has fewer than B batches unfinished.
#[derive(Clone, Copy)]
enum Ends {
    InOrder,
    Split(usize),
    Backlog(usize),
}
/// The workers of [`parallel`]: chunk c is `items[bounds[c]..bounds[c + 1]]`,
/// handed out as `ends` says.
#[allow(clippy::too_many_arguments)]
fn parallel_chunks<T: Sync, N: Num>(
    items: &[T],
    bounds: &[usize],
    threads: usize,
    ends_by: Ends,
    pb: &ProgressBar,
    w: &Mutex<Weights<N>>,
    f: impl Fn(&T, &mut Local<N>) + Sync,
    finish: &(dyn Fn(&mut Local<N>) + Sync),
) {
    let ends = Mutex::new((0usize, items.len(), 0usize));
    static HEAVY_MAX: OnceLock<usize> = OnceLock::new();
    let heavy_max = *HEAVY_MAX
        .get_or_init(|| {
            std::env::var("TG_SIX_HEAVY_MAX").ok().map_or(24, |v| v.parse().unwrap())
        });
    static RAMP: OnceLock<u64> = OnceLock::new();
    let ramp = *RAMP
        .get_or_init(|| {
            std::env::var("TG_SIX_RAMP_MS").ok().map_or(3000, |v| v.parse().unwrap())
        });
    let t0 = Instant::now();
    let cap_now = || {
        let ms = t0.elapsed().as_millis() as u64;
        if ramp == 0 || ms >= ramp {
            heavy_max
        } else {
            4 + (heavy_max.saturating_sub(4) as u64 * ms / ramp) as usize
        }
    };
    let last_line = AtomicU64::new(0);
    let paused_ms = AtomicU64::new(0);
    let announced = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope
                .spawn(|| {
                    let mut local = Local::<N>::default();
                    loop {
                        if let Some(pause) = PAUSE.get().filter(|p| p.exists()) {
                            let first = !announced.swap(true, Ordering::Relaxed);
                            let since = Instant::now();
                            if first {
                                eprintln!(
                                    "{}: paused; remove {} to resume", pb.prefix(), pause
                                    .display()
                                );
                            }
                            while pause.exists() {
                                std::thread::sleep(std::time::Duration::from_secs(2));
                            }
                            if first {
                                paused_ms
                                    .fetch_add(
                                        since.elapsed().as_millis() as u64,
                                        Ordering::Relaxed,
                                    );
                                eprintln!(
                                    "{}: resumed after {:.0}s", pb.prefix(), since.elapsed()
                                    .as_secs_f64()
                                );
                                announced.store(false, Ordering::Relaxed);
                            }
                        }
                        let (start, end, front) = {
                            let mut e = ends.lock().unwrap();
                            if e.0 >= e.1 {
                                break;
                            }
                            let front = match ends_by {
                                Ends::InOrder => true,
                                Ends::Split(k) => e.2 < k,
                                #[cfg(feature = "gpu")]
                                Ends::Backlog(b) => {
                                    gpu::six_pending() < b && e.2 < cap_now()
                                }
                                #[cfg(not(feature = "gpu"))]
                                Ends::Backlog(_) => e.2 < cap_now(),
                            };
                            let take = ((e.1 - e.0) / (2 * threads)).max(1);
                            if front {
                                let c = bounds.partition_point(|&s| s <= e.0);
                                let end = bounds[c].min(e.0 + take).min(e.1);
                                let start = std::mem::replace(&mut e.0, end);
                                e.2 += 1;
                                HEAVY_NOW.fetch_add(1, Ordering::Relaxed);
                                (start, end, true)
                            } else {
                                let c = bounds.partition_point(|&s| s < e.1);
                                let start = bounds[c - 1].max(e.1 - take).max(e.0);
                                let end = std::mem::replace(&mut e.1, start);
                                (start, end, false)
                            }
                        };
                        for item in &items[start..end] {
                            f(item, &mut local);
                        }
                        if front {
                            ends.lock().unwrap().2 -= 1;
                            HEAVY_NOW.fetch_sub(1, Ordering::Relaxed);
                        }
                        pb.inc((end - start) as u64);
                        let now = t0.elapsed().as_millis() as u64;
                        let prev = last_line.load(Ordering::Relaxed);
                        if pb.is_hidden() && !pb.prefix().is_empty()
                            && now >= prev + 30_000
                            && last_line
                                .compare_exchange(
                                    prev,
                                    now,
                                    Ordering::Relaxed,
                                    Ordering::Relaxed,
                                )
                                .is_ok()
                        {
                            let (done, len) = (pb.position(), items.len() as u64);
                            let working = now
                                .saturating_sub(paused_ms.load(Ordering::Relaxed))
                                .max(1);
                            let rate = done as f64 / (working as f64 / 1e3);
                            eprintln!(
                                "{}: {done}/{len} in {:.0}s, {rate:.1}/s, eta {:.0}s", pb
                                .prefix(), now as f64 / 1e3, (len - done) as f64 / rate
                            );
                        }
                    }
                    finish(&mut local);
                    if std::env::var_os("TG_MEM_STATS").is_some() {
                        eprintln!("worker memory: {}", local.six.mem_stats());
                        #[cfg(feature = "gpu")]
                        eprintln!("worker queue: {}", local.sixq.mem_stats());
                        eprintln!(
                            "worker weights: w {} fold {}", local.w.len(), local.fold
                            .len()
                        );
                    }
                    let mut g = w.lock().unwrap();
                    for (k, v) in local.w.into_iter().chain(local.fold) {
                        *g.entry(k).or_default() += v;
                    }
                });
        }
    });
}
/// The states one pick past `level`, all held: every step credited, the
/// children merged by key. Stops the process past `--mem-gb`.
fn grow<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    level: &[(Key, Deco<N>)],
    args: &Args,
    mp: &MultiProgress,
    weights: &Mutex<Weights<N>>,
    started: Instant,
) -> Vec<(Key, Deco<N>)> {
    let size = level[0].0.n + 1;
    let per_state = 200.0 + 2.0 * std::mem::size_of::<N>() as f64;
    let max_states = (args.mem_gb * 1e9 / per_state) as usize;
    let pb = bar(mp, level.len() as u64, &format!("size {} → {size}", size - 1));
    let next: Sharded<Key, Deco<N>> = Sharded::new(1024);
    let created = AtomicUsize::new(0);
    let steps = AtomicU64::new(0);
    parallel::<
        _,
        N,
    >(
        level,
        args.threads,
        &pb,
        weights,
        |(k, d), l| {
            let mut n = 0;
            walk(
                t,
                ar,
                k,
                d,
                Some(&mut l.w),
                &mut l.sc,
                &mut |step, live| {
                    n
                        += emit_children(
                            t,
                            k,
                            &step,
                            live,
                            &mut |ck, cd| {
                                let mut shard = next.shard(fxhash(&ck)).lock().unwrap();
                                let e = shard
                                    .entry(ck)
                                    .or_insert_with(|| {
                                        if created.fetch_add(1, Ordering::Relaxed) == max_states {
                                            eprintln!(
                                                "size {size}: past {max_states} states (--mem-gb {}), stopping. Every state carries the \
                             dead labels of its partly admissible multi-forme bases, which multiply a class \
                             multiset's states by ∏(r+1) — see --partial-stats.",
                                                args.mem_gb
                                            );
                                            std::process::exit(2);
                                        }
                                        Deco::new()
                                    });
                                for (c, x) in cd {
                                    deco_add(e, c, x);
                                }
                            },
                        );
                },
            );
            steps.fetch_add(n, Ordering::Relaxed);
        },
    );
    let level = next.into_vec();
    let msg = format!(
        "{} states, {} steps, {} kernels", level.len(), steps.load(Ordering::Relaxed), ar
        .misses.load(Ordering::Relaxed)
    );
    pb.finish_with_message(msg.clone());
    let multisets: std::collections::HashSet<[u16; 5]> = level
        .iter()
        .map(|(k, _)| k.m)
        .collect();
    let mut dead_hist = vec![0u64; 1];
    let mut deco_len = 0usize;
    for (k, d) in &level {
        let n = k.dead_bases() as usize;
        if dead_hist.len() <= n {
            dead_hist.resize(n + 1, 0);
        }
        dead_hist[n] += 1;
        deco_len += d.len();
    }
    println!(
        "size {size}: {msg}; {} class multisets; states by bases with dead labels {:?}; {:.1} contexts per state; {:.0}s",
        multisets.len(), dead_hist, deco_len as f64 / level.len() as f64, started
        .elapsed().as_secs_f64()
    );
    level
}
/// Party slots 1–4 exactly. Slot 3 is the step out of the held size-2
/// states and slot 4 the one after it, folded with that step's label splits
/// integrated (`fold_after`), so no size-3 state is held. Slot 1 also takes
/// the leads drawn late — after never-leads species only — from the few
/// lead-less states, carried on to size 5.
fn run_slots<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    args: &Args,
    mp: &MultiProgress,
    weights: &Mutex<Weights<N>>,
    started: Instant,
) {
    if args.check_quadrature {
        check_quadrature();
        std::process::exit(0);
    }
    let level1 = grow(
        t,
        ar,
        &[(Key::empty(), vec![(0, N::ratio(1, 1))])],
        args,
        mp,
        weights,
        started,
    );
    let mut level2 = grow(t, ar, &level1, args, mp, weights, started);
    if args.limit > 0 {
        let stride = (level2.len() / args.limit).max(1);
        level2 = level2.into_iter().step_by(stride).take(args.limit).collect();
        println!(
            "--limit: {} size-2 states (every {stride}th); the weights are NOT the exact ones",
            level2.len()
        );
    }
    level2
        .sort_unstable_by(|a, b| {
            (a.0.m, a.0.lead, a.0.tb, a.0.fire, a.0.dead)
                .cmp(&(b.0.m, b.0.lead, b.0.tb, b.0.fire, b.0.dead))
        });
    let mut inclusion: Vec<f64> = Vec::new();
    if args.limit_groups > 0 {
        let stride = (level2.len() / args.limit_groups).max(1);
        let mut keep = Vec::new();
        for (lo, hi) in label_groups(&level2) {
            let (lo, hi) = (lo as usize, hi as usize);
            if lo.div_ceil(stride) * stride < hi {
                keep.extend(lo..hi);
                inclusion.push(((hi - lo) as f64 / stride as f64).min(1.0));
            }
        }
        let keep: std::collections::HashSet<usize> = keep.into_iter().collect();
        level2 = level2
            .into_iter()
            .enumerate()
            .filter(|(i, _)| keep.contains(i))
            .map(|(_, s)| s)
            .collect();
        println!(
            "--limit-groups: {} label groups, {} size-2 states; the weights are NOT the exact ones",
            inclusion.len(), level2.len()
        );
    }
    if args.sample_groups > 0 {
        let groups = label_groups(&level2);
        let stride = (groups.len() / args.sample_groups).max(1);
        let keep: Vec<usize> = groups
            .iter()
            .step_by(stride)
            .flat_map(|&(lo, hi)| lo as usize..hi as usize)
            .collect();
        let keep: std::collections::HashSet<usize> = keep.into_iter().collect();
        level2 = level2
            .into_iter()
            .enumerate()
            .filter(|(i, _)| keep.contains(i))
            .map(|(_, s)| s)
            .collect();
        println!(
            "--sample-groups: every {stride}th of {} label groups, {} size-2 states; the weights are NOT the exact ones",
            groups.len(), level2.len()
        );
    }
    if args.check_groups {
        check_groups(t, ar, &level2, args.threads, mp, args.grid, !args.no_profile);
        std::process::exit(0);
    }
    if args.count_six > 0 {
        assert!(N::QUADRATURE, "--count-six integrates in floats");
        count_six(t, ar, &level2, args.count_six, args.threads);
        std::process::exit(0);
    }
    if args.time_six > 0 {
        assert!(N::QUADRATURE, "--time-six integrates in floats");
        time_six(t, ar, &level2, args.time_six, args.threads, mp);
        std::process::exit(0);
    }
    if args.check_analytic > 0 {
        check_analytic(t, ar, &level2, args.check_analytic, args.threads, mp);
        std::process::exit(0);
    }
    if args.check_analytic5 > 0 {
        check_analytic5(t, ar, &level2, args.check_analytic5, args.threads, mp);
        std::process::exit(0);
    }
    if args.check_groups6 > 0 || args.check_six > 0 {
        assert!(N::QUADRATURE, "--check-groups6 integrates in floats");
        check_groups6(
            t,
            ar,
            &level2,
            args.check_groups6.max(args.check_six),
            args.threads,
            mp,
            args.check_six > 0,
        );
        std::process::exit(0);
    }
    if args.check_analytic6 > 0 {
        assert!(N::QUADRATURE, "--check-analytic6 integrates in floats");
        check_analytic6(t, ar, &level2, args.check_analytic6, args.threads, mp);
        std::process::exit(0);
    }
    if args.cp_stats {
        cp_stats(t, &level2);
        std::process::exit(0);
    }
    if args.check_cp > 0 {
        assert!(N::QUADRATURE, "--check-cp runs in floats");
        check_cp(t, ar, &level2, args.check_cp);
        std::process::exit(0);
    }
    if args.group_stats {
        group_stats(&level2);
        std::process::exit(0);
    }
    let mut ck = Ckpt::new(args);
    if args.slot6 {
        ck.load::<N>(weights);
    }
    if ck.stage == 0 {
        before_six(t, ar, args, mp, weights, started, &level2, &inclusion);
        if args.slot6 {
            ck.stage = 1;
            ck.save::<N>(weights);
        }
    }
    if args.slot6 {
        assert!(N::QUADRATURE, "--slot6 integrates in floats");
        let t0 = Instant::now();
        let cost = six_epochs(t, ar, &level2, args.threads, mp, weights, &mut ck);
        let wall = t0.elapsed().as_secs_f64();
        println!(
            "slot 6: {} size-2 states in {wall:.1}s; {:.0}s", level2.len(), started
            .elapsed().as_secs_f64()
        );
        if !inclusion.is_empty() {
            let spent: f64 = cost.iter().sum();
            let full: f64 = cost.iter().zip(&inclusion).map(|(c, p)| c / p).sum();
            println!(
                "slot 6 estimate: {spent:.0} thread-s here, {full:.0} thread-s for every state: {:.1} h at this run's {:.1} threads' worth",
                full / (spent / wall) / 3600.0, spent / wall
            );
        }
    }
}
/// Every pass of [`run_slots`] before slot 6's own: slots 3–5 and their
/// signatures, and the late leads, whose sixth picks are credited here too.
#[allow(clippy::too_many_arguments)]
fn before_six<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    args: &Args,
    mp: &MultiProgress,
    weights: &Mutex<Weights<N>>,
    started: Instant,
    level2: &[(Key, Deco<N>)],
    inclusion: &[f64],
) {
    let table: Sharded<Ctx, KDeco<N>> = Sharded::new(4096);
    let table5: Sharded<Ctx, KDeco<N>> = Sharded::new(16384);
    let leadless: Sharded<Key, Deco<N>> = Sharded::new(64);
    let pb = bar(mp, level2.len() as u64, "slots 3–4");
    let t0 = Instant::now();
    parallel::<
        _,
        N,
    >(
        &level2,
        args.threads,
        &pb,
        weights,
        |(k, d), l| {
            let Local { w, fold, sc, fold_sc, .. } = l;
            walk(
                t,
                ar,
                k,
                d,
                Some(w),
                sc,
                &mut |step, live| {
                    fold_after(t, ar, k, &step, live, fold_sc, &table, fold);
                    if !step.lead {
                        emit_children(
                            t,
                            k,
                            &step,
                            live,
                            &mut |ck, cd| {
                                let mut shard = leadless.shard(fxhash(&ck)).lock().unwrap();
                                let e = shard.entry(ck).or_default();
                                for (c, x) in cd {
                                    deco_add(e, c, x);
                                }
                            },
                        );
                    }
                },
            );
        },
    );
    pb.finish();
    println!(
        "slots 3–4: {} size-2 states in {:.0}s, {} slot-4 signatures, {} kernels; {:.0}s",
        level2.len(), t0.elapsed().as_secs_f64(), table.len(), ar.misses
        .load(Ordering::Relaxed), started.elapsed().as_secs_f64()
    );
    if args.slot5 {
        let pb = bar(mp, 0, "slot 5");
        let t0 = Instant::now();
        let cost = slot5_groups(
            t,
            ar,
            &level2,
            args.threads,
            &pb,
            &table5,
            weights,
            args.grid,
            !args.no_profile,
        );
        if let Some(log) = SLOT6_STATS.get() {
            use std::io::Write;
            log.lock().unwrap().flush().unwrap();
        }
        if let Some(log) = SPLIT_LOG.get() {
            use std::io::Write;
            log.lock().unwrap().flush().unwrap();
        }
        pb.finish();
        let wall = t0.elapsed().as_secs_f64();
        println!(
            "slot 5: {} size-2 states in {wall:.1}s; {:.0}s", level2.len(), started
            .elapsed().as_secs_f64()
        );
        if !inclusion.is_empty() {
            let spent: f64 = cost.iter().sum();
            let full: f64 = cost.iter().zip(inclusion).map(|(c, p)| c / p).sum();
            println!(
                "slot 5 estimate: {spent:.0} thread-s here, {full:.0} thread-s for every state: {:.1} h at this run's {:.1} threads' worth",
                full / (spent / wall) / 3600.0, spent / wall
            );
        }
    }
    let pb = bar(mp, 0, "slot-4 signatures");
    table_pass(t, ar, table, args.threads, &pb, weights);
    pb.finish();
    if args.slot5 && CHECK_CP_ROWS.load(Ordering::Relaxed) {
        CP_CHECK.report();
    }
    if args.slot5 && COUNT_CHANGES.load(Ordering::Relaxed) {
        CHANGES.report();
    }
    if args.slot5 {
        let (pp, pc) = (
            PSHAPE.0.load(Ordering::Relaxed),
            PSHAPE.1.load(Ordering::Relaxed),
        );
        let (qp, qc) = (
            QSHAPE.0.load(Ordering::Relaxed),
            QSHAPE.1.load(Ordering::Relaxed),
        );
        println!(
            "fourth picks: {pp} by profile, {qp} per class ({:.1}%); class evaluations they stand for: {pc} by profile, {qc} per class ({:.1}%)",
            100.0 * qp as f64 / (pp + qp).max(1) as f64, 100.0 * qc as f64 / (pc + qc)
            .max(1) as f64
        );
    }
    if args.slot5 && QSHAPE.0.load(Ordering::Relaxed) > 0 {
        let c = QSHAPE.1.load(Ordering::Relaxed).max(1) as f64;
        println!(
            "quadrature: {} batches, {c} class evaluations ({:.1} a batch): {:.1} nodes, {:.2} varying and {:.2} random bases a class",
            QSHAPE.0.load(Ordering::Relaxed), c / QSHAPE.0.load(Ordering::Relaxed) as
            f64, QSHAPE.2.load(Ordering::Relaxed) as f64 / c, QSHAPE.3
            .load(Ordering::Relaxed) as f64 / c, QSHAPE.4.load(Ordering::Relaxed) as f64
            / c
        );
    }
    if args.slot5 {
        let n = SHAPE.0.load(Ordering::Relaxed).max(1) as f64;
        println!(
            "{} slot-5 signatures; {n} fifth-pick folds: random bases {:.2}, random entries {:.2} (max {}), x-only entries {:.2}",
            table5.len(), SHAPE.1.load(Ordering::Relaxed) as f64 / n, SHAPE.2
            .load(Ordering::Relaxed) as f64 / n, SHAPE.4.load(Ordering::Relaxed), SHAPE.3
            .load(Ordering::Relaxed) as f64 / n
        );
        let pb = bar(mp, 0, "slot-5 signatures");
        table_pass(t, ar, table5, args.threads, &pb, weights);
        pb.finish();
    }
    let table6: Sharded<Ctx, KDeco<N>> = Sharded::new(16384);
    let mut states = leadless.into_vec();
    for size in 3..=5u8 {
        let next: Sharded<Key, Deco<N>> = Sharded::new(64);
        let pb = bar(mp, states.len() as u64, &format!("lead-less {size}"));
        parallel::<
            _,
            N,
        >(
            &states,
            args.threads,
            &pb,
            weights,
            |(k, d), l| {
                let Local { w, fold, sc, fold_sc, .. } = l;
                let w = if size >= 4 { Some(w) } else { None };
                walk(
                    t,
                    ar,
                    k,
                    d,
                    w,
                    sc,
                    &mut |step, live| {
                        if args.slot6 && size == 4 && step.lead {
                            fold_after(t, ar, k, &step, live, fold_sc, &table6, fold);
                        }
                        if size < 5 && !step.lead {
                            emit_children(
                                t,
                                k,
                                &step,
                                live,
                                &mut |ck, cd| {
                                    let mut shard = next.shard(fxhash(&ck)).lock().unwrap();
                                    let e = shard.entry(ck).or_default();
                                    for (c, x) in cd {
                                        deco_add(e, c, x);
                                    }
                                },
                            );
                        }
                    },
                );
            },
        );
        pb.finish();
        println!("lead-less states at size {size}: {}", states.len());
        states = next.into_vec();
    }
    if args.slot6 {
        println!("{} slot-6 signatures from the late leads", table6.len());
        six::table_pass6::<N>(t, ar, &six::Bits::new(t), table6, args.threads, weights);
    }
}
/// Slot 4 over a sample of the size-2 states two ways — label splits
/// integrated (`fold_after`), and enumerated into realized size-3 states
/// each folded on its own — which must agree (to rounding, for floats).
fn check_analytic<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    level: &[(Key, Deco<N>)],
    n: usize,
    threads: usize,
    mp: &MultiProgress,
) {
    let sample: Vec<&(Key, Deco<N>)> = level
        .iter()
        .step_by((level.len() / n).max(1))
        .collect();
    let (wa, wb) = (
        Mutex::new(Weights::<N>::default()),
        Mutex::new(Weights::<N>::default()),
    );
    let (ta, tb): (Sharded<Ctx, KDeco<N>>, Sharded<Ctx, KDeco<N>>) = (
        Sharded::new(1024),
        Sharded::new(1024),
    );
    let pb = bar(mp, sample.len() as u64, "integrated");
    parallel::<
        _,
        N,
    >(
        &sample,
        threads,
        &pb,
        &wa,
        |&(k, d), l| {
            let Local { fold, sc, fold_sc, .. } = l;
            walk(
                t,
                ar,
                k,
                d,
                None,
                sc,
                &mut |step, live| fold_after(t, ar, k, &step, live, fold_sc, &ta, fold),
            );
        },
    );
    let children = AtomicU64::new(0);
    let pb = bar(mp, sample.len() as u64, "enumerated");
    parallel::<
        _,
        N,
    >(
        &sample,
        threads,
        &pb,
        &wb,
        |&(k, d), l| {
            let Local { fold, sc, fold_sc, .. } = l;
            walk(
                t,
                ar,
                k,
                d,
                None,
                sc,
                &mut |step, live| {
                    let n = emit_children(
                        t,
                        k,
                        &step,
                        live,
                        &mut |ck, cd| {
                            let ctx = Ctx::of_key(t, &ck);
                            let av = Avail::of_key(t, &ck);
                            let total = options(t, &ck, &ctx, &av, fold_sc);
                            fold_next(
                                t,
                                ar,
                                &ck,
                                &ctx,
                                &av,
                                total,
                                fold_sc,
                                &[],
                                &cd,
                                &tb,
                                fold,
                            );
                        },
                    );
                    children.fetch_add(n, Ordering::Relaxed);
                },
            );
        },
    );
    let pb = bar(mp, 0, "tables");
    table_pass(t, ar, ta, threads, &pb, &wa);
    table_pass(t, ar, tb, threads, &pb, &wb);
    let (wa, wb) = (wa.into_inner().unwrap(), wb.into_inner().unwrap());
    let (mut agree, mut cells, mut worst) = (0usize, 0usize, String::new());
    let (mut sa, mut sb) = (N::default(), N::default());
    for key in wa.keys().chain(wb.keys().filter(|k| !wa.contains_key(*k))) {
        let a = wa.get(key).map_or(N::default(), |&x| N::get(x));
        let b = wb.get(key).map_or(N::default(), |&x| N::get(x));
        sa = sa + a;
        sb = sb + b;
        cells += 1;
        if N::same(a, b) {
            agree += 1;
        } else if worst.is_empty() {
            worst = format!(
                "first disagreement: {} slot {} lead {} {}", t.sp[key.0 as usize].name,
                key.1, key.2, code_label(key.3)
            );
        }
    }
    println!(
        "check-analytic over {} size-2 states ({} realized size-3 states): {agree} of {cells} cells agree; \
         total mass {} vs {} ({}). {worst}",
        sample.len(), children.load(Ordering::Relaxed), sa.show(), sb.show(), if
        N::same(sa, sb) { "same" } else { "DIFFERENT" }
    );
}
/// How the size-2 states share their label-free part (members, lead, tb,
/// fire): what a group-major slot-5 pass would batch.
fn group_stats<N: Num>(level: &[(Key, Deco<N>)]) {
    let mut groups: FxMap<([u16; 5], bool, bool, u8), (u32, u32)> = FxMap::default();
    let mut deco_len = 0usize;
    for (k, d) in level {
        let e = groups.entry((k.m, k.lead, k.tb, k.fire)).or_default();
        e.0 += 1;
        e.1 = e.1.max(d.len() as u32);
        deco_len += d.len();
    }
    let mut sizes: Vec<u32> = groups.values().map(|g| g.0).collect();
    sizes.sort_unstable();
    let n = level.len() as f64;
    let weighted = sizes.iter().map(|&s| f64::from(s) * f64::from(s)).sum::<f64>() / n;
    let pct = |p: f64| sizes[((sizes.len() - 1) as f64 * p) as usize];
    let multisets: std::collections::HashSet<[u16; 5]> = groups
        .keys()
        .map(|g| g.0)
        .collect();
    println!(
        "{} size-2 states in {} groups ({} multisets): size mean {:.1}, state-weighted {weighted:.1}; \
         p50 {} p90 {} p99 {} max {}; deco length mean {:.1}, group max mean {:.1}",
        level.len(), sizes.len(), multisets.len(), n / sizes.len() as f64, pct(0.5),
        pct(0.9), pct(0.99), sizes.last().unwrap(), deco_len as f64 / n, groups.values()
        .map(| g | f64::from(g.1)).sum::< f64 > () / sizes.len() as f64,
    );
}
/// `--check-cp N`: steps 1 and 2's label splits integrated rather than held,
/// over N label groups — the identity slot 6's rank-one class sums rest on.
/// For fixed G_b, the held side is Σ over a group's variants of its contexts
/// times ∏_b G_b^{alive_b}, over its size-2 live bases. The integrated side
/// walks each order of the two members from the empty team without splitting
/// labels: with 1/M₁ = ∫t^{M₁−1}, an entry of a base the first pick split
/// contributes (1−q₁) + q₁t·h, h its step-2 survival (1−q₂) + q₂G_b (or 1 if
/// the base leaves play), the second pick's own base t∂/∂t, and the rest is
/// fixed. Floats only.
fn check_cp<N: Num>(t: &Tables, ar: &Arith<N>, level: &[(Key, Deco<N>)], n: usize) {
    let groups = label_groups(level);
    let stride = (groups.len() / n).max(1);
    let g_of = |b: u16| 0.3 + 0.7 * (f64::from(b) * 0.618_033_988_749_895).fract();
    let mut sc = Scratch::default();
    let (mut worst, mut worst_at, mut cells, mut checked) = (
        0.0f64,
        String::new(),
        0usize,
        0usize,
    );
    let k0 = Key::empty();
    let deco0: Deco<N> = vec![(0, N::ratio(1, 1))];
    let mut steps1: Vec<(Key, u16, Deco<N>)> = Vec::new();
    let mut live0: Vec<Live> = Vec::new();
    walk(
        t,
        ar,
        &k0,
        &deco0,
        None,
        &mut sc,
        &mut |step, live| {
            live0 = live.to_vec();
            steps1
                .push((
                    step.child(&k0),
                    step.base,
                    step.deco.iter().map(|&(c, x)| (c, x * step.p)).collect(),
                ));
        },
    );
    for &(lo, hi) in groups.iter().step_by(stride) {
        let group = &level[lo as usize..hi as usize];
        let rep = group[0].0;
        let mut lhs: FxMap<u16, f64> = FxMap::default();
        for (k, deco) in group {
            options(t, k, &Ctx::of_key(t, k), &Avail::of_key(t, k), &mut sc);
            let g: f64 = sc
                .live
                .iter()
                .map(|l| g_of(l.base).powi(i32::from(l.alive)))
                .product();
            for &(code, x) in deco {
                *lhs.entry(code).or_default() += N::to_f64(x) * g;
            }
        }
        let mut rhs: FxMap<u16, f64> = FxMap::default();
        for (child1, base1, deco1) in &steps1 {
            let first = child1.members()[0];
            let Some(at) = rep.members().iter().position(|&c| c == first) else {
                continue
            };
            let second = rep.members()[1 - at];
            let ctx1 = Ctx::of_key(t, child1);
            let mut split1: FxMap<u16, (f64, u8)> = FxMap::default();
            for l in &live0 {
                if l.base == *base1 {
                    continue;
                }
                let m1 = forme_mask(t, &ctx1, l.base);
                if m1 != 0 && m1 != l.mask {
                    split1
                        .insert(
                            l.base,
                            (
                                f64::from(m1.count_ones()) / f64::from(l.mask.count_ones()),
                                l.alive,
                            ),
                        );
                }
            }
            let mut steps2: Vec<(Key, u16, u32, u32, Deco<N>, Vec<Live>, u32)> = Vec::new();
            walk(
                t,
                ar,
                child1,
                deco1,
                None,
                &mut sc,
                &mut |step, live| {
                    if step.class == second {
                        steps2
                            .push((
                                step.child(child1),
                                step.base,
                                step.num,
                                step.den,
                                step.deco.clone(),
                                live.to_vec(),
                                step.total,
                            ));
                    }
                },
            );
            for (child2, base2, num2, den2, deco2, live1, total1) in steps2 {
                if (child2.m, child2.lead, child2.tb, child2.fire)
                    != (rep.m, rep.lead, rep.tb, rep.fire)
                {
                    continue;
                }
                let ctx2 = Ctx::of_key(t, &child2);
                let f1 = total1
                    - split1.values().map(|&(_, mu)| u32::from(mu)).sum::<u32>();
                let mut weight = 1.0 / f64::from(den2);
                let mut factors: Vec<(Option<f64>, u8, f64, bool)> = Vec::new();
                for l in &live1 {
                    let s1 = split1.get(&l.base).map(|&(q, _)| q);
                    if l.base == base2 {
                        if s1.is_none() {
                            weight *= f64::from(num2);
                        }
                        factors.push((s1, l.alive, 1.0, true));
                        continue;
                    }
                    let m2 = forme_mask(t, &ctx2, l.base);
                    let h = if m2 == 0 {
                        1.0
                    } else {
                        let q2 = f64::from(m2.count_ones())
                            / f64::from(l.mask.count_ones());
                        (1.0 - q2) + q2 * g_of(l.base)
                    };
                    factors.push((s1, l.alive, h, false));
                }
                if base2 == NONE || !live1.iter().any(|l| l.base == base2) {
                    weight *= f64::from(num2);
                }
                let (xs, ws) = gauss_jacobi(f1, 48);
                let integral: f64 = xs
                    .iter()
                    .zip(&ws)
                    .map(|(&x, &w)| {
                        w
                            * factors
                                .iter()
                                .map(|&(s1, mu, h, picked)| match (s1, picked) {
                                    (Some(q), false) => {
                                        ((1.0 - q) + q * x * h).powi(i32::from(mu))
                                    }
                                    (Some(q), true) => {
                                        f64::from(mu) * q * x
                                            * ((1.0 - q) + q * x).powi(i32::from(mu) - 1)
                                    }
                                    (None, false) => h.powi(i32::from(mu)),
                                    (None, true) => 1.0,
                                })
                                .product::<f64>()
                    })
                    .sum();
                for &(code, x) in &deco2 {
                    *rhs.entry(code).or_default() += N::to_f64(x) * weight * integral;
                }
            }
        }
        checked += 1;
        for code in lhs.keys().chain(rhs.keys().filter(|c| !lhs.contains_key(*c))) {
            let (a, b) = (
                lhs.get(code).copied().unwrap_or(0.0),
                rhs.get(code).copied().unwrap_or(0.0),
            );
            cells += 1;
            let rel = (a - b).abs() / a.abs().max(b.abs()).max(1e-300);
            if rel > worst {
                worst = rel;
                worst_at = format!(
                    "group {:?} code {code}: held {a:.6e}, integrated {b:.6e}", rep
                    .members()
                );
            }
        }
    }
    println!(
        "check-cp over {checked} groups, {cells} (group, code) cells: worst relative gap {worst:.2e} ({worst_at})"
    );
}
/// `--cp-stats`: what integrating steps 1 and 2's label splits would cost
/// against a group's variants. With 1/M₁ = ∫t₁^{M₁−1} and 1/M₂ = ∫t₂^{M₂−1},
/// a group's variant weights are a sum over (t₁, t₂) nodes of products over
/// bases; the nodes needed follow the t-degrees, μ summed over the bases
/// whose live count is random after the first pick (t₁) and after the second
/// (t₂), for each order of the two members.
fn cp_stats<N: Num>(t: &Tables, level: &[(Key, Deco<N>)]) {
    let full = |b: &Base| (1u32 << b.formes.len()) - 1;
    let (mut groups, mut cheaper, mut states, mut states_cheaper) = (
        0u64,
        0u64,
        0u64,
        0u64,
    );
    let mut hist: std::collections::BTreeMap<u32, (u64, u64)> = Default::default();
    for (lo, hi) in label_groups(level) {
        let size = u64::from(hi - lo);
        let mut rep = level[lo as usize].0;
        rep.dead = [0; 2];
        let ctx2 = Ctx::of_key(t, &rep);
        let mut nodes = 0u32;
        for &first in rep.members() {
            let ctx1 = Ctx::of_key(t, &Key::empty().with(first));
            let (mut d1, mut d2) = (0u32, 0u32);
            for (bi, b) in t.bases.iter().enumerate() {
                if b.mi == u8::MAX {
                    continue;
                }
                let (m1, m2) = (
                    forme_mask(t, &ctx1, bi as u16),
                    forme_mask(t, &ctx2, bi as u16),
                );
                let random1 = m1 != full(b) && m1 != 0;
                if random1 {
                    d1 += u32::from(b.mu);
                }
                if m2 != 0 && (random1 || m2 != m1) {
                    d2 += u32::from(b.mu);
                }
            }
            nodes += (d1 / 2 + 1) * (d2 / 2 + 1);
        }
        groups += 1;
        states += size;
        if u64::from(nodes) < size {
            cheaper += 1;
            states_cheaper += size;
        }
        let e = hist.entry(nodes).or_default();
        e.0 += 1;
        e.1 += size;
    }
    println!(
        "cp-stats: {groups} groups, {states} states; (t1, t2) nodes below the group's variant count in {cheaper} groups \
         holding {states_cheaper} states ({:.1}%)",
        100.0 * states_cheaper as f64 / states as f64
    );
    let mut cum = 0u64;
    for (nodes, (g, s)) in &hist {
        cum += s;
        if g * 200 > groups || s * 50 > states {
            println!(
                "  {nodes:>5} nodes: {g:>7} groups, {s:>9} states (cumulative {:.1}% of states)",
                100.0 * cum as f64 / states as f64
            );
        }
    }
}
/// Slot 5 two ways over size-2 states light enough to enumerate: steps 3
/// and 4's label splits integrated jointly (`fold5_after`), and step 3's
/// enumerated into realized size-3 states, each walked and folded with the
/// one-step `fold_after` (checked by `check_analytic`). The slot-5 cells must
/// agree (to rounding, for floats). The heavy states — up to millions of
/// realized size-3 states each — are what the integration exists for, and
/// are left out.
fn check_analytic5<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    level: &[(Key, Deco<N>)],
    n: usize,
    threads: usize,
    mp: &MultiProgress,
) {
    const LIGHT: u64 = 3000;
    let mut sc = Scratch::default();
    let mut sample: Vec<&(Key, Deco<N>)> = Vec::new();
    let mut realized = 0u64;
    for s in level.iter().step_by((level.len() / (n * 20)).max(1)) {
        let (k, d) = s;
        let mut count = 0u64;
        walk(
            t,
            ar,
            k,
            d,
            None,
            &mut sc,
            &mut |step, live| {
                split_children::<
                    N,
                >(t, step.child(k), step.base, live, &mut |_, _| count += 1);
            },
        );
        if count <= LIGHT {
            realized += count;
            sample.push(s);
            if sample.len() == n {
                break;
            }
        }
    }
    let (wa, wb) = (
        Mutex::new(Weights::<N>::default()),
        Mutex::new(Weights::<N>::default()),
    );
    let (ta, tb): (Sharded<Ctx, KDeco<N>>, Sharded<Ctx, KDeco<N>>) = (
        Sharded::new(4096),
        Sharded::new(4096),
    );
    let pb = bar(mp, sample.len() as u64, "integrated");
    let t0 = Instant::now();
    parallel::<
        _,
        N,
    >(
        &sample,
        threads,
        &pb,
        &wa,
        |&(k, d), l| {
            let Local { fold, sc, sc3, sc4, polys, .. } = l;
            walk(
                t,
                ar,
                k,
                d,
                None,
                sc,
                &mut |step, live| fold5_after(
                    t,
                    ar,
                    k,
                    &step,
                    live,
                    sc3,
                    sc4,
                    polys,
                    &ta,
                    fold,
                ),
            );
        },
    );
    let t_int = t0.elapsed().as_secs_f64();
    let pb = bar(mp, sample.len() as u64, "enumerated");
    let t0 = Instant::now();
    parallel::<
        _,
        N,
    >(
        &sample,
        threads,
        &pb,
        &wb,
        |&(k, d), l| {
            let Local { fold, sc, sc3, sc4, .. } = l;
            walk(
                t,
                ar,
                k,
                d,
                None,
                sc,
                &mut |step, live| {
                    emit_children(
                        t,
                        k,
                        &step,
                        live,
                        &mut |ck, cd| {
                            walk(
                                t,
                                ar,
                                &ck,
                                &cd,
                                None,
                                sc3,
                                &mut |step4, live3| fold_after(
                                    t,
                                    ar,
                                    &ck,
                                    &step4,
                                    live3,
                                    sc4,
                                    &tb,
                                    fold,
                                ),
                            );
                        },
                    );
                },
            );
        },
    );
    let t_enum = t0.elapsed().as_secs_f64();
    let pb = bar(mp, 0, "tables");
    table_pass(t, ar, ta, threads, &pb, &wa);
    table_pass(t, ar, tb, threads, &pb, &wb);
    println!(
        "check-analytic5 over {} light size-2 states ({realized} realized size-3 states; integrated {t_int:.1}s, \
         enumerated {t_enum:.1}s): {}",
        sample.len(), compare_slot::< N > (t, & wa.into_inner().unwrap(), & wb
        .into_inner().unwrap(), 5)
    );
}
/// Slot 6's integrals for one fifth pick, from a size-3 situation: with 1/M₃
/// = ∫x^{M₃−1}, 1/M₄ = ∫y^{M₄−1} and 1/M₅ = ∫z^{M₅−1}, an entry of a size-3
/// live base contributes (1−q₃) + q₃x((1−q₄) + q₄y((1−q₅) + q₅z)), a q zero
/// where the base leaves play. The fourth pick's base gives one entry q₃x
/// (its weight A₃ is `num4` with the caller) and the fifth's one entry
/// q₃x·q₄y; a base the three steps leave whole goes into f₃, f₄ and f₅.
/// Returns E[w₄w₅/(M₃M₄M₅)] over the picks' num/den, and pushes, by size-5
/// live index, each random count's E[w₄w₅A₅/(M₃M₄M₅)] over the same — the
/// moment its corrections need. Gauss–Jacobi in each variable, so exact for
/// the polynomial; floats only, a reference.
#[allow(clippy::too_many_arguments)]
fn integrate6(
    mask2: &[u32; 64],
    live3: &[Live],
    base4: u16,
    live4: &[Live],
    base5: u16,
    live5: &[Live],
    mut f3: u32,
    mut f4: u32,
    mut f5: u32,
    quad: &mut Quad,
    moments: &mut Vec<(usize, f64)>,
) -> f64 {
    let ratio = |a: u32, b: u32| f64::from(a.count_ones()) / f64::from(b.count_ones());
    let mut bases: Vec<(u8, f64, f64, f64, u8, Option<usize>)> = Vec::new();
    let (mut dx, mut dy, mut dz) = (0usize, 0usize, 0usize);
    for l3 in live3 {
        let r = l3.alive;
        if r == 0 {
            continue;
        }
        let q3 = ratio(l3.mask, mask2[usize::from(l3.mi)]);
        if l3.base == base4 {
            if q3 == 1.0 {
                f3 += u32::from(r);
            } else {
                bases.push((r, q3, 0.0, 0.0, 4, None));
                dx += usize::from(r);
            }
            continue;
        }
        let m4 = live4.iter().find(|l| l.mi == l3.mi).map_or(0, |l| l.mask);
        let q4 = if m4 == 0 { 0.0 } else { ratio(m4, l3.mask) };
        if l3.base == base5 {
            bases.push((r, q3, q4, 0.0, 5, None));
            dx += usize::from(r);
            dy += usize::from(r);
            continue;
        }
        let at5 = if m4 == 0 { None } else { live5.iter().position(|l| l.mi == l3.mi) };
        let q5 = at5.map_or(0.0, |j| ratio(live5[j].mask, m4));
        if q3 == 1.0 && q4 == 1.0 && q5 == 1.0 {
            f3 += u32::from(r);
            f4 += u32::from(r);
            f5 += u32::from(r);
            continue;
        }
        bases.push((r, q3, q4, q5, 0, at5));
        dx += usize::from(r);
        if q4 > 0.0 {
            dy += usize::from(r);
        }
        if q5 > 0.0 {
            dz += usize::from(r);
        }
    }
    assert!(
        f3 >= 1 && f4 >= 1 && f5 >= 1, "a pool that can run dry before the sixth pick"
    );
    let (xs, wx) = quad.rule(f3, dx / 2 + 1).clone();
    let (ys, wy) = quad.rule(f4, dy / 2 + 1).clone();
    let (zs, wz) = quad.rule(f5, dz / 2 + 1).clone();
    let random: Vec<usize> = (0..bases.len())
        .filter(|&b| bases[b].4 == 0 && bases[b].5.is_some())
        .collect();
    let mut sums = vec![0.0; 1 + random.len()];
    let mut factor = vec![0.0; bases.len()];
    for (&x, &vx) in xs.iter().zip(&wx) {
        for (&y, &vy) in ys.iter().zip(&wy) {
            for (&z, &vz) in zs.iter().zip(&wz) {
                let entry = |q3: f64, q4: f64, q5: f64| {
                    (1.0 - q3) + q3 * x * ((1.0 - q4) + q4 * y * ((1.0 - q5) + q5 * z))
                };
                for (f, &(r, q3, q4, q5, role, _)) in factor.iter_mut().zip(&bases) {
                    let rest = entry(q3, q4, q5).powi(i32::from(r) - 1);
                    *f = match role {
                        4 => q3 * x * rest,
                        5 => q3 * x * q4 * y * rest,
                        _ => entry(q3, q4, q5) * rest,
                    };
                }
                let w = vx * vy * vz;
                let all: f64 = factor.iter().product();
                sums[0] += w * all;
                for (k, &b) in random.iter().enumerate() {
                    let (r, q3, q4, q5, ..) = bases[b];
                    let moment = f64::from(r) * q3 * x * q4 * y * q5 * z
                        * entry(q3, q4, q5).powi(i32::from(r) - 1);
                    let others: f64 = factor
                        .iter()
                        .enumerate()
                        .filter(|&(i, _)| i != b)
                        .map(|(_, f)| f)
                        .product();
                    sums[1 + k] += w * moment * others;
                }
            }
        }
    }
    moments
        .extend(
            random.iter().enumerate().map(|(k, &b)| (bases[b].5.unwrap(), sums[1 + k])),
        );
    sums[0]
}
/// p ← p·(a + b·z)^r, coefficients from z⁰ up.
fn times_linear_pow(p: &mut Vec<f64>, a: f64, b: f64, r: u8) {
    for _ in 0..r {
        p.push(0.0);
        for k in (1..p.len()).rev() {
            p[k] = p[k] * a + p[k - 1] * b;
        }
        p[0] *= a;
    }
}
/// p / (a + b·z) for a factor that divides p, from the top when b ≥ a and
/// from the bottom otherwise, so no step multiplies an error by more than 1.
fn div_linear_f64(p: &[f64], a: f64, b: f64, out: &mut Vec<f64>) {
    let d = p.len() - 1;
    out.clear();
    out.resize(d.max(1), 0.0);
    if d == 0 {
        out[0] = p[0] / a;
        return;
    }
    if b == 0.0 {
        out.resize(d + 1, 0.0);
        for (o, &c) in out.iter_mut().zip(p) {
            *o = c / a;
        }
        return;
    }
    if b >= a {
        out[d - 1] = p[d] / b;
        for k in (1..d).rev() {
            out[k - 1] = (p[k] - a * out[k]) / b;
        }
    } else {
        out[0] = p[0] / a;
        for k in 1..d {
            out[k] = (p[k] - b * out[k - 1]) / a;
        }
    }
}
/// Σ_{i,j} p[i]·q[j]/(f + i + j): ∫z^{f−1}·p(z)·q(z) dz.
fn integrate_product(p: &[f64], q: &[f64], f: u32) -> f64 {
    let mut s = 0.0;
    for (i, &pi) in p.iter().enumerate() {
        if pi == 0.0 {
            continue;
        }
        for (j, &qj) in q.iter().enumerate() {
            s += pi * qj / (f64::from(f) + (i + j) as f64);
        }
    }
    s
}
/// [`integrate6`] by a cheaper route, the same contract. A base steps 3
/// and 4 leave whole contributes x^r·y^r — folded into f₃ and f₄ — times
/// (1 − q₅ + q₅z)^r: z^r into f₅ when the fifth pick leaves it too, else a
/// factor of the exact polynomial R̃(z). The bases steps 3 or 4 split are
/// (A + B·h(z))^r with A, B depending on x and y: quadrature in x and y, an
/// exact polynomial in z at each node, summed into P(z). Then V =
/// Σ P[i]·R̃[j]/(f₅ + i + j), and each moment is the same with one base's
/// factor replaced by its z∂/∂z — an exact division of the node polynomial,
/// or of R̃. Every coefficient is positive.
#[allow(clippy::too_many_arguments)]
fn integrate6_fast(
    mask2: &[u32; 64],
    live3: &[Live],
    base4: u16,
    live4: &[Live],
    base5: u16,
    live5: &[Live],
    mut f3: u32,
    mut f4: u32,
    mut f5: u32,
    quad: &mut Quad,
    moments: &mut Vec<(usize, f64)>,
) -> f64 {
    let ratio = |a: u32, b: u32| f64::from(a.count_ones()) / f64::from(b.count_ones());
    let mut split: Vec<(u8, f64, f64, f64, u8, Option<usize>)> = Vec::new();
    let mut whole: Vec<(u8, f64, Option<usize>)> = Vec::new();
    let (mut dx, mut dy) = (0usize, 0usize);
    for l3 in live3 {
        let r = l3.alive;
        if r == 0 {
            continue;
        }
        let q3 = ratio(l3.mask, mask2[usize::from(l3.mi)]);
        if l3.base == base4 {
            if q3 == 1.0 {
                f3 += u32::from(r);
            } else {
                split.push((r, q3, 0.0, 0.0, 4, None));
                dx += usize::from(r);
            }
            continue;
        }
        let m4 = live4.iter().find(|l| l.mi == l3.mi).map_or(0, |l| l.mask);
        let q4 = if m4 == 0 { 0.0 } else { ratio(m4, l3.mask) };
        if l3.base == base5 {
            if q3 == 1.0 && q4 == 1.0 {
                f3 += u32::from(r);
                f4 += u32::from(r);
            } else {
                split.push((r, q3, q4, 0.0, 5, None));
                dx += usize::from(r);
                dy += usize::from(r);
            }
            continue;
        }
        let at5 = if m4 == 0 { None } else { live5.iter().position(|l| l.mi == l3.mi) };
        let q5 = at5.map_or(0.0, |j| ratio(live5[j].mask, m4));
        if q3 == 1.0 && q4 == 1.0 {
            f3 += u32::from(r);
            f4 += u32::from(r);
            if q5 == 1.0 {
                f5 += u32::from(r);
            } else if q5 > 0.0 {
                whole.push((r, q5, at5));
            }
            continue;
        }
        split.push((r, q3, q4, q5, 0, at5));
        dx += usize::from(r);
        if q4 > 0.0 {
            dy += usize::from(r);
        }
    }
    assert!(
        f3 >= 1 && f4 >= 1 && f5 >= 1, "a pool that can run dry before the sixth pick"
    );
    let mut rt = vec![1.0];
    for &(r, q5, _) in &whole {
        times_linear_pow(&mut rt, 1.0 - q5, q5, r);
    }
    let (xs, wx) = quad.rule(f3, dx / 2 + 1).clone();
    let (ys, wy) = quad.rule(f4, dy / 2 + 1).clone();
    let need: Vec<usize> = (0..split.len())
        .filter(|&b| split[b].4 == 0 && split[b].5.is_some() && split[b].3 > 0.0)
        .collect();
    let mut p_sum: Vec<f64> = Vec::new();
    let mut m_sum: Vec<Vec<f64>> = vec![Vec::new(); need.len()];
    let (mut node, mut quo) = (Vec::with_capacity(72), Vec::with_capacity(72));
    for (&x, &vx) in xs.iter().zip(&wx) {
        for (&y, &vy) in ys.iter().zip(&wy) {
            let w = vx * vy;
            let mut scalar = w;
            node.clear();
            node.push(1.0);
            for &(r, q3, q4, q5, role, _) in &split {
                match role {
                    4 => scalar *= q3 * x * ((1.0 - q3) + q3 * x).powi(i32::from(r) - 1),
                    5 => {
                        scalar
                            *= q3 * x * q4 * y
                                * ((1.0 - q3) + q3 * x * ((1.0 - q4) + q4 * y))
                                    .powi(i32::from(r) - 1);
                    }
                    _ => {
                        let bb = q3 * x * q4 * y;
                        let a = (1.0 - q3) + q3 * x * (1.0 - q4) + bb * (1.0 - q5);
                        if q5 == 0.0 {
                            scalar *= a.powi(i32::from(r));
                        } else {
                            times_linear_pow(&mut node, a, bb * q5, r);
                        }
                    }
                }
            }
            if p_sum.len() < node.len() {
                p_sum.resize(node.len(), 0.0);
            }
            for (s, &c) in p_sum.iter_mut().zip(&node) {
                *s += c * scalar;
            }
            for (k, &b) in need.iter().enumerate() {
                let (r, q3, q4, q5, ..) = split[b];
                let bb = q3 * x * q4 * y;
                let a = (1.0 - q3) + q3 * x * (1.0 - q4) + bb * (1.0 - q5);
                div_linear_f64(&node, a, bb * q5, &mut quo);
                let m = &mut m_sum[k];
                if m.len() < quo.len() + 1 {
                    m.resize(quo.len() + 1, 0.0);
                }
                let scale = f64::from(r) * bb * q5 * scalar;
                for (i, &c) in quo.iter().enumerate() {
                    m[i + 1] += c * scale;
                }
            }
        }
    }
    let v = integrate_product(&p_sum, &rt, f5);
    for (k, &b) in need.iter().enumerate() {
        moments.push((split[b].5.unwrap(), integrate_product(&m_sum[k], &rt, f5)));
    }
    let mut rq = Vec::with_capacity(rt.len() + 1);
    for &(r, q5, at5) in &whole {
        div_linear_f64(&rt, 1.0 - q5, q5, &mut quo);
        rq.clear();
        rq.push(0.0);
        rq.extend(quo.iter().map(|&c| c * f64::from(r) * q5));
        moments.push((at5.unwrap(), integrate_product(&p_sum, &rq, f5)));
    }
    v
}
/// `TG_CHECK_FAST=1`: [`integrate6_fast`] against [`integrate6`], call by call.
static FAST_GAP: Mutex<(f64, u64)> = Mutex::new((0.0, 0));
#[allow(clippy::too_many_arguments)]
fn integrate6_checked(
    mask2: &[u32; 64],
    live3: &[Live],
    base4: u16,
    live4: &[Live],
    base5: u16,
    live5: &[Live],
    f3: u32,
    f4: u32,
    f5: u32,
    quad: &mut Quad,
    moments: &mut Vec<(usize, f64)>,
) -> f64 {
    static CHECK: OnceLock<bool> = OnceLock::new();
    let v = integrate6_fast(
        mask2,
        live3,
        base4,
        live4,
        base5,
        live5,
        f3,
        f4,
        f5,
        quad,
        moments,
    );
    if *CHECK.get_or_init(|| std::env::var_os("TG_CHECK_FAST").is_some()) {
        let mut slow = Vec::new();
        let v2 = integrate6(
            mask2,
            live3,
            base4,
            live4,
            base5,
            live5,
            f3,
            f4,
            f5,
            quad,
            &mut slow,
        );
        let mut gap = (v - v2).abs() / v.abs().max(v2.abs()).max(1e-300);
        for &(j, m) in &slow {
            let fast = moments.iter().find(|x| x.0 == j).map_or(0.0, |x| x.1);
            gap = gap.max((fast - m).abs() / fast.abs().max(m.abs()).max(1e-300));
        }
        if moments.len() != slow.len() {
            gap = f64::INFINITY;
        }
        let mut g = FAST_GAP.lock().unwrap();
        g.0 = g.0.max(gap);
        g.1 += 1;
    }
    v
}
/// The sixth pick after `step5`, folded through `table6` as the fifth is
/// through `table5` ([`fold5_pick`]), with steps 3–5's splits integrated.
#[allow(clippy::too_many_arguments)]
fn fold6_pick<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    child4: &Key,
    step5: &Step<N>,
    mask2: &[u32; 64],
    live3: &[Live],
    base4: u16,
    live4: &[Live],
    f3: u32,
    f4: u32,
    c4: N,
    sc5: &mut Scratch,
    quad: &mut Quad,
    table6: &Sharded<Ctx, KDeco<N>>,
    w: &mut Weights<N>,
) {
    let child5 = step5.child(child4);
    if !child5.lead {
        return;
    }
    let ctx5 = Ctx::of_key(t, &child5);
    let av5 = Avail::of_key(t, &child5);
    let total5 = options_into(t, &child5, &ctx5, &av5, &mut sc5.live, None);
    if total5 == 0 {
        return;
    }
    let f5 = total5 - sc5.live.iter().map(|l| u32::from(l.alive)).sum::<u32>();
    let mut moments = Vec::new();
    let i0 = integrate6(
        mask2,
        live3,
        base4,
        live4,
        step5.base,
        &sc5.live,
        f3,
        f4,
        f5,
        quad,
        &mut moments,
    );
    let c = c4 * N::ratio(u64::from(step5.num), u64::from(step5.den));
    let moments: Vec<(usize, N)> = moments
        .into_iter()
        .map(|(j, m)| (j, c * N::of_f64(m)))
        .collect();
    fold_credit(
        t,
        ar,
        &child5,
        &ctx5,
        &av5,
        sc5,
        c * N::of_f64(i0),
        &moments,
        step5.deco,
        table6,
        w,
    );
}
/// Slot 6 from one size-2 state's third pick `step3`: every fourth and
/// fifth pick walked, the sixth folded ([`fold6_pick`]).
#[allow(clippy::too_many_arguments)]
fn fold6_after<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    k2: &Key,
    step3: &Step<N>,
    live2: &[Live],
    sc3: &mut Scratch,
    sc4: &mut Scratch,
    sc5: &mut Scratch,
    quad: &mut Quad,
    table6: &Sharded<Ctx, KDeco<N>>,
    w: &mut Weights<N>,
) {
    let child3 = step3.child(k2);
    let d3: Deco<N> = step3.deco.iter().map(|&(c, x)| (c, x * step3.p)).collect();
    let mut mask2 = [0u32; 64];
    for l in live2 {
        mask2[usize::from(l.mi)] = l.mask;
    }
    walk(
        t,
        ar,
        &child3,
        &d3,
        None,
        sc3,
        &mut |step4, live3| {
            let child4 = step4.child(&child3);
            let f3 = step4.total - live3.iter().map(|l| u32::from(l.alive)).sum::<u32>();
            let c4 = N::ratio(u64::from(step4.num), u64::from(step4.den));
            walk(
                t,
                ar,
                &child4,
                step4.deco,
                None,
                sc4,
                &mut |step5, live4| {
                    let f4 = step5.total
                        - live4.iter().map(|l| u32::from(l.alive)).sum::<u32>();
                    fold6_pick(
                        t,
                        ar,
                        &child4,
                        &step5,
                        &mask2,
                        live3,
                        step4.base,
                        live4,
                        f3,
                        f4,
                        c4,
                        sc5,
                        quad,
                        table6,
                        w,
                    );
                },
            );
        },
    );
}
/// Slot 6 from one fourth pick of a situation, group-major: every fifth pick
/// walked, each class's steps 3–5 integrals ([`integrate6`]) weighted by its
/// contexts, and the sixth pick folded through `table6` with its corrections
/// — [`fold6_pick`] over a situation's classes at once. A class's contexts
/// pass through the fourth and fifth picks' own sets before they are
/// weighted; a base whose count after step 5 is fixed takes `alive × V` as
/// its moment, and is skipped where [`fold_credit`] skips it.
#[allow(clippy::too_many_arguments)]
fn slot6_pick<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    s: &Situation<N>,
    p: &Pick5<N>,
    sc: &mut Scratch,
    sc5: &mut Scratch,
    quad: &mut Quad,
    table6: &Sharded<Ctx, KDeco<N>>,
    w: &mut Weights<N>,
) {
    let nc = s.nc();
    let classes = s.keys.len();
    let deco4: Vec<Deco<N>> = (0..classes)
        .map(|c| {
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
                p.c4.tb,
                p.c4.fire,
                d,
            )
        })
        .collect();
    let w4 = |c: usize| match p.at4_base {
        Some(i4) => N::ratio(u64::from(s.r(c, i4)), u64::from(p.den)),
        None => N::ratio(u64::from(p.num), u64::from(p.den)),
    };
    options(t, &p.c4, &p.ctx4, &p.av4, sc);
    let opts = sc.opts.clone();
    let live4 = sc.live.clone();
    let tb_code = |tb: bool| if tb { TB_BIT } else { 0 };
    let mut live3_c = s.live3.clone();
    let mut split = Vec::new();
    for o5 in &opts {
        let (class5, picked5, lead_pick5, base5) = pick_parts(t, &p.c4, o5.pick);
        let flagged5 = matches!(o5.pick, Pick::Forme(_)) && t.sp[picked5 as usize].flags;
        let i5 = (base5 != NONE && t.bases[base5 as usize].mi != u8::MAX)
            .then(|| usize::from(s.at3_of[usize::from(t.bases[base5 as usize].mi)]));
        let mut outcomes: Vec<(bool, u8, Vec<Deco<N>>)> = Vec::new();
        for (c, d) in deco4.iter().enumerate() {
            split.clear();
            if flagged5 {
                kernel_split(t, ar, &p.c4, picked5, lead_pick5, d, &mut split);
            } else {
                split.push((p.c4.tb, p.c4.fire, d.clone()));
            }
            for (tb, fire, d5) in split.drain(..) {
                let at = match outcomes.iter().position(|o| o.0 == tb && o.1 == fire) {
                    Some(at) => at,
                    None => {
                        outcomes.push((tb, fire, vec![Vec::new(); classes]));
                        outcomes.len() - 1
                    }
                };
                outcomes[at].2[c] = d5;
            }
        }
        for (tb5, fire5, decos) in outcomes {
            let mut c5 = p.c4.with(class5);
            c5.lead = p.c4.lead || lead_pick5;
            c5.tb = tb5;
            c5.fire = fire5;
            if !c5.lead {
                continue;
            }
            let ctx5 = Ctx::of_key(t, &c5);
            let av5 = Avail::of_key(t, &c5);
            let total5 = options_into(t, &c5, &ctx5, &av5, &mut sc5.live, None);
            if total5 == 0 {
                continue;
            }
            let f5 = total5 - sc5.live.iter().map(|l| u32::from(l.alive)).sum::<u32>();
            let live5 = sc5.live.clone();
            let mut u0: Deco<N> = Vec::new();
            let mut rows: Vec<(Deco<N>, Deco<N>)> = vec![
                (Vec::new(), Vec::new()); live5.len()
            ];
            let mut moms = Vec::new();
            for c in 0..classes {
                if decos[c].is_empty() {
                    continue;
                }
                for (i, l) in live3_c.iter_mut().enumerate() {
                    l.alive = s.r(c, i);
                }
                let num5 = match i5 {
                    Some(i) => u32::from(s.r(c, i)),
                    None => o5.num,
                };
                if num5 == 0 {
                    continue;
                }
                moms.clear();
                let i0 = integrate6_checked(
                    &s.mask2,
                    &live3_c,
                    p.base4,
                    &live4,
                    base5,
                    &live5,
                    s.f3,
                    p.f4,
                    f5,
                    quad,
                    &mut moms,
                );
                let k = w4(c) * N::ratio(u64::from(num5), u64::from(o5.den));
                let v = k * N::of_f64(i0);
                for &(code, x) in &decos[c] {
                    deco_add(&mut u0, code, x * v);
                }
                for (j, l5) in live5.iter().enumerate() {
                    let base = &t.bases[l5.base as usize];
                    let full = (1u32 << base.formes.len()) - 1;
                    let m = match moms.iter().find(|m| m.0 == j) {
                        Some(&(_, m)) => k * N::of_f64(m),
                        None => {
                            let alive = s
                                .at3_of
                                .get(usize::from(l5.mi))
                                .map_or(
                                    l5.alive,
                                    |&i3| {
                                        if i3 == u8::MAX {
                                            l5.alive
                                        } else {
                                            s.r(c, usize::from(i3))
                                        }
                                    },
                                );
                            if l5.mask == full && alive == base.mu {
                                continue;
                            }
                            N::ratio(u64::from(alive), 1) * v
                        }
                    };
                    for &(code, x) in &decos[c] {
                        deco_add(&mut rows[j].0, code, x * m);
                        deco_add(&mut rows[j].1, code, x * v);
                    }
                }
            }
            if u0.is_empty() {
                continue;
            }
            credit_signature(t, ar, &c5, &ctx5, &av5, &u0, table6, w);
            for (j, l5) in live5.iter().enumerate() {
                let base = &t.bases[l5.base as usize];
                let size = N::ratio(1, u64::from(l5.mask.count_ones()));
                for (fi, &f) in base.formes.iter().enumerate() {
                    if l5.mask & (1 << fi) == 0 {
                        continue;
                    }
                    let (sl, lead) = pick_slot(&c5, &t.sp[f as usize]);
                    let om = ar.omega[f as usize];
                    let (moment, inv) = &rows[j];
                    for &(code, x) in moment {
                        credit(w, f, sl, lead, code | tb_code(c5.tb), x * size);
                    }
                    for &(code, x) in inv {
                        credit(
                            w,
                            f,
                            sl,
                            lead,
                            code | tb_code(c5.tb),
                            N::default() - om * x,
                        );
                    }
                }
            }
        }
    }
}
/// Slot 6 for one (label group, third pick): [`slot5_item`]'s situations, each
/// fourth pick handed to [`slot6_pick`].
fn slot6_item<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    group: &[(Key, Deco<N>)],
    o3: usize,
    l: &mut Local<N>,
    table6: &Sharded<Ctx, KDeco<N>>,
) {
    let Local { fold, sc, sc3, sc4, fold_sc, quad, .. } = l;
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
        let b = &t.bases[base as usize];
        b.mu - k.dead_of(b.mi)
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
        let p = N::ratio(u64::from(num), u64::from(o.den) * u64::from(m3));
        split.clear();
        if flagged3 {
            kernel_split(t, ar, &rep, picked3, lead_pick3, deco, &mut split);
        } else {
            split.push((rep.tb, rep.fire, deco.clone()));
        }
        for (tb, fire, d) in split.drain(..) {
            let d: Deco<N> = d.into_iter().map(|(c, x)| (c, x * p)).collect();
            match outcomes3.iter_mut().find(|e| e.0 == tb && e.1 == fire) {
                Some(e) => e.2.push((v, d)),
                None => outcomes3.push((tb, fire, vec![(v, d)])),
            }
        }
    }
    let mut sc6 = Scratch::default();
    for (tb3, fire3, variants) in outcomes3 {
        let mut c3 = rep.with(class3);
        c3.lead = rep.lead || lead_pick3;
        c3.tb = tb3;
        c3.fire = fire3;
        let s = Situation::new(t, c3, mask2, group, &variants, sc3);
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
                slot6_pick(t, ar, &s, &p, fold_sc, &mut sc6, quad, table6, fold);
            }
        }
    }
}
/// [`slot6_item`] over every (label group, third pick) of `level`.
fn slot6_groups<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    level: &[(Key, Deco<N>)],
    threads: usize,
    pb: &ProgressBar,
    table6: &Sharded<Ctx, KDeco<N>>,
    weights: &Mutex<Weights<N>>,
) {
    let groups = label_groups(level);
    let mut sc = Scratch::default();
    let mut items: Vec<(u32, u16)> = Vec::new();
    for (g, &(lo, _)) in groups.iter().enumerate() {
        let mut rep = level[lo as usize].0;
        rep.dead = [0; 2];
        options(t, &rep, &Ctx::of_key(t, &rep), &Avail::of_key(t, &rep), &mut sc);
        items.extend((0..sc.opts.len()).map(|i| (g as u32, i as u16)));
    }
    items
        .sort_by_key(|&(g, _)| std::cmp::Reverse(
            groups[g as usize].1 - groups[g as usize].0,
        ));
    pb.set_length(items.len() as u64);
    parallel::<
        _,
        N,
    >(
        &items,
        threads,
        pb,
        weights,
        |&(g, o3), l| {
            let (lo, hi) = groups[g as usize];
            slot6_item(
                t,
                ar,
                &level[lo as usize..hi as usize],
                usize::from(o3),
                l,
                table6,
            );
        },
    );
}
/// Slot 6 per state ([`fold6_after`]) and group-major — [`slot6_item`], or
/// with `fast` [`six::item`] — over the same few (label group, third pick)
/// items: the cells must agree.
fn check_groups6<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    level: &[(Key, Deco<N>)],
    n: usize,
    threads: usize,
    mp: &MultiProgress,
    fast: bool,
) {
    let bits = six::Bits::new(t);
    let groups = label_groups(level);
    let mut sc = Scratch::default();
    let mut items: Vec<(u32, u16, Pick)> = Vec::new();
    let big = std::env::var_os("TG_CHECK_BIG").is_some();
    let step = if big { 7 } else { (groups.len() / (n * 3)).max(1) };
    for (g, &(lo, hi)) in groups.iter().enumerate().step_by(step) {
        let size = hi - lo;
        if if big { !(8..=30).contains(&size) } else { size > 6 } {
            continue;
        }
        let mut rep = level[lo as usize].0;
        rep.dead = [0; 2];
        options(t, &rep, &Ctx::of_key(t, &rep), &Avail::of_key(t, &rep), &mut sc);
        let o3 = (g * 7919) % sc.opts.len();
        items.push((g as u32, o3 as u16, sc.opts[o3].pick));
        if items.len() == n {
            break;
        }
    }
    let ident = |pick: Pick| match pick {
        Pick::Class(c) => (c, NONE),
        Pick::Forme(f) => (t.sp[f as usize].class, t.sp[f as usize].base),
    };
    let (wa, wb) = (
        Mutex::new(Weights::<N>::default()),
        Mutex::new(Weights::<N>::default()),
    );
    let (ta, tb): (Sharded<Ctx, KDeco<N>>, Sharded<Ctx, KDeco<N>>) = (
        Sharded::new(4096),
        Sharded::new(4096),
    );
    let states: Vec<(usize, Pick)> = items
        .iter()
        .flat_map(|&(g, _, pick)| {
            (groups[g as usize].0..groups[g as usize].1).map(move |i| (i as usize, pick))
        })
        .collect();
    let pb = bar(mp, states.len() as u64, "per state");
    let t0 = Instant::now();
    parallel::<
        _,
        N,
    >(
        &states,
        threads,
        &pb,
        &wa,
        |&(i, pick), l| {
            let Local { fold, sc, sc3, sc4, fold_sc, quad, .. } = l;
            let (k, d) = &level[i];
            let want = ident(pick);
            walk(
                t,
                ar,
                k,
                d,
                None,
                sc,
                &mut |step, live| {
                    if (step.class, step.base) == want {
                        fold6_after(
                            t,
                            ar,
                            k,
                            &step,
                            live,
                            sc3,
                            sc4,
                            fold_sc,
                            quad,
                            &ta,
                            fold,
                        );
                    }
                },
            );
        },
    );
    let t_state = t0.elapsed().as_secs_f64();
    let pb = bar(mp, items.len() as u64, "group-major");
    let t0 = Instant::now();
    parallel::<
        _,
        N,
    >(
        &items,
        threads,
        &pb,
        &wb,
        |&(g, o3, _), l| {
            let (lo, hi) = groups[g as usize];
            if fast {
                six::item(
                    t,
                    ar,
                    &bits,
                    &level[lo as usize..hi as usize],
                    usize::from(o3),
                    l,
                    &tb,
                );
                six::finish(t, &bits, l, &tb);
            } else {
                slot6_item(
                    t,
                    ar,
                    &level[lo as usize..hi as usize],
                    usize::from(o3),
                    l,
                    &tb,
                );
            }
        },
    );
    let t_group = t0.elapsed().as_secs_f64();
    let pb = bar(mp, 0, "tables");
    table_pass(t, ar, ta, threads, &pb, &wa);
    table_pass(t, ar, tb, threads, &pb, &wb);
    println!(
        "check-groups6 over {} (group, third pick) items, {} states (per state {t_state:.1}s, group-major {t_group:.1}s): {}",
        items.len(), states.len(), compare_slot::< N > (t, & wa.into_inner().unwrap(), &
        wb.into_inner().unwrap(), 6)
    );
    let g = FAST_GAP.lock().unwrap();
    if g.1 > 0 {
        println!(
            "integrate6_fast against integrate6 over {} calls: worst relative gap {:.2e}",
            g.1, g.0
        );
    }
    if fast {
        six::report();
    }
}
/// Slot 6 two ways over N (size-2 state, third pick) pairs: steps 3–5's
/// splits integrated jointly ([`fold6_after`]), and step 3's enumerated into
/// realized size-3 states, each folded by the slot-5 machinery one level
/// deeper — steps 4 and 5 integrated ([`fold5_after`]), already checked. The
/// third picks taken split step 3 into 2–8 realized states: a real split, and
/// light enough to enumerate. The slot-6 cells must agree to rounding.
fn check_analytic6<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    level: &[(Key, Deco<N>)],
    n: usize,
    threads: usize,
    mp: &MultiProgress,
) {
    let mut sc = Scratch::default();
    let mut sample: Vec<(usize, usize)> = Vec::new();
    let mut realized = 0u64;
    'states: for (si, (k, d)) in level
        .iter()
        .enumerate()
        .step_by((level.len() / (n * 4)).max(1))
    {
        let mut picks: Vec<(usize, u64)> = Vec::new();
        let mut index = 0;
        walk(
            t,
            ar,
            k,
            d,
            None,
            &mut sc,
            &mut |step, live| {
                let mut count = 0u64;
                split_children::<
                    N,
                >(t, step.child(k), step.base, live, &mut |_, _| count += 1);
                picks.push((index, count));
                index += 1;
            },
        );
        for &(pi, count) in picks
            .iter()
            .filter(|p| (2..=8).contains(&p.1))
            .step_by(37)
            .take(2)
        {
            realized += count;
            sample.push((si, pi));
            if sample.len() == n {
                break 'states;
            }
        }
    }
    let on_pick = |
        si: usize,
        pi: usize,
        sc: &mut Scratch,
        f: &mut dyn FnMut(&Key, &Step<N>, &[Live])|
    {
        let (k, d) = &level[si];
        let mut index = 0;
        walk(
            t,
            ar,
            k,
            d,
            None,
            sc,
            &mut |step, live| {
                if index == pi {
                    f(k, &step, live);
                }
                index += 1;
            },
        );
    };
    let (wa, wb) = (
        Mutex::new(Weights::<N>::default()),
        Mutex::new(Weights::<N>::default()),
    );
    let (ta, tb): (Sharded<Ctx, KDeco<N>>, Sharded<Ctx, KDeco<N>>) = (
        Sharded::new(4096),
        Sharded::new(4096),
    );
    let pb = bar(mp, sample.len() as u64, "integrated");
    let t0 = Instant::now();
    parallel::<
        _,
        N,
    >(
        &sample,
        threads,
        &pb,
        &wa,
        |&(si, pi), l| {
            let Local { fold, sc, sc3, sc4, fold_sc, quad, .. } = l;
            on_pick(
                si,
                pi,
                sc,
                &mut |k, step, live| fold6_after(
                    t,
                    ar,
                    k,
                    step,
                    live,
                    sc3,
                    sc4,
                    fold_sc,
                    quad,
                    &ta,
                    fold,
                ),
            );
        },
    );
    let t_int = t0.elapsed().as_secs_f64();
    let pb = bar(mp, sample.len() as u64, "enumerated");
    let t0 = Instant::now();
    parallel::<
        _,
        N,
    >(
        &sample,
        threads,
        &pb,
        &wb,
        |&(si, pi), l| {
            let Local { fold, sc, sc3, sc4, fold_sc, polys, .. } = l;
            on_pick(
                si,
                pi,
                sc,
                &mut |k, step, live| {
                    emit_children(
                        t,
                        k,
                        step,
                        live,
                        &mut |ck, cd| {
                            walk(
                                t,
                                ar,
                                &ck,
                                &cd,
                                None,
                                sc3,
                                &mut |step4, live3| fold5_after(
                                    t,
                                    ar,
                                    &ck,
                                    &step4,
                                    live3,
                                    sc4,
                                    fold_sc,
                                    polys,
                                    &tb,
                                    fold,
                                ),
                            );
                        },
                    );
                },
            );
        },
    );
    let t_enum = t0.elapsed().as_secs_f64();
    let pb = bar(mp, 0, "tables");
    table_pass(t, ar, ta, threads, &pb, &wa);
    table_pass(t, ar, tb, threads, &pb, &wb);
    println!(
        "check-analytic6 over {} (size-2 state, third pick) pairs ({realized} realized size-3 states; integrated \
         {t_int:.1}s, enumerated {t_enum:.1}s): {}",
        sample.len(), compare_slot::< N > (t, & wa.into_inner().unwrap(), & wb
        .into_inner().unwrap(), 6)
    );
}
/// How two runs' cells of party slot `slot` agree.
fn compare_slot<N: Num>(
    t: &Tables,
    wa: &Weights<N>,
    wb: &Weights<N>,
    slot: u8,
) -> String {
    let (mut agree, mut cells, mut worst) = (0usize, 0usize, String::new());
    let (mut sa, mut sb) = (N::default(), N::default());
    let mut spread = String::new();
    for key in wa
        .keys()
        .chain(wb.keys().filter(|k| !wa.contains_key(*k)))
        .filter(|k| k.1 == slot)
    {
        let a = wa.get(key).map_or(N::default(), |&x| N::get(x));
        let b = wb.get(key).map_or(N::default(), |&x| N::get(x));
        sa = sa + a;
        sb = sb + b;
        cells += 1;
        if N::same(a, b) {
            agree += 1;
        } else if worst.is_empty() {
            worst = format!(
                "first disagreement: {} lead {} {}: {} vs {}", t.sp[key.0 as usize].name,
                key.2, code_label(key.3), a.show(), b.show()
            );
        }
    }
    if N::QUADRATURE {
        let (mut rel, mut abs) = (0.0f64, 0.0f64);
        for key in wa
            .keys()
            .chain(wb.keys().filter(|k| !wa.contains_key(*k)))
            .filter(|k| k.1 == slot)
        {
            let a = wa.get(key).map_or(0.0, |&x| N::to_f64(N::get(x)));
            let b = wb.get(key).map_or(0.0, |&x| N::to_f64(N::get(x)));
            abs = abs.max((a - b).abs());
            if a.abs().max(b.abs()) > 1e-17 {
                rel = rel.max((a - b).abs() / a.abs().max(b.abs()));
            }
        }
        spread = format!(
            " Worst relative gap {rel:.1e} (cells above 1e-17), largest absolute {abs:.1e}."
        );
    }
    format!(
        "{agree} of {cells} slot-{slot} cells agree; slot-{slot} mass {} vs {} ({}).{spread} {worst}",
        sa.show(), sb.show(), if N::same(sa, sb) { "same" } else { "DIFFERENT" }
    )
}
/// Slot 5 per state ([`fold5_after`]) and group-major ([`slot5_groups`])
/// over the same states: the cells must agree to rounding.
fn check_groups<N: Num>(
    t: &Tables,
    ar: &Arith<N>,
    level: &[(Key, Deco<N>)],
    threads: usize,
    mp: &MultiProgress,
    grid: bool,
    profile: bool,
) {
    let (wa, wb) = (
        Mutex::new(Weights::<N>::default()),
        Mutex::new(Weights::<N>::default()),
    );
    let (ta, tb): (Sharded<Ctx, KDeco<N>>, Sharded<Ctx, KDeco<N>>) = (
        Sharded::new(4096),
        Sharded::new(4096),
    );
    let pb = bar(mp, level.len() as u64, "per state");
    let t0 = Instant::now();
    parallel::<
        _,
        N,
    >(
        level,
        threads,
        &pb,
        &wa,
        |(k, d), l| {
            let Local { fold, sc, sc3, sc4, polys, .. } = l;
            walk(
                t,
                ar,
                k,
                d,
                None,
                sc,
                &mut |step, live| fold5_after(
                    t,
                    ar,
                    k,
                    &step,
                    live,
                    sc3,
                    sc4,
                    polys,
                    &ta,
                    fold,
                ),
            );
        },
    );
    let t_state = t0.elapsed().as_secs_f64();
    let pb = bar(mp, 0, "group-major");
    let t0 = Instant::now();
    slot5_groups(t, ar, level, threads, &pb, &tb, &wb, grid, profile);
    let t_group = t0.elapsed().as_secs_f64();
    let pb = bar(mp, 0, "tables");
    table_pass(t, ar, ta, threads, &pb, &wa);
    table_pass(t, ar, tb, threads, &pb, &wb);
    println!(
        "check-groups over {} size-2 states in {} groups (per state {t_state:.1}s, group-major {t_group:.1}s): {}",
        level.len(), label_groups(level).len(), compare_slot::< N > (t, & wa.into_inner()
        .unwrap(), & wb.into_inner().unwrap(), 5)
    );
}
/// `--gpu`: slot 5's per-class picks go to the GPU, in f32 pairs (~48 bits).
/// f64 would not keep up even at its peak.
#[cfg(feature = "gpu")]
static GPU: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
fn main() {
    if std::env::var_os("CUDA_DEVICE_MAX_CONNECTIONS").is_none() {
        unsafe { std::env::set_var("CUDA_DEVICE_MAX_CONNECTIONS", "32") };
    }
    six::OLD_F.store(std::env::var_os("TG_SIX_OLD_F").is_some(), Ordering::Relaxed);
    let args = parse_args();
    PAUSE.set(format!("{}.pause", args.out).into()).unwrap();
    if args.gpu {
        #[cfg(feature = "gpu")]
        {
            GPU.store(true, Ordering::Relaxed);
            let held = args.threads * (six::depth() + 1);
            assert!(
                held <= gpu::six_slots(),
                "--threads {} × (TG_SIX_DEPTH {} + 1) past the GPU's {} slots", args
                .threads, six::depth(), gpu::six_slots()
            );
        }
        #[cfg(not(feature = "gpu"))]
        panic!(
            "--gpu: built without the gpu feature (cargo build --release --features gpu)"
        );
    }
    match args.exact {
        Some(batch) => {
            Field::init(batch);
            println!(
                "exact batch {batch}: residues modulo {LANES} primes, {} down to {}",
                field().primes[0].p, field().primes[LANES - 1].p
            );
            run::<Fp>(&args);
        }
        None => run::<f64>(&args),
    }
}
fn run<N: Num>(args: &Args) {
    let started = Instant::now();
    let t = build_tables();
    let simple_members: usize = t
        .simple_classes
        .iter()
        .map(|&c| t.classes[c as usize].members.len())
        .sum();
    println!(
        "{} species in {} bases; {} classes ({} simple classes holding {simple_members} species); {} threads",
        t.sp.len(), t.bases.len(), t.classes.len(), t.simple_classes.len(), args.threads
    );
    if args.partial_stats > 0 {
        partial_stats(&t, args.partial_stats);
        return;
    }
    if args.mc_dump > 0 {
        mc_dump(&t, args.mc_dump, args.mc_seed, args.mc_rb, args.threads, &args.out);
        return;
    }
    if args.self_check > 0 {
        let (_, bad) = sample(&t, args.self_check, true, args.threads);
        assert_eq!(
            bad, 0, "fast admissibility disagrees with TeamGenState::admits {bad} times"
        );
        println!(
            "self-check: fast admissibility matches TeamGenState::admits over {} teams",
            args.self_check
        );
    }
    let ar = Arith::<N>::new(&t);
    let mp = MultiProgress::new();
    let weights = Mutex::new(Weights::<N>::default());
    let complete: Vec<u8> = if args.slots {
        run_slots(&t, &ar, args, &mp, &weights, started);
        let mut done = vec![1, 2, 3, 4];
        if args.slot5 {
            done.push(5);
        }
        if args.slot6 {
            done.push(6);
        }
        done
    } else {
        let mut level: Vec<(Key, Deco<N>)> = vec![
            (Key::empty(), vec![(0, N::ratio(1, 1))])
        ];
        for _ in 0..args.max_size {
            level = grow(&t, &ar, &level, args, &mp, &weights, started);
        }
        (2..=args.max_size).collect()
    };
    let weights = weights.into_inner().unwrap();
    let mut by_slot = vec![N::Acc::default(); 8];
    for (&(_, slot, _, _), &x) in &weights {
        by_slot[slot as usize] += x;
    }
    println!(
        "done in {:.0}s; mass by slot: {}", started.elapsed().as_secs_f64(), complete
        .iter().map(|& s | format!("{s}: {}", N::get(by_slot[s as usize]).show()))
        .collect::< Vec < _ >> ().join(", ")
    );
    N::write(&t, &weights, &complete, &args.out);
    println!("wrote {}", args.out);
    if args.check_mc > 0 {
        N::check_mc(&t, &weights, &complete, args.check_mc, args.threads);
    }
}
/// The weights of the complete slots as rows `[species, slot, lead,
/// context, weight]`. A context is a `teamDetails` code: bit i the i-th of
/// `contextFlags`, bits 12–13 the Spikes count (the generator reads it only
/// as `>= 2`, so 2 stands for more).
fn write_rows(t: &Tables, weights: &Weights<f64>, slots: &[u8], path: &str) {
    use serde_json::{Map, Value};
    let mut rows: Vec<(&str, u8, bool, u16, f64)> = weights
        .iter()
        .filter(|(k, _)| slots.contains(&k.1))
        .map(|(&(s, slot, lead, code), &x)| (
            t.sp[s as usize].name,
            slot,
            lead,
            code,
            f64::get(x),
        ))
        .collect();
    rows.sort_by(|a, b| (a.0, a.1, a.2, a.3).cmp(&(b.0, b.1, b.2, b.3)));
    let negative = rows.iter().filter(|r| r.4 < 0.0).count();
    let smallest = rows
        .iter()
        .map(|r| r.4)
        .filter(|&x| x > 0.0)
        .fold(f64::INFINITY, f64::min);
    println!(
        "{} cells; smallest {smallest:e}; {negative} negative (cancellation residue)",
        rows.len()
    );
    let mut root = Map::new();
    root.insert(
        "_about".into(),
        Value::from(
            "P(species in party slot k, as lead or not, under each teamDetails context), computed exactly from the PS-exact team loop. Slot 1 is the lead; the others follow in generation order.",
        ),
    );
    root.insert("contextFlags".into(), Value::from(CONTEXT_FLAGS.to_vec()));
    root.insert(
        "rows".into(),
        Value::Array(
            rows
                .into_iter()
                .map(|(name, slot, lead, code, x)| {
                    Value::Array(
                        vec![
                            Value::from(name), Value::from(slot), Value::from(lead),
                            Value::from(code), Value::from(x)
                        ],
                    )
                })
                .collect(),
        ),
    );
    std::fs::write(path, serde_json::to_string(&Value::Object(root)).unwrap()).unwrap();
}
/// `--mc-dump TEAMS`: TEAMS teams from the port's loop (`sampled_team`),
/// every member counted by (species, party slot, lead, context), as rows
/// `[species, slot, lead, context, count]` beside the team count — the
/// sampled twin of [`write_rows`], slot 6 included. `--mc-seed` picks the
/// streams, so runs of different seeds add.
///
/// `--mc-rb K`: the K-th member Rao–Blackwellized ([`sampled_rb`]) and
/// only its cells written, each an expected count (a float).
fn mc_dump(t: &Tables, teams: u64, seed: u64, rb: usize, threads: usize, path: &str) {
    use serde_json::{Map, Value};
    let t0 = Instant::now();
    let counts: FxMap<(u16, u8, bool, u16), f64> = if rb > 0 {
        sample_rb(t, teams, threads, seed, rb)
    } else {
        sample_seeded(t, teams, false, threads, seed)
            .0
            .into_iter()
            .map(|(k, n)| (k, n as f64))
            .collect()
    };
    let teams = teams / threads as u64 * threads as u64;
    let secs = t0.elapsed().as_secs_f64();
    let mut rows: Vec<(&str, u8, bool, u16, f64)> = counts
        .iter()
        .map(|(&(s, slot, lead, code), &n)| (t.sp[s as usize].name, slot, lead, code, n))
        .collect();
    rows.sort_by(|a, b| (a.0, a.1, a.2, a.3).cmp(&(b.0, b.1, b.2, b.3)));
    println!(
        "mc-dump: {teams} teams (seed {seed}, rb {rb}) in {secs:.0}s, {:.0} teams/s; {} cells",
        teams as f64 / secs, rows.len()
    );
    let mut root = Map::new();
    root.insert(
        "_about".into(),
        Value::from(
            if rb > 0 {
                "Expected counts of (species in party slot k, as lead or not, under each teamDetails context) over `teams` \
             teams from the PS-exact port's team loop, the `rb`-th member Rao-Blackwellized: its distribution given \
             the team before it and one draw of the remaining entries' forme labels, the earlier members sampled \
             (teamgen --mc-dump --mc-rb). Only that member's cells. The streams are per thread: `threads` and `seed` \
             reproduce them."
            } else {
                "Counts of (species in party slot k, as lead or not, under each teamDetails context) over `teams` teams \
             sampled from the PS-exact port's team loop (teamgen --mc-dump). Slots as in slot-weights.json."
            },
        ),
    );
    root.insert("teams".into(), Value::from(teams));
    root.insert("seed".into(), Value::from(seed));
    root.insert("threads".into(), Value::from(threads));
    root.insert("rb".into(), Value::from(rb));
    root.insert("contextFlags".into(), Value::from(CONTEXT_FLAGS.to_vec()));
    root.insert(
        "rows".into(),
        Value::Array(
            rows
                .into_iter()
                .map(|(name, slot, lead, code, n)| Value::Array(
                    vec![
                        Value::from(name), Value::from(slot), Value::from(lead),
                        Value::from(code), Value::from(n)
                    ],
                ))
                .collect(),
        ),
    );
    std::fs::write(path, serde_json::to_string(&Value::Object(root)).unwrap()).unwrap();
}
/// One exact batch's weights, from which the rationals are rebuilt.
/// Little-endian: `EXACTW1\0`, the lane count and its primes, the species
/// names (u16 length, UTF-8), then the rows of the complete slots: species
/// index u16, slot u8, lead u8, context code u16 (as in `write_rows`), and
/// each lane's residue u64.
fn write_residues(t: &Tables, weights: &Weights<Fp>, slots: &[u8], path: &str) {
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"EXACTW1\0");
    out.extend_from_slice(&(LANES as u32).to_le_bytes());
    for pr in &field().primes {
        out.extend_from_slice(&pr.p.to_le_bytes());
    }
    out.extend_from_slice(&(t.sp.len() as u32).to_le_bytes());
    for s in &t.sp {
        out.extend_from_slice(&(s.name.len() as u16).to_le_bytes());
        out.extend_from_slice(s.name.as_bytes());
    }
    let mut rows: Vec<(&(u16, u8, bool, u16), &Fp)> = weights
        .iter()
        .filter(|(k, _)| slots.contains(&k.1))
        .collect();
    rows.sort_by_key(|r| *r.0);
    out.extend_from_slice(&(rows.len() as u64).to_le_bytes());
    let zero = rows.iter().filter(|r| *r.1 == Fp::default()).count();
    for (&(s, slot, lead, code), x) in rows {
        out.extend_from_slice(&s.to_le_bytes());
        out.push(slot);
        out.push(u8::from(lead));
        out.extend_from_slice(&code.to_le_bytes());
        for r in x.canonical() {
            out.extend_from_slice(&r.to_le_bytes());
        }
    }
    println!(
        "{} cells, {zero} exactly zero in every lane", weights.iter().filter(| (k, _) |
        slots.contains(& k.1)).count()
    );
    std::fs::write(path, out).unwrap();
}
/// Two-sided p-value of `n` under Poisson(`e`): exact sums below a mean of
/// 200, the normal approximation above.
fn poisson_two_sided(n: u64, e: f64) -> f64 {
    if e <= 0.0 {
        return if n == 0 { 1.0 } else { 0.0 };
    }
    if e > 200.0 {
        let z = (n as f64 - e).abs() / e.sqrt();
        return (2.0 * 0.5 * libm_erfc(z / std::f64::consts::SQRT_2)).min(1.0);
    }
    let ln_pmf = |k: u64| -> f64 {
        k as f64 * e.ln() - e - (1..=k).map(|i| (i as f64).ln()).sum::<f64>()
    };
    let base = ln_pmf(n);
    let (mut lower, mut upper) = (0.0f64, 0.0f64);
    let mut lp = base;
    for k in (0..=n).rev() {
        lower += lp.exp();
        if k > 0 {
            lp += (k as f64).ln() - e.ln();
        }
    }
    lp = base;
    let mut k = n;
    loop {
        upper += lp.exp();
        k += 1;
        lp += e.ln() - (k as f64).ln();
        if lp < base - 40.0 && k as f64 > e {
            break;
        }
    }
    (2.0 * lower.min(upper)).min(1.0)
}
/// erfc by a rational approximation (Abramowitz–Stegun 7.1.26), enough for
/// a p-value's order of magnitude.
fn libm_erfc(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t
                * (-0.284496736
                    + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    poly * (-x * x).exp()
}
/// The computed weights against `teams` sampled teams of the port, per
/// (species, slot, lead, context) cell.
fn check_mc(
    t: &Tables,
    weights: &Weights<f64>,
    slots: &[u8],
    teams: u64,
    threads: usize,
) {
    let (counts, _) = sample(t, teams, false, threads);
    let teams = teams / threads as u64 * threads as u64;
    let mut cells: FxMap<(u16, u8, bool, u16), (f64, u64)> = FxMap::default();
    for (&k, &x) in weights {
        if slots.contains(&k.1) {
            cells.entry(k).or_default().0 += f64::get(x);
        }
    }
    for (&k, &n) in &counts {
        if slots.contains(&k.1) {
            cells.entry(k).or_default().1 += n;
        }
    }
    let mut zs: Vec<(f64, String)> = cells
        .iter()
        .map(|(&(s, slot, lead, code), &(p, n))| {
            let e = p * teams as f64;
            let sd = (e * (1.0 - p)).max(1.0).sqrt();
            (
                (n as f64 - e) / sd,
                format!(
                    "{} slot {slot} {} {}: exact {e:.1} sampled {n}", t.sp[s as usize]
                    .name, if lead { "lead" } else { "nonlead" }, code_label(code)
                ),
            )
        })
        .collect();
    zs.sort_by(|a, b| b.0.abs().total_cmp(&a.0.abs()));
    let mut ps: Vec<(f64, String)> = cells
        .iter()
        .map(|(&(s, slot, lead, code), &(p, n))| {
            let e = p * teams as f64;
            (
                poisson_two_sided(n, e),
                format!(
                    "{} slot {slot} {} {}: exact {e:.1} sampled {n}", t.sp[s as usize]
                    .name, if lead { "lead" } else { "nonlead" }, code_label(code)
                ),
            )
        })
        .collect();
    ps.sort_by(|a, b| a.0.total_cmp(&b.0));
    let bonferroni = ps.first().map_or(1.0, |p| (p.0 * ps.len() as f64).min(1.0));
    println!(
        "check against {teams} sampled teams: {} cells; smallest p {:.2e}, Bonferroni {bonferroni:.3e} \
         (above ~0.01 is what a correct table looks like); max normal |z| {:.2}",
        ps.len(), ps.first().map_or(1.0, | p | p.0), zs.first().map_or(0.0, | z | z.0
        .abs()),
    );
    for (p, label) in ps.iter().take(8) {
        println!("  p {p:.2e}  {label}");
    }
    let mut agg: FxMap<(u8, u16), (f64, u64)> = FxMap::default();
    for (&(_, slot, _, code), &(p, n)) in &cells {
        let e = agg.entry((slot, code)).or_default();
        e.0 += p;
        e.1 += n;
    }
    let mut az: Vec<(f64, String)> = agg
        .iter()
        .map(|(&(slot, code), &(p, n))| {
            let e = p * teams as f64;
            (
                (n as f64 - e) / (e * (1.0 - p)).max(1.0).sqrt(),
                format!("slot {slot} {}: exact {e:.1} sampled {n}", code_label(code)),
            )
        })
        .collect();
    az.sort_by(|a, b| b.0.abs().total_cmp(&a.0.abs()));
    println!(
        "by (slot, context), summed over species: {} cells, max |z| {:.2}", az.len(), az
        .first().map_or(0.0, | z | z.0.abs())
    );
    for (z, label) in az.iter().take(8) {
        println!("  z {z:+.2}  {label}");
    }
}
