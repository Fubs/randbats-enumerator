//! Exact enumeration of [`random_set`](super::random_set)'s outcome
//! distribution.
//!
//! The generator is procedural, but every random decision it makes is a
//! uniform pick from a short list or a `num/denom` chance — all funneled
//! through [`Decide`]. So the joint distribution over complete sets is
//! computable exactly: walk every branch of the decision tree depth-first,
//! carrying the product of branch probabilities, and merge leaves that
//! produce the same set. Each leaf probability is a product of exact
//! rationals (`1/n`, `num/den`); f64 keeps ~15 significant digits through
//! the handful of factors a path has, so the merged masses are the exact
//! values that statistics from sampled teams estimate.
//! [`enumerate_random_set_exact`] hands each path's factors to the caller
//! for arithmetic of its own. MATH.md §9 records the check against PS's own
//! `randomSet`, enumerated the same way.
//!
//! Cosmetic decisions (gender, shininess, cosmetic formes, move-slot
//! order) do not branch: they leave a set's identity unchanged, and
//! multiplying the tree by them would change nothing but the run time.
//! The returned sets carry a fixed representative for each (majority
//! gender, not shiny, culled move order).
use crate::generated::species_id::SpeciesId;
use crate::pokemon::Gender;
use super::set::{RandomSet, random_set_decide};
use super::team::TeamDetails;
use super::{Decide, MoveId};
/// One replayed decision along the current path.
#[derive(Copy, Clone)]
struct BranchPoint {
    /// Which option this path takes.
    chosen: u32,
    /// How many options the decision has (2 for a chance).
    options: u32,
    /// Probability of `chosen`: uniform `1/options`, or the chance's
    /// `num/denom` split.
    prob: f64,
    /// `Some((num, denom))` for a chance decision — needed to recompute
    /// `prob` when the odometer advances `chosen`.
    chance: Option<(u32, u32)>,
}
/// [`Decide`] that replays a recorded choice prefix and extends it with
/// first options, so a driver loop can walk the whole tree odometer-style.
struct PathDecide {
    path: Vec<BranchPoint>,
    /// Decisions consumed so far in the current run.
    depth: usize,
    /// Product of branch probabilities along the current path.
    prob: f64,
}
impl PathDecide {
    fn new() -> Self {
        Self {
            path: Vec::new(),
            depth: 0,
            prob: 1.0,
        }
    }
    /// Reset for the next run over the (already-advanced) path.
    fn rewind(&mut self) {
        self.depth = 0;
        self.prob = 1.0;
    }
    /// Replay-or-record one decision. `options` must match what the
    /// recorded run saw — the generator is deterministic given its
    /// decisions, so a mismatch is a replay bug, not data.
    fn step(
        &mut self,
        options: u32,
        prob_of: impl Fn(u32) -> f64,
        chance: Option<(u32, u32)>,
    ) -> u32 {
        let bp = if self.depth < self.path.len() {
            let bp = self.path[self.depth];
            debug_assert_eq!(
                bp.options, options, "decision tree replay diverged at depth {}", self
                .depth
            );
            bp
        } else {
            let bp = BranchPoint {
                chosen: 0,
                options,
                prob: prob_of(0),
                chance,
            };
            self.path.push(bp);
            bp
        };
        self.depth += 1;
        self.prob *= bp.prob;
        bp.chosen
    }
    /// The current path's branch probabilities as exact `(num, den)`.
    fn factors(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.path[..self.depth]
            .iter()
            .map(|bp| match bp.chance {
                Some((num, den)) if bp.chosen == 0 => (num, den),
                Some((num, den)) => (den - num, den),
                None => (1, bp.options),
            })
    }
    /// Advance the path odometer to the next unexplored branch.
    /// Returns false when the tree is exhausted.
    fn advance(&mut self) -> bool {
        debug_assert_eq!(self.depth, self.path.len(), "run ended mid-path");
        while let Some(last) = self.path.last_mut() {
            if last.chosen + 1 < last.options {
                last.chosen += 1;
                last.prob = match last.chance {
                    Some((num, den)) => {
                        debug_assert_eq!(last.chosen, 1);
                        f64::from(den - num) / f64::from(den)
                    }
                    None => 1.0 / f64::from(last.options),
                };
                return true;
            }
            self.path.pop();
        }
        false
    }
}
impl Decide for PathDecide {
    fn index(&mut self, n: usize) -> usize {
        if n <= 1 {
            return 0;
        }
        self.step(n as u32, |_| 1.0 / n as f64, None) as usize
    }
    fn chance(&mut self, num: u32, denom: u32) -> bool {
        if num == 0 {
            return false;
        }
        if num >= denom {
            return true;
        }
        self
            .step(
                2,
                |c| {
                    if c == 0 {
                        f64::from(num) / f64::from(denom)
                    } else {
                        f64::from(denom - num) / f64::from(denom)
                    }
                },
                Some((num, denom)),
            ) == 0
    }
    fn gender(&mut self, male_eighths: u8) -> Gender {
        if male_eighths >= 4 { Gender::Male } else { Gender::Female }
    }
    fn cosmetic_chance(&mut self, _num: u32, _denom: u32) -> bool {
        false
    }
    fn cosmetic_index(&mut self, _n: usize) -> usize {
        0
    }
    fn shuffle_moves(&mut self, _moves: &mut [MoveId]) {}
}
/// A distinct complete set the generator can produce, with its exact
/// probability. Probabilities over a call's return sum to 1.
#[derive(Clone, Debug)]
pub struct SetOutcome {
    pub set: RandomSet,
    pub prob: f64,
}
/// Identity for merging leaves: the moves (unordered), species, item,
/// ability, Tera type and role. EVs and IVs follow from these, and gender,
/// shininess and move order are cosmetic.
fn outcome_key(set: &RandomSet) -> ([u16; 4], u16, u16, u16, u8, u8) {
    let mut moves: [u16; 4] = set.moves.map(|m| m.map_or(u16::MAX, |m| m as u16));
    moves.sort_unstable();
    (
        moves,
        set.species as u16,
        set.item as u16,
        set.ability as u16,
        set.tera_type as u8,
        set.role.map_or(u8::MAX, |r| r as u8),
    )
}
/// Every set `random_set(species, team_details, is_lead)` can produce,
/// with exact probabilities, merged over cosmetic differences and
/// sorted by [`outcome_key`] so identical inputs give identical tables.
pub fn enumerate_random_set(
    species: SpeciesId,
    team_details: &TeamDetails,
    is_lead: bool,
) -> Vec<SetOutcome> {
    let mut d = PathDecide::new();
    let mut merged: std::collections::HashMap<
        ([u16; 4], u16, u16, u16, u8, u8),
        SetOutcome,
    > = std::collections::HashMap::new();
    loop {
        let set = random_set_decide(&mut d, species, team_details, is_lead, false);
        let key = outcome_key(&set);
        merged
            .entry(key)
            .and_modify(|o| o.prob += d.prob)
            .or_insert(SetOutcome { set, prob: d.prob });
        if !d.advance() {
            break;
        }
        d.rewind();
    }
    let mut out: Vec<(([u16; 4], u16, u16, u16, u8, u8), SetOutcome)> = merged
        .into_iter()
        .collect();
    out.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
    out.into_iter().map(|(_, o)| o).collect()
}
/// [`enumerate_random_set`] with each set's probability in the caller's
/// exact arithmetic: `leaf` turns one path's branch probabilities, as
/// `(num, den)` factors, into a `T`, and `add` merges paths that produce
/// the same set. Same sets, same order.
pub fn enumerate_random_set_exact<T>(
    species: SpeciesId,
    team_details: &TeamDetails,
    is_lead: bool,
    leaf: impl Fn(&mut dyn Iterator<Item = (u32, u32)>) -> T,
    add: impl Fn(&mut T, T),
) -> Vec<(RandomSet, T)> {
    let mut d = PathDecide::new();
    let mut merged: std::collections::HashMap<
        ([u16; 4], u16, u16, u16, u8, u8),
        (RandomSet, T),
    > = std::collections::HashMap::new();
    loop {
        let set = random_set_decide(&mut d, species, team_details, is_lead, false);
        let p = leaf(&mut d.factors());
        match merged.entry(outcome_key(&set)) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                add(&mut e.get_mut().1, p)
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert((set, p));
            }
        }
        if !d.advance() {
            break;
        }
        d.rewind();
    }
    let mut out: Vec<_> = merged.into_iter().collect();
    out.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
    out.into_iter().map(|(_, o)| o).collect()
}
