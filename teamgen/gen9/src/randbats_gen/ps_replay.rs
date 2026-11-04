//! PS's own draws, replayed through the port: the team-level parity check.
//! `trace.js` records every `PRNG.random` call PS's `randomTeam` makes and
//! the team it built. Fed back through [`PsDraws`], the port must ask for
//! the same range at every draw, in the same order, and build the same
//! team, so a run of traces checks the team loop and every set behind it
//! at once, under the contexts real teams reach (MATH.md §9).
use crate::to_id;
use crate::pokemon::Gender;
use super::set::RandomSet;
use super::team::generate_team_decide;
use super::{Decide, MoveId};
/// One `PRNG.random` call and what it returned, as the trace writes it:
/// `[from, to, value]` for `random(from, to)`, `[n, null, value]` for
/// `random(n)`.
#[derive(Clone, Copy, Debug, serde::Deserialize)]
pub struct PsDraw(pub u32, pub Option<u32>, pub u32);
/// A set as PS's `randomSet` returns it, in the fields the port rolls.
#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PsSet {
    pub species_id: String,
    pub gender: String,
    pub shiny: bool,
    pub level: u8,
    pub moves: Vec<String>,
    pub ability: String,
    pub evs: PsStats,
    pub ivs: PsStats,
    pub item: String,
    pub tera_type: String,
    pub role: String,
}
/// A set's EVs or IVs as PS writes them.
#[derive(Debug, serde::Deserialize)]
pub struct PsStats {
    pub hp: u8,
    pub atk: u8,
    pub def: u8,
    pub spa: u8,
    pub spd: u8,
    pub spe: u8,
}
/// [`Decide`] that answers from PS's recorded draws, and stops at the
/// first one whose range the port did not ask for.
struct PsDraws<'a> {
    draws: &'a [PsDraw],
    at: usize,
    error: Option<String>,
    /// Each ask in order, when [`ps_replay_log`] wants them.
    log: Option<Vec<String>>,
}
impl PsDraws<'_> {
    /// The next draw, which must be PS's `random(range)` — or, with
    /// `from`, its `random(from, from + range)`.
    #[track_caller]
    fn take(&mut self, what: &str, from: Option<u32>, range: u32) -> u32 {
        let caller = std::panic::Location::caller();
        let what = &format!(
            "{what} @{}:{}", caller.file().rsplit('/').next().unwrap_or(""), caller
            .line()
        );
        if self.error.is_some() {
            return 0;
        }
        let asked = match from {
            None => format!("random({range})"),
            Some(f) => format!("random({f}, {})", f + range),
        };
        let Some(&PsDraw(f, to, value)) = self.draws.get(self.at) else {
            self.error = Some(
                format!(
                    "draw {}: the port asked {what} {asked}; PS drew no more", self.at
                ),
            );
            return 0;
        };
        let matches = match (from, to) {
            (None, None) => f == range,
            (Some(from), Some(to)) => f == from && to == from + range,
            _ => false,
        };
        if !matches {
            let drew = match to {
                None => format!("random({f})"),
                Some(to) => format!("random({f}, {to})"),
            };
            self.error = Some(
                format!(
                    "draw {}: the port asked {what} {asked}; PS drew {drew}", self.at
                ),
            );
            return 0;
        }
        if let Some(log) = &mut self.log {
            log.push(format!("{what} {asked} = {value}"));
        }
        self.at += 1;
        value - from.unwrap_or(0)
    }
}
impl Decide for PsDraws<'_> {
    #[track_caller]
    fn index(&mut self, n: usize) -> usize {
        self.take("an index", None, n as u32) as usize
    }
    #[track_caller]
    fn chance(&mut self, num: u32, denom: u32) -> bool {
        self.take("a chance", None, denom) < num
    }
    fn gender(&mut self, _male_eighths: u8) -> Gender {
        if self.take("a gender", None, 2) == 1 { Gender::Female } else { Gender::Male }
    }
    fn cosmetic_chance(&mut self, num: u32, denom: u32) -> bool {
        self.chance(num, denom)
    }
    fn cosmetic_index(&mut self, n: usize) -> usize {
        self.take("a forme", None, n as u32) as usize
    }
    fn shuffle_moves(&mut self, moves: &mut [MoveId]) {
        let len = moves.len() as u32;
        for start in 0..len.saturating_sub(1) {
            let next = start + self.take("a shuffle", Some(start), len - start);
            moves.swap(start as usize, next as usize);
        }
    }
}
/// The team PS's draws build through the port, or where the port first
/// asked for a draw PS did not make.
pub fn generate_team_from_ps_draws(draws: &[PsDraw]) -> Result<Vec<RandomSet>, String> {
    replay(draws, None).0
}
/// The port's asks in order, each with the PS draw that answered it, up to
/// the first divergence. Indexed like the draws after PS's first, the
/// Monotype type sample, which singles makes too.
pub fn ps_replay_log(draws: &[PsDraw]) -> (Vec<String>, Option<String>) {
    let (result, log) = replay(draws, Some(Vec::new()));
    (log.unwrap_or_default(), result.err())
}
fn replay(
    draws: &[PsDraw],
    log: Option<Vec<String>>,
) -> (Result<Vec<RandomSet>, String>, Option<Vec<String>>) {
    let Some((PsDraw(18, None, _), rest)) = draws.split_first() else {
        return (Err("PS's first draw is not its type sample, random(18)".into()), log);
    };
    let mut d = PsDraws {
        draws: rest,
        at: 0,
        error: None,
        log,
    };
    let team = generate_team_decide(&mut d);
    let result = if let Some(e) = d.error.take() {
        Err(e)
    } else if d.at != rest.len() {
        Err(format!("draw {}: the port stopped; PS drew {} in all", d.at, rest.len()))
    } else {
        Ok(team)
    };
    (result, d.log)
}
/// Every field where `ours` differs from the team PS built.
pub fn ps_team_mismatches(ours: &[RandomSet], ps: &[PsSet]) -> Vec<String> {
    if ours.len() != ps.len() {
        return vec![format!("{} members against PS's {}", ours.len(), ps.len())];
    }
    let mut out = Vec::new();
    for (i, (o, p)) in ours.iter().zip(ps).enumerate() {
        let mut field = |name: &str, ours: String, theirs: String| {
            if ours != theirs {
                out.push(
                    format!(
                        "slot {i} ({}) {name}: ours {ours}, PS {theirs}", p.species_id
                    ),
                );
            }
        };
        field("species", to_id(o.species.data().name), p.species_id.clone());
        field("level", o.level.to_string(), p.level.to_string());
        field("ability", to_id(o.ability.metadata().name), to_id(&p.ability));
        let item = if o.item == crate::generated::item_id::ItemId::None {
            String::new()
        } else {
            to_id(o.item.metadata().name)
        };
        field("item", item, to_id(&p.item));
        field("tera", format!("{:?}", o.tera_type), p.tera_type.clone());
        field(
            "role",
            o.role.map_or_else(String::new, |r| r.name().to_string()),
            p.role.clone(),
        );
        let moves: Vec<String> = o
            .moves
            .iter()
            .flatten()
            .map(|m| to_id(m.data().name))
            .collect();
        field("moves", moves.join(","), p.moves.join(","));
        let gender = match o.gender {
            Gender::Male => "M",
            Gender::Female => "F",
            Gender::Genderless => "N",
        };
        field("gender", gender.to_string(), p.gender.clone());
        field("shiny", o.shiny.to_string(), p.shiny.to_string());
        let spread = |s: [u8; 6]| format!("{s:?}");
        field(
            "evs",
            spread([o.evs.hp, o.evs.atk, o.evs.def, o.evs.spa, o.evs.spd, o.evs.spe]),
            spread([p.evs.hp, p.evs.atk, p.evs.def, p.evs.spa, p.evs.spd, p.evs.spe]),
        );
        field(
            "ivs",
            spread([o.ivs.hp, o.ivs.atk, o.ivs.def, o.ivs.spa, o.ivs.spd, o.ivs.spe]),
            spread([p.ivs.hp, p.ivs.atk, p.ivs.def, p.ivs.spa, p.ivs.spd, p.ivs.spe]),
        );
    }
    out
}
