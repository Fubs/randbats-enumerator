//! Pokémon Showdown's Gen 9 Random Battles team generator, ported draw for
//! draw from `data/random-battles/gen9/teams.ts`: the singles path of
//! `randomTeam` and everything it calls (`randomSet`, `randomMoveset`,
//! `cullMovePool`, `getAbility`, `getPriorityItem`, `getItem` and their
//! helpers). Branches only the doubles format can reach are left out; the
//! `is_doubles` parameters of PS's helpers are kept, and the team loop
//! passes `false`.
//!
//! Every random decision goes through one trait, so the same code serves
//! three callers:
//! - [`generate_team`] and [`random_set`] draw from a `rand` RNG. The
//!   draws have PS's distribution, but a seed does not give PS's team for
//!   that seed: PS's PRNG is not ported.
//! - [`generate_team_from_ps_draws`] answers each decision with the draw PS
//!   made, as `trace.js` records it, and must ask for the same range at
//!   every draw and build the same team.
//! - [`enumerate_random_set`] and [`enumerate_random_set_exact`] take every
//!   branch with its probability: the exact distribution of one species'
//!   set under a given `teamDetails`.
//!
//! The set data is [`crate::randbats`]. The generator reads its lists and
//! draws from them as PS does.
use rand::{Rng, RngExt};
use crate::generated::ability_id::AbilityId;
use crate::generated::move_id::MoveId;
use crate::generated::species_id::SpeciesId;
use crate::species::SpeciesTags;
use crate::types::{Effectiveness, Type};
mod ability;
mod counter;
mod enumerate;
mod items;
mod moveset;
mod ps_replay;
mod set;
mod team;
pub use enumerate::{SetOutcome, enumerate_random_set, enumerate_random_set_exact};
pub use ps_replay::{
    PsDraw, PsSet, PsStats, generate_team_from_ps_draws, ps_replay_log,
    ps_team_mismatches,
};
pub use set::{RandomSet, random_set};
pub use team::{
    TeamDetails, TeamGenState, formes_of, generate_team, never_leads, weighted_base_pool,
    weighted_base_runs,
};
/// Moves that restore HP.
pub(crate) const RECOVERY_MOVES: &[MoveId] = &[
    MoveId::Healorder,
    MoveId::Milkdrink,
    MoveId::Moonlight,
    MoveId::Morningsun,
    MoveId::Recover,
    MoveId::Roost,
    MoveId::Shoreup,
    MoveId::Slackoff,
    MoveId::Softboiled,
    MoveId::Strengthsap,
    MoveId::Synthesis,
];
/// Moves that drop stats (Contrary turns into a boost).
pub(crate) const CONTRARY_MOVES: &[MoveId] = &[
    MoveId::Armorcannon,
    MoveId::Closecombat,
    MoveId::Leafstorm,
    MoveId::Makeitrain,
    MoveId::Overheat,
    MoveId::Spinout,
    MoveId::Superpower,
    MoveId::Vcreate,
];
/// Moves that boost Attack.
pub(crate) const PHYSICAL_SETUP: &[MoveId] = &[
    MoveId::Bellydrum,
    MoveId::Bulkup,
    MoveId::Coil,
    MoveId::Curse,
    MoveId::Dragondance,
    MoveId::Honeclaws,
    MoveId::Howl,
    MoveId::Meditate,
    MoveId::Poweruppunch,
    MoveId::Swordsdance,
    MoveId::Tidyup,
    MoveId::Victorydance,
];
/// Moves that boost Special Attack.
pub(crate) const SPECIAL_SETUP: &[MoveId] = &[
    MoveId::Calmmind,
    MoveId::Chargebeam,
    MoveId::Geomancy,
    MoveId::Nastyplot,
    MoveId::Quiverdance,
    MoveId::Tailglow,
    MoveId::Takeheart,
    MoveId::Torchsong,
];
/// Moves that boost both Attack and Special Attack.
pub(crate) const MIXED_SETUP: &[MoveId] = &[
    MoveId::Clangoroussoul,
    MoveId::Growth,
    MoveId::Happyhour,
    MoveId::Holdhands,
    MoveId::Noretreat,
    MoveId::Shellsmash,
    MoveId::Workup,
];
/// Moves that primarily boost Speed.
pub(crate) const SPEED_SETUP: &[MoveId] = &[
    MoveId::Agility,
    MoveId::Autotomize,
    MoveId::Flamecharge,
    MoveId::Rockpolish,
    MoveId::Snowscape,
    MoveId::Trailblaze,
];
/// Conglomerate of every setup move (PS's SETUP).
pub(crate) const SETUP: &[MoveId] = &[
    MoveId::Acidarmor,
    MoveId::Agility,
    MoveId::Autotomize,
    MoveId::Bellydrum,
    MoveId::Bulkup,
    MoveId::Calmmind,
    MoveId::Clangoroussoul,
    MoveId::Coil,
    MoveId::Cosmicpower,
    MoveId::Curse,
    MoveId::Dragondance,
    MoveId::Flamecharge,
    MoveId::Growth,
    MoveId::Honeclaws,
    MoveId::Howl,
    MoveId::Irondefense,
    MoveId::Meditate,
    MoveId::Nastyplot,
    MoveId::Noretreat,
    MoveId::Poweruppunch,
    MoveId::Quiverdance,
    MoveId::Rockpolish,
    MoveId::Shellsmash,
    MoveId::Shelter,
    MoveId::Shiftgear,
    MoveId::Swordsdance,
    MoveId::Tailglow,
    MoveId::Takeheart,
    MoveId::Tidyup,
    MoveId::Trailblaze,
    MoveId::Workup,
    MoveId::Victorydance,
];
/// Speed-control moves (paralysis spreaders, Trick Room, Tailwind, etc.).
pub(crate) const SPEED_CONTROL: &[MoveId] = &[
    MoveId::Electroweb,
    MoveId::Glare,
    MoveId::Icywind,
    MoveId::Lowsweep,
    MoveId::Nuzzle,
    MoveId::Quash,
    MoveId::Tailwind,
    MoveId::Thunderwave,
    MoveId::Trickroom,
];
/// Moves that shouldn't be the only STAB on a set (chip / utility).
pub(crate) const NO_STAB: &[MoveId] = &[
    MoveId::Acidspray,
    MoveId::Accelerock,
    MoveId::Aquajet,
    MoveId::Bounce,
    MoveId::Breakingswipe,
    MoveId::Bulletpunch,
    MoveId::Chatter,
    MoveId::Chloroblast,
    MoveId::Clearsmog,
    MoveId::Covet,
    MoveId::Dragontail,
    MoveId::Doomdesire,
    MoveId::Electroweb,
    MoveId::Eruption,
    MoveId::Explosion,
    MoveId::Fakeout,
    MoveId::Feint,
    MoveId::Flamecharge,
    MoveId::Flipturn,
    MoveId::Futuresight,
    MoveId::Grassyglide,
    MoveId::Iceshard,
    MoveId::Icywind,
    MoveId::Incinerate,
    MoveId::Infestation,
    MoveId::Machpunch,
    MoveId::Meteorbeam,
    MoveId::Mortalspin,
    MoveId::Nuzzle,
    MoveId::Pluck,
    MoveId::Pursuit,
    MoveId::Quickattack,
    MoveId::Rapidspin,
    MoveId::Reversal,
    MoveId::Selfdestruct,
    MoveId::Shadowsneak,
    MoveId::Skydrop,
    MoveId::Snarl,
    MoveId::Strugglebug,
    MoveId::Suckerpunch,
    MoveId::Trailblaze,
    MoveId::Uturn,
    MoveId::Vacuumwave,
    MoveId::Voltswitch,
    MoveId::Watershuriken,
    MoveId::Waterspout,
];
/// Hazard-setting moves.
pub(crate) const HAZARDS: &[MoveId] = &[
    MoveId::Spikes,
    MoveId::Stealthrock,
    MoveId::Stickyweb,
    MoveId::Toxicspikes,
];
/// Protect and its variants.
///
/// Unused in singles: PS reads it for roles whose label contains
/// "Protect", which are all doubles roles. Kept as the ported copy of PS's
/// list.
#[allow(dead_code)]
pub(crate) const PROTECT_MOVES: &[MoveId] = &[
    MoveId::Banefulbunker,
    MoveId::Burningbulwark,
    MoveId::Protect,
    MoveId::Silktrap,
    MoveId::Spikyshield,
];
/// Moves that switch the user out.
pub(crate) const PIVOT_MOVES: &[MoveId] = &[
    MoveId::Chillyreception,
    MoveId::Flipturn,
    MoveId::Partingshot,
    MoveId::Shedtail,
    MoveId::Teleport,
    MoveId::Uturn,
    MoveId::Voltswitch,
];
/// Pairs of moves that should appear together when possible.
pub(crate) const MOVE_PAIRS: &[(MoveId, MoveId)] = &[
    (MoveId::Lightscreen, MoveId::Reflect),
    (MoveId::Sleeptalk, MoveId::Rest),
    (MoveId::Protect, MoveId::Wish),
    (MoveId::Leechseed, MoveId::Protect),
    (MoveId::Leechseed, MoveId::Substitute),
];
/// Pokémon that always want priority STAB and accept it as their only
/// STAB move of that type.
pub(crate) const PRIORITY_POKEMON: &[SpeciesId] = &[
    SpeciesId::Breloom,
    SpeciesId::Brutebonnet,
    SpeciesId::Cacturne,
    SpeciesId::Honchkrow,
    SpeciesId::Mimikyu,
    SpeciesId::Ragingbolt,
    SpeciesId::Scizor,
];
/// PS's `NO_LEAD_POKEMON`, by forme id: species never generated into the
/// lead slot in singles.
pub(crate) const NO_LEAD_SPECIES: &[SpeciesId] = &[
    SpeciesId::Dugtrio,
    SpeciesId::Gothitelle,
    SpeciesId::Ironthorns,
    SpeciesId::Kingambit,
    SpeciesId::Zacian,
    SpeciesId::Zaciancrowned,
    SpeciesId::Zamazenta,
    SpeciesId::Zamazentacrowned,
];
/// Defensive Tera Blast users (extra item rule in `getItem`).
pub(crate) const DEFENSIVE_TERA_BLAST_USERS: &[SpeciesId] = &[
    SpeciesId::Alcremie,
    SpeciesId::Bellossom,
    SpeciesId::Comfey,
    SpeciesId::Fezandipiti,
    SpeciesId::Florges,
];
/// Every random decision the generator makes. [`RngDecide`] draws from an
/// RNG, `ps_replay`'s `PsDraws` answers with PS's recorded draws, and
/// `enumerate`'s `PathDecide` walks every branch with its probability.
///
/// Cosmetic decisions (gender, shininess, cosmetic formes, move-slot
/// order) have methods of their own: they leave a set's identity
/// unchanged, so the enumerator does not branch on them.
pub(crate) trait Decide {
    /// Uniform draw from `0..n`. `n` is at least 1.
    #[track_caller]
    fn index(&mut self, n: usize) -> usize;
    /// `true` with probability `num/denom`.
    #[track_caller]
    fn chance(&mut self, num: u32, denom: u32) -> bool;
    /// Gender roll for a `male_eighths`-in-8 ratio.
    fn gender(&mut self, male_eighths: u8) -> Gender;
    /// A chance whose outcome is cosmetic (shiny).
    fn cosmetic_chance(&mut self, num: u32, denom: u32) -> bool;
    /// A uniform draw from `0..n` whose outcome is cosmetic (`getForme`).
    fn cosmetic_index(&mut self, n: usize) -> usize;
    /// Shuffle whose resulting order is cosmetic (move slots).
    fn shuffle_moves(&mut self, moves: &mut [MoveId]);
}
/// [`Decide`] backed by a `rand` RNG.
pub(crate) struct RngDecide<'a, R: Rng>(pub &'a mut R);
impl<R: Rng> Decide for RngDecide<'_, R> {
    fn index(&mut self, n: usize) -> usize {
        self.0.random_range(0..n)
    }
    fn chance(&mut self, num: u32, denom: u32) -> bool {
        self.0.random_range(0..denom) < num
    }
    fn gender(&mut self, male_eighths: u8) -> Gender {
        if self.0.random_range(0u8..8) < male_eighths {
            Gender::Male
        } else {
            Gender::Female
        }
    }
    fn cosmetic_chance(&mut self, num: u32, denom: u32) -> bool {
        self.chance(num, denom)
    }
    fn cosmetic_index(&mut self, n: usize) -> usize {
        self.index(n)
    }
    fn shuffle_moves(&mut self, moves: &mut [MoveId]) {
        for start in 0..moves.len().saturating_sub(1) {
            let next = self.0.random_range(start..moves.len());
            moves.swap(start, next);
        }
    }
}
/// PS's `fastPop`: O(1) swap-remove from an unsorted Vec.
pub(crate) fn fast_pop<T>(list: &mut Vec<T>, index: usize) -> T {
    let last = list.len() - 1;
    list.swap(index, last);
    list.pop().unwrap()
}
/// PS's `sample`: pick a uniformly-random element from the slice.
#[track_caller]
pub(crate) fn sample<T: Copy, D: Decide>(d: &mut D, list: &[T]) -> T {
    list[d.index(list.len())]
}
/// PS's `sampleNoReplace`: pick + remove a uniformly-random element.
#[track_caller]
pub(crate) fn sample_no_replace<T, D: Decide>(
    d: &mut D,
    list: &mut Vec<T>,
) -> Option<T> {
    if list.is_empty() {
        return None;
    }
    let i = d.index(list.len());
    Some(fast_pop(list, i))
}
/// PS's `randomChance(num, denom)`: returns true with probability num/denom.
#[track_caller]
pub(crate) fn random_chance<D: Decide>(d: &mut D, num: u32, denom: u32) -> bool {
    d.chance(num, denom)
}
/// Remove `item` from `list` if present (no-op otherwise). Mirrors the
/// PS pattern of `if (list.includes(x)) fastPop(list, list.indexOf(x))`.
pub(crate) fn remove_if_present<T: PartialEq>(list: &mut Vec<T>, item: T) {
    if let Some(i) = list.iter().position(|x| *x == item) {
        fast_pop(list, i);
    }
}
/// PS's `species.types` as a Vec for ergonomic iteration.
pub(crate) fn species_types(species: SpeciesId) -> Vec<Type> {
    let t = species.data().types;
    match t.1 {
        Some(t1) => vec![t.0, t1],
        None => vec![t.0],
    }
}
pub use crate::pokemon::Gender;
/// PS's `dex.getEffectiveness` of `attack_type` against `defender` species:
/// `0` neutral, `+1` SE, `+2` 4×, `-1` resisted, `-2` 4× resisted, summed
/// per type with an immunity counted as neutral, so a Dark/Ghost is
/// Fighting-weak to the generator.
pub(crate) fn type_effectiveness_score(attack_type: Type, defender: SpeciesId) -> i32 {
    type_score_against(attack_type, defender.data().types)
}
/// [`type_effectiveness_score`] against a typing rather than a species.
pub(crate) fn type_score_against(
    attack_type: Type,
    types: crate::species::SpeciesTypes,
) -> i32 {
    let score = |e: Effectiveness| match e {
        Effectiveness::Immune => 0,
        Effectiveness::Resistant => -1,
        Effectiveness::Neutral => 0,
        Effectiveness::SuperEffective => 1,
    };
    let s0 = score(attack_type.effectiveness_against(types.0));
    let s1 = types.1.map(|t1| score(attack_type.effectiveness_against(t1))).unwrap_or(0);
    s0 + s1
}
/// `species.tags.contains(NFE)`.
pub(crate) fn species_is_nfe(species: SpeciesId) -> bool {
    species.data().tags.contains(SpeciesTags::NFE)
}
/// All ability slots a species can have.
pub(crate) fn species_ability_pool(species: SpeciesId) -> Vec<AbilityId> {
    let abs = species.data().abilities;
    let mut out = vec![abs.slot_1];
    if let Some(a) = abs.slot_2 {
        out.push(a);
    }
    if let Some(a) = abs.hidden {
        out.push(a);
    }
    if let Some(a) = abs.special {
        out.push(a);
    }
    out
}
