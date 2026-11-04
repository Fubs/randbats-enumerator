//! PS's `randomTeam` in singles: the base-species pool it draws from, its
//! rejection gates, and the `teamDetails` the members accepted so far
//! pass to `randomSet`.
use std::collections::HashMap;
use std::sync::OnceLock;
use rand::Rng;
use crate::generated::ability_id::AbilityId;
use crate::generated::move_id::MoveId;
use crate::generated::randbats_data::RoleId;
use crate::generated::species_id::SpeciesId;
use crate::randbats::randbats_table;
use crate::types::Type;
use super::set::{RandomSet, random_set_decide};
use super::{
    Decide, NO_LEAD_SPECIES, RngDecide, sample, sample_no_replace, species_types,
    type_effectiveness_score, type_score_against,
};
/// Maximum team size in singles.
pub const MAX_TEAM_SIZE: usize = 6;
/// PS's `limitFactor`: `Math.round(maxTeamSize / 6) || 1` — always 1
/// at the singles team size.
const LIMIT_FACTOR: u32 = 1;
/// PS's `RandomTeamsTypes.TeamDetails`: what the members accepted so far
/// bring (weather, hazards, hazard removal, screens, a status cure, a Tera
/// Blast user), which `randomSet` and its helpers read. `Hash` and `Ord`
/// so that tables can be keyed on it.
#[derive(Default, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TeamDetails {
    pub rain: bool,
    pub sun: bool,
    pub sand: bool,
    pub snow: bool,
    pub stealth_rock: bool,
    pub spikes: u8,
    pub toxic_spikes: bool,
    pub sticky_web: bool,
    pub rapid_spin: bool,
    pub defog: bool,
    pub screens: bool,
    pub status_cure: bool,
    pub tera_blast: bool,
}
impl TeamDetails {
    /// OR in another member's contribution (`spikes` layers add): per-member
    /// deltas compose into the details a later member sees.
    pub fn merge(&mut self, other: &TeamDetails) {
        self.rain |= other.rain;
        self.sun |= other.sun;
        self.sand |= other.sand;
        self.snow |= other.snow;
        self.stealth_rock |= other.stealth_rock;
        self.spikes = self.spikes.saturating_add(other.spikes);
        self.toxic_spikes |= other.toxic_spikes;
        self.sticky_web |= other.sticky_web;
        self.rapid_spin |= other.rapid_spin;
        self.defog |= other.defog;
        self.screens |= other.screens;
        self.status_cure |= other.status_cure;
        self.tera_blast |= other.tera_blast;
    }
}
/// `(weighted base pool, base → formes, base → its run in the pool)`, built
/// once from the tables in `getPokemonPool`'s order.
type Pools = (
    Vec<SpeciesId>,
    HashMap<SpeciesId, Vec<SpeciesId>>,
    HashMap<SpeciesId, (usize, usize)>,
);
fn pools() -> &'static Pools {
    static POOLS: OnceLock<Pools> = OnceLock::new();
    POOLS
        .get_or_init(|| {
            let mut entries: Vec<_> = randbats_table().iter().flatten().collect();
            entries.sort_unstable_by_key(|e| e.ps_order);
            let mut by_base: HashMap<SpeciesId, Vec<SpeciesId>> = HashMap::new();
            let mut bases: Vec<SpeciesId> = Vec::new();
            for entry in entries {
                let base = entry.species.data().base_species;
                if !by_base.contains_key(&base) {
                    bases.push(base);
                }
                by_base.entry(base).or_default().push(entry.species);
            }
            let mut weighted: Vec<SpeciesId> = Vec::new();
            let mut runs: HashMap<SpeciesId, (usize, usize)> = HashMap::new();
            for &base in &bases {
                let n = by_base[&base].len();
                let weight = if base == SpeciesId::Squawkabilly {
                    1
                } else {
                    ((n as f32 / 3.0).ceil() as usize).clamp(1, 3)
                };
                runs.insert(base, (weighted.len(), weighted.len() + weight));
                for _ in 0..weight {
                    weighted.push(base);
                }
            }
            (weighted, by_base, runs)
        })
}
/// PS's `baseSpeciesPool` from `getPokemonPool`: each base species
/// ⌈formes / 3⌉ times, at most 3 (Squawkabilly once), in `sets.json`
/// order. `randomTeam` draws a base uniformly from this, then a forme
/// uniformly ([`formes_of`]).
pub fn weighted_base_pool() -> &'static [SpeciesId] {
    &pools().0
}
/// Each base of [`weighted_base_pool`] with its multiplicity, in pool
/// order.
pub fn weighted_base_runs() -> impl Iterator<Item = (SpeciesId, usize)> {
    weighted_base_pool().chunk_by(|a, b| a == b).map(|run| (run[0], run.len()))
}
/// The formes a drawn base resolves to, uniformly.
pub fn formes_of(base: SpeciesId) -> &'static [SpeciesId] {
    pools().1.get(&base).map(|v| v.as_slice()).unwrap_or(&[])
}
/// PS's `NO_LEAD_POKEMON`: the formes `randomTeam` never puts in the lead
/// slot (Dugtrio, Gothitelle, Iron Thorns, Kingambit, and Zacian and
/// Zamazenta in both formes). Keyed on the forme, as PS keys it.
pub fn never_leads(species: SpeciesId) -> bool {
    NO_LEAD_SPECIES.contains(&species)
}
/// `randomTeam`'s Freeze-Dry weakness: Ice-weak, or Water-typed without a
/// 4× Ice resistance.
fn weak_to_freeze_dry(species: SpeciesId) -> bool {
    let ice = type_effectiveness_score(Type::Ice, species);
    ice > 0 || (ice > -2 && species_types(species).contains(&Type::Water))
}
/// The running state of PS's `randomTeam` loop: its counters, its
/// rejection gates ([`Self::admits`]) and its updates after an acceptance
/// ([`Self::record`]). The gates are pure reads and draw nothing, so the
/// loop's draws are PS's.
///
/// [`Self::record`] has two halves: [`Self::record_species`], what a
/// member's species and level imply, and [`Self::note_set_facts`], what
/// its set implies.
///
/// Fixed-size and allocation-free. PS's `typeComboCount` is not kept: only
/// its Monotype format reads it.
#[derive(Clone, Debug)]
pub struct TeamGenState {
    species_on_team: [SpeciesId; MAX_MEMBERS],
    /// `species_on_team`'s base formes, for the species clause.
    bases: [SpeciesId; MAX_MEMBERS],
    members: u8,
    type_count: TypeCounts,
    type_weaknesses: TypeCounts,
    type_double_weaknesses: TypeCounts,
    freeze_dry_weakness_count: u32,
    num_max_level_pokemon: u32,
    pub team_details: TeamDetails,
}
/// The capacity of a state's member arrays.
const MAX_MEMBERS: usize = 12;
/// A count per type, indexed by the type's discriminant.
#[derive(Clone, Copy, Debug, Default)]
struct TypeCounts([u32; Type::Typeless as usize + 1]);
impl TypeCounts {
    fn get(&self, t: Type) -> u32 {
        self.0[t as usize]
    }
    fn bump(&mut self, t: Type) {
        self.0[t as usize] += 1;
    }
    fn bump_each(&mut self, mut mask: u32) {
        while mask != 0 {
            self.0[mask.trailing_zeros() as usize] += 1;
            mask &= mask - 1;
        }
    }
    /// Whether any type in `mask` is counted at least `limit` times.
    fn any_at_least(&self, mut mask: u32, limit: u32) -> bool {
        while mask != 0 {
            if self.0[mask.trailing_zeros() as usize] >= limit {
                return true;
            }
            mask &= mask - 1;
        }
        false
    }
}
/// (weak, double-weak) bitmasks over `Type` of the attacking types a
/// species of these types is weak to (0 < score < 90) and double-weak to
/// (score > 1). The answer depends on the typing alone, so one small table
/// replaces the 18 type-chart scores per candidate the gates would compute.
fn weakness_masks(types: crate::species::SpeciesTypes) -> (u32, u32) {
    const T: usize = Type::Typeless as usize + 1;
    static TABLE: OnceLock<[[(u32, u32); T + 1]; T]> = OnceLock::new();
    let table = TABLE
        .get_or_init(|| {
            let defenders: Vec<Type> = ALL_OFFENSIVE_TYPES
                .iter()
                .copied()
                .chain([Type::Stellar, Type::Typeless])
                .collect();
            let mut table = [[(0, 0); T + 1]; T];
            for &first in &defenders {
                for second in defenders.iter().copied().map(Some).chain([None]) {
                    let cell = &mut table[first
                        as usize][second.map_or(T, |t| t as usize)];
                    for &t in ALL_OFFENSIVE_TYPES {
                        let s = type_score_against(
                            t,
                            crate::species::SpeciesTypes(first, second),
                        );
                        if s > 0 && s < 90 {
                            cell.0 |= 1 << t as u32;
                        }
                        if s > 1 {
                            cell.1 |= 1 << t as u32;
                        }
                    }
                }
            }
            table
        });
    table[types.0 as usize][types.1.map_or(T, |t| t as usize)]
}
impl Default for TeamGenState {
    fn default() -> Self {
        Self {
            species_on_team: [SpeciesId::None; MAX_MEMBERS],
            bases: [SpeciesId::None; MAX_MEMBERS],
            members: 0,
            type_count: TypeCounts::default(),
            type_weaknesses: TypeCounts::default(),
            type_double_weaknesses: TypeCounts::default(),
            freeze_dry_weakness_count: 0,
            num_max_level_pokemon: 0,
            team_details: TeamDetails::default(),
        }
    }
}
impl TeamGenState {
    fn species(&self) -> &[SpeciesId] {
        &self.species_on_team[..self.members as usize]
    }
    /// Members recorded so far.
    pub fn len(&self) -> usize {
        self.members as usize
    }
    pub fn is_empty(&self) -> bool {
        self.members == 0
    }
    pub fn has_base(&self, base: SpeciesId) -> bool {
        self.bases[..self.members as usize].contains(&base)
    }
    /// PS's rejection gates for a candidate `species`. They are pure reads,
    /// so checking them in another order than `randomTeam` changes nothing.
    /// `members` is `pokemon.length` at check time.
    pub fn admits(&self, species: SpeciesId, members: usize, team_size: usize) -> bool {
        if species.data().base_species == SpeciesId::Zoroark && members + 1 >= team_size
        {
            return false;
        }
        self.admits_composition(species)
    }
    /// The composition slice of [`Self::admits`]: every gate that constrains
    /// what a finished team contains, none that constrains where.
    pub fn admits_composition(&self, species: SpeciesId) -> bool {
        if self.has_base(species.data().base_species) {
            return false;
        }
        if matches!(
            species, SpeciesId::Ogerpon | SpeciesId::Ogerponhearthflame |
            SpeciesId::Terapagos
        ) && self.team_details.tera_blast
        {
            return false;
        }
        let types = species.data().types;
        if self.type_count.get(types.0) >= 2 * LIMIT_FACTOR
            || types.1.is_some_and(|t| self.type_count.get(t) >= 2 * LIMIT_FACTOR)
        {
            return false;
        }
        let (weak, double_weak) = weakness_masks(types);
        if self.type_weaknesses.any_at_least(weak, 3 * LIMIT_FACTOR)
            || self.type_double_weaknesses.any_at_least(double_weak, LIMIT_FACTOR)
        {
            return false;
        }
        if self.type_weaknesses.get(Type::Fire) >= 3 * LIMIT_FACTOR
            && type_score_against(Type::Fire, types) == 0
            && super::species_ability_pool(species)
                .iter()
                .any(|a| matches!(* a, AbilityId::Dryskin | AbilityId::Fluffy))
        {
            return false;
        }
        if weak_to_freeze_dry(species)
            && self.freeze_dry_weakness_count >= 4 * LIMIT_FACTOR
        {
            return false;
        }
        let entry_level = species.randbats_entry().map(|e| e.level).unwrap_or(80);
        if entry_level == 100 && self.num_max_level_pokemon >= LIMIT_FACTOR {
            return false;
        }
        get_pokemon_compatibility(species, self.species())
    }
    /// Everything the counters learn from a member's species and level
    /// alone: the species clause, the type, weakness, Freeze-Dry and
    /// level-100 counts, and the species half of the Tera Blast accounting
    /// (an Ogerpon or Terapagos is the team's Tera Blast user).
    pub fn record_species(&mut self, species: SpeciesId, level: u8) {
        self.species_on_team[self.members as usize] = species;
        self.bases[self.members as usize] = species.data().base_species;
        self.members += 1;
        let types = species.data().types;
        self.type_count.bump(types.0);
        if let Some(t) = types.1 {
            self.type_count.bump(t);
        }
        let (weak, double_weak) = weakness_masks(types);
        self.type_weaknesses.bump_each(weak);
        self.type_double_weaknesses.bump_each(double_weak);
        if weak_to_freeze_dry(species) {
            self.freeze_dry_weakness_count += 1;
        }
        if level == 100 {
            self.num_max_level_pokemon += 1;
        }
        if matches!(
            species, SpeciesId::Ogerpon | SpeciesId::Ogerponhearthflame |
            SpeciesId::Terapagos
        ) {
            self.team_details.tera_blast = true;
        }
    }
    /// The set-dependent half of PS's post-acceptance block: the flags a
    /// member's ability, moves and role set, and Dry Skin or Fluffy
    /// counting as a Fire weakness.
    pub(crate) fn note_facts(
        &mut self,
        species: SpeciesId,
        ability: Option<AbilityId>,
        moves: &[MoveId],
        tera_blast_role: bool,
    ) {
        let d = &mut self.team_details;
        let has = |m: MoveId| moves.contains(&m);
        if matches!(ability, Some(AbilityId::Dryskin | AbilityId::Fluffy))
            && type_effectiveness_score(Type::Fire, species) == 0
        {
            self.type_weaknesses.bump(Type::Fire);
        }
        if ability == Some(AbilityId::Drizzle) || has(MoveId::Raindance) {
            d.rain = true;
        }
        if matches!(ability, Some(AbilityId::Drought | AbilityId::Orichalcumpulse))
            || has(MoveId::Sunnyday)
        {
            d.sun = true;
        }
        if ability == Some(AbilityId::Sandstream) {
            d.sand = true;
        }
        if ability == Some(AbilityId::Snowwarning) || has(MoveId::Snowscape)
            || has(MoveId::Chillyreception)
        {
            d.snow = true;
        }
        if has(MoveId::Healbell) {
            d.status_cure = true;
        }
        if has(MoveId::Spikes) || has(MoveId::Ceaselessedge) {
            d.spikes = d.spikes.saturating_add(1);
        }
        if has(MoveId::Toxicspikes) || ability == Some(AbilityId::Toxicdebris) {
            d.toxic_spikes = true;
        }
        if has(MoveId::Stealthrock) || has(MoveId::Stoneaxe) {
            d.stealth_rock = true;
        }
        if has(MoveId::Stickyweb) {
            d.sticky_web = true;
        }
        if has(MoveId::Defog) {
            d.defog = true;
        }
        if has(MoveId::Rapidspin) || has(MoveId::Mortalspin) {
            d.rapid_spin = true;
        }
        if has(MoveId::Auroraveil) || (has(MoveId::Reflect) && has(MoveId::Lightscreen))
        {
            d.screens = true;
        }
        if tera_blast_role {
            d.tera_blast = true;
        }
    }
    /// What a member's set implies for the team's flags and counters.
    pub fn note_set_facts(&mut self, set: &RandomSet) {
        let moves: Vec<MoveId> = set.moves.iter().flatten().copied().collect();
        self.note_facts(
            set.species,
            Some(set.ability),
            &moves,
            set.role == Some(RoleId::TeraBlastUser),
        );
    }
    /// PS's full post-acceptance tracking block.
    pub fn record(&mut self, set: &RandomSet) {
        self.record_species(set.species, set.level);
        self.note_set_facts(set);
    }
}
/// PS's `randomTeam`: a team of six drawn from `rng`, in PS's order (the
/// lead first, the others in the order they were generated).
pub fn generate_team<R: Rng>(rng: &mut R) -> Vec<RandomSet> {
    generate_team_decide(&mut RngDecide(rng))
}
/// [`generate_team`] over any [`Decide`].
pub(crate) fn generate_team_decide<D: Decide>(rng: &mut D) -> Vec<RandomSet> {
    let is_doubles = false;
    let mut base_species_pool: Vec<SpeciesId> = weighted_base_pool().to_vec();
    let mut pokemon: Vec<RandomSet> = Vec::new();
    let mut state = TeamGenState::default();
    let mut leads_remaining: usize = 1;
    while !base_species_pool.is_empty() && pokemon.len() < MAX_TEAM_SIZE {
        let Some(base_species) = sample_no_replace(rng, &mut base_species_pool) else {
            break;
        };
        let formes = formes_of(base_species);
        if formes.is_empty() {
            continue;
        }
        let species = sample(rng, formes);
        if !state.admits(species, pokemon.len(), MAX_TEAM_SIZE) {
            continue;
        }
        let set = if leads_remaining > 0 {
            if never_leads(species) {
                if pokemon.len() + leads_remaining == MAX_TEAM_SIZE {
                    continue;
                }
                let s = random_set_decide(
                    rng,
                    species,
                    &state.team_details,
                    false,
                    is_doubles,
                );
                pokemon.push(s.clone());
                s
            } else {
                let s = random_set_decide(
                    rng,
                    species,
                    &state.team_details,
                    true,
                    is_doubles,
                );
                pokemon.insert(0, s.clone());
                leads_remaining -= 1;
                s
            }
        } else {
            let s = random_set_decide(
                rng,
                species,
                &state.team_details,
                false,
                is_doubles,
            );
            pokemon.push(s.clone());
            s
        };
        if pokemon.len() == MAX_TEAM_SIZE {
            break;
        }
        state.record(&set);
    }
    pokemon
}
/// PS's `getPokemonCompatibility` in singles: no Blissey with Chansey or
/// Illumise with Volbeat, no two web or screen setters, no Toxicroak with
/// a sun setter.
fn get_pokemon_compatibility(species: SpeciesId, on_team: &[SpeciesId]) -> bool {
    let blissey_chansey = [SpeciesId::Blissey, SpeciesId::Chansey];
    if blissey_chansey.contains(&species) {
        if blissey_chansey.iter().any(|s| on_team.contains(s) && *s != species) {
            return false;
        }
    }
    let illumise_volbeat = [SpeciesId::Illumise, SpeciesId::Volbeat];
    if illumise_volbeat.contains(&species) {
        if illumise_volbeat.iter().any(|s| on_team.contains(s) && *s != species) {
            return false;
        }
    }
    let web_setters: &[SpeciesId] = &[
        SpeciesId::Ariados,
        SpeciesId::Smeargle,
        SpeciesId::Masquerain,
        SpeciesId::Kricketune,
        SpeciesId::Leavanny,
        SpeciesId::Galvantula,
        SpeciesId::Vikavolt,
        SpeciesId::Ribombee,
        SpeciesId::Araquanid,
        SpeciesId::Spidops,
    ];
    if web_setters.contains(&species) && on_team.iter().any(|p| web_setters.contains(p))
    {
        return false;
    }
    let screen_setters: &[SpeciesId] = &[
        SpeciesId::Meowstic,
        SpeciesId::Grimmsnarl,
        SpeciesId::Ninetalesalola,
        SpeciesId::Abomasnow,
    ];
    if screen_setters.contains(&species)
        && on_team.iter().any(|p| screen_setters.contains(p))
    {
        return false;
    }
    let sun_setters: &[SpeciesId] = &[
        SpeciesId::Ninetales,
        SpeciesId::Torkoal,
        SpeciesId::Groudon,
        SpeciesId::Koraidon,
    ];
    if species == SpeciesId::Toxicroak && on_team.iter().any(|p| sun_setters.contains(p))
    {
        return false;
    }
    if sun_setters.contains(&species) && on_team.contains(&SpeciesId::Toxicroak) {
        return false;
    }
    true
}
const ALL_OFFENSIVE_TYPES: &[Type] = &[
    Type::Bug,
    Type::Dark,
    Type::Dragon,
    Type::Electric,
    Type::Fairy,
    Type::Fighting,
    Type::Fire,
    Type::Flying,
    Type::Ghost,
    Type::Grass,
    Type::Ground,
    Type::Ice,
    Type::Normal,
    Type::Poison,
    Type::Psychic,
    Type::Rock,
    Type::Steel,
    Type::Water,
];
