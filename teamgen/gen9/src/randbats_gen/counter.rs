//! PS's `MoveCounter`, `queryMoves` and `getMoveType`: what a moveset
//! counts as, which the culling, ability and item rules read.
use crate::generated::ability_id::AbilityId;
use crate::generated::move_id::MoveId;
use crate::generated::species_id::SpeciesId;
use crate::moves::{Move, MoveCategory, MoveFlags, MultiHit};
use crate::types::Type;
use super::{
    CONTRARY_MOVES, HAZARDS, MIXED_SETUP, NO_STAB, PHYSICAL_SETUP, PRIORITY_POKEMON,
    RECOVERY_MOVES, SETUP, SPECIAL_SETUP, SPEED_CONTROL, SPEED_SETUP,
};
const IN_RECOVERY: u16 = 1 << 0;
const IN_CONTRARY: u16 = 1 << 1;
const IN_PHYSICAL_SETUP: u16 = 1 << 2;
const IN_SPECIAL_SETUP: u16 = 1 << 3;
const IN_MIXED_SETUP: u16 = 1 << 4;
const IN_SPEED_SETUP: u16 = 1 << 5;
const IN_SPEED_CONTROL: u16 = 1 << 6;
const IN_SETUP: u16 = 1 << 7;
const IN_HAZARDS: u16 = 1 << 8;
const IN_NO_STAB: u16 = 1 << 9;
/// The generator's move lists `move_id` is in, as `IN_*` bits: one table
/// lookup where `query_moves` scanned ten lists per move.
fn list_membership(move_id: MoveId) -> u16 {
    static TABLE: std::sync::OnceLock<Box<[u16]>> = std::sync::OnceLock::new();
    TABLE
        .get_or_init(|| {
            let lists: [(&[MoveId], u16); 10] = [
                (RECOVERY_MOVES, IN_RECOVERY),
                (CONTRARY_MOVES, IN_CONTRARY),
                (PHYSICAL_SETUP, IN_PHYSICAL_SETUP),
                (SPECIAL_SETUP, IN_SPECIAL_SETUP),
                (MIXED_SETUP, IN_MIXED_SETUP),
                (SPEED_SETUP, IN_SPEED_SETUP),
                (SPEED_CONTROL, IN_SPEED_CONTROL),
                (SETUP, IN_SETUP),
                (HAZARDS, IN_HAZARDS),
                (NO_STAB, IN_NO_STAB),
            ];
            let mut table = vec![0u16; MoveId::COUNT + 1];
            for (list, bit) in lists {
                for &m in list {
                    table[m as usize] |= bit;
                }
            }
            table.into()
        })[move_id as usize]
}
/// The counter's key space. PS keys its multiset by string; the keys are a
/// closed set, so here each is an index (a set costs several hundred
/// lookups).
///
/// The type slots lead, in `Type`'s own discriminant order, so
/// [`type_counter_key`] is an index rather than a match.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum CounterKey {
    Normal = 0,
    Fire,
    Water,
    Electric,
    Grass,
    Ice,
    Fighting,
    Poison,
    Ground,
    Flying,
    Psychic,
    Bug,
    Rock,
    Ghost,
    Dragon,
    Dark,
    Steel,
    Fairy,
    Stellar,
    Typeless,
    Physical,
    Special,
    Status,
    Contrary,
    Damage,
    Drain,
    Hazards,
    Inaccurate,
    Ironfist,
    Mixedsetup,
    Physicalsetup,
    Priority,
    Recoil,
    Recovery,
    Setup,
    Sheerforce,
    Skilllink,
    Sound,
    Specialsetup,
    Speedcontrol,
    Speedsetup,
    Stab,
    Stabtera,
    Strongjaw,
    Technician,
}
const COUNTER_KEYS: usize = 45;
/// PS's `MoveCounter` — a multiset of move-flavor counts plus the damaging
/// move view the caller reads back.
#[derive(Clone, Debug)]
pub struct MoveCounter {
    counts: [u32; COUNTER_KEYS],
    /// Distinct damaging move ids, in the order they were counted. PS keeps
    /// whole move objects; the only things read back are how many there are
    /// and the first one's type, and dedup needs the ids.
    damaging_ids: [MoveId; 8],
    damaging_len: u8,
    first_damaging_type: Option<Type>,
}
impl Default for MoveCounter {
    fn default() -> Self {
        Self {
            counts: [0; COUNTER_KEYS],
            damaging_ids: [MoveId::None; 8],
            damaging_len: 0,
            first_damaging_type: None,
        }
    }
}
impl MoveCounter {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add(&mut self, key: CounterKey) {
        self.counts[key as usize] += 1;
    }
    pub fn set(&mut self, key: CounterKey, value: u32) {
        self.counts[key as usize] = value;
    }
    pub fn get(&self, key: CounterKey) -> u32 {
        self.counts[key as usize]
    }
    pub fn damaging_is_empty(&self) -> bool {
        self.damaging_len == 0
    }
    pub fn damaging_len(&self) -> usize {
        usize::from(self.damaging_len)
    }
    /// Declared type of the first damaging move counted.
    pub fn first_damaging_type(&self) -> Option<Type> {
        self.first_damaging_type
    }
    fn add_damaging(&mut self, mv: Move) {
        let seen = &self.damaging_ids[..usize::from(self.damaging_len)];
        if seen.contains(&mv.id) {
            return;
        }
        if self.first_damaging_type.is_none() {
            self.first_damaging_type = Some(mv.typ);
        }
        debug_assert!(
            usize::from(self.damaging_len) < self.damaging_ids.len(),
            "more damaging moves than a set can hold"
        );
        if let Some(slot) = self.damaging_ids.get_mut(usize::from(self.damaging_len)) {
            *slot = mv.id;
            self.damaging_len += 1;
        }
    }
}
/// PS's `getMoveType`: the type `mv` counts as for STAB and coverage.
pub fn get_move_type(
    mv: &Move,
    species: SpeciesId,
    abilities: &[AbilityId],
    tera_type: Type,
) -> Type {
    if mv.id == MoveId::Terablast {
        return tera_type;
    }
    if matches!(
        mv.id, MoveId::Judgment | MoveId::Multiattack | MoveId::Revelationdance
    ) {
        return species.data().types.0;
    }
    if mv.id == MoveId::Ragingbull {
        let name = species.data().name;
        if name.starts_with("Tauros-Paldea") {
            if name.ends_with("Combat") {
                return Type::Fighting;
            }
            if name.ends_with("Blaze") {
                return Type::Fire;
            }
            if name.ends_with("Aqua") {
                return Type::Water;
            }
        }
    }
    if mv.id == MoveId::Ivycudgel {
        let name = species.data().name;
        if name.starts_with("Ogerpon") {
            if name.contains("Wellspring") {
                return Type::Water;
            }
            if name.contains("Hearthflame") {
                return Type::Fire;
            }
            if name.contains("Cornerstone") {
                return Type::Rock;
            }
        }
    }
    if mv.id == MoveId::Terrainpulse {
        if abilities.contains(&AbilityId::Grassysurge)
            || abilities.contains(&AbilityId::Seedsower)
        {
            return Type::Grass;
        }
        if abilities.contains(&AbilityId::Psychicsurge) {
            return Type::Psychic;
        }
        if abilities.contains(&AbilityId::Electricsurge) {
            return Type::Electric;
        }
    }
    if mv.id == MoveId::Weatherball {
        const SUN: [AbilityId; 4] = [
            AbilityId::Chlorophyll,
            AbilityId::Drought,
            AbilityId::Megasol,
            AbilityId::Orichalcumpulse,
        ];
        let data = species.data();
        let dex = &data.abilities;
        let mega = data.forme_name.is_some_and(|f| f.starts_with("Mega"));
        let dex_sun = [Some(dex.slot_1), dex.slot_2, dex.hidden, dex.special]
            .into_iter()
            .flatten()
            .any(|a| SUN.contains(&a));
        if abilities.iter().any(|a| SUN.contains(a)) || mega && dex_sun {
            return Type::Fire;
        }
        if abilities.contains(&AbilityId::Drizzle) {
            return Type::Water;
        }
    }
    if mv.typ == Type::Normal {
        if abilities.contains(&AbilityId::Aerilate) {
            return Type::Flying;
        }
        if abilities.contains(&AbilityId::Galvanize) {
            return Type::Electric;
        }
        if abilities.contains(&AbilityId::Pixilate) {
            return Type::Fairy;
        }
        if abilities.contains(&AbilityId::Refrigerate) {
            return Type::Ice;
        }
    }
    mv.typ
}
/// PS's `move.damage || move.damageCallback`: `queryMoves` counts such a
/// move as `Damage`, not by its category.
pub(crate) fn deals_set_damage(move_id: MoveId) -> bool {
    let mv = move_id.data();
    mv.set_damage
}
/// PS's `queryMoves`: the [`MoveCounter`] of a moveset.
pub fn query_moves(
    moves: &[MoveId],
    species: SpeciesId,
    tera_type: Type,
    abilities: &[AbilityId],
) -> MoveCounter {
    let mut counter = MoveCounter::new();
    let types = species.data().types;
    if moves.is_empty() {
        return counter;
    }
    let mut categories = (0u32, 0u32, 0u32);
    for &move_id in moves {
        let move_id = if move_id == MoveId::Naturepower {
            MoveId::Triattack
        } else {
            move_id
        };
        let mv = move_id.data();
        let lists = list_membership(move_id);
        let move_type = get_move_type(mv, species, abilities, tera_type);
        if deals_set_damage(move_id) {
            counter.add(CounterKey::Damage);
            counter.add_damaging(*mv);
        } else {
            match mv.category {
                MoveCategory::Physical => categories.0 += 1,
                MoveCategory::Special => categories.1 += 1,
                MoveCategory::Status => categories.2 += 1,
            }
        }
        if move_id == MoveId::Lowkick
            || (mv.base_power > 0 && mv.base_power <= 60
                && !matches!(move_id, MoveId::Nuzzle | MoveId::Rapidspin))
        {
            counter.add(CounterKey::Technician);
        }
        if matches!(mv.multihit, MultiHit::Range(_, 5)) {
            counter.add(CounterKey::Skilllink);
        }
        if mv.recoil.is_some() {
            counter.add(CounterKey::Recoil);
        }
        if mv.drain.is_some() {
            counter.add(CounterKey::Drain);
        }
        let has_bp = mv.base_power > 0 || mv.has_base_power_callback;
        if has_bp {
            let counts_for_stab = lists & IN_NO_STAB == 0
                || (PRIORITY_POKEMON.contains(&species) && mv.priority > 0);
            if counts_for_stab {
                counter.add(type_counter_key(move_type));
                let species_types_match = types.0 == move_type
                    || types.1 == Some(move_type);
                if species_types_match {
                    counter.add(CounterKey::Stab);
                }
                if tera_type == move_type {
                    counter.add(CounterKey::Stabtera);
                }
                counter.add_damaging(*mv);
            }
            if mv.flags.contains(MoveFlags::BITE) {
                counter.add(CounterKey::Strongjaw);
            }
            if mv.flags.contains(MoveFlags::PUNCH) {
                counter.add(CounterKey::Ironfist);
            }
            if mv.flags.contains(MoveFlags::SOUND) {
                counter.add(CounterKey::Sound);
            }
            if mv.priority > 0
                || (move_id == MoveId::Grassyglide
                    && abilities.contains(&AbilityId::Grassysurge))
            {
                counter.add(CounterKey::Priority);
            }
        }
        if mv.secondary.is_some() {
            counter.add(CounterKey::Sheerforce);
        }
        match mv.accuracy {
            crate::moves::Accuracy::Percent(p) if p < 90 => {
                counter.add(CounterKey::Inaccurate)
            }
            _ => {}
        }
        if lists & IN_RECOVERY != 0 {
            counter.add(CounterKey::Recovery);
        }
        if lists & IN_CONTRARY != 0 {
            counter.add(CounterKey::Contrary);
        }
        if lists & IN_PHYSICAL_SETUP != 0 {
            counter.add(CounterKey::Physicalsetup);
        }
        if lists & IN_SPECIAL_SETUP != 0 {
            counter.add(CounterKey::Specialsetup);
        }
        if lists & IN_MIXED_SETUP != 0 {
            counter.add(CounterKey::Mixedsetup);
        }
        if lists & IN_SPEED_SETUP != 0 {
            counter.add(CounterKey::Speedsetup);
        }
        if lists & IN_SPEED_CONTROL != 0 {
            counter.add(CounterKey::Speedcontrol);
        }
        if lists & IN_SETUP != 0 {
            counter.add(CounterKey::Setup);
        }
        if lists & IN_HAZARDS != 0 {
            counter.add(CounterKey::Hazards);
        }
    }
    counter.set(CounterKey::Physical, categories.0);
    counter.set(CounterKey::Special, categories.1);
    counter.set(CounterKey::Status, categories.2);
    counter
}
/// The counter slot for a type's move count (PS's `counter.add('Bug')`).
///
/// `CounterKey`'s type slots are declared in `Type`'s discriminant order, so
/// this is a table index.
pub fn type_counter_key(t: Type) -> CounterKey {
    TYPE_KEYS[t as usize]
}
static TYPE_KEYS: [CounterKey; 20] = [
    CounterKey::Normal,
    CounterKey::Fire,
    CounterKey::Water,
    CounterKey::Electric,
    CounterKey::Grass,
    CounterKey::Ice,
    CounterKey::Fighting,
    CounterKey::Poison,
    CounterKey::Ground,
    CounterKey::Flying,
    CounterKey::Psychic,
    CounterKey::Bug,
    CounterKey::Rock,
    CounterKey::Ghost,
    CounterKey::Dragon,
    CounterKey::Dark,
    CounterKey::Steel,
    CounterKey::Fairy,
    CounterKey::Stellar,
    CounterKey::Typeless,
];
