//! The Gen 9 Random Battles set tables: PS's
//! `data/random-battles/gen9/sets.json`, one [`RandbatsEntry`] per species
//! in the format.
//!
//! Each PS set becomes one [`RandbatsRole`], carrying its movepool,
//! abilities and Tera types. PS picks a set, and a Tera type in it,
//! uniformly, and its moves and ability by rule; the generator in
//! [`crate::randbats_gen`] makes those choices as PS does.
use crate::generated::ability_id::AbilityId;
use crate::generated::move_id::MoveId;
use crate::generated::species_id::SpeciesId;
use crate::types::Type;
/// One of a set's abilities.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct WeightedAbility {
    pub id: AbilityId,
}
/// One of a set's Tera types.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct WeightedType {
    pub typ: Type,
}
/// A move in a set's movepool. Which moves a set gets is decided by
/// `randomMoveset`'s rules.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct WeightedMove {
    pub id: MoveId,
}
/// One PS set of a species: its role label, movepool, abilities and Tera
/// types. Two sets with the same role label are two `RandbatsRole`s.
#[derive(Copy, Clone, Debug)]
pub struct RandbatsRole {
    /// The set's role ("Bulky Attacker", "Setup Sweeper", …). The enum is
    /// generated from the labels in the data; `RoleId::name` gives the
    /// label back.
    pub id: crate::generated::randbats_data::RoleId,
    pub abilities: &'static [WeightedAbility],
    pub tera_types: &'static [WeightedType],
    pub moves: &'static [WeightedMove],
}
/// One species' entry in `sets.json`.
#[derive(Copy, Clone, Debug)]
pub struct RandbatsEntry {
    pub species: SpeciesId,
    /// The species' place in PS's `sets.json`, the order `getPokemonPool`
    /// builds the base pool in.
    pub ps_order: u16,
    /// How many formes PS's `getForme` samples from before anything else in
    /// the set (Vivillon's patterns, Pikachu's caps), 0 for no draw.
    pub forme_draw: u8,
    /// The species' level in `sets.json`.
    pub level: u8,
    pub roles: &'static [RandbatsRole],
}
/// Every species' entry, indexed by `SpeciesId`; `None` outside the format.
pub fn randbats_table() -> &'static [Option<RandbatsEntry>; SpeciesId::COUNT] {
    &crate::generated::randbats_data::RANDBATS_BY_SPECIES
}
impl SpeciesId {
    /// The species' entry, or `None` if it is not in the format.
    #[inline]
    pub fn randbats_entry(self) -> Option<&'static RandbatsEntry> {
        randbats_table()[self as usize].as_ref()
    }
}
