//! Species data, as the generator reads it.
use crate::generated::ability_id::AbilityId;
use crate::generated::item_id::ItemId;
use crate::generated::move_id::MoveId;
use crate::generated::species_id::SpeciesId;
use crate::types::Type;
#[derive(Copy, Clone, Debug)]
pub struct Species {
    pub id: SpeciesId,
    pub name: &'static str,
    pub types: SpeciesTypes,
    pub base_stats: StatBlock,
    pub abilities: SpeciesAbilities,
    pub gender_ratio: GenderRatio,
    /// The species itself for a base forme.
    pub base_species: SpeciesId,
    /// `Some("Origin")` for Dialga-Origin.
    pub forme_name: Option<&'static str>,
    pub required_item: Option<ItemId>,
    /// In PS's order: Arceus takes the first.
    pub required_items: &'static [ItemId],
    /// Keldeo-Resolute's Secret Sword.
    pub required_move: Option<MoveId>,
    pub tags: SpeciesTags,
}
/// A primary type and an optional second.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SpeciesTypes(pub Type, pub Option<Type>);
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct StatBlock {
    pub hp: u16,
    pub atk: u16,
    pub def: u16,
    pub spa: u16,
    pub spd: u16,
    pub spe: u16,
}
#[derive(Copy, Clone, Debug)]
pub struct SpeciesAbilities {
    pub slot_1: AbilityId,
    pub slot_2: Option<AbilityId>,
    pub hidden: Option<AbilityId>,
    pub special: Option<AbilityId>,
}
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GenderRatio {
    Genderless,
    AlwaysMale,
    AlwaysFemale,
    /// Males in eighths, `0..=8`.
    Ratio { male_eighths: u8 },
}
/// Species flags the generator reads.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct SpeciesTags(u8);
impl SpeciesTags {
    /// Not fully evolved.
    pub const NFE: Self = Self(1);
    pub const fn empty() -> Self {
        Self(0)
    }
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}
/// Zacian and Zamazenta holding their signature item battle as the Crowned
/// forme, which PS resolves when the team is built.
pub fn crowned_forme(species: SpeciesId, item: ItemId) -> SpeciesId {
    match (species, item) {
        (SpeciesId::Zacian, ItemId::Rustedsword) => SpeciesId::Zaciancrowned,
        (SpeciesId::Zamazenta, ItemId::Rustedshield) => SpeciesId::Zamazentacrowned,
        _ => species,
    }
}
