//! Pokémon types and the type chart.
/// The 18 types, Stellar (Terastallization only) and Typeless. The
/// discriminants index the type chart and the generator's per-type tables.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Type {
    Normal = 0,
    Fire = 1,
    Water = 2,
    Electric = 3,
    Grass = 4,
    Ice = 5,
    Fighting = 6,
    Poison = 7,
    Ground = 8,
    Flying = 9,
    Psychic = 10,
    Bug = 11,
    Rock = 12,
    Ghost = 13,
    Dragon = 14,
    Dark = 15,
    Steel = 16,
    Fairy = 17,
    Stellar = 18,
    Typeless = 19,
}
impl Type {
    /// The types in the chart: every type but `Typeless`.
    pub const CHART_SIZE: usize = 19;
    pub const fn name(self) -> &'static str {
        match self {
            Type::Normal => "Normal",
            Type::Fire => "Fire",
            Type::Water => "Water",
            Type::Electric => "Electric",
            Type::Grass => "Grass",
            Type::Ice => "Ice",
            Type::Fighting => "Fighting",
            Type::Poison => "Poison",
            Type::Ground => "Ground",
            Type::Flying => "Flying",
            Type::Psychic => "Psychic",
            Type::Bug => "Bug",
            Type::Rock => "Rock",
            Type::Ghost => "Ghost",
            Type::Dragon => "Dragon",
            Type::Dark => "Dark",
            Type::Steel => "Steel",
            Type::Fairy => "Fairy",
            Type::Stellar => "Stellar",
            Type::Typeless => "???",
        }
    }
    /// `self` attacking `defender`, from PS's type chart. `Typeless` is
    /// neutral either way.
    pub const fn effectiveness_against(self, defender: Type) -> Effectiveness {
        if (self as usize) >= Self::CHART_SIZE || (defender as usize) >= Self::CHART_SIZE
        {
            return Effectiveness::Neutral;
        }
        crate::generated::type_chart::TYPE_CHART[self as usize][defender as usize]
    }
}
/// One type against one type: ×0, ×½, ×1 or ×2.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Effectiveness {
    Immune = 0,
    Resistant = 1,
    Neutral = 2,
    SuperEffective = 3,
}
