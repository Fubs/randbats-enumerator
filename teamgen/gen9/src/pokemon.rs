//! The parts of a set that are not species data.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum Gender {
    Male,
    Female,
    #[default]
    Genderless,
}
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct EvSpread {
    pub hp: u8,
    pub atk: u8,
    pub def: u8,
    pub spa: u8,
    pub spd: u8,
    pub spe: u8,
}
impl EvSpread {
    pub const fn new(hp: u8, atk: u8, def: u8, spa: u8, spd: u8, spe: u8) -> Self {
        Self {
            hp,
            atk,
            def,
            spa,
            spd,
            spe,
        }
    }
}
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct IvSpread {
    pub hp: u8,
    pub atk: u8,
    pub def: u8,
    pub spa: u8,
    pub spd: u8,
    pub spe: u8,
}
impl IvSpread {
    /// 31 in every stat.
    pub const fn perfect() -> Self {
        Self {
            hp: 31,
            atk: 31,
            def: 31,
            spa: 31,
            spd: 31,
            spe: 31,
        }
    }
}
