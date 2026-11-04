//! Move data, as the generator reads it.
use crate::generated::move_id::MoveId;
use crate::types::Type;
#[derive(Copy, Clone, Debug)]
pub struct Move {
    pub id: MoveId,
    pub name: &'static str,
    pub typ: Type,
    pub category: MoveCategory,
    pub base_power: u16,
    pub accuracy: Accuracy,
    pub priority: i8,
    pub flags: MoveFlags,
    pub multihit: MultiHit,
    pub drain: Option<Fraction>,
    pub recoil: Option<Fraction>,
    pub secondary: Option<Secondary>,
    /// PS's `basePowerCallback`.
    pub has_base_power_callback: bool,
    /// PS's `damage || damageCallback`: fixed or computed damage.
    pub set_damage: bool,
}
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MoveCategory {
    Physical,
    Special,
    Status,
}
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Accuracy {
    Percent(u8),
    AlwaysHits,
}
/// The move flags the generator reads.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct MoveFlags(u8);
impl MoveFlags {
    pub const BITE: Self = Self(1);
    pub const PUNCH: Self = Self(1 << 1);
    pub const SOUND: Self = Self(1 << 2);
    pub const fn empty() -> Self {
        Self(0)
    }
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MultiHit {
    None,
    Fixed(u8),
    Range(u8, u8),
}
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Fraction {
    pub num: u8,
    pub den: u8,
}
/// The move has a secondary effect; what it is, the generator does not read.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Secondary;
