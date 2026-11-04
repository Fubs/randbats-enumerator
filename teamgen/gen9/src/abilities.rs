//! Ability data.
use crate::generated::ability_id::AbilityId;
#[derive(Copy, Clone, Debug)]
pub struct AbilityMetadata {
    pub id: AbilityId,
    pub name: &'static str,
}
