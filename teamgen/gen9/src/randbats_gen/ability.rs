//! PS's `getAbility` and `shouldCullAbility`.
use super::Decide;
use crate::generated::ability_id::AbilityId;
use crate::generated::move_id::MoveId;
use crate::generated::species_id::SpeciesId;
use crate::types::{Effectiveness, Type};
use crate::generated::randbats_data::RoleId;
use super::counter::{CounterKey, MoveCounter};
use super::sample;
use super::team::TeamDetails;
/// PS's `shouldCullAbility`: per-ability "is this ability bad for the
/// current set?" check.
#[allow(clippy::too_many_arguments)]
pub fn should_cull_ability(
    ability: AbilityId,
    types: &[Type],
    moves: &[MoveId],
    abilities: &[AbilityId],
    counter: &MoveCounter,
    team_details: &TeamDetails,
    species: SpeciesId,
    is_lead: bool,
    is_doubles: bool,
    tera_type: Type,
    role: RoleId,
) -> bool {
    let _ = (types, abilities, is_lead, is_doubles, tera_type, role);
    match ability {
        AbilityId::Chlorophyll | AbilityId::Solarpower => !team_details.sun,
        AbilityId::Defiant => {
            species == SpeciesId::Thundurus && counter.get(CounterKey::Status) > 0
        }
        AbilityId::Hydration | AbilityId::Swiftswim => !team_details.rain,
        AbilityId::Ironfist => counter.get(CounterKey::Ironfist) == 0,
        AbilityId::Skilllink => counter.get(CounterKey::Skilllink) == 0,
        AbilityId::Overgrow => counter.get(CounterKey::Grass) == 0,
        AbilityId::Prankster => counter.get(CounterKey::Status) == 0,
        AbilityId::Sandforce | AbilityId::Sandrush => !team_details.sand,
        AbilityId::Slushrush => !team_details.snow,
        AbilityId::Swarm => counter.get(CounterKey::Bug) == 0,
        AbilityId::Torrent => {
            counter.get(CounterKey::Water) == 0 && !moves.contains(&MoveId::Flipturn)
        }
        _ => false,
    }
}
/// PS's `getAbility`: pick the most appropriate ability from the role's
/// pool given the current moveset / team / species hardcodes.
#[allow(clippy::too_many_arguments)]
pub fn get_ability<R: Decide>(
    rng: &mut R,
    types: &[Type],
    moves: &[MoveId],
    abilities: &[AbilityId],
    counter: &MoveCounter,
    team_details: &TeamDetails,
    species: SpeciesId,
    is_lead: bool,
    is_doubles: bool,
    tera_type: Type,
    role: RoleId,
) -> AbilityId {
    if abilities.len() <= 1 {
        return abilities.first().copied().unwrap_or(AbilityId::None);
    }
    if species == SpeciesId::Drifblim {
        return if moves.contains(&MoveId::Defog) {
            AbilityId::Aftermath
        } else {
            AbilityId::Unburden
        };
    }
    if abilities.contains(&AbilityId::Flashfire)
        && tera_effectiveness(Type::Fire, tera_type) >= 1
    {
        return AbilityId::Flashfire;
    }
    if (species == SpeciesId::Thundurus || species == SpeciesId::Tornadus)
        && counter.get(CounterKey::Physical) == 0
    {
        return AbilityId::Prankster;
    }
    if species == SpeciesId::Swampert
        && (counter.get(CounterKey::Water) > 0 || moves.contains(&MoveId::Flipturn))
    {
        return AbilityId::Torrent;
    }
    if species == SpeciesId::Toucannon && counter.get(CounterKey::Skilllink) > 0 {
        return AbilityId::Skilllink;
    }
    if abilities.contains(&AbilityId::Slushrush) && moves.contains(&MoveId::Snowscape) {
        return AbilityId::Slushrush;
    }
    if species == SpeciesId::Golduck && team_details.rain {
        return AbilityId::Swiftswim;
    }
    let mut allowed: Vec<AbilityId> = Vec::new();
    for &ability in abilities {
        if !should_cull_ability(
            ability,
            types,
            moves,
            abilities,
            counter,
            team_details,
            species,
            is_lead,
            is_doubles,
            tera_type,
            role,
        ) {
            allowed.push(ability);
        }
    }
    if !allowed.is_empty() {
        return sample(rng, &allowed);
    }
    let weather_abilities: Vec<AbilityId> = abilities
        .iter()
        .copied()
        .filter(|a| {
            matches!(
                * a, AbilityId::Chlorophyll | AbilityId::Hydration | AbilityId::Sandforce
                | AbilityId::Sandrush | AbilityId::Slushrush | AbilityId::Solarpower |
                AbilityId::Swiftswim
            )
        })
        .collect();
    if !weather_abilities.is_empty() {
        return sample(rng, &weather_abilities);
    }
    sample(rng, abilities)
}
/// PS's `dex.getEffectiveness(attack_type, teraType)` against one type. An
/// immunity scores −99 here and 0 in PS; the one caller, Flash Fire's
/// check, asks about Fire, which no type is immune to.
fn tera_effectiveness(attack_type: Type, tera_type: Type) -> i32 {
    match attack_type.effectiveness_against(tera_type) {
        Effectiveness::Immune => -99,
        Effectiveness::Resistant => -1,
        Effectiveness::Neutral => 0,
        Effectiveness::SuperEffective => 1,
    }
}
