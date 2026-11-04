//! PS's `randomSet`: one species' set, from the choice of PS set to its
//! moves, ability, item, Tera type, level, EVs and IVs.
use rand::Rng;
use super::{Decide, RngDecide};
use crate::generated::ability_id::AbilityId;
use crate::generated::item_id::ItemId;
use crate::generated::move_id::MoveId;
use crate::generated::randbats_data::RoleId;
use crate::generated::species_id::SpeciesId;
use crate::moves::MoveCategory;
use crate::randbats::{RandbatsEntry, RandbatsRole};
use crate::pokemon::{EvSpread, Gender, IvSpread};
use crate::types::Type;
use super::ability::get_ability;
use super::counter::{deals_set_damage, query_moves};
use super::items::{get_item, get_priority_item};
use super::moveset::random_moveset;
use super::team::TeamDetails;
use super::{sample, species_ability_pool, species_types, type_effectiveness_score};
/// One generated set, PS's `RandomTeamsTypes.RandomSet`. The cosmetic
/// forme `getForme` picks (Vivillon's pattern, Pikachu's cap) is drawn but
/// not kept: `species` is the species `randomSet` was called with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RandomSet {
    pub species: SpeciesId,
    pub level: u8,
    pub gender: Gender,
    pub moves: [Option<MoveId>; 4],
    pub ability: AbilityId,
    pub item: ItemId,
    pub tera_type: Type,
    pub evs: EvSpread,
    pub ivs: IvSpread,
    /// The set's role; the generator always picks one.
    pub role: Option<RoleId>,
    pub shiny: bool,
}
/// Largest `sets` array in the randbats data, with headroom.
const MAX_ROLES: usize = 8;
/// `randomSet`'s choice of PS set: uniform over the sets that pass its
/// gates (no Fast Bulky Setup Paradox lead, no second Tera Blast user).
/// PS throws when none passes; the port falls back to every set.
fn pick_role<'a, R: Decide>(
    rng: &mut R,
    entry: &'a RandbatsEntry,
    is_lead: bool,
    team_details: &TeamDetails,
) -> &'a RandbatsRole {
    let mut possible = [0usize; MAX_ROLES];
    let mut n = 0;
    for (i, role) in entry.roles.iter().enumerate() {
        if is_lead && role.id == RoleId::FastBulkySetup
            && role
                .abilities
                .iter()
                .any(|a| {
                    matches!(a.id, AbilityId::Protosynthesis | AbilityId::Quarkdrive)
                })
        {
            continue;
        }
        if team_details.tera_blast && role.id == RoleId::TeraBlastUser {
            continue;
        }
        if let Some(slot) = possible.get_mut(n) {
            *slot = i;
            n += 1;
        }
    }
    debug_assert!(
        entry.roles.len() <= MAX_ROLES,
        "{:?} has more roles than the inline buffer holds", entry.species
    );
    if n == 0 {
        for (i, slot) in possible.iter_mut().enumerate().take(entry.roles.len()) {
            *slot = i;
        }
        n = entry.roles.len();
    }
    &entry.roles[sample(rng, &possible[..n])]
}
/// PS's `randomSet`: a set for `species`, drawn from `rng`.
pub fn random_set<R: Rng>(
    rng: &mut R,
    species: SpeciesId,
    team_details: &TeamDetails,
    is_lead: bool,
    is_doubles: bool,
) -> RandomSet {
    random_set_decide(&mut RngDecide(rng), species, team_details, is_lead, is_doubles)
}
/// [`random_set`] over any [`Decide`]: the one body behind the RNG entry
/// point, the replay of PS's draws and the exact enumerator.
pub(crate) fn random_set_decide<R: Decide>(
    rng: &mut R,
    species: SpeciesId,
    team_details: &TeamDetails,
    is_lead: bool,
    is_doubles: bool,
) -> RandomSet {
    let entry = species
        .randbats_entry()
        .expect("random_set: species not in randbats pool");
    if entry.forme_draw > 0 {
        rng.cosmetic_index(entry.forme_draw as usize);
    }
    let role = pick_role(rng, entry, is_lead, team_details);
    let role_id = role.id;
    let move_pool: Vec<MoveId> = role.moves.iter().map(|m| m.id).collect();
    let abilities: Vec<AbilityId> = role.abilities.iter().map(|a| a.id).collect();
    let abilities = if abilities.is_empty() {
        species_ability_pool(species)
    } else {
        abilities
    };
    let tera_pool: Vec<Type> = role.tera_types.iter().map(|t| t.typ).collect();
    let tera_type = if tera_pool.is_empty() {
        species.data().types.0
    } else {
        sample(rng, &tera_pool)
    };
    let types = species_types(species);
    let moves_vec = random_moveset(
        rng,
        &types,
        &abilities,
        team_details,
        species,
        is_lead,
        is_doubles,
        move_pool,
        tera_type,
        role_id,
    );
    let counter = query_moves(&moves_vec, species, tera_type, &abilities);
    let ability = get_ability(
        rng,
        &types,
        &moves_vec,
        &abilities,
        &counter,
        team_details,
        species,
        is_lead,
        is_doubles,
        tera_type,
        role_id,
    );
    let item = match get_priority_item(
        rng,
        ability,
        &types,
        &moves_vec,
        &counter,
        team_details,
        species,
        is_lead,
        is_doubles,
        tera_type,
        role_id,
    ) {
        Some(i) => i,
        None => {
            get_item(
                rng,
                ability,
                &types,
                &moves_vec,
                &counter,
                team_details,
                species,
                is_lead,
                tera_type,
                role_id,
            )
        }
    };
    let level = entry.level;
    let mut evs = EvSpread::new(85, 85, 85, 85, 85, 85);
    let mut ivs = IvSpread::perfect();
    let sr_immunity = ability == AbilityId::Magicguard || item == ItemId::Heavydutyboots;
    let mut sr_weakness = if sr_immunity {
        0
    } else {
        type_effectiveness_score(Type::Rock, species)
    };
    if [MoveId::Axekick, MoveId::Highjumpkick, MoveId::Jumpkick, MoveId::Supercellslam]
        .iter()
        .any(|m| moves_vec.contains(m))
    {
        sr_weakness = 2;
    }
    while evs.hp > 1 {
        let base_hp = species.data().base_stats.hp as i32;
        let hp = ((2 * base_hp + ivs.hp as i32 + (evs.hp / 4) as i32 + 100)
            * level as i32 / 100 + 10) as u32;
        if (moves_vec.contains(&MoveId::Substitute) && item == ItemId::Sitrusberry)
            || species == SpeciesId::Minior
        {
            if hp % 4 == 0 {
                break;
            }
        } else if (moves_vec.contains(&MoveId::Bellydrum)
            || moves_vec.contains(&MoveId::Filletaway)
            || moves_vec.contains(&MoveId::Shedtail))
            && (item == ItemId::Sitrusberry || ability == AbilityId::Gluttony)
        {
            if hp % 2 == 0 {
                break;
            }
        } else if moves_vec.contains(&MoveId::Substitute)
            && moves_vec.contains(&MoveId::Endeavor)
        {
            if hp % 4 > 0 {
                break;
            }
        } else {
            if is_doubles {
                break;
            }
            if sr_weakness <= 0 || ability == AbilityId::Regenerator
                || matches!(item, ItemId::Leftovers | ItemId::Lifeorb)
            {
                break;
            }
            let denom = (4 / sr_weakness.max(1)) as u32;
            if denom > 0 {
                if item != ItemId::Sitrusberry && hp % denom > 0 {
                    break;
                }
                if item == ItemId::Sitrusberry && hp % denom == 0 {
                    break;
                }
            }
        }
        evs.hp = evs.hp.saturating_sub(4);
    }
    if no_attack_stat_moves(&moves_vec, species, ability)
        && !moves_vec.contains(&MoveId::Transform)
    {
        evs.atk = 0;
        ivs.atk = 0;
    }
    if moves_vec.contains(&MoveId::Gyroball) || moves_vec.contains(&MoveId::Trickroom) {
        evs.spe = 0;
        ivs.spe = 0;
    }
    let mut shuffled: Vec<MoveId> = moves_vec.into_iter().take(4).collect();
    rng.shuffle_moves(&mut shuffled);
    let gender = if species.data().base_species == SpeciesId::Greninja {
        Gender::Male
    } else {
        match species.data().gender_ratio {
            crate::species::GenderRatio::Genderless => Gender::Genderless,
            crate::species::GenderRatio::AlwaysMale => Gender::Male,
            crate::species::GenderRatio::AlwaysFemale => Gender::Female,
            _ => rng.gender(4),
        }
    };
    let shiny = rng.cosmetic_chance(1, 1024);
    let mut moves_arr: [Option<MoveId>; 4] = [None; 4];
    for (i, m) in shuffled.into_iter().enumerate() {
        moves_arr[i] = Some(m);
    }
    let species = crate::species::crowned_forme(species, item);
    if species == SpeciesId::Zaciancrowned || species == SpeciesId::Zamazentacrowned {
        let replacement = if species == SpeciesId::Zaciancrowned {
            MoveId::Behemothblade
        } else {
            MoveId::Behemothbash
        };
        for m in moves_arr.iter_mut() {
            if *m == Some(MoveId::Ironhead) {
                *m = Some(replacement);
            }
        }
    }
    RandomSet {
        species,
        level,
        gender,
        moves: moves_arr,
        ability,
        item,
        tera_type,
        evs,
        ivs,
        role: Some(role_id),
        shiny,
    }
}
/// PS's `noAttackStatMoves`: no move in the set reads the user's Attack.
fn no_attack_stat_moves(
    moves: &[MoveId],
    species: SpeciesId,
    ability: AbilityId,
) -> bool {
    moves
        .iter()
        .all(|&m| {
            let mv = m.data();
            if deals_set_damage(m) {
                return true;
            }
            if m == MoveId::Shellsidearm {
                return false;
            }
            if m == MoveId::Terablast
                && (species == SpeciesId::Porygon2
                    || matches!(ability, AbilityId::Contrary | AbilityId::Defiant)
                    || moves.contains(&MoveId::Shiftgear)
                    || species.data().base_stats.atk > species.data().base_stats.spa)
            {
                return false;
            }
            mv.category != MoveCategory::Physical
                || matches!(m, MoveId::Bodypress | MoveId::Foulplay)
        })
}
