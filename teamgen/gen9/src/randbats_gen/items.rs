//! PS's `getPriorityItem` and `getItem`, singles only.
use super::Decide;
use crate::generated::ability_id::AbilityId;
use crate::generated::item_id::ItemId;
use crate::generated::move_id::MoveId;
use crate::generated::species_id::SpeciesId;
use crate::types::Type;
use crate::generated::randbats_data::RoleId;
use super::counter::{CounterKey, MoveCounter};
use super::team::TeamDetails;
use super::{
    DEFENSIVE_TERA_BLAST_USERS, PIVOT_MOVES, random_chance, sample, species_is_nfe,
    type_effectiveness_score,
};
/// PS's `getPriorityItem`: the items particular abilities, species and
/// sets call for, ahead of `getItem`'s general rules.
#[allow(clippy::too_many_arguments)]
pub fn get_priority_item<R: Decide>(
    rng: &mut R,
    ability: AbilityId,
    types: &[Type],
    moves: &[MoveId],
    counter: &MoveCounter,
    team_details: &TeamDetails,
    species: SpeciesId,
    is_lead: bool,
    is_doubles: bool,
    tera_type: Type,
    role: RoleId,
) -> Option<ItemId> {
    let _ = (tera_type, is_lead);
    if !is_doubles {
        if role == RoleId::FastBulkySetup
            && (ability == AbilityId::Quarkdrive || ability == AbilityId::Protosynthesis)
        {
            return Some(ItemId::Boosterenergy);
        }
        if species == SpeciesId::Lokix {
            return Some(
                if role == RoleId::FastAttacker {
                    ItemId::Silverpowder
                } else {
                    ItemId::Lifeorb
                },
            );
        }
    }
    let required_items = species.data().required_items;
    if !required_items.is_empty() {
        if species.data().base_species == SpeciesId::Arceus {
            return Some(required_items[0]);
        }
        return Some(sample(rng, required_items));
    }
    if let Some(item) = species.data().required_item {
        return Some(item);
    }
    if species == SpeciesId::Pikachu {
        return Some(ItemId::Lightball);
    }
    if role == RoleId::AvPivot {
        return Some(ItemId::Assaultvest);
    }
    if species == SpeciesId::Regieleki {
        return Some(ItemId::Magnet);
    }
    if types.contains(&Type::Normal) && moves.contains(&MoveId::Doubleedge)
        && moves.contains(&MoveId::Fakeout)
    {
        return Some(ItemId::Silkscarf);
    }
    if species == SpeciesId::Froslass || moves.contains(&MoveId::Populationbomb)
        || (ability == AbilityId::Hustle && counter.get(CounterKey::Setup) > 0
            && !is_doubles && random_chance(rng, 1, 2))
    {
        return Some(ItemId::Widelens);
    }
    if species == SpeciesId::Smeargle {
        return Some(ItemId::Focussash);
    }
    if moves.contains(&MoveId::Clangoroussoul)
        || (species == SpeciesId::Toxtricity && moves.contains(&MoveId::Shiftgear))
    {
        return Some(ItemId::Throatspray);
    }
    if (species.data().base_species == SpeciesId::Magearna
        && role == RoleId::TeraBlastUser)
        || ((species == SpeciesId::Calyrexice || species == SpeciesId::Necrozmaduskmane)
            && is_doubles)
    {
        return Some(ItemId::Weaknesspolicy);
    }
    if [MoveId::Dragonenergy, MoveId::Lastrespects, MoveId::Waterspout]
        .iter()
        .any(|m| moves.contains(m))
    {
        return Some(ItemId::Choicescarf);
    }
    if !is_doubles
        && (ability == AbilityId::Imposter
            || (species == SpeciesId::Magnezone && role == RoleId::FastAttacker))
    {
        return Some(ItemId::Choicescarf);
    }
    if species == SpeciesId::Rampardos && (role == RoleId::FastAttacker || is_doubles) {
        return Some(ItemId::Choicescarf);
    }
    if species == SpeciesId::Palkia && counter.get(CounterKey::Status) > 0 {
        return Some(ItemId::Lustrousorb);
    }
    if moves.contains(&MoveId::Courtchange)
        || (!is_doubles
            && (species == SpeciesId::Luvdisc
                || (species == SpeciesId::Terapagos && !moves.contains(&MoveId::Rest))))
    {
        return Some(ItemId::Heavydutyboots);
    }
    if matches!(
        ability, AbilityId::Cheekpouch | AbilityId::Cudchew | AbilityId::Harvest |
        AbilityId::Ripen
    ) || moves.contains(&MoveId::Bellydrum) || moves.contains(&MoveId::Filletaway)
    {
        return Some(ItemId::Sitrusberry);
    }
    if [MoveId::Healingwish, MoveId::Switcheroo, MoveId::Trick]
        .iter()
        .any(|m| moves.contains(m))
    {
        let spe = species.data().base_stats.spe;
        if (60..=108).contains(&spe) && role != RoleId::Wallbreaker
            && counter.get(CounterKey::Priority) == 0
        {
            return Some(ItemId::Choicescarf);
        }
        return Some(
            if counter.get(CounterKey::Physical) > counter.get(CounterKey::Special) {
                ItemId::Choiceband
            } else {
                ItemId::Choicespecs
            },
        );
    }
    if counter.get(CounterKey::Status) > 0
        && (species == SpeciesId::Latias || species == SpeciesId::Latios)
    {
        return Some(ItemId::Souldew);
    }
    if species == SpeciesId::Scyther && !is_doubles {
        return Some(
            if is_lead && !moves.contains(&MoveId::Uturn) {
                ItemId::Eviolite
            } else {
                ItemId::Heavydutyboots
            },
        );
    }
    if ability == AbilityId::Poisonheal || ability == AbilityId::Quickfeet {
        return Some(ItemId::Toxicorb);
    }
    if species_is_nfe(species) {
        return Some(ItemId::Eviolite);
    }
    if (ability == AbilityId::Guts || moves.contains(&MoveId::Facade))
        && !moves.contains(&MoveId::Sleeptalk)
    {
        return Some(
            if types.contains(&Type::Fire) || ability == AbilityId::Toxicboost {
                ItemId::Toxicorb
            } else {
                ItemId::Flameorb
            },
        );
    }
    if ability == AbilityId::Magicguard
        || (ability == AbilityId::Sheerforce && counter.get(CounterKey::Sheerforce) > 0)
    {
        return Some(ItemId::Lifeorb);
    }
    if ability == AbilityId::Angershell {
        return Some(
            sample(
                rng,
                &[
                    ItemId::Expertbelt,
                    ItemId::Lumberry,
                    ItemId::Scopelens,
                    ItemId::Sitrusberry,
                ],
            ),
        );
    }
    if moves.contains(&MoveId::Dragondance) && is_doubles {
        return Some(ItemId::Clearamulet);
    }
    if counter.get(CounterKey::Skilllink) > 0 && ability != AbilityId::Skilllink
        && species != SpeciesId::Breloom
    {
        return Some(ItemId::Loadeddice);
    }
    if ability == AbilityId::Unburden {
        return Some(
            if moves.contains(&MoveId::Closecombat) || moves.contains(&MoveId::Leafstorm)
            {
                ItemId::Whiteherb
            } else {
                ItemId::Sitrusberry
            },
        );
    }
    if moves.contains(&MoveId::Shellsmash) && ability != AbilityId::Weakarmor {
        return Some(ItemId::Whiteherb);
    }
    if moves.contains(&MoveId::Meteorbeam)
        || (moves.contains(&MoveId::Electroshot) && !team_details.rain)
    {
        return Some(ItemId::Powerherb);
    }
    if moves.contains(&MoveId::Acrobatics) && ability != AbilityId::Protosynthesis {
        return Some(ItemId::None);
    }
    if moves.contains(&MoveId::Auroraveil)
        || (moves.contains(&MoveId::Lightscreen) && moves.contains(&MoveId::Reflect))
    {
        return Some(ItemId::Lightclay);
    }
    if ability == AbilityId::Gluttony {
        let pick = sample(
            rng,
            &[
                ItemId::Aguavberry,
                ItemId::Figyberry,
                ItemId::Iapapaberry,
                ItemId::Magoberry,
                ItemId::Wikiberry,
            ],
        );
        return Some(pick);
    }
    if species == SpeciesId::Giratina && !is_doubles && moves.contains(&MoveId::Rest)
        && !moves.contains(&MoveId::Sleeptalk)
    {
        return Some(ItemId::Leftovers);
    }
    if moves.contains(&MoveId::Rest) && !moves.contains(&MoveId::Sleeptalk)
        && ability != AbilityId::Naturalcure && ability != AbilityId::Shedskin
    {
        return Some(ItemId::Chestoberry);
    }
    if species != SpeciesId::Yanmega
        && type_effectiveness_score(Type::Rock, species) >= 2
        && (!types.contains(&Type::Flying) || !is_doubles)
    {
        return Some(ItemId::Heavydutyboots);
    }
    None
}
/// PS's `getItem`: the item when `getPriorityItem` returns none.
#[allow(clippy::too_many_arguments)]
pub fn get_item<R: Decide>(
    rng: &mut R,
    ability: AbilityId,
    types: &[Type],
    moves: &[MoveId],
    counter: &MoveCounter,
    team_details: &TeamDetails,
    species: SpeciesId,
    is_lead: bool,
    tera_type: Type,
    role: RoleId,
) -> ItemId {
    let _ = (team_details, tera_type);
    let life_orb_reqs = [MoveId::Flamecharge, MoveId::Nuzzle, MoveId::Rapidspin]
        .iter()
        .all(|m| !moves.contains(m));
    let total_moves = moves.len() as u32;
    if species != SpeciesId::Jirachi && counter.get(CounterKey::Physical) >= total_moves
        && [
            MoveId::Dragontail,
            MoveId::Fakeout,
            MoveId::Firstimpression,
            MoveId::Flamecharge,
            MoveId::Rapidspin,
            MoveId::Trailblaze,
        ]
            .iter()
            .all(|m| !moves.contains(m))
    {
        let scarf_reqs = role != RoleId::Wallbreaker
            && (species.data().base_stats.atk >= 100 || ability == AbilityId::Hugepower
                || ability == AbilityId::Purepower)
            && (60..=108).contains(&species.data().base_stats.spe)
            && ability != AbilityId::Speedboost
            && counter.get(CounterKey::Priority) == 0;
        return if scarf_reqs && random_chance(rng, 1, 2) {
            ItemId::Choicescarf
        } else {
            ItemId::Choiceband
        };
    }
    if counter.get(CounterKey::Special) >= total_moves
        || (counter.get(CounterKey::Special) >= total_moves.saturating_sub(1)
            && [MoveId::Flipturn, MoveId::Uturn].iter().any(|m| moves.contains(m)))
    {
        let scarf_reqs = role != RoleId::Wallbreaker
            && species.data().base_stats.spa >= 100
            && (60..=108).contains(&species.data().base_stats.spe)
            && ability != AbilityId::Speedboost && ability != AbilityId::Tintedlens
            && !moves.contains(&MoveId::Uturn) && counter.get(CounterKey::Priority) == 0;
        return if scarf_reqs && random_chance(rng, 1, 2) {
            ItemId::Choicescarf
        } else {
            ItemId::Choicespecs
        };
    }
    if counter.get(CounterKey::Speedsetup) > 0
        && counter.get(CounterKey::Physicalsetup) == 0 && role == RoleId::BulkySetup
    {
        return ItemId::Weaknesspolicy;
    }
    if counter.get(CounterKey::Status) == 0
        && !matches!(
            role, RoleId::FastAttacker | RoleId::Wallbreaker | RoleId::TeraBlastUser
        )
    {
        return ItemId::Assaultvest;
    }
    if species == SpeciesId::Golem {
        return if counter.get(CounterKey::Speedsetup) > 0 {
            ItemId::Weaknesspolicy
        } else {
            ItemId::Custapberry
        };
    }
    if moves.contains(&MoveId::Substitute) {
        return ItemId::Leftovers;
    }
    if moves.contains(&MoveId::Stickyweb) && is_lead
        && species.data().base_stats.hp as u32 + species.data().base_stats.def as u32
            + species.data().base_stats.spd as u32 <= 235
    {
        return ItemId::Focussash;
    }
    if type_effectiveness_score(Type::Rock, species) >= 1 {
        return ItemId::Heavydutyboots;
    }
    if moves.contains(&MoveId::Chillyreception)
        || (role == RoleId::FastSupport
            && PIVOT_MOVES
                .iter()
                .chain([MoveId::Defog, MoveId::Mortalspin, MoveId::Rapidspin].iter())
                .any(|m| moves.contains(m)) && !types.contains(&Type::Flying)
            && ability != AbilityId::Levitate)
    {
        return ItemId::Heavydutyboots;
    }
    if moves.contains(&MoveId::Dragondance) && role == RoleId::BulkySetup {
        return ItemId::Weaknesspolicy;
    }
    if ability == AbilityId::Roughskin
        || (ability == AbilityId::Regenerator
            && (role == RoleId::BulkySupport || role == RoleId::BulkyAttacker)
            && species.data().base_stats.hp as u32 + species.data().base_stats.def as u32
                >= 180 && random_chance(rng, 1, 2))
        || (ability != AbilityId::Regenerator && counter.get(CounterKey::Setup) == 0
            && counter.get(CounterKey::Recovery) > 0
            && type_effectiveness_score(Type::Fighting, species) < 1
            && species.data().base_stats.hp as u32 + species.data().base_stats.def as u32
                > 200 && random_chance(rng, 1, 2))
    {
        return ItemId::Rockyhelmet;
    }
    if moves.contains(&MoveId::Outrage) && counter.get(CounterKey::Setup) > 0 {
        return ItemId::Lumberry;
    }
    if moves.contains(&MoveId::Protect) && ability != AbilityId::Speedboost {
        return ItemId::Leftovers;
    }
    if role == RoleId::FastSupport && is_lead && counter.get(CounterKey::Recovery) == 0
        && counter.get(CounterKey::Recoil) == 0
        && (counter.get(CounterKey::Hazards) > 0 || counter.get(CounterKey::Setup) > 0)
        && (species.data().base_stats.hp as u32 + species.data().base_stats.def as u32
            + species.data().base_stats.spd as u32) < 258
    {
        return ItemId::Focussash;
    }
    if counter.get(CounterKey::Setup) == 0 && ability != AbilityId::Levitate
        && type_effectiveness_score(Type::Ground, species) >= 2
    {
        return ItemId::Airballoon;
    }
    if matches!(
        role, RoleId::BulkyAttacker | RoleId::BulkySupport | RoleId::BulkySetup
    ) {
        return ItemId::Leftovers;
    }
    if species == SpeciesId::Pawmot && moves.contains(&MoveId::Nuzzle) {
        return ItemId::Leppaberry;
    }
    if role == RoleId::FastSupport || role == RoleId::FastBulkySetup {
        return if counter.get(CounterKey::Physical) + counter.get(CounterKey::Special)
            > counter.get(CounterKey::Status) && life_orb_reqs
        {
            ItemId::Lifeorb
        } else {
            ItemId::Leftovers
        };
    }
    if role == RoleId::TeraBlastUser && DEFENSIVE_TERA_BLAST_USERS.contains(&species) {
        return ItemId::Leftovers;
    }
    if life_orb_reqs
        && matches!(
            role, RoleId::FastAttacker | RoleId::SetupSweeper | RoleId::TeraBlastUser |
            RoleId::Wallbreaker
        )
    {
        return ItemId::Lifeorb;
    }
    ItemId::Leftovers
}
