//! PS's `randomMoveset` and what it calls: `cullMovePool`, `addMove`,
//! `incompatibleMoves` and the per-type `moveEnforcementCheckers`.
use super::Decide;
use crate::generated::ability_id::AbilityId;
use crate::generated::move_id::MoveId;
use crate::generated::species_id::SpeciesId;
use crate::moves::MoveCategory;
use crate::types::Type;
use crate::generated::randbats_data::RoleId;
use super::counter::{CounterKey, MoveCounter, get_move_type, query_moves};
use super::team::TeamDetails;
use super::{
    HAZARDS, MOVE_PAIRS, NO_STAB, PHYSICAL_SETUP, PIVOT_MOVES, PRIORITY_POKEMON,
    RECOVERY_MOVES, SETUP, SPEED_SETUP, fast_pop, remove_if_present, sample,
    sample_no_replace,
};
/// Maximum number of moves on a randbats set.
pub const MAX_MOVE_COUNT: usize = 4;
/// PS's `runEnforcementChecker` for a type: whether `randomMoveset` should
/// still add a STAB move of `typ`.
pub fn run_enforcement_checker(
    typ: Type,
    move_pool: &[MoveId],
    moves: &[MoveId],
    abilities: &[AbilityId],
    types: &[Type],
    counter: &MoveCounter,
    species: SpeciesId,
    team_details: &TeamDetails,
    is_lead: bool,
    is_doubles: bool,
    role: RoleId,
) -> bool {
    let _ = (moves, team_details, is_lead, role);
    match typ {
        Type::Bug => {
            move_pool.contains(&MoveId::Megahorn)
                || move_pool.contains(&MoveId::Xscissor)
                || (counter.get(CounterKey::Bug) == 0
                    && (types.contains(&Type::Electric)
                        || types.contains(&Type::Psychic)))
        }
        Type::Dark => {
            if counter.get(CounterKey::Dark) < 2 && PRIORITY_POKEMON.contains(&species)
                && role == RoleId::Wallbreaker
            {
                return true;
            }
            counter.get(CounterKey::Dark) == 0
        }
        Type::Dragon => counter.get(CounterKey::Dragon) == 0,
        Type::Electric => counter.get(CounterKey::Electric) == 0,
        Type::Fairy => counter.get(CounterKey::Fairy) == 0,
        Type::Fighting => counter.get(CounterKey::Fighting) == 0,
        Type::Fire => counter.get(CounterKey::Fire) == 0,
        Type::Flying => counter.get(CounterKey::Flying) == 0,
        Type::Ghost => counter.get(CounterKey::Ghost) == 0,
        Type::Grass => {
            counter.get(CounterKey::Grass) == 0
                && (move_pool.contains(&MoveId::Leafstorm)
                    || species.data().base_stats.atk >= 100
                    || types.contains(&Type::Electric)
                    || abilities.contains(&AbilityId::Seedsower))
        }
        Type::Ground => counter.get(CounterKey::Ground) == 0,
        Type::Ice => {
            move_pool.contains(&MoveId::Freezedry)
                || move_pool.contains(&MoveId::Blizzard)
                || counter.get(CounterKey::Ice) == 0
        }
        Type::Normal => {
            move_pool.contains(&MoveId::Boomburst)
                || move_pool.contains(&MoveId::Hypervoice)
        }
        Type::Poison => {
            if types.contains(&Type::Ground) {
                return false;
            }
            counter.get(CounterKey::Poison) == 0
        }
        Type::Psychic => {
            if (is_doubles || species == SpeciesId::Bruxish)
                && move_pool.contains(&MoveId::Psychicfangs)
            {
                return true;
            }
            if species == SpeciesId::Hoopaunbound && move_pool.contains(&MoveId::Psychic)
            {
                return true;
            }
            if [Type::Dark, Type::Steel, Type::Water].iter().any(|m| types.contains(m)) {
                return false;
            }
            counter.get(CounterKey::Psychic) == 0
        }
        Type::Rock => {
            counter.get(CounterKey::Rock) == 0 && species.data().base_stats.atk >= 80
        }
        Type::Steel => {
            counter.get(CounterKey::Steel) == 0
                && (is_doubles || species.data().base_stats.atk >= 90
                    || move_pool.contains(&MoveId::Gigatonhammer)
                    || move_pool.contains(&MoveId::Makeitrain))
        }
        Type::Water => {
            counter.get(CounterKey::Water) == 0
                && (!types.contains(&Type::Ground) || is_doubles)
        }
        _ => false,
    }
}
/// Membership of the moves chosen so far, as bits: `cull_move_pool` asks
/// every incompatible pair whether it touches one.
pub struct MoveBits([u64; MoveId::COUNT.div_ceil(64) + 1]);
impl MoveBits {
    pub fn of(moves: &[MoveId]) -> Self {
        let mut bits = Self([0; MoveId::COUNT.div_ceil(64) + 1]);
        for &m in moves {
            bits.0[m as usize / 64] |= 1 << (m as usize % 64);
        }
        bits
    }
    pub fn has(&self, m: MoveId) -> bool {
        self.0[m as usize / 64] >> (m as usize % 64) & 1 != 0
    }
}
/// A PS parameter that is a move id or a list of them, read in place: the
/// selector is borrowed, never collected, since `incompatible_moves` runs
/// dozens of times per cull.
pub trait MoveSelector {
    fn as_moves(&self) -> &[MoveId];
}
impl MoveSelector for MoveId {
    fn as_moves(&self) -> &[MoveId] {
        std::slice::from_ref(self)
    }
}
impl MoveSelector for &[MoveId] {
    fn as_moves(&self) -> &[MoveId] {
        self
    }
}
impl<const N: usize> MoveSelector for [MoveId; N] {
    fn as_moves(&self) -> &[MoveId] {
        self
    }
}
/// PS's `incompatibleMoves`: when one move from `moves_a` is already
/// chosen, remove every move in `moves_b` from the pool, and vice versa,
/// while the pool and the moves together exceed `MAX_MOVE_COUNT`.
pub fn incompatible_moves<A: MoveSelector, B: MoveSelector>(
    chosen: &MoveBits,
    moves: &[MoveId],
    move_pool: &mut Vec<MoveId>,
    moves_a: A,
    moves_b: B,
) {
    let arr_a = moves_a.as_moves();
    let arr_b = moves_b.as_moves();
    if !arr_a.iter().chain(arr_b).any(|&m| chosen.has(m)) {
        return;
    }
    if moves.len() + move_pool.len() <= MAX_MOVE_COUNT {
        return;
    }
    for &moveid1 in moves {
        if arr_b.contains(&moveid1) {
            for &moveid2 in arr_a {
                if moveid1 != moveid2 {
                    if let Some(i) = move_pool.iter().position(|m| *m == moveid2) {
                        fast_pop(move_pool, i);
                        if moves.len() + move_pool.len() <= MAX_MOVE_COUNT {
                            return;
                        }
                    }
                }
            }
        }
        if arr_a.contains(&moveid1) {
            for &moveid2 in arr_b {
                if moveid1 != moveid2 {
                    if let Some(i) = move_pool.iter().position(|m| *m == moveid2) {
                        fast_pop(move_pool, i);
                        if moves.len() + move_pool.len() <= MAX_MOVE_COUNT {
                            return;
                        }
                    }
                }
            }
        }
    }
}
/// PS's `cullMovePool`: trim the move pool down toward `MAX_MOVE_COUNT`
/// by removing redundant / incompatible / team-already-covered options.
#[allow(clippy::too_many_arguments)]
pub fn cull_move_pool(
    types: &[Type],
    moves: &[MoveId],
    abilities: &[AbilityId],
    counter: &MoveCounter,
    move_pool: &mut Vec<MoveId>,
    team_details: &TeamDetails,
    species: SpeciesId,
    is_lead: bool,
    is_doubles: bool,
    tera_type: Type,
    _role: RoleId,
) {
    if moves.len() + move_pool.len() <= MAX_MOVE_COUNT {
        return;
    }
    let chosen = MoveBits::of(moves);
    if moves.len() == MAX_MOVE_COUNT - 2 {
        let mut unpaired = move_pool.clone();
        for (a, b) in MOVE_PAIRS {
            if move_pool.contains(a) && move_pool.contains(b) {
                if let Some(i) = unpaired.iter().position(|m| m == a) {
                    fast_pop(&mut unpaired, i);
                }
                if let Some(i) = unpaired.iter().position(|m| m == b) {
                    fast_pop(&mut unpaired, i);
                }
            }
        }
        if unpaired.len() == 1 {
            remove_if_present(move_pool, unpaired[0]);
        }
    }
    if moves.len() == MAX_MOVE_COUNT - 1 {
        for (a, b) in MOVE_PAIRS {
            if move_pool.contains(a) && move_pool.contains(b) {
                remove_if_present(move_pool, *a);
                remove_if_present(move_pool, *b);
            }
        }
    }
    if team_details.screens {
        if !is_doubles {
            remove_if_present(move_pool, MoveId::Auroraveil);
        }
        if move_pool.len() >= MAX_MOVE_COUNT + 2 {
            remove_if_present(move_pool, MoveId::Reflect);
            remove_if_present(move_pool, MoveId::Lightscreen);
        }
    }
    if team_details.sticky_web {
        remove_if_present(move_pool, MoveId::Stickyweb);
        if moves.len() + move_pool.len() <= MAX_MOVE_COUNT {
            return;
        }
    }
    if team_details.stealth_rock {
        remove_if_present(move_pool, MoveId::Stealthrock);
        if moves.len() + move_pool.len() <= MAX_MOVE_COUNT {
            return;
        }
    }
    if team_details.defog || team_details.rapid_spin {
        remove_if_present(move_pool, MoveId::Defog);
        remove_if_present(move_pool, MoveId::Rapidspin);
        if moves.len() + move_pool.len() <= MAX_MOVE_COUNT {
            return;
        }
    }
    if team_details.toxic_spikes {
        remove_if_present(move_pool, MoveId::Toxicspikes);
        if moves.len() + move_pool.len() <= MAX_MOVE_COUNT {
            return;
        }
    }
    if team_details.spikes >= 2 {
        remove_if_present(move_pool, MoveId::Spikes);
        if moves.len() + move_pool.len() <= MAX_MOVE_COUNT {
            return;
        }
    }
    if team_details.status_cure {
        remove_if_present(move_pool, MoveId::Healbell);
        if moves.len() + move_pool.len() <= MAX_MOVE_COUNT {
            return;
        }
    }
    if moves.len() + move_pool.len() > MAX_MOVE_COUNT {
        let trick_set: &[MoveId] = &[
            MoveId::Healingwish,
            MoveId::Switcheroo,
            MoveId::Trick,
        ];
        for &m1 in moves {
            if trick_set.contains(&m1) {
                let mut i = 0;
                while i < move_pool.len() {
                    let m2 = move_pool[i];
                    if m2 != m1 && m2.data().category == MoveCategory::Status {
                        fast_pop(move_pool, i);
                        if moves.len() + move_pool.len() <= MAX_MOVE_COUNT {
                            return;
                        }
                    } else {
                        i += 1;
                    }
                }
            }
            if m1.data().category == MoveCategory::Status {
                for &m2 in trick_set {
                    if m1 != m2 {
                        if let Some(i) = move_pool.iter().position(|p| *p == m2) {
                            fast_pop(move_pool, i);
                            if moves.len() + move_pool.len() <= MAX_MOVE_COUNT {
                                return;
                            }
                        }
                    }
                }
            }
        }
    }
    static PAIRS: &[(&[MoveId], &[MoveId])] = &[
        (SETUP, PIVOT_MOVES),
        (SETUP, HAZARDS),
        (
            SETUP,
            &[MoveId::Defog, MoveId::Nuzzle, MoveId::Toxic, MoveId::Yawn, MoveId::Haze],
        ),
        (PHYSICAL_SETUP, PHYSICAL_SETUP),
        (&[MoveId::Substitute], PIVOT_MOVES),
        (SPEED_SETUP, &[MoveId::Aquajet, MoveId::Rest, MoveId::Trickroom]),
        (&[MoveId::Curse], &[MoveId::Irondefense, MoveId::Rapidspin]),
        (&[MoveId::Dragondance], &[MoveId::Dracometeor]),
        (&[MoveId::Yawn], &[MoveId::Roar]),
        (&[MoveId::Trick], &[MoveId::Uturn]),
        (
            &[MoveId::Psychic, MoveId::Psychicnoise],
            &[MoveId::Psyshock, MoveId::Psychicnoise],
        ),
        (&[MoveId::Surf], &[MoveId::Hydropump]),
        (&[MoveId::Liquidation], &[MoveId::Wavecrash]),
        (&[MoveId::Aquajet], &[MoveId::Flipturn]),
        (&[MoveId::Gigadrain], &[MoveId::Leafstorm]),
        (&[MoveId::Powerwhip], &[MoveId::Hornleech]),
        (&[MoveId::Airslash], &[MoveId::Hurricane]),
        (&[MoveId::Knockoff], &[MoveId::Foulplay]),
        (&[MoveId::Throatchop], &[MoveId::Crunch, MoveId::Lashout]),
        (&[MoveId::Doubleedge], &[MoveId::Bodyslam, MoveId::Headbutt]),
        (
            &[MoveId::Fireblast, MoveId::Magmastorm],
            &[MoveId::Fierydance, MoveId::Flamethrower, MoveId::Lavaplume],
        ),
        (&[MoveId::Thunderpunch], &[MoveId::Wildcharge]),
        (&[MoveId::Thunderbolt], &[MoveId::Discharge]),
        (
            &[MoveId::Gunkshot],
            &[MoveId::Direclaw, MoveId::Poisonjab, MoveId::Sludgebomb],
        ),
        (&[MoveId::Aurasphere], &[MoveId::Focusblast]),
        (&[MoveId::Closecombat], &[MoveId::Drainpunch]),
        (&[MoveId::Dragonpulse, MoveId::Spacialrend], &[MoveId::Dracometeor]),
        (&[MoveId::Dragonclaw], &[MoveId::Outrage]),
        (&[MoveId::Heavyslam], &[MoveId::Flashcannon]),
        (&[MoveId::Alluringvoice], &[MoveId::Dazzlinggleam]),
        (&[MoveId::Taunt], &[MoveId::Disable]),
        (
            &[MoveId::Thunderwave, MoveId::Toxic],
            &[MoveId::Thunderwave, MoveId::Willowisp],
        ),
        (
            &[MoveId::Thunderwave, MoveId::Toxic, MoveId::Willowisp],
            &[MoveId::Toxicspikes],
        ),
        (&[MoveId::Nastyplot], &[MoveId::Rockslide, MoveId::Knockoff]),
        (&[MoveId::Switcheroo], &[MoveId::Fakeout]),
        (&[MoveId::Toxic], &[MoveId::Clearsmog]),
        (&[MoveId::Healbell], &[MoveId::Stealthrock]),
        (&[MoveId::Mirrorcoat], &[MoveId::Hydropump, MoveId::Bodypress]),
    ];
    static TOUCHING: std::sync::OnceLock<Vec<u128>> = std::sync::OnceLock::new();
    let touching = TOUCHING
        .get_or_init(|| {
            assert!(PAIRS.len() <= 128, "the pair set is a u128");
            let mut by_move = vec![0u128; MoveId::COUNT + 1];
            for (i, (a, b)) in PAIRS.iter().enumerate() {
                for &m in a.iter().chain(b.iter()) {
                    by_move[m as usize] |= 1 << i;
                }
            }
            by_move
        });
    let mut pairs = moves.iter().fold(0u128, |set, &m| set | touching[m as usize]);
    while pairs != 0 {
        let (a, b) = PAIRS[pairs.trailing_zeros() as usize];
        incompatible_moves(&chosen, moves, move_pool, a, b);
        pairs &= pairs - 1;
    }
    if !types.contains(&Type::Ice) {
        incompatible_moves(&chosen, moves, move_pool, MoveId::Icebeam, MoveId::Icywind);
    }
    if !is_doubles {
        incompatible_moves(&chosen, moves, move_pool, MoveId::Taunt, MoveId::Encore);
    }
    if !types.contains(&Type::Dark) && tera_type != Type::Dark {
        incompatible_moves(
            &chosen,
            moves,
            move_pool,
            MoveId::Knockoff,
            MoveId::Suckerpunch,
        );
    }
    if !abilities.contains(&AbilityId::Prankster) {
        incompatible_moves(&chosen, moves, move_pool, MoveId::Thunderwave, MoveId::Yawn);
    }
    match species {
        SpeciesId::Barraskewda => {
            incompatible_moves(
                &chosen,
                moves,
                move_pool,
                [MoveId::Psychicfangs, MoveId::Throatchop],
                [MoveId::Poisonjab, MoveId::Throatchop],
            );
        }
        SpeciesId::Quagsire => {
            incompatible_moves(
                &chosen,
                moves,
                move_pool,
                MoveId::Spikes,
                MoveId::Icebeam,
            );
        }
        SpeciesId::Cyclizar => {
            incompatible_moves(
                &chosen,
                moves,
                move_pool,
                MoveId::Taunt,
                MoveId::Knockoff,
            );
        }
        SpeciesId::Coalossal => {
            incompatible_moves(
                &chosen,
                moves,
                move_pool,
                MoveId::Flamethrower,
                MoveId::Overheat,
            );
        }
        _ => {}
    }
    let _ = (counter, is_lead);
}
/// PS's `addMove`: add `mv` to `moves`, remove it from `move_pool`, recount
/// and cull again. Returns the new counter.
#[allow(clippy::too_many_arguments)]
pub fn add_move(
    mv: MoveId,
    moves: &mut Vec<MoveId>,
    types: &[Type],
    abilities: &[AbilityId],
    team_details: &TeamDetails,
    species: SpeciesId,
    is_lead: bool,
    is_doubles: bool,
    move_pool: &mut Vec<MoveId>,
    tera_type: Type,
    role: RoleId,
) -> MoveCounter {
    if !moves.contains(&mv) {
        moves.push(mv);
    }
    if let Some(i) = move_pool.iter().position(|m| *m == mv) {
        fast_pop(move_pool, i);
    }
    let counter = query_moves(moves, species, tera_type, abilities);
    cull_move_pool(
        types,
        moves,
        abilities,
        &counter,
        move_pool,
        team_details,
        species,
        is_lead,
        is_doubles,
        tera_type,
        role,
    );
    counter
}
/// PS's `randomMoveset`: the culls, the enforced additions and the random
/// fill, up to `MAX_MOVE_COUNT` moves (the whole culled pool when it is no
/// larger).
#[allow(clippy::too_many_arguments, unused_assignments)]
pub fn random_moveset<R: Decide>(
    rng: &mut R,
    types: &[Type],
    abilities: &[AbilityId],
    team_details: &TeamDetails,
    species: SpeciesId,
    is_lead: bool,
    is_doubles: bool,
    mut move_pool: Vec<MoveId>,
    tera_type: Type,
    role: RoleId,
) -> Vec<MoveId> {
    let mut moves: Vec<MoveId> = Vec::with_capacity(4);
    let mut counter = query_moves(&moves, species, tera_type, abilities);
    cull_move_pool(
        types,
        &moves,
        abilities,
        &counter,
        &mut move_pool,
        team_details,
        species,
        is_lead,
        is_doubles,
        tera_type,
        role,
    );
    if move_pool.len() <= MAX_MOVE_COUNT {
        for moveid in move_pool.drain(..) {
            if !moves.contains(&moveid) {
                moves.push(moveid);
            }
        }
        return moves;
    }
    macro_rules! add {
        ($mv:expr) => {
            { counter = add_move($mv, & mut moves, types, abilities, team_details,
            species, is_lead, is_doubles, & mut move_pool, tera_type, role,); }
        };
    }
    if role == RoleId::TeraBlastUser && move_pool.contains(&MoveId::Terablast) {
        add!(MoveId::Terablast);
    }
    if let Some(required) = species.data().required_move && move_pool.contains(&required)
    {
        add!(required);
    }
    if move_pool.contains(&MoveId::Facade) && abilities.contains(&AbilityId::Guts) {
        add!(MoveId::Facade);
    }
    for moveid in [
        MoveId::Nightshade,
        MoveId::Revelationdance,
        MoveId::Revivalblessing,
        MoveId::Stickyweb,
    ] {
        if move_pool.contains(&moveid) {
            add!(moveid);
        }
    }
    if role == RoleId::BulkySupport && !team_details.defog && !team_details.rapid_spin {
        if move_pool.contains(&MoveId::Rapidspin) {
            add!(MoveId::Rapidspin);
        }
        if move_pool.contains(&MoveId::Defog) {
            add!(MoveId::Defog);
        }
    }
    if [MoveId::Dragontail, MoveId::Roar, MoveId::Whirlwind]
        .iter()
        .any(|m| move_pool.contains(m)) && species.data().base_stats.spe <= 60
        && move_pool.contains(&MoveId::Stealthrock) && !team_details.stealth_rock
    {
        add!(MoveId::Stealthrock);
    }
    if !team_details.screens && move_pool.contains(&MoveId::Auroraveil) {
        add!(MoveId::Auroraveil);
    }
    if !is_doubles && types.len() == 1
        && (types.contains(&Type::Normal) || types.contains(&Type::Fighting))
        && move_pool.contains(&MoveId::Knockoff)
    {
        add!(MoveId::Knockoff);
    }
    if species == SpeciesId::Smeargle && move_pool.contains(&MoveId::Spore) {
        add!(MoveId::Spore);
    }
    if matches!(role, RoleId::BulkyAttacker | RoleId::BulkySetup | RoleId::Wallbreaker)
        || PRIORITY_POKEMON.contains(&species)
    {
        let mut priority_moves: Vec<MoveId> = Vec::new();
        for &moveid in &move_pool {
            let mv = moveid.data();
            let mt = get_move_type(mv, species, abilities, tera_type);
            let is_priority = mv.priority > 0
                || (moveid == MoveId::Grassyglide
                    && abilities.contains(&AbilityId::Grassysurge));
            let has_bp = mv.base_power > 0 || mv.has_base_power_callback;
            if types.contains(&mt) && is_priority && has_bp {
                priority_moves.push(moveid);
            }
        }
        if !priority_moves.is_empty() {
            let pick = sample(rng, &priority_moves);
            add!(pick);
        }
    }
    for &t in types {
        let mut stab_moves: Vec<MoveId> = Vec::new();
        for &moveid in &move_pool {
            let mv = moveid.data();
            let mt = get_move_type(mv, species, abilities, tera_type);
            let has_bp = mv.base_power > 0 || mv.has_base_power_callback;
            if !NO_STAB.contains(&moveid) && has_bp && t == mt {
                stab_moves.push(moveid);
            }
        }
        while super::moveset::run_enforcement_checker(
            t,
            &move_pool,
            &moves,
            abilities,
            types,
            &counter,
            species,
            team_details,
            is_lead,
            is_doubles,
            role,
        ) {
            if stab_moves.is_empty() {
                break;
            }
            let Some(pick) = sample_no_replace(rng, &mut stab_moves) else {
                break;
            };
            add!(pick);
        }
    }
    if counter.get(CounterKey::Stabtera) == 0 && role != RoleId::BulkySupport {
        let mut stab_moves: Vec<MoveId> = Vec::new();
        for &moveid in &move_pool {
            let mv = moveid.data();
            let mt = get_move_type(mv, species, abilities, tera_type);
            let has_bp = mv.base_power > 0 || mv.has_base_power_callback;
            if !NO_STAB.contains(&moveid) && has_bp && tera_type == mt {
                stab_moves.push(moveid);
            }
        }
        if !stab_moves.is_empty() {
            let pick = sample(rng, &stab_moves);
            add!(pick);
        }
    }
    if counter.get(CounterKey::Stab) == 0 {
        let mut stab_moves: Vec<MoveId> = Vec::new();
        for &moveid in &move_pool {
            let mv = moveid.data();
            let mt = get_move_type(mv, species, abilities, tera_type);
            let has_bp = mv.base_power > 0 || mv.has_base_power_callback;
            if !NO_STAB.contains(&moveid) && has_bp && types.contains(&mt) {
                stab_moves.push(moveid);
            }
        }
        if !stab_moves.is_empty() {
            let pick = sample(rng, &stab_moves);
            add!(pick);
        }
    }
    if matches!(
        role, RoleId::BulkySupport | RoleId::BulkyAttacker | RoleId::BulkySetup
    ) {
        let recovery: Vec<MoveId> = move_pool
            .iter()
            .copied()
            .filter(|m| RECOVERY_MOVES.contains(m))
            .collect();
        if !recovery.is_empty() {
            let pick = sample(rng, &recovery);
            add!(pick);
        }
    }
    if role == RoleId::AvPivot {
        let pivots: Vec<MoveId> = move_pool
            .iter()
            .copied()
            .filter(|m| matches!(* m, MoveId::Uturn | MoveId::Voltswitch))
            .collect();
        if !pivots.is_empty() {
            let pick = sample(rng, &pivots);
            add!(pick);
        }
    }
    if matches!(role, RoleId::BulkySetup | RoleId::FastBulkySetup | RoleId::SetupSweeper)
        || role == RoleId::TeraBlastUser
    {
        let non_speed_setup: Vec<MoveId> = move_pool
            .iter()
            .copied()
            .filter(|m| SETUP.contains(m) && !SPEED_SETUP.contains(m))
            .collect();
        if !non_speed_setup.is_empty() {
            let pick = sample(rng, &non_speed_setup);
            add!(pick);
        } else {
            let setup: Vec<MoveId> = move_pool
                .iter()
                .copied()
                .filter(|m| SETUP.contains(m))
                .collect();
            if !setup.is_empty() {
                let pick = sample(rng, &setup);
                add!(pick);
            }
        }
    }
    if counter.damaging_is_empty() {
        let attacking: Vec<MoveId> = move_pool
            .iter()
            .copied()
            .filter(|m| {
                !NO_STAB.contains(m) && m.data().category != MoveCategory::Status
            })
            .collect();
        if !attacking.is_empty() {
            let pick = sample(rng, &attacking);
            add!(pick);
        }
    }
    if !matches!(role, RoleId::AvPivot | RoleId::FastSupport | RoleId::BulkySupport)
        && counter.damaging_len() == 1
    {
        let current_attack_type = counter
            .first_damaging_type()
            .expect("a single damaging move has a type");
        let mut coverage: Vec<MoveId> = Vec::new();
        for &moveid in &move_pool {
            let mv = moveid.data();
            let mt = get_move_type(mv, species, abilities, tera_type);
            let has_bp = mv.base_power > 0 || mv.has_base_power_callback;
            if !NO_STAB.contains(&moveid) && has_bp && current_attack_type != mt {
                coverage.push(moveid);
            }
        }
        if !coverage.is_empty() {
            let pick = sample(rng, &coverage);
            add!(pick);
        }
    }
    while moves.len() < MAX_MOVE_COUNT && !move_pool.is_empty() {
        if moves.len() + move_pool.len() <= MAX_MOVE_COUNT {
            for moveid in move_pool.drain(..) {
                if !moves.contains(&moveid) {
                    moves.push(moveid);
                }
            }
            break;
        }
        let pick = sample(rng, &move_pool);
        add!(pick);
        for (a, b) in MOVE_PAIRS {
            if pick == *a && move_pool.contains(b) {
                add!(* b);
            }
            if pick == *b && move_pool.contains(a) {
                add!(* a);
            }
        }
    }
    moves
}
