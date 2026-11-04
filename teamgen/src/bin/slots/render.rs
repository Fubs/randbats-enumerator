//! A species' weighted sets as pkmn/randbats' `data/stats` writes them: what
//! its `update` script's sampler converges to, at full precision (`update`
//! rounds when it publishes).
use std::collections::HashMap;
use randbats_gen9::generated::item_id::ItemId;
use randbats_gen9::generated::randbats_data::RoleId;
use randbats_gen9::randbats::RandbatsEntry;
use randbats_gen9::randbats_gen::SetOutcome;
use serde_json::Value;
use crate::json::{Json, locale_cmp};
const EV_DEFAULT: u8 = 85;
const IV_DEFAULT: u8 = 31;
const STATS: [&str; 6] = ["hp", "atk", "def", "spa", "spd", "spe"];
/// A Neumaier-compensated sum: a species' marginal adds up thousands of
/// context-weighted copies of the same exact table, and a plain f64 sum
/// prints 1/3 as 0.3333333333333259.
#[derive(Default, Clone, Copy)]
pub struct Sum {
    s: f64,
    c: f64,
}
impl std::ops::AddAssign<f64> for Sum {
    fn add_assign(&mut self, x: f64) {
        let t = self.s + x;
        self.c
            += if self.s.abs() >= x.abs() { (self.s - t) + x } else { (x - t) + self.s };
        self.s = t;
    }
}
impl Sum {
    pub fn get(self) -> f64 {
        self.s + self.c
    }
}
/// One stat across a group's sets. pkmn lists a stat when its sampled sets
/// agree on a value that is not the default (its tally forgets
/// default-valued sets seen before the first other one); in the limit of
/// its sampler, that is every set of positive probability having it.
#[derive(Default, Clone, Copy)]
enum Stat {
    #[default]
    Unseen,
    Same(u8),
    Mixed,
}
impl Stat {
    fn see(&mut self, v: u8) {
        *self = match *self {
            Stat::Unseen => Stat::Same(v),
            Stat::Same(u) if u == v => Stat::Same(v),
            _ => Stat::Mixed,
        };
    }
}
#[derive(Default)]
struct Masses {
    total: Sum,
    abilities: HashMap<String, Sum>,
    items: HashMap<String, Sum>,
    tera: HashMap<String, Sum>,
    moves: HashMap<String, Sum>,
    evs: [Stat; 6],
    ivs: [Stat; 6],
}
impl Masses {
    fn add(&mut self, p: f64, o: &SetOutcome) {
        let s = &o.set;
        self.total += p;
        *self.abilities.entry(s.ability.metadata().name.to_string()).or_default() += p;
        if s.item != ItemId::None {
            *self.items.entry(s.item.metadata().name.to_string()).or_default() += p;
        }
        *self.tera.entry(format!("{:?}", s.tera_type)).or_default() += p;
        for mv in s.moves.iter().flatten() {
            *self.moves.entry(mv.data().name.to_string()).or_default() += p;
        }
        if p > 0.0 {
            let evs = [s.evs.hp, s.evs.atk, s.evs.def, s.evs.spa, s.evs.spd, s.evs.spe];
            let ivs = [s.ivs.hp, s.ivs.atk, s.ivs.def, s.ivs.spa, s.ivs.spd, s.ivs.spe];
            for k in 0..6 {
                self.evs[k].see(evs[k]);
                self.ivs[k].see(ivs[k]);
            }
        }
    }
}
/// A probability as pkmn writes one: 0 and 1 as integers, anything else at
/// full f64 precision.
fn prob(x: f64) -> Json {
    Json::Leaf(
        if (x - 1.0).abs() < 1e-12 {
            Value::from(1)
        } else if x == 0.0 {
            Value::from(0)
        } else {
            Value::from(x)
        },
    )
}
/// The order key of a share: equal shares tie even when the float sums
/// that made them differ in the last bits, so the order is the limit's and
/// not rounding noise.
fn order_key(x: f64) -> f64 {
    (x * 1e12).round()
}
/// pkmn's `CMP.entries`: descending share, ties by name, each a share of
/// `total`. A role no set takes has shares of 0 (pkmn's 0/0 prints null).
fn sorted(m: &HashMap<String, Sum>, total: Sum) -> Json {
    let total = total.get();
    let mut v: Vec<(&String, f64)> = m
        .iter()
        .map(|(k, w)| (k, if total == 0.0 { 0.0 } else { w.get() / total }))
        .collect();
    v.sort_by(|a, b| {
        order_key(b.1).total_cmp(&order_key(a.1)).then_with(|| locale_cmp(a.0, b.0))
    });
    Json::Object(v.into_iter().map(|(k, w)| (k.clone(), prob(w))).collect())
}
fn overrides(stats: &[Stat; 6], default: u8) -> Option<Json> {
    let out: Vec<(String, Json)> = STATS
        .iter()
        .zip(stats)
        .filter_map(|(name, s)| match *s {
            Stat::Same(v) if v != default => {
                Some((name.to_string(), Json::Leaf(Value::from(v))))
            }
            _ => None,
        })
        .collect();
    (!out.is_empty()).then_some(Json::Object(out))
}
/// What a file's render met that pkmn's sampler would show differently or
/// not at all.
#[derive(Default)]
pub struct Notes {
    /// Roles of the species with no set of positive probability here.
    pub empty_roles: Vec<String>,
}
/// One species' entry from its weighted set outcomes (weights need not sum
/// to 1; every share is taken of their total). Every role, and every move
/// and Tera type of a role's pools, is listed even at probability 0, as
/// pkmn's `getRoles` seeds them; abilities and items only when seen.
pub fn render<'a>(
    entry: &RandbatsEntry,
    outcomes: impl IntoIterator<Item = (f64, &'a SetOutcome)>,
    notes: &mut Notes,
) -> Json {
    let mut species = Masses::default();
    let mut roles: Vec<(RoleId, Masses)> = Vec::new();
    for r in entry.roles {
        let seeded = Masses {
            moves: r
                .moves
                .iter()
                .map(|m| (m.id.data().name.to_string(), Sum::default()))
                .collect(),
            tera: r
                .tera_types
                .iter()
                .map(|t| (format!("{:?}", t.typ), Sum::default()))
                .collect(),
            ..Masses::default()
        };
        match roles.iter_mut().find(|x| x.0 == r.id) {
            Some(x) => x.1 = seeded,
            None => roles.push((r.id, seeded)),
        }
    }
    let mut level = None;
    for (p, o) in outcomes {
        level.get_or_insert(o.set.level);
        species.add(p, o);
        let role = o.set.role.expect("generated sets carry a role");
        roles
            .iter_mut()
            .find(|x| x.0 == role)
            .expect("a set's role is one of its species'")
            .1
            .add(p, o);
    }
    let share = |m: &Masses| order_key(m.total.get() / species.total.get());
    roles.sort_by(|a, b| share(&b.1).total_cmp(&share(&a.1)));
    let mut rendered_roles = Vec::new();
    for (id, r) in &roles {
        let name = id.name().to_string();
        if r.total.get() == 0.0 {
            notes.empty_roles.push(name.clone());
        }
        let mut role = vec![
            ("weight".to_string(), prob(r.total.get() / species.total.get()),),
            ("abilities".to_string(), sorted(& r.abilities, r.total)),
        ];
        if !r.items.is_empty() {
            role.push(("items".into(), sorted(&r.items, r.total)));
        }
        role.push(("teraTypes".into(), sorted(&r.tera, r.total)));
        role.push(("moves".into(), sorted(&r.moves, r.total)));
        if let Some(evs) = overrides(&r.evs, EV_DEFAULT) {
            role.push(("evs".into(), evs));
        }
        if let Some(ivs) = overrides(&r.ivs, IV_DEFAULT) {
            role.push(("ivs".into(), ivs));
        }
        rendered_roles.push((name, Json::Object(role)));
    }
    let mut out = vec![
        ("level".to_string(), Json::Leaf(Value::from(level
        .expect("a species has sets"))),), ("abilities".to_string(), sorted(& species
        .abilities, species.total),),
    ];
    if !species.items.is_empty() {
        out.push(("items".into(), sorted(&species.items, species.total)));
    }
    out.push(("roles".into(), Json::Object(rendered_roles)));
    if let Some(evs) = overrides(&species.evs, EV_DEFAULT) {
        out.push(("evs".into(), evs));
    }
    if let Some(ivs) = overrides(&species.ivs, IV_DEFAULT) {
        out.push(("ivs".into(), ivs));
    }
    Json::Object(out)
}
