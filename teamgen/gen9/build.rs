//! Generates the data tables from a dump of Pokémon Showdown (`dump.js`).
//!
//! The dump is read from `RANDBATS_DUMP`, or from `target/dump.json` in the
//! workspace, where `node dump.js` writes it. Every id enum is numbered in
//! the dump's order, which is sorted by id.
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::path::PathBuf;
use serde_json::Value;
const TYPES: [&str; 19] = [
    "Normal",
    "Fire",
    "Water",
    "Electric",
    "Grass",
    "Ice",
    "Fighting",
    "Poison",
    "Ground",
    "Flying",
    "Psychic",
    "Bug",
    "Rock",
    "Ghost",
    "Dragon",
    "Dark",
    "Steel",
    "Fairy",
    "Stellar",
];
fn main() {
    println!("cargo:rerun-if-env-changed=RANDBATS_DUMP");
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let workspace = manifest.parent().unwrap().to_path_buf();
    let path = match std::env::var_os("RANDBATS_DUMP") {
        Some(p) => PathBuf::from(p),
        None => workspace.join("target").join("dump.json"),
    };
    println!("cargo:rerun-if-changed={}", path.display());
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) => {
            fail(
                &format!(
                    "cannot read the Pokémon Showdown dump at {} ({e}).\n\
             Write it with `node dump.js` in {} (Pokémon Showdown must be built first: \
             `npm install && node build` in vendor/pokemon-showdown), or point RANDBATS_DUMP at one.",
                    path.display(), workspace.display()
                ),
            )
        }
    };
    let dump: Value = serde_json::from_str(&text)
        .unwrap_or_else(|e| fail(&format!("{}: {e}", path.display())));
    if std::env::var_os("RANDBATS_DUMP").is_none() && let Some(root) = workspace.parent()
    {
        check_fresh(
            &root.join("vendor").join("pokemon-showdown"),
            str_of(&dump["ps"]),
            &path,
        );
    }
    let data = &dump["data"];
    check_format(&data["format"]);
    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    let write = |name: &str, body: String| {
        std::fs::write(out.join(name), body)
            .unwrap_or_else(|e| fail(&format!("writing {name}: {e}")));
    };
    let species = list(data, "species");
    let moves = list(data, "moves");
    let items = list(data, "items");
    let abilities = list(data, "abilities");
    write(
        "species_id.rs",
        ids(
            "SpeciesId",
            species,
            "data",
            "crate::species::Species",
            "crate::generated::species_data::SPECIES",
        ),
    );
    write(
        "move_id.rs",
        ids(
            "MoveId",
            moves,
            "data",
            "crate::moves::Move",
            "crate::generated::moves_data::MOVES",
        ),
    );
    write(
        "item_id.rs",
        ids(
            "ItemId",
            items,
            "metadata",
            "crate::items::ItemMetadata",
            "crate::generated::items_data::ITEMS",
        ),
    );
    write(
        "ability_id.rs",
        ids(
            "AbilityId",
            abilities,
            "metadata",
            "crate::abilities::AbilityMetadata",
            "crate::generated::abilities_data::ABILITIES",
        ),
    );
    write("species_data.rs", species_data(species));
    write("moves_data.rs", moves_data(moves));
    write(
        "items_data.rs",
        named_data("ItemMetadata", "ItemId", "crate::items", "ITEMS", items, "(no item)"),
    );
    write(
        "abilities_data.rs",
        named_data(
            "AbilityMetadata",
            "AbilityId",
            "crate::abilities",
            "ABILITIES",
            abilities,
            "(no ability)",
        ),
    );
    write("randbats_data.rs", randbats_data(&data["randbatsSets"], species));
    write("type_chart.rs", type_chart(&data["typechart"]));
    write(
        "source.rs",
        format!(
            "pub const PS_COMMIT: &str = {:?};\npub const DATA_FINGERPRINT: &str = {:?};\npub const LOGIC_FINGERPRINT: &str = {:?};\n",
            str_of(& dump["ps"]), str_of(& dump["fingerprint"]), str_of(& dump["logic"]),
        ),
    );
}
/// The default dump must be of the vendored checkout as it stands, when git
/// can say what that is.
fn check_fresh(ps: &std::path::Path, dumped: &str, path: &std::path::Path) {
    let Ok(out) = std::process::Command::new("git")
        .arg("-C")
        .arg(ps)
        .args(["rev-parse", "HEAD"])
        .output() else {
        return;
    };
    let head = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if out.status.success() && head != dumped {
        fail(
            &format!(
                "{} is a dump of Pokémon Showdown {dumped}, but {} is at {head}. Rerun `node dump.js`.",
                path.display(), ps.display()
            ),
        );
    }
}
fn fail(msg: &str) -> ! {
    eprintln!("randbats-gen9: {msg}");
    std::process::exit(1)
}
fn list<'a>(data: &'a Value, key: &str) -> &'a [Value] {
    data[key]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_else(|| fail(&format!("dump: `{key}` is not a list")))
}
fn str_of(v: &Value) -> &str {
    v.as_str().unwrap_or_else(|| fail(&format!("dump: expected a string, found {v}")))
}
fn opt_str(v: &Value) -> Option<&str> {
    if v.is_null() { None } else { Some(str_of(v)) }
}
fn int(v: &Value) -> i64 {
    v.as_i64().unwrap_or_else(|| fail(&format!("dump: expected an integer, found {v}")))
}
/// The port is written for these rules of the format; a change to any of
/// them is a change to the generator.
fn check_format(f: &Value) {
    let expect = |what: &str, ok: bool| {
        if !ok {
            fail(
                &format!(
                    "gen9randombattle's {what} changed ({f}); the port assumes otherwise"
                ),
            );
        }
    };
    expect("mod", f["mod"] == "gen9");
    expect("game type", f["gameType"] == "singles");
    expect("team", f["team"] == "random");
    expect("team size", f["maxTeamSize"] == 6);
    expect("move count", f["maxMoveCount"] == 4);
    expect("level adjustment", f["adjustLevel"].is_null());
    let rules: BTreeSet<&str> = f["rules"]
        .as_array()
        .into_iter()
        .flatten()
        .map(str_of)
        .collect();
    for rule in [
        "forceofthefallenmod",
        "pickedteamsize",
        "sametypeclause",
        "teampreview",
        "terastalclause",
    ] {
        expect(rule, !rules.contains(rule));
    }
    for rule in ["forcemonotype", "forceteratype"] {
        expect(rule, f["valueRules"].get(rule).is_none());
    }
    for key in ["banlist", "restricted", "unbanlist", "customRules"] {
        expect(key, f[key].as_array().is_some_and(Vec::is_empty));
    }
}
/// PS id to variant: the first letter upper-cased, a leading digit behind `_`.
fn variant(id: &str) -> String {
    let mut out = String::with_capacity(id.len() + 1);
    let mut chars = id.chars();
    if let Some(first) = chars.next() {
        if first.is_ascii_digit() {
            out.push('_');
            out.push(first);
        } else {
            out.extend(first.to_uppercase());
        }
    }
    out.extend(chars);
    out
}
fn type_variant(name: &str) -> &'static str {
    TYPES
        .iter()
        .find(|t| **t == name)
        .copied()
        .unwrap_or_else(|| fail(&format!("dump: unknown type {name:?}")))
}
/// A species' or move's type: PS's `???`, and MissingNo.'s `Bird`, are typeless.
fn dex_type(name: &str) -> &'static str {
    match name {
        "???" | "Bird" => "Typeless",
        _ => type_variant(name),
    }
}
fn ids(name: &str, entries: &[Value], accessor: &str, ty: &str, table: &str) -> String {
    let ids: Vec<&str> = entries.iter().map(|e| str_of(&e["id"])).collect();
    if ids.windows(2).any(|w| w[0] >= w[1]) {
        fail(&format!("dump: {name}s are not sorted by id"));
    }
    let mut b = String::new();
    writeln!(b, "#[repr(u16)]").unwrap();
    writeln!(b, "#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]")
        .unwrap();
    writeln!(b, "pub enum {name} {{\n    None = 0,").unwrap();
    for (i, id) in ids.iter().enumerate() {
        writeln!(b, "    {} = {},", variant(id), i + 1).unwrap();
    }
    writeln!(b, "}}\n\nimpl {name} {{").unwrap();
    writeln!(b, "    pub const COUNT: usize = {};", ids.len() + 1).unwrap();
    writeln!(
        b,
        "    pub fn {accessor}(self) -> &'static {ty} {{\n        &{table}[self as usize]\n    }}"
    )
        .unwrap();
    writeln!(
        b,
        "    pub fn name(self) -> &'static str {{\n        self.{accessor}().name\n    }}"
    )
        .unwrap();
    writeln!(b, "    /// PS's id: lowercase, alphanumeric only.").unwrap();
    writeln!(b, "    pub fn ps_id(self) -> &'static str {{").unwrap();
    writeln!(
        b, "        static IDS: [&str; {}] = [\"\", {}];", ids.len() + 1, quoted(& ids)
    )
        .unwrap();
    writeln!(b, "        IDS[self as usize]\n    }}").unwrap();
    writeln!(b, "    pub fn from_ps_id(s: &str) -> Option<Self> {{").unwrap();
    writeln!(
        b, "        static IDS: [(&str, {name}); {}] = [(\"\", {name}::None), ", ids
        .len() + 1
    )
        .unwrap();
    for id in &ids {
        write!(b, "({id:?}, {name}::{}), ", variant(id)).unwrap();
    }
    writeln!(b, "];").unwrap();
    writeln!(
        b,
        "        IDS.binary_search_by(|(id, _)| (*id).cmp(s)).ok().map(|i| IDS[i].1)\n    }}\n}}"
    )
        .unwrap();
    b
}
fn quoted(ids: &[&str]) -> String {
    ids.iter().map(|id| format!("{id:?}")).collect::<Vec<_>>().join(", ")
}
fn opt_variant(enum_name: &str, v: &Value) -> String {
    match opt_str(v) {
        None => "None".into(),
        Some(id) => format!("Some({enum_name}::{})", variant(id)),
    }
}
fn species_data(entries: &[Value]) -> String {
    let mut b = String::from(
        "use crate::generated::ability_id::AbilityId;\n\
         use crate::generated::item_id::ItemId;\n\
         use crate::generated::move_id::MoveId;\n\
         use crate::generated::species_id::SpeciesId;\n\
         use crate::species::*;\n\
         use crate::types::Type;\n\n",
    );
    writeln!(b, "pub static SPECIES: [Species; SpeciesId::COUNT] = [").unwrap();
    writeln!(
        b,
        "Species {{ id: SpeciesId::None, name: \"(no species)\", types: SpeciesTypes(Type::Typeless, None), \
         base_stats: StatBlock {{ hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 }}, \
         abilities: SpeciesAbilities {{ slot_1: AbilityId::None, slot_2: None, hidden: None, special: None }}, \
         gender_ratio: GenderRatio::Genderless, base_species: SpeciesId::None, forme_name: None, \
         required_item: None, required_items: &[], required_move: None, tags: SpeciesTags::empty() }},"
    )
        .unwrap();
    for e in entries {
        let types = e["types"].as_array().unwrap_or_else(|| fail("dump: species types"));
        let second = match opt_str(&types[1]) {
            None => "None".to_string(),
            Some(t) => format!("Some(Type::{})", dex_type(t)),
        };
        let st = &e["baseStats"];
        let ab = &e["abilities"];
        let slot_1 = match str_of(&ab["slot1"]) {
            "" => "AbilityId::None".to_string(),
            id => format!("AbilityId::{}", variant(id)),
        };
        let g = &e["genderRatio"];
        let gender = match str_of(&g["kind"]) {
            "genderless" => "GenderRatio::Genderless".to_string(),
            "alwaysMale" => "GenderRatio::AlwaysMale".to_string(),
            "alwaysFemale" => "GenderRatio::AlwaysFemale".to_string(),
            "ratio" => {
                format!(
                    "GenderRatio::Ratio {{ male_eighths: {} }}", int(& g["maleEighths"])
                )
            }
            k => fail(&format!("dump: gender ratio kind {k:?}")),
        };
        let forme = match opt_str(&e["formeName"]) {
            None => "None".to_string(),
            Some(f) => format!("Some({f:?})"),
        };
        let required_items: Vec<String> = e["requiredItems"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|i| format!("ItemId::{}", variant(str_of(i))))
            .collect();
        let tags = if e["nfe"] == true {
            "SpeciesTags::NFE"
        } else {
            "SpeciesTags::empty()"
        };
        writeln!(
            b,
            "Species {{ id: SpeciesId::{}, name: {:?}, types: SpeciesTypes(Type::{}, {second}), \
             base_stats: StatBlock {{ hp: {}, atk: {}, def: {}, spa: {}, spd: {}, spe: {} }}, \
             abilities: SpeciesAbilities {{ slot_1: {slot_1}, slot_2: {}, hidden: {}, special: {} }}, \
             gender_ratio: {gender}, base_species: SpeciesId::{}, forme_name: {forme}, required_item: {}, \
             required_items: &[{}], required_move: {}, tags: {tags} }},",
            variant(str_of(& e["id"])), str_of(& e["name"]), dex_type(str_of(&
            types[0])), int(& st["hp"]), int(& st["atk"]), int(& st["def"]), int(&
            st["spa"]), int(& st["spd"]), int(& st["spe"]), opt_variant("AbilityId", &
            ab["slot2"]), opt_variant("AbilityId", & ab["hidden"]),
            opt_variant("AbilityId", & ab["special"]), variant(str_of(&
            e["baseSpecies"])), opt_variant("ItemId", & e["requiredItem"]),
            required_items.join(", "), opt_variant("MoveId", & e["requiredMove"]),
        )
            .unwrap();
    }
    b.push_str("];\n");
    b
}
fn fraction(v: &Value) -> String {
    match v.as_array() {
        None => "None".into(),
        Some(nd) => {
            format!("Some(Fraction {{ num: {}, den: {} }})", int(& nd[0]), int(& nd[1]))
        }
    }
}
fn moves_data(entries: &[Value]) -> String {
    let mut b = String::from(
        "use crate::generated::move_id::MoveId;\nuse crate::moves::*;\nuse crate::types::Type;\n\n",
    );
    writeln!(b, "pub static MOVES: [Move; MoveId::COUNT] = [").unwrap();
    writeln!(
        b,
        "Move {{ id: MoveId::None, name: \"(no move)\", typ: Type::Typeless, category: MoveCategory::Status, \
         base_power: 0, accuracy: Accuracy::AlwaysHits, priority: 0, flags: MoveFlags::empty(), \
         multihit: MultiHit::None, drain: None, recoil: None, secondary: None, \
         has_base_power_callback: false, set_damage: false }},"
    )
        .unwrap();
    for e in entries {
        let category = match str_of(&e["category"]) {
            c @ ("Physical" | "Special" | "Status") => c,
            c => fail(&format!("dump: move category {c:?}")),
        };
        let accuracy = match &e["accuracy"] {
            Value::String(s) if s == "always" => "Accuracy::AlwaysHits".to_string(),
            v => format!("Accuracy::Percent({})", int(v)),
        };
        let mut flags = Vec::new();
        for (key, flag) in [("bite", "BITE"), ("punch", "PUNCH"), ("sound", "SOUND")] {
            if e["flags"][key] == true {
                flags.push(flag);
            }
        }
        let flags = flags
            .iter()
            .fold(
                "MoveFlags::empty()".to_string(),
                |acc, f| { format!("{acc}.union(MoveFlags::{f})") },
            );
        let multihit = match &e["multihit"] {
            Value::Null => "MultiHit::None".to_string(),
            Value::Array(r) => {
                format!("MultiHit::Range({}, {})", int(& r[0]), int(& r[1]))
            }
            v => format!("MultiHit::Fixed({})", int(v)),
        };
        writeln!(
            b,
            "Move {{ id: MoveId::{}, name: {:?}, typ: Type::{}, category: MoveCategory::{category}, base_power: {}, \
             accuracy: {accuracy}, priority: {}, flags: {flags}, multihit: {multihit}, drain: {}, recoil: {}, \
             secondary: {}, has_base_power_callback: {}, set_damage: {} }},",
            variant(str_of(& e["id"])), str_of(& e["name"]), dex_type(str_of(&
            e["type"])), int(& e["basePower"]), int(& e["priority"]), fraction(&
            e["drain"]), fraction(& e["recoil"]), if e["secondary"] == true {
            "Some(Secondary)" } else { "None" }, e["hasBasePowerCallback"] == true,
            e["setDamage"] == true,
        )
            .unwrap();
    }
    b.push_str("];\n");
    b
}
fn named_data(
    ty: &str,
    id: &str,
    module: &str,
    table: &str,
    entries: &[Value],
    none: &str,
) -> String {
    let short = id.trim_end_matches("Id").to_lowercase();
    let mut b = format!(
        "use crate::generated::{short}_id::{id};\nuse {module}::{ty};\n\n"
    );
    writeln!(b, "pub static {table}: [{ty}; {id}::COUNT] = [").unwrap();
    writeln!(b, "{ty} {{ id: {id}::None, name: {none:?} }},").unwrap();
    for e in entries {
        writeln!(
            b, "{ty} {{ id: {id}::{}, name: {:?} }},", variant(str_of(& e["id"])),
            str_of(& e["name"])
        )
            .unwrap();
    }
    b.push_str("];\n");
    b
}
fn type_chart(chart: &Value) -> String {
    let names: BTreeSet<&str> = chart
        .as_object()
        .into_iter()
        .flatten()
        .map(|(k, _)| k.as_str())
        .collect();
    let ours: BTreeSet<&str> = TYPES.iter().copied().collect();
    if names != ours {
        fail(&format!("dump: the type chart's types are {names:?}"));
    }
    let mut b = String::from("use crate::types::Effectiveness::{self, *};\n\n");
    writeln!(
        b, "pub(crate) static TYPE_CHART: [[Effectiveness; {0}]; {0}] = [", TYPES.len()
    )
        .unwrap();
    for atk in TYPES {
        let row: Vec<&str> = TYPES
            .iter()
            .map(|def| match str_of(&chart[atk][*def]) {
                "neutral" => "Neutral",
                "superEffective" => "SuperEffective",
                "resistant" => "Resistant",
                "immune" => "Immune",
                e => fail(&format!("dump: effectiveness {e:?}")),
            })
            .collect();
        writeln!(b, "    [{}],", row.join(", ")).unwrap();
    }
    b.push_str("];\n");
    b
}
/// `"Tera Blast user"` to `TeraBlastUser`.
fn role_variant(label: &str) -> String {
    let mut out = String::new();
    for word in label.split_whitespace() {
        let mut chars = word.chars();
        if let Some(first) = chars.next() {
            out.extend(first.to_uppercase());
            out.extend(chars.flat_map(char::to_lowercase));
        }
    }
    out
}
/// Each PS set becomes one role, its lists in PS's order.
fn randbats_data(sets: &Value, species: &[Value]) -> String {
    let known: BTreeSet<&str> = species.iter().map(|e| str_of(&e["id"])).collect();
    let sets = sets
        .as_object()
        .unwrap_or_else(|| fail("dump: `randbatsSets` is not an object"));
    let mut roles: BTreeSet<&str> = BTreeSet::new();
    for entry in sets.values() {
        for set in entry["sets"].as_array().into_iter().flatten() {
            roles.insert(str_of(&set["role"]));
        }
    }
    let mut b = String::from(
        "use crate::generated::ability_id::AbilityId;\n\
         use crate::generated::move_id::MoveId;\n\
         use crate::generated::species_id::SpeciesId;\n\
         use crate::randbats::*;\n\
         use crate::types::Type;\n\n",
    );
    writeln!(b, "/// A set's role label.").unwrap();
    writeln!(
        b,
        "#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]\n#[repr(u8)]\npub enum RoleId {{"
    )
        .unwrap();
    for (i, r) in roles.iter().enumerate() {
        writeln!(b, "    {} = {i},", role_variant(r)).unwrap();
    }
    writeln!(b, "}}\n\nimpl RoleId {{\n    pub const COUNT: usize = {};", roles.len())
        .unwrap();
    writeln!(
        b,
        "    /// The label as PS writes it.\n    pub fn name(self) -> &'static str {{\n        match self {{"
    )
        .unwrap();
    for r in &roles {
        writeln!(b, "            RoleId::{} => {r:?},", role_variant(r)).unwrap();
    }
    writeln!(b, "        }}\n    }}\n}}\n").unwrap();
    writeln!(
        b,
        "pub static RANDBATS_BY_SPECIES: [Option<RandbatsEntry>; SpeciesId::COUNT] = {{"
    )
        .unwrap();
    writeln!(
        b,
        "    let mut out: [Option<RandbatsEntry>; SpeciesId::COUNT] = [None; SpeciesId::COUNT];"
    )
        .unwrap();
    let by_id: BTreeMap<&str, &Value> = sets
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect();
    for (id, entry) in by_id {
        if !known.contains(id) {
            fail(&format!("dump: randbats species {id:?} is not a species"));
        }
        let sets = entry["sets"]
            .as_array()
            .unwrap_or_else(|| fail(&format!("dump: {id} has no sets")));
        if sets.is_empty() {
            fail(&format!("dump: {id} has no sets"));
        }
        let list = |set: &Value, key: &str, lit: &dyn Fn(&str) -> String| -> String {
            set[key]
                .as_array()
                .into_iter()
                .flatten()
                .map(|v| lit(str_of(v)))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let mut role_lits = Vec::new();
        for set in sets {
            role_lits
                .push(
                    format!(
                        "RandbatsRole {{ id: RoleId::{}, abilities: &[{}], tera_types: &[{}], moves: &[{}] }}",
                        role_variant(str_of(& set["role"])), list(set, "abilities", &| a
                        | format!("WeightedAbility {{ id: AbilityId::{} }}",
                        variant(a))), list(set, "teraTypes", &| t |
                        format!("WeightedType {{ typ: Type::{} }}", type_variant(t))),
                        list(set, "movepool", &| m |
                        format!("WeightedMove {{ id: MoveId::{} }}", variant(m))),
                    ),
                );
        }
        writeln!(
            b,
            "    out[SpeciesId::{v} as usize] = Some(RandbatsEntry {{ species: SpeciesId::{v}, ps_order: {}, forme_draw: {}, level: {}, \
             roles: &[{}] }});",
            int(& entry["order"]), int(& entry["formeDraw"]), int(& entry["level"]),
            role_lits.join(", "), v = variant(id),
        )
            .unwrap();
    }
    writeln!(b, "    out\n}};").unwrap();
    b
}
