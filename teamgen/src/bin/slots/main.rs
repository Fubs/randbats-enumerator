//! The party-slot files, in pkmn/randbats' schema, from the exact slot weights
//! `exact_team_weights` writes (`--slots --slot5 [--slot6]`): each species'
//! sets given it lands in a party slot, or in one of the first k, as the
//! exact set tables of every (lead, `teamDetails`) it is rolled under there,
//! weighted by the probability of each.
//!
//! ```text
//! slots --weights SLOT_WEIGHTS.json --out DIR [--rb DUMP.json...] [--keep 1,2,3,4]
//!       [--meta META.json] [--generated YYYY-MM-DD] [--rb-threads T] [--rb-table OUT.json]
//!       [--partial] [--threads N]
//! ```
//!
//! The slots `--weights` carries are exact (it must be `exact_team_weights`'
//! output, not a merged estimate). A slot it lacks can be estimated from
//! `exact_team_weights --mc-dump N --mc-rb K` dumps (`--rb`, every dump of
//! the same K, seeds distinct): their expected counts summed and normalized
//! to the slot's mass, as MATH.md §7 describes.
//! `--keep` restricts which exact slots are read (the slot-5 backtest drops
//! slot 5 and estimates it instead).
//!
//! DIR gets `lead.json` (slot 1), `slot-K.json`, `first-K-slots.json` (in one
//! of slots 1–K; `first-6-slots.json` is the whole team), `non-lead.json`
//! (slots 2–6), `slot-probabilities.json` ({species: {slot: P(slot holds
//! it)}}) and `manifest.json` (which slots are exact; `--meta`'s keys, such
//! as the PS commit, merged in). Every slot 1–6 must be covered unless
//! `--partial`, which writes only the files whose slots are. MATH.md §8
//! describes the files and how they mix the weights.
mod json;
mod render;
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use randbats_gen9::generated::randbats_data::RANDBATS_BY_SPECIES;
use randbats_gen9::generated::species_id::SpeciesId;
use randbats_gen9::randbats::RandbatsEntry;
use randbats_gen9::randbats_gen::{SetOutcome, TeamDetails, enumerate_random_set};
use serde_json::Value;
use json::{Json, locale_cmp};
/// Party slot 1 is the lead; the rest follow in generation order. The
/// non-lead file pools slots 2–6: the lead is seen at a battle's start, so
/// any other member is known to stand in one of them.
const FILES: [(&str, &[u8]); 12] = [
    ("lead.json", &[1]),
    ("slot-2.json", &[2]),
    ("slot-3.json", &[3]),
    ("slot-4.json", &[4]),
    ("slot-5.json", &[5]),
    ("slot-6.json", &[6]),
    ("first-2-slots.json", &[1, 2]),
    ("first-3-slots.json", &[1, 2, 3]),
    ("first-4-slots.json", &[1, 2, 3, 4]),
    ("first-5-slots.json", &[1, 2, 3, 4, 5]),
    ("first-6-slots.json", &[1, 2, 3, 4, 5, 6]),
    ("non-lead.json", &[2, 3, 4, 5, 6]),
];
const CONTEXT_FLAGS: [&str; 12] = [
    "rain",
    "sun",
    "sand",
    "snow",
    "stealthRock",
    "toxicSpikes",
    "stickyWeb",
    "rapidSpin",
    "defog",
    "screens",
    "statusCure",
    "teraBlast",
];
/// Weights below this are the residue of a cancellation in the fold (a
/// species credited through its signature, then taken back because its
/// base is on the team), not probabilities.
const RESIDUE: f64 = 1e-17;
/// `_about` of `exact_team_weights`' own weights; a file without it (a
/// merged estimate, say) would have its estimated slot labelled exact.
const EXACT_ABOUT: &str = "computed exactly";
fn rb_method(k: u8) -> String {
    format!(
        "Rao-Blackwellized sampling (exact_team_weights --mc-dump N --mc-rb {k}): teams drawn through the PS-exact \
         port of the team loop up to pick {k}, and the remaining pool entries' forme labels drawn; pick {k}'s exact \
         distribution given those is added at its teamDetails context. Unbiased, not exact."
    )
}
fn details_of(code: u16) -> TeamDetails {
    let bit = |i: u16| code & (1 << i) != 0;
    TeamDetails {
        rain: bit(0),
        sun: bit(1),
        sand: bit(2),
        snow: bit(3),
        stealth_rock: bit(4),
        toxic_spikes: bit(5),
        sticky_web: bit(6),
        rapid_spin: bit(7),
        defog: bit(8),
        screens: bit(9),
        status_cure: bit(10),
        tera_blast: bit(11),
        spikes: ((code >> 12) & 3) as u8,
    }
}
struct Args {
    weights: PathBuf,
    out: PathBuf,
    rb: Vec<PathBuf>,
    keep: Option<BTreeSet<u8>>,
    meta: Option<PathBuf>,
    generated: Option<String>,
    rb_threads: Option<u64>,
    rb_table: Option<PathBuf>,
    partial: bool,
    threads: usize,
}
const USAGE: &str = "slots --weights SLOT_WEIGHTS.json --out DIR [--rb DUMP.json...] [--keep 1,2,3,4] [--meta META.json] \
    [--generated YYYY-MM-DD] [--rb-threads T] [--rb-table OUT.json] [--partial] [--threads N]";
fn parse_args() -> Args {
    let mut a = Args {
        weights: PathBuf::new(),
        out: PathBuf::new(),
        rb: Vec::new(),
        keep: None,
        meta: None,
        generated: None,
        rb_threads: None,
        rb_table: None,
        partial: false,
        threads: std::thread::available_parallelism().map_or(8, |n| n.get()),
    };
    let mut it = std::env::args().skip(1).peekable();
    while let Some(flag) = it.next() {
        let mut val = || {
            it.next().unwrap_or_else(|| panic!("{flag} takes a value\n{USAGE}"))
        };
        match flag.as_str() {
            "--weights" => a.weights = val().into(),
            "--out" => a.out = val().into(),
            "--meta" => a.meta = Some(val().into()),
            "--generated" => a.generated = Some(val()),
            "--rb-threads" => a.rb_threads = Some(val().parse().unwrap()),
            "--rb-table" => a.rb_table = Some(val().into()),
            "--threads" => a.threads = val().parse().unwrap(),
            "--keep" => {
                a.keep = Some(val().split(',').map(|s| s.parse().unwrap()).collect());
            }
            "--partial" => a.partial = true,
            "--rb" => {
                while let Some(p) = it.next_if(|s| !s.starts_with("--")) {
                    a.rb.push(p.into());
                }
            }
            _ => panic!("unknown argument {flag}\n{USAGE}"),
        }
    }
    assert!(
        ! a.weights.as_os_str().is_empty() && ! a.out.as_os_str().is_empty(), "{USAGE}"
    );
    a
}
fn read_json(path: &Path) -> Value {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("parsing {}: {e}", path.display()))
}
fn check_flags(root: &Value, path: &Path) {
    let flags: Vec<&str> = root["contextFlags"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(
        flags, CONTEXT_FLAGS, "{}: the context bits are in another order", path.display()
    );
}
/// A weight row: species name, slot, lead, context code, weight.
type Row = (String, u8, bool, u16, f64);
/// The estimated slot from `--mc-rb K` dumps: every (species, context)
/// cell's expected count summed over the dumps (by seed, each in file
/// order), then divided by their total, so the slot's mass is 1.
struct Estimate {
    slot: u8,
    rows: Vec<Row>,
    teams: u64,
    seeds: Vec<u64>,
    threads: u64,
}
/// One `--mc-rb` dump: its estimated slot's cells in file order.
struct Dump {
    seed: u64,
    slot: u8,
    teams: u64,
    threads: u64,
    cells: Vec<((String, u16), f64)>,
}
fn read_dump(path: &Path, rb_threads: Option<u64>) -> Dump {
    let root = read_json(path);
    check_flags(&root, path);
    let slot = root["rb"].as_u64().unwrap() as u8;
    assert!(
        slot >= 2, "{}: --rb takes Rao-Blackwellized dumps (`--mc-rb K`, K >= 2)", path
        .display()
    );
    let threads = root["threads"]
        .as_u64()
        .or(rb_threads)
        .unwrap_or_else(|| {
            panic!(
                "{} does not record its thread count; pass --rb-threads", path.display()
            )
        });
    let cells = root["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row.as_array().unwrap())
        .filter(|r| r[1].as_u64().unwrap() == slot as u64)
        .map(|r| {
            assert!(! r[2].as_bool().unwrap(), "slot {slot} is never the lead");
            (
                (r[0].as_str().unwrap().to_string(), r[3].as_u64().unwrap() as u16),
                r[4].as_f64().unwrap(),
            )
        })
        .collect();
    Dump {
        seed: root["seed"].as_u64().unwrap(),
        slot,
        teams: root["teams"].as_u64().unwrap(),
        threads,
        cells,
    }
}
fn estimate(paths: &[PathBuf], rb_threads: Option<u64>) -> Estimate {
    let mut dumps: Vec<Dump> = paths.iter().map(|p| read_dump(p, rb_threads)).collect();
    dumps.sort_by_key(|d| d.seed);
    let seeds: Vec<u64> = dumps.iter().map(|d| d.seed).collect();
    assert!(
        seeds.windows(2).all(| w | w[0] != w[1]),
        "two dumps of one seed share their streams: {seeds:?}"
    );
    let (slot, threads) = (dumps[0].slot, dumps[0].threads);
    assert!(dumps.iter().all(| d | d.slot == slot), "--rb dumps of different slots");
    assert!(
        dumps.iter().all(| d | d.threads == threads),
        "--rb dumps of different thread counts"
    );
    let teams = dumps.iter().map(|d| d.teams).sum();
    let mut index: HashMap<(String, u16), usize> = HashMap::new();
    let mut cells: Vec<((String, u16), f64)> = Vec::new();
    for d in dumps {
        for (key, n) in d.cells {
            match index.get(&key) {
                Some(&i) => cells[i].1 += n,
                None => {
                    index.insert(key.clone(), cells.len());
                    cells.push((key, n));
                }
            }
        }
    }
    let mut z = render::Sum::default();
    for c in &cells {
        z += c.1;
    }
    let z = z.get();
    let zoroark: f64 = cells
        .iter()
        .filter(|c| c.0.0.starts_with("Zoroark"))
        .fold(0.0, |s, c| s + c.1);
    println!(
        "slot {slot} estimated: {teams} teams over seeds {seeds:?}, {} cells, mass {:.12} of the teams, Zoroark {:e}",
        cells.len(), z / teams as f64, zoroark / z
    );
    let rows = cells
        .into_iter()
        .map(|((name, code), n)| (name, slot, false, code, n / z))
        .collect();
    Estimate {
        slot,
        rows,
        teams,
        seeds,
        threads,
    }
}
/// Σ xs rounded once, as Python's `math.fsum` (Shewchuk's partials).
fn fsum(xs: impl IntoIterator<Item = f64>) -> f64 {
    let mut partials: Vec<f64> = Vec::new();
    for mut x in xs {
        let mut i = 0;
        for j in 0..partials.len() {
            let mut y = partials[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                partials[i] = lo;
                i += 1;
            }
            x = hi;
        }
        partials.truncate(i);
        if x != 0.0 {
            partials.push(x);
        }
    }
    let Some(mut hi) = partials.pop() else {
        return 0.0;
    };
    let mut lo = 0.0;
    while let Some(y) = partials.pop() {
        let x = hi;
        hi = x + y;
        let yr = hi - x;
        lo = y - yr;
        if lo != 0.0 {
            break;
        }
    }
    if let Some(&next) = partials.last()
        && ((lo < 0.0 && next < 0.0) || (lo > 0.0 && next > 0.0))
    {
        let y = lo * 2.0;
        let x = hi + y;
        if y == x - hi {
            hi = x;
        }
    }
    hi
}
/// Today in UTC, as YYYY-MM-DD (Hinnant's `civil_from_days`).
fn today() -> String {
    let days = (std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() / 86400) as i64;
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}
fn write(dir: &Path, file: &str, j: &Json) {
    let path = dir.join(file);
    std::fs::write(&path, j.pretty())
        .unwrap_or_else(|e| panic!("writing {}: {e}", path.display()));
}
fn main() {
    let args = parse_args();
    let start = std::time::Instant::now();
    let root = read_json(&args.weights);
    check_flags(&root, &args.weights);
    let about = root["_about"].as_str().unwrap_or("");
    assert!(
        about.contains(EXACT_ABOUT),
        "{} is not exact_team_weights' exact output (its _about: {about:?})", args
        .weights.display()
    );
    let entries: HashMap<&str, &RandbatsEntry> = RANDBATS_BY_SPECIES
        .iter()
        .flatten()
        .map(|e| (e.species.data().name, e))
        .collect();
    let mut rows: Vec<Row> = Vec::new();
    let mut exact: BTreeSet<u8> = BTreeSet::new();
    for row in root["rows"].as_array().unwrap() {
        let r = row.as_array().unwrap();
        let slot = r[1].as_u64().unwrap() as u8;
        if args.keep.as_ref().is_some_and(|k| !k.contains(&slot)) {
            continue;
        }
        exact.insert(slot);
        rows.push((
            r[0].as_str().unwrap().to_string(),
            slot,
            r[2].as_bool().unwrap(),
            r[3].as_u64().unwrap() as u16,
            r[4].as_f64().unwrap(),
        ));
    }
    drop(root);
    if let Some(keep) = &args.keep {
        assert!(
            keep.is_subset(& exact),
            "--keep {keep:?}, but the weights hold slots {exact:?}"
        );
    }
    let est = (!args.rb.is_empty()).then(|| estimate(&args.rb, args.rb_threads));
    if let Some(e) = &est {
        assert!(
            ! exact.contains(& e.slot),
            "slot {} is exact in the weights and estimated by --rb; --keep the others", e
            .slot
        );
        if let Some(path) = &args.rb_table {
            let mut t: Vec<&Row> = e.rows.iter().collect();
            t.sort_by(|a, b| (&a.0, a.3).cmp(&(&b.0, b.3)));
            let v = Value::Array(
                t
                    .iter()
                    .map(|r| {
                        Value::from(
                            vec![
                                Value::from(r.0.as_str()), Value::from(r.3), Value::from(r
                                .4),
                            ],
                        )
                    })
                    .collect(),
            );
            std::fs::write(path, v.to_string()).unwrap();
        }
        rows.extend(e.rows.iter().cloned());
    }
    let covered: BTreeSet<u8> = exact
        .iter()
        .copied()
        .chain(est.as_ref().map(|e| e.slot))
        .collect();
    let missing: Vec<u8> = (1..=6).filter(|s| !covered.contains(s)).collect();
    assert!(
        args.partial || missing.is_empty(),
        "no weights for slot(s) {missing:?} (--partial writes what they leave)"
    );
    rows.sort_by(|a, b| (&a.0, a.1, a.2, a.3).cmp(&(&b.0, b.1, b.2, b.3)));
    let mut cells: HashMap<SpeciesId, Vec<(u8, bool, u16, f64)>> = HashMap::new();
    let mut slot_probs: Vec<(String, u8, f64)> = Vec::new();
    let (mut kept, mut residue) = (0usize, 0usize);
    let mut group: Vec<f64> = Vec::new();
    for (i, (name, slot, lead, code, w)) in rows.iter().enumerate() {
        if w.abs() >= RESIDUE {
            assert!(* w > 0.0, "a negative weight {w} for {name}");
            kept += 1;
            let entry = entries
                .get(name.as_str())
                .unwrap_or_else(|| panic!("{name} is not in the live pool"));
            cells.entry(entry.species).or_default().push((*slot, *lead, *code, *w));
            group.push(*w);
        } else {
            residue += 1;
        }
        let last = rows.get(i + 1).is_none_or(|n| (&n.0, n.1) != (name, *slot));
        if last && !group.is_empty() {
            slot_probs.push((name.clone(), *slot, fsum(group.drain(..))));
        }
    }
    let mut species: Vec<SpeciesId> = cells.keys().copied().collect();
    species.sort_by(|a, b| locale_cmp(a.data().name, b.data().name));
    println!(
        "{kept} weights over {} species ({residue} residues below {RESIDUE:e} dropped)",
        species.len()
    );
    for s in &covered {
        let mass = fsum(slot_probs.iter().filter(|p| p.1 == *s).map(|p| p.2));
        println!("slot {s}: mass {mass:.15}");
    }
    let files: Vec<(&str, &[u8])> = FILES
        .iter()
        .copied()
        .filter(|(_, slots)| slots.iter().all(|s| covered.contains(s)))
        .collect();
    let rendered: Vec<Mutex<Vec<Option<(Json, render::Notes)>>>> = species
        .iter()
        .map(|_| Mutex::new(Vec::new()))
        .collect();
    let cursor = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..args.threads {
            scope
                .spawn(|| {
                    loop {
                        let i = cursor.fetch_add(1, Ordering::Relaxed);
                        if i >= species.len() {
                            break;
                        }
                        let s = species[i];
                        let entry = entries[s.data().name];
                        let mine = &cells[&s];
                        let mut tables: HashMap<(bool, u16), Vec<SetOutcome>> = HashMap::new();
                        for &(_, lead, code, _) in mine {
                            tables
                                .entry((lead, code))
                                .or_insert_with(|| enumerate_random_set(
                                    s,
                                    &details_of(code),
                                    lead,
                                ));
                        }
                        let out = files
                            .iter()
                            .map(|(_, slots)| {
                                let mut cells = mine
                                    .iter()
                                    .filter(|r| slots.contains(&r.0))
                                    .peekable();
                                cells.peek()?;
                                let mut notes = render::Notes::default();
                                let j = render::render(
                                    entry,
                                    cells
                                        .flat_map(|&(_, lead, code, w)| {
                                            tables[&(lead, code)].iter().map(move |o| (w * o.prob, o))
                                        }),
                                    &mut notes,
                                );
                                Some((j, notes))
                            })
                            .collect();
                        *rendered[i].lock().unwrap() = out;
                    }
                });
        }
    });
    std::fs::create_dir_all(&args.out).unwrap();
    for (f, (file, _)) in files.iter().enumerate() {
        let mut out = Vec::new();
        let mut empty_roles = Vec::new();
        for (i, s) in species.iter().enumerate() {
            let mut r = rendered[i].lock().unwrap();
            if let Some((j, notes)) = r[f].take() {
                empty_roles
                    .extend(
                        notes
                            .empty_roles
                            .iter()
                            .map(|role| format!("{} {role}", s.data().name)),
                    );
                out.push((s.data().name.to_string(), j));
            }
        }
        let n = out.len();
        write(&args.out, file, &Json::Object(out));
        println!("{file}: {n} species");
        if !empty_roles.is_empty() {
            println!("  roles no set takes here (weight 0): {}", empty_roles.join(", "));
        }
    }
    let mut by_species: Vec<(String, Vec<(String, Json)>)> = Vec::new();
    for (name, slot, p) in &slot_probs {
        if by_species.last().is_none_or(|l| &l.0 != name) {
            by_species.push((name.clone(), Vec::new()));
        }
        by_species
            .last_mut()
            .unwrap()
            .1
            .push((slot.to_string(), Json::Leaf(Value::from(*p))));
    }
    by_species.sort_by(|a, b| locale_cmp(&a.0, &b.0));
    write(
        &args.out,
        "slot-probabilities.json",
        &Json::Object(
            by_species.into_iter().map(|(n, v)| (n, Json::Object(v))).collect(),
        ),
    );
    let mut manifest = vec![
        ("format".to_string(), Json::Leaf(Value::from("gen9randombattle")),),
        ("exact_slots".to_string(), Json::Leaf(Value::from(exact.iter().copied()
        .collect::< Vec < u8 >> ())),), ("estimated_slots".to_string(),
        Json::Leaf(Value::from(est.iter().map(| e | e.slot).collect::< Vec < u8 >>
        ())),),
    ];
    if let Some(e) = &est {
        manifest
            .push((
                format!("slot{}_estimate", e.slot),
                Json::Object(
                    vec![
                        ("method".into(), Json::Leaf(Value::from(rb_method(e.slot)))),
                        ("teams".into(), Json::Leaf(Value::from(e.teams))), ("seeds"
                        .into(), Json::Leaf(Value::from(e.seeds.clone()))), ("threads"
                        .into(), Json::Leaf(Value::from(e.threads))),
                    ],
                ),
            ));
    }
    manifest
        .push((
            "generated".into(),
            Json::Leaf(Value::from(args.generated.clone().unwrap_or_else(today))),
        ));
    if let Some(path) = &args.meta {
        let Value::Object(meta) = read_json(path) else {
            panic!("{}: --meta takes a JSON object", path.display())
        };
        for (k, v) in meta {
            assert!(
                manifest.iter().all(| m | m.0 != k),
                "--meta may not set the manifest's own {k:?}"
            );
            manifest.push((k, Json::Leaf(v)));
        }
    }
    write(&args.out, "manifest.json", &Json::Object(manifest));
    println!("wrote {} in {:.0}s", args.out.display(), start.elapsed().as_secs_f64());
}
