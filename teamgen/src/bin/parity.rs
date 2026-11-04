//! Team-generator parity with PS, draw for draw (MATH.md §9): replays each
//! trace `trace.js` recorded through the port and reports every team the
//! port did not rebuild exactly.
//!
//! ```text
//! node trace.js --teams 100000 --sites | target/release/parity --expect 100000
//! ```
//!
//! Reads trace lines from the file named on the command line, or stdin.
//! Divergences are tallied by where they happened (the port's ask, and
//! PS's call site with `--sites`), each with its first seed. Exits 0 only
//! when every team was rebuilt exactly, at least one was read, and, with
//! `--expect N`, exactly N were. `--seed S` instead prints the trace whose
//! seed is `S`, PS's draws beside the port's asks, up to the divergence.
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::sync::Mutex;
use randbats_gen9::randbats_gen::{
    PsDraw, PsSet, formes_of, generate_team_from_ps_draws, ps_replay_log,
    ps_team_mismatches, weighted_base_pool,
};
#[derive(serde::Deserialize)]
struct Trace {
    seed: String,
    draws: Vec<PsDraw>,
    team: Vec<PsSet>,
    #[serde(default)]
    sites: Vec<String>,
}
/// What went wrong, keyed coarsely enough to group repeats, with the
/// full report of its first occurrence.
fn classify(t: &Trace) -> Option<(String, String)> {
    match generate_team_from_ps_draws(&t.draws) {
        Err(e) => {
            let at: Option<usize> = e
                .strip_prefix("draw ")
                .and_then(|r| r.split(':').next())
                .and_then(|n| n.parse().ok());
            let site = at
                .and_then(|i| t.sites.get(i + 1))
                .map(String::as_str)
                .unwrap_or("?");
            let ask = e
                .split("the port asked ")
                .nth(1)
                .and_then(|r| r.split(" random(").next())
                .unwrap_or("?");
            Some((
                format!("draw: port asked {ask}, PS at {site}"),
                format!("{}: {e} [PS at {site}]", t.seed),
            ))
        }
        Ok(team) => {
            let bad = ps_team_mismatches(&team, &t.team);
            let first = bad.first()?;
            let field = first
                .split(") ")
                .nth(1)
                .and_then(|r| r.split(':').next())
                .unwrap_or("?");
            Some((format!("field: {field}"), format!("{}: {}", t.seed, bad.join("; "))))
        }
    }
}
/// Whether a draw's `site` (`trace.js`'s `line:function` frames, innermost
/// first) has `callee` called straight from `randomTeam`: the base-species
/// and forme draws, found by name so that `teams.ts` line shifts don't
/// matter.
fn called_from_random_team(site: &str, callee: &str) -> bool {
    let frames: Vec<&str> = site.split('<').collect();
    frames
        .windows(2)
        .any(|w| {
            w[0].rsplit(':').next() == Some(callee) && w[1].ends_with(":randomTeam")
        })
}
/// One trace, PS's draws beside the port's asks, through the divergence.
fn show(t: &Trace) {
    let (log, err) = ps_replay_log(&t.draws);
    let fmt = |d: &PsDraw| match d.1 {
        None => format!("random({})", d.0),
        Some(to) => format!("random({}, {to})", d.0),
    };
    let mut pool = weighted_base_pool().to_vec();
    let mut base = None;
    for (i, d) in t.draws.iter().skip(1).enumerate().take(log.len() + 3) {
        let site = t.sites.get(i + 1).map(String::as_str).unwrap_or("");
        let port = log.get(i).map(String::as_str).unwrap_or("");
        let mut drawn = String::new();
        if called_from_random_team(site, "sampleNoReplace")
            && (d.2 as usize) < pool.len()
        {
            base = Some(pool.swap_remove(d.2 as usize));
        } else if called_from_random_team(site, "sample") {
            if let Some(f) = base
                .take()
                .and_then(|b| formes_of(b).get(d.2 as usize).copied())
            {
                drawn = format!("-> {}", f.data().name);
            }
        }
        println!(
            "{i:>4}  PS {:<16} = {:<4} {site:<60} | port {port} {drawn}", fmt(d), d.2
        );
    }
    if let Some(e) = err {
        println!("{e}");
    }
    let team: Vec<&str> = t.team.iter().map(|s| s.species_id.as_str()).collect();
    println!("PS team: {}", team.join(" "));
}
fn main() {
    let mut args = std::env::args().skip(1);
    let mut path = None;
    let mut seed = None;
    let mut expect: Option<u64> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--seed" => seed = args.next(),
            "--expect" => {
                expect = Some(
                    args
                        .next()
                        .and_then(|n| n.parse().ok())
                        .expect("--expect takes a team count"),
                );
            }
            _ => path = Some(a),
        }
    }
    let input: Box<dyn BufRead + Send> = match path {
        Some(path) => {
            Box::new(BufReader::new(std::fs::File::open(path).expect("open trace file")))
        }
        None => Box::new(BufReader::new(std::io::stdin())),
    };
    if let Some(seed) = seed {
        for line in input.lines() {
            let line = line.expect("read trace line");
            if line.contains(&seed) {
                show(&serde_json::from_str(&line).expect("parse trace line"));
                return;
            }
        }
        panic!("no trace with seed {seed}");
    }
    let input = Mutex::new(input.lines());
    let tally: Mutex<(u64, BTreeMap<String, (u64, String)>)> = Mutex::new((
        0,
        BTreeMap::new(),
    ));
    std::thread::scope(|s| {
        for _ in 0..std::thread::available_parallelism().map_or(8, |n| n.get()) {
            s.spawn(|| {
                loop {
                    let Some(line) = input.lock().unwrap().next() else { break };
                    let line = line.expect("read trace line");
                    let t: Trace = serde_json::from_str(&line)
                        .expect("parse trace line");
                    let verdict = classify(&t);
                    let mut g = tally.lock().unwrap();
                    g.0 += 1;
                    if let Some((key, report)) = verdict {
                        g.1.entry(key).or_insert((0, report)).0 += 1;
                    }
                }
            });
        }
    });
    let (teams, kinds) = tally.into_inner().unwrap();
    let bad: u64 = kinds.values().map(|(n, _)| n).sum();
    println!("{teams} teams, {} rebuilt exactly, {bad} not", teams - bad);
    let mut kinds: Vec<_> = kinds.into_iter().collect();
    kinds.sort_by(|a, b| b.1.0.cmp(&a.1.0));
    for (key, (n, first)) in kinds {
        println!("\n{n:>8}  {key}\n          first: {first}");
    }
    if teams == 0 {
        eprintln!("no traces read");
        std::process::exit(1);
    }
    if let Some(n) = expect.filter(|&n| n != teams) {
        eprintln!("expected {n} traces, read {teams}");
        std::process::exit(1);
    }
    if bad > 0 {
        std::process::exit(1);
    }
}
