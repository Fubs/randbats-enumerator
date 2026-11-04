#!/usr/bin/env node
'use strict';

// Writes the Pokémon Showdown data the Gen 9 Random Battles generator port
// reads, as one JSON file the `randbats-gen9` build script turns into Rust
// tables, and fingerprints it.
//
//   node teamgen/dump.js [--ps DIR] [--out FILE]   write the dump
//   node teamgen/dump.js [--ps DIR] --fingerprint  print {ps, fingerprint, logic}
//
// DIR defaults to the vendored checkout (../vendor/pokemon-showdown), which
// must be built (`node build` there, so that dist/ exists). FILE defaults to
// target/dump.json beside this script, where the build script looks for it.
//
// `fingerprint` hashes the data the port reads, and the few fields PS's
// generator reads that the port does not (no set reaches them); `logic`
// hashes the PS source files the port is a line-by-line translation of. A
// fingerprint change
// means the tables must be regenerated; a logic change means the port must be
// checked against PS again (`trace.js | parity`).

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const DEFAULT_PS = path.join(__dirname, '..', 'vendor', 'pokemon-showdown');
const DEFAULT_OUT = path.join(__dirname, 'target', 'dump.json');

// The PS sources the port is a translation of.
const LOGIC = ['data/random-battles/gen9/teams.ts', 'sim/prng.ts', 'sim/dex.ts'];

const FORMAT = 'gen9randombattle';

function loadPS(psDir) {
  const entry = path.join(psDir, 'dist', 'sim');
  if (!fs.existsSync(path.join(entry, 'index.js'))) {
    throw new Error(`${entry} not found: build Pokémon Showdown first (\`node build\` in ${psDir})`);
  }
  return require(entry);
}

// Sorted keys at every level; arrays keep their order.
function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value !== null && typeof value === 'object') {
    const out = {};
    for (const key of Object.keys(value).sort()) out[key] = canonical(value[key]);
    return out;
  }
  return value;
}

const sha256 = data => crypto.createHash('sha256').update(data).digest('hex');

function genderRatio(s) {
  if (s.gender === 'N') return { kind: 'genderless' };
  if (s.gender === 'M') return { kind: 'alwaysMale' };
  if (s.gender === 'F') return { kind: 'alwaysFemale' };
  if (s.genderRatio && typeof s.genderRatio.M === 'number') {
    return { kind: 'ratio', maleEighths: Math.round(s.genderRatio.M * 8) };
  }
  return { kind: 'ratio', maleEighths: 4 };
}

function byId(a, b) {
  return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
}

function species(dex, toID) {
  return dex.species.all()
    .filter(s => !s.isCosmeticForme)
    .map(s => {
      const abilities = s.abilities || {};
      return {
        id: toID(s.name),
        name: s.name,
        types: [s.types[0], s.types[1] || null],
        baseStats: {
          hp: s.baseStats.hp, atk: s.baseStats.atk, def: s.baseStats.def,
          spa: s.baseStats.spa, spd: s.baseStats.spd, spe: s.baseStats.spe,
        },
        abilities: {
          slot1: toID(abilities[0] || ''),
          slot2: abilities[1] ? toID(abilities[1]) : null,
          hidden: abilities.H ? toID(abilities.H) : null,
          special: abilities.S ? toID(abilities.S) : null,
        },
        genderRatio: genderRatio(s),
        baseSpecies: toID(s.baseSpecies || s.name),
        formeName: s.forme || null,
        requiredItem: s.requiredItem ? toID(s.requiredItem) : null,
        // Source order: the generator gives Arceus requiredItems[0].
        requiredItems: Array.isArray(s.requiredItems) && s.requiredItems.length
          ? s.requiredItems.map(i => toID(i))
          : s.requiredItem ? [toID(s.requiredItem)] : [],
        requiredMove: s.requiredMove ? toID(s.requiredMove) : null,
        nfe: !!s.nfe,
        // Read by PS's generator, not by the port (no set it reaches turns
        // on it): here so that a change to it moves the fingerprint.
        isMega: !!s.isMega,
      };
    })
    .sort(byId);
}

function moves(dex, toID) {
  return dex.moves.all()
    .map(m => ({
      id: toID(m.name),
      name: m.name,
      type: m.type,
      category: m.category,
      basePower: m.basePower || 0,
      accuracy: m.accuracy === true ? 'always' : m.accuracy,
      priority: m.priority || 0,
      flags: { bite: !!m.flags.bite, punch: !!m.flags.punch, sound: !!m.flags.sound },
      multihit: m.multihit || null,
      drain: m.drain || null,
      recoil: m.recoil || null,
      secondary: !!m.secondary,
      hasBasePowerCallback: !!m.basePowerCallback,
      setDamage: !!(m.damage || m.damageCallback),
      // Read by PS's generator, not by the port (no set it reaches turns on
      // them): here so that a change to one moves the fingerprint.
      hasCrashDamage: !!m.hasCrashDamage,
      hasSheerForceBoost: !!m.hasSheerForceBoost,
    }))
    .sort(byId);
}

function named(list, toID) {
  return list.map(x => ({ id: toID(x.name), name: x.name })).sort(byId);
}

// typechart[attacker][defender], PS's damageTaken codes as words.
function typechart(dex) {
  const chart = dex.data.TypeChart;
  const names = Object.keys(chart).map(t => t[0].toUpperCase() + t.slice(1)).sort();
  const label = code => ['neutral', 'superEffective', 'resistant', 'immune'][code || 0];
  const out = {};
  for (const atk of names) {
    out[atk] = {};
    for (const def of names) out[atk][def] = label(chart[def.toLowerCase()].damageTaken[atk]);
  }
  return out;
}

// The format's rules the generator reads; the build script checks them
// against the ones the port assumes.
function format(Dex) {
  const f = Dex.formats.get(FORMAT);
  if (!f.exists) throw new Error(`${FORMAT} is not a format`);
  const rules = Dex.formats.getRuleTable(f);
  return {
    mod: f.mod,
    gameType: f.gameType,
    team: f.team,
    maxTeamSize: rules.maxTeamSize,
    maxMoveCount: rules.maxMoveCount,
    adjustLevel: rules.adjustLevel || null,
    rules: [...rules.keys()].sort(),
    valueRules: Object.fromEntries([...rules.valueRules].sort()),
    banlist: f.banlist,
    restricted: f.restricted,
    unbanlist: f.unbanlist,
    customRules: f.customRules || [],
  };
}

// gen9 `sets.json`, ids in place of names, every array in PS's order (a draw
// is an index into it). `order` is the species' place in the file, the order
// the base pool is built in; `formeDraw` is the range of the cosmetic forme
// draw `getForme` makes before anything else in the set, 0 for none.
function randbatsSets(psDir, dex, Teams, toID) {
  const file = path.join(psDir, 'data', 'random-battles', 'gen9', 'sets.json');
  const raw = JSON.parse(fs.readFileSync(file, 'utf8'));
  const generator = Teams.getGenerator(FORMAT);
  const formeDraw = id => {
    let range = 0;
    const random = generator.prng.random.bind(generator.prng);
    generator.prng.random = (from, to) => {
      if (range !== 0 || to !== undefined) throw new Error(`getForme(${id}) drew more than one random(n)`);
      range = from;
      return random(from, to);
    };
    try {
      generator.getForme(dex.species.get(id));
    } finally {
      generator.prng.random = random;
    }
    return range;
  };
  const out = {};
  Object.keys(raw).forEach((id, order) => {
    out[id] = {
      order,
      formeDraw: formeDraw(id),
      level: raw[id].level,
      sets: raw[id].sets.map(s => ({
        role: s.role,
        movepool: s.movepool.map(m => toID(m)),
        abilities: s.abilities.map(a => toID(a)),
        teraTypes: [...s.teraTypes],
      })),
    };
  });
  return out;
}

function psCommit(psDir) {
  let commit;
  try {
    commit = execFileSync('git', ['-C', psDir, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
  } catch (err) {
    throw new Error(`cannot read the Pokémon Showdown commit in ${psDir}: ${err.message}`);
  }
  if (!/^[0-9a-f]{40}$/.test(commit)) throw new Error(`unexpected commit '${commit}' in ${psDir}`);
  return commit;
}

function logic(psDir) {
  const hash = crypto.createHash('sha256');
  for (const file of LOGIC) {
    hash.update(file + '\0');
    hash.update(fs.readFileSync(path.join(psDir, file)));
    hash.update('\0');
  }
  return hash.digest('hex');
}

function dump(psDir = DEFAULT_PS) {
  psDir = path.resolve(psDir);
  const { Dex, Teams, toID } = loadPS(psDir);
  const dex = Dex.mod('gen9');
  const data = canonical({
    species: species(dex, toID),
    moves: moves(dex, toID),
    items: named(dex.items.all(), toID),
    abilities: named(dex.abilities.all(), toID),
    typechart: typechart(dex),
    format: format(Dex),
    randbatsSets: randbatsSets(psDir, dex, Teams, toID),
  });
  return {
    ps: psCommit(psDir),
    fingerprint: sha256(JSON.stringify(data)),
    logic: logic(psDir),
    data,
  };
}

function fingerprint(psDir = DEFAULT_PS) {
  const { ps, fingerprint, logic } = dump(psDir);
  return { ps, fingerprint, logic };
}

module.exports = { dump, fingerprint };

if (require.main === module) {
  const args = process.argv.slice(2);
  let psDir = DEFAULT_PS;
  let out = DEFAULT_OUT;
  let print = false;
  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--ps': psDir = args[++i]; break;
      case '--out': out = args[++i]; break;
      case '--fingerprint': print = true; break;
      default:
        console.error(`unknown argument: ${args[i]}`);
        process.exit(2);
    }
  }
  try {
    if (print) {
      console.log(JSON.stringify(fingerprint(psDir)));
    } else {
      const result = dump(psDir);
      fs.mkdirSync(path.dirname(path.resolve(out)), { recursive: true });
      fs.writeFileSync(out, JSON.stringify(result) + '\n');
      console.error(`wrote ${out} (PS ${result.ps}, fingerprint ${result.fingerprint.slice(0, 12)})`);
    }
  } catch (err) {
    console.error(err.message);
    process.exit(1);
  }
}
