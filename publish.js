#!/usr/bin/env node
'use strict';

// Renders teamgen's output into data/gen9randombattle/, in the per-species schema of
// pkmn/randbats' data/stats/gen9randombattle.json and at full double precision.
//
//   node publish.js [DIR]   DIR defaults to teamgen/output/gen9randombattle
//
// The vendored Pokémon Showdown must be built and must match the output's manifest: the same
// `fingerprint` and `logic`.

const fs = require('fs');
const path = require('path');
const { FORMAT, DATA, PS, SETS, compare, inputs, stamp, write, writeManifest } = require('./check.js');

const STATS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
// Sorting ignores float residue (0.49999999999999994 against 0.5), so near-ties fall to the name.
const order = p => Number(p.toPrecision(12));
const byValue = (a, b) => order(b[1]) - order(a[1]) || a[0].localeCompare(b[0]);
const byName = (a, b) => a[0].localeCompare(b[0]);
const byStat = (a, b) => STATS.indexOf(a[0]) - STATS.indexOf(b[0]);

const distribution = entries => Object.fromEntries(Array.from(entries).sort(byValue));
const positive = obj => Object.entries(obj || {}).filter(e => e[1] > 0);
const spread = obj => {
  const entries = Object.entries(obj || {}).filter(e => e[1] !== -1).sort(byStat);
  return entries.length ? Object.fromEntries(entries) : undefined;
};

// What pkmn's sampler converges to: every movepool move and role Tera type is listed, at 0 when
// it is never generated; abilities and items only when they can be.
const render = (dex, sets, exact) => {
  const out = {};
  for (const [name, data] of Object.entries(exact).sort(byName)) {
    const s = { level: data.level, abilities: distribution(positive(data.abilities)) };
    const items = positive(data.items);
    if (items.length) s.items = distribution(items);

    // The output's roles in its order, then any other role in the sets data, as the sampler
    // seeds every role. A label used by several sets keeps the last one's movepool, as the
    // sampler does. The sort is stable, so roles of exactly equal weight keep the output's order.
    const roles = new Map(Object.keys(data.roles).map(role => [role, undefined]));
    for (const set of sets[dex.toID(name)]?.sets || []) roles.set(set.role, set);
    const weight = role => data.roles[role]?.weight || 0;
    s.roles = {};
    for (const [role, set] of Array.from(roles).sort((a, b) => order(weight(b[0])) - order(weight(a[0])))) {
      const r = data.roles[role] || {};
      const o = { weight: weight(role), abilities: distribution(positive(r.abilities)) };
      const roleItems = positive(r.items);
      if (roleItems.length) o.items = distribution(roleItems);
      if (set?.teraTypes || r.teraTypes) {
        const tera = new Map((set?.teraTypes || []).map(type => [type, 0]));
        for (const [type, p] of positive(r.teraTypes)) tera.set(type, p);
        o.teraTypes = distribution(tera);
      }
      const moves = new Map((set?.movepool || []).map(m => [dex.toID(m), 0]));
      for (const [move, p] of positive(r.moves)) moves.set(dex.toID(move), p);
      o.moves = distribution(Array.from(moves).map(([id, p]) => [dex.moves.get(id).name, p]));
      o.evs = spread(r.evs);
      o.ivs = spread(r.ivs);
      s.roles[role] = o;
    }

    s.evs = spread(data.evs);
    s.ivs = spread(data.ivs);
    out[name] = s;
  }
  return out;
};

// Every slot listed for every species, 0 where it never stands.
const renderSlots = probabilities =>
  Object.fromEntries(Object.entries(probabilities).sort(byName).map(([name, p]) =>
    [name, Object.fromEntries([1, 2, 3, 4, 5, 6].map(k => [k, p[k] || 0]))]));

const main = () => {
  const dir = path.resolve(process.argv[2] || path.join(__dirname, 'teamgen', 'output', FORMAT));
  const c = compare(dir);
  if (!c.readable) throw new Error(c.problem);
  if (!c.fingerprint_match || !c.logic_match) {
    throw new Error(
      `${path.relative(__dirname, dir)} was computed at smogon/pokemon-showdown@${c.manifest_ps}, ` +
      `and vendor/pokemon-showdown (${c.ps}) differs in ` +
      [!c.fingerprint_match && 'fingerprint', !c.logic_match && 'logic'].filter(Boolean).join(' and '));
  }

  const { Dex } = require(PS);
  const dex = Dex.forFormat(Dex.formats.get(FORMAT));
  const sets = JSON.parse(fs.readFileSync(path.join(PS, 'data', 'random-battles', 'gen9', 'sets.json'), 'utf8'));
  const read = name => JSON.parse(fs.readFileSync(path.join(dir, `${name}.json`), 'utf8'));

  fs.rmSync(DATA, { recursive: true, force: true });
  fs.mkdirSync(DATA, { recursive: true });
  for (const name of SETS) write(path.join(DATA, `${name}.json`), render(dex, sets, read(name)));
  write(path.join(DATA, 'slot-probabilities.json'), renderSlots(read('slot-probabilities')));
  const { format, ...rest } = c.manifest;
  writeManifest(DATA, stamp({ format, fresh: true, ...rest }, c.ps, inputs()));
  console.log(`published ${path.relative(__dirname, dir)} (smogon/pokemon-showdown@${c.manifest_ps}) to ` +
    path.relative(__dirname, DATA));
};

try {
  main();
} catch (err) {
  console.error(err.message);
  process.exit(1);
}
