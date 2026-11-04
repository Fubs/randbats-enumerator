#!/usr/bin/env node
'use strict';

// Whether PS changed nothing the published files depend on but species' levels, and the patch
// that carries the new levels into them.
//
//   node levels.js --base     the PS commit whose inputs the published files describe
//   node levels.js --old DIR --new DIR [--teams N] [--start S] [--workers W] [--every SECONDS]
//                             the comparison of two built PS checkouts, as JSON
//
// gen9randombattle's generator reads a level in three places: randomTeam's limit of one level-100
// member, the set's own level, and randomSet's HP EV tuning, which draws nothing. A change of
// levels alone, none to or from 100, therefore leaves every probability as it was. The EVs can
// still move, and a future rule could read a level elsewhere, so PS settles it: the same seeds
// must build the same teams at both commits, those levels apart.

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const readline = require('readline');
const { spawn } = require('child_process');
const { FORMAT, DATA, SETS, writeManifest, write } = require('./check.js');

const TRACE = path.join(__dirname, 'teamgen', 'trace.js');
const toID = s => String(s).toLowerCase().replace(/[^a-z0-9]+/g, '');

const readManifest = () => JSON.parse(fs.readFileSync(path.join(DATA, 'manifest.json'), 'utf8'));

// The PS commit whose inputs the files describe: the last level patch's, or the computed one.
const base = manifest => manifest.patches?.length ? manifest.patches.at(-1).ps : manifest.ps;

// What differs between the generator's inputs at the two checkouts.
const inputs = (oldDir, newDir) => {
  const { dump } = require('./teamgen/dump.js');
  const a = dump(oldDir);
  const b = dump(newDir);
  const changes = {};
  const other = [];
  const walk = (x, y, at) => {
    if (JSON.stringify(x) === JSON.stringify(y)) return;
    if (x && y && typeof x === 'object' && typeof y === 'object' && Array.isArray(x) === Array.isArray(y)) {
      for (const k of new Set([...Object.keys(x), ...Object.keys(y)])) walk(x[k], y[k], [...at, k]);
    } else if (at.length === 3 && at[0] === 'randbatsSets' && at[2] === 'level') {
      changes[at[1]] = [x, y];
    } else {
      other.push(at.join('.'));
    }
  };
  walk(a.data, b.data, []);
  return {
    old_ps: a.ps, new_ps: b.ps,
    old_fingerprint: a.fingerprint, new_fingerprint: b.fingerprint,
    changes, other,
  };
};

// One trace line, with the changed species' levels checked against `side` and then left out.
const normalize = (line, changes, side) => {
  const t = JSON.parse(line);
  for (const set of t.team) {
    const change = changes[set.speciesId];
    if (!change) continue;
    if (set.level !== change[side]) return { seed: t.seed, bad: `${set.speciesId} at level ${set.level}` };
    delete set.level;
  }
  const hash = crypto.createHash('sha256').update(JSON.stringify([t.draws, t.team])).digest('hex');
  return { seed: t.seed, hash };
};

const trace = (dir, opts, onLine) => new Promise((resolve, reject) => {
  const args = [TRACE, '--teams', opts.teams, '--start', opts.start, '--workers', opts.workers, '--ps', dir];
  const child = spawn(process.execPath, args.map(String), { stdio: ['ignore', 'pipe', 'inherit'] });
  const lines = readline.createInterface({ input: child.stdout, crlfDelay: Infinity });
  lines.on('line', onLine);
  child.once('error', reject);
  child.once('close', code => (code === 0 ? resolve() : reject(new Error(`trace.js --ps ${dir} exited ${code}`))));
});

// The same seeds through both checkouts: how many teams differ beyond the changed levels.
const sameTeams = async (oldDir, newDir, changes, opts) => {
  const seen = new Map();
  let first = null;
  let differing = 0;
  let done = 0;
  const started = Date.now();
  const timer = setInterval(() => {
    const s = Math.round((Date.now() - started) / 1000);
    console.error(`levels: ${done.toLocaleString('en-US')} of ${(2 * opts.teams).toLocaleString('en-US')} traces in ${s}s`);
  }, opts.every * 1000);
  const differ = (seed, why) => {
    differing++;
    if (!first) first = `${seed}: ${why}`;
  };
  try {
    await trace(oldDir, opts, line => {
      const n = normalize(line, changes, 0);
      done++;
      if (n.bad) differ(n.seed, `the old checkout built ${n.bad}`);
      else seen.set(n.seed, n.hash);
    });
    let compared = 0;
    await trace(newDir, opts, line => {
      const n = normalize(line, changes, 1);
      done++;
      compared++;
      if (n.bad) differ(n.seed, `the new checkout built ${n.bad}`);
      else if (!seen.has(n.seed)) differ(n.seed, 'missing from the old checkout');
      else if (seen.get(n.seed) !== n.hash) differ(n.seed, 'a different team or different draws');
    });
    return { teams: compared, differing, first_difference: first };
  } finally {
    clearInterval(timer);
  }
};

const compare = async (oldDir, newDir, opts) => {
  const d = inputs(oldDir, newDir);
  const out = { ...d, level_only: false, patchable: false };
  const crossing = Object.entries(d.changes).filter(([, [a, b]]) => a === 100 || b === 100);
  if (d.old_fingerprint === d.new_fingerprint) {
    out.reason = 'the inputs are the same';
  } else if (d.other.length) {
    out.reason = `more than levels changed: ${d.other.slice(0, 5).join(', ')}${d.other.length > 5 ? ', …' : ''}`;
  } else if (crossing.length) {
    out.reason = `a level went to or from 100: ${crossing.map(([s, [a, b]]) => `${s} ${a} -> ${b}`).join(', ')}`;
  } else {
    out.level_only = true;
    out.same_teams = await sameTeams(oldDir, newDir, d.changes, opts);
    out.patchable = out.same_teams.differing === 0 && out.same_teams.teams === opts.teams;
    out.reason = out.patchable
      ? `only levels changed, and ${opts.teams.toLocaleString('en-US')} teams are the same at both commits`
      : `only levels changed, but ${out.same_teams.differing} of ${out.same_teams.teams} teams differ (${out.same_teams.first_difference})`;
  }
  return out;
};

const describe = levels =>
  Object.entries(levels).map(([name, [a, b]]) => `${name} ${a} -> ${b}`).join(', ');

// Carries the new levels into the published files and records the patch in the manifest.
const apply = (result, ps) => {
  const manifest = readManifest();
  if (manifest.fingerprint !== result.old_fingerprint) {
    throw new Error('the published files do not describe the old checkout');
  }
  const levels = {};
  for (const name of SETS) {
    const file = path.join(DATA, `${name}.json`);
    const data = JSON.parse(fs.readFileSync(file, 'utf8'));
    for (const [species, s] of Object.entries(data)) {
      const change = result.changes[toID(species)];
      if (!change) continue;
      if (s.level !== change[0]) throw new Error(`${name}.json: ${species} is at level ${s.level}, not ${change[0]}`);
      s.level = change[1];
      levels[species] = change;
    }
    write(file, data);
  }
  const { format, ...rest } = manifest;
  writeManifest(DATA, {
    format,
    ...rest,
    fingerprint: result.new_fingerprint,
    patches: [...(manifest.patches || []), { ps, levels }],
  });
  return { levels, summary: `levels patched: ${describe(levels)}` };
};

module.exports = { base, compare, apply, describe };

if (require.main === module) {
  const args = process.argv.slice(2);
  const opts = { teams: 100000, start: 0, workers: require('os').cpus().length, every: 60 };
  let oldDir;
  let newDir;
  let printBase = false;
  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--base': printBase = true; break;
      case '--old': oldDir = path.resolve(args[++i]); break;
      case '--new': newDir = path.resolve(args[++i]); break;
      case '--teams': opts.teams = Number(args[++i]); break;
      case '--start': opts.start = Number(args[++i]); break;
      case '--workers': opts.workers = Number(args[++i]); break;
      case '--every': opts.every = Number(args[++i]); break;
      default:
        console.error(`unknown argument: ${args[i]}`);
        process.exit(2);
    }
  }
  (async () => {
    if (printBase) return console.log(base(readManifest()));
    if (!oldDir || !newDir) throw new Error('--old and --new name two built PS checkouts');
    console.log(JSON.stringify(await compare(oldDir, newDir, opts)));
  })().catch(err => {
    console.error(err.message);
    process.exit(1);
  });
}
