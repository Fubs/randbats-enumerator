#!/usr/bin/env node
'use strict';

// Records every PRNG draw Pokémon Showdown's gen9randombattle `randomTeam`
// makes, and the team it builds: one JSON line per team,
// `{"seed", "draws": [[from, to, value], ...], "team": [...]}`. The `parity`
// binary replays the draws through the port, which must build the same team.
// From inside teamgen/, where the binaries are built:
//
//   node trace.js --teams N [--start S] [--workers W] [--sites] [--out FILE] [--ps DIR] \
//     | target/release/parity --expect N
//
// Team i is rolled from the seed `sodium,<i as 64 hex digits>`. `--sites`
// adds, per draw, the teams.ts frames that made it.

const fs = require('fs');
const os = require('os');
const path = require('path');
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');

const DEFAULT_PS = path.join(__dirname, '..', 'vendor', 'pokemon-showdown');
const BATCH = 500;

const seedOf = i => 'sodium,' + i.toString(16).padStart(64, '0');

// The random-battles/gen9/teams frames behind a draw, innermost first, as
// `line:function`.
function siteOf() {
  const frames = (new Error().stack || '').split('\n').filter(f => f.includes('random-battles/gen9/teams'));
  return frames.slice(0, 3).map(f => {
    const m = f.match(/at (?:[\w.]+\.)?(\w+) .*teams\.[jt]s:(\d+)/);
    return m ? `${m[2]}:${m[1]}` : f.trim();
  }).join('<');
}

function runWorker(job) {
  // Sites as teams.ts lines, through PS's source maps.
  if (job.sites) process.setSourceMapsEnabled(true);
  const { Teams } = require(path.join(job.psPath, 'dist', 'sim'));
  let lines = [];
  for (let i = job.start; i < job.start + job.count; i++) {
    const seed = seedOf(i);
    const gen = Teams.getGenerator('gen9randombattle', seed);
    const prng = gen.prng;
    const random = prng.random.bind(prng);
    const draws = [];
    const sites = [];
    prng.random = (from, to) => {
      if (from === undefined) throw new Error(`seed ${seed}: a float draw, which the port has no counterpart for`);
      const value = random(from, to);
      draws.push([from, to === undefined ? null : to, value]);
      if (job.sites) sites.push(siteOf());
      return value;
    };
    const team = gen.getTeam().map(s => ({
      speciesId: s.speciesId,
      gender: s.gender,
      shiny: s.shiny,
      level: s.level,
      moves: s.moves,
      ability: s.ability,
      evs: s.evs,
      ivs: s.ivs,
      item: s.item,
      teraType: s.teraType,
      role: s.role,
    }));
    lines.push(JSON.stringify(job.sites ? { seed, draws, team, sites } : { seed, draws, team }));
    if (lines.length >= BATCH) {
      parentPort.postMessage(lines.join('\n') + '\n');
      lines = [];
    }
  }
  if (lines.length) parentPort.postMessage(lines.join('\n') + '\n');
}

async function main() {
  let teams = 300;
  let start = 0;
  let workers = Math.min(os.cpus().length, 8);
  let out = '';
  let sites = false;
  let psPath = DEFAULT_PS;
  const args = process.argv.slice(2);
  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--teams': teams = parseInt(args[++i], 10); break;
      case '--start': start = parseInt(args[++i], 10); break;
      case '--workers': workers = parseInt(args[++i], 10); break;
      case '--out': out = args[++i]; break;
      case '--sites': sites = true; break;
      case '--ps': psPath = args[++i]; break;
      default:
        console.error(`unknown argument: ${args[i]}`);
        process.exit(2);
    }
  }
  psPath = path.resolve(psPath);
  if (!fs.existsSync(path.join(psPath, 'dist', 'sim', 'index.js'))) {
    throw new Error(`${psPath}/dist/sim not found: build Pokémon Showdown first (\`node build\` there)`);
  }
  if (!(teams >= 0) || !(workers >= 1) || !(start >= 0)) throw new Error('--teams, --start and --workers take counts');
  const sink = out ? fs.createWriteStream(out) : process.stdout;
  // A reader that goes away (`| head`) ends the run, not an error.
  sink.on('error', err => {
    if (err.code === 'EPIPE') process.exit(0);
    throw err;
  });
  const per = Math.ceil(teams / workers);
  const runs = [];
  for (let w = 0; w < workers; w++) {
    const job = { psPath, start: start + w * per, count: Math.max(0, Math.min(per, teams - w * per)), sites };
    if (job.count === 0) continue;
    runs.push(new Promise((resolve, reject) => {
      const worker = new Worker(__filename, { workerData: job });
      worker.on('message', chunk => sink.write(chunk));
      worker.once('error', reject);
      worker.once('exit', code => (code === 0 ? resolve() : reject(new Error(`worker exited ${code}`))));
    }));
  }
  await Promise.all(runs);
  if (out) await new Promise(resolve => sink.end(resolve));
}

if (isMainThread) {
  main().catch(err => {
    console.error(err.message || err);
    process.exit(1);
  });
} else {
  runWorker(workerData);
}
