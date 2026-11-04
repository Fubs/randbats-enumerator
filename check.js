#!/usr/bin/env node
'use strict';

// Whether the published data still describes the generator in the Pokémon Showdown checkout at
// vendor/pokemon-showdown (built: `node build` there).
//
//   node check.js          the comparison, as JSON
//   node check.js --mark   the verdict, as JSON, recorded in the published manifest when the
//                          status changes; PARITY=pass|fail|error is the parity check's result
//
// The data is fresh while the manifest's `fingerprint` (what the generator reads) and `logic`
// (the PS sources the port translates) match the checkout. When only `logic` differs, it stays
// fresh if PARITY=pass. A parity check that could not run leaves the recorded status as it was.

const fs = require('fs');
const path = require('path');
const stringify = require('json-stringify-pretty-compact');

const FORMAT = 'gen9randombattle';
const DATA = path.join(__dirname, 'data', FORMAT);
const PS = path.join(__dirname, 'vendor', 'pokemon-showdown');
const SETS = [
  'lead', 'slot-2', 'slot-3', 'slot-4', 'slot-5', 'slot-6',
  'first-2-slots', 'first-3-slots', 'first-4-slots', 'first-5-slots', 'first-6-slots',
  'non-lead',
];
const FILES = [...SETS, 'slot-probabilities'];

const STALE = {
  inputs: 'PS has changed what the generator reads since these files were computed',
  port: 'PS has changed the generator, and the port no longer rebuilds its teams',
};

// The data at pkmn's width for stats files, the manifest at the default.
const write = (file, data, maxLength = 10000) => fs.writeFileSync(file, stringify(data, { maxLength }));
const writeManifest = (dir, manifest) => write(path.join(dir, 'manifest.json'), manifest, 80);

// The published files against the checkout, before any parity check.
const compare = (dir = DATA) => {
  const problem = (reason, extra) => ({ readable: false, problem: reason, ...extra });
  let manifest;
  try {
    manifest = JSON.parse(fs.readFileSync(path.join(dir, 'manifest.json'), 'utf8'));
  } catch (err) {
    return problem(`cannot read ${path.relative(__dirname, dir)}/manifest.json: ${err.code || err.message}`);
  }
  if (manifest.format !== FORMAT) return problem(`the manifest is for '${manifest.format}'`);
  for (const name of FILES) {
    const file = path.join(dir, `${name}.json`);
    try {
      JSON.parse(fs.readFileSync(file, 'utf8'));
    } catch (err) {
      return problem(`cannot read ${path.relative(__dirname, file)}: ${err.code || err.message}`);
    }
  }
  const current = require('./teamgen/dump.js').fingerprint(PS);
  const fingerprint_match = manifest.fingerprint === current.fingerprint;
  const logic_match = manifest.logic === current.logic;
  return {
    readable: true,
    ps: current.ps,
    manifest_ps: manifest.ps,
    fresh: manifest.fresh,
    fingerprint_match,
    logic_match,
    needs_parity: !fingerprint_match || !logic_match,
    fingerprint: current.fingerprint,
    logic: current.logic,
    manifest_fingerprint: manifest.fingerprint,
    manifest_logic: manifest.logic,
    manifest,
  };
};

// `fresh` is what the manifest should say: true, false, or null to leave it as it is.
const verdict = (c, parity) => {
  if (!c.readable) return { kind: 'unreadable', fresh: null, reason: c.problem };
  const change = (a, b) => `${String(a).slice(0, 12)} -> ${String(b).slice(0, 12)}`;
  if (parity === 'fail') {
    return { kind: 'port', fresh: false, stale: 'port', reason: 'the port failed parity against this PS' };
  }
  if (!c.fingerprint_match) {
    const reason = `PS changed the generator's inputs (fingerprint ${change(c.manifest_fingerprint, c.fingerprint)})`;
    if (parity === 'pass') return { kind: 'recompute', fresh: false, stale: 'inputs', reason };
    return { kind: 'recompute-unchecked', fresh: false, stale: 'inputs', reason };
  }
  if (!c.logic_match) {
    const logic = `the generator's sources changed (logic ${change(c.manifest_logic, c.logic)})`;
    if (parity === 'pass') return { kind: 'fresh', fresh: true, reason: `${logic}; the port still passes parity` };
    return { kind: 'parity-error', fresh: null, reason: `${logic}, and parity could not run` };
  }
  return { kind: 'fresh', fresh: true, reason: 'fingerprint and logic match the manifest' };
};

// The manifest with the verdict recorded, or null when it already says so.
const record = (manifest, v, ps) => {
  if (v.fresh === null) return null;
  const { format, fresh, stale, ...rest } = manifest;
  if (v.fresh) {
    if (fresh === true && !stale) return null;
    return { format, fresh: true, ...rest };
  }
  const reason = STALE[v.stale];
  if (fresh === false && stale?.reason === reason) return null;
  const since = fresh === false && stale?.ps || ps;
  return { format, fresh: false, stale: { ps: since, reason }, ...rest };
};

const ISSUE = {
  recompute: [
    'Recompute needed',
    'PS changed what the generator reads, and the port still rebuilds its teams draw for draw. ' +
      'Run `teamgen/run.sh` at this PS commit, then `node publish.js`, and push the files.',
  ],
  'recompute-unchecked': [
    'Recompute needed; parity could not run',
    'PS changed what the generator reads. The parity check could not run (a tooling failure: see ' +
      'the run), so whether the port still matches PS is unknown. `teamgen/run.sh` checks parity ' +
      'before it computes anything.',
  ],
  port: [
    'Port needs updating',
    ({ teams, start }) => 'The port no longer rebuilds the teams PS generates. In `teamgen/`, ' +
      `\`node trace.js --teams ${teams} --start ${start} --sites | target/release/parity\` ` +
      'replays the teams of this check and tallies the failures by where they diverged. Update ' +
      'the port until parity passes, then recompute.',
  ],
  'parity-error': [
    'Parity could not run',
    "The generator's sources changed and the parity check could not run (a tooling failure, not " +
      'a port failure), so whether the data still holds is unknown. The published status is ' +
      'unchanged. Rerun the workflow, and fix the build if it keeps failing.',
  ],
  unreadable: [
    'The published files cannot be read',
    'Republish them with `node publish.js`.',
  ],
};

// The parity check's teams, as the workflow sets them.
const parityTeams = () => ({
  teams: Number(process.env.PARITY_TEAMS) || 100000,
  start: Number(process.env.PARITY_START) || 0,
});

const issueBody = (c, v, parity, levels) => {
  const [title, todo] = ISSUE[v.kind];
  const ps = commit => commit ? `smogon/pokemon-showdown@${commit}` : 'unknown';
  const teams = parityTeams();
  let checked = { pass: 'passed', fail: 'failed', error: 'could not run' }[parity] || 'not run';
  if (parity === 'pass' || parity === 'fail') {
    checked += ` (${teams.teams.toLocaleString('en-US')} teams from seed index ${teams.start})`;
  }
  const lines = [
    `**${title}.** ${typeof todo === 'function' ? todo(teams) : todo}`,
    '',
    `- PS: ${ps(c.ps)}`,
    `- Data computed at: ${ps(c.manifest_ps)}` +
      (c.manifest?.patches?.length ? `, levels patched at ${ps(c.manifest.patches.at(-1).ps)}` : ''),
    `- Reason: ${v.reason}`,
    `- Parity: ${checked}`,
  ];
  if (levels) lines.push(`- Level-only patch: no, ${levels.reason}`);
  const { GITHUB_SERVER_URL: server, GITHUB_REPOSITORY: repo, GITHUB_RUN_ID: run } = process.env;
  if (server && repo && run) lines.push(`- Run: ${server}/${repo}/actions/runs/${run}`);
  lines.push('', '_Kept up to date by the Check workflow, which closes it once the data is fresh again._');
  return lines.join('\n');
};

// levels.js's comparison, when the workflow made one (LEVELS names its file).
const readLevels = () => {
  try {
    return process.env.LEVELS ? JSON.parse(fs.readFileSync(process.env.LEVELS, 'utf8')) : null;
  } catch {
    return null;
  }
};

const mark = parity => {
  let c = compare();
  const levels = readLevels();
  // A level-only change is patched in when it was measured against the inputs the files describe,
  // reaches the inputs at hand, and the port needs no parity check or passed it.
  let patch = null;
  if (c.readable && !c.fingerprint_match && levels?.patchable &&
      levels.old_fingerprint === c.manifest_fingerprint && levels.new_fingerprint === c.fingerprint &&
      (c.logic_match || parity === 'pass')) {
    patch = require('./levels.js').apply(levels, c.ps);
    c = compare();
  }
  const v = verdict(c, parity);
  if (patch) v.reason = `${patch.summary}; ${v.reason}`;
  const updated = c.readable ? record(c.manifest, v, c.ps) : null;
  if (updated) writeManifest(DATA, updated);
  const short = c.ps ? c.ps.slice(0, 8) : 'unknown';
  const at = `smogon/pokemon-showdown@${short}`;
  let closing = null;
  if (v.kind === 'fresh') {
    closing = `Fresh at ${c.ps ? `smogon/pokemon-showdown@${c.ps}` : 'this PS'}.`;
    if (patch) closing += ` ${patch.summary[0].toUpperCase()}${patch.summary.slice(1)}.`;
  }
  return {
    ...v,
    ps: c.ps,
    manifest_ps: c.manifest_ps,
    changed: !!(updated || patch),
    commit_message: patch ? `${FORMAT}: levels patched at ${at}`
      : updated && `${FORMAT}: ${updated.fresh ? 'fresh' : 'stale'} at ${at}`,
    closing_comment: closing,
    issue_body: v.kind === 'fresh' ? null : issueBody(c, v, parity, levels && !patch ? levels : null),
  };
};

module.exports = { FORMAT, DATA, PS, SETS, FILES, compare, write, writeManifest };

if (require.main === module) {
  try {
    if (process.argv.includes('--mark')) {
      console.log(JSON.stringify(mark(process.env.PARITY)));
    } else {
      const { manifest, ...c } = compare();
      console.log(JSON.stringify(c));
    }
  } catch (err) {
    console.error(err.message);
    process.exit(1);
  }
}
