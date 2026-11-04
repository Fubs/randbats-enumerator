#!/usr/bin/env node
'use strict';

// Copies stdin to stdout and reports on stderr, every --every seconds and at the end, how many
// lines (here, traced teams) have passed, out of --total, with an estimate of the time left.
//
//   node trace.js ... | node .github/progress.js --total N [--every SECONDS] [--label TEXT] | parity ...

const { Transform, pipeline } = require('stream');

let total = 0;
let every = 180;
let label = 'progress';
const args = process.argv.slice(2);
for (let i = 0; i < args.length; i++) {
  switch (args[i]) {
    case '--total': total = Number(args[++i]); break;
    case '--every': every = Number(args[++i]); break;
    case '--label': label = args[++i]; break;
    default:
      console.error(`unknown argument: ${args[i]}`);
      process.exit(2);
  }
}

const start = Date.now();
let lines = 0;
const duration = s => {
  s = Math.round(s);
  return s >= 60 ? `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, '0')}s` : `${s}s`;
};
const report = done => {
  const elapsed = (Date.now() - start) / 1000;
  let line = `${label}: ${lines.toLocaleString('en-US')}`;
  if (total) line += ` of ${total.toLocaleString('en-US')} (${(100 * lines / total).toFixed(1)}%)`;
  line += ` in ${duration(elapsed)}`;
  if (!done && total && lines) line += `, about ${duration(elapsed * (total - lines) / lines)} to go`;
  console.error(line);
};

const timer = setInterval(() => report(false), every * 1000);
const count = new Transform({
  transform(chunk, _, callback) {
    for (let i = chunk.indexOf(10); i !== -1; i = chunk.indexOf(10, i + 1)) lines++;
    callback(null, chunk);
  },
});
pipeline(process.stdin, count, process.stdout, err => {
  clearInterval(timer);
  report(true);
  if (err) {
    console.error(`${label}: ${err.message}`);
    process.exit(1);
  }
});
