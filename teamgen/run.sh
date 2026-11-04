#!/usr/bin/env bash
# The exact Gen 9 Random Battles party-slot data, from the vendored Pokémon
# Showdown into output/gen9randombattle/, the staging directory the
# repository's publish.js renders into data/gen9randombattle/.
#
#   teamgen/run.sh
#
# 1. `node dump.js`: the PS data the generator reads, and its fingerprint.
# 2. `cargo build --release --features gpu`.
# 3. Parity: PARITY_TEAMS of PS's teams, traced, must each rebuild draw for
#    draw (about 40 seconds for 10⁶).
# 4. exact_team_weights --slots --slot5: slots 1–5 exactly, held against
#    CHECK_MC sampled teams (about 90 minutes on 32 threads and a GPU).
# 5. Slot 6, estimated: CHUNKS dumps of TEAMS teams each (about 20 minutes).
#    With SLOT6=1, step 4 computes slot 6 exactly as well (`--slot6`, days
#    rather than minutes, resumable from its checkpoint) and this step is
#    skipped.
# 6. slots: the files, and manifest.json with dump.js's ps, fingerprint and
#    logic in it, and `code`, a hash of this crate's sources.
#
# The weights and the dumps go to work/<fingerprint>-<logic>-<code>/,
# gitignored, and are reused from there when a run is repeated on the same
# inputs and the same code.
#
# Environment:
#   GPU=0                      build and run without CUDA (slot 5 then takes hours)
#   PARITY_TEAMS=1000000       the parity check's teams
#   WORKERS=N                  its trace workers (default: every CPU)
#   THREADS=N                  exact_team_weights' threads (default: every CPU;
#                              with the GPU at most 85, the default clamped to it)
#   CHECK_MC=40000000          sampled teams the exact slots are checked against
#   SLOT6=1                    slot 6 exactly, not estimated
#   CHUNKS=2 TEAMS=200000000   the slot-6 estimate's dumps: 4·10⁸ teams
#   RB_THREADS=30 SEED0=300    the dumps' threads and seeds (seeds 301, 302): a
#                              dump's streams are per thread, so its thread
#                              count is part of its seed
#   GENERATED=TIMESTAMP        manifest.json's ISO 8601 time (default now, UTC)
#   OUT=DIR                    where the files go (default output/gen9randombattle)
#   LIMIT_GROUPS=N             testing only: a sample of the states, so the
#                              weights are NOT exact
set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-1}"
CHECK_MC="${CHECK_MC:-40000000}"
CHUNKS="${CHUNKS:-2}"
TEAMS="${TEAMS:-200000000}"
RB_THREADS="${RB_THREADS:-30}"
SEED0="${SEED0:-300}"
OUT="${OUT:-output/gen9randombattle}"
if [ -n "${LIMIT_GROUPS:-}" ] && [ "$(realpath -m "$OUT")" = "$(realpath -m output/gen9randombattle)" ]; then
    echo "LIMIT_GROUPS weights are not exact: set OUT to somewhere other than the staging directory" >&2
    exit 1
fi

PS=../vendor/pokemon-showdown
if [ ! -f "$PS/dist/sim/index.js" ]; then
    echo "build Pokémon Showdown first: (cd vendor/pokemon-showdown && npm install && node build)" >&2
    exit 1
fi

node dump.js
FEATURES=()
[ "$GPU" != 0 ] && FEATURES=(--features gpu)
cargo build --release "${FEATURES[@]}" --bin exact_team_weights --bin slots --bin parity
BIN=target/release/exact_team_weights
SLOTS=target/release/slots

PARITY_TEAMS="${PARITY_TEAMS:-1000000}"
WORKERS="${WORKERS:-$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)}"
echo "parity: $PARITY_TEAMS teams"
if ! node trace.js --teams "$PARITY_TEAMS" --workers "$WORKERS" | target/release/parity --expect "$PARITY_TEAMS"; then
    echo "the parity check failed (above): weights from this port would not be PS's" >&2
    exit 1
fi

# The sources that decide the weights, Cargo.lock included (the build above
# writes it when it is missing).
CODE="$(node - <<'EOF'
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const files = [];
const walk = p => fs.statSync(p).isDirectory() ? fs.readdirSync(p).forEach(f => walk(path.join(p, f))) : files.push(p);
['src', 'gen9', 'cuda', 'build.rs', 'Cargo.toml', 'Cargo.lock', 'dump.js', 'trace.js'].forEach(walk);
const hash = crypto.createHash('sha256');
for (const f of files.sort()) hash.update(f + '\0').update(fs.readFileSync(f)).update('\0');
console.log(hash.digest('hex'));
EOF
)"
META="$(node -e 'const m = JSON.parse(process.argv[1]); m.code = process.argv[2]; console.log(JSON.stringify(m))' \
    "$(node dump.js --fingerprint)" "$CODE")"
KEY="$(node -e 'const m = JSON.parse(process.argv[1]); console.log([m.fingerprint, m.logic, m.code].map(h => h.slice(0, 12)).join("-"))' "$META")"
WORK="work/$KEY${LIMIT_GROUPS:+/limit-$LIMIT_GROUPS}"
mkdir -p "$WORK/rb"
printf '%s\n' "$META" > "$WORK/meta.json"
echo "inputs: $META"
SLOT6="${SLOT6:-0}"
if [ "$SLOT6" != 0 ]; then
    WEIGHTS="$WORK/slot-weights-1-6.json"
else
    WEIGHTS="$WORK/slot-weights.json"
fi

WEIGHT_ARGS=(--slots --slot5 --check-mc "$CHECK_MC" --out "$WEIGHTS.tmp")
[ "$SLOT6" != 0 ] && WEIGHT_ARGS+=(--slot6)
if [ "$GPU" != 0 ]; then
    WEIGHT_ARGS+=(--gpu)
    # The GPU's 256 in-flight slots, (TG_SIX_DEPTH + 1) a thread: 85 threads
    # at the default depth of 2.
    MAX_THREADS=$((256 / (${TG_SIX_DEPTH:-2} + 1)))
    if [ -z "${THREADS:-}" ]; then
        THREADS="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)"
        if [ "$THREADS" -gt "$MAX_THREADS" ]; then
            echo "THREADS: $THREADS CPUs, clamped to the GPU's $MAX_THREADS"
            THREADS="$MAX_THREADS"
        fi
    elif [ "$THREADS" -gt "$MAX_THREADS" ]; then
        echo "THREADS=$THREADS: with the GPU, threads × $((${TG_SIX_DEPTH:-2} + 1)) ≤ 256 (its in-flight slots), so at most $MAX_THREADS" >&2
        exit 1
    fi
fi
[ -n "${THREADS:-}" ] && WEIGHT_ARGS+=(--threads "$THREADS")
[ -n "${LIMIT_GROUPS:-}" ] && WEIGHT_ARGS+=(--limit-groups "$LIMIT_GROUPS")

if [ -s "$WEIGHTS" ]; then
    echo "reusing $WEIGHTS"
else
    "$BIN" "${WEIGHT_ARGS[@]}" 2>&1 | tee "$WORK/slot-weights.log"
    # The sampled check's Bonferroni p over every cell: a correct table sits
    # above ~0.01, and below 0.001 the weights are not published.
    p="$(grep -oE 'Bonferroni [0-9.eE+-]+' "$WORK/slot-weights.log" | tail -n 1 | cut -d' ' -f2 || true)"
    if [ -z "$p" ]; then
        echo "no sampled check in $WORK/slot-weights.log" >&2
        exit 1
    fi
    if awk -v p="$p" 'BEGIN { exit !(p < 0.001) }'; then
        if [ -z "${LIMIT_GROUPS:-}" ]; then
            echo "the sampled check failed (Bonferroni p $p): the weights are wrong" >&2
            exit 1
        fi
        echo "sampled check: Bonferroni p $p, as expected of LIMIT_GROUPS weights"
    fi
    mv "$WEIGHTS.tmp" "$WEIGHTS"
fi

ARGS=(--weights "$WEIGHTS" --out "$OUT" --meta "$WORK/meta.json")
[ -n "${GENERATED:-}" ] && ARGS+=(--generated "$GENERATED")

if [ "$SLOT6" = 0 ]; then
    DUMPS=()
    for i in $(seq 1 "$CHUNKS"); do
        seed=$((SEED0 + i))
        dump="$WORK/rb/k6_seed$seed.json"
        if [ -s "$dump" ]; then
            echo "reusing $dump"
        else
            "$BIN" --self-check 0 --threads "$RB_THREADS" --mc-dump "$TEAMS" --mc-rb 6 --mc-seed "$seed" --out "$dump.tmp" \
                > "$WORK/rb/k6_seed$seed.log" 2>&1
            grep mc-dump "$WORK/rb/k6_seed$seed.log"
            mv "$dump.tmp" "$dump"
        fi
        DUMPS+=("$dump")
    done
    ARGS+=(--rb "${DUMPS[@]}" --rb-threads "$RB_THREADS")
fi
"$SLOTS" "${ARGS[@]}"
