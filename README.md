# Exact Gen 9 Random Battles Set Distributions by Party Slot

Each species' set distribution in [Pokémon Showdown](https://github.com/smogon/pokemon-showdown)'s
Gen 9 Random Battles, given the party slot it stands in, computed from the team generator rather
than sampled from it. PS builds a team one member at a time, and a member's set depends on whether
it leads and on what the members before it bring: Komala carries Rapid Spin every time it leads,
but 93% of the time in slot 2. The files have the per-species schema of
[pkmn/randbats](https://github.com/pkmn/randbats)'
[`data/stats/gen9randombattle.json`](https://github.com/pkmn/randbats/blob/main/data/stats/gen9randombattle.json). Slots 1–5 are exact, up to floating-point
rounding. Slot 6 is a Rao–Blackwellized estimate.

## Files

[`data/gen9randombattle/`](data/gen9randombattle/) holds:

| File | A species' sets, given that it stands in |
|---|---|
| `lead.json` | slot 1 |
| `slot-K.json`, K = 2…6 | slot K |
| `first-K-slots.json`, K = 2…6 | one of slots 1…K; `first-6-slots.json` is the whole team |
| `non-lead.json` | any slot but the lead, which is seen at once: the file for a battle's other members |
| `slot-probabilities.json` | not sets: the probability that slot k holds each species, k = 1…6 |
| `manifest.json` | the PS commit and inputs the files describe, any level patches, and whether they are fresh |

A party slot is PS's party order at the start of a battle, the lead and then the other members in
the order they were generated, not the order in which they are revealed. The slot-6 estimate
enters `slot-6.json`, `first-6-slots.json`, `non-lead.json` and the `"6"` column of
`slot-probabilities.json`; backtested on slot 5, the same estimator came within 5.6·10⁻⁵ of the
exact values.

## Freshness

Every [hour](.github/workflows/check.yml), PS's master branch is built and compared with the
manifest by [`check.js`](check.js). The files stay fresh while the generator's inputs are
unchanged, and when only its code changes, as long as 300,000 PS teams still rebuild draw for draw
through the port. A change of levels alone, none to or from 100, is patched into the files by
[`levels.js`](levels.js) once PS builds the same 100,000 teams at both commits apart from those
levels. Any other change marks the files stale in their manifest until they are recomputed, and an
issue stays open while something needs attention.

## Recomputing

The computation is [`teamgen/`](teamgen/): a Rust port of PS's team generator, held to it draw for
draw, and a dynamic program over its team loop, described in [`teamgen/README.md`](teamgen/README.md)
and [`teamgen/MATH.md`](teamgen/MATH.md). It takes about 90 minutes on a Ryzen 9 9950X3D with an
RTX 4090, so it runs outside CI:

```bash
git clone https://github.com/smogon/pokemon-showdown vendor/pokemon-showdown
(cd vendor/pokemon-showdown && npm install && node build)
teamgen/run.sh
npm install --no-save json-stringify-pretty-compact@3.0.0
node publish.js
```

Run at the manifest's `ps`, this reproduces the files as computed; the manifest's `patches` list
the levels changed since.
