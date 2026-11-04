# Exact Party-Slot Set Distributions for Gen 9 Random Battles

## Abstract

Pokémon Showdown (PS) builds a Gen 9 Random Battle team one member at a time, and each member's
set depends on whether it leads and on what the members generated before it provide. A species'
set distribution therefore depends on its place in the team. This crate computes, without
sampling, the probability that each species stands in each party slot under each context of team
flags, and combines these weights with exactly enumerated set tables into per-slot set statistics
in the schema of pkmn/randbats [2]. The computation is a dynamic program over PS's team loop, run
on a Rust port of the generator that is held to PS draw for draw. Slots 1–5 are exact up to
floating-point rounding. Slot 6, whose exact pass is implemented but takes about 50 hours, is a
Rao–Blackwellized sampling estimate; backtested on slot 5, the same estimator came within
5.6·10⁻⁵ of exact. The results are staged in `output/gen9randombattle/` with a manifest naming
the PS inputs they were computed from; the repository publishes them, and checks every hour that
PS has not changed those inputs since.

## 1 Introduction

PS's `randomTeam` [1] (`data/random-battles/gen9/teams.ts`) draws base species without
replacement from a weighted pool, picks one of the base's formes uniformly, and rejects the draw
if the team cannot take it: the species clause, the type and weakness limits, at most one
level-100 member, incompatible pairs, and a few position rules. Each accepted species rolls its
set with `randomSet`, which is told whether the species leads and what earlier members provide
(`teamDetails`: weather, hazards, hazard removal, screens, status cure, Tera Blast and the Spikes
count). A support set whose hazard removal is already covered drops it, and some options exist
only for leads, so a set depends on when it was generated:

| Probability, given the species is in party slot k | 1 (lead) | 2 | 3 | 4 | 5 | 6 (estimate) |
|---|---|---|---|---|---|---|
| Komala carries Rapid Spin | 1 | 0.934 | 0.874 | 0.816 | 0.762 | 0.712 |
| Weezing-Galar carries Defog | 1 | 0.933 | 0.871 | 0.812 | 0.755 | 0.703 |
| Kricketune holds Focus Sash | 1 | 0 | 0 | 0 | 0 | 0 |

Pooled over the whole team (the slot-6 estimate included), Komala carries Rapid Spin with
probability 0.844, a value that describes no single slot. The sampled statistics [2] tally 10⁵
teams per format and pool all slots; there, Komala's figure rests on about 1,500 sets, a standard
error near 0.01. Sampled per-slot tables would be noisier still, and worst for the rarest sets.
(These values are at the PS commit of 2026-10-18.)

"Exact" means that no sampling enters: every probability is the generator's own, summed over
every way the team loop can run, as a fair coin's 1/2 is read off the coin rather than estimated
from flips. The sums are evaluated in double precision, with GPU stages of about 48 bits, and
independent code paths agree to about 10⁻¹² relative.

## 2 Party slots and the staged files

A *party slot* is a member's place in PS's party order at the start of a battle: `randomTeam`
moves the lead to the front and keeps the rest in generation order. Slot 1 is the lead, the first
accepted species that may lead; eight never do (Dugtrio, Gothitelle, Iron Thorns, Kingambit, and
Zacian and Zamazenta in both formes). Slot k ≥ 2 is the (k−1)-th non-lead member generated. A
party slot is not the order in which members are revealed in battle.

`output/gen9randombattle/` holds fourteen files:

| File | A species' sets, given that it stands in |
|---|---|
| `lead.json` | slot 1 |
| `slot-K.json`, K = 2…6 | slot K |
| `first-K-slots.json`, K = 2…6 | one of slots 1…K; `first-6-slots.json` is the whole team |
| `non-lead.json` | one of slots 2…6 |
| `slot-probabilities.json` | not sets: `{species: {k: P(slot k holds it)}}`, k = 1…6, omitting slots a species never takes (the never-leads have no `"1"`, Zoroark and Zoroark-Hisui no `"6"`); the published file lists those as 0 |
| `manifest.json` | what the files were computed from |

The set files have the per-species shape of pkmn/randbats' `data/stats/gen9randombattle.json` [2]
and hold what its sampler converges to, at full double precision ([MATH.md](MATH.md) §8). The
lead file omits the never-leads, and the slot-6 file Zoroark and Zoroark-Hisui, which never stand
there. A species is on a team at most once, so for any set of slots S,

  P(set | s, in one of S) = Σ_{k∈S} p_k(s) · P(set | s, in k) / Σ_{k∈S} p_k(s),

with p_k(s) from `slot-probabilities.json`; the pooled files are these mixtures, taken over the
exact set tables. The lead is seen as soon as a battle starts, so every other member is known to
stand in one of slots 2–6: `non-lead.json` is the file for those. Ariados holds Focus Sash only
as the lead, 17% of the time in `first-6-slots.json` and never in `non-lead.json`.

`manifest.json` records:

| Field | Meaning |
|---|---|
| `format` | `"gen9randombattle"` |
| `exact_slots` | the slots computed exactly, `[1, 2, 3, 4, 5]` |
| `estimated_slots` | the slots estimated, `[6]` |
| `slot6_estimate` | the estimate's `method`, its `teams`, and its dumps' `seeds` and `threads` (a dump's random streams are per thread, so its thread count is part of its seed) |
| `generated` | when the files were rendered, as an ISO 8601 UTC timestamp |
| `ps` | the PS commit the inputs were read from |
| `fingerprint` | SHA-256 of a canonical dump of what the generator reads from PS: species, moves, items, abilities, the type chart, the format's rules and `sets.json`, with the few fields PS's generator reads that no set makes the port read |
| `logic` | SHA-256 of the PS sources the port translates: `data/random-battles/gen9/teams.ts`, `sim/prng.ts` and `sim/dex.ts` |
| `code` | SHA-256 of this crate's sources (`src/`, `gen9/`, `cuda/`, `build.rs`, `Cargo.toml`, `Cargo.lock`, `dump.js`, `trace.js`), by path and content; a record only, which the repository's check does not compare (a comment-only change moves it) |

`ps`, `fingerprint` and `logic` are what `node dump.js --fingerprint` prints, and `run.sh` adds
`code`. A `fingerprint` change means the inputs changed and the weights must be recomputed. A
`logic` change alone means the port must be checked against PS again.

## 3 Publication

`publish.js`, in the repository root, renders the staged files into `data/gen9randombattle/`,
the published files, and refuses staged files whose `fingerprint` or `logic` differs from the
vendored PS. The repository's hourly check (`check.js`) then compares them with PS's master: they
stay *fresh* while `fingerprint` and `logic` match, or while only `logic` differs and the port
still rebuilds 3·10⁵ PS teams draw for draw (§6). Otherwise they are marked stale in their
manifest, and one issue stays open, saying *recompute needed* if the port still agrees with PS
and *port needs updating* if it does not. The repository's README describes the check.

## 4 Method

Lemmas and sections cited here are those of [MATH.md](MATH.md). The weights are
W_k(s, c) = P(s stands in slot k, rolled under context c), each slot summing to 1. A file
conditions on an event E, "in slot k" or "in one of slots S", and gives each species the mixture

  P(set | s, E) ∝ Σ_{k∈E} Σ_c W_k(s, c) · K_{s,[k=1]}(set | c),

where K_{s,ℓ}(· | c) is the species' exact set table as lead (ℓ = 1) or not, under context c.

- **Labels in advance.** A forme label is fixed when its pool entry is drawn, and the team only
  grows, so a rejection carries no information (Lemma 1). The process is therefore the one whose
  labels are all drawn in advance, and the next acceptance is uniform over the entries whose label
  the team admits (Corollary 3). The only hidden state is which labels are dead (Lemma 6).
- **Classes and flags.** Single-forme species without special rules are interchangeable when they
  share a typing and level-100 status (Lemma 4). Acceptance reads sets only through the Tera Blast
  and Dry Skin/Fluffy rules, so the context rides beside each state as a distribution (Lemma 5).
- **Slots 1–4.** Teams through size 2 are held with their dead labels (8.75·10⁶ states), and slot 3
  is the step out of them. Slot 4 integrates the label splits instead of enumerating them, in
  closed form: with f entries certain to stay admissible and A_b ~ Bin(r_b, q_b) survivors of each
  partly admissible base b, E[1/M] = ∫₀¹ x^{f−1} ∏_b (1 − q_b + q_b x)^{r_b} dx (MATH.md §5).
- **Slot 5.** Steps 3 and 4 are integrated jointly in (x, y), exactly by Gauss–Jacobi quadrature
  [6], on the GPU in pairs of single-precision floats (about 48 bits) [7] (MATH.md §6).
- **Slot 6.** The exact pass integrates steps 3–5 in (x, y, z) (MATH.md §7). It is implemented and
  checked but not yet run in full: about 50 hours on the hardware below. Until it runs, slot 6 is
  estimated. Teams are sampled through the port to five members, and the sixth pick is not drawn;
  instead every remaining pool entry's label is drawn, and the pick's exact distribution given the
  five and those labels is added. This Rao–Blackwellization [3, 4, 5] gives every species the
  sixth pick could be an expected count from each team, where a plainly sampled team gives one
  species one count (MATH.md §7, "The slot-6 estimate"). The estimate uses 4·10⁸ teams.

## 5 Recomputing

The data goes stale when PS changes what the generator reads, in practice the monthly Random
Battles balance patch and the occasional set or `teams.ts` change. The recompute is too heavy for
CI, so it runs on a machine with a GPU. From the repository root, at the PS commit to compute
from:

```bash
git clone https://github.com/smogon/pokemon-showdown vendor/pokemon-showdown
(cd vendor/pokemon-showdown && npm install && node build)
teamgen/run.sh
node publish.js
```

**Prerequisites.** Node.js, at the version the vendored PS requires (22.18 or later); stable Rust
1.88 or later; and, for the GPU path, the CUDA toolkit with nvcc and a host C++ compiler it
accepts. `CUDA_PATH` or `CUDA_HOME` names the toolkit, by default the one whose nvcc is on `PATH`.
nvcc builds for the machine's GPU, or for `TEAMGEN_CUDA_ARCH=sm_NN`. `.cargo/config.toml` builds
for the host CPU (`target-cpu=native`), since the weights' arithmetic is fused multiply-adds and
slot 6 runs about 9× slower without them.

**What `run.sh` does.**
1. `node dump.js` writes `target/dump.json`, the PS data the port reads, from which
   `gen9/build.rs` generates the port's tables. `node dump.js --fingerprint` gives `ps`,
   `fingerprint` and `logic`.
2. `cargo build --release --features gpu` builds `exact_team_weights`, `slots` and `parity`.
3. Parity: `node trace.js --teams 1000000 | target/release/parity --expect 1000000` must rebuild
   every PS team draw for draw (§6), or the run stops. It runs every time, even when the
   intermediates below are reused; about 40 seconds at 16 workers.
4. `exact_team_weights --slots --slot5 --gpu --check-mc 40000000` computes slots 1–5 and holds
   them to 4·10⁷ sampled teams (§6). Weights whose Bonferroni-adjusted p is below 0.001 are not
   published.
5. Slot 6 is estimated from `CHUNKS` (2) runs of
   `exact_team_weights --mc-dump 200000000 --mc-rb 6`, one dump each, at seeds 301 and 302 and 30
   threads. With `SLOT6=1`, step 4 computes slot 6 exactly too and this step is skipped.
6. `slots` renders the fourteen files into `output/gen9randombattle/`, with `ps`, `fingerprint`,
   `logic` and `code` in the manifest.

Intermediates go to `work/<key>/`, the key made of the first 12 hex digits of each of
`fingerprint`, `logic` and `code`. It is ignored by git and reused when a run is repeated on the
same inputs and the same code.

| Variable | Default | Meaning |
|---|---|---|
| `GPU` | `1` | `0` builds and runs without CUDA; slot 5 then takes hours (not timed) |
| `PARITY_TEAMS`, `WORKERS` | `1000000`, every CPU | the parity check's teams and its trace workers |
| `THREADS` | every CPU | `exact_team_weights`' threads for the weights. With the GPU, at most 256 / (`TG_SIX_DEPTH` + 1), 85 by default (its 256 in-flight slots): an unset `THREADS` is clamped to that, and a larger one is refused |
| `CHECK_MC` | `40000000` | sampled teams the exact slots are checked against |
| `SLOT6` | `0` | `1` computes slot 6 exactly instead of estimating it |
| `CHUNKS`, `TEAMS` | `2`, `200000000` | the slot-6 estimate's dumps and teams per dump |
| `RB_THREADS`, `SEED0` | `30`, `300` | the dumps' threads, and their seeds `SEED0 + 1`… |
| `GENERATED` | now (UTC) | the manifest's timestamp |
| `OUT` | `output/gen9randombattle` | where the files go; a relative path is taken from `teamgen/`, where `run.sh` runs |
| `LIMIT_GROUPS` | unset | testing only: a sample of the states, so the weights are **not** exact; refused unless `OUT` is elsewhere |

**Other switches.** `exact_team_weights` also has research and diagnostic switches: flags such as
`--cp-stats`, `--slot6-stats`, `--group-stats`, `--grid`, `--sample-groups` and `--partial-stats`,
and `TG_*` environment variables. `run.sh` needs none of them. Some give wrong weights by design
(`TG_SKIP_KERNEL`, which skips the GPU's arithmetic to time the CPU alone, for one), so leave
them unset for a run whose output is published.

**The exact slot 6.** `SLOT6=1 teamgen/run.sh` adds `--slot6` to step 4, which then computes
slot 6 in 1,024 epochs and checkpoints after slots 1–5 and after each epoch
(`work/<key>/slot-weights-1-6.json.tmp.ckpt`); rerunning the same command resumes after the last
checkpoint. `slots` then renders all six slots as exact, and the manifest says so.

**When the port disagrees.** If the issue says *port needs updating*, PS changed the generator
itself. In this directory, `node dump.js && cargo build --release --bin parity`, then
`node trace.js --teams N --start S --sites | target/release/parity`, with the teams and the first
seed the issue names, replays the check's teams and tallies those the port did not rebuild by
where they diverged, with the `teams.ts` lines behind each draw. The port in
`gen9/src/randbats_gen/` is then changed until every team rebuilds, and the data recomputed.

**Hardware and timings.** An AMD Ryzen 9 9950X3D (16 cores, 32 threads) and an NVIDIA RTX 4090.

| Step | Time |
|---|---|
| Dump and build | under 1 s and 36 s (a clean build, stable, with `gpu`) |
| Parity gate, 10⁶ teams | under 1 min |
| Slots 1–5, with the sampled check | 76 min: slots 1–4 11 min, slot 5 64 min, the check about 1 min |
| Slot-6 estimate, 2 × 2·10⁸ teams | 14 min (421 s and 402 s, about 480,000 teams/s each at 30 threads) |
| Render | 12 s |
| `run.sh`, end to end | 90.5 min, measured before the parity gate was added |
| Exact slot 6 (not yet run in full) | about 50 h, estimated: 1,024 epochs of 172–174 s each (one epoch's share, timed), and about 0.5 h for the rest |
| Parity, 10⁶ teams (`trace.js \| parity`) | 39 s at 16 workers |

## 6 Verification

| Check | Result |
|---|---|
| The port against PS | `node trace.js --teams N \| target/release/parity --expect N` records every random draw PS's `randomTeam` makes and the team it builds, replays the draws through the port, and exits 0 only if every team is rebuilt exactly. 10⁶ teams are. `run.sh` runs it on 10⁶ teams before every recompute, and the repository's hourly check on 3·10⁵, from seeds fixed by the inputs, whenever `logic` or `fingerprint` changes. At the PS of 2026-10-18, PS's own `randomSet`, enumerated over every branch, gave the same 107,802 set tables as the port: 1,017 (species, lead flag) pairs, every species as a non-lead and all but Iron Thorns as a lead, times 106 contexts (no flag, each of 14 single flags with Spikes 1 and 2 counted apart, the 90 pairs of them, and all flags). That PS-side enumerator is not part of this crate |
| Mass | every slot sums to 1 (slot 5 to 10⁻¹⁵) |
| Sampled teams, slots 1–5 | `--check-mc` samples teams through the port's ordinary loop and judges every (species, slot, lead, context) cell by its exact two-sided Poisson p (normal above an expected count of 200), Bonferroni-adjusted over all cells. A correct table sits above about 0.01; `run.sh` refuses weights below 0.001. On 2026-10-18 data: 3.57·10⁶ cells, smallest p 1.65·10⁻⁶, Bonferroni 1.0. The run that made the first published files (2026-11-01, the PS commit in their manifest): 3.57·10⁶ cells, smallest p 2.85·10⁻⁶, Bonferroni 1.0. Most cells expect under one team at 4·10⁷, so the test is sharp only on the larger ones |
| Code paths | slot 4's integrated step against enumeration on 3,001 states, to 2·10⁻¹⁴ relative; slot 5's grouped fold against a per-state one on 3,000 states, to 1.1·10⁻¹²; slot 6's against a per-state fold, to 6·10⁻¹², and its GPU loop against the CPU's, to 1.7·10⁻¹³ |
| The slot-6 estimator on slot 5 | the same estimator at the fifth pick, 2·10⁸ teams, against the exact slot 5, both rendered by `slots`: every value of `slot-5.json` within 5.6·10⁻⁵ (role weights 4.7·10⁻⁵, mean 1.9·10⁻⁶), and `first-5-slots.json` within 1.3·10⁻⁵. At the four decimals of pkmn's sampled files, the estimate is near the rounding step and far below their noise |
| The slot-6 estimate against sampling | 3·10⁹ plainly sampled teams on other seeds: species totals' largest \|z\| 3.17 over 507 species (Bonferroni 0.77); contexts within species χ² 1.023 per degree of freedom, against about 1.015 expected from the estimate's own noise |

The backtest uses the crate's binaries, run from this directory. With slots 1–5 exact in
`WEIGHTS` (`run.sh` leaves them in `work/<key>/slot-weights.json`):

```bash
target/release/exact_team_weights --self-check 0 --threads 30 --mc-dump 200000000 --mc-rb 5 \
    --mc-seed 201 --out k5.json
target/release/slots --weights WEIGHTS --partial --out exact5
target/release/slots --weights WEIGHTS --keep 1,2,3,4 --rb k5.json --partial --out estimate5
```

and compare `exact5/slot-5.json` with `estimate5/slot-5.json`.

## 7 Layout

| Path | Contents |
|---|---|
| `run.sh` | The recompute (§5) |
| `dump.js` | The PS data the port reads, and its `fingerprint` and `logic`; the repository's `check.js` calls its `fingerprint()` |
| `trace.js` | PS's draws and teams, one JSON line a team, for `parity` |
| `gen9/` | Crate `randbats-gen9`: the generator port (`src/randbats_gen/`), the data tables `build.rs` generates from the dump, and the support modules the port reads (types, species, moves, items, abilities, set fields) |
| `src/main.rs` | Binary `exact_team_weights`: slots 1–5, the checks, and the sampler behind the estimate (`--mc-dump N --mc-rb K`) |
| `src/six.rs` | Slot 6, exactly |
| `src/gpu.rs`, `cuda/`, `build.rs` | The `gpu` feature: slot 5's and slot 6's kernels, built by nvcc |
| `src/bin/parity.rs` | Binary `parity`: PS's traces replayed through the port |
| `src/bin/slots/` | Binary `slots`: the staged files, from the weights and the slot-6 dumps |
| `output/gen9randombattle/` | The staged files (§2), which `publish.js` renders; ignored by git |
| `MATH.md` | The mathematics and the measurements |

## 8 Limitations

- **One PS commit.** A change to the PS data the port reads (`fingerprint`) stales the data, and
  the published files are marked stale until it is recomputed. A change to the generator's
  sources alone (`logic`) stales it only if the port then fails parity (§3).
- **Slot 6 is an estimate**, and so are the files that pool it: `slot-6.json`, `first-6-slots.json`,
  `non-lead.json`, and the `"6"` column of `slot-probabilities.json`.
- **Averaged over teammates.** Each file conditions only on the species' own slot. A member's set
  also depends on what the members generated before it brought (a Stealth Rock setter, say), and
  the files average over that.
- **Floating point.** The files are double-precision sums with ~48-bit GPU stages, and the checks
  agree to about 10⁻¹² relative, not to zero.
- **Scope.** Singles `gen9randombattle` teams as PS generates them; no other format or generation.

## References

[1] Pokémon Showdown, `data/random-battles/gen9/teams.ts`.
https://github.com/smogon/pokemon-showdown

[2] pkmn/randbats, `data/stats/gen9randombattle.json`. https://github.com/pkmn/randbats

[3] C. R. Rao (1945). Information and the accuracy attainable in the estimation of statistical
parameters. *Bull. Calcutta Math. Soc.* 37, 81–91.

[4] D. Blackwell (1947). Conditional expectation and unbiased sequential estimation. *Ann. Math.
Statist.* 18(1), 105–110.

[5] G. Casella, C. P. Robert (1996). Rao–Blackwellisation of sampling schemes. *Biometrika* 83(1),
81–94.

[6] G. H. Golub, J. H. Welsch (1969). Calculation of Gauss quadrature rules. *Math. Comp.* 23(106),
221–230.

[7] T. J. Dekker (1971). A floating-point technique for extending the available precision. *Numer.
Math.* 18(3), 224–242.
