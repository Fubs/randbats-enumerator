# The randbats team loop, as mathematics

The weights of PS's gen9 random-battle team loop, computed exactly: the
mathematics and the measurements. The weights are how often each species
stands in each party slot, as the lead or not, and which `teamDetails` it
is rolled under there. Every set table is already exact, enumerated
through a draw-for-draw port of PS's generator ("the port" below; §9);
these weights are what turns the set tables into per-slot set
distributions. Paths are relative to this crate, and README.md says how
to run it. Sections 1–7 are the method and its measurements, §8 the
files it writes, and §9 the port.

## 1. The object

- **Pool.** A multiset E of N = 428 entries. Entry e belongs to base b(e),
  which has multiplicity μ_b ∈ {1,2,3} (`getPokemonPool`: ⌈formes/3⌉, at most
  3, except Squawkabilly, whose 4 formes get μ = 1) and formes F_b,
  φ_b = |F_b|. The species set Σ = ⋃ F_b has 509 members.
  Write ω(s) = μ_b/φ_b for s ∈ F_b: a species' share of a uniform draw.
- **Admissibility.** A family 𝓘 ⊆ 2^Σ: one species per base; for every type
  t at most 2 members of type t, at most 3 weak to t, at most 1 doubly weak
  (PS's `getEffectiveness`, an immunity scoring 0); at most 4 Freeze-Dry
  weak; at most 1 at level 100; the compatibility exclusions (web setters,
  screen setters, Blissey/Chansey, Illumise/Volbeat, Toxicroak/sun). 𝓘 is
  hereditary. The position rules (Zoroark not sixth, a no-lead species not
  sixth without a lead) and the set-dependent ones (Ogerpon/Terapagos after a
  Tera Blast member, Dry Skin/Fluffy as a Fire weakness) are monotone too.
- **Process.** `sampleNoReplace` over the pool is a uniformly random
  permutation of E. Equivalently: independent arrival times U_e ~ U(0,1) and
  labels X_e ~ U(F_{b(e)}); visit entries by arrival; accept e iff
  A ∪ {X_e} ∈ 𝓘, until |A| = 6. The lead is the first accepted species not on
  the no-lead list. This is the random greedy (random sequential adsorption)
  process on a hereditary family.
- **Decoration.** An accepted species s rolls its set from its exact kernel
  K_{s,ℓ}(· | c), c being the OR of earlier members' flags.
- **Wanted.** W_T(ℓ, c) = P(T accepted, lead flag ℓ, context c).

## 2. Structure

**Lemma 1 (rejections are invisible).** A label is fixed at arrival, 𝓘 is
hereditary and the team only grows, so a rejected entry would be rejected at
every later moment. A rejected draw carries no information about the
future.

**Lemma 2 (the law of a team).** For an ordered admissible team
S = (s₁,…,s₆) with prefixes A_k, let α_e(k) be the fraction of entry e's
formes admissible given A_k. Then

  P(S) = ∏_k ω(s_k) · ∫_{0<τ₁<⋯<τ₆<1} ∏_{e unused} (1 − Σ_{k=0}^{5} α_e(k)(τ_{k+1} − τ_k)) dτ,

τ₀ = 0. (Entries are independent; the entry used for s_k arrives at τ_k with
label s_k; every other entry arriving during stage k draws an inadmissible
label; arrivals after τ₆ are free.)

**Corollary 3 (product formula).** If every α_e is 0 or 1 — no multi-forme
base partly admissible — α_e(k) = 1{k < δ_e} for a drop index δ_e, the
integrand is ∏(1 − τ_{δ_e}), and iterated Beta integrals give

  P(S) = ∏_{k=0}^{5} ω(s_{k+1}) / M(A_k),   M(A) = Σ_{s admissible given A} ω(s):

the next species is uniform over the admissible mass. Fractional α keeps
Lemma 2 exact: integrals of polynomials over a simplex are Dirichlet moments,
rational in closed form.

**Lemma 4 (exchangeability).** Call single-forme species without a special
rule equivalent when they share their typing and level-100 status
(weaknesses and Freeze-Dry follow from the typing). P(S) is invariant under
permutations within classes, so the class sequence C has law
∏_j (n_j)_{m_j} · ∏_k ω/M_k(C), with falling factorials choosing distinct
members. A class appears at most twice: its types cap at 2.

**Lemma 5 (flags are a linear decoration).** Except through the Tera Blast
and Dry Skin/Fluffy rules, acceptance never reads a set, so the flags are a
Markov chain riding on the species sequence. A class picked once contributes
its averaged kernel K̄_j = (1/n_j)Σ K_s; a class picked twice, the exact
distinct-members correction (n/(n−1))[K̄∘⋯∘K̄ − (1/n²)Σ_s K_s∘⋯∘K_s]. Flag
behaviour is no part of a class's identity.

**Lemma 6 (the minimal state).** The future depends on a partial team A
through its cap profile — per type, members (capped at 2), weaknesses (at
3), double weaknesses (at 1); Freeze-Dry, level 100, compatibility, Tera
Blast — the species-clause correction, which reads the members' classes,
and **the forme labels of the multi-forme bases A has made partly
admissible**. Partial teams almost never share a profile: it lives in an
18-type × 3-kind space, and the simple species spread over many typings:
342 over 162 in the pool of 2026-09-14, 367 over 168 ordered typings in the
2026-10-18 pool, which the vendored PS still has.

Draw every entry's label in advance —
it is fixed at arrival and independent of the order — and the process is
Corollary 3's: the next acceptance is uniform over the entries whose label
is admissible. A base whose formes go from F to F′ ≠ ∅ keeps each live
label with probability |F′|/|F|, and whether it did is not a function of
A: it is a hidden bit per entry, alive or dead, that M and every later
step read. Expanding Lemma 2's integrand instead — each partly admissible
entry's factor (1/φ)Σ_f(1 − τ_{δ_f}) split into the formes already dropped
and those still alive — gives the same dimension, as does carrying
"arrived and burned" per entry. One sees it a fourth way: the factor
(1 − P_b − F_b)^r of a base, P_b its past drops and F_b its future ones,
expands into r + 1 terms the past cannot integrate away. Any exact
representation carries ∏(r_b + 1) components per class multiset, over the
bases b partly admissible with r_b live entries.

## 3. What pen and paper reaches

- The exact law of any single team (Lemma 2, Corollary 3).
- **The lead.** The first acceptance is from the empty team, where
  everything is admissible:
  P(T leads) = ω(T)/N + Σ_{n no-lead} (ω(n)/N) · ω(T)·1{T adm. given n}/M({n}) + (no-lead pairs) + ⋯,
  a short sum over the eight no-lead species. The lead's context is empty
  unless a no-lead species came first, and then it is their exact set tables.
- **The cap-free process.** With only the species clause, the clock
  representation gives P(T on team) as one integral of a polynomial,
  (1/φ_b) ∫₀¹ μ_b(1−u)^{μ_b−1} · [z^{<6}] ∏_{b′≠b}((1−u)^{μ_{b′}} + (1−(1−u)^{μ_{b′}}) z) du,
  b = T's base (its first entry's arrival density, times "fewer than six
  other bases arrived first")
  — an exact rational. The caps break the product: whether a third
  Water-type is accepted depends on the order of arrivals, not their count.
- **Not the rest.** P(T on team) and every non-lead context are sums over
  the partial-team profiles of Lemma 6.

## 4. Measured size (the PS pool of 2026-09-14)

The counts in this section were measured on the PS pool of 2026-09-14, not
the current one.

| | |
|---|---|
| Classes (Lemma 4; flags out, multi-forme formes and rule species apart) | K = 323: 162 typing classes for 342 simple species, 124 for 130 multi-forme formes, 37 rule species |
| Reachable class multisets, sizes 1 / 2 / 3 | 323 / 51,423 / 5,296,954 |
| Size 4 | past 1.5·10⁸ (stopped at a cap); estimated ~4·10⁸ |

The table above counts class multisets only, and so leaves out Lemma 6's
forme labels, which multiply the states by orders of magnitude. The pool
has 54 multi-forme bases holding 61
entries (μ = 1 for 48, 2 for 5, 3 for 1). Along 200,000 teams sampled from
the port:

| Size | Partly admissible bases | Their live entries | ∏(r+1): median | p90 | mean |
|---|---|---|---|---|---|
| 1 | 0.52 | 0.61 | 1 | 4 | 2.6 |
| 2 | 1.75 | 2.42 | 2 | 96 | 136 |
| 3 | 4.77 | 6.67 | 16 | 20,736 | 1.0·10⁵ |
| 4 | 8.72 | 11.95 | 1,728 | 884,736 | 5.5·10⁵ |
| 5 | 11.82 | 15.97 | 27,648 | 1.8·10⁶ | 9.4·10⁵ |

The exact DP (`src/main.rs`, labels as state)
agrees with 4M sampled teams at positions 1–2 (20,237 cells, Bonferroni
p = 1.0), and measures the multiplier directly: 76,297 class multisets at
size 2 become 8,754,407 states. Size 3 passes 40M states before it stops
at its memory cap. The heavy tail decides the totals, since a state count
is a mean over multisets: about 10⁹ multisets at size 4 times a mean in
the 10³–10⁵ range is 10¹²–10¹⁴ states, terabytes to petabytes, and the
transitions out of them several hundred times that. Carrying "arrived and
burned" instead of "label dead" lags the dimension one size (389,000
states at size 2) but enumerates the burns of every partly admissible base
at every step — the size-3 multiplier, per transition.

So the exact weights are not a computation for one machine by holding
states; §§5–7 integrate the label splits instead. The mathematics
still sets the shape any attempt would take — flags as decorations, labels
as the only hidden state, the last pick folded by signature (w/M, with the
step-5 label splits integrated as ∫x^{c−1}∏(1 − q + qx)^r dx) — but held
as states, the label dimension is exponential in the partly admissible
bases.

## 5. Party slots 1–4

The weights of the first four party slots are within reach, and computed.
`exact_team_weights --slots` gives P(species in party slot k, lead,
`teamDetails` context) for k = 1…4 with no sampling. Party slot means PS's
order: `randomTeam` `unshift`s the lead and appends the rest in generation
order.

- **Held states.** Sizes 1–2 are held: 8,754,407 label states.
- **Slot 3** is the step out of them.
- **Slot 4** is the step after that, folded without the size-3 states it
  passes through. That step's label splits are integrated rather than
  enumerated: E[1/M] = ∫₀¹ x^{f−1} ∏_b (1 − q_b + q_b x)^{r_b} dx, and
  E[A_b/M] likewise (1/M = ∫x^{M−1}, E[x^A] = (1−q+qx)^r). Enumerated,
  3,001 sampled states alone realized 4.3·10⁹ size-3 states, so the whole
  pass would be about 10¹³.
- **Late leads.** Leads drawn after never-leads species only come from the
  lead-less states, carried to size 5.

The answer is held to everything at hand:
- Each slot's mass is 1.
- 40M sampled teams agree on all 1.12M (species, slot, lead, context)
  cells: smallest exact Poisson p 4.9·10⁻⁶, Bonferroni 1.0.
- The integrated fold equals enumeration: to 2·10⁻¹⁴ in floats on 3,001
  states, and exactly, in every residue, on 201.

In floats it takes 11 minutes on 32 threads, and the files of slots 1–4
render from it in seconds (§8).

**Exact rationals.** Everything the walk of slots 1–4 forms is a rational
with small factors: w/M, a split's binomial, ω = μ/φ, the integrals, and
the set tables' branch probabilities (`enumerate_random_set_exact` hands
each leaf's factors to the caller). So the same walk also runs in residues:
`--exact BATCH` writes one batch of residues modulo 32 primes below 2^63
(Montgomery arithmetic), at about 100 minutes a batch on 32 threads, ~9.5×
the float run. This crate does not rebuild the rationals from the batches;
that takes the Chinese remainder theorem and rational reconstruction,
checked on held-out primes. Rebuilt that way, slot 2's shares match the
floats to 3·10⁻¹⁶, and they are small: the huge path denominators cancel
in each share's ratio, leaving at most 105 characters a share.

## 6. Party slot 5

Computed exactly (2026-10-18, the PS pool of that date): `exact_team_weights
--slots --slot5 --gpu` (crate `teamgen`, `--features gpu`), 87 minutes on 32
threads and one RTX 4090, slot 5 itself 75 of them. `slots` renders the
files of slots 1–5 from its output (§8).

- **The integral.** Steps 3 and 4's label splits are integrated jointly:
  with 1/M₃ = ∫x^{M₃−1} and 1/M₄ = ∫y^{M₄−1}, an entry of a size-3 live base
  contributes (1−q₃) + q₃x((1−q₄) + q₄y), the fourth pick's own base x∂/∂x
  on one entry, and each base whose count after step 4 is random a y∂/∂y
  moment, which its corrections need. V = E[w₄/(M₃M₄)] is the integral of a
  polynomial against x^{f₃−1}y^{f₄−1}: Gauss–Jacobi in each variable is
  exact (Golub–Welsch in u = 1−x, Newton-polished, 1.6·10⁻¹⁵).
- **Group-major.** A label group — states sharing members, lead, Tera Blast
  and Fire flags — differs only in dead labels. One (group, third pick)
  item takes each fourth pick's options, kernel, signature deposit and
  corrections once, on contexts summed over the group, each variant weighted
  by its own integral. Variants with the same alive counts on every size-3
  live base have the same integral: they are summed first, into classes
  (median 16k a situation, up to 41k).
- **Profiles.** A fourth pick that changes no live base's admissible formes
  has q₄ = 1, so each entry is (1−q₃) + q₃xy and the class sums collapse to a
  polynomial in xy per situation, contracted with 1/((f₃+k)(f₄+k)). 24.5% of
  fourth picks qualify. A split base's moment divides its factor back out of
  the class polynomial exactly (top-down when q ≥ a, else bottom-up).
- **The rest on the GPU,** in f32 pairs (~48 bits). A block takes 256
  key-sorted classes of a pick, a thread each: the node tables 32 at a time
  out of shared memory, the bases every lane of a warp shares multiplied once
  a node, accumulators specialized on the random-base count, each block under
  the smallest rule exact for its own classes. Phase B, the integrand and its
  moments, is 99% of the work; the moment accumulations are 65–80% of its
  instructions.
- **Checks.** Per-state against group-major over 3,000 states: every cell
  agrees, worst relative gap 1.1·10⁻¹². The full run: each slot's mass is 1
  (slot 5 to 10⁻¹⁵); 40M sampled teams agree on all 3.57M cells, smallest
  exact Poisson p 1.65·10⁻⁶, Bonferroni 1.0.

Two measured facts point at slot 6. A situation's class contexts are one
profile scaled, in every situation checked: the codes come from the
members' kernels, not the labels. And integrating steps 1 and 2's splits
as well — t₁ for 1/M₁, t₂ for 1/M₂ — would make a group's class weights a sum
of rank-one products over bases, one per (t₁, t₂) node: 14–40 nodes for the
groups holding most states (`--cp-stats`), against thousands of variants
and ~16k classes.

## 7. Party slot 6

Designed, coded and checked, not yet run in full: `exact_team_weights
--slots --slot5 --slot6 --gpu` (`src/six.rs`, its node loop on the GPU in
`cuda/six.cu`), whose output `slots` renders as exact (§8); `SLOT6=1
run.sh` runs it end to end. The pass is
34.1M (group, third pick) items in 1024 checkpointed epochs, estimated at
~50 h on 32 threads and one RTX 4090: ~49.3 h of epochs and ~0.5 h of the
rest. PS lands the gen9 random-battle balance patch around the 1st of each
month, and any patch stales these weights, so the pass is best run just
after one; until then the sixth-slot files are a sampled estimate (below).

- **The integral.** Steps 3, 4 and 5's label splits are integrated jointly:
  x for 1/M₃, y for 1/M₄, z for 1/M₅. A pick is a (situation, fourth pick);
  its fifth picks, each outcome of a flagged set its own, are grouped by the
  context after them — a group is one sixth-pick pool, so one Gauss–Jacobi
  rule in z, weight z^{m_min−1}. The node loop fills its arrays at Chebyshev
  points in z (first kind, so no E₀ vanishes), and each group reads them at
  its own nodes by barycentric interpolation.
- **Bases.** A size-4 live base whole at steps 3–4 and unchanged by the
  fifth pick (W) folds into a z-factor or a constant; its credits come as
  U − E, all deposits less those that changed it. Split or varying ones (P)
  are the node loop's factors.
- **Rows.** The situation's classes, or its rank-one terms — steps 1–2
  integrated, each random base (α + βE)^μ — when fewer. Rows alike over the
  varying P bases merge before the node loop; the code weights are one
  profile scaled almost always (`single`).
- **The node loop,** per (x, y) node and eight z lanes: the rows in tiles of
  32 through their bases' f, E·f′/f and change ratios, then H, Hv and each
  key's sums (a key: a distinct set of varying-base changes) as register
  sums over the tile, eight sums a pass, the keys' change products built
  down a trie of row sweeps. Patterns without a key take their moments as a
  4×4 register-blocked product over the nodes. Each node's table is built in
  the loop: all of them at once outgrew a thread's share of L2.
- **On the GPU.** `--gpu` runs the node loop in f32 pairs (~48 bits). A
  block takes a (pick, z range); per node it shares a table of E, E², E³
  and ρ per slot, rank-one rows' f and E·f′/f per row tile, and each key's
  row sums. A pick whose rows' code weights are not one profile scaled
  (~1.4% of picks, 6.7 codes, 24 rows) has weights spanning two profiles,
  in every one measured, so `prepare` finds an orthonormal basis of the
  span (Gram–Schmidt, twice) and each row's coordinates, kept only if they
  rebuild every row to 10⁻¹⁴; the node loop runs the basis rows and
  `Prep::expand` makes the codes' arrays.
- **Deposits.** Per group, a core vector for u and each changed W base's
  moment; identity-routed P credits wait for the pick's end, on their z
  weights summed by (pattern, Tera Blast); the table entry, U and the
  group's W exceptions come once a group, on its deposits summed. The
  species clause: the fourth pick's members once a group, each fifth
  pick's own member by pool size.
- **Epochs.** A whole-pass signature table does not fit in memory: a 1/1000
  sample holds 7.8M signatures and 116M entries, growing ~1.9× per 3× the
  items, so ~400M signatures and ~150 GB for the pass. The pass runs in
  1024 stratified epochs (`six_epochs`), each epoch's table credited when
  it ends, by refusal mask (`table_pass6`). Weights are checkpointed after
  slots 1–5 and after every epoch (`<out>.ckpt`): written aside, synced,
  the one before kept by hard link as `.ckpt.prev`, and swapped in by one
  rename. A rerun resumes after the last checkpoint; a damaged one is moved
  to `.ckpt.bad` and the one before is loaded.
- **Checks.** `--check-six` against the per-state fold
  (`TG_FORCE_CP` / `TG_NO_CP` / `TG_CHECK_BIG` choose the paths): every
  cell agrees, worst relative gap 7.8·10⁻¹⁴ on small groups and 6·10⁻¹² on
  big ones, by classes and by rank-one terms alike. `TG_SIX_GPU_CHECK`
  holds every GPU pick to the CPU loop: worst gap 1.5·10⁻¹³ over 2.7·10⁹
  entries, and 1.7·10⁻¹³ over 2.8M arrays with the multi-row picks on the
  GPU too. End to end, `TG_SIX_EPOCHS=8 exact_team_weights --slots --slot5
  --slot6 --gpu --limit-groups 8` (~4.5 min) against an earlier build cell
  by cell: all 2,563,996 cells, slots 1–5 exact, slot 6's mass to the 17th
  digit, its worst cell 6.5·10⁻¹³ (at 10⁻⁹ of the largest; 2.0·10⁻¹³ above
  10⁻⁶ of it, 4·10⁻¹⁵ above 10⁻³). `--limit-groups 2`, four epochs, killed
  (SIGKILL) after epoch 2 and rerun: resumed at 2/4, all 1,742,740 cells
  within 1.1·10⁻¹³ of the uninterrupted run (1.3·10⁻¹³ again with the
  checkpoint written on its own thread). Reruns agree to ~10⁻¹³, not
  bit for bit: which loop a pick takes depends on the GPU's backlog at the
  time.

**Cost.** One epoch's share of the items (`--time-six 1024`) takes
172–174 s, hence the ~50 h. The node loop runs on the GPU, and workers
never block on it: each keeps two batches in flight, and a pick's node
loop goes to the CPU instead once the GPU's backlog passes 24, so neither
side idles at an epoch's ends. A single-row pick (94% of picks) deposits
only scalar multiples of a few code vectors fixed when it is prepared, so
its deposits add as scalars and expand once. The pass is CPU-bound, by the
16 cores rather than the 32 threads, with the GPU's backlog ~7. On the CPU
alone it was measured at about 117 h (1.35·10⁷ thread-seconds, on 0.33% of
the items), before the CPU side's later speedups.

### The slot-6 estimate

Until the pass runs, the sixth slot is a sampled estimate, in the exact
files' schema, so the exact pass's files replace it as they are.
`manifest.json` lists slot 6 under `estimated_slots`, with the estimate's
method, teams, seeds and threads. The renderer makes it:
`slots --rb DUMP...` reads `exact_team_weights --mc-dump N --mc-rb 6`
dumps, sums every (species, context) cell's expected count over them (by
seed, each in file order) and divides by the total, so the slot's mass is
1; `run.sh` writes the dumps and renders. Every file that pools slot 6
carries the estimate: `slot-6.json`, `first-6-slots.json`, `non-lead.json`,
and the `"6"` column of `slot-probabilities.json`.
- **How.** `exact_team_weights --mc-dump N --mc-rb 6` draws teams through
  the PS-exact port's loop (`sampled_team`) to five members, sets rolled
  and `teamDetails` recorded, and does not draw the sixth pick. Every
  remaining pool entry is drawn at most once, so its forme label can be
  drawn in advance; it draws them, and adds the pick's distribution given
  the five and those labels, at their context: uniform over the entries
  whose label is admissible (Corollary 3). Unbiased to order 10⁻¹⁰, but
  still sampling: the first five and the labels are drawn, so it is never
  exact. The 10⁻¹⁰ is the chance that none of the first five may lead: the
  sixth pick is then the lead, in slot 1, and the fifth-generated member
  stands in slot 6, which the estimate never counts (it never happened in
  the 4·10⁸ teams). Each team gives every
  admissible species an observation of its context, where a plainly
  sampled team (`--mc-dump` alone) gives one species one: 2·10⁶ such
  teams matched 10⁹ plain ones. The normalized table is a ratio of these
  counts, so it is consistent, with a bias of order 1/n. The slot-6
  files: 4·10⁸ teams (seeds 301 and 302 at 30 threads), 2.42M cells.
- **Backtest.** The same at the fifth pick (`--mc-rb 5`, 2·10⁸ teams)
  against the exact slot 5, rendered through `slots` (README.md §6 has the
  commands): role weights within 4.7·10⁻⁵ (mean 1.9·10⁻⁶), items 3.8·10⁻⁵,
  moves 5.6·10⁻⁵, abilities 3.1·10⁻⁵; `first-5-slots.json` within
  1.3·10⁻⁵; species totals 2.1·10⁻⁵ in total variation.
- **Check.** Against 3·10⁹ plainly sampled teams (the port's ordinary
  loop, `--mc-dump` without `--mc-rb`, other seeds): species totals max
  |z| 3.17 over 507 (Bonferroni 0.77); contexts within species χ² 1.023 per degree of
  freedom, ~1.015 expected from the estimate's own noise (the exact slot 5
  scores 0.999). Zoroark is 0 in slot 6, as the loop refuses it there.
- **The alternatives, on the slot-5 backtest** (TV over cells / species
  totals / contexts within species): carrying slot 4 forward 0.11 / 0.022
  / 0.11; extrapolating each quantity's trend across slots (in the contexts' down-set domain, where
  each slot multiplies by a slowly moving ratio: P(context 0) falls by
  0.7565, 0.7566, 0.7558, 0.7549 a slot) 4.7·10⁻³ / 3.3·10⁻³ / 3.0·10⁻³;
  10⁹ plain teams 3.2·10⁻³ / 2.7·10⁻⁴ / 3.2·10⁻³; plain counts shrunk
  toward an extrapolated prior (empirical Bayes) 1.8·10⁻³ / 2.7·10⁻⁴ /
  1.8·10⁻³; the Rao–Blackwellized 2·10⁸ 4.1·10⁻⁴ / 2.1·10⁻⁵ / 4.0·10⁻⁴.
  The trends are smooth enough to extrapolate well (Alcremie's Bulky Setup
  0.5, 0.5205, 0.5393, 0.5570, 0.5738, then 0.5898 estimated), but the
  species totals are where they fall short.

## 8. The staging files

`slots` (`src/bin/slots/`) renders the weights into the staging directory,
`output/gen9randombattle/`, from which the repository's `publish.js`
publishes `data/gen9randombattle/` (README.md §3). The slots the weights
carry are rendered as exact; a slot they lack is estimated from `--rb`
dumps (§7). Every slot 1–6 must be covered, unless `--partial`, which
writes only the files whose slots are. The weights file is an
intermediate and is never published.

- **Files.** `lead.json` (slot 1), `slot-K.json`, `first-K-slots.json` (in
  one of slots 1–K; `first-6-slots.json` is the whole team, what pkmn's
  sampled `data/stats/gen9randombattle.json` estimates), `non-lead.json`
  (slots 2–6), `slot-probabilities.json`, and
  `manifest.json`: the exact and estimated slots, the estimate's teams,
  seeds and threads, the date, and `--meta`'s keys (the PS commit, the
  `fingerprint` and the `logic` of `dump.js`).
- **Mixtures.** A file for the slots S gives each species s the mixture
  Σ_{k∈S} Σ_c W_k(s, c) · K_{s,[k=1]}(· | c), normalized, of its exact set
  tables, each rolled under its lead flag and context, and summarizes it.
  A species is on a team at most once, so `slot-probabilities.json`'s
  p_k(s) = Σ_c W_k(s, c) recombine the slot files into any pooled one:
  P(· | s, in S) = Σ_{k∈S} p_k(s) P(· | s, k) / Σ_{k∈S} p_k(s), species
  marginals with weights p_k(s), a role's with p_k(s)·P(role | s, k).
  Sums are Neumaier-compensated, and each p_k(s) is its weights' correctly
  rounded sum (Shewchuk's partials, as Python's `math.fsum`).
- **Residues.** A weight that should be exactly zero can come out as a
  residue of at most 1.1·10⁻²⁵ in magnitude (about half of them negative),
  left by a credit and its correction cancelling. The smallest kept weight
  is 6.5·10⁻¹⁴ in slot 4 and 5·10⁻¹⁶ in slot 5, so the renderer's cut at
  10⁻¹⁷ (469 slot-4 and 1,013 slot-5 cells, on 2026-10-18 data) falls in a
  gap of nine orders of magnitude.
- **What pkmn's sampler converges to.** Display names, moves included.
  Every movepool move and Tera type of a role is listed, 0 where
  impossible, as `getRoles` seeds them; abilities and items only when seen,
  and no `items` where no set holds one (Jumpluff). A role is a label:
  Forretress' two Bulky Support sets are one role, seeded with the last
  one's pools. A role no set takes in a file (the five Fast Bulky Setups,
  which never lead) has weight 0 and shares 0, where pkmn would print null
  for 0/0. An EV or IV is listed when every set of positive probability has
  the same value and it is not the default; on 2026-10-18 data, pkmn's
  sampled files list the same ones.
  Full precision, a share within 10⁻¹² of 1 printed as 1. Species and
  equal shares are in `localeCompare` order (checked against Node on all
  60k maps), entries by descending share, roles by descending weight with
  ties in sets.json's order; shares within about 10⁻¹² of each other tie
  (the sort key rounds x·10¹²). pkmn sorts entries after rounding to four
  decimals; `publish.js` sorts the published files again at full precision,
  comparing shares to 12 significant digits.
- **Checks (2026-10-18 data).** `slots` parses floats correctly rounded
  (serde_json's `float_roundtrip`); serde_json's default would move up to 4%
  of a file's shares by 1–2 ulp (≤ 2.2·10⁻¹⁶). Two builds
  of `slots` on the same weights and dumps wrote the same fourteen files
  byte for byte. The `--mc-dump` files themselves vary in the last ulp from
  run to run, because the per-thread sums merge in finishing order.
- **Cost.** The render takes ~12 s and 1.6 GB on 16 threads. The dumps
  ran at 3.5·10⁵ teams/s on 16 threads, 4·10⁸ teams in ~19 minutes, as
  measured; `run.sh` runs them at 30 (`RB_THREADS`), not timed here.

## 9. The port

`gen9/` (crate `randbats-gen9`) is PS's `data/random-battles/gen9/teams.ts`
translated draw for draw: every `random(n)` PS makes, the port makes, in
the same order and with the same range, so a stream of PS's draws builds
the same team in both.

- **Its data.** `dump.js` writes what the port reads from the built
  vendored PS — species, moves, items, abilities, the type chart, the
  format's rules and gen9 `sets.json` in PS's order (a draw is an index
  into it) — with the few fields PS's generator reads that no set makes
  the port read, so that a change to one still moves the fingerprint, and
  `gen9/build.rs` generates the port's tables from it. The
  build asserts the format rules the port assumes (singles, six members,
  four moves, no level adjustment, no monotype or Tera rules, no custom
  bans) and refuses a default dump (`target/dump.json`, without
  `RANDBATS_DUMP`) of a PS commit other than the vendored one.
  The dump's hash is `fingerprint`; `logic` hashes `teams.ts`,
  `sim/prng.ts` and `sim/dex.ts`, the sources the port translates.
- **Parity.** `trace.js` rolls team i from the seed `sodium,<i as 64 hex
  digits>` in PS, records every draw with its range and value and the team
  built, and refuses a float draw, which the port has no counterpart for;
  `--sites` adds the `teams.ts` lines behind each draw. `parity` replays
  each trace through the port and exits 0 only if every team is rebuilt
  exactly, at least one was read, and, with `--expect N`, exactly N were.
  10⁶ teams rebuild exactly (39 s at 16 workers; 10⁵ at 4 workers, 10.7 s).
- **Set tables.** `enumerate_random_set` walks every branch of
  `randomSet` for a species, lead flag and context, each leaf with its
  exact probability. At the PS of 2026-10-18, PS's own `randomSet`,
  enumerated the same way over every branch, gave the same 107,802 tables:
  1,017 (species, lead flag) pairs, every species as a non-lead and all but
  Iron Thorns as a lead, times 106 contexts (1 with no flag, 14 with one,
  Spikes 1 and 2 counted apart, 90 with two, and 1 with all). That PS-side
  enumerator is not part of this crate.
