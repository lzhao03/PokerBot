# Deferred and Removed Features

This ledger records capabilities intentionally removed in commit `dcbadfc`
(`Simplify poker and CFR model`). Removal here means "not worth carrying in
the current implementation," not "never build this again."

When restoring an item, implement it in a focused commit, add its verification,
and change its status here instead of deleting the entry.

The complete previous implementation remains recoverable with:

```sh
git diff dcbadfc^ dcbadfc
git show dcbadfc^:<path>
```

## High-value workflow features

### Training checkpoints

- **Status:** Deferred; first feature to reconsider for long training runs.
- **Removed:** `SolverCheckpoint`, `SerializedRngState`, `checkpoint()`,
  `restore()`, `SaveCheckpoint()`, and `LoadCheckpoint()`.
- **Previous behavior:** Persisted the model fingerprint, CFR rows, regrets,
  average-strategy sums, iteration count, cumulative utility, and exact
  `mt19937` state. Restore validated versions, fingerprints, row spans, array
  shapes, finite values, and resumed training bit-for-bit.
- **Why removed:** No production caller existed and the manual binary format,
  reader, validation, and restore path accounted for roughly 250 source lines.
- **Restore when:** Training jobs need interruption recovery, migration between
  machines, or reproducible continuation.
- **Recovery locations:** `src/solver.h`, `src/solver.cc`, and
  `tests/cfr_solver_test.cc` in `dcbadfc^`. The old file magic was `PKCHECK1`.

### Policy loading and artifact compatibility

- **Status:** Deferred; likely needed when policies are consumed by this repo.
- **Removed:** `LoadPolicy()`, `ByteReader`, and `ReadBytes()`.
- **Previous behavior:** Loaded `PKPOLCY1` files and rejected invalid magic,
  versions, truncation, duplicate rows, trailing bytes, bad spans, nonfinite
  probabilities, and non-normalized rows.
- **Why removed:** Only a round-trip test called the loader; executables only
  wrote policies.
- **What remains:** Atomic `SavePolicy()` using the smaller `PKPOLCY2` format.
- **Restore when:** Evaluation, serving, inspection, or continued training must
  consume a saved policy.
- **Recovery locations:** `src/solver.h`, `src/solver.cc`, and the policy
  round-trip test in `tests/cfr_solver_test.cc` from `dcbadfc^`.
- **Compatibility note:** A restored loader should make an explicit decision
  about reading v1 and v2. Model fingerprints also changed to schema 2.

### Byte-budgeted policy export

- **Status:** Deferred.
- **Removed:** `extract_average_policy(max_serialized_bytes)` and benchmark flag
  `--policy_max_bytes`.
- **Previous behavior:** Ranked infosets by accumulated strategy mass and kept
  the highest-mass rows that fit a deterministic serialized-byte budget.
- **Why removed:** The deployment constraint was unproven; its only non-test
  caller was an optional benchmark flag whose default was unlimited.
- **Restore when:** Policies must fit a hard storage, memory, or transport limit.
- **Recovery locations:** `ExtractAveragePolicy` in `src/solver.cc`, its overload
  in `src/solver.h`, the lossy-policy tests, and `tools/cfr_benchmark.cc` in
  `dcbadfc^`.

## Poker and policy inspection features

### Semantic hand evaluation

- **Status:** Deferred from the runtime API; retained only as a test oracle.
- **Removed:** Runtime `HandRank`, `HandEvaluation`, `EvaluateFiveCards()`, and
  raw `HandValue()`.
- **Previous behavior:** Returned named hand categories, kickers, comparisons,
  and internal Cactus values for diagnostics or user interfaces.
- **Why removed:** The solver only needs `CompareHands()`; semantic evaluation
  was otherwise used by tests and the table generator.
- **What remains:** Seven-card showdown comparison and an independent
  five-card scoring model in `tests/card_utils_test.cc`.
- **Restore when:** A UI, hand-history inspector, debugging tool, or public API
  needs names, kickers, or stable raw values.
- **Recovery locations:** `src/hand_evaluator.*` and the removed generator
  tooling from `dcbadfc^`.

### Direct history-edge action metadata

- **Status:** Deferred.
- **Removed:** `HistoryEdge {GameAction, child}`, `EdgeRange`, the four history
  node wrappers, and stored edge actions.
- **Previous behavior:** External tree consumers could read the action attached
  to every child directly.
- **Why removed:** Node wrappers duplicated `BettingState`; solver traversal only
  needs child IDs, and action order is reproducible with
  `SelectAbstractActions(config, decision)`.
- **What remains:** `HistoryNode {BettingState, children_begin, child_count}` and
  the complete child tree.
- **Restore when:** Tree visualization, policy serving, debugging, or external
  traversal needs self-describing edges without the solver configuration.
- **Recovery locations:** `src/solver.h`, history construction/fingerprinting in
  `src/solver.cc`, and the direct-transition test in `tests/cfr_solver_test.cc`
  from `dcbadfc^`.

### Public abstraction diagnostics

- **Status:** Internalized rather than behaviorally removed.
- **Removed from the public API:** `BoardFeatures`, `BoardBucketId`,
  `PrivateBucketId`, coarse bucket-count constants, `BoardFeaturesFor()`,
  `BoardTextureBucket()`, and `Handcrafted36Bucket()`.
- **Why removed:** They expose the current heuristic implementation rather than
  the Poker/CFR observation contract.
- **What remains:** Exact/texture public observations, exact/handcrafted private
  observations, and both recall modes through `ObservePublic()` and
  `ObservePrivate()`.
- **Restore when:** Abstraction analysis or model-building tools genuinely need
  stable access to intermediate features and buckets. Prefer a tool-facing API
  rather than making solver internals public again.

## Diagnostics and operational telemetry

### Per-run serial/parallel training results

- **Status:** Deferred.
- **Removed:** `TrainingResult`, `iterations_completed`, `serial_iterations`,
  `parallel_iterations`, `get_cfr_update_count()`, and corresponding CLI output.
- **Why removed:** Completed iterations duplicated the input and update count
  duplicated `SolverStats::decision_visits`; only the serial/parallel split was
  unique information.
- **What remains:** Total solver iterations and `SolverStats` for decisions,
  chance samples, and terminals.
- **Restore when:** Performance tuning needs to measure prefill versus parallel
  work per `run()` call.
- **Recovery locations:** `src/solver.h`, `CFRSolver::run()` in `src/solver.cc`,
  `src/main.cc`, and `tools/cfr_benchmark.cc` from `dcbadfc^`.

### Solver INFO logging

- **Status:** Deferred; low priority.
- **Removed:** Main's `--log` flag, Abseil log initialization, and the solver's
  end-of-run INFO summary.
- **Why removed:** The messages duplicated the main/benchmark summaries and
  there were no actual VLOG call sites.
- **Restore when:** The solver library needs structured or callback-based
  progress reporting. Prefer a deliberate reporting interface over implicit
  global logging.

### Redundant evaluation result fields

- **Status:** Intentionally omitted unless consumers demonstrate a need.
- **Removed:** `ValueEstimate::samples`, `BestResponseResult::responder`, and
  `BestResponseResult::training_iterations_completed`.
- **Why removed:** They repeated unchanged request inputs.
- **Restore when:** Serialized result records must be self-contained without the
  request/configuration that produced them.

## Experimental and build-time features

### Sampled CFR traversal variants

- **Status:** External sampling is available through `external_sampling` and
  `--external_sampling`; outcome and public-chance sampling remain deferred.
- **Current behavior:** Samples opponent actions while enumerating actions for
  the player being updated. Evaluation remains a full traversal.
- **Why defer the rest:** Additional variants add estimator variance, sampling
  weights, and different average-strategy accounting. The default
  chance-sampled traversal remains the simpler correctness baseline.
- **Convergence requirements:** The textbook external-sampling guarantee assumes
  a finite two-player zero-sum perfect-recall game and unbiased updates. The
  current infoset cap, current-bucket-only recall, CFR+ regret clipping, and
  concurrent updates fall outside that proof.
- **Extend when:** Measured wall-clock convergence justifies another sampling
  mode rather than further tuning external sampling.
- **Verification:** Compare sampled updates against exact CFR on toy games, track
  exploitability versus nodes touched and wall time, and first validate the
  single-threaded uncapped perfect-recall implementation.

### Separate production benchmark target

- **Status:** Not planned for restoration.
- **Removed:** `//tools:cfr_prod_benchmark` and its compile-time defaults macro.
- **Why removed:** It compiled the same source as `cfr_benchmark` only to change
  iteration and evaluation defaults from 100 to 1.
- **Equivalent command:**

  ```sh
  bazel run //tools:cfr_benchmark -- --iterations=1 --eval_samples=1
  ```

### RapidCheck dependency

- **Status:** Available to re-add if property-based tests are introduced.
- **Removed:** The unused module dependency, archive override, and custom Bazel
  patches under `third_party/rapidcheck`.
- **Why removed:** No test or source file referenced RapidCheck.
- **Restore when:** A property test benefits enough to justify the dependency;
  first check whether current Bazel support removes the need for local patches.

### Equity private abstraction design

- **Status:** Research idea, not removed runtime functionality.
- **Removed:** `docs/equity_private_abstraction.md`.
- **Why removed:** It described an equity-bucketing subsystem already absent
  from the parent source and explicitly called itself a deletion candidate.
- **Restore when:** Experiments demonstrate a quality/memory benefit over exact
  canonical or handcrafted-36 private observations.
- **Recovery location:** `docs/equity_private_abstraction.md` in `dcbadfc^`.

## Compatibility and reproducibility notes

- Policy files changed from `PKPOLCY1` to `PKPOLCY2`; v2 omits the stored action
  count and no loader currently exists.
- Model fingerprints changed to schema 2. Old policies/checkpoints are not
  compatible without an explicit migration path.
- Street-specific board types were replaced by one `Board`. Poker behavior is
  preserved, but invalid street transitions are now checked at runtime instead
  of being impossible to express in the type system.
- Deal sampling now draws independently from each weighted range and rejects
  overlapping hands. This is the same product distribution conditioned on no
  blockers, but consumes RNG differently and can be slower for highly
  overlapping ranges.
- Seeded training is therefore not bit-for-bit compatible with `dcbadfc^`.
  The small deterministic test changed from 720 to 714 infosets and from about
  `-1.01572` to `-0.428839` EV after ten iterations.

## Core capabilities retained

The current solver still includes heads-up Poker rules, typed cards and combos,
range parsing, blocker-aware chance sampling, exact and texture public card
abstractions, exact and handcrafted-36 private abstractions, recall modes,
CFR+ training, parallel fixed-table updates, average-policy extraction and
saving, expected-value evaluation, approximate best responses, exploitability,
model fingerprints, and the benchmark executable.
