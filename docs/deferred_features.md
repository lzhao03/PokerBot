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

### Compact infoset keys

- **Status:** Deferred until infoset-map memory is a demonstrated constraint.
- **Current behavior:** `InfoSetTable` uses one
  `flat_hash_map<InfoSetKey, uint32_t>` with a structural 16-byte key.
- **Potential design:** Losslessly encode the current 33-bit public observation,
  32-bit history, and 21-bit private observation into a fixed 96-bit key.
- **Measured tradeoff:** The removed conditional 64-bit encoding saved about
  8.4 MB (13.2% of process RSS) at 500,000 infosets and improved traversal
  throughput by roughly 3-8%, but required two map representations and packing
  branches. Benchmark a single 96-bit representation before restoring packing.

### Training checkpoints

- **Status:** Deferred; first feature to reconsider for long training runs.
- **Current partial behavior:** `--deep_checkpoint_interval` saves numbered Deep
  CFR average-policy snapshots during a run. These snapshots can be loaded for
  evaluation or deployment, but they cannot resume training state.
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

### Policy decoding and artifact compatibility

- **Status:** Restored as the compact `PKCODEC1` codec.
- **Current behavior:** `EncodePolicy()` and `DecodePolicy()` validate compact
  policy buffers; `SavePolicy()` atomically replaces files and `LoadPolicy()`
  reads and decodes them. Uniform rows are omitted and use the policy's uniform
  fallback.
- **Compatibility note:** Legacy `PKPOLCY1` and `PKPOLCY2` artifacts are not
  accepted by the compact codec.

### Byte-budgeted policy export

- **Status:** Deferred implementation; the current 1 MB deployment goal now
  justifies revisiting it.
- **Removed:** `extract_average_policy(max_serialized_bytes)` and benchmark flag
  `--policy_max_bytes`.
- **Previous behavior:** Ranked infosets by accumulated strategy mass and kept
  the highest-mass rows that fit a deterministic serialized-byte budget.
- **Why removed:** The deployment constraint was unproven; its only non-test
  caller was an optional benchmark flag whose default was unlimited.
- **Next design:** Rank rows and neural distillation samples using reach plus
  regret or value sensitivity, then validate held-out playing strength rather
  than probability error alone. Measure implicit key ordering and entropy coding
  only when they reduce total model-and-decoder bytes.
- **Recovery locations:** `ExtractAveragePolicy` in `src/solver.cc`, its overload
  in `src/solver.h`, the lossy-policy tests, and `tools/cfr_benchmark.cc` in
  `dcbadfc^`.

### Neural policy compression and dynamic bet sizing

- **Status:** Neural strategy compression is implemented; dynamic bet sizing is
  deferred.
- **Current behavior:** `FitNeuralPolicy()` trains the shared policy network from
  a tabular teacher. Deep CFR uses the same representation, and policies support
  native save/load plus portable `PNN1` browser export.
- **Current limitation:** The network predicts probabilities over the fixed legal
  actions produced by the betting abstraction.
- **Artifact compression:** Portable `PNN1` stores float32 parameters. Float16 or
  int8 weights with explicit per-layer scales remain deferred; compare total
  model-and-decoder bytes, inference speed, and strategy quality.
- **What remains deferred:** Let the network propose legal bet sizes dynamically
  and evaluate artifact size, strategy quality, and inference cost against the
  fixed-action models.

### Solver workflow quality of life

- **Status:** Deferred until repeated training and policy analysis justify a
  broader tooling surface.
- **Policy inspector:** Report file size, row count, quantization settings,
  action-count distribution, fingerprint, and decoding validity.
- **Single-state policy query:** Given hole cards, board, and action history,
  report abstractions, infoset key, legal actions, probabilities, and whether
  uniform fallback was used.
- **Run manifest:** Write a non-deployed sidecar with the normalized
  configuration, seed, iteration count, Git commit, throughput, peak RSS, and
  model hash.
- **Training preflight:** Report betting-history size, maximum action count,
  estimated memory at the infoset cap, and likely policy size without training.
- **Progress reporting:** `cfr_benchmark --progress_interval` reports iteration
  milestones, with throughput and infoset totals reported after training.
  Periodic throughput, elapsed time, ETA, RSS, cap utilization, and estimated
  encoded policy size remain deferred.
- **Policy comparison:** `//tools:policy_compare` compares matching tabular,
  Deep CFR, and distilled policies using seat-swapped EV and rough
  exploitability. Probability divergence and reach-weighted coverage remain
  deferred.
- **Strict validation:** Optionally fail on cap exhaustion, incompatible
  fingerprints, poor policy coverage, or an invalid exploitability estimate.
- **Saved configuration presets:** Keep named, checked-in configurations for
  reproducible common training runs instead of repeating long flag lists.
- **Deterministic hand replay:** Save a seed or compact hand history that the CLI
  and browser can replay when debugging strategy disagreements.

### Convergence-aware policy quality evaluation

- **Status:** Deferred beyond the current sampled approximate-best-response
  estimates.
- **Current limitation:** Evaluation reports sampling error and lookup coverage,
  but it does not establish that the trained response converged. A finite-budget
  approximate response can therefore produce an invalid negative NashConv.
- **Desired endpoint:** Sweep increasing response budgets, compare responses with
  the original-policy fallback, report cap and coverage failures, reject invalid
  estimates, and verify the estimator against exact toy games.
- **Use:** Track policy quality against wall time and artifact size so training,
  abstraction, and compression changes have a common regression signal.

## Poker and policy inspection features

### Optional betting-tree raise cap

- **Status:** Deferred; useful as an abstraction and resource-control experiment,
  not as a poker-rules correction.
- **Current behavior:** Passive actions close a street and aggressive actions
  reopen a response, so legal raises continue until stack and minimum-raise rules
  end them.
- **Previous behavior:** `max_raises_per_street` optionally truncated the tree;
  it was removed in commit `876e676`.
- **Restore when:** A measured memory or convergence experiment needs a shallower
  betting tree. Include the cap in the model fingerprint and keep uncapped play
  as the default.

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
  observations, and both recall modes through `PublicPosition` and
  `ObservePrivate()`.
- **Restore when:** Abstraction analysis or model-building tools genuinely need
  stable access to intermediate features and buckets. Prefer a tool-facing API
  rather than making solver internals public again.

### WASM-owned browser poker semantics

- **Status:** Complete for the browser demo.
- **Current behavior:** The WASM replay API owns betting transitions, legal
  abstract actions, street progression, showdown evaluation, and payouts.
  TypeScript owns deck sampling, presentation state, statistics, and rendering.
- **Constraint:** Do not add more poker, abstraction, or policy semantics to
  TypeScript; extend the WASM boundary instead so C++ remains the source of
  truth.
- **Verification:** `//web:policy_decoder_test` checks compact decoding,
  canonical action replay, terminal payouts, card-derived lookup, and portable
  neural inference.

### Reveal a winning bot hand

- **Status:** Deferred until the next poker-table UI pass.
- **Desired behavior:** Reveal the bot's hole cards whenever it wins, including
  pots won after the player folds. Showdown hands are already revealed.

## Diagnostics and operational telemetry

### Browser statistics cleanup

- **Status:** Deferred until the core game and policy experience is stable.
- **Current behavior:** `App.svelte` presents persisted session results and a
  dense per-street action table backed by `stats.ts`.
- **Cleanup:** Revisit which metrics earn space, tighten naming and layout, and
  make the expanded statistics view work cleanly on mobile without adding more
  logic to `App.svelte`.
- **Restore when:** The demo is ready for a dedicated usability pass.

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

### Lightweight tabular-only CLI binary

- **Status:** Deferred while one CLI is simpler to operate.
- **Current behavior:** `//src:poker_solver` selects tabular or Deep CFR with
  `--algorithm` and therefore links LibTorch even for tabular runs.
- **Desired endpoint:** Keep the shared CLI parsing and solve-spec construction,
  but offer a lightweight tabular target that does not fetch or link LibTorch.
- **Restore when:** LibTorch download size, link time, or deployment size becomes
  a practical problem for tabular-only use.

### Portable and scalable Deep CFR training

- **Status:** Deferred while local Apple Silicon CPU training is sufficient.
- **Current limitations:** Neural targets link the macOS ARM LibTorch archive and
  Accelerate, and Deep CFR requires `--threads=1`.
- **Desired endpoint:** Support Linux training and measured batching or parallel
  traversal improvements; add CUDA only when GPU profiling justifies it.
- **Verification:** Compare wall-clock policy quality, peak memory, and seeded
  single-worker behavior rather than throughput alone.

### Semantic neural history and generalization

- **Status:** Research item; low priority until one network must span multiple
  betting trees, stack depths, or dynamic bet sizes.
- **Current behavior:** Neural features include the numeric history ID plus
  semantic card, stack, pot, street, and actor features.
- **Previous experiment:** Recent-action features were added and reverted in
  `0e2e4b3` after showing no measured improvement.
- **Revisit when:** A held-out cross-configuration benchmark can demonstrate that
  a semantic action history improves strategy quality enough to replace the
  simpler encoding.

### Full Bazel web build

- **Status:** Deferred while the Vercel build remains small and reliable.
- **Current behavior:** `web/vercel-build.sh` builds only the WASM target with
  Bazel, then invokes Vite directly.
- **Desired endpoint:** One deliberate Bazel build verifies the full C++ tree
  and produces the deployable web artifacts, without duplicating build logic in
  shell scripts and package scripts.
- **Restore when:** CI reproducibility or another web artifact justifies owning
  the additional Bazel JavaScript integration.

### Sampled CFR traversal variants

- **Status:** External sampling is available through `external_sampling` and
  `--external_sampling`; outcome and public-chance sampling remain deferred.
- **Current behavior:** Samples opponent actions while enumerating actions for
  the player being updated, including approximate best-response training.
  Policy evaluation can optionally sample actions; default tabular evaluation
  remains a full traversal.
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

### Depth-limited and continual solving

- **Status:** Conditional research feature; unnecessary for the current static
  policy artifact and browser demo.
- **Previous behavior:** A continuation-value interface and exact-hand nested CFR
  solver were removed in `2aad5c1`; that implementation was not range-aware and
  should not be restored directly.
- **Desired endpoint:** If online solving becomes a product goal, use range-aware
  or learned continuation values with validated depth cutoffs and safe resolving.
- **Restore when:** Real-time adaptation is worth the runtime, model, and
  correctness complexity compared with serving a precomputed policy.

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

- Compact policy files use `PKCODEC1`; legacy `PKPOLCY1` and `PKPOLCY2`
  artifacts are not accepted.
- Model fingerprints use schema 4. Old policies/checkpoints are not
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
CFR+ training, parallel fixed-table updates, Deep CFR, tabular-to-neural policy
fitting, portable neural export, average-policy extraction and saving,
expected-value evaluation, approximate best responses, exploitability, model
fingerprints, and the benchmark executable.
