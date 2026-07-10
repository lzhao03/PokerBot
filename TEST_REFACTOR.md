# Lean poker test suite

This rewrite consolidates the supplied suite from 14 test translation units to 5:

- `src/cards_test.cc`
- `src/game_test.cc`
- `src/range_test.cc`
- `src/cfr_test.cc`
- `src/cfr_coarse_test.cc`

`cfr_coarse_test.cc` must remain in a separate test target compiled with
`POKER_COARSE_PUBLIC_BUCKETS=1`. Mixing it into the normal target would either
trip its compile-time guard or silently run the rest of the suite under the
wrong public-bucket mode.

## Build migration

Replace the normal test target's old source list with:

```text
src/cards_test.cc
src/game_test.cc
src/range_test.cc
src/cfr_test.cc
```

Replace the coarse-public-bucket target's source with:

```text
src/cfr_coarse_test.cc
```

Then delete these old files:

```text
src/betting_abstraction_test.cc
src/card_abstraction_test.cc
src/card_utils_test.cc
src/cfr_board_bucket_test.cc
src/cfr_equivalence_test.cc
src/cfr_solver_test.cc
src/cfr_traversal_guard_test.cc
src/combo_test.cc
src/game_rules_test.cc
src/game_semantics_property_test.cc
src/hand_evaluator_test.cc
src/hand_range_test.cc
src/strategy_store_test.cc
src/training_range_test.cc
```

## What changed

The rewrite keeps four kinds of tests only:

1. Exhaustive checks for small finite spaces, such as combo encoding.
2. Property tests for broad invariants, such as game-state conservation,
   abstraction invariance, and range-model equivalence.
3. Focused boundary regressions, such as wheel straights, short all-ins,
   exhausted decks, stale range scratch, and CFR+ clipping.
4. One integration test for each materially different CFR execution mode:
   mutable warmup, frozen storage, read-only evaluation, capped storage, and
   coarse public buckets.

Overlapping examples were folded into table-driven or property tests. Repeated
`Combo`, `ExactRange`, solver configuration, and fixture code now exists once
per behavioral domain instead of once per original file.

## Size comparison

| Metric | Supplied suite | Rewrite |
|---|---:|---:|
| Test source files | 14 | 5 |
| Test cases | 46 | 31 |
| Source lines | 2,313 | 1,884 |
| Explicit `CHECK`/`REQUIRE`/`RC_ASSERT` calls | 231 | 215 |

This is a conservative reduction: 64% fewer files and 33% fewer named test
cases while retaining about 93% of the explicit assertions. Property checks
still exercise many generated states per named test case.

## Execution policy

Run the four normal files on every change. Keep the generated evaluator-table
comparison identifiable by `[slow]`; exclude it only from very short local
loops, not from CI. Run the coarse target in CI because it uses a distinct
compile-time configuration and cannot be covered by the normal binary.

## Validation status

The consolidated normal and coarse targets compile and pass under
`bazel test //...`.
