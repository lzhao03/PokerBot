# Equity Private Abstraction

## Status

This subsystem is optional and is a candidate for deletion. Keep it only if it
improves exploitability at comparable memory and wall-clock cost.

It is not part of poker rules, hand evaluation, public-card abstraction, CFR,
or policy persistence.

## Problem It Solves

The solver needs a private observation ID at each decision. The two simple
choices have opposite costs:

- `ExactCanonical` preserves every private distinction but creates many
  infosets.
- `Handcrafted36` is cheap but groups hands using arbitrary rank, made-hand,
  and draw heuristics.

An equity abstraction is intended to provide a smaller private state space
whose buckets group hands with similar present and future showdown value.

## Current Semantics

The current implementation maps private states as follows:

- Preflop: 169 canonical hand classes.
- Flop: 16 EHS-squared quantiles, each split by median EHS, for 32 buckets.
- Turn: the same 32-bucket structure.
- River: 64 current-equity quantiles.

`EHS` is expected showdown equity over sampled opponent hands and future
runouts. `EHS-squared` preserves some information about the distribution of
future equity rather than only its mean.

The quantile boundaries are fitted offline. The checked-in binary model stores
only those boundaries and sampling metadata. It is not a strategy, neural
network, state-to-bucket table, or precomputed equity database.

At runtime, a previously unseen canonical hand and board still requires
deterministic rollout sampling. The result is then stored in a mutable cache.

## Minimum Semantic Contract

If this feature remains, it should do exactly these things:

1. Accept an exact hole-card combination and exact typed board.
2. Produce a private bucket ID for the current street.
3. Return the same bucket for every global suit renaming of the same state.
4. Return the same bucket across runs, independent of cache state or traversal
   order.
5. Return a bucket within the configured street's fixed range.
6. Use an immutable, validated set of bucket boundaries.
7. Expose a stable model fingerprint that becomes part of the solver, policy,
   and checkpoint fingerprints.
8. Leave exact cards unchanged for blockers, chance sampling, and showdown.

The narrow runtime operation should conceptually be:

```cpp
PrivateBucketId EquityPrivateBucket(
    ComboId hand,
    const Board& board,
    const EquityBucketModel& model) noexcept;
```

This operation returns only a current-street bucket. It should not own recall
policy. `CardAbstraction` should decide whether the infoset uses only the
current bucket or packs the sequence of preflop, flop, turn, and river buckets.

## Minimum Model

The immutable model needs only fields that affect bucket identity:

```text
feature version
rollout seed and sample counts
flop EHS-squared cutoffs and EHS medians
turn EHS-squared cutoffs and EHS medians
river equity cutoffs
stable fingerprint
```

Training sample count and fitting seed are useful provenance, but they do not
belong in the runtime calculation unless they are intentionally part of model
identity.

## Integration Boundary

The intended data flow is:

```text
Offline
  sample exact states
  -> calculate deterministic equity features
  -> fit quantile boundaries
  -> emit immutable model data

Solver startup
  load model at the CLI/config boundary
  -> validate shape, values, version, and fingerprint once
  -> construct CardAbstraction

Traversal
  ObservePrivate(hand, public_position)
  -> calculate or retrieve current equity bucket
  -> apply configured recall encoding
  -> build InfoSetKey
```

The CFR traversal should not know about EHS, quantiles, model files, rollout
sampling, or caches.

## Not Minimum Requirements

These are implementation choices and should not be required by the abstraction
interface:

- Saving a model from the runtime solver library.
- Custom byte writers embedded in `card_abstraction.cc`.
- A mutable unbounded cache.
- Public `EquityFeatures`, `EquityBucket`, or model-finalization APIs.
- Runtime selection among multiple model file formats.
- Public-card abstraction behavior.
- Recall-history packing.

If only one equity model is supported, the simplest representation is a
generated C++ header containing validated constant boundaries and a fixed
fingerprint. That removes runtime model serialization entirely.

If external model files remain supported, loading belongs at the application
boundary. Model writing belongs to the offline builder. The runtime abstraction
should receive an already validated immutable model.

## Suggested Ownership If Retained

```text
card_abstraction.cc
  mode dispatch
  public observation construction
  private observation and recall encoding

equity_abstraction.cc
  deterministic feature calculation
  current-street equity bucket mapping
  immutable model validation

tools/build_equity_abstraction.cc
  fitting
  model emission
```

Persistence helpers should not sit beside board-texture encoding.

## Required Evidence Before Keeping It

Compare `EquityPotential` against `Handcrafted36` with the same:

- Betting tree
- Infoset or memory limit
- Training wall-clock time
- Chance samples
- Evaluation seeds

Record:

- Approximate exploitability or NashConv
- Infoset count and memory
- Decision visits per second
- Runtime cache size and miss cost
- Missing-policy lookup rate

Keep the subsystem only if it produces a meaningful quality improvement. If it
does not, delete the model, builder, serialization, rollout feature code, cache,
CLI flags, enum value, and associated tests together.
