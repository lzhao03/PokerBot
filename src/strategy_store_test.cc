#include "src/strategy_store.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>

namespace poker {
namespace {

void Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << "\n";
    std::exit(1);
  }
}

void ExpectNear(double actual, double expected, const char* message) {
  if (std::fabs(actual - expected) > 1e-9) {
    std::cerr << "FAILED: " << message << " actual=" << actual
              << " expected=" << expected << "\n";
    std::exit(1);
  }
}

struct StoreFixture {
  SolverConfig config;
  CardAbstraction card_abstraction;
  SolverStorage storage;
  TraversalStats stats;
  StrategyStore store{config, card_abstraction, storage, &stats};
};

ActionBlock CreateBlock(StoreFixture& fixture, absl::Span<const int> actions) {
  std::optional<ActionBlock> block =
      fixture.store.get_or_create({0, 0, 0}, actions);
  Expect(block.has_value(), "expected action block to be created");
  return *block;
}

void CheckRegretMatchingMissingBlockIsUniform() {
  StoreFixture fixture;
  double probabilities[3] = {};
  fixture.store.regret_matching_or_uniform(
      std::nullopt, 3, RegretLoadMode::kPlain,
      absl::Span<double>(probabilities));

  ExpectNear(probabilities[0], 1.0 / 3.0,
             "missing block should use uniform probability");
  ExpectNear(probabilities[1], 1.0 / 3.0,
             "missing block should use uniform probability");
  ExpectNear(probabilities[2], 1.0 / 3.0,
             "missing block should use uniform probability");
}

void CheckRegretMatchingPositiveRegretsNormalize() {
  StoreFixture fixture;
  const int actions[] = {10, 20, 30};
  ActionBlock block = CreateBlock(fixture, actions);
  const size_t offset = block.action_offset();
  fixture.storage.cumulative->cumulative_regrets[offset] = 0.0f;
  fixture.storage.cumulative->cumulative_regrets[offset + 1] = 2.0f;
  fixture.storage.cumulative->cumulative_regrets[offset + 2] = 6.0f;

  double probabilities[3] = {};
  block.regret_matching(RegretLoadMode::kPlain,
                        absl::Span<double>(probabilities));

  ExpectNear(probabilities[0], 0.0, "zero regret should have zero probability");
  ExpectNear(probabilities[1], 0.25, "positive regrets should normalize");
  ExpectNear(probabilities[2], 0.75, "positive regrets should normalize");
}

void CheckCfrPlusRegretClipsAtZero() {
  StoreFixture fixture;
  const int actions[] = {10, 20};
  ActionBlock block = CreateBlock(fixture, actions);
  fixture.storage.cumulative->cumulative_regrets[block.action_offset()] = 1.0f;

  block.add_cfr_plus_regret(
      0, -3.0f, RegretUpdateOptions{RegretUpdateMode::kPlain, false});

  ExpectNear(fixture.storage.cumulative
                 ->cumulative_regrets[block.action_offset()],
             0.0, "CFR+ regret update should clip at zero");
}

void CheckAverageStrategyFallbackAndRemap() {
  StoreFixture fixture;
  const int actions[] = {10, 20};
  ActionBlock block = CreateBlock(fixture, actions);

  double probabilities[2] = {};
  block.average_strategy(false, actions, 0.5,
                         absl::Span<double>(probabilities));
  ExpectNear(probabilities[0], 0.5,
             "zero average-strategy mass should fall back to uniform");
  ExpectNear(probabilities[1], 0.5,
             "zero average-strategy mass should fall back to uniform");

  const size_t offset = block.action_offset();
  fixture.storage.cumulative->cumulative_strategies[offset] = 1.0f;
  fixture.storage.cumulative->cumulative_strategies[offset + 1] = 3.0f;
  const int reversed_actions[] = {20, 10};
  block.average_strategy(false, reversed_actions, 0.5,
                         absl::Span<double>(probabilities));
  ExpectNear(probabilities[0], 0.75,
             "average strategy should remap requested action ids");
  ExpectNear(probabilities[1], 0.25,
             "average strategy should remap requested action ids");
}

void CheckFrozenLookupMatchesSlowSlabLookup() {
  StoreFixture fixture;
  fixture.storage.mutable_tables->public_state_rows.resize(1);
  fixture.storage.mutable_tables->public_state_rows[0].action_count = 2;
  const int actions[] = {10, 20};
  ActionBlock block = CreateBlock(fixture, actions);
  fixture.storage.mutable_tables->private_bucket_rows.resize(1);
  fixture.storage.mutable_tables->private_bucket_rows[0][0] = 0;

  Expect(fixture.store.prebuild_frozen_info_set_action_offsets(),
         "frozen action-offset prebuild should succeed");
  std::optional<ActionBlock> frozen_block =
      fixture.store.find_frozen(0, 0, 0, 2);

  Expect(frozen_block.has_value(), "frozen lookup should find existing block");
  Expect(frozen_block->action_offset() == block.action_offset(),
         "frozen lookup should match slab action offset");
}

}  // namespace
}  // namespace poker

int main() {
  poker::CheckRegretMatchingMissingBlockIsUniform();
  poker::CheckRegretMatchingPositiveRegretsNormalize();
  poker::CheckCfrPlusRegretClipsAtZero();
  poker::CheckAverageStrategyFallbackAndRemap();
  poker::CheckFrozenLookupMatchesSlowSlabLookup();
  return 0;
}
