#include "src/strategy_store.h"

#include "doctest/doctest.h"

#include <memory>
#include <optional>

namespace poker {
namespace {

struct StoreFixture {
  SolverConfig config;
  SolverStorage storage;
  TraversalStats stats;
  StrategyStore store{config, storage, &stats};
};

ActionBlock CreateBlock(StoreFixture& fixture, absl::Span<const int> actions) {
  std::optional<ActionBlock> block =
      fixture.store.get_or_create({0, 0, 0}, actions);
  REQUIRE(block.has_value());
  return *block;
}

TEST_CASE("missing action block regret matching is uniform") {
  StoreFixture fixture;
  double probabilities[3] = {};
  fixture.store.regret_matching_or_uniform(
      std::nullopt, 3, RegretLoadMode::kPlain,
      absl::Span<double>(probabilities));

  CHECK(probabilities[0] == doctest::Approx(1.0 / 3.0));
  CHECK(probabilities[1] == doctest::Approx(1.0 / 3.0));
  CHECK(probabilities[2] == doctest::Approx(1.0 / 3.0));
}

TEST_CASE("positive regrets normalize into action probabilities") {
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

  CHECK(probabilities[0] == doctest::Approx(0.0));
  CHECK(probabilities[1] == doctest::Approx(0.25));
  CHECK(probabilities[2] == doctest::Approx(0.75));
}

TEST_CASE("CFR plus regret update clips at zero") {
  StoreFixture fixture;
  const int actions[] = {10, 20};
  ActionBlock block = CreateBlock(fixture, actions);
  fixture.storage.cumulative->cumulative_regrets[block.action_offset()] = 1.0f;

  block.add_cfr_plus_regret(
      0, -3.0f, RegretUpdateOptions{RegretUpdateMode::kPlain, false});

  CHECK(fixture.storage.cumulative->cumulative_regrets[block.action_offset()] ==
        doctest::Approx(0.0));
}

TEST_CASE("average strategy falls back to uniform and remaps action ids") {
  StoreFixture fixture;
  const int actions[] = {10, 20};
  ActionBlock block = CreateBlock(fixture, actions);

  double probabilities[2] = {};
  block.average_strategy(false, actions, 0.5,
                         absl::Span<double>(probabilities));
  CHECK(probabilities[0] == doctest::Approx(0.5));
  CHECK(probabilities[1] == doctest::Approx(0.5));

  const size_t offset = block.action_offset();
  fixture.storage.cumulative->cumulative_strategies[offset] = 1.0f;
  fixture.storage.cumulative->cumulative_strategies[offset + 1] = 3.0f;
  const int reversed_actions[] = {20, 10};
  block.average_strategy(false, reversed_actions, 0.5,
                         absl::Span<double>(probabilities));
  CHECK(probabilities[0] == doctest::Approx(0.75));
  CHECK(probabilities[1] == doctest::Approx(0.25));
}

TEST_CASE("frozen lookup matches slow slab lookup") {
  StoreFixture fixture;
  fixture.storage.mutable_tables->public_state_rows.resize(1);
  fixture.storage.mutable_tables->public_state_rows[0].action_count = 2;
  const int actions[] = {10, 20};
  ActionBlock block = CreateBlock(fixture, actions);

  REQUIRE(fixture.store.prebuild_frozen_info_set_action_offsets());
  std::optional<ActionBlock> frozen_block =
      fixture.store.find_frozen(0, 0, 0, 2);

  REQUIRE(frozen_block.has_value());
  CHECK(frozen_block->action_offset() == block.action_offset());
}

}  // namespace
}  // namespace poker
