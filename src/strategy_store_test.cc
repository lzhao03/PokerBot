#include "src/strategy_store.h"

#include "doctest/doctest.h"

#include <memory>
#include <optional>

namespace poker {
namespace {

struct StoreFixture {
  StoreFixture() {
    storage.mutable_tables->nodes.resize(1);
    storage.mutable_tables->betting_nodes.resize(1);
    storage.mutable_tables->nodes[0].betting_node_id = 0;
    storage.mutable_tables->betting_nodes[0].kind =
        StrategyTables::NodeKind::kDecision;
    storage.mutable_tables->betting_nodes[0].state.player_to_act = 0;
  }

  SolverConfig config;
  SolverStorage storage;
  TraversalStats stats;
  StrategyStore store{config, storage, &stats};
};

ActionBlock CreateBlock(StoreFixture& fixture, size_t action_count) {
  fixture.storage.mutable_tables->betting_nodes[0].action_count =
      static_cast<uint8_t>(action_count);
  std::optional<ActionBlock> block =
      fixture.store.get_or_create({0, 0}, action_count);
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
  ActionBlock block = CreateBlock(fixture, 3);
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
  ActionBlock block = CreateBlock(fixture, 2);
  fixture.storage.cumulative->cumulative_regrets[block.action_offset()] = 1.0f;

  block.add_cfr_plus_regret(
      0, -3.0f, RegretUpdateOptions{RegretUpdateMode::kPlain, false});

  CHECK(fixture.storage.cumulative->cumulative_regrets[block.action_offset()] ==
        doctest::Approx(0.0));
}

TEST_CASE("average strategy falls back to uniform and normalizes mass") {
  StoreFixture fixture;
  ActionBlock block = CreateBlock(fixture, 2);

  double probabilities[2] = {};
  block.average_strategy(false, 0.5, absl::Span<double>(probabilities));
  CHECK(probabilities[0] == doctest::Approx(0.5));
  CHECK(probabilities[1] == doctest::Approx(0.5));

  const size_t offset = block.action_offset();
  fixture.storage.cumulative->cumulative_strategies[offset] = 1.0f;
  fixture.storage.cumulative->cumulative_strategies[offset + 1] = 3.0f;
  block.average_strategy(false, 0.5, absl::Span<double>(probabilities));
  CHECK(probabilities[0] == doctest::Approx(0.25));
  CHECK(probabilities[1] == doctest::Approx(0.75));
}

TEST_CASE("frozen lookup matches slow slab lookup") {
  StoreFixture fixture;
  auto& tables = *fixture.storage.mutable_tables;
  tables.betting_nodes[0].action_count = 2;
  ActionBlock block = CreateBlock(fixture, 2);

  REQUIRE(fixture.store.prebuild_frozen_info_set_action_offsets());
  std::optional<ActionBlock> frozen_block =
      fixture.store.find_frozen(0, 0, 2);

  REQUIRE(frozen_block.has_value());
  CHECK(frozen_block->action_offset() == block.action_offset());
}

}  // namespace
}  // namespace poker
