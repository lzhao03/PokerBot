#include "tests/rules_test_support.h"

#include "doctest/doctest.h"
#include "src/game_state.h"

namespace poker {
namespace {

TEST_CASE("heads-up blinds define pot and call amount") {
  const ExactPublicState state = test::InitialHeadsUpState(20, 20, 1, 2);

  CHECK(state.betting.stack == std::array<Chips, kPlayerCount>{19, 18});
  CHECK(state.betting.total_committed ==
        std::array<Chips, kPlayerCount>{1, 2});
  CHECK(state.betting.street_committed ==
        std::array<Chips, kPlayerCount>{1, 2});
  CHECK(state.betting.last_full_raise == 2);
  CHECK(Pot(state.betting) == 3);
  CHECK(HighestStreetCommitment(state.betting) == 2);
  CHECK(ToCall(state.betting, 0) == 1);
  CHECK(ToCall(state.betting, 1) == 0);
  CHECK(MaxContestableAdditional(state.betting, 0) == 19);
}

}  // namespace
}  // namespace poker
