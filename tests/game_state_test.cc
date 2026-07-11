#include "tests/rules_test_support.h"

#include "doctest/doctest.h"
#include "src/poker.h"

namespace poker {
namespace {

TEST_CASE("heads-up blinds define pot and call amount") {
  const ExactPublicState state = test::InitialHeadsUpState(20, 20, 1, 2);
  const BettingData& betting = Data(state.betting);

  CHECK(betting.stack == std::array<Chips, kPlayerCount>{19, 18});
  CHECK(betting.total_committed ==
        std::array<Chips, kPlayerCount>{1, 2});
  CHECK(betting.street_committed ==
        std::array<Chips, kPlayerCount>{1, 2});
  CHECK(betting.last_full_raise == 2);
  CHECK(Pot(betting) == 3);
  CHECK(HighestStreetCommitment(betting) == 2);
  CHECK(ToCall(betting, Player::kA) == 1);
  CHECK(ToCall(betting, Player::kB) == 0);
  CHECK(MaxContestableAdditional(betting, Player::kA) == 19);
}

}  // namespace
}  // namespace poker
