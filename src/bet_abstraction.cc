#include "src/bet_abstraction.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace poker {

AbstractActions SelectAbstractActions(const BetAbstractionConfig& config,
                                      const DecisionState& state) {
  const LegalActionSpace legal = LegalActions(state);
  AbstractActions actions;
  if (legal.facing_action()) {
    actions.push_back({ActionKind::kFold, 0});
    actions.push_back({ActionKind::kCall, legal.call_to});
  } else {
    actions.push_back({ActionKind::kCheck, 0});
  }

  const ActionKind kind =
      legal.wager_open() ? ActionKind::kRaise : ActionKind::kBet;
  const auto& fractions =
      config.pot_fractions[static_cast<size_t>(state.data.street)];
  const Chips pot_after_call = Pot(state.data) + legal.to_call();
  for (double fraction : fractions) {
    const Chips raise_by = std::max(
        Chips{1},
        static_cast<Chips>(std::llround(fraction * pot_after_call)));
    const Chips target = legal.highest_to + raise_by;
    if (target >= legal.min_full_raise_to && target < legal.all_in_to) {
      actions.push_back({kind, target});
    }
  }

  std::sort(actions.begin() + (legal.facing_action() ? 2 : 1), actions.end(),
            [](const GameAction& left, const GameAction& right) {
              return left.target_street_commitment <
                     right.target_street_commitment;
            });
  actions.erase(std::unique(actions.begin(), actions.end()), actions.end());
  if (legal.can_aggress()) {
    actions.push_back({ActionKind::kAllIn, legal.all_in_to});
  }

  for (const GameAction& action : actions) {
    assert(IsLegalAction(state, action));
  }
  return actions;
}

}  // namespace poker
