#include "src/bet_abstraction.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <utility>

namespace poker {

AbstractActions SelectAbstractActions(const BetAbstractionConfig& config,
                                      const DecisionState& state) {
  const BettingData& data = state.data;
  const size_t player = Index(state.actor);
  const Chips current_to = data.street_committed[player];
  const Chips highest_to = CurrentWager(data);
  const Chips to_call = highest_to - current_to;
  const Chips call_to = std::min(highest_to, current_to + data.stack[player]);
  const Chips all_in_to = current_to + MaxContestableAdditional(data, state.actor);
  const Chips min_full_raise_to = (highest_to > 0 ? highest_to : current_to) + data.last_full_raise;
  AbstractActions actions;
  if (to_call > 0) {
    actions.push_back({ActionKind::kFold, 0});
    actions.push_back({ActionKind::kCall, call_to});
  } else {
    actions.push_back({ActionKind::kCheck, 0});
  }

  const ActionKind kind = highest_to > 0 ? ActionKind::kRaise : ActionKind::kBet;
  const auto& fractions = config.pot_fractions[std::to_underlying(data.street)];
  const Chips pot_after_call = Pot(data) + to_call;
  for (double fraction : fractions) {
    const Chips raise_by = std::max(
        Chips{1},
        static_cast<Chips>(std::llround(fraction * pot_after_call)));
    const Chips target = highest_to + raise_by;
    if (target >= min_full_raise_to && target < all_in_to) {
      actions.push_back({kind, target});
    }
  }

  std::sort(actions.begin() + (to_call > 0 ? 2 : 1), actions.end(),
            [](const GameAction& left, const GameAction& right) {
              return left.target_street_commitment <
                     right.target_street_commitment;
            });
  actions.erase(std::unique(actions.begin(), actions.end()), actions.end());
  if (all_in_to > call_to) {
    actions.push_back({ActionKind::kAllIn, all_in_to});
  }

  for (const GameAction& action : actions) {
    assert(IsLegalAction(state, action));
  }
  return actions;
}

}  // namespace poker
