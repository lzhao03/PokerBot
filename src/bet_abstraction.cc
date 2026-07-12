#include "src/bet_abstraction.h"

#include <algorithm>
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
  const Chips all_in_to =
      current_to + MaxContestableAdditional(data, state.actor);
  AbstractActions actions;
  if (to_call > 0) {
    actions.emplace_back(ActionKind::Fold, 0);
    actions.emplace_back(ActionKind::Call, call_to);
  } else {
    actions.emplace_back(ActionKind::Check, 0);
  }

  const ActionKind kind = highest_to > 0 ? ActionKind::Raise : ActionKind::Bet;
  const auto& fractions =
      config.pot_fractions[std::to_underlying(data.street)];
  const Chips pot_after_call = Pot(data) + to_call;
  for (double fraction : fractions) {
    const Chips raise_by =
        static_cast<Chips>(std::ceil(fraction * pot_after_call));
    const Chips target = highest_to + raise_by;
    if (IsLegalAction(state, {kind, target}) &&
        actions.back().target_street_commitment != target) {
      actions.emplace_back(kind, target);
    }
  }

  if (IsLegalAction(state, {ActionKind::AllIn, all_in_to})) {
    actions.emplace_back(ActionKind::AllIn, all_in_to);
  }

  return actions;
}

}  // namespace poker
