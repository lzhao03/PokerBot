#include "src/history.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace poker {
namespace {

HistoryId AppendHistory(HistoryTree& tree, const BettingState& state, const BettingRules& rules,
                        const BetAbstractionConfig& abstraction) {
  const HistoryId id{static_cast<uint32_t>(tree.nodes.size())};
  if (const auto* decision = std::get_if<DecisionState>(&state)) {
    const AbstractActions actions = SelectAbstractActions(abstraction, *decision);
    const uint32_t begin = static_cast<uint32_t>(tree.children.size());
    tree.children.resize(begin + actions.size(), id);
    tree.nodes.push_back({state, begin, static_cast<uint8_t>(actions.size())});
    for (size_t index = 0; index < actions.size(); ++index) {
      const auto child = ApplyAction(*decision, actions[index]);
      assert(child.ok());
      tree.children[begin + index] = AppendHistory(tree, *child, rules, abstraction);
    }
  } else if (const auto* chance = std::get_if<ChanceState>(&state)) {
    const uint32_t begin = static_cast<uint32_t>(tree.children.size());
    tree.children.push_back(id);
    tree.nodes.push_back({state, begin, 1});
    tree.children[begin] =
        AppendHistory(tree, AdvanceBettingStreet(*chance, rules), rules, abstraction);
  } else {
    tree.nodes.push_back({state, static_cast<uint32_t>(tree.children.size()), 0});
  }
  return id;
}

}  // namespace

HistoryTree BuildHistoryTree(const BettingState& root, const BettingRules& rules,
                             const BetAbstractionConfig& abstraction) {
  HistoryTree tree;
  tree.nodes.reserve(4096);
  tree.children.reserve(4096);
  AppendHistory(tree, root, rules, abstraction);
  return tree;
}

}  // namespace poker
