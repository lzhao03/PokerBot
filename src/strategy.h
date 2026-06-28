#pragma once

#include <unordered_map>
#include <string>
#include <vector>
#include "src/poker.pb.h"

namespace poker {

class Strategy {
public:
  // Probability distribution over actions at each information set
  using ActionProbabilities = std::unordered_map<int, double>; // Action ID -> Probability
  using InfoSetStrategy = std::unordered_map<std::string, ActionProbabilities>;
  
  Strategy() = default;
  
  // Core strategy operations
  ActionProbabilities get_strategy(const std::string& info_set_key) const {
    auto it = strategy_.find(info_set_key);
    return (it != strategy_.end()) ? it->second : ActionProbabilities();
  }
  
  const InfoSetStrategy& get_full_strategy() const { return strategy_; }
  
  void update(const std::string& info_set_key, const ActionProbabilities& new_strategy) {
    strategy_[info_set_key] = new_strategy;
    normalize(info_set_key);
  }
  
  // Action probability operations
  double get_action_probability(const std::string& info_set_key, int action_id) const {
    auto it = strategy_.find(info_set_key);
    if (it != strategy_.end()) {
      auto action_it = it->second.find(action_id);
      return (action_it != it->second.end()) ? action_it->second : 0.0;
    }
    return 0.0;
  }
  
  void set_action_probability(const std::string& info_set_key, int action_id, double probability) {
    strategy_[info_set_key][action_id] = probability;
  }
  
  // Information set operations
  std::vector<std::string> get_info_sets() const {
    std::vector<std::string> info_sets;
    info_sets.reserve(strategy_.size());
    for (const auto& pair : strategy_) {
      info_sets.push_back(pair.first);
    }
    return info_sets;
  }
  
  bool has_info_set(const std::string& info_set_key) const {
    return strategy_.find(info_set_key) != strategy_.end();
  }
  
  void clear() { strategy_.clear(); }
  
  // Normalize the probabilities in an information set to sum to 1
  void normalize(const std::string& info_set_key);

private:
  InfoSetStrategy strategy_;
};

} // namespace poker
