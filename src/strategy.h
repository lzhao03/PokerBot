#pragma once

#include <cstddef>
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
  ActionProbabilities get_strategy(const std::string& info_set_key) const;
  
  InfoSetStrategy get_full_strategy() const;
  
  void update(const std::string& info_set_key,
              const ActionProbabilities& new_strategy);
  
  // Action probability operations
  double get_action_probability(const std::string& info_set_key,
                                int action_id) const;
  
  void set_action_probability(const std::string& info_set_key,
                              int action_id,
                              double probability);
  
  // Information set operations
  std::vector<std::string> get_info_sets() const;
  
  bool has_info_set(const std::string& info_set_key) const;
  bool empty() const;
  
  void clear();
  
  // Normalize the probabilities in an information set to sum to 1
  void normalize(const std::string& info_set_key);

private:
  struct InfoSetData {
    std::string key;
    std::vector<int> action_ids;
    std::vector<double> probabilities;
  };

  int get_or_create_info_set_id(const std::string& info_set_key);
  size_t ensure_action(InfoSetData& info_set, int action_id);

  std::unordered_map<std::string, int> info_set_ids_;
  std::vector<InfoSetData> info_sets_;
};

} // namespace poker
