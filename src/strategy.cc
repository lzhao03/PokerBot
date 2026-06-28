#include "src/strategy.h"
#include <numeric>
#include <algorithm>

namespace poker {

void Strategy::normalize(const std::string& info_set_key) {
  auto it = strategy_.find(info_set_key);
  if (it != strategy_.end()) {
    double sum = 0.0;
    
    // Calculate sum of probabilities
    for (const auto& action_prob : it->second) {
      sum += action_prob.second;
    }
    
    // Normalize if sum is greater than 0
    if (sum > 0.0) {
      for (auto& action_prob : it->second) {
        action_prob.second /= sum;
      }
    } else {
      // If all probabilities are 0, set uniform strategy
      double uniform_prob = 1.0 / it->second.size();
      for (auto& action_prob : it->second) {
        action_prob.second = uniform_prob;
      }
    }
  }
}

} // namespace poker
