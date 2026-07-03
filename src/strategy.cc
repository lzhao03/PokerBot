#include "src/strategy.h"
#include <algorithm>
#include <numeric>
#include <utility>

namespace poker {

int Strategy::get_or_create_info_set_id(const std::string& info_set_key) {
  auto existing = info_set_ids_.find(info_set_key);
  if (existing != info_set_ids_.end()) {
    return existing->second;
  }

  const int id = static_cast<int>(info_sets_.size());
  InfoSetData data;
  data.key = info_set_key;
  info_sets_.push_back(std::move(data));
  info_set_ids_.emplace(info_sets_.back().key, id);
  return id;
}

size_t Strategy::ensure_action(InfoSetData& info_set, int action_id) {
  auto existing = std::find(info_set.action_ids.begin(),
                            info_set.action_ids.end(), action_id);
  if (existing != info_set.action_ids.end()) {
    return static_cast<size_t>(existing - info_set.action_ids.begin());
  }

  info_set.action_ids.push_back(action_id);
  info_set.probabilities.push_back(0.0);
  return info_set.action_ids.size() - 1;
}

Strategy::ActionProbabilities Strategy::get_strategy(
    const std::string& info_set_key) const {
  auto existing = info_set_ids_.find(info_set_key);
  if (existing == info_set_ids_.end()) {
    return ActionProbabilities();
  }
  const InfoSetData& info_set = info_sets_[existing->second];

  ActionProbabilities strategy;
  strategy.reserve(info_set.action_ids.size());
  for (size_t i = 0; i < info_set.action_ids.size(); ++i) {
    strategy[info_set.action_ids[i]] = info_set.probabilities[i];
  }
  return strategy;
}

Strategy::InfoSetStrategy Strategy::get_full_strategy() const {
  InfoSetStrategy strategy;
  strategy.reserve(info_sets_.size());
  for (const InfoSetData& info_set : info_sets_) {
    ActionProbabilities action_probabilities;
    action_probabilities.reserve(info_set.action_ids.size());
    for (size_t i = 0; i < info_set.action_ids.size(); ++i) {
      action_probabilities[info_set.action_ids[i]] = info_set.probabilities[i];
    }
    strategy.emplace(info_set.key, std::move(action_probabilities));
  }
  return strategy;
}

void Strategy::update(const std::string& info_set_key,
                      const ActionProbabilities& new_strategy) {
  const int info_set_id = get_or_create_info_set_id(info_set_key);
  InfoSetData& info_set = info_sets_[info_set_id];
  info_set.action_ids.clear();
  info_set.probabilities.clear();
  info_set.action_ids.reserve(new_strategy.size());
  info_set.probabilities.reserve(new_strategy.size());
  for (const auto& action_prob : new_strategy) {
    info_set.action_ids.push_back(action_prob.first);
    info_set.probabilities.push_back(action_prob.second);
  }
  normalize(info_set_key);
}

double Strategy::get_action_probability(const std::string& info_set_key,
                                        int action_id) const {
  auto existing_info_set = info_set_ids_.find(info_set_key);
  if (existing_info_set == info_set_ids_.end()) {
    return 0.0;
  }
  const InfoSetData& info_set = info_sets_[existing_info_set->second];

  auto action = std::find(info_set.action_ids.begin(),
                          info_set.action_ids.end(), action_id);
  if (action == info_set.action_ids.end()) {
    return 0.0;
  }
  const size_t index =
      static_cast<size_t>(action - info_set.action_ids.begin());
  return info_set.probabilities[index];
}

void Strategy::set_action_probability(const std::string& info_set_key,
                                      int action_id,
                                      double probability) {
  const int info_set_id = get_or_create_info_set_id(info_set_key);
  InfoSetData& info_set = info_sets_[info_set_id];
  const size_t action_index = ensure_action(info_set, action_id);
  info_set.probabilities[action_index] = probability;
}

std::vector<std::string> Strategy::get_info_sets() const {
  std::vector<std::string> info_sets;
  info_sets.reserve(info_sets_.size());
  for (const InfoSetData& info_set : info_sets_) {
    info_sets.push_back(info_set.key);
  }
  return info_sets;
}

bool Strategy::has_info_set(const std::string& info_set_key) const {
  return info_set_ids_.find(info_set_key) != info_set_ids_.end();
}

void Strategy::clear() {
  info_set_ids_.clear();
  info_sets_.clear();
}

void Strategy::normalize(const std::string& info_set_key) {
  auto existing = info_set_ids_.find(info_set_key);
  if (existing == info_set_ids_.end()) {
    return;
  }
  InfoSetData& info_set = info_sets_[existing->second];
  if (info_set.probabilities.empty()) {
    return;
  }

  double sum = std::accumulate(info_set.probabilities.begin(),
                               info_set.probabilities.end(), 0.0);
  if (sum > 0.0) {
    for (double& probability : info_set.probabilities) {
      probability /= sum;
    }
  } else {
    const double uniform_prob = 1.0 / info_set.probabilities.size();
    for (double& probability : info_set.probabilities) {
      probability = uniform_prob;
    }
  }
}

} // namespace poker
