#include "src/card_abstraction.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"

ABSL_FLAG(std::string, output, "",
          "output model path");
ABSL_FLAG(uint64_t, fit_seed, 1, "raw state sampling seed");
ABSL_FLAG(uint64_t, rollout_seed, 0x4551554954595631ULL,
          "deterministic equity rollout seed");
ABSL_FLAG(uint32_t, samples, 100000, "raw deals sampled per street");
ABSL_FLAG(uint32_t, opponent_samples, 16,
          "opponent hands sampled per future runout");
ABSL_FLAG(uint32_t, runout_samples, 8,
          "future runouts sampled on flop and turn");

namespace poker {
namespace {

uint64_t Mix64(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

size_t BoundedIndex(uint64_t& state, size_t bound) {
  const uint64_t unsigned_bound = static_cast<uint64_t>(bound);
  const uint64_t threshold = -unsigned_bound % unsigned_bound;
  uint64_t value;
  do {
    state = Mix64(state);
    value = state;
  } while (value < threshold);
  return static_cast<size_t>(value % unsigned_bound);
}

std::array<Card, 7> SampleCards(uint64_t& state) {
  std::array<Card, kDeckCardCount> deck = kDeck;
  std::array<Card, 7> cards;
  for (size_t index = 0; index < cards.size(); ++index) {
    const size_t selected =
        index + BoundedIndex(state, deck.size() - index);
    std::swap(deck[index], deck[selected]);
    cards[index] = deck[index];
  }
  return cards;
}

std::vector<float> QuantileCutoffs(std::vector<float> values,
                                   size_t buckets) {
  std::sort(values.begin(), values.end());
  std::vector<float> cutoffs;
  cutoffs.reserve(buckets - 1);
  for (size_t bucket = 1; bucket < buckets; ++bucket) {
    const size_t index = bucket * values.size() / buckets;
    const float lower = values[index - 1];
    const float upper = values[index];
    cutoffs.push_back(lower == upper ? lower : std::midpoint(lower, upper));
  }
  return cutoffs;
}

size_t Quantile(float value, const std::vector<float>& cutoffs) {
  return static_cast<size_t>(
      std::upper_bound(cutoffs.begin(), cutoffs.end(), value) -
      cutoffs.begin());
}

std::vector<float> OuterMedians(
    const std::vector<EquityFeatures>& samples,
    const std::vector<float>& cutoffs) {
  std::vector<std::vector<float>> groups(cutoffs.size() + 1);
  for (EquityFeatures features : samples) {
    groups[Quantile(features.ehs2, cutoffs)].push_back(features.ehs);
  }
  std::vector<float> medians;
  medians.reserve(groups.size());
  for (auto& group : groups) {
    if (group.empty()) return {};
    const size_t middle = group.size() / 2;
    std::nth_element(group.begin(), group.begin() + middle, group.end());
    medians.push_back(group[middle]);
  }
  return medians;
}

bool PrintStats(StreetKind street,
                const std::vector<EquityFeatures>& samples,
                const EquityBucketModel& model) {
  const size_t buckets = street == StreetKind::River ? 64 : 32;
  struct Moments {
    uint64_t count = 0;
    double ehs = 0.0;
    double ehs_square = 0.0;
    double ehs2 = 0.0;
    double ehs2_square = 0.0;
  };
  std::vector<Moments> moments(buckets);
  for (EquityFeatures features : samples) {
    Moments& item = moments[EquityBucket(street, features, model)];
    ++item.count;
    item.ehs += features.ehs;
    item.ehs_square += static_cast<double>(features.ehs) * features.ehs;
    item.ehs2 += features.ehs2;
    item.ehs2_square += static_cast<double>(features.ehs2) * features.ehs2;
  }
  bool balanced = true;
  const double ideal = static_cast<double>(samples.size()) / buckets;
  for (size_t bucket = 0; bucket < moments.size(); ++bucket) {
    const Moments& item = moments[bucket];
    if (item.count < 0.5 * ideal || item.count > 2.0 * ideal) {
      balanced = false;
    }
    if (item.count == 0) {
      std::cout << static_cast<int>(street) << '\t' << bucket
                << "\t0\t0\t0\n";
      continue;
    }
    const double count = static_cast<double>(item.count);
    const double ehs_variance =
        item.ehs_square / count - std::pow(item.ehs / count, 2);
    const double ehs2_variance =
        item.ehs2_square / count - std::pow(item.ehs2 / count, 2);
    std::cout << static_cast<int>(street) << '\t' << bucket << '\t'
              << item.count << '\t' << ehs_variance << '\t'
              << ehs2_variance << '\n';
  }
  return balanced;
}

int Run() {
  EquityBucketModel model;
  model.rollout_seed = absl::GetFlag(FLAGS_rollout_seed);
  model.fit_seed = absl::GetFlag(FLAGS_fit_seed);
  model.training_samples = absl::GetFlag(FLAGS_samples);
  model.opponent_samples = absl::GetFlag(FLAGS_opponent_samples);
  model.runout_samples = absl::GetFlag(FLAGS_runout_samples);
  if (absl::GetFlag(FLAGS_output).empty()) {
    std::cerr << "Error: --output is required\n";
    return 1;
  }
  if (model.training_samples < 64) {
    std::cerr << "Error: --samples must be at least 64\n";
    return 1;
  }

  std::array<std::vector<EquityFeatures>, 4> features;
  for (StreetKind street : {StreetKind::Flop, StreetKind::Turn,
                            StreetKind::River}) {
    features[static_cast<size_t>(street)].reserve(model.training_samples);
  }
  uint64_t state = model.fit_seed;
  for (uint32_t sample = 0; sample < model.training_samples; ++sample) {
    const auto cards = SampleCards(state);
    const ComboId hand = CardsToComboId(cards[0], cards[1]);
    const FlopBoard flop =
        DealFlop(PreflopBoard{}, {cards[2], cards[3], cards[4]});
    const TurnBoard turn = DealTurn(flop, cards[5]);
    const RiverBoard river = DealRiver(turn, cards[6]);
    features[static_cast<size_t>(StreetKind::Flop)].push_back(
        EvaluateEquityFeatures(hand, Board{flop}, model));
    features[static_cast<size_t>(StreetKind::Turn)].push_back(
        EvaluateEquityFeatures(hand, Board{turn}, model));
    features[static_cast<size_t>(StreetKind::River)].push_back(
        EvaluateEquityFeatures(hand, Board{river}, model));
  }

  for (StreetKind street : {StreetKind::Flop, StreetKind::Turn}) {
    const size_t index = static_cast<size_t>(street);
    std::vector<float> values;
    values.reserve(features[index].size());
    for (EquityFeatures item : features[index]) values.push_back(item.ehs2);
    model.ehs2_cutoffs[index] = QuantileCutoffs(std::move(values), 16);
    model.ehs_medians[index] =
        OuterMedians(features[index], model.ehs2_cutoffs[index]);
    if (model.ehs_medians[index].size() != 16) {
      std::cerr << "Error: tied features produced an empty outer quantile\n";
      return 1;
    }
  }
  std::vector<float> river_values;
  river_values.reserve(features[static_cast<size_t>(StreetKind::River)].size());
  for (EquityFeatures item :
       features[static_cast<size_t>(StreetKind::River)]) {
    river_values.push_back(item.ehs);
  }
  model.river_equity_cutoffs =
      QuantileCutoffs(std::move(river_values), 64);

  auto finalized = FinalizeEquityBucketModel(std::move(model));
  if (!finalized.ok()) {
    std::cerr << "Error: " << finalized.status() << '\n';
    return 1;
  }
  std::cout << "street\tbucket\tcount\tehs_variance\tehs2_variance\n";
  bool balanced = true;
  for (StreetKind street : {StreetKind::Flop, StreetKind::Turn,
                            StreetKind::River}) {
    balanced &= PrintStats(
        street, features[static_cast<size_t>(street)], *finalized);
  }
  if (!balanced) {
    std::cerr << "Error: fitted bucket occupancy is unbalanced\n";
    return 1;
  }
  const absl::Status saved = SaveEquityBucketModel(
      *finalized, absl::GetFlag(FLAGS_output));
  if (!saved.ok()) {
    std::cerr << "Error: " << saved << '\n';
    return 1;
  }
  return 0;
}

}  // namespace
}  // namespace poker

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage("Build an equity-potential abstraction model.");
  absl::ParseCommandLine(argc, argv);
  return poker::Run();
}
