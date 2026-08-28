#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/bet_abstraction.h"
#include "src/evaluation.h"
#include "src/neural_evaluation.h"
#include "src/neural_policy.h"
#include "src/policy_codec.h"
#include "src/solver.h"

ABSL_FLAG(std::string, tabular, "", "tabular .policy file");
ABSL_FLAG(std::string, deep, "", "Deep CFR .pt file");
ABSL_FLAG(std::string, distilled, "", "distilled .pt file");
ABSL_FLAG(std::string, private_recall, "current",
          "private recall shared by all models: current or history");

namespace {

constexpr uint64_t kEvaluationSamples = 100'000;
constexpr uint64_t kSeed = 1;

absl::StatusOr<poker::SolveSpec> ComparisonSpec() {
  poker::SolverConfig config;
  config.bet_abstraction = poker::SmallBettingConfig();
  config.card_abstraction.public_mode =
      poker::PublicCardMode::CompactTexture;
  config.card_abstraction.private_kind =
      poker::PrivateAbstractionKind::Handcrafted36;
  const std::string recall = absl::GetFlag(FLAGS_private_recall);
  if (recall == "current") {
    config.card_abstraction.recall_mode =
        poker::RecallMode::CurrentBucketOnly;
  } else if (recall == "history") {
    config.card_abstraction.recall_mode = poker::RecallMode::BucketHistory;
  } else {
    return absl::InvalidArgumentError(
        "private_recall must be current or history");
  }
  const poker::ComboRange range = poker::UniformComboRange();
  return poker::SolveSpec{
      config,
      poker::MakeInitialState(config.betting_rules, {200, 200}, {1, 2}),
      {range, range},
  };
}

struct Candidate {
  std::string_view name;
  std::filesystem::path path;
  poker::StrategyLookup strategy;
};

}  // namespace

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage(
      "Compare matching 100BB compact-H36 small-betting policies.");
  absl::ParseCommandLine(argc, argv);
  const std::filesystem::path tabular_path =
      absl::GetFlag(FLAGS_tabular);
  const std::filesystem::path deep_path = absl::GetFlag(FLAGS_deep);
  const std::filesystem::path distilled_path =
      absl::GetFlag(FLAGS_distilled);
  if (tabular_path.empty() || deep_path.empty() || distilled_path.empty()) {
    std::cerr << "Error: --tabular, --deep, and --distilled are required\n";
    return 1;
  }

  auto spec = ComparisonSpec();
  if (!spec.ok()) {
    std::cerr << "Error: " << spec.status() << '\n';
    return 1;
  }
  const poker::SolverConfig& config = spec->config;
  auto deals = poker::DealDistribution::Create(
      spec->ranges[poker::Index(poker::Player::A)],
      spec->ranges[poker::Index(poker::Player::B)]);
  if (!deals.ok()) {
    std::cerr << "Error: " << deals.status() << '\n';
    return 1;
  }
  const poker::HistoryTree history = poker::BuildHistoryTree(
      spec->root.betting, config.betting_rules, config.bet_abstraction);
  const poker::PublicPosition initial_public(
      config.card_abstraction, spec->root.board);
  const poker::ModelFingerprint model = poker::ModelFingerprintFor(
      config, spec->root, spec->ranges);
  auto tabular = poker::LoadPolicy(tabular_path);
  if (!tabular.ok()) {
    std::cerr << "Error loading --tabular: " << tabular.status() << '\n';
    return 1;
  }
  if (tabular->model != model) {
    std::cerr << "Error: tabular model does not match the comparison game\n";
    return 1;
  }
  auto deep = poker::LoadNeuralPolicy(deep_path, model);
  if (!deep.ok()) {
    std::cerr << "Error loading --deep: " << deep.status() << '\n';
    return 1;
  }
  auto distilled = poker::LoadNeuralPolicy(distilled_path, model);
  if (!distilled.ok()) {
    std::cerr << "Error loading --distilled: " << distilled.status() << '\n';
    return 1;
  }

  const std::array<Candidate, 3> candidates = {{
      {"tabular", tabular_path, poker::MakeStrategyLookup(*tabular)},
      {"deep", deep_path,
       poker::MakeStrategyLookup(
           history, config.card_abstraction, model, *deep)},
      {"distilled", distilled_path,
       poker::MakeStrategyLookup(
           history, config.card_abstraction, model, *distilled)},
  }};

  std::cout << std::setprecision(8)
            << "model_fingerprint\t" << std::to_underlying(model)
            << '\n'
            << "evaluation_samples\t" << kEvaluationSamples << '\n'
            << "best_response_info_set_cap_per_player\t"
            << config.max_info_sets << '\n'
            << "policy\tbytes\tapprox_exploitability\tstandard_error"
               "\tresponse_rows_total\topponent_misses\tresponse_misses\n";
  for (const Candidate& candidate : candidates) {
    const auto estimate =
        poker::EstimateExploitability(
            config, *deals, history, initial_public, model,
            candidate.strategy);
    if (!estimate.ok()) {
      std::cerr << "Error evaluating " << candidate.name << ": "
                << estimate.status() << '\n';
      return 1;
    }
    const auto& player_a = estimate->player_a_response;
    const auto& player_b = estimate->player_b_response;
    const double standard_error = 0.5 * std::hypot(
        player_a.standard_error, player_b.standard_error);
    std::cout << candidate.name << '\t'
              << std::filesystem::file_size(candidate.path) << '\t'
              << estimate->exploitability << '\t' << standard_error << '\t'
              << player_a.response_policy.rows.size() +
                     player_b.response_policy.rows.size()
              << '\t'
              << player_a.missing_opponent_lookups +
                     player_b.missing_opponent_lookups
              << '\t'
              << player_a.missing_response_lookups +
                     player_b.missing_response_lookups
              << '\n';
  }

  std::cout << "left\tright\tleft_as_a\tleft_as_a_se\tleft_as_b"
               "\tleft_as_b_se\tleft_average\tleft_average_se"
               "\tmissing_lookups\n";
  for (size_t left = 0; left < candidates.size(); ++left) {
    for (size_t right = left + 1; right < candidates.size(); ++right) {
      const auto as_a = poker::EstimateExpectedValue(
          config, *deals, history, initial_public, candidates[left].strategy,
          candidates[right].strategy, kEvaluationSamples, kSeed, false, true);
      const auto as_b = poker::EstimateExpectedValue(
          config, *deals, history, initial_public, candidates[right].strategy,
          candidates[left].strategy, kEvaluationSamples, kSeed, false, true);
      if (!as_a.ok() || !as_b.ok()) {
        std::cerr << "Error evaluating " << candidates[left].name << " vs "
                  << candidates[right].name << ": "
                  << (!as_a.ok() ? as_a.status() : as_b.status()) << '\n';
        return 1;
      }
      const double average = 0.5 * (as_a->mean - as_b->mean);
      const double standard_error =
          0.5 * std::hypot(as_a->standard_error, as_b->standard_error);
      std::cout << candidates[left].name << '\t'
                << candidates[right].name << '\t' << as_a->mean << '\t'
                << as_a->standard_error << '\t' << -as_b->mean << '\t'
                << as_b->standard_error << '\t' << average << '\t'
                << standard_error << '\t'
                << as_a->missing_policy_lookups +
                       as_b->missing_policy_lookups
                << '\n';
    }
  }
  return 0;
}
