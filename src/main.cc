#include "src/solver.h"

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/strings/numbers.h"

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/resource.h>
#include <utility>
#include <vector>

ABSL_FLAG(int, iterations, 100, "CFR iterations");
ABSL_FLAG(int, starting_stack, 100, "starting stack in chips");
ABSL_FLAG(int, small_blind, 1, "small blind in chips");
ABSL_FLAG(int, big_blind, 2, "big blind in chips");
ABSL_FLAG(int, chance_samples, 1, "chance samples per chance node");
ABSL_FLAG(int, max_info_sets, 500000, "maximum infosets");
ABSL_FLAG(int64_t, max_memory_mb, 4096,
          "hard memory limit in MB; 0 is unlimited");
ABSL_FLAG(bool, accumulate_average_strategy, true,
          "store the average strategy");
ABSL_FLAG(bool, log, false, "show INFO logs and VLOG(1) progress");
ABSL_FLAG(std::string, private_abstraction, "handcrafted36",
          "exact, handcrafted36, or equity");
ABSL_FLAG(std::string, private_recall, "auto",
          "auto, current, or history");
ABSL_FLAG(std::string, equity_model, "",
          "equity abstraction model path");
ABSL_FLAG(std::vector<std::string>, pot_fractions,
          std::vector<std::string>({"0.25", "0.5", "1.0"}),
          "pot fractions after calling for every street");
ABSL_FLAG(std::vector<std::string>, preflop_pot_fractions, {},
          "preflop pot fractions after calling");
ABSL_FLAG(std::vector<std::string>, flop_pot_fractions, {},
          "flop pot fractions after calling");
ABSL_FLAG(std::vector<std::string>, turn_pot_fractions, {},
          "turn pot fractions after calling");
ABSL_FLAG(std::vector<std::string>, river_pot_fractions, {},
          "river pot fractions after calling");

namespace {

void SetMemoryLimit(int64_t megabytes) {
  if (megabytes <= 0) {
    return;
  }
  const rlim_t bytes = static_cast<rlim_t>(megabytes) * 1024ULL * 1024ULL;
  struct rlimit limit;
  limit.rlim_cur = bytes;
  limit.rlim_max = bytes;
  if (setrlimit(RLIMIT_AS, &limit) != 0) {
    std::cerr << "Warning: failed to set memory limit to " << megabytes
              << " MB: " << std::strerror(errno) << "\n";
  }
}

absl::StatusOr<std::vector<double>> ParsePotFractions(
    const std::vector<std::string>& values) {
  std::vector<double> sizes;
  sizes.reserve(values.size());
  for (const std::string& value : values) {
    double size = 0.0;
    if (!absl::SimpleAtod(value, &size)) {
      return absl::InvalidArgumentError("invalid pot fraction: " + value);
    }
    sizes.push_back(size);
  }
  return sizes;
}

absl::Status OverridePotFractions(poker::SolverConfigOptions& config,
                                  poker::StreetKind street,
                                  const std::vector<std::string>& values) {
  if (!values.empty()) {
    const auto fractions = ParsePotFractions(values);
    if (!fractions.ok()) {
      return fractions.status();
    }
    config.bet_abstraction.pot_fractions[std::to_underlying(street)] =
        *fractions;
  }
  return absl::OkStatus();
}

absl::StatusOr<poker::SolverConfig> ConfigFromFlags() {
  poker::SolverConfigOptions config;
  config.starting_stack = absl::GetFlag(FLAGS_starting_stack);
  config.small_blind = absl::GetFlag(FLAGS_small_blind);
  config.big_blind = absl::GetFlag(FLAGS_big_blind);
  config.chance_samples = absl::GetFlag(FLAGS_chance_samples);
  config.max_info_sets = absl::GetFlag(FLAGS_max_info_sets);
  config.accumulate_average_strategy =
      absl::GetFlag(FLAGS_accumulate_average_strategy);

  const std::string private_abstraction =
      absl::GetFlag(FLAGS_private_abstraction);
  if (private_abstraction == "exact") {
    config.card_abstraction.private_kind =
        poker::PrivateAbstractionKind::ExactCanonical;
  } else if (private_abstraction == "handcrafted36") {
    config.card_abstraction.private_kind =
        poker::PrivateAbstractionKind::Handcrafted36;
  } else if (private_abstraction == "equity") {
    config.card_abstraction.private_kind =
        poker::PrivateAbstractionKind::EquityPotential;
    const std::string path = absl::GetFlag(FLAGS_equity_model);
    if (path.empty()) {
      return absl::InvalidArgumentError(
          "--equity_model is required for equity abstraction");
    }
    auto model = poker::LoadEquityBucketModel(path);
    if (!model.ok()) return model.status();
    config.card_abstraction.equity_model = std::move(*model);
  } else {
    return absl::InvalidArgumentError("invalid private abstraction");
  }

  const std::string recall = absl::GetFlag(FLAGS_private_recall);
  if (recall == "auto") {
    config.card_abstraction.recall_mode =
        config.card_abstraction.private_kind ==
                poker::PrivateAbstractionKind::Handcrafted36
            ? poker::RecallMode::BucketHistory
            : poker::RecallMode::CurrentBucketOnly;
  } else if (recall == "current") {
    config.card_abstraction.recall_mode =
        poker::RecallMode::CurrentBucketOnly;
  } else if (recall == "history") {
    config.card_abstraction.recall_mode = poker::RecallMode::BucketHistory;
  } else {
    return absl::InvalidArgumentError("invalid private recall mode");
  }

  const auto fractions =
      ParsePotFractions(absl::GetFlag(FLAGS_pot_fractions));
  if (!fractions.ok()) {
    return fractions.status();
  }
  config.bet_abstraction.pot_fractions.fill(*fractions);
  for (const auto& [street, values] : {
           std::pair{poker::StreetKind::Preflop,
                     absl::GetFlag(FLAGS_preflop_pot_fractions)},
           std::pair{poker::StreetKind::Flop,
                     absl::GetFlag(FLAGS_flop_pot_fractions)},
           std::pair{poker::StreetKind::Turn,
                     absl::GetFlag(FLAGS_turn_pot_fractions)},
           std::pair{poker::StreetKind::River,
                     absl::GetFlag(FLAGS_river_pot_fractions)},
       }) {
    const absl::Status status = OverridePotFractions(config, street, values);
    if (!status.ok()) {
      return status;
    }
  }
  return poker::SolverConfig::Create(std::move(config));
}

bool CapHit(size_t count, int cap) {
  return count >= static_cast<size_t>(cap);
}

void PrintRunSummary(const poker::CFRSolver& solver,
                     const poker::SolverConfig& config,
                     double seconds) {
  const size_t info_sets = solver.get_info_set_count();
  const size_t history_nodes = solver.get_history_count();
  const uint64_t visits = solver.get_stats().decision_visits;

  std::cout << "iterations=" << solver.get_iterations_run() << "\n";
  std::cout << "info_sets=" << info_sets << "\n";
  std::cout << "max_info_sets=" << config.max_info_sets() << "\n";
  std::cout << "info_set_cap_hit=" << CapHit(info_sets, config.max_info_sets())
            << "\n";
  std::cout << "player_a_ev=" << solver.get_expected_value(poker::Player::A)
            << "\n";
  std::cout << "seconds=" << seconds << "\n";
  std::cout << "history_nodes=" << history_nodes << "\n";
  std::cout << "equity_cache_entries="
            << solver.card_abstraction().cache_size() << "\n";
  std::cout << "decision_visits=" << visits << "\n";
  if (seconds > 0.0) {
    std::cout << "decision_visits_per_second="
              << static_cast<double>(visits) / seconds << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage("Train the heads-up poker CFR solver.");
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  if (absl::GetFlag(FLAGS_log)) {
    absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
    absl::SetGlobalVLogLevel(1);
  }

  const int64_t memory_limit_mb = absl::GetFlag(FLAGS_max_memory_mb);
  if (memory_limit_mb < 0) {
    std::cerr << "Error: --max_memory_mb must be non-negative\n";
    return 1;
  }
  const auto config = ConfigFromFlags();
  if (!config.ok()) {
    std::cerr << "Error: " << config.status() << "\n";
    return 1;
  }
  const poker::ComboRange a_range = poker::UniformComboRange();
  const poker::ComboRange b_range = poker::UniformComboRange();
  const poker::Chips stack = config->starting_stack();
  const poker::ExactPublicState root = poker::MakeInitialState(
      poker::BettingRules{config->big_blind()}, {stack, stack},
      {config->small_blind(), config->big_blind()});
  SetMemoryLimit(memory_limit_mb);
  auto solver = poker::CFRSolver::Create(
      {*config, root, {a_range, b_range}});
  if (!solver.ok()) {
    std::cerr << "Error: " << solver.status() << "\n";
    return 1;
  }
  const auto start = std::chrono::steady_clock::now();
  const poker::TrainingResult result =
      (*solver)->run(absl::GetFlag(FLAGS_iterations));
  const auto end = std::chrono::steady_clock::now();

  const std::chrono::duration<double> elapsed = end - start;
  PrintRunSummary(**solver, *config, elapsed.count());
  std::cout << "stop_reason="
            << (result.stop_reason ==
                        poker::TrainingStopReason::IterationsCompleted
                    ? "iterations_completed"
                    : "info_set_limit")
            << "\n";
  return 0;
}
