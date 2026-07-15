#include "src/solver.h"

#include "absl/status/statusor.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/strings/numbers.h"

#include <array>
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
ABSL_FLAG(int, threads, 1, "training worker threads after infoset prefill");
ABSL_FLAG(int64_t, max_memory_mb, 4096,
          "hard memory limit in MB; 0 is unlimited");
ABSL_FLAG(bool, accumulate_average_strategy, true,
          "store the average strategy");
ABSL_FLAG(bool, external_sampling, false,
          "sample opponent actions during training");
ABSL_FLAG(std::string, public_abstraction, "texture",
          "exact, texture, or compact_texture");
ABSL_FLAG(std::string, private_abstraction, "handcrafted36",
          "exact or handcrafted36");
ABSL_FLAG(std::string, private_recall, "auto",
          "auto, current, or history");
ABSL_FLAG(std::string, betting_abstraction, "default",
          "default or small_betting");
ABSL_FLAG(std::vector<std::string>, pot_fractions, {},
          "override pot fractions after calling for every street");
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
  const rlimit limit{bytes, bytes};
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

absl::StatusOr<poker::SolverConfig> ConfigFromFlags() {
  poker::SolverConfig config;
  const poker::Chips stack = absl::GetFlag(FLAGS_starting_stack);
  const poker::Chips small_blind = absl::GetFlag(FLAGS_small_blind);
  const poker::Chips big_blind = absl::GetFlag(FLAGS_big_blind);
  if (stack <= 0 || small_blind <= 0 || big_blind < small_blind ||
      stack < big_blind) {
    return absl::InvalidArgumentError("invalid stack or blind configuration");
  }
  config.betting_rules.minimum_bet = big_blind;
  config.chance_samples = absl::GetFlag(FLAGS_chance_samples);
  config.max_info_sets = absl::GetFlag(FLAGS_max_info_sets);
  config.accumulate_average_strategy =
      absl::GetFlag(FLAGS_accumulate_average_strategy);
  config.external_sampling = absl::GetFlag(FLAGS_external_sampling);

  const std::string public_abstraction =
      absl::GetFlag(FLAGS_public_abstraction);
  if (public_abstraction == "exact") {
    config.card_abstraction.public_mode =
        poker::PublicCardMode::ExactCanonical;
  } else if (public_abstraction == "compact_texture") {
    config.card_abstraction.public_mode =
        poker::PublicCardMode::CompactTexture;
  } else if (public_abstraction != "texture") {
    return absl::InvalidArgumentError("invalid public abstraction");
  }

  const std::string private_abstraction =
      absl::GetFlag(FLAGS_private_abstraction);
  if (private_abstraction != "exact" &&
      private_abstraction != "handcrafted36") {
    return absl::InvalidArgumentError("invalid private abstraction");
  }
  config.card_abstraction.private_kind = private_abstraction == "exact"
      ? poker::PrivateAbstractionKind::ExactCanonical
      : poker::PrivateAbstractionKind::Handcrafted36;

  const std::string recall = absl::GetFlag(FLAGS_private_recall);
  if (recall != "auto" && recall != "current" && recall != "history") {
    return absl::InvalidArgumentError("invalid private recall mode");
  }
  const bool retain_bucket_history =
      recall == "history" ||
      (recall == "auto" &&
       config.card_abstraction.private_kind ==
           poker::PrivateAbstractionKind::Handcrafted36);
  config.card_abstraction.recall_mode = retain_bucket_history
          ? poker::RecallMode::BucketHistory
          : poker::RecallMode::CurrentBucketOnly;

  const std::string betting_abstraction =
      absl::GetFlag(FLAGS_betting_abstraction);
  if (betting_abstraction == "small_betting") {
    config.bet_abstraction = poker::SmallBettingConfig();
  } else if (betting_abstraction != "default") {
    return absl::InvalidArgumentError("invalid betting abstraction");
  }
  const auto global_fractions = absl::GetFlag(FLAGS_pot_fractions);
  if (!global_fractions.empty()) {
    const auto fractions = ParsePotFractions(global_fractions);
    if (!fractions.ok()) return fractions.status();
    config.bet_abstraction.pot_fractions.fill(*fractions);
  }
  const std::array overrides = {
      absl::GetFlag(FLAGS_preflop_pot_fractions),
      absl::GetFlag(FLAGS_flop_pot_fractions),
      absl::GetFlag(FLAGS_turn_pot_fractions),
      absl::GetFlag(FLAGS_river_pot_fractions),
  };
  for (size_t street = 0; street < overrides.size(); ++street) {
    if (overrides[street].empty()) continue;
    const auto override = ParsePotFractions(overrides[street]);
    if (!override.ok()) return override.status();
    config.bet_abstraction.pot_fractions[street] = *override;
  }
  return poker::SolverConfig::Create(std::move(config));
}

void PrintRunSummary(const poker::CFRSolver& solver,
                     const poker::SolverConfig& config,
                     double seconds) {
  const size_t info_sets = solver.get_info_set_count();
  const size_t history_nodes = solver.get_history_count();
  const uint64_t visits = solver.get_stats().decision_visits;

  std::cout << "iterations=" << solver.get_iterations_run() << "\n";
  std::cout << "info_sets=" << info_sets << "\n";
  std::cout << "max_info_sets=" << config.max_info_sets << "\n";
  std::cout << "info_set_cap_hit="
            << (info_sets >= static_cast<size_t>(config.max_info_sets))
            << "\n";
  std::cout << "player_a_ev=" << solver.get_expected_value(poker::Player::A)
            << "\n";
  std::cout << "seconds=" << seconds << "\n";
  std::cout << "history_nodes=" << history_nodes << "\n";
  std::cout << "decision_visits=" << visits << "\n";
  if (seconds > 0.0) {
    std::cout << "decision_visits_per_second="
              << visits / seconds << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage("Train the heads-up poker CFR solver.");
  absl::ParseCommandLine(argc, argv);
  const int64_t memory_limit_mb = absl::GetFlag(FLAGS_max_memory_mb);
  if (memory_limit_mb < 0) {
    std::cerr << "Error: --max_memory_mb must be non-negative\n";
    return 1;
  }
  const int threads = absl::GetFlag(FLAGS_threads);
  if (threads <= 0) {
    std::cerr << "Error: --threads must be positive\n";
    return 1;
  }
  const auto config = ConfigFromFlags();
  if (!config.ok()) {
    std::cerr << "Error: " << config.status() << "\n";
    return 1;
  }
  const poker::ComboRange range = poker::UniformComboRange();
  const poker::Chips stack = absl::GetFlag(FLAGS_starting_stack);
  const poker::Chips small_blind = absl::GetFlag(FLAGS_small_blind);
  const poker::ExactPublicState root = poker::MakeInitialState(
      config->betting_rules, {stack, stack},
      {small_blind, config->betting_rules.minimum_bet});
  SetMemoryLimit(memory_limit_mb);
  auto solver = poker::CFRSolver::Create(
      {*config, root, {range, range}});
  if (!solver.ok()) {
    std::cerr << "Error: " << solver.status() << "\n";
    return 1;
  }
  const auto start = std::chrono::steady_clock::now();
  solver->run(absl::GetFlag(FLAGS_iterations), threads);
  const auto end = std::chrono::steady_clock::now();

  const std::chrono::duration<double> elapsed = end - start;
  PrintRunSummary(*solver, *config, elapsed.count());
  std::cout << "threads=" << threads << "\n";
  return 0;
}
