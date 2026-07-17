#include "src/solver.h"
#include "src/evaluation.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/status/status.h"

ABSL_FLAG(int, iterations, 100, "CFR iterations");
ABSL_FLAG(int, eval_samples, 100, "evaluation samples");
ABSL_FLAG(bool, evaluate, true, "evaluate and extract a policy after training");
ABSL_FLAG(std::string, range, "premium",
          "premium, all, or a poker range");
ABSL_FLAG(double, training_seconds, 0.0,
          "train for this wall-clock duration; 0 uses iterations");
ABSL_FLAG(uint64_t, prefill_iterations, 0,
          "single-thread iterations before timed training");
ABSL_FLAG(std::string, public_abstraction, "texture",
          "exact, texture, or compact_texture");
ABSL_FLAG(std::string, private_abstraction, "handcrafted36",
          "exact or handcrafted36");
ABSL_FLAG(std::string, private_recall, "auto",
          "auto, current, or history");
ABSL_FLAG(std::string, betting_abstraction, "default",
          "default or small_betting");
ABSL_FLAG(uint64_t, evaluation_seed, 1, "policy evaluation seed");
ABSL_FLAG(uint64_t, best_response_iterations, 0,
          "approximate best-response iterations; 0 disables it");
ABSL_FLAG(int, starting_stack, 100, "starting stack in chips");
ABSL_FLAG(int, max_info_sets, 500000, "maximum infosets");
ABSL_FLAG(int, chance_samples, 1, "chance samples per chance node");
ABSL_FLAG(bool, accumulate_average_strategy, true,
          "accumulate average-strategy values during training");
ABSL_FLAG(bool, external_sampling, false,
          "sample opponent actions during training");
ABSL_FLAG(int, threads, 1, "training worker threads");
ABSL_FLAG(uint64_t, progress_interval, 0,
          "log progress after this many iterations; 0 disables logging");
ABSL_FLAG(bool, reach_coverage, false,
          "measure infosets needed to cover 99% of policy reach");
ABSL_FLAG(std::string, policy_output, "", "optional policy output path");

namespace {

absl::StatusOr<poker::ComboRange> BenchmarkRange(std::string_view text) {
  if (text == "premium") {
    return poker::ParseRange("AA,KK,QQ,JJ,AKs,AQs,AKo");
  }
  if (text == "all") {
    return poker::UniformComboRange();
  }
  return poker::ParseRange(text);
}

double Rate(double count, double seconds) {
  return seconds > 0.0 ? count / seconds : 0.0;
}

absl::StatusOr<poker::SolverConfig> BenchmarkConfig() {
  poker::SolverConfig options;
  if (absl::GetFlag(FLAGS_starting_stack) <
      options.betting_rules.minimum_bet) {
    return absl::InvalidArgumentError("starting stack is too small");
  }
  options.max_info_sets = absl::GetFlag(FLAGS_max_info_sets);
  options.chance_samples = absl::GetFlag(FLAGS_chance_samples);
  options.accumulate_average_strategy =
      absl::GetFlag(FLAGS_accumulate_average_strategy);
  options.external_sampling = absl::GetFlag(FLAGS_external_sampling);
  const std::string public_mode = absl::GetFlag(FLAGS_public_abstraction);
  if (public_mode == "exact") {
    options.card_abstraction.public_mode =
        poker::PublicCardMode::ExactCanonical;
  } else if (public_mode == "compact_texture") {
    options.card_abstraction.public_mode =
        poker::PublicCardMode::CompactTexture;
  } else if (public_mode != "texture") {
    return absl::InvalidArgumentError("invalid public abstraction");
  }
  const std::string betting = absl::GetFlag(FLAGS_betting_abstraction);
  if (betting == "small_betting") {
    options.bet_abstraction = poker::SmallBettingConfig();
  } else if (betting != "default") {
    return absl::InvalidArgumentError("invalid betting abstraction");
  }
  const std::string kind = absl::GetFlag(FLAGS_private_abstraction);
  if (kind == "exact") {
    options.card_abstraction.private_kind =
        poker::PrivateAbstractionKind::ExactCanonical;
  } else if (kind == "handcrafted36") {
    options.card_abstraction.private_kind =
        poker::PrivateAbstractionKind::Handcrafted36;
  } else {
    return absl::InvalidArgumentError("invalid private abstraction");
  }
  const std::string recall = absl::GetFlag(FLAGS_private_recall);
  if (recall == "auto") {
    options.card_abstraction.recall_mode =
        kind == "handcrafted36" ? poker::RecallMode::BucketHistory
                                : poker::RecallMode::CurrentBucketOnly;
  } else if (recall == "current") {
    options.card_abstraction.recall_mode =
        poker::RecallMode::CurrentBucketOnly;
  } else if (recall == "history") {
    options.card_abstraction.recall_mode =
        poker::RecallMode::BucketHistory;
  } else {
    return absl::InvalidArgumentError("invalid private recall mode");
  }
  return poker::SolverConfig::Create(std::move(options));
}

template <typename Function>
double Measure(std::string_view name, Function function) {
  const auto start = std::chrono::steady_clock::now();
  const auto result = function();
  const auto end = std::chrono::steady_clock::now();
  const double seconds = std::chrono::duration<double>(end - start).count();
  std::cout << name << '\t' << seconds << '\t' << result << '\n';
  return seconds;
}

}  // namespace

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage("Benchmark the heads-up poker CFR solver.");
  absl::ParseCommandLine(argc, argv);
  const auto config_result = BenchmarkConfig();
  if (!config_result.ok()) {
    std::cerr << "Error: " << config_result.status() << '\n';
    return 1;
  }
  const poker::SolverConfig config = *config_result;

  const std::string range = absl::GetFlag(FLAGS_range);
  const auto parsed_range = BenchmarkRange(range);
  if (!parsed_range.ok()) {
    std::cerr << "Error: " << parsed_range.status() << '\n';
    return 1;
  }
  const poker::ComboRange a_range = *parsed_range;
  const poker::ComboRange b_range = *parsed_range;
  const poker::Chips stack = absl::GetFlag(FLAGS_starting_stack);
  const poker::ExactPublicState root = poker::MakeInitialState(
      config.betting_rules, {stack, stack},
      {1, config.betting_rules.minimum_bet});

  std::cout << "case\tseconds\tresult\n";
  Measure("range_expand", [&] { return a_range.count(); });

  std::optional<poker::CFRSolver> solver;
  std::string build_error;
  const uint64_t progress_interval =
      absl::GetFlag(FLAGS_progress_interval);
  const int threads = absl::GetFlag(FLAGS_threads);
  if (threads <= 0) {
    std::cerr << "Error: threads must be positive\n";
    return 1;
  }
  if (progress_interval > 0) std::cerr << "building_history\n";
  Measure("build_history", [&] {
    auto result = poker::CFRSolver::Create(
        {config, root, {a_range, b_range}});
    if (!result.ok()) {
      build_error = result.status().ToString();
      return size_t{0};
    }
    solver = std::move(*result);
    return solver->history_count();
  });
  if (!build_error.empty()) {
    std::cerr << "Error: " << build_error << '\n';
    return 1;
  }
  if (progress_interval > 0) {
    std::cerr << "history_nodes\t" << solver->history_count() << '\n';
  }
  const uint64_t prefill_iterations =
      absl::GetFlag(FLAGS_prefill_iterations);
  if (prefill_iterations > 0) {
    Measure("prefill", [&] {
      solver->run(prefill_iterations);
      return solver->info_set_count();
    });
    solver->reset_stats();
  }
  const double training_seconds = Measure("train_range", [&] {
    const double seconds = absl::GetFlag(FLAGS_training_seconds);
    if (seconds <= 0.0) {
      const uint64_t iterations =
          static_cast<uint64_t>(absl::GetFlag(FLAGS_iterations));
      if (progress_interval == 0) {
        solver->run(iterations, threads);
      } else {
        while (solver->iterations() < iterations) {
          const uint64_t batch = std::min(
              progress_interval,
              iterations - solver->iterations());
          solver->run(batch, threads);
          std::cerr << "training_iterations\t"
                    << solver->iterations() << '\n';
        }
      }
    } else {
      const auto deadline = std::chrono::steady_clock::now() +
          std::chrono::duration<double>(seconds);
      while (std::chrono::steady_clock::now() < deadline) {
        solver->run(static_cast<uint64_t>(threads), threads);
        if (progress_interval > 0 &&
            solver->iterations() % progress_interval == 0) {
          std::cerr << "training_iterations\t"
                    << solver->iterations() << '\n';
        }
      }
    }
    return solver->expected_value(poker::Player::A);
  });
  const auto training = solver->stats();
  std::cout << "iterations\t" << solver->iterations() << '\n'
            << "threads\t" << threads << '\n'
            << "decision_visits\t" << training.decision_visits << '\n'
            << "decision_visits_per_second\t"
            << Rate(training.decision_visits, training_seconds) << '\n'
            << "chance_samples\t" << training.chance_samples << '\n'
            << "terminal_visits\t" << training.terminal_visits << '\n'
            << "infosets\t" << solver->info_set_count() << '\n'
            << "history_nodes\t" << solver->history_count() << '\n'
            << "regret_bytes\t" << solver->regret_bytes() << '\n'
            << "strategy_bytes\t" << solver->strategy_bytes() << '\n';

  if (!absl::GetFlag(FLAGS_evaluate)) return 0;

  solver->reset_stats();
  Measure("evaluate_range", [&] {
    if (!config.accumulate_average_strategy) {
      return solver->evaluate_current(
          absl::GetFlag(FLAGS_eval_samples));
    }
    const auto value = solver->evaluate_average(
        absl::GetFlag(FLAGS_eval_samples));
    return value.ok() ? *value : 0.0;
  });

  const auto policy = solver->extract_average_policy();
  if (policy.ok()) {
    std::cout << "policy_rows\t" << policy->rows.size() << '\n'
              << "policy_probability_bytes\t"
              << policy->probabilities.size() * sizeof(float) << '\n';
    const std::string output = absl::GetFlag(FLAGS_policy_output);
    if (!output.empty()) {
      const absl::Status saved = poker::SavePolicy(*policy, output);
      if (!saved.ok()) {
        std::cerr << "Error: " << saved << '\n';
        return 1;
      }
      std::cout << "policy_file_bytes\t"
                << std::filesystem::file_size(output) << '\n';
    }
    const auto profile = poker::EstimateExpectedValue(
        solver->game(), *policy, *policy,
        static_cast<uint64_t>(absl::GetFlag(FLAGS_eval_samples)),
        absl::GetFlag(FLAGS_evaluation_seed),
        absl::GetFlag(FLAGS_reach_coverage));
    if (profile.ok()) {
      std::cout << "policy_ev\t" << profile->mean << '\n'
                << "policy_standard_error\t" << profile->standard_error
                << '\n'
                << "policy_lookups\t" << profile->policy_lookups << '\n'
                << "missing_policy_lookups\t"
                << profile->missing_policy_lookups << '\n'
                << "weighted_policy_lookups\t"
                << profile->weighted_policy_lookups << '\n'
                << "weighted_missing_policy_lookups\t"
                << profile->weighted_missing_policy_lookups << '\n';
      if (absl::GetFlag(FLAGS_reach_coverage)) {
        std::cout << "observed_info_sets\t"
                  << profile->observed_info_sets << '\n'
                  << "info_sets_for_99_percent_reach\t"
                  << profile->info_sets_for_99_percent_reach << '\n';
      }
    }
    const uint64_t response_iterations =
        absl::GetFlag(FLAGS_best_response_iterations);
    if (response_iterations > 0) {
      const auto exploitability = poker::EstimateExploitability(
          solver->game(), *policy,
          {response_iterations,
           static_cast<uint64_t>(absl::GetFlag(FLAGS_eval_samples)),
           absl::GetFlag(FLAGS_evaluation_seed)});
      if (exploitability.ok()) {
        std::cout << "nash_conv\t" << exploitability->nash_conv << '\n'
                  << "exploitability\t" << exploitability->exploitability
                  << '\n'
                  << "missing_response_lookups\t"
                  << exploitability->player_a_response
                             .missing_opponent_lookups +
                         exploitability->player_b_response
                             .missing_opponent_lookups
                  << '\n';
      }
    }
  }
  return 0;
}
