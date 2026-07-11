#pragma once

#include <array>
#include <filesystem>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/card_canonicalization.h"
#include "src/fingerprint.h"
#include "src/poker.h"

namespace poker {

using PrivateBucketId = uint16_t;

inline constexpr uint32_t kCoarsePrivateStreetObservationCount = 36;
inline constexpr uint32_t kCoarsePublicStreetObservationCount = 108;

enum class PublicCardMode : uint8_t {
  kExactCanonical,
  kTexture,
};

enum class PrivateAbstractionKind : uint8_t {
  kExactCanonical,
  kHandcrafted36,
};

enum class RecallMode : uint8_t {
  kCurrentBucketOnly,
  kBucketHistory,
};

struct CardAbstractionConfig {
  PublicCardMode public_mode = PublicCardMode::kTexture;
  PrivateAbstractionKind private_kind =
      PrivateAbstractionKind::kHandcrafted36;
  RecallMode recall_mode = RecallMode::kBucketHistory;
};

struct BoardFeatures {
  std::array<uint8_t, 13> rank_counts = {};
  std::array<uint8_t, 4> suit_counts = {};
  uint16_t rank_mask = 0;
  uint8_t card_count = 0;
  uint8_t max_rank_count = 0;
  uint8_t max_suit_count = 0;
  uint8_t max_rank = 0;

  friend bool operator==(const BoardFeatures&, const BoardFeatures&) = default;
};

struct EquityFeatures {
  float ehs = 0.0f;
  float ehs2 = 0.0f;

  friend bool operator==(const EquityFeatures&,
                         const EquityFeatures&) = default;
};

struct EquityBucketModel {
  uint32_t format_version = 1;
  uint32_t feature_version = 1;
  uint64_t rollout_seed = 0;
  uint64_t fit_seed = 0;
  uint32_t training_samples = 0;
  uint32_t opponent_samples = 0;
  uint32_t runout_samples = 0;
  std::array<std::vector<float>, 4> ehs2_cutoffs;
  std::array<std::vector<float>, 4> ehs_medians;
  std::vector<float> river_equity_cutoffs;
  ModelFingerprint fingerprint;

  friend bool operator==(const EquityBucketModel&,
                         const EquityBucketModel&) = default;
};

absl::Status ValidateEquityBucketModel(const EquityBucketModel& model);
absl::StatusOr<EquityBucketModel> FinalizeEquityBucketModel(
    EquityBucketModel model);
absl::Status SaveEquityBucketModel(
    const EquityBucketModel& model,
    const std::filesystem::path& path);
absl::StatusOr<EquityBucketModel> LoadEquityBucketModel(
    const std::filesystem::path& path);
PrivateBucketId EquityBucket(StreetKind street,
                             EquityFeatures features,
                             const EquityBucketModel& model) noexcept;

class PublicPosition {
 public:
  static PublicPosition Root(const CardAbstractionConfig& config,
                             StreetKind street,
                             Board board);

  PublicPosition after_chance(const CardAbstractionConfig& config,
                              StreetKind street,
                              Board board) const;

  StreetKind street() const noexcept { return street_; }
  const Board& board() const noexcept { return board_; }
  PublicObservationId observation() const noexcept { return observation_; }
  const BoardFeatures& features() const noexcept { return features_; }

 private:
  PublicPosition(StreetKind street,
                 Board board,
                 PublicObservationId observation,
                 BoardFeatures features)
      : street_(street),
        board_(std::move(board)),
        observation_(observation),
        features_(features) {}

  StreetKind street_ = StreetKind::kPreflop;
  Board board_ = PreflopBoard{};
  PublicObservationId observation_;
  BoardFeatures features_;
};

BoardFeatures BoardFeaturesFor(const Board& board) noexcept;
BoardBucketId BoardTextureBucket(StreetKind street,
                                 const BoardFeatures& features) noexcept;
PrivateBucketId CoarsePrivateBucket(ComboId hand,
                                    StreetKind street,
                                    const BoardFeatures& features) noexcept;

PublicObservationId ObservePublic(const CardAbstractionConfig& config,
                                  StreetKind street,
                                  const Board& board) noexcept;
PrivateObservationId ObservePrivate(const CardAbstractionConfig& config,
                                    ComboId hand,
                                    const PublicPosition& position) noexcept;

PrivateObservationId InitialPrivateObservation(
    const CardAbstractionConfig& config,
    ComboId hand) noexcept;
PrivateObservationId AdvancePrivateObservation(
    const CardAbstractionConfig& config,
    PrivateObservationId previous,
    ComboId hand,
    const PublicPosition& child) noexcept;

}  // namespace poker
