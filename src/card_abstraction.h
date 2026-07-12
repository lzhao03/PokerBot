#pragma once

#include <array>
#include <filesystem>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
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
  ExactCanonical,
  Texture,
};

enum class PrivateAbstractionKind : uint8_t {
  ExactCanonical,
  Handcrafted36,
  EquityPotential,
};

enum class RecallMode : uint8_t {
  CurrentBucketOnly,
  BucketHistory,
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

struct CardAbstractionConfig {
  PublicCardMode public_mode = PublicCardMode::Texture;
  PrivateAbstractionKind private_kind =
      PrivateAbstractionKind::Handcrafted36;
  RecallMode recall_mode = RecallMode::BucketHistory;
  std::optional<EquityBucketModel> equity_model;
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
EquityFeatures EvaluateEquityFeatures(
    ComboId hand,
    const Board& board,
    const EquityBucketModel& model) noexcept;

class CardAbstraction;

class PublicPosition {
 public:
  static PublicPosition Root(const CardAbstraction& abstraction,
                             StreetKind street,
                             Board board);

  PublicPosition after_chance(const CardAbstraction& abstraction,
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

  StreetKind street_ = StreetKind::Preflop;
  Board board_ = PreflopBoard{};
  PublicObservationId observation_;
  BoardFeatures features_;
};

class CardAbstraction {
 public:
  static absl::StatusOr<CardAbstraction> Create(
      CardAbstractionConfig config);

  const CardAbstractionConfig& config() const noexcept { return config_; }
  size_t cache_size() const noexcept { return equity_cache_.size(); }
  PrivateBucketId equity_bucket(ComboId hand,
                                const Board& board) const noexcept;

 private:
  struct CacheKey {
    PublicObservationId public_observation;
    PrivateObservationId private_observation;

    friend bool operator==(const CacheKey&, const CacheKey&) = default;

    template <typename H>
    friend H AbslHashValue(H hash, const CacheKey& key) {
      return H::combine(std::move(hash), key.public_observation.value(),
                        key.private_observation.value());
    }
  };

  explicit CardAbstraction(CardAbstractionConfig config)
      : config_(std::move(config)) {}

  CardAbstractionConfig config_;
  mutable absl::flat_hash_map<CacheKey, PrivateBucketId> equity_cache_;

  friend PublicObservationId ObservePublic(
      const CardAbstraction&, const Board&) noexcept;
  friend PrivateObservationId ObservePrivate(
      const CardAbstraction&, ComboId, const PublicPosition&) noexcept;
};

BoardFeatures BoardFeaturesFor(const Board& board) noexcept;
BoardBucketId BoardTextureBucket(StreetKind street,
                                 const BoardFeatures& features) noexcept;
PrivateBucketId Handcrafted36Bucket(ComboId hand,
                                    StreetKind street,
                                    const BoardFeatures& features) noexcept;

PublicObservationId ObservePublic(const CardAbstraction& abstraction,
                                  const Board& board) noexcept;
PrivateObservationId ObservePrivate(const CardAbstraction& abstraction,
                                    ComboId hand,
                                    const PublicPosition& position) noexcept;

}  // namespace poker
