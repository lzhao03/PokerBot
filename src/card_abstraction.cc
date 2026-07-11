#include "src/card_abstraction.h"

#include "src/hand_evaluator.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <optional>
#include <system_error>
#include <vector>

namespace poker {
namespace {

inline constexpr int kPublicObservationBitsPerStreet = 7;
inline constexpr int kSuitBuckets = 4;
inline constexpr int kStraightBuckets = 3;
inline constexpr int kHighBuckets = 3;

static_assert(kCoarsePublicStreetObservationCount <
              (uint32_t{1} << kPublicObservationBitsPerStreet));

constexpr uint8_t StraightWindowDensity(uint16_t rank_mask) {
  uint8_t best = 0;
  for (int start = 0; start <= 8; ++start) {
    const uint8_t count = static_cast<uint8_t>(__builtin_popcount(
        static_cast<unsigned int>((rank_mask >> start) & 0x1F)));
    best = std::max(best, count);
  }
  const uint16_t wheel_mask = static_cast<uint16_t>((1u << 12) | 0x0F);
  const uint8_t wheel_count = static_cast<uint8_t>(__builtin_popcount(
      static_cast<unsigned int>(rank_mask & wheel_mask)));
  return std::max(best, wheel_count);
}

constexpr std::array<uint8_t, 8192> BuildStraightDensityTable() {
  std::array<uint8_t, 8192> table = {};
  for (size_t mask = 0; mask < table.size(); ++mask) {
    table[mask] = StraightWindowDensity(static_cast<uint16_t>(mask));
  }
  return table;
}

inline constexpr auto kStraightDensity = BuildStraightDensityTable();

int HighRankGroup(int rank) noexcept {
  if (rank >= 14) {
    return 0;
  }
  if (rank >= 12) {
    return 1;
  }
  return rank >= 9 ? 2 : 3;
}

int LowRankGroup(int rank) noexcept {
  if (rank >= 10) {
    return 0;
  }
  return rank >= 7 ? 1 : 2;
}

int HoleStrengthBucket(int high, int low, bool pair, bool suited) noexcept {
  if (pair || high == 14 || (high >= 13 && low >= 10)) {
    return 0;
  }
  const int gap = high - low;
  return (high >= 11 && low >= 8) || (suited && gap <= 2) ? 1 : 2;
}

int MadeBucket(const ComboInfo& combo,
               const BoardFeatures& features) noexcept {
  std::array<uint8_t, 13> rank_counts = features.rank_counts;
  ++rank_counts[static_cast<size_t>(PokerRank(combo.card0) - 2)];
  ++rank_counts[static_cast<size_t>(PokerRank(combo.card1) - 2)];

  int pairs = 0;
  int max_count = 0;
  for (uint8_t count : rank_counts) {
    pairs += count >= 2 ? 1 : 0;
    max_count = std::max(max_count, static_cast<int>(count));
  }
  if (max_count >= 3) {
    return 3;
  }
  if (pairs >= 2) {
    return 2;
  }
  return pairs == 1 ? 1 : 0;
}

int DrawBucket(const ComboInfo& combo,
               const BoardFeatures& features) noexcept {
  std::array<uint8_t, 4> suit_counts = features.suit_counts;
  uint16_t rank_mask = features.rank_mask;
  auto add_card = [&](Card card) {
    ++suit_counts[static_cast<size_t>(card.suit())];
    rank_mask |= static_cast<uint16_t>(1u << (PokerRank(card) - 2));
  };
  add_card(combo.card0);
  add_card(combo.card1);

  for (uint8_t count : suit_counts) {
    if (count >= 4) {
      return 2;
    }
  }
  return kStraightDensity[rank_mask] >= 4 ? 1 : 0;
}

constexpr int PublicShift(StreetKind street) noexcept {
  assert(street != StreetKind::kPreflop);
  return (static_cast<int>(street) - 1) * kPublicObservationBitsPerStreet;
}

PublicObservationId AdvanceCoarsePublic(PublicObservationId previous,
                                        StreetKind street,
                                        BoardBucketId bucket) noexcept {
  assert(bucket < kCoarsePublicStreetObservationCount);
  const int shift = PublicShift(street);
  constexpr uint64_t kSlotMask =
      (uint64_t{1} << kPublicObservationBitsPerStreet) - 1;
  const uint64_t slot_mask = kSlotMask << shift;
  assert((previous.value() & slot_mask) == 0);
  return PublicObservationId(
      previous.value() | ((static_cast<uint64_t>(bucket) + 1) << shift));
}

PublicObservationId ObserveCoarsePublic(StreetKind street,
                                        const Board& board) noexcept {
  PublicObservationId observation;
  if (street == StreetKind::kPreflop) {
    return observation;
  }

  const auto cards = BoardCards(board);
  const FlopBoard flop =
      DealFlop(PreflopBoard{}, {cards[0], cards[1], cards[2]});
  observation = AdvanceCoarsePublic(
      observation, StreetKind::kFlop,
      BoardTextureBucket(StreetKind::kFlop, BoardFeaturesFor(Board{flop})));
  if (street == StreetKind::kFlop) {
    return observation;
  }

  const TurnBoard turn = DealTurn(flop, cards[3]);
  observation = AdvanceCoarsePublic(
      observation, StreetKind::kTurn,
      BoardTextureBucket(StreetKind::kTurn, BoardFeaturesFor(Board{turn})));
  if (street == StreetKind::kTurn) {
    return observation;
  }

  const RiverBoard river = DealRiver(turn, cards[4]);
  return AdvanceCoarsePublic(
      observation, StreetKind::kRiver,
      BoardTextureBucket(StreetKind::kRiver, BoardFeaturesFor(Board{river})));
}

using Bytes = std::vector<uint8_t>;

void AppendU32(Bytes& bytes, uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8) {
    bytes.push_back(static_cast<uint8_t>(value >> shift));
  }
}

void AppendU64(Bytes& bytes, uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8) {
    bytes.push_back(static_cast<uint8_t>(value >> shift));
  }
}

void AppendVector(Bytes& bytes, const std::vector<float>& values) {
  AppendU32(bytes, static_cast<uint32_t>(values.size()));
  for (float value : values) AppendU32(bytes, std::bit_cast<uint32_t>(value));
}

void AddVector(FingerprintBuilder& hash,
               const std::vector<float>& values) noexcept {
  hash.add_u32(static_cast<uint32_t>(values.size()));
  for (float value : values) hash.add_float(value);
}

ModelFingerprint EquityModelFingerprint(
    const EquityBucketModel& model) noexcept {
  FingerprintBuilder hash;
  hash.add_u32(0x4551424d);  // EQBM
  hash.add_u32(model.format_version);
  hash.add_u32(model.feature_version);
  hash.add_u64(model.rollout_seed);
  hash.add_u64(model.fit_seed);
  hash.add_u32(model.training_samples);
  hash.add_u32(model.opponent_samples);
  hash.add_u32(model.runout_samples);
  for (const auto& values : model.ehs2_cutoffs) AddVector(hash, values);
  for (const auto& values : model.ehs_medians) AddVector(hash, values);
  AddVector(hash, model.river_equity_cutoffs);
  return hash.finish();
}

bool ValidCutoffs(const std::vector<float>& values) {
  return std::is_sorted(values.begin(), values.end()) &&
         std::all_of(values.begin(), values.end(), [](float value) {
           return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
         });
}

bool ValidValues(const std::vector<float>& values) {
  return std::all_of(values.begin(), values.end(), [](float value) {
    return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
  });
}

absl::Status ValidateEquityModelShape(const EquityBucketModel& model) {
  if (model.format_version != 1 || model.feature_version != 1) {
    return absl::InvalidArgumentError("unsupported equity model version");
  }
  if (model.training_samples == 0 || model.opponent_samples == 0 ||
      model.runout_samples == 0) {
    return absl::InvalidArgumentError("equity model sampling is empty");
  }
  if (model.opponent_samples > 4096 || model.runout_samples > 4096 ||
      static_cast<uint64_t>(model.opponent_samples) *
              model.runout_samples >
          1'000'000) {
    return absl::InvalidArgumentError("equity model sampling is too large");
  }
  for (StreetKind street : {StreetKind::kPreflop, StreetKind::kRiver}) {
    const size_t index = static_cast<size_t>(street);
    if (!model.ehs2_cutoffs[index].empty() ||
        !model.ehs_medians[index].empty()) {
      return absl::InvalidArgumentError(
          "equity model has unexpected street cutoffs");
    }
  }
  for (StreetKind street : {StreetKind::kFlop, StreetKind::kTurn}) {
    const size_t index = static_cast<size_t>(street);
    if (model.ehs2_cutoffs[index].size() != 15 ||
        model.ehs_medians[index].size() != 16 ||
        !ValidCutoffs(model.ehs2_cutoffs[index]) ||
        !ValidValues(model.ehs_medians[index])) {
      return absl::InvalidArgumentError(
          "equity model requires 32 flop and turn buckets");
    }
  }
  if (model.river_equity_cutoffs.size() != 63 ||
      !ValidCutoffs(model.river_equity_cutoffs)) {
    return absl::InvalidArgumentError(
        "equity model requires 64 river buckets");
  }
  return absl::OkStatus();
}

class ByteReader {
 public:
  explicit ByteReader(absl::Span<const uint8_t> bytes) : bytes_(bytes) {}

  std::optional<uint8_t> u8() {
    if (offset_ == bytes_.size()) return std::nullopt;
    return bytes_[offset_++];
  }

  std::optional<uint32_t> u32() {
    if (offset_ + 4 > bytes_.size()) return std::nullopt;
    uint32_t value = 0;
    for (int shift = 0; shift < 32; shift += 8) {
      value |= static_cast<uint32_t>(bytes_[offset_++]) << shift;
    }
    return value;
  }

  std::optional<uint64_t> u64() {
    if (offset_ + 8 > bytes_.size()) return std::nullopt;
    uint64_t value = 0;
    for (int shift = 0; shift < 64; shift += 8) {
      value |= static_cast<uint64_t>(bytes_[offset_++]) << shift;
    }
    return value;
  }

  std::optional<std::vector<float>> floats(size_t maximum) {
    const auto count = u32();
    if (!count || *count > maximum) return std::nullopt;
    std::vector<float> values;
    values.reserve(*count);
    for (uint32_t index = 0; index < *count; ++index) {
      const auto bits = u32();
      if (!bits) return std::nullopt;
      values.push_back(std::bit_cast<float>(*bits));
    }
    return values;
  }

  bool done() const { return offset_ == bytes_.size(); }

 private:
  absl::Span<const uint8_t> bytes_;
  size_t offset_ = 0;
};

absl::Status WriteBytes(const Bytes& bytes,
                        const std::filesystem::path& path) {
  std::filesystem::path temporary = path;
  temporary += ".tmp";
  std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
  if (!output) return absl::UnavailableError("could not open model file");
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  output.close();
  if (!output) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    return absl::DataLossError("could not write model file");
  }
  std::error_code error;
  std::filesystem::rename(temporary, path, error);
  if (error) {
    std::filesystem::remove(temporary, error);
    return absl::UnavailableError("could not replace model file");
  }
  return absl::OkStatus();
}

absl::StatusOr<Bytes> ReadBytes(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) return absl::NotFoundError("could not open model file");
  const std::streamoff end = input.tellg();
  if (end < 0 || end > 4096) {
    return absl::DataLossError("invalid equity model size");
  }
  Bytes bytes(static_cast<size_t>(end));
  input.seekg(0);
  input.read(reinterpret_cast<char*>(bytes.data()), end);
  if (!input) return absl::DataLossError("could not read model file");
  return bytes;
}

uint64_t Mix64(uint64_t value) noexcept {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

uint64_t SampleSeed(const EquityBucketModel& model,
                    const CanonicalCardObservation& observation,
                    uint64_t runout,
                    uint64_t opponent) noexcept {
  uint64_t seed = Mix64(model.rollout_seed);
  seed = Mix64(seed ^ observation.public_observation.value());
  seed = Mix64(seed ^ observation.private_observation.value());
  seed = Mix64(seed ^ runout);
  return Mix64(seed ^ opponent);
}

size_t BoundedIndex(uint64_t& state, size_t bound) noexcept {
  assert(bound > 0);
  const uint64_t unsigned_bound = static_cast<uint64_t>(bound);
  const uint64_t threshold = -unsigned_bound % unsigned_bound;
  uint64_t value;
  do {
    state = Mix64(state);
    value = state;
  } while (value < threshold);
  return static_cast<size_t>(value % unsigned_bound);
}

std::array<Card, 2> SampleUnblocked(CardMask blocked,
                                    size_t count,
                                    uint64_t seed) noexcept {
  assert(count <= 2);
  std::array<Card, kDeckCardCount> available = {};
  size_t available_count = 0;
  for (Card card : kDeck) {
    if ((blocked & CardBit(card)) == 0) available[available_count++] = card;
  }
  assert(available_count >= count);
  std::array<Card, 2> sampled = {};
  for (size_t index = 0; index < count; ++index) {
    const size_t chosen = index + BoundedIndex(seed, available_count - index);
    std::swap(available[index], available[chosen]);
    sampled[index] = available[index];
  }
  return sampled;
}

RiverBoard CompleteBoard(const Board& board,
                         const std::array<Card, 2>& cards) noexcept {
  if (const auto* flop = std::get_if<FlopBoard>(&board)) {
    return DealRiver(DealTurn(*flop, cards[0]), cards[1]);
  }
  const auto* turn = std::get_if<TurnBoard>(&board);
  assert(turn != nullptr);
  return DealRiver(*turn, cards[0]);
}

float EstimateShowdownEquity(
    ComboId hand,
    const RiverBoard& board,
    uint32_t samples,
    const EquityBucketModel& model,
    const CanonicalCardObservation& observation,
    uint64_t runout) noexcept {
  const CardMask blocked = ComboMask(hand) | board.mask();
  const uint16_t hero_value = HandValue(hand, board);
  double value = 0.0;
  for (uint32_t sample = 0; sample < samples; ++sample) {
    const auto cards = SampleUnblocked(
        blocked, 2, SampleSeed(model, observation, runout, sample));
    const ComboId opponent = CardsToComboId(cards[0], cards[1]);
    const uint16_t opponent_value = HandValue(opponent, board);
    const int comparison = hero_value < opponent_value
                               ? 1
                               : (hero_value > opponent_value ? -1 : 0);
    value += comparison > 0 ? 1.0 : (comparison == 0 ? 0.5 : 0.0);
  }
  return static_cast<float>(value / samples);
}

}  // namespace

absl::Status ValidateEquityBucketModel(const EquityBucketModel& model) {
  const absl::Status shape = ValidateEquityModelShape(model);
  if (!shape.ok()) return shape;
  if (model.fingerprint != EquityModelFingerprint(model)) {
    return absl::DataLossError("equity model fingerprint mismatch");
  }
  return absl::OkStatus();
}

absl::StatusOr<EquityBucketModel> FinalizeEquityBucketModel(
    EquityBucketModel model) {
  const absl::Status shape = ValidateEquityModelShape(model);
  if (!shape.ok()) return shape;
  model.fingerprint = EquityModelFingerprint(model);
  return model;
}

absl::Status SaveEquityBucketModel(
    const EquityBucketModel& model,
    const std::filesystem::path& path) {
  const absl::Status valid = ValidateEquityBucketModel(model);
  if (!valid.ok()) return valid;
  Bytes bytes = {'P', 'E', 'Q', 'M', 'O', 'D', '1', 0};
  AppendU32(bytes, model.format_version);
  AppendU32(bytes, model.feature_version);
  AppendU64(bytes, model.rollout_seed);
  AppendU64(bytes, model.fit_seed);
  AppendU32(bytes, model.training_samples);
  AppendU32(bytes, model.opponent_samples);
  AppendU32(bytes, model.runout_samples);
  for (const auto& values : model.ehs2_cutoffs) AppendVector(bytes, values);
  for (const auto& values : model.ehs_medians) AppendVector(bytes, values);
  AppendVector(bytes, model.river_equity_cutoffs);
  for (std::byte byte : model.fingerprint.bytes) {
    bytes.push_back(std::to_integer<uint8_t>(byte));
  }
  return WriteBytes(bytes, path);
}

absl::StatusOr<EquityBucketModel> LoadEquityBucketModel(
    const std::filesystem::path& path) {
  const auto bytes = ReadBytes(path);
  if (!bytes.ok()) return bytes.status();
  constexpr std::array<uint8_t, 8> kMagic = {
      'P', 'E', 'Q', 'M', 'O', 'D', '1', 0};
  if (bytes->size() < kMagic.size() ||
      !std::equal(kMagic.begin(), kMagic.end(), bytes->begin())) {
    return absl::DataLossError("invalid equity model magic");
  }
  ByteReader reader(absl::MakeConstSpan(*bytes).subspan(kMagic.size()));
  const auto format = reader.u32();
  const auto feature = reader.u32();
  const auto rollout_seed = reader.u64();
  const auto fit_seed = reader.u64();
  const auto training_samples = reader.u32();
  const auto opponent_samples = reader.u32();
  const auto runout_samples = reader.u32();
  if (!format || !feature || !rollout_seed || !fit_seed ||
      !training_samples || !opponent_samples || !runout_samples) {
    return absl::DataLossError("truncated equity model header");
  }

  EquityBucketModel model;
  model.format_version = *format;
  model.feature_version = *feature;
  model.rollout_seed = *rollout_seed;
  model.fit_seed = *fit_seed;
  model.training_samples = *training_samples;
  model.opponent_samples = *opponent_samples;
  model.runout_samples = *runout_samples;
  for (auto& values : model.ehs2_cutoffs) {
    auto loaded = reader.floats(63);
    if (!loaded) return absl::DataLossError("truncated equity cutoffs");
    values = std::move(*loaded);
  }
  for (auto& values : model.ehs_medians) {
    auto loaded = reader.floats(64);
    if (!loaded) return absl::DataLossError("truncated equity medians");
    values = std::move(*loaded);
  }
  auto river = reader.floats(63);
  if (!river) return absl::DataLossError("truncated river cutoffs");
  model.river_equity_cutoffs = std::move(*river);
  for (std::byte& byte : model.fingerprint.bytes) {
    const auto value = reader.u8();
    if (!value) {
      return absl::DataLossError("truncated model fingerprint");
    }
    byte = static_cast<std::byte>(*value);
  }
  if (!reader.done()) return absl::DataLossError("trailing model data");
  const absl::Status valid = ValidateEquityBucketModel(model);
  if (!valid.ok()) return valid;
  return model;
}

PrivateBucketId EquityBucket(StreetKind street,
                             EquityFeatures features,
                             const EquityBucketModel& model) noexcept {
  assert(features.ehs >= 0.0f && features.ehs <= 1.0f);
  assert(features.ehs2 >= 0.0f && features.ehs2 <= 1.0f);
  if (street == StreetKind::kRiver) {
    return static_cast<PrivateBucketId>(std::upper_bound(
        model.river_equity_cutoffs.begin(),
        model.river_equity_cutoffs.end(), features.ehs) -
        model.river_equity_cutoffs.begin());
  }
  assert(street == StreetKind::kFlop || street == StreetKind::kTurn);
  const size_t index = static_cast<size_t>(street);
  const size_t outer = static_cast<size_t>(std::upper_bound(
      model.ehs2_cutoffs[index].begin(), model.ehs2_cutoffs[index].end(),
      features.ehs2) - model.ehs2_cutoffs[index].begin());
  return static_cast<PrivateBucketId>(
      2 * outer + (features.ehs >= model.ehs_medians[index][outer] ? 1 : 0));
}

EquityFeatures EvaluateEquityFeatures(
    ComboId hand,
    const Board& board,
    const EquityBucketModel& model) noexcept {
  assert(BoardCount(board) >= 3);
  const CanonicalCardState canonical = CanonicalizeCardState(hand, board);
  assert((ComboMask(canonical.hand) & BoardMask(canonical.board)) == 0);

  if (const auto* river = std::get_if<RiverBoard>(&canonical.board)) {
    const uint32_t samples = model.runout_samples * model.opponent_samples;
    const float equity = EstimateShowdownEquity(
        canonical.hand, *river, samples, model, canonical.observation, 0);
    return {equity, equity * equity};
  }

  const size_t cards_needed =
      std::holds_alternative<FlopBoard>(canonical.board) ? 2 : 1;
  const CardMask blocked =
      ComboMask(canonical.hand) | BoardMask(canonical.board);
  double ehs = 0.0;
  double ehs2 = 0.0;
  for (uint32_t runout = 0; runout < model.runout_samples; ++runout) {
    const auto cards = SampleUnblocked(
        blocked, cards_needed,
        SampleSeed(model, canonical.observation, runout,
                   std::numeric_limits<uint64_t>::max()));
    const RiverBoard river = CompleteBoard(canonical.board, cards);
    const float equity = EstimateShowdownEquity(
        canonical.hand, river, model.opponent_samples, model,
        canonical.observation, runout);
    ehs += equity;
    ehs2 += static_cast<double>(equity) * equity;
  }
  return {
      static_cast<float>(ehs / model.runout_samples),
      static_cast<float>(ehs2 / model.runout_samples),
  };
}

BoardFeatures BoardFeaturesFor(const Board& board) noexcept {
  BoardFeatures features;
  features.card_count = BoardCount(board);
  for (Card card : BoardCards(board)) {
    const size_t rank = static_cast<size_t>(card.rank());
    const size_t suit = static_cast<size_t>(card.suit());
    ++features.rank_counts[rank];
    ++features.suit_counts[suit];
    features.max_rank_count =
        std::max(features.max_rank_count, features.rank_counts[rank]);
    features.max_suit_count =
        std::max(features.max_suit_count, features.suit_counts[suit]);
    features.max_rank =
        std::max(features.max_rank, static_cast<uint8_t>(PokerRank(card)));
    features.rank_mask |= static_cast<uint16_t>(1u << rank);
  }
  return features;
}

BoardBucketId BoardTextureBucket(StreetKind,
                                 const BoardFeatures& features) noexcept {
  if (features.card_count == 0) {
    return 0;
  }
  const int paired = features.max_rank_count >= 3
                         ? 2
                         : (features.max_rank_count == 2 ? 1 : 0);
  const int suited = features.max_suit_count >= 4
                         ? 3
                         : (features.max_suit_count >= 3
                                ? 2
                                : (features.max_suit_count == 2 ? 1 : 0));
  const int density = kStraightDensity[features.rank_mask & 0x1FFF];
  const int straight = density >= 4 ? 2 : (density >= 3 ? 1 : 0);
  const int high = features.max_rank >= 14
                       ? 0
                       : (features.max_rank >= 11 ? 1 : 2);
  return static_cast<BoardBucketId>(
      (((paired * kSuitBuckets) + suited) * kStraightBuckets + straight) *
          kHighBuckets +
      high);
}

PrivateBucketId CoarsePrivateBucket(
    ComboId hand,
    StreetKind street,
    const BoardFeatures& features) noexcept {
  const ComboInfo& combo = GetComboInfo(hand);
  const int rank0 = PokerRank(combo.card0);
  const int rank1 = PokerRank(combo.card1);
  const int high = std::max(rank0, rank1);
  const int low = std::min(rank0, rank1);
  const bool pair = rank0 == rank1;
  const bool suited = combo.card0.suit() == combo.card1.suit();
  if (street == StreetKind::kPreflop || features.card_count == 0) {
    const int shape = pair ? 0 : (suited ? 1 : 2);
    return static_cast<PrivateBucketId>(
        shape * 12 + HighRankGroup(high) * 3 + LowRankGroup(low));
  }
  return static_cast<PrivateBucketId>(
      MadeBucket(combo, features) * 9 + DrawBucket(combo, features) * 3 +
      HoleStrengthBucket(high, low, pair, suited));
}

namespace {

PrivateBucketId PreflopHandClass(ComboId hand) noexcept {
  const ComboInfo& combo = GetComboInfo(hand);
  int first = static_cast<int>(combo.card0.rank());
  int second = static_cast<int>(combo.card1.rank());
  if (first < second) std::swap(first, second);
  if (first == second) return static_cast<PrivateBucketId>(first);
  const int offset = first * (first - 1) / 2 + second;
  const bool suited = combo.card0.suit() == combo.card1.suit();
  return static_cast<PrivateBucketId>((suited ? 13 : 91) + offset);
}

PrivateObservationId HandcraftedObservation(
    const CardAbstractionConfig& config,
    ComboId hand,
    const PublicPosition& position) noexcept {
  const PrivateBucketId current = CoarsePrivateBucket(
      hand, position.street(), position.features());
  if (config.recall_mode == RecallMode::kCurrentBucketOnly) {
    return PrivateObservationId(static_cast<uint64_t>(current) + 1);
  }

  uint64_t observation = static_cast<uint64_t>(CoarsePrivateBucket(
      hand, StreetKind::kPreflop, BoardFeatures{})) + 1;
  if (position.street() == StreetKind::kPreflop) {
    return PrivateObservationId(observation);
  }
  const auto cards = BoardCards(position.board());
  const FlopBoard flop =
      DealFlop(PreflopBoard{}, {cards[0], cards[1], cards[2]});
  observation |=
      (static_cast<uint64_t>(CoarsePrivateBucket(
           hand, StreetKind::kFlop, BoardFeaturesFor(Board{flop}))) + 1)
      << 6;
  if (position.street() == StreetKind::kFlop) {
    return PrivateObservationId(observation);
  }
  const TurnBoard turn = DealTurn(flop, cards[3]);
  observation |=
      (static_cast<uint64_t>(CoarsePrivateBucket(
           hand, StreetKind::kTurn, BoardFeaturesFor(Board{turn}))) + 1)
      << 12;
  if (position.street() == StreetKind::kTurn) {
    return PrivateObservationId(observation);
  }
  const RiverBoard river = DealRiver(turn, cards[4]);
  observation |=
      (static_cast<uint64_t>(CoarsePrivateBucket(
           hand, StreetKind::kRiver, BoardFeaturesFor(Board{river}))) + 1)
      << 18;
  return PrivateObservationId(observation);
}

PrivateObservationId EquityObservation(
    const CardAbstraction& abstraction,
    ComboId hand,
    const PublicPosition& position) noexcept {
  const uint64_t preflop =
      static_cast<uint64_t>(PreflopHandClass(hand)) + 1;
  if (position.street() == StreetKind::kPreflop) {
    return PrivateObservationId(preflop);
  }
  const PrivateBucketId current =
      abstraction.equity_bucket(hand, position.board());
  if (abstraction.config().recall_mode == RecallMode::kCurrentBucketOnly) {
    return PrivateObservationId(static_cast<uint64_t>(current) + 1);
  }

  const auto cards = BoardCards(position.board());
  const FlopBoard flop =
      DealFlop(PreflopBoard{}, {cards[0], cards[1], cards[2]});
  uint64_t observation = preflop;
  observation |=
      (static_cast<uint64_t>(abstraction.equity_bucket(hand, Board{flop})) + 1)
      << 8;
  if (position.street() == StreetKind::kFlop) {
    return PrivateObservationId(observation);
  }
  const TurnBoard turn = DealTurn(flop, cards[3]);
  observation |=
      (static_cast<uint64_t>(abstraction.equity_bucket(hand, Board{turn})) + 1)
      << 14;
  if (position.street() == StreetKind::kTurn) {
    return PrivateObservationId(observation);
  }
  const RiverBoard river = DealRiver(turn, cards[4]);
  observation |=
      (static_cast<uint64_t>(abstraction.equity_bucket(hand, Board{river})) + 1)
      << 20;
  return PrivateObservationId(observation);
}

}  // namespace

absl::StatusOr<CardAbstraction> CardAbstraction::Create(
    CardAbstractionConfig config) {
  if (config.private_kind == PrivateAbstractionKind::kEquityPotential) {
    if (!config.equity_model.has_value()) {
      return absl::InvalidArgumentError(
          "equity private abstraction requires a model");
    }
    const absl::Status valid =
        ValidateEquityBucketModel(*config.equity_model);
    if (!valid.ok()) return valid;
  } else if (config.equity_model.has_value()) {
    return absl::InvalidArgumentError(
        "equity model provided for a non-equity abstraction");
  }
  return CardAbstraction(std::move(config));
}

PrivateBucketId CardAbstraction::equity_bucket(
    ComboId hand,
    const Board& board) const noexcept {
  assert(config_.equity_model.has_value());
  const CanonicalCardObservation canonical =
      CanonicalizeObservation(hand, board);
  const CacheKey key{canonical.public_observation,
                     canonical.private_observation};
  const auto found = equity_cache_.find(key);
  if (found != equity_cache_.end()) return found->second;
  const PrivateBucketId bucket = EquityBucket(
      static_cast<StreetKind>(BoardCount(board) - 2),
      EvaluateEquityFeatures(hand, board, *config_.equity_model),
      *config_.equity_model);
  equity_cache_.emplace(key, bucket);
  return bucket;
}

PublicObservationId ObservePublic(const CardAbstraction& abstraction,
                                  StreetKind street,
                                  const Board& board) noexcept {
  switch (abstraction.config().public_mode) {
    case PublicCardMode::kExactCanonical:
      return CanonicalPublicObservation(board);
    case PublicCardMode::kTexture:
      return ObserveCoarsePublic(street, board);
  }
}

PrivateObservationId ObservePrivate(const CardAbstraction& abstraction,
                                    ComboId hand,
                                    const PublicPosition& position) noexcept {
  switch (abstraction.config().private_kind) {
    case PrivateAbstractionKind::kExactCanonical:
      return CanonicalizeObservation(hand, position.board())
          .private_observation;
    case PrivateAbstractionKind::kHandcrafted36:
      return HandcraftedObservation(abstraction.config(), hand, position);
    case PrivateAbstractionKind::kEquityPotential:
      return EquityObservation(abstraction, hand, position);
  }
}

PublicPosition PublicPosition::Root(const CardAbstraction& abstraction,
                                    StreetKind street,
                                    Board board) {
  const BoardFeatures features = BoardFeaturesFor(board);
  const PublicObservationId observation =
      ObservePublic(abstraction, street, board);
  return PublicPosition(street, std::move(board), observation, features);
}

PublicPosition PublicPosition::after_chance(
    const CardAbstraction& abstraction,
    StreetKind street,
    Board board) const {
  const BoardFeatures features = BoardFeaturesFor(board);
  PublicObservationId observation;
  if (abstraction.config().public_mode == PublicCardMode::kExactCanonical) {
    observation = CanonicalPublicObservation(board);
  } else {
    observation = AdvanceCoarsePublic(
        observation_, street, BoardTextureBucket(street, features));
  }
  return PublicPosition(street, std::move(board), observation, features);
}

}  // namespace poker
