#include "src/solver.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <type_traits>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"

namespace poker {
namespace {

class Sha256 {
 public:
  void add(absl::Span<const uint8_t> bytes) noexcept {
    total_bytes_ += bytes.size();
    for (uint8_t byte : bytes) {
      block_[block_size_++] = byte;
      if (block_size_ == block_.size()) {
        transform();
        block_size_ = 0;
      }
    }
  }

  ModelFingerprint finish() noexcept {
    const uint64_t bit_count = total_bytes_ * 8;
    const uint8_t marker = 0x80;
    add(absl::Span<const uint8_t>(&marker, 1));
    const uint8_t zero = 0;
    while (block_size_ != 56) {
      add(absl::Span<const uint8_t>(&zero, 1));
    }
    std::array<uint8_t, 8> length = {};
    for (size_t index = 0; index < length.size(); ++index) {
      length[7 - index] =
          static_cast<uint8_t>(bit_count >> static_cast<unsigned>(index * 8));
    }
    add(length);

    ModelFingerprint fingerprint;
    for (size_t word = 0; word < state_.size(); ++word) {
      for (size_t byte = 0; byte < 4; ++byte) {
        fingerprint.bytes[word * 4 + byte] = static_cast<std::byte>(
            state_[word] >> static_cast<unsigned>((3 - byte) * 8));
      }
    }
    return fingerprint;
  }

 private:
  static constexpr std::array<uint32_t, 64> kRoundConstants = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
      0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
      0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
      0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
      0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
      0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
      0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
      0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
      0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
      0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
      0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
      0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
      0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
      0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
  };

  void transform() noexcept {
    std::array<uint32_t, 64> words = {};
    for (size_t index = 0; index < 16; ++index) {
      const size_t offset = index * 4;
      words[index] =
          (static_cast<uint32_t>(block_[offset]) << 24) |
          (static_cast<uint32_t>(block_[offset + 1]) << 16) |
          (static_cast<uint32_t>(block_[offset + 2]) << 8) |
          static_cast<uint32_t>(block_[offset + 3]);
    }
    for (size_t index = 16; index < words.size(); ++index) {
      const uint32_t s0 = std::rotr(words[index - 15], 7) ^
                          std::rotr(words[index - 15], 18) ^
                          (words[index - 15] >> 3);
      const uint32_t s1 = std::rotr(words[index - 2], 17) ^
                          std::rotr(words[index - 2], 19) ^
                          (words[index - 2] >> 10);
      words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }

    uint32_t a = state_[0];
    uint32_t b = state_[1];
    uint32_t c = state_[2];
    uint32_t d = state_[3];
    uint32_t e = state_[4];
    uint32_t f = state_[5];
    uint32_t g = state_[6];
    uint32_t h = state_[7];
    for (size_t index = 0; index < words.size(); ++index) {
      const uint32_t s1 =
          std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
      const uint32_t choice = (e & f) ^ (~e & g);
      const uint32_t temp1 =
          h + s1 + choice + kRoundConstants[index] + words[index];
      const uint32_t s0 =
          std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
      const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t temp2 = s0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<uint32_t, 8> state_ = {
      0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  };
  std::array<uint8_t, 64> block_ = {};
  size_t block_size_ = 0;
  uint64_t total_bytes_ = 0;
};

class FingerprintBuilder {
 public:
  void add_u8(uint8_t value) noexcept {
    hash_.add(absl::Span<const uint8_t>(&value, 1));
  }

  void add_u32(uint32_t value) noexcept {
    std::array<uint8_t, 4> bytes = {};
    for (size_t index = 0; index < bytes.size(); ++index) {
      bytes[index] =
          static_cast<uint8_t>(value >> static_cast<unsigned>(index * 8));
    }
    hash_.add(bytes);
  }

  void add_u64(uint64_t value) noexcept {
    std::array<uint8_t, 8> bytes = {};
    for (size_t index = 0; index < bytes.size(); ++index) {
      bytes[index] =
          static_cast<uint8_t>(value >> static_cast<unsigned>(index * 8));
    }
    hash_.add(bytes);
  }

  void add_i32(int32_t value) noexcept {
    add_u32(static_cast<uint32_t>(value));
  }

  void add_float(float value) noexcept {
    add_u32(std::bit_cast<uint32_t>(value));
  }

  void add_double(double value) noexcept {
    add_u64(std::bit_cast<uint64_t>(value));
  }

  ModelFingerprint finish() noexcept { return hash_.finish(); }

 private:
  Sha256 hash_;
};

void AddBettingData(FingerprintBuilder& hash,
                    const BettingData& data) noexcept {
  for (Chips value : data.stack) hash.add_i32(value);
  for (Chips value : data.total_committed) hash.add_i32(value);
  for (Chips value : data.street_committed) hash.add_i32(value);
  hash.add_i32(data.last_full_raise);
  hash.add_u8(static_cast<uint8_t>(data.street));
  hash.add_u8(data.pending_action_mask);
}

void AddBettingState(FingerprintBuilder& hash,
                     const BettingState& state) noexcept {
  std::visit([&](const auto& phase) {
    using Phase = std::decay_t<decltype(phase)>;
    if constexpr (std::is_same_v<Phase, DecisionState>) {
      hash.add_u8(0);
      AddBettingData(hash, phase.data);
      hash.add_u8(static_cast<uint8_t>(phase.actor));
    } else if constexpr (std::is_same_v<Phase, ChanceState>) {
      hash.add_u8(1);
      AddBettingData(hash, phase.data);
    } else if constexpr (std::is_same_v<Phase, FoldTerminalState>) {
      hash.add_u8(2);
      AddBettingData(hash, phase.data);
      hash.add_u8(static_cast<uint8_t>(phase.folded));
    } else {
      hash.add_u8(3);
      AddBettingData(hash, phase.data);
    }
  }, state);
}

void AddBoard(FingerprintBuilder& hash, const Board& board) noexcept {
  hash.add_u8(BoardCount(board));
  for (Card card : BoardCards(board)) {
    hash.add_u8(static_cast<uint8_t>(card.index()));
  }
}

void AddRange(FingerprintBuilder& hash, const ComboRange& range) noexcept {
  hash.add_u32(range.active_count);
  for (uint16_t index = 0; index < range.active_count; ++index) {
    hash.add_u32(static_cast<uint32_t>(range.active[index].index()));
  }
  for (float weight : range.weights) hash.add_float(weight);
}

ModelFingerprint FingerprintModel(const SolveSpec& spec,
                                  const HistoryTree& history) noexcept {
  FingerprintBuilder hash;
  hash.add_u32(1);  // Fingerprint schema.
  hash.add_u32(1);  // Exact poker rules.
  hash.add_u32(1);  // Card abstraction implementation.
  hash.add_u32(1);  // Perfect-recall observation encoding.

  const SolverConfig& config = spec.config;
  hash.add_i32(config.starting_stack());
  hash.add_i32(config.small_blind());
  hash.add_i32(config.big_blind());
  hash.add_i32(config.chance_samples());
  hash.add_i32(config.max_info_sets());
  hash.add_u8(config.accumulate_average_strategy() ? 1 : 0);
  hash.add_u8(static_cast<uint8_t>(config.card_abstraction().public_mode));
  hash.add_u8(static_cast<uint8_t>(config.card_abstraction().private_mode));
  for (const auto& fractions :
       config.bet_abstraction().pot_fractions) {
    hash.add_u32(static_cast<uint32_t>(fractions.size()));
    for (double fraction : fractions) hash.add_double(fraction);
  }

  AddBettingState(hash, spec.root.betting);
  AddBoard(hash, spec.root.board);
  for (const ComboRange& range : spec.ranges) AddRange(hash, range);

  hash.add_u32(history.root.value());
  hash.add_u32(static_cast<uint32_t>(history.nodes.size()));
  for (const HistoryNode& node : history.nodes) {
    std::visit([&](const auto& value) {
      using Node = std::decay_t<decltype(value)>;
      if constexpr (std::is_same_v<Node, DecisionNode>) {
        hash.add_u8(0);
        AddBettingData(hash, value.state.data);
        hash.add_u8(static_cast<uint8_t>(value.state.actor));
        hash.add_u32(value.edges.begin);
        hash.add_u8(value.edges.count);
      } else if constexpr (std::is_same_v<Node, ChanceNode>) {
        hash.add_u8(1);
        AddBettingData(hash, value.state.data);
        hash.add_u32(value.child.value());
      } else if constexpr (std::is_same_v<Node, FoldTerminalNode>) {
        hash.add_u8(2);
        AddBettingData(hash, value.state.data);
        hash.add_u8(static_cast<uint8_t>(value.state.folded));
      } else {
        hash.add_u8(3);
        AddBettingData(hash, value.state.data);
      }
    }, node);
  }
  hash.add_u32(static_cast<uint32_t>(history.edges.size()));
  for (const HistoryEdge& edge : history.edges) {
    hash.add_u8(static_cast<uint8_t>(edge.action.kind));
    hash.add_i32(edge.action.target_street_commitment);
    hash.add_u32(edge.child.value());
  }
  return hash.finish();
}

using Bytes = std::vector<uint8_t>;

void AppendU8(Bytes& bytes, uint8_t value) {
  bytes.push_back(value);
}

void AppendU32(Bytes& bytes, uint32_t value) {
  for (size_t index = 0; index < 4; ++index) {
    bytes.push_back(
        static_cast<uint8_t>(value >> static_cast<unsigned>(index * 8)));
  }
}

void AppendU64(Bytes& bytes, uint64_t value) {
  for (size_t index = 0; index < 8; ++index) {
    bytes.push_back(
        static_cast<uint8_t>(value >> static_cast<unsigned>(index * 8)));
  }
}

void AppendFingerprint(Bytes& bytes, ModelFingerprint fingerprint) {
  for (std::byte byte : fingerprint.bytes) {
    bytes.push_back(std::to_integer<uint8_t>(byte));
  }
}

class ByteReader {
 public:
  explicit ByteReader(absl::Span<const uint8_t> bytes) : bytes_(bytes) {}

  std::optional<uint8_t> read_u8() {
    if (remaining() < 1) return std::nullopt;
    return bytes_[offset_++];
  }

  std::optional<uint32_t> read_u32() {
    if (remaining() < 4) return std::nullopt;
    uint32_t value = 0;
    for (size_t index = 0; index < 4; ++index) {
      value |= static_cast<uint32_t>(bytes_[offset_++])
               << static_cast<unsigned>(index * 8);
    }
    return value;
  }

  std::optional<uint64_t> read_u64() {
    if (remaining() < 8) return std::nullopt;
    uint64_t value = 0;
    for (size_t index = 0; index < 8; ++index) {
      value |= static_cast<uint64_t>(bytes_[offset_++])
               << static_cast<unsigned>(index * 8);
    }
    return value;
  }

  std::optional<ModelFingerprint> read_fingerprint() {
    if (remaining() < ModelFingerprint{}.bytes.size()) return std::nullopt;
    ModelFingerprint fingerprint;
    for (std::byte& byte : fingerprint.bytes) {
      byte = static_cast<std::byte>(bytes_[offset_++]);
    }
    return fingerprint;
  }

  std::optional<absl::Span<const uint8_t>> read_bytes(size_t count) {
    if (remaining() < count) return std::nullopt;
    const auto result = bytes_.subspan(offset_, count);
    offset_ += count;
    return result;
  }

  size_t remaining() const { return bytes_.size() - offset_; }

 private:
  absl::Span<const uint8_t> bytes_;
  size_t offset_ = 0;
};

absl::Status WriteBytes(const std::filesystem::path& path,
                        absl::Span<const uint8_t> bytes) {
  std::filesystem::path temporary = path;
  temporary += ".tmp";
  std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
  if (!output) {
    return absl::UnavailableError("could not open output file");
  }
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  output.close();
  if (!output) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    return absl::DataLossError("could not write output file");
  }
  std::error_code error;
  std::filesystem::rename(temporary, path, error);
  if (error) {
    std::filesystem::remove(temporary, error);
    return absl::UnavailableError("could not replace output file");
  }
  return absl::OkStatus();
}

absl::StatusOr<Bytes> ReadBytes(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    return absl::NotFoundError("could not open input file");
  }
  const std::streamoff end = input.tellg();
  if (end < 0 || static_cast<uint64_t>(end) >
                     std::numeric_limits<size_t>::max()) {
    return absl::DataLossError("invalid input file size");
  }
  Bytes bytes(static_cast<size_t>(end));
  input.seekg(0);
  input.read(reinterpret_cast<char*>(bytes.data()), end);
  if (!input) {
    return absl::DataLossError("could not read input file");
  }
  return bytes;
}

bool KeyLess(const InfoSetKey& left, const InfoSetKey& right) {
  if (left.history != right.history) {
    return left.history < right.history;
  }
  if (left.public_observation != right.public_observation) {
    return left.public_observation < right.public_observation;
  }
  return left.private_observation < right.private_observation;
}

constexpr int kHandTypeCount = 169;

enum class HandShape {
  kPair,
  kSuited,
  kOffsuit,
  kAny,
};

struct HandType {
  int high = 0;
  int low = 0;
  HandShape shape = HandShape::kPair;
};

std::optional<int> ParseRank(char rank) {
  switch (rank) {
    case 'A': return 14;
    case 'K': return 13;
    case 'Q': return 12;
    case 'J': return 11;
    case 'T': return 10;
    default:
      return rank >= '2' && rank <= '9'
                 ? std::optional<int>(rank - '0')
                 : std::nullopt;
  }
}

int NonPairOffset(int high, int low) {
  high -= 2;
  low -= 2;
  return high * (high - 1) / 2 + low;
}

std::optional<int> HandTypeIndex(HandType type) {
  if (type.high < type.low) {
    std::swap(type.high, type.low);
  }
  if (type.low < 2 || type.high > 14) {
    return std::nullopt;
  }
  if (type.high == type.low) {
    return type.shape == HandShape::kPair
               ? std::optional<int>(type.high - 2)
               : std::nullopt;
  }
  const int offset = NonPairOffset(type.high, type.low);
  if (type.shape == HandShape::kSuited) {
    return 13 + offset;
  }
  return type.shape == HandShape::kOffsuit
             ? std::optional<int>(91 + offset)
             : std::nullopt;
}

std::optional<HandType> DecodeHandType(int index) {
  if (index < 0 || index >= kHandTypeCount) {
    return std::nullopt;
  }
  if (index < 13) {
    return HandType{index + 2, index + 2, HandShape::kPair};
  }
  const bool suited = index < 91;
  int offset = suited ? index - 13 : index - 91;
  int high = 1;
  while (offset >= high * (high + 1) / 2) {
    ++high;
  }
  const int low = offset - high * (high - 1) / 2;
  return HandType{high + 2, low + 2,
                  suited ? HandShape::kSuited : HandShape::kOffsuit};
}

std::optional<HandType> ParseHandType(std::string_view text) {
  if (text.size() != 2 && text.size() != 3) {
    return std::nullopt;
  }
  const auto first = ParseRank(text[0]);
  const auto second = ParseRank(text[1]);
  if (!first || !second) {
    return std::nullopt;
  }
  HandType type{std::max(*first, *second), std::min(*first, *second),
                HandShape::kPair};
  if (type.high == type.low) {
    return text.size() == 2 ? std::optional<HandType>(type) : std::nullopt;
  }
  if (text.size() == 2) {
    type.shape = HandShape::kAny;
  } else if (text[2] == 's') {
    type.shape = HandShape::kSuited;
  } else if (text[2] == 'o') {
    type.shape = HandShape::kOffsuit;
  } else {
    return std::nullopt;
  }
  return type;
}

std::vector<ComboId> Expand(HandType type) {
  constexpr std::array<Suit, 4> suits = {
      Suit::kHearts, Suit::kDiamonds,
      Suit::kClubs, Suit::kSpades};
  std::vector<ComboId> combos;
  for (size_t first = 0; first < suits.size(); ++first) {
    for (size_t second = 0; second < suits.size(); ++second) {
      if (type.high == type.low && first >= second) {
        continue;
      }
      const bool suited = first == second;
      if (type.high != type.low && type.shape != HandShape::kAny &&
          suited != (type.shape == HandShape::kSuited)) {
        continue;
      }
      combos.push_back(CardsToComboId(
          Card(static_cast<Rank>(type.high - 2), suits[first]),
          Card(static_cast<Rank>(type.low - 2), suits[second])));
    }
  }
  return combos;
}

std::string_view Trim(std::string_view text) {
  const size_t first = text.find_first_not_of(" \t");
  if (first == std::string_view::npos) {
    return {};
  }
  const size_t last = text.find_last_not_of(" \t");
  return text.substr(first, last - first + 1);
}

ComboRange ExpandSelectedCombos(const std::vector<int>& selected) {
  ComboRange range;
  for (int index : selected) {
    const auto combos = Expand(*DecodeHandType(index));
    for (ComboId combo : combos) {
      range.add(combo, 1.0f);
    }
  }
  return range;
}

HistoryId AppendHistory(HistoryTree& tree,
                        const BettingState& state,
                        const BettingRules& rules,
                        const SolverConfig& config) {
  const HistoryId id(static_cast<uint32_t>(tree.nodes.size()));
  if (const auto* decision = std::get_if<DecisionState>(&state)) {
    const LegalActionSpace legal = LegalActions(*decision);
    const AbstractActions actions = SelectAbstractActions(
        config.bet_abstraction(), *decision, legal);
    assert(actions.size() <= std::numeric_limits<uint8_t>::max());
    const uint32_t begin = static_cast<uint32_t>(tree.edges.size());
    for (const GameAction& action : actions) {
      tree.edges.push_back({action, id});
    }
    tree.nodes.push_back(
        DecisionNode{*decision, {begin, static_cast<uint8_t>(actions.size())}});
    for (size_t index = 0; index < actions.size(); ++index) {
      const auto child_state = ApplyAction(*decision, actions[index]);
      assert(child_state.ok());
      const HistoryId child = AppendHistory(tree, *child_state, rules, config);
      tree.edges[begin + index] = {actions[index], child};
    }
    return id;
  }

  if (const auto* chance = std::get_if<ChanceState>(&state)) {
    tree.nodes.push_back(ChanceNode{*chance, id});
    const BettingState child_state = AdvanceBettingStreet(*chance, rules);
    const HistoryId child = AppendHistory(tree, child_state, rules, config);
    auto* node = std::get_if<ChanceNode>(&tree.nodes[id.index()]);
    assert(node != nullptr);
    node->child = child;
    return id;
  }

  if (const auto* fold = std::get_if<FoldTerminalState>(&state)) {
    tree.nodes.push_back(FoldTerminalNode{*fold});
  } else {
    const auto* showdown = std::get_if<ShowdownState>(&state);
    assert(showdown != nullptr);
    tree.nodes.push_back(ShowdownNode{*showdown});
  }
  return id;
}

HistoryTree BuildHistoryTree(const BettingState& root,
                             const BettingRules& rules,
                             const SolverConfig& config) {
  HistoryTree tree;
  tree.nodes.reserve(4096);
  tree.edges.reserve(4096);
  tree.root = AppendHistory(tree, root, rules, config);
  return tree;
}

}  // namespace

absl::StatusOr<DealDistribution> DealDistribution::Create(
    const ComboRange& player_a,
    const ComboRange& player_b) {
  DealDistribution distribution;
  distribution.a_hands_.reserve(player_a.count());
  distribution.a_cumulative_.reserve(player_a.count());
  distribution.b_offsets_.reserve(player_a.count());
  distribution.b_counts_.reserve(player_a.count());
  distribution.b_hands_.reserve(player_a.count() * player_b.count());
  distribution.b_cumulative_.reserve(player_a.count() * player_b.count());

  for (uint16_t a_index = 0; a_index < player_a.active_count; ++a_index) {
    const ComboId a = player_a.active[a_index];
    const float a_weight = player_a.weight(a);
    if (a_weight <= 0.0f) {
      continue;
    }
    const uint32_t offset =
        static_cast<uint32_t>(distribution.b_hands_.size());
    float b_total = 0.0f;
    for (uint16_t b_index = 0; b_index < player_b.active_count; ++b_index) {
      const ComboId b = player_b.active[b_index];
      const float b_weight = player_b.weight(b);
      if (b_weight <= 0.0f || (ComboMask(a) & ComboMask(b)) != 0) {
        continue;
      }
      b_total += b_weight;
      distribution.b_hands_.push_back(b);
      distribution.b_cumulative_.push_back(b_total);
    }
    if (b_total <= 0.0f) {
      continue;
    }
    distribution.a_hands_.push_back(a);
    distribution.b_offsets_.push_back(offset);
    distribution.b_counts_.push_back(static_cast<uint16_t>(
        distribution.b_hands_.size() - offset));
    distribution.total_ += a_weight * b_total;
    distribution.a_cumulative_.push_back(distribution.total_);
  }
  if (distribution.total_ <= 0.0f) {
    return absl::InvalidArgumentError(
        "ranges contain no non-overlapping hands");
  }
  return distribution;
}

Deal DealDistribution::sample(std::mt19937& rng) const {
  const size_t a_index = SampleIndex(a_cumulative_, total_, rng);
  const size_t offset = b_offsets_[a_index];
  const uint16_t count = b_counts_[a_index];
  const float b_total = b_cumulative_[offset + count - 1];
  const size_t relative = SampleIndex(
      absl::Span<const float>(b_cumulative_).subspan(offset, count),
      b_total, rng);
  const ComboId a = a_hands_[a_index];
  const ComboId b = b_hands_[offset + relative];
  return {{HoleCards(a), HoleCards(b)}, ComboMask(a) | ComboMask(b)};
}

size_t DealDistribution::SampleIndex(absl::Span<const float> cumulative,
                                     float total,
                                     std::mt19937& rng) {
  std::uniform_real_distribution<float> distribution(0.0f, total);
  const auto found = std::upper_bound(
      cumulative.begin(), cumulative.end(), distribution(rng));
  return found == cumulative.end()
             ? cumulative.size() - 1
             : static_cast<size_t>(found - cumulative.begin());
}

namespace {

void FillUniform(absl::Span<double> probabilities) {
  if (!probabilities.empty()) {
    std::fill(probabilities.begin(), probabilities.end(),
              1.0 / probabilities.size());
  }
}

void RegretMatch(const CfrState& state,
                 const InfoSetRow* row,
                 absl::Span<double> probabilities) {
  if (row == nullptr) {
    FillUniform(probabilities);
    return;
  }

  double sum = 0.0;
  for (size_t action = 0; action < probabilities.size(); ++action) {
    const size_t index = static_cast<size_t>(row->action_offset) + action;
    const double regret = std::max(0.0, static_cast<double>(state.regret_sum[index]));
    probabilities[action] = regret;
    sum += regret;
  }
  if (sum <= 0.0) {
    FillUniform(probabilities);
    return;
  }
  for (double& probability : probabilities) {
    probability /= sum;
  }
}

void AverageStrategy(const CfrState& state,
                     const InfoSetRow* row,
                     absl::Span<double> probabilities) {
  if (row == nullptr) {
    FillUniform(probabilities);
    return;
  }

  double sum = 0.0;
  for (size_t action = 0; action < probabilities.size(); ++action) {
    const size_t index = static_cast<size_t>(row->action_offset) + action;
    const float value = state.strategy_sum[index];
    probabilities[action] = std::max(0.0, static_cast<double>(value));
    sum += probabilities[action];
  }
  if (sum <= 0.0) {
    FillUniform(probabilities);
    return;
  }
  for (double& probability : probabilities) {
    probability /= sum;
  }
}

void AddCfrPlusRegret(CfrState& state,
                      InfoSetRow row,
                      size_t action,
                      float delta) {
  const size_t index = static_cast<size_t>(row.action_offset) + action;
  state.regret_sum[index] = std::max(0.0f, state.regret_sum[index] + delta);
}

void AddStrategySum(CfrState& state,
                    InfoSetRow row,
                    absl::Span<const double> probabilities,
                    double weight) {
  for (size_t action = 0; action < probabilities.size(); ++action) {
    const size_t index = static_cast<size_t>(row.action_offset) + action;
    state.strategy_sum[index] +=
        static_cast<float>(weight * probabilities[action]);
  }
}

}  // namespace

bool Policy::strategy(InfoSetKey key, absl::Span<float> output) const {
  const auto found = rows.find(key);
  if (found == rows.end() || found->second.action_count != output.size() ||
      found->second.action_offset + output.size() > probabilities.size()) {
    if (!output.empty()) {
      std::fill(output.begin(), output.end(), 1.0f / output.size());
    }
    return false;
  }
  const size_t offset = found->second.action_offset;
  std::copy_n(probabilities.begin() + static_cast<ptrdiff_t>(offset),
              output.size(), output.begin());
  return true;
}

namespace {

absl::Status ValidatePolicy(const Policy& policy) {
  std::vector<PolicyRow> rows;
  rows.reserve(policy.rows.size());
  for (const auto& [key, row] : policy.rows) {
    (void)key;
    if (row.action_count == 0 ||
        row.action_offset + row.action_count > policy.probabilities.size()) {
      return absl::DataLossError("invalid policy row span");
    }
    double sum = 0.0;
    for (size_t action = 0; action < row.action_count; ++action) {
      const float probability =
          policy.probabilities[row.action_offset + action];
      if (!std::isfinite(probability) || probability < 0.0f ||
          probability > 1.0f) {
        return absl::DataLossError("invalid policy probability");
      }
      sum += probability;
    }
    if (std::abs(sum - 1.0) > 1e-5) {
      return absl::DataLossError("policy row is not normalized");
    }
    rows.push_back(row);
  }
  std::sort(rows.begin(), rows.end(), [](PolicyRow left, PolicyRow right) {
    return left.action_offset < right.action_offset;
  });
  size_t offset = 0;
  for (PolicyRow row : rows) {
    if (row.action_offset != offset) {
      return absl::DataLossError("policy rows are not contiguous");
    }
    offset += row.action_count;
  }
  return offset == policy.probabilities.size()
             ? absl::OkStatus()
             : absl::DataLossError("policy array size does not match rows");
}

constexpr std::array<uint8_t, 8> kPolicyMagic = {
    'P', 'K', 'P', 'O', 'L', 'C', 'Y', '1'};

}  // namespace

absl::Status SavePolicy(const Policy& policy,
                        const std::filesystem::path& path) {
  const absl::Status valid = ValidatePolicy(policy);
  if (!valid.ok()) return valid;

  std::vector<std::pair<InfoSetKey, PolicyRow>> rows(
      policy.rows.begin(), policy.rows.end());
  std::sort(rows.begin(), rows.end(), [](const auto& left, const auto& right) {
    return KeyLess(left.first, right.first);
  });

  Bytes bytes;
  bytes.insert(bytes.end(), kPolicyMagic.begin(), kPolicyMagic.end());
  AppendU32(bytes, 1);
  AppendFingerprint(bytes, policy.model);
  AppendU64(bytes, rows.size());
  AppendU64(bytes, policy.probabilities.size());
  for (const auto& [key, row] : rows) {
    AppendU32(bytes, key.history.value());
    AppendU64(bytes, key.public_observation.value());
    AppendU64(bytes, key.private_observation.value());
    AppendU64(bytes, row.action_offset);
    AppendU8(bytes, row.action_count);
  }
  for (float probability : policy.probabilities) {
    AppendU32(bytes, std::bit_cast<uint32_t>(probability));
  }
  return WriteBytes(path, bytes);
}

absl::StatusOr<Policy> LoadPolicy(const std::filesystem::path& path) {
  const auto file = ReadBytes(path);
  if (!file.ok()) return file.status();
  ByteReader reader(*file);
  for (uint8_t expected : kPolicyMagic) {
    const auto actual = reader.read_u8();
    if (!actual || *actual != expected) {
      return absl::DataLossError("invalid policy file magic");
    }
  }
  const auto version = reader.read_u32();
  const auto model = reader.read_fingerprint();
  const auto row_count = reader.read_u64();
  const auto probability_count = reader.read_u64();
  if (!version || *version != 1 || !model || !row_count ||
      !probability_count || *row_count > reader.remaining() / 29 ||
      *probability_count > reader.remaining() / sizeof(float)) {
    return absl::DataLossError("invalid policy file header");
  }

  Policy policy;
  policy.model = *model;
  policy.rows.reserve(static_cast<size_t>(*row_count));
  for (uint64_t index = 0; index < *row_count; ++index) {
    const auto history = reader.read_u32();
    const auto public_observation = reader.read_u64();
    const auto private_observation = reader.read_u64();
    const auto offset = reader.read_u64();
    const auto action_count = reader.read_u8();
    if (!history || !public_observation || !private_observation || !offset ||
        !action_count || *offset > std::numeric_limits<size_t>::max()) {
      return absl::DataLossError("truncated policy row");
    }
    const InfoSetKey key{HistoryId(*history),
                         PublicObservationId(*public_observation),
                         PrivateObservationId(*private_observation)};
    const PolicyRow row{static_cast<size_t>(*offset), *action_count};
    if (!policy.rows.emplace(key, row).second) {
      return absl::DataLossError("duplicate policy row");
    }
  }
  policy.probabilities.resize(static_cast<size_t>(*probability_count));
  for (float& probability : policy.probabilities) {
    const auto bits = reader.read_u32();
    if (!bits) return absl::DataLossError("truncated policy probabilities");
    probability = std::bit_cast<float>(*bits);
  }
  if (reader.remaining() != 0) {
    return absl::DataLossError("trailing policy data");
  }
  const absl::Status valid = ValidatePolicy(policy);
  if (!valid.ok()) return valid;
  return policy;
}

namespace {

SerializedRngState SerializeRng(const std::mt19937& rng) {
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << rng;
  return {output.str()};
}

absl::StatusOr<std::mt19937> DeserializeRng(
    const SerializedRngState& serialized) {
  std::istringstream input(serialized.text);
  input.imbue(std::locale::classic());
  std::mt19937 rng;
  input >> rng;
  if (!input) {
    return absl::DataLossError("invalid RNG state");
  }
  input >> std::ws;
  if (!input.eof()) {
    return absl::DataLossError("trailing RNG state data");
  }
  return rng;
}

absl::Status ValidateCheckpointState(const CfrState& state) {
  if (!std::isfinite(state.cumulative_root_utility)) {
    return absl::DataLossError("nonfinite cumulative utility");
  }
  for (float value : state.regret_sum) {
    if (!std::isfinite(value)) {
      return absl::DataLossError("nonfinite regret value");
    }
  }
  for (float value : state.strategy_sum) {
    if (!std::isfinite(value)) {
      return absl::DataLossError("nonfinite strategy value");
    }
  }
  if (!state.strategy_sum.empty() &&
      state.strategy_sum.size() != state.regret_sum.size()) {
    return absl::DataLossError("CFR arrays have different sizes");
  }

  std::vector<size_t> offsets;
  offsets.reserve(state.rows.size());
  for (const auto& [key, row] : state.rows) {
    (void)key;
    offsets.push_back(row.action_offset);
  }
  std::sort(offsets.begin(), offsets.end());
  if (offsets.empty()) {
    return state.regret_sum.empty()
               ? absl::OkStatus()
               : absl::DataLossError("CFR arrays have no rows");
  }
  if (offsets.front() != 0 || offsets.back() >= state.regret_sum.size()) {
    return absl::DataLossError("invalid CFR row offsets");
  }
  for (size_t index = 1; index < offsets.size(); ++index) {
    if (offsets[index] <= offsets[index - 1]) {
      return absl::DataLossError("duplicate CFR row offsets");
    }
  }
  return absl::OkStatus();
}

constexpr std::array<uint8_t, 8> kCheckpointMagic = {
    'P', 'K', 'C', 'H', 'E', 'C', 'K', '1'};

}  // namespace

absl::Status SaveCheckpoint(const SolverCheckpoint& checkpoint,
                            const std::filesystem::path& path) {
  if (checkpoint.format_version != 1) {
    return absl::InvalidArgumentError("unsupported checkpoint version");
  }
  const absl::Status valid = ValidateCheckpointState(checkpoint.state);
  if (!valid.ok()) return valid;
  if (!DeserializeRng(checkpoint.rng).ok()) {
    return absl::InvalidArgumentError("invalid checkpoint RNG state");
  }

  std::vector<std::pair<InfoSetKey, InfoSetRow>> rows(
      checkpoint.state.rows.begin(), checkpoint.state.rows.end());
  std::sort(rows.begin(), rows.end(), [](const auto& left, const auto& right) {
    return KeyLess(left.first, right.first);
  });

  Bytes bytes;
  bytes.insert(bytes.end(), kCheckpointMagic.begin(),
               kCheckpointMagic.end());
  AppendU32(bytes, checkpoint.format_version);
  AppendFingerprint(bytes, checkpoint.model);
  AppendU64(bytes, checkpoint.state.iterations);
  AppendU64(bytes, std::bit_cast<uint64_t>(
                       checkpoint.state.cumulative_root_utility));
  AppendU64(bytes, checkpoint.rng.text.size());
  AppendU64(bytes, rows.size());
  AppendU64(bytes, checkpoint.state.regret_sum.size());
  AppendU64(bytes, checkpoint.state.strategy_sum.size());
  bytes.insert(bytes.end(), checkpoint.rng.text.begin(),
               checkpoint.rng.text.end());
  for (const auto& [key, row] : rows) {
    AppendU32(bytes, key.history.value());
    AppendU64(bytes, key.public_observation.value());
    AppendU64(bytes, key.private_observation.value());
    AppendU64(bytes, row.action_offset);
  }
  for (float value : checkpoint.state.regret_sum) {
    AppendU32(bytes, std::bit_cast<uint32_t>(value));
  }
  for (float value : checkpoint.state.strategy_sum) {
    AppendU32(bytes, std::bit_cast<uint32_t>(value));
  }
  return WriteBytes(path, bytes);
}

absl::StatusOr<SolverCheckpoint> LoadCheckpoint(
    const std::filesystem::path& path) {
  const auto file = ReadBytes(path);
  if (!file.ok()) return file.status();
  ByteReader reader(*file);
  for (uint8_t expected : kCheckpointMagic) {
    const auto actual = reader.read_u8();
    if (!actual || *actual != expected) {
      return absl::DataLossError("invalid checkpoint file magic");
    }
  }
  const auto version = reader.read_u32();
  const auto model = reader.read_fingerprint();
  const auto iterations = reader.read_u64();
  const auto utility = reader.read_u64();
  const auto rng_size = reader.read_u64();
  const auto row_count = reader.read_u64();
  const auto regret_count = reader.read_u64();
  const auto strategy_count = reader.read_u64();
  if (!version || *version != 1 || !model || !iterations || !utility ||
      !rng_size || !row_count || !regret_count || !strategy_count ||
      *rng_size > reader.remaining() || *row_count > reader.remaining() / 28 ||
      *regret_count > reader.remaining() / sizeof(float) ||
      *strategy_count > reader.remaining() / sizeof(float)) {
    return absl::DataLossError("invalid checkpoint file header");
  }
  const auto rng_bytes = reader.read_bytes(static_cast<size_t>(*rng_size));
  if (!rng_bytes) {
    return absl::DataLossError("truncated checkpoint RNG state");
  }

  SolverCheckpoint checkpoint;
  checkpoint.format_version = *version;
  checkpoint.model = *model;
  checkpoint.state.iterations = *iterations;
  checkpoint.state.cumulative_root_utility =
      std::bit_cast<double>(*utility);
  checkpoint.rng.text.assign(
      reinterpret_cast<const char*>(rng_bytes->data()), rng_bytes->size());
  checkpoint.state.rows.reserve(static_cast<size_t>(*row_count));
  for (uint64_t index = 0; index < *row_count; ++index) {
    const auto history = reader.read_u32();
    const auto public_observation = reader.read_u64();
    const auto private_observation = reader.read_u64();
    const auto offset = reader.read_u64();
    if (!history || !public_observation || !private_observation || !offset ||
        *offset > std::numeric_limits<size_t>::max()) {
      return absl::DataLossError("truncated checkpoint row");
    }
    const InfoSetKey key{HistoryId(*history),
                         PublicObservationId(*public_observation),
                         PrivateObservationId(*private_observation)};
    if (!checkpoint.state.rows
             .emplace(key, InfoSetRow{static_cast<size_t>(*offset)})
             .second) {
      return absl::DataLossError("duplicate checkpoint row");
    }
  }
  checkpoint.state.regret_sum.resize(static_cast<size_t>(*regret_count));
  for (float& value : checkpoint.state.regret_sum) {
    const auto bits = reader.read_u32();
    if (!bits) return absl::DataLossError("truncated checkpoint regrets");
    value = std::bit_cast<float>(*bits);
  }
  checkpoint.state.strategy_sum.resize(static_cast<size_t>(*strategy_count));
  for (float& value : checkpoint.state.strategy_sum) {
    const auto bits = reader.read_u32();
    if (!bits) return absl::DataLossError("truncated checkpoint strategies");
    value = std::bit_cast<float>(*bits);
  }
  if (reader.remaining() != 0) {
    return absl::DataLossError("trailing checkpoint data");
  }
  const absl::Status valid = ValidateCheckpointState(checkpoint.state);
  if (!valid.ok()) return valid;
  if (!DeserializeRng(checkpoint.rng).ok()) {
    return absl::DataLossError("invalid checkpoint RNG state");
  }
  return checkpoint;
}

absl::StatusOr<SolverConfig> SolverConfig::Create(
    SolverConfigOptions options) {
  if (options.starting_stack <= 0 || options.small_blind <= 0 ||
      options.big_blind < options.small_blind ||
      options.starting_stack < options.big_blind) {
    return absl::InvalidArgumentError("invalid stack or blind configuration");
  }
  if (options.chance_samples <= 0) {
    return absl::InvalidArgumentError("chance_samples must be positive");
  }
  if (options.max_info_sets <= 0) {
    return absl::InvalidArgumentError("max_info_sets must be positive");
  }
  for (const auto& fractions : options.bet_abstraction.pot_fractions) {
    if (fractions.size() >
        std::numeric_limits<uint8_t>::max() - size_t{3}) {
      return absl::InvalidArgumentError("too many pot fractions");
    }
    for (double fraction : fractions) {
      if (!std::isfinite(fraction) || fraction <= 0.0) {
        return absl::InvalidArgumentError(
            "pot fractions must be finite and positive");
      }
    }
  }
  return SolverConfig(std::move(options));
}

SolverConfig SolverConfig::Default() {
  auto config = Create(SolverConfigOptions{});
  assert(config.ok());
  return *config;
}

absl::StatusOr<ComboRange> ParseRange(std::string_view text) {
  std::array<bool, kHandTypeCount> seen = {};
  std::vector<int> selected;
  auto select = [&](HandType type) {
    auto add = [&](HandType candidate) {
      const auto index = HandTypeIndex(candidate);
      if (index && !seen[static_cast<size_t>(*index)]) {
        seen[static_cast<size_t>(*index)] = true;
        selected.push_back(*index);
      }
    };
    if (type.shape == HandShape::kAny) {
      type.shape = HandShape::kSuited;
      add(type);
      type.shape = HandShape::kOffsuit;
    }
    add(type);
  };
  while (!text.empty()) {
    const size_t comma = text.find(',');
    const std::string_view part = Trim(text.substr(0, comma));
    text = comma == std::string_view::npos ? std::string_view()
                                           : text.substr(comma + 1);
    if (part.empty()) {
      return absl::InvalidArgumentError("range contains an empty item");
    }
    if (part.size() == 3 && part[0] == part[1] && part[2] == '+') {
      const auto rank = ParseRank(part[0]);
      if (!rank) {
        return absl::InvalidArgumentError("invalid pair range");
      }
      for (int value = *rank; value <= 14; ++value) {
        select(HandType{value, value, HandShape::kPair});
      }
      continue;
    }
    const auto type = ParseHandType(part);
    if (!type) {
      return absl::InvalidArgumentError("invalid hand range item");
    }
    select(*type);
  }
  if (selected.empty()) {
    return absl::InvalidArgumentError("range is empty");
  }
  return ExpandSelectedCombos(selected);
}

ComboRange UniformComboRange() {
  ComboRange range;
  for (size_t first = 0; first < kDeck.size(); ++first) {
    for (size_t second = first + 1; second < kDeck.size(); ++second) {
      range.add(CardsToComboId(kDeck[first], kDeck[second]));
    }
  }
  return range;
}

ComboRange SingleComboRange(ComboId combo, float weight) {
  ComboRange range;
  range.add(combo, weight);
  return range;
}

CFRSolver::CFRSolver(SolveSpec spec, DealDistribution deals)
    : spec_(std::move(spec)),
      betting_rules_{spec_.config.big_blind()},
      deals_(std::move(deals)),
      rng_(12345) {
  history_ = BuildHistoryTree(
      spec_.root.betting, betting_rules_, spec_.config);
  model_ = FingerprintModel(spec_, history_);
  const size_t rows = static_cast<size_t>(spec_.config.max_info_sets());
  size_t max_actions = 3;
  for (StreetKind street : {StreetKind::kPreflop, StreetKind::kFlop,
                            StreetKind::kTurn, StreetKind::kRiver}) {
    const auto& sizes = spec_.config.bet_abstraction().pot_fractions;
    max_actions = std::max(
        max_actions, sizes[static_cast<size_t>(street)].size() + 3);
  }
  state_.rows.reserve(rows);
  state_.regret_sum.reserve(rows * max_actions);
  if (spec_.config.accumulate_average_strategy()) {
    state_.strategy_sum.reserve(rows * max_actions);
  }
}

absl::StatusOr<std::unique_ptr<CFRSolver>> CFRSolver::Create(
    SolveSpec spec) {
  if (!IsValidBettingData(Data(spec.root.betting))) {
    return absl::InvalidArgumentError("invalid root betting state");
  }
  auto deals = DealDistribution::Create(spec.ranges[Index(Player::kA)],
                                        spec.ranges[Index(Player::kB)]);
  if (!deals.ok()) {
    return deals.status();
  }
  return std::unique_ptr<CFRSolver>(
      new CFRSolver(std::move(spec), std::move(*deals)));
}

Position CFRSolver::root_position() const {
  return {history_.root,
          PublicPosition::Root(spec_.config.card_abstraction(),
                               Data(spec_.root.betting).street,
                               spec_.root.board)};
}

Position CFRSolver::action_child(Position position,
                                 uint8_t action_index) const {
  assert(position.history.index() < history_.nodes.size());
  const auto* node =
      std::get_if<DecisionNode>(&history_.nodes[position.history.index()]);
  assert(node != nullptr && action_index < node->edges.count);
  position.history = history_.edges[
      node->edges.begin + action_index].child;
  return position;
}

Position CFRSolver::sample_chance_child(Position position,
                                        const Deal& deal) {
  assert(position.history.index() < history_.nodes.size());
  const auto* node =
      std::get_if<ChanceNode>(&history_.nodes[position.history.index()]);
  assert(node != nullptr);

  const BettingData& data = node->state.data;
  const auto sampled = SampleStreetCards(data.street,
                                         position.public_state.board(),
                                         deal.blocked_mask, rng_);
  assert(sampled.ok());
  const ExactPublicState child = AdvanceChance(
      node->state, position.public_state.board(), *sampled, betting_rules_);
  position.history = node->child;
  position.public_state = position.public_state.after_chance(
      spec_.config.card_abstraction(), Data(child.betting).street,
      child.board);
  return position;
}

std::array<PrivateObservationId, kPlayerCount>
CFRSolver::private_observations_for_position(
    const Deal& deal,
    const Position& position) const {
  std::array<PrivateObservationId, kPlayerCount> observations;
  for (size_t player = 0; player < kPlayerCount; ++player) {
    const ComboId hand = deal.hand(static_cast<Player>(player)).combo();
    observations[player] = ObservePrivate(
        spec_.config.card_abstraction(), hand, position.public_state);
  }
  return observations;
}

void CFRSolver::advance_private_observations(
    TraversalFrame& frame,
    const Deal& deal,
    const Position& child) const {
  for (size_t player = 0; player < kPlayerCount; ++player) {
    const ComboId hand = deal.hand(static_cast<Player>(player)).combo();
    frame.private_observations[player] = AdvancePrivateObservation(
        spec_.config.card_abstraction(), frame.private_observations[player],
        hand, child.public_state);
  }
}

std::optional<InfoSetRow> CFRSolver::find_row(InfoSetKey key) const {
  const auto row = state_.rows.find(key);
  if (row == state_.rows.end()) {
    return std::nullopt;
  }
  return row->second;
}

std::optional<InfoSetRow> CFRSolver::find_or_create_row(
    InfoSetKey key,
    uint8_t action_count) {
  if (const auto row = find_row(key)) {
    return row;
  }
  if (state_.rows.size() >=
      static_cast<size_t>(spec_.config.max_info_sets())) {
    return std::nullopt;
  }

  const size_t offset = state_.regret_sum.size();
  state_.regret_sum.resize(offset + action_count, 0.0f);
  if (spec_.config.accumulate_average_strategy()) {
    state_.strategy_sum.resize(offset + action_count, 0.0f);
  }
  const InfoSetRow row{offset};
  return state_.rows.emplace(key, row).first->second;
}

double CFRSolver::traverse(Position position,
                           TraversalFrame frame,
                           TraversalContext& context) {
  const Deal& deal = context.deal;
  assert(position.history.index() < history_.nodes.size());
  return std::visit([&](const auto& node) -> double {
    using Node = std::decay_t<decltype(node)>;
    if constexpr (std::is_same_v<Node, FoldTerminalNode>) {
      ++stats_.terminal_visits;
      return TerminalUtility(node.state, Player::kA);
    } else if constexpr (std::is_same_v<Node, ShowdownNode>) {
      ++stats_.terminal_visits;
      const auto* board =
          std::get_if<RiverBoard>(&position.public_state.board());
      assert(board != nullptr);
      return TerminalUtility(node.state, *board, deal.hand(Player::kA),
                             deal.hand(Player::kB));
    } else if constexpr (std::is_same_v<Node, ChanceNode>) {
      const int samples = spec_.config.chance_samples();
      stats_.chance_samples += static_cast<uint64_t>(samples);
      double value = 0.0;
      for (int sample = 0; sample < samples; ++sample) {
        const Position child = sample_chance_child(position, deal);
        TraversalFrame child_frame = frame;
        advance_private_observations(child_frame, deal, child);
        value += traverse(child, child_frame, context);
      }
      return value / samples;
    } else {
      const Player player = node.state.actor;
      const size_t player_index = Index(player);
      const uint8_t action_count = node.edges.count;
      assert(action_count > 0);
      const InfoSetKey key{
          position.history, position.public_state.observation(),
          frame.private_observations[player_index]};
      const bool training = context.mode == TraversalMode::kTrain;
      const bool updates = training && context.update_player == player;
      std::optional<InfoSetRow> row;
      if (updates) {
        row = find_or_create_row(key, action_count);
        if (!row.has_value()) {
          context.info_set_limit_reached = true;
        }
      } else {
        row = find_row(key);
      }
      const InfoSetRow* strategy_row = row ? &*row : nullptr;

      absl::InlinedVector<double, 8> probabilities(action_count, 0.0);
      absl::InlinedVector<double, 8> values(action_count, 0.0);
      if (context.mode == TraversalMode::kEvaluateAverage) {
        AverageStrategy(state_, strategy_row, absl::MakeSpan(probabilities));
      } else {
        RegretMatch(state_, strategy_row, absl::MakeSpan(probabilities));
      }

      double node_value = 0.0;
      for (uint8_t action = 0; action < action_count; ++action) {
        TraversalFrame child_frame = frame;
        child_frame.reach[player_index] *= probabilities[action];
        values[action] =
            traverse(action_child(position, action), child_frame, context);
        node_value += probabilities[action] * values[action];
      }
      if (!training) {
        return node_value;
      }
      ++stats_.decision_visits;
      if (!updates || !row.has_value()) {
        return node_value;
      }

      const double sign = player == Player::kA ? 1.0 : -1.0;
      const double opponent_reach = frame.reach[Index(Opponent(player))];
      for (uint8_t action = 0; action < action_count; ++action) {
        const double regret =
            opponent_reach * sign * (values[action] - node_value);
        AddCfrPlusRegret(state_, *row, action, static_cast<float>(regret));
      }
      if (spec_.config.accumulate_average_strategy()) {
        const double weight = frame.reach[player_index] *
                              static_cast<double>(context.iteration + 1);
        AddStrategySum(state_, *row, absl::MakeConstSpan(probabilities),
                       weight);
      }
      return node_value;
    }
  }, history_.nodes[position.history.index()]);
}

TrainingResult CFRSolver::run(uint64_t iterations) {
  TrainingResult result;
  if (iterations <= 0) {
    return result;
  }

  const Position root = root_position();
  for (uint64_t i = 0; i < iterations; ++i) {
    const Deal deal = deals_.sample(rng_);
    TraversalFrame frame;
    frame.private_observations = private_observations_for_position(deal, root);
    const Player update_player =
        state_.iterations % kPlayerCount == 0 ? Player::kA : Player::kB;
    TraversalContext context{
        deal, TraversalMode::kTrain, update_player, state_.iterations};
    state_.cumulative_root_utility +=
        traverse(root, frame, context);
    ++state_.iterations;
    ++result.iterations_completed;
    if (context.info_set_limit_reached) {
      result.stop_reason = TrainingStopReason::kInfoSetLimit;
      break;
    }
  }

  log_training_summary();
  return result;
}

double CFRSolver::evaluate_deal(const Deal& deal, TraversalMode mode) {
  const Position root = root_position();
  TraversalFrame frame;
  frame.private_observations = private_observations_for_position(deal, root);
  TraversalContext context{deal, mode};
  return traverse(root, frame, context);
}

double CFRSolver::evaluate_deals(int samples, TraversalMode mode) {
  if (samples <= 0) {
    return 0.0;
  }
  double value = 0.0;
  for (int sample = 0; sample < samples; ++sample) {
    value += evaluate_deal(deals_.sample(rng_), mode);
  }
  return value / samples;
}

double CFRSolver::evaluate_current(HoleCards player_a,
                                   HoleCards player_b) {
  const Deal deal{{player_a, player_b},
                  ComboMask(player_a.combo()) | ComboMask(player_b.combo())};
  return evaluate_deal(deal, TraversalMode::kEvaluateCurrent);
}

double CFRSolver::evaluate_current(int samples) {
  return evaluate_deals(samples, TraversalMode::kEvaluateCurrent);
}

absl::StatusOr<double> CFRSolver::evaluate_average(
    HoleCards player_a,
    HoleCards player_b) {
  if (!spec_.config.accumulate_average_strategy()) {
    return absl::FailedPreconditionError(
        "average strategy accumulation is disabled");
  }
  const Deal deal{{player_a, player_b},
                  ComboMask(player_a.combo()) | ComboMask(player_b.combo())};
  return evaluate_deal(deal, TraversalMode::kEvaluateAverage);
}

absl::StatusOr<double> CFRSolver::evaluate_average(int samples) {
  if (!spec_.config.accumulate_average_strategy()) {
    return absl::FailedPreconditionError(
        "average strategy accumulation is disabled");
  }
  return evaluate_deals(samples, TraversalMode::kEvaluateAverage);
}

absl::StatusOr<Policy> ExtractAveragePolicy(
    const CfrState& state,
    const HistoryTree& history,
    ModelFingerprint model) {
  Policy policy;
  policy.model = model;
  policy.rows.reserve(state.rows.size());
  policy.probabilities.resize(state.strategy_sum.size());
  for (const auto& [key, row] : state.rows) {
    if (key.history.index() >= history.nodes.size()) {
      return absl::DataLossError("infoset references an invalid history");
    }
    const auto* node =
        std::get_if<DecisionNode>(&history.nodes[key.history.index()]);
    if (node == nullptr) {
      return absl::DataLossError("infoset history is not a decision");
    }
    const uint8_t count = node->edges.count;
    if (row.action_offset + count > state.strategy_sum.size()) {
      return absl::DataLossError("infoset strategy span is invalid");
    }
    policy.rows.emplace(key, PolicyRow{row.action_offset, count});
    double sum = 0.0;
    for (size_t action = 0; action < count; ++action) {
      const float value = state.strategy_sum[row.action_offset + action];
      if (!std::isfinite(value)) {
        return absl::DataLossError("nonfinite average strategy value");
      }
      sum += std::max(0.0f, value);
    }
    for (size_t action = 0; action < count; ++action) {
      const float value = state.strategy_sum[row.action_offset + action];
      policy.probabilities[row.action_offset + action] =
          sum > 0.0 ? static_cast<float>(std::max(0.0f, value) / sum)
                    : 1.0f / count;
    }
  }
  const absl::Status valid = ValidatePolicy(policy);
  if (!valid.ok()) return valid;
  return policy;
}

absl::StatusOr<Policy> CFRSolver::extract_average_policy() const {
  if (!spec_.config.accumulate_average_strategy()) {
    return absl::FailedPreconditionError(
        "average strategy accumulation is disabled");
  }
  return ExtractAveragePolicy(state_, history_, model_);
}

SolverCheckpoint CFRSolver::checkpoint() const {
  return {1, model_, state_, SerializeRng(rng_)};
}

absl::Status CFRSolver::restore(SolverCheckpoint checkpoint) {
  if (checkpoint.format_version != 1) {
    return absl::InvalidArgumentError("unsupported checkpoint version");
  }
  if (checkpoint.model != model_) {
    return absl::FailedPreconditionError(
        "checkpoint model does not match solver");
  }
  const absl::Status valid = ValidateCheckpointState(checkpoint.state);
  if (!valid.ok()) return valid;
  if (checkpoint.state.rows.size() >
      static_cast<size_t>(spec_.config.max_info_sets())) {
    return absl::ResourceExhaustedError(
        "checkpoint exceeds the infoset limit");
  }
  if (spec_.config.accumulate_average_strategy()) {
    if (checkpoint.state.strategy_sum.size() !=
        checkpoint.state.regret_sum.size()) {
      return absl::DataLossError("checkpoint has no average strategy");
    }
  } else if (!checkpoint.state.strategy_sum.empty()) {
    return absl::DataLossError("checkpoint has unexpected strategy values");
  }

  std::vector<std::pair<size_t, uint8_t>> spans;
  spans.reserve(checkpoint.state.rows.size());
  for (const auto& [key, row] : checkpoint.state.rows) {
    if (key.history.index() >= history_.nodes.size()) {
      return absl::DataLossError("checkpoint references invalid history");
    }
    const auto* node =
        std::get_if<DecisionNode>(&history_.nodes[key.history.index()]);
    if (node == nullptr) {
      return absl::DataLossError("checkpoint row is not a decision");
    }
    if (row.action_offset > checkpoint.state.regret_sum.size() ||
        node->edges.count >
            checkpoint.state.regret_sum.size() - row.action_offset) {
      return absl::DataLossError("checkpoint row exceeds CFR arrays");
    }
    spans.push_back({row.action_offset, node->edges.count});
  }
  std::sort(spans.begin(), spans.end());
  size_t offset = 0;
  for (const auto& [row_offset, count] : spans) {
    if (row_offset != offset) {
      return absl::DataLossError("checkpoint rows are not contiguous");
    }
    offset += count;
  }
  if (offset != checkpoint.state.regret_sum.size()) {
    return absl::DataLossError("checkpoint CFR array size is invalid");
  }

  auto rng = DeserializeRng(checkpoint.rng);
  if (!rng.ok()) return rng.status();
  state_ = std::move(checkpoint.state);
  rng_ = *rng;
  stats_ = {};

  const size_t rows = static_cast<size_t>(spec_.config.max_info_sets());
  size_t max_actions = 3;
  for (const auto& fractions :
       spec_.config.bet_abstraction().pot_fractions) {
    max_actions = std::max(max_actions, fractions.size() + 3);
  }
  state_.rows.reserve(rows);
  state_.regret_sum.reserve(rows * max_actions);
  if (spec_.config.accumulate_average_strategy()) {
    state_.strategy_sum.reserve(rows * max_actions);
  }
  return absl::OkStatus();
}

double CFRSolver::get_expected_value(Player player) const {
  if (state_.iterations == 0) {
    return 0.0;
  }
  const double player_a_ev =
      state_.cumulative_root_utility / state_.iterations;
  return player == Player::kA ? player_a_ev : -player_a_ev;
}

void CFRSolver::log_training_summary() const {
  LOG(INFO) << "CFR iterations completed";
  LOG(INFO) << "Iterations run: " << state_.iterations;
  LOG(INFO) << "Information sets: " << state_.rows.size();
  LOG(INFO) << "History nodes: " << history_.nodes.size();
  LOG(INFO) << "Player A average EV: " << get_expected_value(Player::kA);
}

}  // namespace poker
