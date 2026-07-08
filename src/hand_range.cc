#include "src/hand_range.h"

#include <algorithm>
#include <optional>
#include <sstream>
#include <string_view>

namespace poker {

namespace {

constexpr int kHandTypeCount = 169;

struct HandType {
  enum class Shape {
    kPair,
    kSuited,
    kOffsuit,
    kAnyNonPair,
  };

  int high_rank = 0;
  int low_rank = 0;
  Shape shape = Shape::kPair;
};

std::optional<int> RankFromChar(char rank) {
  switch (rank) {
    case 'A':
      return 14;
    case 'K':
      return 13;
    case 'Q':
      return 12;
    case 'J':
      return 11;
    case 'T':
      return 10;
    case '9':
    case '8':
    case '7':
    case '6':
    case '5':
    case '4':
    case '3':
    case '2':
      return rank - '0';
  }
  return std::nullopt;
}

char RankToChar(int rank) {
  switch (rank) {
    case 14:
      return 'A';
    case 13:
      return 'K';
    case 12:
      return 'Q';
    case 11:
      return 'J';
    case 10:
      return 'T';
    case 9:
    case 8:
    case 7:
    case 6:
    case 5:
    case 4:
    case 3:
    case 2:
      return static_cast<char>('0' + rank);
  }
  return '?';
}

bool IsValidRank(int rank) {
  return rank >= 2 && rank <= 14;
}

int NonPairOffset(int high_rank, int low_rank) {
  const int high = high_rank - 2;
  const int low = low_rank - 2;
  return (high * (high - 1) / 2) + low;
}

std::optional<HandType> DecodeHandTypeIndex(int index) {
  if (index < 0 || index >= kHandTypeCount) {
    return std::nullopt;
  }
  if (index < 13) {
    const int rank = index + 2;
    return HandType{rank, rank, HandType::Shape::kPair};
  }

  const bool suited = index < 91;
  int offset = suited ? index - 13 : index - 91;
  int high = 1;
  while (offset >= (high * (high + 1) / 2)) {
    ++high;
  }
  const int low = offset - (high * (high - 1) / 2);
  return HandType{high + 2, low + 2,
                  suited ? HandType::Shape::kSuited
                         : HandType::Shape::kOffsuit};
}

std::optional<int> EncodeHandTypeIndex(HandType type) {
  if (!IsValidRank(type.high_rank) || !IsValidRank(type.low_rank)) {
    return std::nullopt;
  }
  if (type.high_rank < type.low_rank) {
    std::swap(type.high_rank, type.low_rank);
  }
  if (type.high_rank == type.low_rank) {
    return type.shape == HandType::Shape::kPair
               ? std::optional<int>(type.high_rank - 2)
               : std::nullopt;
  }

  const int offset = NonPairOffset(type.high_rank, type.low_rank);
  if (type.shape == HandType::Shape::kSuited) {
    return 13 + offset;
  }
  if (type.shape == HandType::Shape::kOffsuit) {
    return 91 + offset;
  }
  return std::nullopt;
}

ComboId MakeCombo(int first_rank, SuitKind first_suit, int second_rank,
                  SuitKind second_suit) {
  return CardsToComboId(MakeCardId(first_rank, first_suit),
                        MakeCardId(second_rank, second_suit));
}

void AddWeightedCombo(WeightedHandRange& weighted_combos, ComboId combo_id,
                      double weight) {
  for (size_t i = 0; i < weighted_combos.size(); ++i) {
    if (weighted_combos.combos[i] == combo_id) {
      weighted_combos.weights[i] += weight;
      return;
    }
  }

  weighted_combos.add(combo_id, weight);
}

std::string ExactHandKey(ComboId combo_id) {
  const ComboInfo& combo = GetComboInfo(combo_id);
  std::ostringstream oss;
  const CardId cards[2] = {combo.card0, combo.card1};
  for (int i = 0; i < 2; ++i) {
    if (i > 0) {
      oss << "-";
    }
    oss << RankFromCardId(cards[i]) << ":"
        << 1 + SuitIndex(SuitFromCardId(cards[i]));
  }
  return oss.str();
}

std::vector<ComboId> ExpandHandType(HandType type) {
  std::vector<ComboId> combos;
  const SuitKind suits[] = {SuitKind::kHearts, SuitKind::kDiamonds,
                            SuitKind::kClubs, SuitKind::kSpades};

  if (!IsValidRank(type.high_rank) || !IsValidRank(type.low_rank)) {
    return combos;
  }

  if (type.high_rank < type.low_rank) {
    std::swap(type.high_rank, type.low_rank);
  }

  if (type.high_rank == type.low_rank) {
    for (int i = 0; i < 4; ++i) {
      for (int j = i + 1; j < 4; ++j) {
        combos.push_back(MakeCombo(type.high_rank, suits[i], type.high_rank,
                                   suits[j]));
      }
    }
    return combos;
  }

  for (SuitKind first_suit : suits) {
    for (SuitKind second_suit : suits) {
      const bool suited = first_suit == second_suit;
      if (type.shape == HandType::Shape::kAnyNonPair ||
          (type.shape == HandType::Shape::kSuited && suited) ||
          (type.shape == HandType::Shape::kOffsuit && !suited)) {
        combos.push_back(MakeCombo(type.high_rank, first_suit, type.low_rank,
                                   second_suit));
      }
    }
  }
  return combos;
}

std::vector<ComboId> ExpandHandTypeIndex(int index) {
  const std::optional<HandType> type = DecodeHandTypeIndex(index);
  return type.has_value() ? ExpandHandType(*type) : std::vector<ComboId>();
}

std::string FormatHandType(HandType type) {
  if (type.high_rank < type.low_rank) {
    std::swap(type.high_rank, type.low_rank);
  }

  std::string key;
  key += RankToChar(type.high_rank);
  key += RankToChar(type.low_rank);
  if (type.high_rank != type.low_rank) {
    if (type.shape == HandType::Shape::kSuited) {
      key += 's';
    } else if (type.shape == HandType::Shape::kOffsuit) {
      key += 'o';
    }
  }
  return key;
}

std::optional<HandType> ParseHandType(std::string_view text) {
  if (text.size() != 2 && text.size() != 3) {
    return std::nullopt;
  }

  std::optional<int> first_rank = RankFromChar(text[0]);
  std::optional<int> second_rank = RankFromChar(text[1]);
  if (!first_rank.has_value() || !second_rank.has_value()) {
    return std::nullopt;
  }

  HandType type{std::max(*first_rank, *second_rank),
                std::min(*first_rank, *second_rank),
                HandType::Shape::kPair};
  if (type.high_rank == type.low_rank) {
    return text.size() == 2 ? std::optional<HandType>(type) : std::nullopt;
  }
  if (text.size() == 2) {
    type.shape = HandType::Shape::kAnyNonPair;
    return type;
  }
  if (text[2] == 's') {
    type.shape = HandType::Shape::kSuited;
    return type;
  }
  if (text[2] == 'o') {
    type.shape = HandType::Shape::kOffsuit;
    return type;
  }
  return std::nullopt;
}

std::optional<ComboId> RepresentativeComboForHandTypeIndex(int index) {
  std::optional<HandType> type = DecodeHandTypeIndex(index);
  if (!type.has_value()) {
    return std::nullopt;
  }
  if (type->shape == HandType::Shape::kPair) {
    return MakeCombo(type->high_rank, SuitKind::kSpades, type->high_rank,
                     SuitKind::kHearts);
  }
  return MakeCombo(type->high_rank, SuitKind::kSpades, type->low_rank,
                   type->shape == HandType::Shape::kSuited
                       ? SuitKind::kSpades
                       : SuitKind::kHearts);
}

}  // namespace

HandRange::HandRange() 
  : weighted_combos_cache_valid_(false),
    total_weight_(0.0) {
}

void HandRange::add_combo(ComboId combo_id, double weight) {
  if (combo_id >= kComboCount || weight <= 0.0) {
    return;
  }

  for (auto& pair : exact_hand_weights_) {
    if (pair.first == combo_id) {
      total_weight_ += weight - pair.second;
      pair.second = weight;
      invalidate_caches();
      return;
    }
  }

  exact_hand_weights_.emplace_back(combo_id, weight);
  total_weight_ += weight;
  invalidate_caches();
}

void HandRange::add_hand_by_index(int index, double weight) {
  if (index < 0 || index >= kHandTypeCount || weight <= 0.0) {
    return;
  }
  
  bool found = false;
  for (auto& pair : hand_weights_) {
    if (pair.first == index) {
      total_weight_ += weight - pair.second;
      pair.second = weight;
      found = true;
      break;
    }
  }
  
  if (!found) {
    hand_weights_.emplace_back(index, weight);
    total_weight_ += weight;
  }
  
  invalidate_caches();
}

double HandRange::get_probability(ComboId combo_id) const {
  if (combo_id >= kComboCount || total_weight_ <= 0.0) {
    return 0.0;
  }

  double hand_weight = 0.0;
  for (const auto& pair : exact_hand_weights_) {
    if (pair.first == combo_id) {
      hand_weight += pair.second;
    }
  }

  int index = combo_to_index(combo_id);
  if (index >= 0) {
    for (const auto& pair : hand_weights_) {
      if (pair.first == index) {
        std::vector<ComboId> combos = ExpandHandTypeIndex(index);
        if (!combos.empty()) {
          hand_weight += pair.second / combos.size();
        }
        break;
      }
    }
  }
  
  return hand_weight / total_weight_;
}

const WeightedHandRange& HandRange::get_all_weighted_combos() const {
  if (weighted_combos_cache_valid_) {
    return cached_weighted_combos_;
  }

  cached_weighted_combos_.clear();
  for (const auto& hand_weight : exact_hand_weights_) {
    AddWeightedCombo(cached_weighted_combos_, hand_weight.first,
                     hand_weight.second);
  }

  for (const auto& hand_weight : hand_weights_) {
    std::vector<ComboId> combos = ExpandHandTypeIndex(hand_weight.first);
    if (combos.empty()) {
      continue;
    }

    double combo_weight = hand_weight.second / combos.size();
    for (ComboId combo : combos) {
      AddWeightedCombo(cached_weighted_combos_, combo, combo_weight);
    }
  }

  weighted_combos_cache_valid_ = true;
  return cached_weighted_combos_;
}

void HandRange::clear() {
  hand_weights_.clear();
  exact_hand_weights_.clear();
  total_weight_ = 0.0;
  invalidate_caches();
}

void HandRange::set_uniform_range() {
  // Clear current range
  clear();
  
  // Assign uniform weight to all 169 hand types, not exact combos.
  double weight = 1.0 / kHandTypeCount;
  
  for (int i = 0; i < kHandTypeCount; ++i) {
    add_hand_by_index(i, weight);
  }
}

void HandRange::set_from_string(const std::string& range_str) {
  clear();
  
  // Split the range string by commas
  std::istringstream iss(range_str);
  std::string component;
  
  while (std::getline(iss, component, ',')) {
    // Trim whitespace
    component.erase(0, component.find_first_not_of(" \t"));
    component.erase(component.find_last_not_of(" \t") + 1);
    
    if (!component.empty()) {
      parse_range_component(component);
    }
  }
  
  // Normalize the weights
  normalize();
}

std::string HandRange::to_string() const {
  std::ostringstream oss;
  
  // Sort indices for consistent output
  std::vector<int> indices;
  indices.reserve(hand_weights_.size());
  
  for (const auto& pair : hand_weights_) {
    indices.push_back(pair.first);
  }
  
  std::sort(indices.begin(), indices.end());

  std::vector<std::string> exact_hands;
  exact_hands.reserve(exact_hand_weights_.size());
  for (const auto& pair : exact_hand_weights_) {
    exact_hands.push_back("[" + ExactHandKey(pair.first) + "]");
  }
  std::sort(exact_hands.begin(), exact_hands.end());
  
  bool needs_separator = false;
  for (int index : indices) {
    if (needs_separator) {
      oss << ",";
    }
    
    std::optional<HandType> type = DecodeHandTypeIndex(index);
    if (type.has_value()) {
      oss << FormatHandType(*type);
    }
    needs_separator = true;
  }

  for (const std::string& hand : exact_hands) {
    if (needs_separator) {
      oss << ",";
    }

    oss << hand;
    needs_separator = true;
  }
  
  return oss.str();
}

double HandRange::get_total_weight() const {
  return total_weight_;
}

void HandRange::normalize() {
  if (total_weight_ <= 0.0) {
    return;
  }
  
  // Normalize weights in the vector
  for (auto& pair : hand_weights_) {
    pair.second /= total_weight_;
  }

  for (auto& pair : exact_hand_weights_) {
    pair.second /= total_weight_;
  }
  
  total_weight_ = 1.0;
  invalidate_caches();
}

int HandRange::combo_to_index(ComboId combo_id) {
  if (combo_id >= kComboCount) {
    return -1;
  }
  
  const ComboInfo& combo = GetComboInfo(combo_id);
  int rank1 = RankFromCardId(combo.card0);
  int rank2 = RankFromCardId(combo.card1);
  SuitKind suit1 = SuitFromCardId(combo.card0);
  SuitKind suit2 = SuitFromCardId(combo.card1);
  
  HandType type{std::max(rank1, rank2), std::min(rank1, rank2),
                rank1 == rank2
                    ? HandType::Shape::kPair
                    : (suit1 == suit2 ? HandType::Shape::kSuited
                                       : HandType::Shape::kOffsuit)};
  return EncodeHandTypeIndex(type).value_or(-1);
}

std::optional<ComboId> HandRange::index_to_combo(int index) {
  return RepresentativeComboForHandTypeIndex(index);
}

std::string HandRange::combo_to_string(ComboId combo_id) {
  if (combo_id >= kComboCount) {
    return "";
  }
  
  const ComboInfo& combo = GetComboInfo(combo_id);
  int rank1 = RankFromCardId(combo.card0);
  int rank2 = RankFromCardId(combo.card1);
  SuitKind suit1 = SuitFromCardId(combo.card0);
  SuitKind suit2 = SuitFromCardId(combo.card1);
  
  // Ensure rank1 >= rank2
  if (rank1 < rank2) {
    std::swap(rank1, rank2);
    std::swap(suit1, suit2);
  }
  
  const HandType type{rank1, rank2,
                      rank1 == rank2
                          ? HandType::Shape::kPair
                          : (suit1 == suit2 ? HandType::Shape::kSuited
                                             : HandType::Shape::kOffsuit)};
  return FormatHandType(type);
}

int HandRange::string_to_index(const std::string& hand_str) {
  const std::optional<HandType> type = ParseHandType(hand_str);
  if (!type.has_value()) {
    return -1;
  }
  return EncodeHandTypeIndex(*type).value_or(-1);
}

void HandRange::parse_range_component(const std::string& component) {
  if (component.size() == 3 && component[0] == component[1] &&
      component[2] == '+') {
    std::optional<int> start_rank = RankFromChar(component[0]);
    if (!start_rank.has_value()) {
      return;
    }

    // Add all pairs from the specified rank up to AA
    for (int rank = *start_rank; rank <= 14; ++rank) {
      add_hand_by_index(rank - 2, 1.0);
    }
    return;
  }

  if (component.find('+') != std::string::npos) {
    return;
  }

  const std::optional<HandType> type = ParseHandType(component);
  if (!type.has_value()) {
    return;
  }

  if (type->shape == HandType::Shape::kAnyNonPair) {
    HandType suited = *type;
    suited.shape = HandType::Shape::kSuited;
    HandType offsuit = *type;
    offsuit.shape = HandType::Shape::kOffsuit;
    add_hand_by_index(EncodeHandTypeIndex(suited).value_or(-1), 1.0);
    add_hand_by_index(EncodeHandTypeIndex(offsuit).value_or(-1), 1.0);
    return;
  }

  const std::optional<int> index = EncodeHandTypeIndex(*type);
  if (index.has_value()) {
    add_hand_by_index(*index, 1.0);
  }
}

void HandRange::invalidate_caches() {
  weighted_combos_cache_valid_ = false;
  cached_weighted_combos_.clear();
}

} // namespace poker
