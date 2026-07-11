#include "src/hand_range.h"

#include <algorithm>
#include <array>
#include <optional>
#include <string_view>
#include <vector>

namespace poker {
namespace {

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

std::optional<int> Rank(char rank) {
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
  const auto first = Rank(text[0]);
  const auto second = Rank(text[1]);
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
  constexpr std::array<SuitKind, 4> suits = {
      SuitKind::kHearts, SuitKind::kDiamonds,
      SuitKind::kClubs, SuitKind::kSpades};
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
          MakeCardId(type.high, suits[first]),
          MakeCardId(type.low, suits[second])));
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

ComboRange ExpandSelected(const std::vector<int>& selected) {
  ComboRange range;
  for (int index : selected) {
    const auto combos = Expand(*DecodeHandType(index));
    const float weight = 1.0f / combos.size();
    for (ComboId combo : combos) {
      range.add(combo, weight);
    }
  }
  return range;
}

}  // namespace

bool ComboRange::empty() const {
  return active_count == 0;
}

ComboRange ParseRange(std::string_view text) {
  std::array<bool, kHandTypeCount> seen = {};
  std::vector<int> selected;
  auto select = [&](HandType type) {
    auto add = [&](HandType candidate) {
      const auto index = HandTypeIndex(candidate);
      if (index && !seen[*index]) {
        seen[*index] = true;
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
    if (part.size() == 3 && part[0] == part[1] && part[2] == '+') {
      if (const auto rank = Rank(part[0])) {
        for (int value = *rank; value <= 14; ++value) {
          select(HandType{value, value, HandShape::kPair});
        }
      }
    } else if (const auto type = ParseHandType(part)) {
      select(*type);
    }
  }
  return ExpandSelected(selected);
}

ComboRange UniformRange() {
  std::vector<int> selected(kHandTypeCount);
  for (int index = 0; index < kHandTypeCount; ++index) {
    selected[index] = index;
  }
  return ExpandSelected(selected);
}

ComboRange SingleComboRange(ComboId combo, float weight) {
  ComboRange range;
  range.add(combo, weight);
  return range;
}

}  // namespace poker
