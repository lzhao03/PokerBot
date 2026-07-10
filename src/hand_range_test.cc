#include "src/hand_range.h"

#include "doctest/doctest.h"
#include "src/training_range.h"

#include <array>

namespace poker {
namespace {

TEST_CASE("range syntax expands to the expected combo count") {
  struct ParserCase {
    const char* text;
    uint16_t active_count;
  };
  const std::array<ParserCase, 7> cases = {{
      {"AA", 6},
      {"AKs", 4},
      {"AKo", 12},
      {"AK", 16},
      {"AA,KK", 12},
      {"QQ+", 18},
      {"89s+", 0},
  }};

  for (const ParserCase& test : cases) {
    CAPTURE(test.text);
    HandRange range;
    range.set_from_string(test.text);
    CHECK(BuildTrainingRange(range).active_count == test.active_count);
  }
}

TEST_CASE("all hand-type indices round-trip through a representative combo") {
  for (int index = 0; index < 169; ++index) {
    CAPTURE(index);
    const std::optional<ComboId> combo = HandRange::index_to_combo(index);
    REQUIRE(combo.has_value());
    CHECK(HandRange::combo_to_index(*combo) == index);
  }
  CHECK(!HandRange::index_to_combo(-1).has_value());
  CHECK(!HandRange::index_to_combo(169).has_value());
}

}  // namespace
}  // namespace poker
