#include "src/hand_range.h"

#include <cmath>
#include <stdexcept>

namespace {

void Expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void CheckUpdatingHandWeightReplacesTotal() {
  poker::HandRange range;
  int aces = poker::HandRange::string_to_index("AA");
  int kings = poker::HandRange::string_to_index("KK");

  range.add_hand_by_index(aces, 1.0);
  range.add_hand_by_index(kings, 1.0);
  range.add_hand_by_index(aces, 3.0);

  poker::Hand aces_hand = poker::HandRange::index_to_hand(aces);
  poker::Hand kings_hand = poker::HandRange::index_to_hand(kings);
  Expect(std::abs(range.get_total_weight() - 4.0) < 0.000001,
         "updating a hand should replace its old total weight");
  Expect(std::abs(range.get_probability(aces_hand) - 0.75) < 0.000001,
         "updated hand probability should use replaced total weight");
  Expect(std::abs(range.get_probability(kings_hand) - 0.25) < 0.000001,
         "other hand probability should use replaced total weight");
}

}  // namespace

int main() {
  CheckUpdatingHandWeightReplacesTotal();
  return 0;
}
