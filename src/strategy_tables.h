#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "src/public_graph.h"
#include "src/poker_types.h"

namespace poker {

class StrategyTables {
 public:
  static constexpr uint32_t kInvalidActionOffset =
      std::numeric_limits<uint32_t>::max();

  struct InfoSetRow {
    uint32_t action_offset = 0;
    uint16_t action_count = 0;
  };

  struct InfoSetKey {
    NodeId node_id = 0;
    PrivateObservationId private_observation = 0;
  };

  struct GrowingPublicInfoSets {
    absl::flat_hash_map<PrivateObservationId, InfoSetRow> rows;
  };

  struct FrozenInfoSetEntry {
    PrivateObservationId private_observation = 0;
    uint32_t action_offset = kInvalidActionOffset;
  };

  struct FrozenPublicInfoSetRange {
    uint32_t begin = 0;
    uint32_t count = 0;
  };

  PublicGraph graph;
  std::vector<FrozenInfoSetEntry> frozen_info_set_entries;
  std::vector<FrozenPublicInfoSetRange> frozen_info_set_ranges;
  size_t info_set_count = 0;
  std::vector<std::unique_ptr<GrowingPublicInfoSets>> growing_info_sets;
};

}  // namespace poker
