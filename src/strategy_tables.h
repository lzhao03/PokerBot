#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "src/combo.h"
#include "src/game_tree.h"
#include "src/poker_types.h"

namespace poker {

class FrozenStrategyTables {
 public:
  using PrivateBucketId = uint16_t;
  using PublicBucketId = uint64_t;

  struct BettingHistoryKey {
    static constexpr int kInlineHistoryValues = 48;

    int street = 0;
    int pot = 0;
    int stack_a = 0;
    int stack_b = 0;
    int all_in = 0;
    int folded_player = 0;
    int player_to_act = 0;
    int player_contribution_size = 0;
    std::array<int, 2> player_contributions = {0, 0};
    int history_size = 0;
    std::array<int, kInlineHistoryValues> history_values = {};
    std::vector<int> history_overflow;

    bool operator==(const BettingHistoryKey& other) const;
  };

  struct BettingHistoryKeyHash {
    size_t operator()(const BettingHistoryKey& key) const;
  };

  struct PublicStateKey {
    uint32_t betting_history_id = 0;
    PublicBucketId public_bucket = 0;

    bool operator==(const PublicStateKey& other) const;
  };

  struct PublicStateKeyHash {
    size_t operator()(const PublicStateKey& key) const;
  };

  struct BettingHistoryRow {
    BettingHistoryRow() {
      action_ids.fill(0);
      action_child_ids.fill(GameTree::Node::kInvalidBettingHistoryId);
    }

    int street = 0;
    int pot = 0;
    std::array<int, 2> stack = {0, 0};
    int all_in = 0;
    int folded_player = 0;
    int player_to_act = 0;
    std::array<int, 2> player_contributions = {0, 0};
    uint8_t action_count = 0;
    std::array<int, GameTree::kMaxActionsPerNode> action_ids;
    std::array<uint32_t, GameTree::kMaxActionsPerNode> action_child_ids;
    uint32_t chance_child_id = GameTree::Node::kInvalidBettingHistoryId;
  };

  struct InfoSetRow {
    uint32_t action_offset = 0;
    uint16_t action_count = 0;
  };

  struct InfoSetAddress {
    uint32_t public_state_id = 0;
    int player = 0;
    PrivateBucketId private_bucket = 0;
  };

  struct PublicStateRow {
    PublicStateRow() {
      action_ids.fill(0);
      action_child_ids.fill(GameTree::Node::kInvalidPublicStateId);
    }

    CompactPublicState state;
    uint32_t betting_history_id = GameTree::Node::kInvalidBettingHistoryId;
    PublicBucketId public_bucket = 0;
    bool is_terminal = false;
    bool is_chance_node = false;
    int player_to_act = -1;
    uint8_t action_count = 0;
    std::array<GameAction, GameTree::kMaxActionsPerNode> actions = {};
    std::array<int, GameTree::kMaxActionsPerNode> action_ids = {};
    std::array<uint32_t, GameTree::kMaxActionsPerNode> action_child_ids = {};
  };

  static constexpr int kPrivateBucketChunkSize = 64;
  static constexpr int kPrivateBucketChunkCount =
      (kComboCount + kPrivateBucketChunkSize - 1) / kPrivateBucketChunkSize;

  struct PrivateRowChunk {
    PrivateRowChunk() { rows.fill(-1); }

    std::array<int32_t, kPrivateBucketChunkSize> rows;
  };

  struct PublicInfoSetSlabPlayer {
    std::array<std::unique_ptr<PrivateRowChunk>, kPrivateBucketChunkCount>
        private_row_chunks;
    std::vector<InfoSetRow> rows;
  };

  struct PublicInfoSetSlab {
    std::array<PublicInfoSetSlabPlayer, kPlayerCount> players;
  };

  absl::flat_hash_map<BettingHistoryKey, uint32_t, BettingHistoryKeyHash>
      betting_history_ids;
  absl::flat_hash_map<PublicStateKey, uint32_t, PublicStateKeyHash>
      public_state_ids;
  std::vector<PublicStateRow> public_state_rows;
  absl::flat_hash_map<uint64_t, uint32_t> public_chance_child_ids;
  std::vector<BettingHistoryRow> betting_history_rows;
  size_t info_set_count = 0;
  std::vector<int> action_ids;
  std::vector<std::unique_ptr<PublicInfoSetSlab>> public_info_set_slabs;
};

struct MutableCumulativeArrays {
  std::vector<float> cumulative_regrets;
  std::vector<float> cumulative_strategies;
};

}  // namespace poker
