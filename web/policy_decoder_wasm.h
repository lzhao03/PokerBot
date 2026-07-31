#pragma once

#include <cstddef>
#include <cstdint>

namespace poker::web {

inline constexpr uint64_t kTabularModel = 0x5bf84653a150904fULL;
inline constexpr uint64_t kNeuralModel = 0x6fbb89e52780fea2ULL;
inline constexpr int kTabularPolicy = 1;
inline constexpr int kNeuralPolicy = 2;

enum BrowserPhase : int32_t {
  DecisionPhase,
  ChancePhase,
  FoldPhase,
  ShowdownPhase,
};

enum BrowserStateField : size_t {
  Phase,
  Street,
  Actor,
  Stack0,
  Stack1,
  Bet0,
  Bet1,
  Pot,
  CurrentWager,
  CallAmount,
  WinnerMask,
  CardsNeeded,
  BrowserStateFieldCount,
};

}  // namespace poker::web

extern "C" {

uint8_t* poker_allocate(size_t size);
void poker_free(void* memory);
int poker_load_policy(const uint8_t* bytes, size_t size);
int poker_load_neural_policy(const uint8_t* bytes, size_t size);
int poker_query(int policy_kind,
                int dealer,
                const uint8_t* input_kinds,
                const int32_t* input_targets,
                size_t input_count,
                const uint8_t* cards,
                size_t board_count,
                uint8_t* output_kinds,
                int32_t* output_targets,
                float* output_probabilities);
int poker_query_found();
int poker_replay(int dealer,
                 const uint8_t* input_kinds,
                 const int32_t* input_targets,
                 size_t input_count,
                 const uint8_t* cards,
                 size_t board_count,
                 int32_t* output_state,
                 uint8_t* output_kinds,
                 int32_t* output_targets);

}
