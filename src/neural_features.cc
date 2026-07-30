#include "src/neural_features.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace poker {

NeuralFeatureVector EncodeNeuralFeatures(
    HistoryId history,
    PublicObservationId public_observation,
    PrivateObservationId private_observation,
    const HistoryNode& node,
    const CardAbstractionConfig& cards) {
  const DecisionState& decision = std::get<DecisionState>(node.state);
  const BettingData& betting = decision.data;
  NeuralFeatureVector features = {};
  size_t output = 0;
  const auto append_bits = [&](auto value) {
    for (size_t bit = 0; bit < sizeof(value) * 8; ++bit) {
      features[output++] = static_cast<float>((value >> bit) & 1);
    }
  };
  append_bits(std::to_underlying(history));

  const size_t public_begin = output;
  if (cards.public_mode == PublicCardMode::CompactTexture) {
    const uint64_t observation = std::to_underlying(public_observation);
    size_t bucket_offset = 0;
    for (size_t street = 0; street < kCompactPublicBuckets.size(); ++street) {
      const size_t bucket = static_cast<size_t>(
          (observation >> (street * kPublicObservationBitsPerStreet)) & 0x7f);
      if (bucket != 0) {
        features[output + bucket_offset + bucket - 1] = 1.0f;
      }
      bucket_offset += kCompactPublicBuckets[street];
    }
  } else {
    append_bits(std::to_underlying(public_observation));
  }
  output = public_begin + 16 + 16 + 64;

  const size_t private_begin = output;
  if (cards.private_kind == PrivateAbstractionKind::Handcrafted36) {
    const uint32_t observation = std::to_underlying(private_observation);
    if (cards.recall_mode == RecallMode::BucketHistory) {
      for (size_t street = 0; street < kPrivateObservationPlaces.size();
           ++street) {
        const uint32_t bucket =
            (observation / kPrivateObservationPlaces[street]) %
            kPrivateObservationRadix;
        if (bucket != 0) {
          features[output + street * kPrivateBucketCount + bucket - 1] = 1.0f;
        }
      }
    } else if (observation != 0) {
      features[output + observation - 1] = 1.0f;
    }
  } else {
    append_bits(std::to_underlying(private_observation));
  }
  output = private_begin + 4 * kPrivateBucketCount;

  features[output++] = decision.actor == Player::B ? 1.0f : 0.0f;
  features[output++] =
      static_cast<float>(node.child_count) / kMaxActionsPerNode;
  for (StreetKind street : {StreetKind::Preflop, StreetKind::Flop,
                            StreetKind::Turn, StreetKind::River}) {
    features[output++] = betting.street == street ? 1.0f : 0.0f;
  }

  const Chips total_chips =
      Pot(betting) + betting.stack[0] + betting.stack[1];
  const float scale =
      1.0f / static_cast<float>(std::max(Chips{1}, total_chips));
  const auto scaled = [scale](Chips value) {
    return static_cast<float>(value) * scale;
  };
  for (Chips value : betting.stack) features[output++] = scaled(value);
  for (Chips value : betting.total_committed) {
    features[output++] = scaled(value);
  }
  for (Chips value : betting.street_committed) {
    features[output++] = scaled(value);
  }
  features[output++] = scaled(betting.last_full_raise);
  features[output++] = betting.actions_remaining / 2.0f;
  features[output++] = scaled(Pot(betting));
  assert(output == features.size());
  return features;
}

}  // namespace poker
