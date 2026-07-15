#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "web/policy_decoder_wasm.h"

#ifdef __EMSCRIPTEN__
#include "emscripten/emscripten.h"
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

namespace {

constexpr std::array<uint8_t, 8> kMagic = {
    'P', 'K', 'C', 'O', 'D', 'E', 'C', '1'};
constexpr size_t kMaxActions = 8;
constexpr uint16_t kMaxUnits = 256;

uint64_t Binomial(size_t n, size_t k) {
  if (k > n) return 0;
  k = std::min(k, n - k);
  uint64_t result = 1;
  for (size_t i = 1; i <= k; ++i) {
    result = result * static_cast<uint64_t>(n - k + i) / i;
  }
  return result;
}

uint64_t DistributionCount(size_t units, size_t actions) {
  return Binomial(units + actions - 1, actions - 1);
}

class Reader {
 public:
  Reader(const uint8_t* bytes, size_t size)
      : current_(bytes), end_(bytes + size) {}

  template <typename Integer>
  bool integer(Integer& value) {
    if (remaining() < sizeof(Integer)) return false;
    value = 0;
    for (size_t index = 0; index < sizeof(Integer); ++index) {
      value |= static_cast<Integer>(current_[index]) << (index * 8);
    }
    current_ += sizeof(Integer);
    return true;
  }

  bool varint(uint64_t& value) {
    value = 0;
    for (size_t shift = 0; shift < 64; shift += 7) {
      uint8_t byte;
      if (!integer(byte) || (shift == 63 && byte > 1)) return false;
      value |= static_cast<uint64_t>(byte & 0x7f) << shift;
      if ((byte & 0x80) == 0) return true;
    }
    return false;
  }

  size_t remaining() const {
    return static_cast<size_t>(end_ - current_);
  }

 private:
  const uint8_t* current_;
  const uint8_t* end_;
};

struct Key {
  uint64_t public_observation;
  uint32_t history;
  uint32_t private_observation;

  friend bool operator==(const Key&, const Key&) = default;
};

auto Order(const Key& key) {
  return std::tuple(key.history, key.public_observation,
                    key.private_observation);
}

struct Row {
  Key key;
  uint64_t code;
};

struct Policy {
  uint64_t model = 0;
  uint16_t units = 0;
  uint8_t max_actions = 0;
  std::array<std::vector<Row>, kMaxActions + 1> rows;
};

bool DecodePolicy(const uint8_t* bytes, size_t size, Policy& output) {
  Reader reader(bytes, size);
  for (uint8_t expected : kMagic) {
    uint8_t actual;
    if (!reader.integer(actual) || actual != expected) return false;
  }

  Policy decoded;
  if (!reader.integer(decoded.model) || !reader.integer(decoded.units) ||
      !reader.integer(decoded.max_actions) || decoded.units == 0 ||
      decoded.units > kMaxUnits || decoded.max_actions == 0 ||
      decoded.max_actions > kMaxActions) {
    return false;
  }

  for (size_t actions = 2; actions <= decoded.max_actions; ++actions) {
    uint32_t row_count;
    if (!reader.integer(row_count) || row_count > reader.remaining() / 4) {
      return false;
    }
    std::vector<Row>& rows = decoded.rows[actions];
    rows.reserve(row_count);
    uint64_t history = 0;
    uint64_t public_observation = 0;
    uint64_t private_observation = 0;
    for (uint32_t index = 0; index < row_count; ++index) {
      uint64_t history_delta;
      uint64_t public_delta;
      uint64_t private_delta;
      uint64_t code;
      if (!reader.varint(history_delta) || !reader.varint(public_delta) ||
          !reader.varint(private_delta) || !reader.varint(code) ||
          history_delta > std::numeric_limits<uint32_t>::max() - history) {
        return false;
      }
      history += history_delta;
      if (history_delta != 0) public_observation = private_observation = 0;
      if (public_delta >
          std::numeric_limits<uint64_t>::max() - public_observation) {
        return false;
      }
      public_observation += public_delta;
      if (public_delta != 0) private_observation = 0;
      if (private_delta >
              std::numeric_limits<uint32_t>::max() - private_observation ||
          (index > 0 && history_delta == 0 && public_delta == 0 &&
           private_delta == 0) ||
          code >= DistributionCount(decoded.units, actions)) {
        return false;
      }
      private_observation += private_delta;
      rows.push_back({
          {public_observation, static_cast<uint32_t>(history),
           static_cast<uint32_t>(private_observation)},
          code});
    }
  }
  if (reader.remaining() != 0) return false;
  output = std::move(decoded);
  return true;
}

void DecodeProbabilities(uint64_t code,
                         size_t actions,
                         uint16_t units,
                         float* output) {
  size_t remaining = units;
  for (size_t action = 0; action + 1 < actions; ++action) {
    const size_t remaining_actions = actions - action - 1;
    size_t action_units = 0;
    while (action_units < remaining) {
      const uint64_t skipped =
          DistributionCount(remaining - action_units, remaining_actions);
      if (code < skipped) break;
      code -= skipped;
      ++action_units;
    }
    output[action] = static_cast<float>(action_units) / units;
    remaining -= action_units;
  }
  output[actions - 1] = static_cast<float>(remaining) / units;
}

Policy policy;

uint64_t PublicObservation(uint32_t low, uint32_t high) {
  return low | (uint64_t{high} << 32);
}

}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE uint8_t* poker_allocate(size_t size) {
  return static_cast<uint8_t*>(std::malloc(size));
}

EMSCRIPTEN_KEEPALIVE void poker_free(void* memory) {
  std::free(memory);
}

EMSCRIPTEN_KEEPALIVE int poker_load_policy(const uint8_t* bytes, size_t size) {
  Policy decoded;
  if (bytes == nullptr || !DecodePolicy(bytes, size, decoded)) return 0;
  policy = std::move(decoded);
  return 1;
}

EMSCRIPTEN_KEEPALIVE void poker_unload_policy() {
  policy = {};
}

// Returns 1 for a stored row, 0 for uniform fallback, and -1 on misuse.
EMSCRIPTEN_KEEPALIVE int poker_strategy(uint32_t public_low,
                                       uint32_t public_high,
                                       uint32_t history,
                                       uint32_t private_observation,
                                       size_t action_count,
                                       float* output) {
  if (output == nullptr || action_count == 0 ||
      action_count > policy.max_actions) {
    return -1;
  }
  const Key key{PublicObservation(public_low, public_high), history,
                private_observation};
  const std::vector<Row>& rows = policy.rows[action_count];
  const auto found = std::lower_bound(
      rows.begin(), rows.end(), key,
      [](const Row& row, const Key& target) {
        return Order(row.key) < Order(target);
      });
  if (found == rows.end() || found->key != key) {
    std::fill_n(output, action_count, 1.0f / action_count);
    return 0;
  }
  DecodeProbabilities(found->code, action_count, policy.units, output);
  return 1;
}

EMSCRIPTEN_KEEPALIVE uint32_t poker_model_low() {
  return static_cast<uint32_t>(policy.model);
}

EMSCRIPTEN_KEEPALIVE uint32_t poker_model_high() {
  return static_cast<uint32_t>(policy.model >> 32);
}

}  // extern "C"
