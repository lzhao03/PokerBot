#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "absl/types/span.h"

namespace poker {

struct ModelFingerprint {
  std::array<std::byte, 32> bytes = {};

  friend bool operator==(const ModelFingerprint&,
                         const ModelFingerprint&) = default;
};

class FingerprintBuilder {
 public:
  void add_bytes(absl::Span<const uint8_t> bytes) noexcept;
  void add_u8(uint8_t value) noexcept;
  void add_u32(uint32_t value) noexcept;
  void add_u64(uint64_t value) noexcept;
  void add_i32(int32_t value) noexcept;
  void add_float(float value) noexcept;
  void add_double(double value) noexcept;
  ModelFingerprint finish() noexcept;

 private:
  void transform() noexcept;

  std::array<uint32_t, 8> state_ = {
      0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  };
  std::array<uint8_t, 64> block_ = {};
  size_t block_size_ = 0;
  uint64_t total_bytes_ = 0;
};

}  // namespace poker
