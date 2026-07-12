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

ModelFingerprint Sha256(absl::Span<const uint8_t> bytes) noexcept;

}  // namespace poker
