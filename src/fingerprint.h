#pragma once

#include <array>
#include <cstdint>

#include "absl/types/span.h"

namespace poker {

struct ModelFingerprint {
  std::array<uint8_t, 32> bytes = {};

  friend bool operator==(const ModelFingerprint&,
                         const ModelFingerprint&) = default;
};

ModelFingerprint Sha256(absl::Span<const uint8_t> bytes) noexcept;

}  // namespace poker
