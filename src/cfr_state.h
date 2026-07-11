#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

namespace poker {

inline constexpr size_t kCacheLineBytes = 64;
inline constexpr size_t kCumulativeActionBlockAlignment =
    kCacheLineBytes / sizeof(float);

template <typename T>
class CacheLineAlignedAllocator {
 public:
  using value_type = T;

  CacheLineAlignedAllocator() noexcept = default;
  template <typename U>
  CacheLineAlignedAllocator(const CacheLineAlignedAllocator<U>&) noexcept {}

  T* allocate(size_t n) {
    if (n > std::numeric_limits<size_t>::max() / sizeof(T)) {
      throw std::bad_array_new_length();
    }
    return static_cast<T*>(
        ::operator new(n * sizeof(T), std::align_val_t{kCacheLineBytes}));
  }

  void deallocate(T* ptr, size_t) noexcept {
    ::operator delete(ptr, std::align_val_t{kCacheLineBytes});
  }
};

template <typename T, typename U>
bool operator==(const CacheLineAlignedAllocator<T>&,
                const CacheLineAlignedAllocator<U>&) noexcept {
  return true;
}

template <typename T, typename U>
bool operator!=(const CacheLineAlignedAllocator<T>&,
                const CacheLineAlignedAllocator<U>&) noexcept {
  return false;
}

struct CfrState {
  using ActionArray =
      std::vector<float, CacheLineAlignedAllocator<float>>;

  ActionArray regret_sum;
  ActionArray strategy_sum;
  uint64_t iterations = 0;
  double cumulative_root_utility = 0.0;
};

}  // namespace poker
