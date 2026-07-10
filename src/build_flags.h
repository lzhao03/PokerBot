#pragma once

#ifndef POKER_BENCHMARK_PROD_DEFAULTS
#define POKER_BENCHMARK_PROD_DEFAULTS 0
#endif

#ifndef POKER_COARSE_PUBLIC_BUCKETS
#define POKER_COARSE_PUBLIC_BUCKETS 0
#endif

#ifndef POKER_COARSE_PRIVATE_BUCKETS
#define POKER_COARSE_PRIVATE_BUCKETS 0
#endif

#ifndef POKER_ENABLE_CAS_RETRY_STATS
#define POKER_ENABLE_CAS_RETRY_STATS 0
#endif

#ifndef POKER_ENABLE_TRAVERSAL_STATS
#define POKER_ENABLE_TRAVERSAL_STATS 1
#endif

namespace poker {

constexpr bool kProdBenchmarkDefaults = POKER_BENCHMARK_PROD_DEFAULTS != 0;
constexpr bool kCoarsePublicBuckets = POKER_COARSE_PUBLIC_BUCKETS != 0;
constexpr bool kCoarsePrivateBuckets = POKER_COARSE_PRIVATE_BUCKETS != 0;
constexpr bool kCasRetryStatsEnabled = POKER_ENABLE_CAS_RETRY_STATS != 0;
constexpr bool kTraversalStatsEnabled = POKER_ENABLE_TRAVERSAL_STATS != 0;

}  // namespace poker
