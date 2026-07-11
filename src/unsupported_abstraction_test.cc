#include "src/cfr_solver.h"

#include "doctest/doctest.h"

namespace poker {
namespace {

#if !POKER_COARSE_PUBLIC_BUCKETS || !POKER_COARSE_PRIVATE_BUCKETS
#error "unsupported_abstraction_test requires both coarse abstractions"
#endif

TEST_CASE("coarse public and private abstractions are unsupported") {
  const SolverConfig config;
  CHECK_THROWS_WITH_AS(
      CFRSolver{config},
      "coarse public + coarse private abstraction does not provide "
      "exhaustive history-aware private observation support",
      std::invalid_argument);
}

}  // namespace
}  // namespace poker
