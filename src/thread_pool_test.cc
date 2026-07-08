#include "src/thread_pool.h"

#include "doctest/doctest.h"

#include <atomic>
#include <future>
#include <stdexcept>
#include <vector>

namespace poker {
namespace {

TEST_CASE("thread pool runs submitted tasks") {
  ThreadPoolExecutor executor(4);

  std::vector<std::future<int>> futures;
  for (int i = 0; i < 20; ++i) {
    futures.push_back(executor.submit([i]() { return i * i; }));
  }

  int sum = 0;
  for (std::future<int>& future : futures) {
    sum += future.get();
  }
  CHECK(sum == 2470);
}

TEST_CASE("default worker count is positive") {
  ThreadPoolExecutor executor(0);
  CHECK(executor.max_workers() >= 1);
}

TEST_CASE("task exceptions propagate through futures") {
  ThreadPoolExecutor executor(2);
  std::future<int> future = executor.submit([]() -> int {
    throw std::runtime_error("task failed");
  });

  CHECK_THROWS_AS((void)future.get(), std::runtime_error);
}

TEST_CASE("thread pool destructor drains queued tasks") {
  std::atomic<int> completed = 0;
  {
    ThreadPoolExecutor executor(1);
    for (int i = 0; i < 5; ++i) {
      executor.submit([&completed]() { ++completed; });
    }
  }
  CHECK(completed == 5);
}

}  // namespace
}  // namespace poker
