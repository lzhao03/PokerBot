#include "src/thread_pool.h"

#include <atomic>
#include <future>
#include <stdexcept>
#include <vector>

namespace poker {
namespace {

void Expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void CheckRunsTasks() {
  ThreadPoolExecutor executor(4);

  std::vector<std::future<int>> futures;
  for (int i = 0; i < 20; ++i) {
    futures.push_back(executor.submit([i]() { return i * i; }));
  }

  int sum = 0;
  for (std::future<int>& future : futures) {
    sum += future.get();
  }
  Expect(sum == 2470, "executor should run every submitted task");
}

void CheckDefaultWorkerCountPositive() {
  ThreadPoolExecutor executor(0);
  Expect(executor.max_workers() >= 1,
         "default executor should create at least one worker");
}

void CheckExceptionPropagatesThroughFuture() {
  ThreadPoolExecutor executor(2);
  std::future<int> future = executor.submit([]() -> int {
    throw std::runtime_error("task failed");
  });

  bool threw = false;
  try {
    future.get();
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Expect(threw, "future should propagate task exceptions");
}

void CheckDestructorDrainsQueuedTasks() {
  std::atomic<int> completed = 0;
  {
    ThreadPoolExecutor executor(1);
    for (int i = 0; i < 5; ++i) {
      executor.submit([&completed]() { ++completed; });
    }
  }
  Expect(completed == 5, "executor destructor should drain queued tasks");
}

}  // namespace
}  // namespace poker

int main() {
  poker::CheckRunsTasks();
  poker::CheckDefaultWorkerCountPositive();
  poker::CheckExceptionPropagatesThroughFuture();
  poker::CheckDestructorDrainsQueuedTasks();
  return 0;
}
