#ifndef POKER_THREAD_POOL_H_
#define POKER_THREAD_POOL_H_

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace poker {

// Minimal local executor for CFR fan-out. Revisit a Boost/standard executor
// backend if we need richer scheduling semantics or broader async integration.
class ThreadPoolExecutor {
 public:
  explicit ThreadPoolExecutor(int max_workers = 0);
  ~ThreadPoolExecutor();

  ThreadPoolExecutor(const ThreadPoolExecutor&) = delete;
  ThreadPoolExecutor& operator=(const ThreadPoolExecutor&) = delete;

  int max_workers() const;

  template <class F>
  auto submit(F&& fn) -> std::future<std::invoke_result_t<F>> {
    using Result = std::invoke_result_t<F>;
    auto task =
        std::make_shared<std::packaged_task<Result()>>(std::forward<F>(fn));
    std::future<Result> future = task->get_future();
    submit_task([task]() { (*task)(); });
    return future;
  }

 private:
  void submit_task(std::function<void()> task);
  void worker_loop();

  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  bool stopping_ = false;
};

}  // namespace poker

#endif  // POKER_THREAD_POOL_H_
