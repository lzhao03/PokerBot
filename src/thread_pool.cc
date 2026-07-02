#include "src/thread_pool.h"

#include <stdexcept>

namespace poker {

ThreadPoolExecutor::ThreadPoolExecutor(int max_workers) {
  int worker_count = max_workers;
  if (worker_count <= 0) {
    worker_count = static_cast<int>(std::thread::hardware_concurrency());
    if (worker_count <= 0) {
      worker_count = 1;
    }
  }

  workers_.reserve(worker_count);
  for (int i = 0; i < worker_count; ++i) {
    workers_.emplace_back([this]() { worker_loop(); });
  }
}

ThreadPoolExecutor::~ThreadPoolExecutor() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  cv_.notify_all();

  for (std::thread& worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

int ThreadPoolExecutor::max_workers() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return static_cast<int>(workers_.size());
}

void ThreadPoolExecutor::submit_task(std::function<void()> task) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_) {
      throw std::runtime_error("cannot submit task after executor shutdown");
    }
    tasks_.push(std::move(task));
  }
  cv_.notify_one();
}

void ThreadPoolExecutor::worker_loop() {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this]() { return stopping_ || !tasks_.empty(); });
      if (stopping_ && tasks_.empty()) {
        return;
      }
      task = std::move(tasks_.front());
      tasks_.pop();
    }
    task();
  }
}

}  // namespace poker
