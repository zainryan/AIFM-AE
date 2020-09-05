#pragma once

#include "sync.h"
#include "thread.h"

#include "cb.hpp"
#include "helpers.hpp"
#include "pointer.hpp"

#include <functional>
#include <optional>
#include <type_traits>

namespace far_memory {

uint32_t get_prefetch_win_size(uint32_t object_data_size);

template <typename Index_t, typename Pattern_t> class Prefetcher {
private:
  using Induce = std::function<Pattern_t(Index_t, Index_t)>;
  using Infer = std::function<Index_t(Index_t, Pattern_t)>;
  using Mapping = std::function<GenericUniquePtr *(Index_t)>;

  struct Trace {
    uint64_t counter;
    uint64_t idx;
    bool nt;
  };

  struct SlaveStatus {
    GenericUniquePtr *task;
    bool is_active;
    bool is_exited;
    rt::CondVar cv;
  };

  constexpr static uint32_t kIdxTracesSize = 256;
  constexpr static uint32_t kHitTimesThresh = 8;
  constexpr static uint32_t kGenTasksBurstSize = 8;
  constexpr static uint32_t kMaxSlaveWaitUs = 5;
  constexpr static uint32_t kMaxNumPrefetchSlaveThreads = 16;

  const uint32_t kPrefetchWinSize_; // In terms of number of objects.
  Induce kInduce_;
  Infer kInfer_;
  Pattern_t pattern_;
  Mapping kMapping_;
  uint32_t object_data_size_;
  Index_t last_idx_;
  uint64_t hit_times_ = 0;
  uint32_t num_objs_to_prefetch = 0;
  Index_t next_prefetch_idx_;
  bool nt_ = false;
  Trace traces_[kIdxTracesSize];
  uint32_t traces_head_ = 0;
  uint32_t traces_tail_ = 0;
  uint64_t traces_counter_ = 0;
  std::vector<rt::Thread> prefetch_threads_;
  CachelineAligned(SlaveStatus) slave_status_[kMaxNumPrefetchSlaveThreads];
  rt::CondVar cv_prefetch_master_;
  bool master_exited = false;
  bool exit_ = false;

  void generate_prefetch_tasks();
  void prefetch_master_fn();
  void prefetch_slave_fn(uint32_t tid);

public:
  Prefetcher(Induce induce, Infer infer, Mapping mapping,
             uint32_t object_data_size);
  ~Prefetcher();
  void add_trace(bool nt, Index_t idx);
  void static_prefetch(Index_t start_idx, Pattern_t pattern, uint32_t num);
};

template <typename Index_t, typename Pattern_t>
Prefetcher<Index_t, Pattern_t>::Prefetcher(Induce induce, Infer infer,
                                           Mapping mapping,
                                           uint32_t object_data_size)
    : kPrefetchWinSize_(get_prefetch_win_size(object_data_size)),
      kInduce_(induce), kInfer_(infer), kMapping_(mapping),
      object_data_size_(object_data_size) {
  prefetch_threads_.emplace_back(rt::Thread([&]() { prefetch_master_fn(); }));
  for (uint32_t i = 0; i < kMaxNumPrefetchSlaveThreads; i++) {
    auto &status = slave_status_[i].data;
    status.task = nullptr;
    status.is_active = status.is_exited = false;
    wmb();
    prefetch_threads_.emplace_back(
        rt::Thread([&, i]() { prefetch_slave_fn(i); }));
  }
  traces_[0].counter = 0;
}

template <typename Index_t, typename Pattern_t>
Prefetcher<Index_t, Pattern_t>::~Prefetcher() {
  exit_ = true;
  wmb();
  while (!ACCESS_ONCE(master_exited)) {
    cv_prefetch_master_.Signal();
    thread_yield();
  }
  for (uint32_t i = 0; i < kMaxNumPrefetchSlaveThreads; i++) {
    auto &status = slave_status_[i].data;
    while (!ACCESS_ONCE(status.is_exited)) {
      ACCESS_ONCE(status.is_active) = true;
      status.cv.Signal();
      thread_yield();
    }
  }
  for (auto &thread : prefetch_threads_) {
    thread.Join();
  }
}

template <typename Index_t, typename Pattern_t>
void Prefetcher<Index_t, Pattern_t>::generate_prefetch_tasks() {
  for (uint32_t i = 0; i < kGenTasksBurstSize; i++) {
    if (!num_objs_to_prefetch) {
      return;
    }
    num_objs_to_prefetch--;
    GenericUniquePtr *task = kMapping_(next_prefetch_idx_);
    next_prefetch_idx_ = kInfer_(next_prefetch_idx_, pattern_);
    if (!task) {
      continue;
    }
    bool dispatched = false;
    std::optional<uint32_t> inactive_slave_id = std::nullopt;
  dispatch:
    for (uint32_t i = 0; i < kMaxNumPrefetchSlaveThreads; i++) {
      auto &status = slave_status_[i].data;
      if (!ACCESS_ONCE(status.is_active)) {
        inactive_slave_id = i;
        continue;
      }
      if (ACCESS_ONCE(status.task) == nullptr) {
        ACCESS_ONCE(status.task) = task;
        dispatched = true;
        break;
      }
    }
    if (!dispatched) {
      if (likely(inactive_slave_id)) {
        auto &status = slave_status_[*inactive_slave_id].data;
        status.task = task;
        status.is_active = true;
        wmb();
        status.cv.Signal();
      } else {
        goto dispatch;
      }
    }
  }
}

template <typename Index_t, typename Pattern_t>
void Prefetcher<Index_t, Pattern_t>::prefetch_slave_fn(uint32_t tid) {
  auto &status = slave_status_[tid].data;
  GenericUniquePtr **task_ptr = &status.task;
  bool *is_active = &status.is_active;
  bool *is_exited = &status.is_exited;
  rt::CondVar *cv = &status.cv;
  cv->Wait();

  while (likely(!ACCESS_ONCE(exit_))) {
    if (likely(ACCESS_ONCE(*task_ptr))) {
      GenericUniquePtr *task = *task_ptr;
      ACCESS_ONCE(*task_ptr) = nullptr;
      task->swap_in(nt_);
    } else {
      auto start_us = microtime();
      while (ACCESS_ONCE(*task_ptr) == nullptr &&
             microtime() - start_us <= kMaxSlaveWaitUs) {
        cpu_relax();
      }
      if (unlikely(ACCESS_ONCE(*task_ptr) == nullptr)) {
        ACCESS_ONCE(*is_active) = false;
        do {
          cv->Wait();
        } while (!ACCESS_ONCE(*is_active));
      }
    }
  }
  ACCESS_ONCE(*is_exited) = true;
}

template <typename Index_t, typename Pattern_t>
void Prefetcher<Index_t, Pattern_t>::prefetch_master_fn() {
  uint64_t local_counter = 0;

  while (likely(!ACCESS_ONCE(exit_))) {
    auto [counter, idx, nt] = traces_[traces_head_];

    if (likely(local_counter < counter)) {
      local_counter = counter;
      traces_head_ = (traces_head_ + 1) % kIdxTracesSize;

      if (unlikely(idx == last_idx_)) {
        continue;
      }

      auto new_pattern = kInduce_(last_idx_, idx);

      if (pattern_ != new_pattern) {
        hit_times_ = num_objs_to_prefetch = 0;
      } else if (++hit_times_ >= kHitTimesThresh) {
        if (unlikely(hit_times_ == kHitTimesThresh)) {
          next_prefetch_idx_ = kInfer_(idx, pattern_);
          num_objs_to_prefetch = kPrefetchWinSize_;
        } else {
          num_objs_to_prefetch++;
        }
      }
      pattern_ = new_pattern;
      last_idx_ = idx;
      if (unlikely(nt_ != nt)) {
        // nt_ is shared by all slaves. Use the store instruction only when
        // neccesary to reduce cache traffic.
        nt_ = nt;
      }
    } else if (!num_objs_to_prefetch) {
      cv_prefetch_master_.Wait();
      continue;
    }
    generate_prefetch_tasks();
  }
  ACCESS_ONCE(master_exited) = true;
}

template <typename Index_t, typename Pattern_t>
void Prefetcher<Index_t, Pattern_t>::add_trace(bool nt, Index_t idx) {
  // add_trace() is at the call path of the frontend mutator thread.
  // The goal is to make it extremely short and fast, therefore not compromising
  // the mutator performance when prefetching is enabled. The most overheads are
  // transferred to the backend prefetching threads.
  traces_[traces_tail_++] = {
      .counter = ++traces_counter_, .idx = idx, .nt = nt};
  traces_tail_ %= kIdxTracesSize;
  if (unlikely(cv_prefetch_master_.HasWaiters())) {
    cv_prefetch_master_.Signal();
  }
}

template <typename Index_t, typename Pattern_t>
void Prefetcher<Index_t, Pattern_t>::static_prefetch(Index_t start_idx,
                                                     Pattern_t pattern,
                                                     uint32_t num) {
  next_prefetch_idx_ = start_idx;
  pattern_ = pattern;
  num_objs_to_prefetch = num;
}

} // namespace far_memory
