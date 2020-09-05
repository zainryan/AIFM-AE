#pragma once

extern "C" {
#include <base/compiler.h>
}

#include "helpers.hpp"
#include "sync.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <type_traits>

namespace far_memory {

template <typename T, bool Sync, uint64_t Capacity = 0> class CircularBuffer {
private:
  using FixedArray = T[Capacity + 1];
  constexpr static bool kIsDynamic = (Capacity == 0);

  typename std::conditional<kIsDynamic, std::unique_ptr<T[]>, FixedArray>::type
      items_;
  uint32_t head_ = 0;
  uint32_t tail_ = 0;
  uint32_t capacity_ = Capacity + 1;
  rt::Spin spin_;

public:
  CircularBuffer() {}

  template <bool B = kIsDynamic,
            typename = typename std::enable_if<B, void>::type>
  CircularBuffer(uint32_t size) {
    capacity_ = size + 1;
    preempt_disable();
    items_ = std::unique_ptr<T[]>(new T[capacity_]);
    preempt_enable();
  }

  CircularBuffer<T, Sync> &operator=(CircularBuffer<T, Sync> &&other) {
    if constexpr (kIsDynamic) {
      items_ = std::move(other.items_);
    } else {
      memcpy(items_, other.items_, sizeof(items_));
    }
    head_ = other.head_;
    tail_ = other.tail_;
    capacity_ = other.capacity_;
    return *this;
  }

  CircularBuffer(CircularBuffer<T, Sync> &&other) { *this = std::move(other); }

  uint32_t capacity() const { return capacity_ - 1; }

  uint32_t size() const {
    uint32_t ret;
    auto tail = load_acquire(&tail_);
    auto head = head_;
    if (tail < head) {
      ret = tail + capacity_ - head;
    } else {
      ret = tail - head;
    }
    return ret;
  }

  template <typename D> bool push_front(D &&d) {
    if constexpr (Sync) {
      spin_.Lock();
    }
    auto head = load_acquire(&head_);
    auto new_head = (head + capacity_ - 1) % capacity_;
    bool success = (new_head != tail_);
    if (likely(success)) {
      items_[new_head] = std::move(d);
      store_release(&head_, new_head);
    }
    if constexpr (Sync) {
      spin_.Unlock();
    }
    return success;
  }

  template <typename D> bool push_back(D &&d) {
    if constexpr (Sync) {
      spin_.Lock();
    }
    auto tail = load_acquire(&tail_);
    auto new_tail = (tail + 1) % capacity_;
    bool success = (new_tail != head_);
    if (likely(success)) {
      items_[tail] = std::move(d);
      store_release(&tail_, new_tail);
    }
    if constexpr (Sync) {
      spin_.Unlock();
    }
    return success;
  }

  template <typename D> std::optional<T> push_back_override(D &&d) {
    std::optional<T> overrided;
    if constexpr (Sync) {
      spin_.Lock();
    }
    auto tail = load_acquire(&tail_);
    auto new_tail = (tail + 1) % capacity_;
    if (unlikely(new_tail == head_)) {
      overrided = std::move(items_[head_]);
      head_ = (head_ + 1) % capacity_;
    }
    items_[tail] = std::move(d);
    store_release(&tail_, new_tail);
    if constexpr (Sync) {
      spin_.Unlock();
    }
    return overrided;
  }

  template <typename D> bool pop_front(D *d) {
    if constexpr (Sync) {
      spin_.Lock();
    }
    auto head = load_acquire(&head_);
    auto tail = tail_;
    bool success = (head != tail);
    if (likely(success)) {
      *d = std::move(items_[head]);
      store_release(&head_, (head + 1) % capacity_);
    }
    if constexpr (Sync) {
      spin_.Unlock();
    }
    return success;
  }

  bool work_steal(CircularBuffer<T, Sync, Capacity> *cb) {
    static_assert(Sync);
    spin_.Lock();
    auto self_spin_releaser = helpers::finally([&]() { spin_.Unlock(); });
    if (!(cb->size() / 2)) {
      return false;
    }
    if (!cb->spin_.TryLock()) {
      return false;
    }
    auto cb_spin_releaser = helpers::finally([&]() { cb->spin_.Unlock(); });
    uint32_t cb_size = cb->size();
    if (very_unlikely(!(cb_size / 2))) {
      return false;
    }
    auto tail = load_acquire(&tail_);
    auto steal_size = std::min(cb_size / 2, capacity() - size());
    for (uint32_t i = 0; i < steal_size; i++) {
      items_[tail] = std::move(cb->items_[cb->head_]);
      cb->head_ = (cb->head_ + 1) % (cb->capacity_);
      tail = (tail + 1) % (capacity_);
      assert(tail < head_ + capacity_);
      assert(cb->head_ != cb->tail_);
    }
    store_release(&tail_, tail);
    return true;
  }

  void clear() {
    auto tail = load_acquire(&tail_);
    store_release(&head_, tail);
  }

  void for_each(const std::function<void(T)> &f) {
    if constexpr (Sync) {
      spin_.Lock();
    }
    auto idx = load_acquire(&head_);
    while (idx != tail_) {
      f(items_[idx]);
      idx = (idx + 1) % capacity_;
    }
    if constexpr (Sync) {
      spin_.Unlock();
    }
  }
};
} // namespace far_memory
