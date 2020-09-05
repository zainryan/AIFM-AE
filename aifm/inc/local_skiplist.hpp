#pragma once

#include "sync.h"

#include "helpers.hpp"
#include "slab.hpp"

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <random>

namespace far_memory {

class GenericLocalSkiplist {
protected:
#pragma pack(push, 1)
  struct Entry {
    using Width = uint32_t;

    Width width;
    Entry *up;
    Entry *down;
    Entry *left;
    Entry *right;
    void *key;
    rt::Spin lock;
  };
#pragma pack(pop)

  constexpr static uint32_t kProbInv = 4;
  constexpr static uint64_t kMaxEntries =
      std::numeric_limits<Entry::Width>::max();
  constexpr static uint64_t kMaxLevels =
      helpers::static_log(kProbInv, kMaxEntries);

  uint32_t item_size_;
  void *negative_infinite_;
  void *positive_infinite_;
  Slab slab_;
  std::default_random_engine generator_;
  std::uniform_int_distribution<int> distribution_{0, kProbInv};
  Entry *head_;
  Entry *tail_;
  uint32_t levels_ = 0;
  std::function<bool(const void *, const void *)> is_equal_;
  std::function<bool(const void *, const void *)> is_smaller_;
  std::function<bool(const void *, const void *)> is_greater_;

  GenericLocalSkiplist(uint32_t item_size, uint64_t data_size);
  Entry *_find_closest(const void *key, Entry **level_traces = nullptr);
  bool should_bubble_up();
  void bubble_up(Entry *down_ptr, Entry **level_traces);
  void prune_empty_level(Entry *left_boudary, Entry *right_boundary);
  bool insert(const void *key);
  bool exist(const void *key);
  bool remove(const void *key);
};

template <typename T> class LocalSkiplist : public GenericLocalSkiplist {
public:
  LocalSkiplist(uint64_t data_size);
  ~LocalSkiplist();
  bool insert(const T &key);
  bool exist(const T &key);
  bool remove(const T &key);
  uint32_t rank(const T &key);
  T select(uint32_t rank);
};

template <typename T>
FORCE_INLINE LocalSkiplist<T>::LocalSkiplist(uint64_t data_size)
    : GenericLocalSkiplist(sizeof(T), data_size) {
  // Param a must be the key from the API argument instead of the in-list
  // pointer.
  is_equal_ = [&](const void *a, const void *b) -> bool {
    assert(b != positive_infinite_);
    if (unlikely(b == negative_infinite_)) {
      return false;
    }
    return *(reinterpret_cast<const T *>(a)) ==
           *(reinterpret_cast<const T *>(b));
  };
  is_smaller_ = [&](const void *a, const void *b) -> bool {
    assert(b != negative_infinite_);
    if (unlikely(b == positive_infinite_)) {
      return true;
    }
    return *(reinterpret_cast<const T *>(a)) <
           *(reinterpret_cast<const T *>(b));
  };
  is_greater_ = [&](const void *a, const void *b) -> bool {
    assert(b != positive_infinite_);
    if (unlikely(b == negative_infinite_)) {
      return true;
    }
    return *(reinterpret_cast<const T *>(a)) >
           *(reinterpret_cast<const T *>(b));
  };
}

template <typename T> LocalSkiplist<T>::~LocalSkiplist() {}

template <typename T> bool LocalSkiplist<T>::insert(const T &key) {
  return GenericLocalSkiplist::insert(&key);
}

template <typename T> bool LocalSkiplist<T>::exist(const T &key) {
  return GenericLocalSkiplist::exist(&key);
}

template <typename T> bool LocalSkiplist<T>::remove(const T &key) {
  return GenericLocalSkiplist::remove(&key);
}

} // namespace far_memory
