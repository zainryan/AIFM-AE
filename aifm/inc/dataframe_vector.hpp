#pragma once

#include "dataframe_types.hpp"
#include "deref_scope.hpp"
#include "device.hpp"
#include "helpers.hpp"
#include "pointer.hpp"

#include <cstdint>
#include <vector>

namespace far_memory {

class GenericDataFrameVector {
private:
  enum OpCode { Resize = 0 };

  const uint32_t kChunkSize_;
  const uint32_t kChunkNumEntries_;
  FarMemDevice *device_;
  uint8_t ds_id_;
  uint64_t size_ = 0;
  std::vector<GenericUniquePtr> chunk_ptrs_;
  template <typename T> friend class DataFrameVector;
  template <typename T> friend class ServerDataFrameVector;

  void expand(uint64_t num);

public:
  GenericDataFrameVector(const uint32_t kChunkSize, uint32_t kChunkNumEntries,
                         uint8_t ds_id, uint8_t dt_id);
  bool empty() const;
  uint64_t size() const;
  void clear();
};

template <typename T> class DataFrameVector : GenericDataFrameVector {
private:
  static_assert(is_basic_dataframe_types<T>());

  constexpr static uint32_t kPreferredChunkSize = 512;
  constexpr static uint32_t kRealChunkNumEntries =
      std::max(static_cast<uint32_t>(1),
               helpers::round_up_power_of_two(kPreferredChunkSize / sizeof(T)));
  constexpr static uint32_t kRealChunkSize = sizeof(T) * kRealChunkNumEntries;
  constexpr static uint32_t kSizePerExpansion = 4 << 20; // 4 MiB.
  constexpr static uint32_t kNumEntriesPerExpansion =
      (kSizePerExpansion - 1) / sizeof(T) + 1;

  friend class FarMemTest;
  template <typename U> friend class ServerDataFrameVector;

  std::pair<uint64_t, uint64_t> get_chunk_stats(uint64_t index);
  void expand(uint64_t num);

public:
  DataFrameVector(uint8_t ds_id);
  uint64_t capacity() const;
  template <typename U> void push_back(const DerefScope &scope, const U &&u);
  void pop_back(const DerefScope &scope);
  void reserve(uint64_t count);
  T &front_mut(const DerefScope &scope);
  const T &front(const DerefScope &scope);
  T &back_mut(const DerefScope &scope);
  const T &back(const DerefScope &scope);
  T &at_mut(const DerefScope &scope, uint64_t index);
  const T &at(const DerefScope &scope, uint64_t index);
};

FORCE_INLINE void GenericDataFrameVector::clear() { size_ = 0; }

FORCE_INLINE bool GenericDataFrameVector::empty() const { return size() == 0; }

FORCE_INLINE uint64_t GenericDataFrameVector::size() const { return size_; }

template <typename T>
DataFrameVector<T>::DataFrameVector(uint8_t ds_id)
    : GenericDataFrameVector(kRealChunkSize, kRealChunkNumEntries, ds_id,
                             get_dataframe_type_id<T>()) {}

template <typename T> uint64_t DataFrameVector<T>::capacity() const {
  return chunk_ptrs_.size() * kRealChunkNumEntries;
}

template <typename T>
std::pair<uint64_t, uint64_t>
DataFrameVector<T>::get_chunk_stats(uint64_t index) {
  // It's superfast since chunk size is the power of 2.
  return std::make_pair(index / kRealChunkNumEntries,
                        index % kRealChunkNumEntries);
}

template <typename T> void DataFrameVector<T>::expand(uint64_t num) {
  GenericDataFrameVector::expand((num - 1) / kRealChunkNumEntries + 1);
}

template <typename T>
template <typename U>
void DataFrameVector<T>::push_back(const DerefScope &scope, const U &&u) {
  static_assert(std::is_same<std::decay_t<U>, std::decay_t<T>>::value,
                "U must be the same as T");
  auto [chunk_idx, chunk_offset] = get_chunk_stats(size_++);
  assert(chunk_ptrs_.size() >= chunk_idx);
  if (unlikely(chunk_ptrs_.size() == chunk_idx)) {
    expand(kNumEntriesPerExpansion);
  }
  auto *raw_mut_ptr = chunk_ptrs_[chunk_idx].deref_mut(scope);
  __builtin_memcpy(reinterpret_cast<T *>(raw_mut_ptr) + chunk_offset, &u,
                   sizeof(u));
}

template <typename T>
void DataFrameVector<T>::pop_back(const DerefScope &scope) {
  size_--;
}

template <typename T> void DataFrameVector<T>::reserve(uint64_t count) {
  BUG_ON(DerefScope::is_in_deref_scope());
  auto diff = static_cast<int64_t>(count) - static_cast<int64_t>(capacity());
  if (diff > 0) {
    expand(diff);
  }
}

template <typename T>
T &DataFrameVector<T>::front_mut(const DerefScope &scope) {
  return at_mut(scope, 0);
}

template <typename T>
const T &DataFrameVector<T>::front(const DerefScope &scope) {
  return at(scope, 0);
}

template <typename T> T &DataFrameVector<T>::back_mut(const DerefScope &scope) {
  return at_mut(scope, size() - 1);
}

template <typename T>
const T &DataFrameVector<T>::back(const DerefScope &scope) {
  return at(scope, size() - 1);
}

template <typename T>
T &DataFrameVector<T>::at_mut(const DerefScope &scope, uint64_t index) {
  auto [chunk_idx, chunk_offset] = get_chunk_stats(index);
  assert(chunk_ptrs_.size() > chunk_idx);
  auto *raw_mut_ptr = chunk_ptrs_[chunk_idx].deref_mut(scope);
  return *(reinterpret_cast<T *>(raw_mut_ptr) + chunk_offset);
}

template <typename T>
const T &DataFrameVector<T>::at(const DerefScope &scope, uint64_t index) {
  auto [chunk_idx, chunk_offset] = get_chunk_stats(index);
  assert(chunk_ptrs_.size() > chunk_idx);
  auto *raw_ptr = chunk_ptrs_[chunk_idx].deref(scope);
  return *(reinterpret_cast<const T *>(raw_ptr) + chunk_offset);
}

} // namespace far_memory
