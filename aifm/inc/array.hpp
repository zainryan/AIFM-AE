#pragma once

#include "sync.h"
#include "thread.h"

#include "deref_scope.hpp"
#include "helpers.hpp"
#include "pointer.hpp"
#include "prefetcher.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

namespace far_memory {

class FarMemManager;

class GenericArray {
protected:
  using Index_t = uint64_t;
  using Pattern_t = int64_t;

  std::unique_ptr<GenericUniquePtr[]> ptrs_;
  uint32_t kItemSize_;
  uint64_t kNumItems_;
  bool dynamic_prefetch_enabled_ = true;
  std::function<Pattern_t(Index_t, Index_t)> induce_;
  std::function<Index_t(Index_t, Pattern_t)> infer_;
  std::function<GenericUniquePtr *(Index_t)> mapping_;
  Prefetcher<Index_t, Pattern_t> prefetcher_;

  GenericArray(FarMemManager *manager, uint32_t item_size, uint64_t num_items);
  ~GenericArray();
  NOT_COPYABLE(GenericArray);
  NOT_MOVEABLE(GenericArray);

public:
  void disable_prefetch();
  void enable_prefetch();
  void static_prefetch(Index_t start, Index_t step, uint32_t num);
  GenericUniquePtr *at(bool nt, Index_t idx);
};

template <typename T, uint64_t... Dims> class Array : public GenericArray {
private:
  using num_dim_t = uint8_t;
  static_assert(std::numeric_limits<num_dim_t>::max() >= sizeof...(Dims));

  friend class FarMemManager;
  friend class FarMemTest;

  Array(FarMemManager *manager);
  NOT_COPYABLE(Array);
  NOT_MOVEABLE(Array);

  template <auto DimIdx> static constexpr uint64_t get_dim_size();

  static constexpr uint64_t _size(uint64_t N) { return N; }

  template <typename... Args>
  static constexpr uint64_t _size(uint64_t N, Args... rest_dims) {
    return N * _size(rest_dims...);
  }

public:
  static constexpr uint64_t kSize = _size(Dims...);

  template <auto DimIdx, typename... Indices>
  static constexpr int64_t _get_flat_idx(Indices... indices);
  template <typename... Indices>
  static constexpr int64_t get_flat_idx(Indices... indices);
  template <typename... Indices> void check_indices(Indices... indices);
  template <bool Nt = false, typename... Indices>
  const T &at(const DerefScope &scope, Indices... indices) noexcept;
  template <bool Nt = false, typename... Indices>
  const T &at_safe(const DerefScope &scope, Indices... indices);
  template <bool Nt = false, typename... Indices> T read(Indices... indices);
  template <bool Nt = false, typename... Indices>
  T read_safe(Indices... indices);
  template <bool Nt = false, typename... Indices>
  T &at_mut(const DerefScope &scope, Indices... indices) noexcept;
  template <bool Nt = false, typename... Indices>
  T &at_mut_safe(const DerefScope &scope, Indices... indices);
  template <bool Nt = false, typename U, typename... Indices>
  void write(U &&u, Indices... indices);
  template <bool Nt = false, typename U, typename... Indices>
  void write_safe(U &&u, Indices... indices);
  template <typename... ArgsStart, typename... ArgsStep>
  void static_prefetch(std::tuple<ArgsStart...> start,
                       std::tuple<ArgsStep...> step, uint32_t num);
};

FORCE_INLINE GenericUniquePtr *GenericArray::at(bool nt, Index_t idx) {
  if (ACCESS_ONCE(dynamic_prefetch_enabled_)) {
    prefetcher_.add_trace(nt, idx);
  }
  return &ptrs_[idx];
}

template <typename T, uint64_t... Dims>
FORCE_INLINE Array<T, Dims...>::Array(FarMemManager *manager)
    : GenericArray(manager, sizeof(T), kSize) {}

template <typename T, uint64_t... Dims>
template <auto DimIdx>
FORCE_INLINE constexpr uint64_t Array<T, Dims...>::get_dim_size() {
  if constexpr (DimIdx == sizeof...(Dims)) {
    return 1;
  } else {
    return helpers::variadic_get<DimIdx>(Dims...) * get_dim_size<DimIdx + 1>();
  }
}

template <typename T, uint64_t... Dims>
template <auto DimIdx, typename... Indices>
FORCE_INLINE constexpr int64_t
Array<T, Dims...>::_get_flat_idx(Indices... indices) {
  if constexpr (DimIdx == sizeof...(Dims)) {
    return 0;
  } else {
    return static_cast<int64_t>(helpers::variadic_get<DimIdx>(indices...)) *
               static_cast<int64_t>(get_dim_size<DimIdx + 1>()) +
           static_cast<int64_t>(_get_flat_idx<DimIdx + 1>(indices...));
  }
}

template <typename T, uint64_t... Dims>
template <typename... Indices>
FORCE_INLINE constexpr int64_t
Array<T, Dims...>::get_flat_idx(Indices... indices) {
  static_assert(sizeof...(Dims) == sizeof...(indices));
  return _get_flat_idx<0>(indices...);
}

template <typename T, uint64_t... Dims>
template <typename... Indices>
FORCE_INLINE void Array<T, Dims...>::check_indices(Indices... indices) {
  static_assert(sizeof...(Dims) == sizeof...(indices));
  for (num_dim_t i = 0; i < sizeof...(Dims); i++) {
    if (unlikely(helpers::variadic_get<i>(indices...) >=
                     helpers::variadic_get<i>(Dims...) ||
                 helpers::variadic_get<i>(indices...) < 0)) {
      throw std::invalid_argument("Index of out range.");
    }
  }
}

template <typename T, uint64_t... Dims>
template <bool Nt, typename... Indices>
FORCE_INLINE const T &Array<T, Dims...>::at(const DerefScope &scope,
                                            Indices... indices) noexcept {
  auto idx = get_flat_idx(indices...);
  auto ptr = reinterpret_cast<UniquePtr<T> *>(GenericArray::at(Nt, idx));
  return *(ptr->template deref<Nt>(scope));
}

template <typename T, uint64_t... Dims>
template <bool Nt, typename... Indices>
FORCE_INLINE const T &Array<T, Dims...>::at_safe(const DerefScope &scope,
                                                 Indices... indices) {
  check_indices(indices...);
  return at(scope, indices...);
}

template <typename T, uint64_t... Dims>
template <bool Nt, typename... Indices>
FORCE_INLINE T Array<T, Dims...>::read(Indices... indices) {
  DerefScope scope;
  return at(scope, indices...);
}

template <typename T, uint64_t... Dims>
template <bool Nt, typename... Indices>
FORCE_INLINE T Array<T, Dims...>::read_safe(Indices... indices) {
  check_indices(indices...);
  return read(indices...);
}

template <typename T, uint64_t... Dims>
template <bool Nt, typename... Indices>
FORCE_INLINE T &Array<T, Dims...>::at_mut(const DerefScope &scope,
                                          Indices... indices) noexcept {
  auto idx = get_flat_idx(indices...);
  auto ptr = reinterpret_cast<UniquePtr<T> *>(GenericArray::at(Nt, idx));
  return *(ptr->template deref_mut<Nt>(scope));
}

template <typename T, uint64_t... Dims>
template <bool Nt, typename... Indices>
FORCE_INLINE T &Array<T, Dims...>::at_mut_safe(const DerefScope &scope,
                                               Indices... indices) {
  check_indices(indices...);
  return at_mut(scope, indices...);
}

template <typename T, uint64_t... Dims>
template <bool Nt, typename U, typename... Indices>
FORCE_INLINE void Array<T, Dims...>::write(U &&u, Indices... indices) {
  static_assert(std::is_same<std::decay_t<U>, std::decay_t<T>>::value,
                "U must be the same as T");
  DerefScope scope;
  at_mut(scope, indices...) = u;
}

template <typename T, uint64_t... Dims>
template <bool Nt, typename U, typename... Indices>
FORCE_INLINE void Array<T, Dims...>::write_safe(U &&u, Indices... indices) {
  check_indices(indices...);
  write_safe(indices..., u);
}

template <typename T, uint64_t... Dims>
template <typename... ArgsStart, typename... ArgsStep>
FORCE_INLINE void
Array<T, Dims...>::static_prefetch(std::tuple<ArgsStart...> start,
                                   std::tuple<ArgsStep...> step, uint32_t num) {
  // Get their flat indices.
  auto start_flat_idx =
      std::apply([&](auto &&... args) { return get_flat_idx(args...); }, start);
  auto step_flat_idx =
      std::apply([&](auto &&... args) { return get_flat_idx(args...); }, step);
  GenericArray::static_prefetch(start_flat_idx, step_flat_idx, num);
}

} // namespace far_memory
