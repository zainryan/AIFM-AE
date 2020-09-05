#pragma once

#include "list.hpp"

namespace far_memory {
template <typename T> class Queue {
private:
  Queue(const DerefScope &scope);

  List<T> list_;
  friend class FarMemManager;

public:
  bool empty() const;
  uint64_t size() const;
  const T &cfront(const DerefScope &scope) const;
  const T &cback(const DerefScope &scope) const;
  T &front(const DerefScope &scope) const;
  T &back(const DerefScope &scope) const;
  void push(const DerefScope &scope, const T &data);
  void pop(const DerefScope &scope);
};

template <typename T>
FORCE_INLINE Queue<T>::Queue(const DerefScope &scope)
    : list_(scope, /* enable_emerge = */ false, /* customized_split = */ true) {
}

template <typename T> FORCE_INLINE bool Queue<T>::empty() const {
  return list_.empty();
}

template <typename T> FORCE_INLINE uint64_t Queue<T>::size() const {
  return list_.size();
}

template <typename T>
FORCE_INLINE const T &Queue<T>::cfront(const DerefScope &scope) const {
  return list_.cfront(scope);
}

template <typename T>
FORCE_INLINE const T &Queue<T>::cback(const DerefScope &scope) const {
  return list_.cback(scope);
}

template <typename T>
FORCE_INLINE T &Queue<T>::front(const DerefScope &scope) const {
  return list_.front(scope);
}

template <typename T>
FORCE_INLINE T &Queue<T>::back(const DerefScope &scope) const {
  return list_.back(scope);
}

template <typename T>
FORCE_INLINE void Queue<T>::push(const DerefScope &scope, const T &data) {
  list_.push_back(scope, data);
}

template <typename T> FORCE_INLINE void Queue<T>::pop(const DerefScope &scope) {
  list_.pop_front(scope);
}

} // namespace far_memory
