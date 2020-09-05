#pragma once

extern "C" {
#include <runtime/preempt.h>
}

#include "helpers.hpp"

#include <cstring>
#include <memory>
#include <stack>
#include <type_traits>
#include <vector>

namespace far_memory {

template <typename T> class LocalList;

template <typename NodePtr> struct GenericLocalListNode {
  NodePtr next;
  NodePtr prev;
  uint8_t data[0];
};

#pragma pack(push, 1)
template <typename NodePtr> struct GenericLocalListData {
  GenericLocalListNode<NodePtr> head;
  GenericLocalListNode<NodePtr> tail;
  NodePtr head_ptr;
  NodePtr tail_ptr;
  uint8_t data[0];
};
#pragma pack(pop)

template <typename NodePtr, typename DerefFn, typename AllocateFn,
          typename FreeFn>
class GenericLocalList {
private:
  using Node = GenericLocalListNode<NodePtr>;
  using ListData = GenericLocalListData<NodePtr>;
  using DerefFnTraits = helpers::FunctionTraits<DerefFn>;
  using AllocateFnTraits = helpers::FunctionTraits<AllocateFn>;
  using FreeFnTraits = helpers::FunctionTraits<FreeFn>;
  using FnStateType = typename DerefFnTraits::template Arg<1>::Type;
  constexpr static bool kFnHasState = (DerefFnTraits::Arity == 2);
  using StateType =
      std::conditional<kFnHasState, FnStateType, uint8_t[0]>::type;

  // DerefFn: (NodePtr, [optional]FnStateType)->Node &
  static_assert(DerefFnTraits::Arity == 1 || DerefFnTraits::Arity == 2);
  static_assert(std::is_same<
                NodePtr, typename DerefFnTraits::template Arg<0>::Type>::value);
  static_assert(
      std::is_same<FnStateType,
                   typename DerefFnTraits::template Arg<1>::Type>::value);
  static_assert(
      std::is_same<Node &, typename DerefFnTraits::ResultType>::value);

  // AllocateFn: ([optional]FnStateType)->NodePtr
  static_assert(AllocateFnTraits::Arity + 1 == DerefFnTraits::Arity);
  static_assert(
      std::is_same<FnStateType,
                   typename AllocateFnTraits::template Arg<0>::Type>::value);
  static_assert(
      std::is_same<NodePtr, typename AllocateFnTraits::ResultType>::value);

  // FreeFn: (NodePtr, [optional]FnStateType)->void
  static_assert(static_cast<int>(FreeFnTraits::Arity) == DerefFnTraits::Arity);
  static_assert(std::is_same<
                NodePtr, typename FreeFnTraits::template Arg<0>::Type>::value);
  static_assert(
      std::is_same<FnStateType,
                   typename FreeFnTraits::template Arg<1>::Type>::value);
  static_assert(std::is_same<void, typename FreeFnTraits::ResultType>::value);

  template <bool Reverse> class IteratorImpl {
  private:
    NodePtr ptr_;
    StateType state_;

    friend class GenericLocalList;
    friend class GenericList;
    template <typename T> friend class LocalList;

    Node &deref(NodePtr ptr) const;
    uint8_t *insert();
    IteratorImpl<Reverse> erase(uint8_t **data_ptr);
    NodePtr allocate();
    void free(NodePtr node_ptr);

  public:
    IteratorImpl();
    IteratorImpl(NodePtr ptr, StateType state);
    template <bool OReverse> IteratorImpl(const IteratorImpl<OReverse> &o);
    template <bool OReverse>
    IteratorImpl &operator=(const IteratorImpl<OReverse> &o);
    IteratorImpl &operator++();
    IteratorImpl operator++(int);
    IteratorImpl &operator--();
    IteratorImpl operator--(int);
    bool operator==(const IteratorImpl &o) const;
    bool operator!=(const IteratorImpl &o) const;
    uint8_t *operator*() const;
  };

  using Iterator = IteratorImpl</* Reverse */ false>;
  using ReverseIterator = IteratorImpl</* Reverse */ true>;

  GenericLocalList();

  ListData *list_data_;
  StateType state_;
  friend class GenericList;
  template <typename T> friend class LocalList;
  template <typename T> friend class List;

public:
  GenericLocalList(ListData *list_data);
  GenericLocalList(ListData *list_data, StateType state);
  void init(NodePtr head_ptr, NodePtr tail_ptr);
  void set_list_data(ListData *list_data);
  Iterator begin() const;
  Iterator end() const;
  ReverseIterator rbegin() const;
  ReverseIterator rend() const;
  template <bool Reverse> uint8_t *insert(const IteratorImpl<Reverse> &iter);
  template <bool Reverse>
  IteratorImpl<Reverse> erase(const IteratorImpl<Reverse> &iter,
                              uint8_t **data_ptr);
};

template <typename T> class LocalList {
private:
  using NodePtr = uint8_t *;
  using NodePool = std::stack<NodePtr, std::vector<NodePtr>>;
  using AutoNodeCleaner = std::vector<std::unique_ptr<uint8_t>>;

  constexpr static uint32_t kReplenishNumNodes = 8192;
  constexpr static auto kDerefNodePtrFn =
      [](NodePtr ptr, NodePool *pool) -> GenericLocalListNode<NodePtr> & {
    return *reinterpret_cast<GenericLocalListNode<NodePtr> *>(ptr);
  };
  constexpr static auto kAllocateNodeFn = [](NodePool *node_pool) {
    if (unlikely(node_pool->empty())) {
      preempt_disable();
      auto auto_node_cleaner = reinterpret_cast<AutoNodeCleaner *>(
          reinterpret_cast<uint64_t>(node_pool) + sizeof(NodePool));
      constexpr auto kNodeSize =
          sizeof(GenericLocalListNode<NodePtr>) + sizeof(T);
      auto mem =
          reinterpret_cast<uint8_t *>(malloc(kReplenishNumNodes * kNodeSize));
      auto_node_cleaner->push_back(std::unique_ptr<uint8_t>(mem));
      auto node = reinterpret_cast<NodePtr>(mem);
      for (uint64_t i = 0; i < kReplenishNumNodes; i++) {
        node_pool->push(node);
        node += kNodeSize;
      }
      preempt_enable();
    }
    auto ret = node_pool->top();
    node_pool->pop();
    return ret;
  };
  constexpr static auto kFreeNodeFn = [](NodePtr ptr, NodePool *node_pool) {
    node_pool->push(ptr);
    reinterpret_cast<T *>(ptr + sizeof(GenericLocalListNode<NodePtr>))->~T();
  };

  using TGenericLocalList =
      GenericLocalList<NodePtr, decltype(kDerefNodePtrFn),
                       decltype(kAllocateNodeFn), decltype(kFreeNodeFn)>;

  template <bool Reverse> class IteratorImpl {
  private:
    TGenericLocalList::template IteratorImpl<Reverse> generic_iter_;
    friend class LocalList;

  public:
    IteratorImpl();
    template <bool OReverse>
    IteratorImpl(
        TGenericLocalList::template IteratorImpl<OReverse> generic_iter);
    template <bool OReverse> IteratorImpl(const IteratorImpl<OReverse> &o);
    template <bool OReverse>
    IteratorImpl &operator=(const IteratorImpl<OReverse> &o);
    IteratorImpl &operator++();
    IteratorImpl operator++(int);
    IteratorImpl &operator--();
    IteratorImpl operator--(int);
    bool operator==(const IteratorImpl &o) const;
    bool operator!=(const IteratorImpl &o) const;
    T &operator*() const;
    T *operator->() const;
  };

  using Iterator = IteratorImpl</* Reverse */ false>;
  using ReverseIterator = IteratorImpl</* Reverse */ true>;

  NodePool node_pool_;
  AutoNodeCleaner auto_cleaner_;
  TGenericLocalList::ListData list_data_;
  TGenericLocalList generic_local_list_;
  uint64_t size_ = 0;
  friend class GenericList;

public:
  ~LocalList();
  LocalList();
  Iterator begin() const;
  Iterator end() const;
  ReverseIterator rbegin() const;
  ReverseIterator rend() const;
  T &front() const;
  T &back() const;
  uint64_t size() const;
  bool empty() const;
  void push_front(const T &data);
  void pop_front();
  void push_back(const T &data);
  void pop_back();
  template <bool Reverse>
  void insert(const IteratorImpl<Reverse> &iter, const T &data);
  template <bool Reverse>
  IteratorImpl<Reverse> erase(const IteratorImpl<Reverse> &iter);
};

template <typename NodePtr, typename DerefFn, typename AllocateFn,
          typename FreeFn>
template <bool Reverse>
FORCE_INLINE GenericLocalList<NodePtr, DerefFn, AllocateFn,
                              FreeFn>::IteratorImpl<Reverse>::IteratorImpl() {}

template <typename NodePtr, typename DerefFn, typename AllocateFn,
          typename FreeFn>
template <bool Reverse>
FORCE_INLINE
GenericLocalList<NodePtr, DerefFn, AllocateFn,
                 FreeFn>::IteratorImpl<Reverse>::IteratorImpl(NodePtr ptr,
                                                              StateType state)
    : ptr_(ptr), state_(state) {}

template <typename NodePtr, typename DerefFn, typename AllocateFn,
          typename FreeFn>
template <bool Reverse>
template <bool OReverse>
FORCE_INLINE
GenericLocalList<NodePtr, DerefFn, AllocateFn, FreeFn>::IteratorImpl<
    Reverse>::IteratorImpl(const IteratorImpl<OReverse> &o)
    : ptr_(o.ptr_), state_(o.state_) {}

template <typename NodePtr, typename DerefFn, typename AllocateFn,
          typename FreeFn>
template <bool Reverse>
template <bool OReverse>
FORCE_INLINE GenericLocalList<NodePtr, DerefFn, AllocateFn,
                              FreeFn>::template IteratorImpl<Reverse> &
    GenericLocalList<NodePtr, DerefFn, AllocateFn,
                     FreeFn>::template IteratorImpl<Reverse>::
    operator=(const IteratorImpl<OReverse> &o) {
  this->ptr_ = o.ptr_;
  this->state_ = o.state_;
  return *this;
}

template <typename NodePtr, typename DerefFn, typename AllocateFn,
          typename FreeFn>
template <bool Reverse>
FORCE_INLINE GenericLocalListNode<NodePtr> &
GenericLocalList<NodePtr, DerefFn, AllocateFn,
                 FreeFn>::IteratorImpl<Reverse>::deref(NodePtr ptr) const {
  const DerefFn kDerefNodePtrFn;
  if constexpr (kFnHasState) {
    return kDerefNodePtrFn(ptr, state_);
  } else {
    return kDerefNodePtrFn(ptr);
  }
}

template <typename NodePtr, typename DerefFn, typename AllocateFn,
          typename FreeFn>
template <bool Reverse>
FORCE_INLINE NodePtr GenericLocalList<
    NodePtr, DerefFn, AllocateFn, FreeFn>::IteratorImpl<Reverse>::allocate() {
  const AllocateFn kAllocateNodeFn;
  if constexpr (kFnHasState) {
    return kAllocateNodeFn(state_);
  } else {
    return kAllocateNodeFn();
  }
}

template <typename NodePtr, typename DerefFn, typename AllocateFn,
          typename FreeFn>
template <bool Reverse>
FORCE_INLINE void
GenericLocalList<NodePtr, DerefFn, AllocateFn,
                 FreeFn>::IteratorImpl<Reverse>::free(NodePtr node_ptr) {
  const FreeFn kFreeNodeFn;
  if constexpr (kFnHasState) {
    return kFreeNodeFn(node_ptr, state_);
  } else {
    return kFreeNodeFn(node_ptr);
  }
}

template <typename NodePtr, typename DerefFn, typename AllocateFn,
          typename FreeFn>
template <bool Reverse>
FORCE_INLINE uint8_t *
GenericLocalList<NodePtr, DerefFn, AllocateFn,
                 FreeFn>::IteratorImpl<Reverse>::insert() {
  auto *node = &deref(ptr_);
  auto new_node_ptr = allocate();
  auto *new_node = &deref(new_node_ptr);

  if constexpr (Reverse) {
    new_node->next = node->next;
    auto *next_node = &deref(node->next);
    next_node->prev = new_node_ptr;
    new_node->prev = ptr_;
    node->next = new_node_ptr;
  } else {
    new_node->prev = node->prev;
    auto *prev_node = &deref(node->prev);
    prev_node->next = new_node_ptr;
    new_node->next = ptr_;
    node->prev = new_node_ptr;
  }
  return reinterpret_cast<uint8_t *>(new_node->data);
}

template <typename NodePtr, typename DerefFn, typename AllocateFn,
          typename FreeFn>
template <bool Reverse>
FORCE_INLINE GenericLocalList<NodePtr, DerefFn, AllocateFn,
                              FreeFn>::template IteratorImpl<Reverse>
GenericLocalList<NodePtr, DerefFn, AllocateFn,
                 FreeFn>::IteratorImpl<Reverse>::erase(uint8_t **data_ptr) {
  auto &node = deref(ptr_);
  *data_ptr = reinterpret_cast<uint8_t *>(node.data);
  auto &prev_node = deref(node.prev);
  prev_node.next = node.next;
  auto &next_node = deref(node.next);
  next_node.prev = node.prev;
  free(ptr_);
  if constexpr (Reverse) {
    return IteratorImpl<Reverse>(node.prev, state_);
  } else {
    return IteratorImpl<Reverse>(node.next, state_);
  }
}

template <typename NodePtr, typename DerefFn, typename AllocateFn,
          typename FreeFn>
template <bool Reverse>
FORCE_INLINE GenericLocalList<NodePtr, DerefFn, AllocateFn,
                              FreeFn>::template IteratorImpl<Reverse> &
GenericLocalList<NodePtr, DerefFn, AllocateFn, FreeFn>::IteratorImpl<Reverse>::
operator++() {
  if constexpr (Reverse) {
    ptr_ = deref(ptr_).prev;
  } else {
    ptr_ = deref(ptr_).next;
  }
  return *this;
}

template <typename NodePtr, typename DerefFn, typename AllocateFn,
          typename FreeFn>
template <bool Reverse>
FORCE_INLINE GenericLocalList<NodePtr, DerefFn, AllocateFn,
                              FreeFn>::template IteratorImpl<Reverse>
GenericLocalList<NodePtr, DerefFn, AllocateFn, FreeFn>::IteratorImpl<Reverse>::
operator++(int) {
  auto ret = *this;
  operator++();
  return ret;
}

template <typename NodePtr, typename DerefFn, typename AllocateFn,
          typename FreeFn>
template <bool Reverse>
FORCE_INLINE GenericLocalList<NodePtr, DerefFn, AllocateFn,
                              FreeFn>::template IteratorImpl<Reverse> &
GenericLocalList<NodePtr, DerefFn, AllocateFn, FreeFn>::IteratorImpl<Reverse>::
operator--() {
  if constexpr (Reverse) {
    ptr_ = deref(ptr_).next;
  } else {
    ptr_ = deref(ptr_).prev;
  }
  return *this;
}

template <typename NodePtr, typename DerefFn, typename AllocateFn,
          typename FreeFn>
template <bool Reverse>
FORCE_INLINE GenericLocalList<NodePtr, DerefFn, AllocateFn,
                              FreeFn>::template IteratorImpl<Reverse>
GenericLocalList<NodePtr, DerefFn, AllocateFn, FreeFn>::IteratorImpl<Reverse>::
operator--(int) {
  auto ret = *this;
  operator--();
  return ret;
}

template <typename NodePtr, typename DerefFn, typename AllocateFn,
          typename FreeFn>
template <bool Reverse>
FORCE_INLINE bool
GenericLocalList<NodePtr, DerefFn, AllocateFn, FreeFn>::IteratorImpl<Reverse>::
operator==(const IteratorImpl &o) const {
  return this->ptr_ == o.ptr_;
}

template <typename NodePtr, typename DerefFn, typename AllocateFn,
          typename FreeFn>
template <bool Reverse>
FORCE_INLINE bool
GenericLocalList<NodePtr, DerefFn, AllocateFn, FreeFn>::IteratorImpl<Reverse>::
operator!=(const IteratorImpl &o) const {
  return this->ptr_ != o.ptr_;
}

template <typename NodePtr, typename DerefFn, typename AllocateFn,
          typename FreeFn>
template <bool Reverse>
FORCE_INLINE uint8_t *GenericLocalList<NodePtr, DerefFn, AllocateFn,
                                       FreeFn>::IteratorImpl<Reverse>::
operator*() const {
  return reinterpret_cast<uint8_t *>(deref(ptr_).data);
}

template <typename NodePtr, typename DerefFn, typename AllocateFn,
          typename FreeFn>
FORCE_INLINE
GenericLocalList<NodePtr, DerefFn, AllocateFn, FreeFn>::GenericLocalList() {}

template <typename NodePtr, typename DerefFn, typename AllocateFn,
          typename FreeFn>
FORCE_INLINE
GenericLocalList<NodePtr, DerefFn, AllocateFn, FreeFn>::GenericLocalList(
    ListData *list_data)
    : list_data_(list_data) {}

template <typename NodePtr, typename DerefFn, typename AllocateFn,
          typename FreeFn>
FORCE_INLINE
GenericLocalList<NodePtr, DerefFn, AllocateFn, FreeFn>::GenericLocalList(
    ListData *list_data, StateType state)
    : list_data_(list_data), state_(state) {}

template <typename NodePtr, typename DerefFn, typename AllocateFn,
          typename FreeFn>
FORCE_INLINE void
GenericLocalList<NodePtr, DerefFn, AllocateFn, FreeFn>::init(NodePtr head_ptr,
                                                             NodePtr tail_ptr) {
  list_data_->head_ptr = head_ptr;
  list_data_->tail_ptr = tail_ptr;
  list_data_->head.next = tail_ptr;
  list_data_->tail.prev = head_ptr;
}

template <typename NodePtr, typename DerefFn, typename AllocateFn,
          typename FreeFn>
FORCE_INLINE void
GenericLocalList<NodePtr, DerefFn, AllocateFn, FreeFn>::set_list_data(
    ListData *list_data) {
  list_data_ = list_data;
}

template <typename NodePtr, typename DerefFn, typename AllocateFn,
          typename FreeFn>
FORCE_INLINE GenericLocalList<NodePtr, DerefFn, AllocateFn, FreeFn>::Iterator
GenericLocalList<NodePtr, DerefFn, AllocateFn, FreeFn>::begin() const {
  return Iterator(list_data_->head.next, state_);
}

template <typename NodePtr, typename DerefFn, typename AllocateFn,
          typename FreeFn>
FORCE_INLINE GenericLocalList<NodePtr, DerefFn, AllocateFn, FreeFn>::Iterator
GenericLocalList<NodePtr, DerefFn, AllocateFn, FreeFn>::end() const {
  return Iterator(list_data_->tail_ptr, state_);
}

template <typename NodePtr, typename DerefFn, typename AllocateFn,
          typename FreeFn>
FORCE_INLINE
    GenericLocalList<NodePtr, DerefFn, AllocateFn, FreeFn>::ReverseIterator
    GenericLocalList<NodePtr, DerefFn, AllocateFn, FreeFn>::rbegin() const {
  return ReverseIterator(list_data_->tail.prev, state_);
}

template <typename NodePtr, typename DerefFn, typename AllocateFn,
          typename FreeFn>
FORCE_INLINE
    GenericLocalList<NodePtr, DerefFn, AllocateFn, FreeFn>::ReverseIterator
    GenericLocalList<NodePtr, DerefFn, AllocateFn, FreeFn>::rend() const {
  return ReverseIterator(list_data_->head_ptr, state_);
}

template <typename NodePtr, typename DerefFn, typename AllocateFn,
          typename FreeFn>
template <bool Reverse>
FORCE_INLINE uint8_t *
GenericLocalList<NodePtr, DerefFn, AllocateFn, FreeFn>::insert(
    const IteratorImpl<Reverse> &iter) {
  return const_cast<IteratorImpl<Reverse> *>(&iter)->insert();
}

template <typename NodePtr, typename DerefFn, typename AllocateFn,
          typename FreeFn>
template <bool Reverse>
FORCE_INLINE GenericLocalList<NodePtr, DerefFn, AllocateFn,
                              FreeFn>::template IteratorImpl<Reverse>
GenericLocalList<NodePtr, DerefFn, AllocateFn, FreeFn>::erase(
    const IteratorImpl<Reverse> &iter, uint8_t **data_ptr) {
  return const_cast<IteratorImpl<Reverse> *>(&iter)->erase(data_ptr);
}

template <typename T>
template <bool Reverse>
FORCE_INLINE LocalList<T>::template IteratorImpl<Reverse>::IteratorImpl() {}

template <typename T>
template <bool Reverse>
template <bool OReverse>
FORCE_INLINE LocalList<T>::template IteratorImpl<Reverse>::IteratorImpl(
    TGenericLocalList::template IteratorImpl<OReverse> generic_iter)
    : generic_iter_(generic_iter) {}

template <typename T>
template <bool Reverse>
template <bool OReverse>
FORCE_INLINE LocalList<T>::template IteratorImpl<Reverse>::IteratorImpl(
    const IteratorImpl<OReverse> &o)
    : generic_iter_(o.generic_iter_) {}

template <typename T>
template <bool Reverse>
template <bool OReverse>
FORCE_INLINE LocalList<T>::template IteratorImpl<Reverse> &
    LocalList<T>::template IteratorImpl<Reverse>::
    operator=(const IteratorImpl<OReverse> &o) {
  generic_iter_ = o.generic_iter_;
  return *this;
}

template <typename T>
template <bool Reverse>
FORCE_INLINE LocalList<T>::template IteratorImpl<Reverse> &
LocalList<T>::IteratorImpl<Reverse>::operator++() {
  generic_iter_.operator++();
  return *this;
}

template <typename T>
template <bool Reverse>
FORCE_INLINE LocalList<T>::template IteratorImpl<Reverse>
LocalList<T>::IteratorImpl<Reverse>::operator++(int) {
  auto ret = *this;
  generic_iter_.operator++();
  return ret;
}

template <typename T>
template <bool Reverse>
FORCE_INLINE LocalList<T>::template IteratorImpl<Reverse> &
LocalList<T>::IteratorImpl<Reverse>::operator--() {
  generic_iter_.operator--();
  return *this;
}

template <typename T>
template <bool Reverse>
FORCE_INLINE LocalList<T>::template IteratorImpl<Reverse>
LocalList<T>::IteratorImpl<Reverse>::operator--(int) {
  auto ret = *this;
  generic_iter_.operator--();
  return ret;
}

template <typename T>
template <bool Reverse>
FORCE_INLINE bool LocalList<T>::IteratorImpl<Reverse>::
operator==(const IteratorImpl &o) const {
  return this->generic_iter_ == o.generic_iter_;
}

template <typename T>
template <bool Reverse>
FORCE_INLINE bool LocalList<T>::IteratorImpl<Reverse>::
operator!=(const IteratorImpl &o) const {
  return this->generic_iter_ != o.generic_iter_;
}

template <typename T>
template <bool Reverse>
FORCE_INLINE T &LocalList<T>::IteratorImpl<Reverse>::operator*() const {
  return *reinterpret_cast<T *>(*generic_iter_);
}

template <typename T>
template <bool Reverse>
FORCE_INLINE T *LocalList<T>::IteratorImpl<Reverse>::operator->() const {
  return reinterpret_cast<T *>(*generic_iter_);
}

template <typename T> FORCE_INLINE LocalList<T>::~LocalList() {
  while (unlikely(!empty())) {
    pop_back();
  }
}

template <typename T>
FORCE_INLINE LocalList<T>::LocalList()
    : generic_local_list_(&list_data_, &node_pool_) {
  generic_local_list_.init(reinterpret_cast<NodePtr>(&(list_data_.head)),
                           reinterpret_cast<NodePtr>(&(list_data_.tail)));
}

template <typename T>
FORCE_INLINE LocalList<T>::Iterator LocalList<T>::begin() const {
  auto ret = Iterator(generic_local_list_.begin());
  return ret;
}

template <typename T>
FORCE_INLINE LocalList<T>::Iterator LocalList<T>::end() const {
  auto ret = Iterator(generic_local_list_.end());
  return ret;
}

template <typename T>
FORCE_INLINE LocalList<T>::ReverseIterator LocalList<T>::rbegin() const {
  auto ret = ReverseIterator(generic_local_list_.rbegin());
  return ret;
}

template <typename T>
FORCE_INLINE LocalList<T>::ReverseIterator LocalList<T>::rend() const {
  auto ret = ReverseIterator(generic_local_list_.rend());
  return ret;
}

template <typename T> FORCE_INLINE T &LocalList<T>::front() const {
  auto begin_iter = generic_local_list_.begin();
  return *reinterpret_cast<T *>(*begin_iter);
}

template <typename T> FORCE_INLINE T &LocalList<T>::back() const {
  auto rbegin_iter = generic_local_list_.rbegin();
  return *reinterpret_cast<T *>(*rbegin_iter);
}

template <typename T> FORCE_INLINE uint64_t LocalList<T>::size() const {
  return size_;
}

template <typename T> FORCE_INLINE bool LocalList<T>::empty() const {
  return size_ == 0;
}

template <typename T>
FORCE_INLINE void LocalList<T>::push_front(const T &data) {
  auto begin_iter = generic_local_list_.begin();
  auto *data_ptr = generic_local_list_.insert(begin_iter);
  memcpy(data_ptr, &data, sizeof(data));
  size_++;
}

template <typename T> FORCE_INLINE void LocalList<T>::pop_front() {
  auto begin_iter = generic_local_list_.begin();
  uint8_t *data_ptr;
  generic_local_list_.erase(begin_iter, &data_ptr);
  size_--;
}

template <typename T> FORCE_INLINE void LocalList<T>::push_back(const T &data) {
  auto rbegin_iter = generic_local_list_.rbegin();
  auto *data_ptr = generic_local_list_.insert(rbegin_iter);
  memcpy(data_ptr, &data, sizeof(data));
  size_++;
}

template <typename T> FORCE_INLINE void LocalList<T>::pop_back() {
  auto rbegin_iter = generic_local_list_.rbegin();
  uint8_t *data_ptr;
  generic_local_list_.erase(rbegin_iter, &data_ptr);
  size_--;
}

template <typename T>
template <bool Reverse>
FORCE_INLINE void LocalList<T>::insert(const IteratorImpl<Reverse> &iter,
                                       const T &data) {
  auto *data_ptr = generic_local_list_.insert(iter.generic_iter_);
  memcpy(data_ptr, &data, sizeof(data));
  size_++;
}

template <typename T>
template <bool Reverse>
FORCE_INLINE LocalList<T>::template IteratorImpl<Reverse>
LocalList<T>::erase(const IteratorImpl<Reverse> &iter) {
  size_--;
  uint8_t *data_ptr;
  auto ret = IteratorImpl<Reverse>(
      generic_local_list_.erase(iter.generic_iter_, &data_ptr));
  return ret;
}

} // namespace far_memory
