#pragma once

extern "C" {
#include <base/assert.h>
}

#include "thread.h"

#include "deref_scope.hpp"
#include "local_list.hpp"
#include "pointer.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>

namespace far_memory {

class GenericList {
private:
  struct ChunkListData {
    uint64_t meta;
    uint8_t data[0];
  };

#pragma pack(push, 1)
  struct ChunkNodePtr {
    uint8_t idx;
    uint16_t addr_offset;

    ChunkNodePtr();
    ChunkNodePtr(const ChunkNodePtr &o);
    constexpr ChunkNodePtr(uint8_t _idx, uint16_t _addr_offset);
    bool operator==(const ChunkNodePtr &o) const;
    bool operator!=(const ChunkNodePtr &o) const;
  };
#pragma pack(pop)

  struct ChunkListState {
    uint64_t list_data_ptr_addr : 48;
    uint16_t kChunkListNodeSize : 16;
  };

  constexpr static auto kDerefNodePtrFn =
      [](ChunkNodePtr ptr,
         ChunkListState state) -> GenericLocalListNode<ChunkNodePtr> & {
    uint64_t *list_data_ptr = reinterpret_cast<uint64_t *>(
        static_cast<uint64_t>(state.list_data_ptr_addr));
    return *reinterpret_cast<GenericLocalListNode<ChunkNodePtr> *>(
        ptr.addr_offset + (*list_data_ptr));
  };
  constexpr static auto kAllocateNodeFn = [](ChunkListState state) {
    uint64_t *list_data_ptr = reinterpret_cast<uint64_t *>(
        static_cast<uint64_t>(state.list_data_ptr_addr));
    auto chunk_list_data = reinterpret_cast<ChunkListData *>(
        static_cast<uint64_t>(*list_data_ptr) - sizeof(ChunkListData));
    auto *meta = &(chunk_list_data->meta);
    assert(*meta);
    auto idx = helpers::bsf_64(*meta);
    auto meta_mask = (static_cast<uint64_t>(1) << idx);
    *meta ^= meta_mask;
    return ChunkNodePtr(idx, sizeof(GenericLocalListData<ChunkNodePtr>) +
                                 idx * state.kChunkListNodeSize);
  };
  constexpr static auto kFreeNodeFn = [](ChunkNodePtr ptr,
                                         ChunkListState state) {
    uint64_t *list_data_ptr = reinterpret_cast<uint64_t *>(
        static_cast<uint64_t>(state.list_data_ptr_addr));
    auto chunk_list_data = reinterpret_cast<ChunkListData *>(
        static_cast<uint64_t>(*list_data_ptr) - sizeof(ChunkListData));
    auto *meta = &(chunk_list_data->meta);
    *meta |= (static_cast<uint64_t>(1) << ptr.idx);
  };

  using ChunkList =
      GenericLocalList<ChunkNodePtr, decltype(kDerefNodePtrFn),
                       decltype(kAllocateNodeFn), decltype(kFreeNodeFn)>;

#pragma pack(push, 1)
  struct LocalNode {
    GenericUniquePtr ptr;
    ChunkList chunk_list;
    uint8_t cnt = 0;
    bool swapping_in = false;
    uint8_t paddings[6];

    bool is_invalid() const;
  };
#pragma pack(pop)
  // Far-mem pointer should never cross the cacheline boundary.
  static_assert(sizeof(LocalNode) == 32);

  template <bool Reverse> class GenericIteratorImpl {
  private:
    uint8_t *insert(const DerefScope &scope);
    GenericIteratorImpl erase(const DerefScope &scope, uint8_t **data_ptr);

    LocalList<LocalNode>::IteratorImpl<Reverse> local_iter_;
    ChunkList::template IteratorImpl<Reverse> chunk_iter_;
    GenericList *list_;
    friend class GenericList;

  public:
    GenericIteratorImpl(
        const DerefScope &scope,
        const LocalList<LocalNode>::IteratorImpl<Reverse> &local_iter,
        GenericList *list);
    GenericIteratorImpl(const GenericIteratorImpl<Reverse> &o);
    GenericIteratorImpl &operator=(const GenericIteratorImpl<Reverse> &o);
    void inc(const DerefScope &scope);
    void dec(const DerefScope &scope);
    bool operator==(const GenericIteratorImpl &o);
    bool operator!=(const GenericIteratorImpl &o);
    const uint8_t *deref(const DerefScope &scope);
    uint8_t *deref_mut(const DerefScope &scope);
  };

  using GenericIterator = GenericIteratorImpl</* Reverse = */ false>;
  using ReverseGenericIterator = GenericIteratorImpl</* Reverse = */ true>;

  constexpr static uint16_t kMinNumNodesPerChunk = 8;
  constexpr static uint16_t kMaxNumNodesPerChunk = 64;
  constexpr static uint16_t kInvalidCnt = kMaxNumNodesPerChunk + 1;
  constexpr static uint16_t kDefaultChunkSize = 4096;
  constexpr static double kMergeThreshRatio = 0.75;

  const uint16_t kItemSize_;
  const uint16_t kNumNodesPerChunk_;
  const uint16_t kChunkListNodeSize_;
  const uint16_t kChunkSize_;
  const uint64_t kInitMeta_;
  const uint16_t kMergeThresh_;
  const uint32_t kPrefetchNumNodes_;
  LocalList<LocalNode> local_list_;
  uint64_t size_ = 0;
  bool enable_merge_;
  bool customized_split_; // Customized for Queue and Stack.
  bool enable_prefetch_ = false;
  bool prefetch_reversed_ = true;
  LocalList<LocalNode>::Iterator prefetch_iter_;

  template <typename T> friend class List;
  friend class FarMemTest;

  GenericList(const DerefScope &scope, const uint16_t kItemSize,
              const uint16_t kNumNodesPerChunk, bool enable_merge,
              bool customized_split = false);
  void init_local_node(const DerefScope &scope, LocalNode *local_node);
  template <bool Reverse>
  LocalList<LocalNode>::IteratorImpl<Reverse>
  add_local_list_node(const DerefScope &scope,
                      const LocalList<LocalNode>::IteratorImpl<Reverse> &iter);
  template <bool Reverse>
  GenericList::GenericIteratorImpl<Reverse>
  remove_local_list_node(const DerefScope &scope,
                         const GenericIteratorImpl<Reverse> &iter);
  template <bool Reverse>
  GenericIteratorImpl<Reverse>
  split_local_list_node(const DerefScope &scope,
                        const GenericIteratorImpl<Reverse> &iter);
  template <bool Reverse>
  void merge_local_list_node(
      const DerefScope &scope,
      const LocalList<LocalNode>::IteratorImpl<Reverse> &local_iter,
      const LocalList<LocalNode>::IteratorImpl<Reverse> &next_local_iter);

  template <bool Mut>
  static void update_chunk_list_addr(const DerefScope &scope,
                                     LocalNode *local_node);

  template <bool Reverse>
  void
  prefetch_fsm(const LocalList<LocalNode>::IteratorImpl<Reverse> &local_iter);
  void prefetch_once();
  void do_prefetch(LocalNode *local_node);

public:
  GenericIterator begin(const DerefScope &scope) const;
  GenericIterator end(const DerefScope &scope) const;
  ReverseGenericIterator rbegin(const DerefScope &scope) const;
  ReverseGenericIterator rend(const DerefScope &scope) const;
  const uint8_t *cfront(const DerefScope &scope) const;
  const uint8_t *cback(const DerefScope &scope) const;
  uint8_t *front(const DerefScope &scope) const;
  uint8_t *back(const DerefScope &scope) const;
  uint64_t size() const;
  bool empty() const;
  uint8_t *new_front(const DerefScope &scope);
  uint8_t *pop_front(const DerefScope &scope);
  uint8_t *new_back(const DerefScope &scope);
  uint8_t *pop_back(const DerefScope &scope);
  template <bool Reverse>
  uint8_t *insert(const DerefScope &scope, GenericIteratorImpl<Reverse> *iter);
  template <bool Reverse>
  GenericIteratorImpl<Reverse> erase(const DerefScope &scope,
                                     const GenericIteratorImpl<Reverse> &iter,
                                     uint8_t **data_ptr);
};

template <typename T> class List : public GenericList {
private:
  constexpr static uint16_t kNumNodesPerChunk =
      std::max(static_cast<uint16_t>(kMinNumNodesPerChunk),
               std::min(static_cast<uint16_t>(kMaxNumNodesPerChunk),
                        static_cast<uint16_t>(
                            (kDefaultChunkSize - sizeof(ChunkListData) -
                             sizeof(ChunkList::ListData)) /
                            (sizeof(T) + sizeof(ChunkList::Node)))));

  static_assert(kNumNodesPerChunk <= 8 * sizeof(ChunkListData::meta));

  template <bool Reverse>
  class IteratorImpl : public GenericIteratorImpl<Reverse> {
  private:
    IteratorImpl(const GenericIteratorImpl<Reverse> &generic_iter);
    friend class List;

  public:
    IteratorImpl(const IteratorImpl &o);
    IteratorImpl &operator=(const IteratorImpl &o);
    const T &deref(const DerefScope &scope);
    T &deref_mut(const DerefScope &scope);
  };

  using Iterator = IteratorImpl</* Reverse = */ false>;
  using ReverseIterator = IteratorImpl</* Reverse = */ true>;

  List(const DerefScope &scope, bool enable_merge,
       bool customized_split = false);

  template <typename D> friend class Queue;
  template <typename D> friend class Stack;
  friend class FarMemManager;
  friend class FarMemTest;

public:
  ~List();
  Iterator begin(const DerefScope &scope) const;
  Iterator end(const DerefScope &scope) const;
  ReverseIterator rbegin(const DerefScope &scope) const;
  ReverseIterator rend(const DerefScope &scope) const;
  const T &cfront(const DerefScope &scope) const;
  const T &cback(const DerefScope &scope) const;
  T &front(const DerefScope &scope) const;
  T &back(const DerefScope &scope) const;
  void push_front(const DerefScope &scope, const T &data);
  void push_back(const DerefScope &scope, const T &data);
  void pop_front(const DerefScope &scope);
  void pop_back(const DerefScope &scope);
  template <bool Reverse>
  void insert(const DerefScope &scope, IteratorImpl<Reverse> *iter,
              const T &data);
  template <bool Reverse>
  IteratorImpl<Reverse> erase(const DerefScope &scope,
                              const IteratorImpl<Reverse> &iter);
};

FORCE_INLINE bool GenericList::LocalNode::is_invalid() const {
  return cnt == kInvalidCnt;
}

FORCE_INLINE bool GenericList::ChunkNodePtr::
operator==(const ChunkNodePtr &o) const {
  return idx == o.idx && addr_offset == o.addr_offset;
}

FORCE_INLINE bool GenericList::ChunkNodePtr::
operator!=(const ChunkNodePtr &o) const {
  return idx != o.idx || addr_offset != o.addr_offset;
}

FORCE_INLINE GenericList::ChunkNodePtr::ChunkNodePtr() {}

FORCE_INLINE GenericList::ChunkNodePtr::ChunkNodePtr(const ChunkNodePtr &o)
    : idx(o.idx), addr_offset(o.addr_offset) {}

FORCE_INLINE constexpr GenericList::ChunkNodePtr::ChunkNodePtr(
    uint8_t _idx, uint16_t _addr_offset)
    : idx(_idx), addr_offset(_addr_offset) {}

template <bool Mut>
FORCE_INLINE void GenericList::update_chunk_list_addr(const DerefScope &scope,
                                                      LocalNode *local_node) {
  ChunkListData *chunk_list_data;
  if constexpr (Mut) {
    chunk_list_data =
        reinterpret_cast<ChunkListData *>(local_node->ptr.deref_mut(scope));
  } else {
    chunk_list_data = reinterpret_cast<ChunkListData *>(
        const_cast<void *>(local_node->ptr.deref(scope)));
  }
  local_node->chunk_list.set_list_data(
      reinterpret_cast<ChunkList::ListData *>(chunk_list_data->data));
}

FORCE_INLINE void GenericList::prefetch_once() {
  auto *local_node = &(*prefetch_iter_);
  if (unlikely(local_node->is_invalid())) {
    return;
  }
  do_prefetch(local_node);
  if (prefetch_reversed_) {
    --prefetch_iter_;
  } else {
    ++prefetch_iter_;
  }
}

FORCE_INLINE void GenericList::do_prefetch(LocalNode *local_node) {
  if (likely(!(local_node->swapping_in))) {
    local_node->swapping_in = true;
    rt::Thread([=] {
      local_node->ptr.swap_in(false);
      barrier();
      local_node->swapping_in = false;
    })
        .Detach();
  }
}

template <bool Reverse>
FORCE_INLINE void GenericList::prefetch_fsm(
    const LocalList<LocalNode>::IteratorImpl<Reverse> &local_iter) {
  if (Reverse == prefetch_reversed_) {
    if (!enable_prefetch_) {
      enable_prefetch_ = true;
      prefetch_iter_ = local_iter;
      for (uint32_t i = 0; i < kPrefetchNumNodes_; i++) {
        prefetch_once();
      }
    } else {
      prefetch_once();
    }
  } else {
    enable_prefetch_ = false;
    prefetch_reversed_ = Reverse;
  }
}

template <bool Reverse>
FORCE_INLINE GenericList::GenericIteratorImpl<Reverse>::GenericIteratorImpl(
    const DerefScope &scope,
    const LocalList<LocalNode>::IteratorImpl<Reverse> &local_iter,
    GenericList *list)
    : local_iter_(local_iter), list_(list) {
  update_chunk_list_addr</* Mut = */ false>(scope, &(*local_iter_));
  auto &chunk_list = local_iter_->chunk_list;
  if constexpr (Reverse) {
    chunk_iter_ = chunk_list.rbegin();
  } else {
    chunk_iter_ = chunk_list.begin();
  }
}

template <bool Reverse>
FORCE_INLINE GenericList::GenericIteratorImpl<Reverse>::GenericIteratorImpl(
    const GenericIteratorImpl<Reverse> &o)
    : local_iter_(o.local_iter_), chunk_iter_(o.chunk_iter_), list_(o.list_) {}

template <bool Reverse>
FORCE_INLINE GenericList::GenericIteratorImpl<Reverse> &
GenericList::GenericIteratorImpl<Reverse>::
operator=(const GenericIteratorImpl<Reverse> &o) {
  local_iter_ = o.local_iter_;
  chunk_iter_ = o.chunk_iter_;
  list_ = o.list_;
  return *this;
}

template <bool Reverse>
FORCE_INLINE void
GenericList::GenericIteratorImpl<Reverse>::inc(const DerefScope &scope) {
  chunk_iter_++;
  update_chunk_list_addr</* Mut = */ false>(scope, &(*local_iter_));
  decltype(chunk_iter_) end_iter;
  auto *chunk_list = &(local_iter_->chunk_list);
  if constexpr (Reverse) {
    end_iter = chunk_list->rend();
  } else {
    end_iter = chunk_list->end();
  }
  if (unlikely(chunk_iter_ == end_iter)) {
    ++local_iter_;
    list_->prefetch_fsm(local_iter_);
    chunk_list = &(local_iter_->chunk_list);
    update_chunk_list_addr</* Mut = */ false>(scope, &(*local_iter_));
    if constexpr (Reverse) {
      chunk_iter_ = chunk_list->rbegin();
    } else {
      chunk_iter_ = chunk_list->begin();
    }
  }
}

template <bool Reverse>
FORCE_INLINE void
GenericList::GenericIteratorImpl<Reverse>::dec(const DerefScope &scope) {
  update_chunk_list_addr</* Mut = */ false>(scope, &(*local_iter_));
  decltype(chunk_iter_) begin_iter;
  auto *chunk_list = &(local_iter_->chunk_list);
  if constexpr (Reverse) {
    begin_iter = chunk_list->rbegin();
  } else {
    begin_iter = chunk_list->begin();
  }
  if (unlikely(chunk_iter_ == begin_iter)) {
    --local_iter_;
    list_->prefetch_fsm(local_iter_);
    chunk_list = &(local_iter_->chunk_list);
    update_chunk_list_addr</* Mut = */ false>(scope, &(*local_iter_));
    if constexpr (Reverse) {
      chunk_iter_ = chunk_list->rend();
    } else {
      chunk_iter_ = chunk_list->end();
    }
  }
  chunk_iter_--;
}

template <bool Reverse>
FORCE_INLINE bool GenericList::GenericIteratorImpl<Reverse>::
operator==(const GenericIteratorImpl &o) {
  return this->local_iter_ == o.local_iter_ &&
         this->chunk_iter_ == o.chunk_iter_;
}

template <bool Reverse>
FORCE_INLINE bool GenericList::GenericIteratorImpl<Reverse>::
operator!=(const GenericIteratorImpl &o) {
  return this->local_iter_ != o.local_iter_ ||
         this->chunk_iter_ != o.chunk_iter_;
}

template <bool Reverse>
FORCE_INLINE const uint8_t *
GenericList::GenericIteratorImpl<Reverse>::deref(const DerefScope &scope) {
  update_chunk_list_addr</* Mut = */ false>(scope, &(*local_iter_));
  return *chunk_iter_;
}

template <bool Reverse>
FORCE_INLINE uint8_t *
GenericList::GenericIteratorImpl<Reverse>::deref_mut(const DerefScope &scope) {
  update_chunk_list_addr</* Mut = */ true>(scope, &(*local_iter_));
  return *chunk_iter_;
}

template <bool Reverse>
FORCE_INLINE LocalList<GenericList::LocalNode>::IteratorImpl<Reverse>
GenericList::add_local_list_node(
    const DerefScope &scope,
    const LocalList<LocalNode>::IteratorImpl<Reverse> &iter) {
  local_list_.insert(iter, LocalNode());
  auto new_iter = iter;
  --new_iter;
  init_local_node(scope, &(*new_iter));
  return new_iter;
}

template <bool Reverse>
FORCE_INLINE uint8_t *
GenericList::GenericIteratorImpl<Reverse>::insert(const DerefScope &scope) {
  update_chunk_list_addr</* Mut = */ true>(scope, &(*local_iter_));
  return local_iter_->chunk_list.insert(chunk_iter_);
}

template <bool Reverse>
FORCE_INLINE GenericList::template GenericIteratorImpl<Reverse>
GenericList::GenericIteratorImpl<Reverse>::erase(const DerefScope &scope,
                                                 uint8_t **data_ptr) {
  auto ret = *this;
  update_chunk_list_addr</* Mut = */ true>(scope, &(*local_iter_));
  ret.chunk_iter_ = local_iter_->chunk_list.erase(chunk_iter_, data_ptr);
  return ret;
}

template <bool Reverse>
FORCE_INLINE uint8_t *GenericList::insert(const DerefScope &scope,
                                          GenericIteratorImpl<Reverse> *iter) {
  size_++;
  auto *cnt = &(iter->local_iter_->cnt);
  if (unlikely(*cnt == kNumNodesPerChunk_)) {
    *iter = split_local_list_node(scope, *iter);
    cnt = &(iter->local_iter_->cnt);
  }
  (*cnt)++;
  return iter->insert(scope);
}

template <bool Reverse>
FORCE_INLINE GenericList::GenericIteratorImpl<Reverse>
GenericList::remove_local_list_node(const DerefScope &scope,
                                    const GenericIteratorImpl<Reverse> &iter) {
  while (unlikely(iter.local_iter_->swapping_in)) {
    thread_yield();
  }
  auto new_local_iter = local_list_.erase(iter.local_iter_);
  return GenericIteratorImpl(scope, new_local_iter, this);
}

template <bool Reverse>
FORCE_INLINE GenericList::GenericIteratorImpl<Reverse>
GenericList::erase(const DerefScope &scope,
                   const GenericIteratorImpl<Reverse> &iter,
                   uint8_t **data_ptr) {
  size_--;
  auto &iter_mut = *const_cast<GenericIteratorImpl<Reverse> *>(&iter);
  auto ret = iter_mut.erase(scope, data_ptr);
  auto cnt = --iter.local_iter_->cnt;
  if (unlikely(cnt == 0)) {
    auto next_local_iter = iter_mut.local_iter_;
    ++next_local_iter;
    prefetch_fsm(next_local_iter);
    return remove_local_list_node(scope, iter_mut);
  } else {
    if (enable_merge_) {
      auto next_local_iter = iter_mut.local_iter_;
      ++next_local_iter;
      prefetch_fsm(next_local_iter);
      if (unlikely(cnt + next_local_iter->cnt <= kMergeThresh_)) {
        merge_local_list_node(scope, iter_mut.local_iter_, next_local_iter);
      }
    }
    return ret;
  }
}

FORCE_INLINE GenericList::GenericIterator
GenericList::begin(const DerefScope &scope) const {
  return GenericIterator(scope, ++(local_list_.begin()),
                         const_cast<GenericList *>(this));
}

FORCE_INLINE GenericList::GenericIterator
GenericList::end(const DerefScope &scope) const {
  return GenericIterator(scope, --(local_list_.end()),
                         const_cast<GenericList *>(this));
}

FORCE_INLINE GenericList::ReverseGenericIterator
GenericList::rbegin(const DerefScope &scope) const {
  return ReverseGenericIterator(scope, ++(local_list_.rbegin()),
                                const_cast<GenericList *>(this));
}

FORCE_INLINE GenericList::ReverseGenericIterator
GenericList::rend(const DerefScope &scope) const {
  return ReverseGenericIterator(scope, --(local_list_.rend()),
                                const_cast<GenericList *>(this));
}

FORCE_INLINE const uint8_t *GenericList::cfront(const DerefScope &scope) const {
  return begin(scope).deref(scope);
}

FORCE_INLINE const uint8_t *GenericList::cback(const DerefScope &scope) const {
  return rbegin(scope).deref(scope);
}

FORCE_INLINE uint8_t *GenericList::front(const DerefScope &scope) const {
  return begin(scope).deref_mut(scope);
}

FORCE_INLINE uint8_t *GenericList::back(const DerefScope &scope) const {
  return rbegin(scope).deref_mut(scope);
}

FORCE_INLINE uint64_t GenericList::size() const { return size_; }

FORCE_INLINE bool GenericList::empty() const { return size_ == 0; }

FORCE_INLINE uint8_t *GenericList::pop_front(const DerefScope &scope) {
  uint8_t *data_ptr;
  erase(scope, begin(scope), &data_ptr);
  return data_ptr;
}

FORCE_INLINE uint8_t *GenericList::pop_back(const DerefScope &scope) {
  uint8_t *data_ptr;
  erase(scope, rbegin(scope), &data_ptr);
  return data_ptr;
}

FORCE_INLINE uint8_t *GenericList::new_front(const DerefScope &scope) {
  if (unlikely(empty())) {
    add_local_list_node(scope, --local_list_.end());
  }
  auto iter = begin(scope);
  return insert(scope, &iter);
}

FORCE_INLINE uint8_t *GenericList::new_back(const DerefScope &scope) {
  if (unlikely(empty())) {
    add_local_list_node(scope, --local_list_.end());
  }
  auto iter = rbegin(scope);
  return insert(scope, &iter);
}

template <bool Reverse>
FORCE_INLINE GenericList::GenericIteratorImpl<Reverse>
GenericList::split_local_list_node(const DerefScope &scope,
                                   const GenericIteratorImpl<Reverse> &iter) {
  GenericIteratorImpl<Reverse> ret = iter;
  auto new_local_iter = add_local_list_node(scope, iter.local_iter_);
  auto &cur_chunk_list = iter.local_iter_->chunk_list;
  auto &new_chunk_list = new_local_iter->chunk_list;
  decltype(iter.chunk_iter_) cur_chunk_iter;
  decltype(iter.chunk_iter_) new_chunk_iter;
  if constexpr (Reverse) {
    cur_chunk_iter = cur_chunk_list.rbegin();
    new_chunk_iter = new_chunk_list.rend();
  } else {
    cur_chunk_iter = cur_chunk_list.begin();
    new_chunk_iter = new_chunk_list.end();
  }
  if (customized_split_) {
    // Optimized for Queue and Stack since they only push elements to the back.
    // In this case, we add a new block and avoid spliting (i.e., copying data).
    ret.local_iter_ = new_local_iter;
    ret.chunk_iter_ = new_chunk_iter;
  } else {
    for (uint32_t i = 0; i < kNumNodesPerChunk_ / 2; i++) {
      uint8_t *data_ptr;
      bool matched = (cur_chunk_iter == iter.chunk_iter_);
      cur_chunk_iter = cur_chunk_list.erase(cur_chunk_iter, &data_ptr);
      auto new_data_ptr = new_chunk_list.insert(new_chunk_iter);
      memcpy(new_data_ptr, data_ptr, kItemSize_);
      if (unlikely(matched)) {
        ret.local_iter_ = new_local_iter;
        ret.chunk_iter_ = new_chunk_iter;
        --ret.chunk_iter_;
      }
    }
    new_local_iter->cnt = kNumNodesPerChunk_ / 2;
    iter.local_iter_->cnt = kNumNodesPerChunk_ - kNumNodesPerChunk_ / 2;
  }
  return ret;
}

template <bool Reverse>
FORCE_INLINE void GenericList::merge_local_list_node(
    const DerefScope &scope,
    const LocalList<LocalNode>::IteratorImpl<Reverse> &local_iter,
    const LocalList<LocalNode>::IteratorImpl<Reverse> &next_local_iter) {
  update_chunk_list_addr</* Mut = */ true>(scope, &(*next_local_iter));
  auto next_cnt = next_local_iter->cnt;
  local_iter->cnt += next_cnt;
  auto &local_chunk_list = local_iter->chunk_list;
  auto &next_chunk_list = next_local_iter->chunk_list;
  ChunkList::IteratorImpl<Reverse> chunk_iter;
  ChunkList::IteratorImpl<Reverse> next_chunk_iter;
  if constexpr (Reverse) {
    chunk_iter = local_chunk_list.rend();
    next_chunk_iter = next_chunk_list.rbegin();
  } else {
    chunk_iter = local_chunk_list.end();
    next_chunk_iter = next_chunk_list.begin();
  }
  for (uint32_t i = 0; i < next_cnt; i++) {
    uint8_t *data_ptr;
    next_chunk_iter = next_chunk_list.erase(next_chunk_iter, &data_ptr);
    auto *new_data_ptr = local_chunk_list.insert(chunk_iter);
    memcpy(new_data_ptr, data_ptr, kItemSize_);
  }
  while (unlikely(next_local_iter->swapping_in)) {
    thread_yield();
  }
  local_list_.erase(next_local_iter);
}

template <typename T>
template <bool Reverse>
FORCE_INLINE List<T>::IteratorImpl<Reverse>::IteratorImpl(const IteratorImpl &o)
    : GenericIteratorImpl<Reverse>(o) {}

template <typename T>
template <bool Reverse>
FORCE_INLINE List<T>::template IteratorImpl<Reverse> &
List<T>::IteratorImpl<Reverse>::operator=(const IteratorImpl &o) {
  GenericIteratorImpl<Reverse>::operator=(o);
  return *this;
}

template <typename T>
template <bool Reverse>
FORCE_INLINE List<T>::IteratorImpl<Reverse>::IteratorImpl(
    const GenericIteratorImpl<Reverse> &generic_iter)
    : GenericIteratorImpl<Reverse>(generic_iter) {}

template <typename T>
template <bool Reverse>
FORCE_INLINE const T &
List<T>::IteratorImpl<Reverse>::deref(const DerefScope &scope) {
  return *reinterpret_cast<const T *>(
      GenericIteratorImpl<Reverse>::deref(scope));
}

template <typename T>
template <bool Reverse>
FORCE_INLINE T &
List<T>::IteratorImpl<Reverse>::deref_mut(const DerefScope &scope) {
  return *reinterpret_cast<T *>(GenericIteratorImpl<Reverse>::deref_mut(scope));
}

template <typename T> FORCE_INLINE List<T>::~List() {
  bool in_scope = DerefScope::is_in_deref_scope();
  if (!in_scope) {
    DerefScope::enter_deref_scope();
  }
  uint8_t idx = 0;
  while (!empty()) {
    if (unlikely(++idx == 0)) {
      DerefScope::exit_deref_scope();
      DerefScope::enter_deref_scope();
    }
    pop_back(*static_cast<DerefScope *>(nullptr));
  }
  if (!in_scope) {
    DerefScope::exit_deref_scope();
  }
}

template <typename T>
FORCE_INLINE List<T>::List(const DerefScope &scope, bool enable_merge,
                           bool customized_split)
    : GenericList(scope, sizeof(T), kNumNodesPerChunk, enable_merge,
                  customized_split) {}

template <typename T>
FORCE_INLINE List<T>::Iterator List<T>::begin(const DerefScope &scope) const {
  return Iterator(GenericList::begin(scope));
}

template <typename T>
FORCE_INLINE List<T>::Iterator List<T>::end(const DerefScope &scope) const {
  return Iterator(GenericList::end(scope));
}

template <typename T>
FORCE_INLINE List<T>::ReverseIterator
List<T>::rbegin(const DerefScope &scope) const {
  return ReverseIterator(GenericList::rbegin(scope));
}

template <typename T>
FORCE_INLINE List<T>::ReverseIterator
List<T>::rend(const DerefScope &scope) const {
  return ReverseIterator(GenericList::rend(scope));
}

template <typename T>
FORCE_INLINE const T &List<T>::cfront(const DerefScope &scope) const {
  return *reinterpret_cast<const T *>(GenericList::cfront(scope));
}

template <typename T>
FORCE_INLINE T &List<T>::front(const DerefScope &scope) const {
  return *reinterpret_cast<T *>(GenericList::front(scope));
}

template <typename T>
FORCE_INLINE const T &List<T>::cback(const DerefScope &scope) const {
  return *reinterpret_cast<const T *>(GenericList::cback(scope));
}

template <typename T>
FORCE_INLINE T &List<T>::back(const DerefScope &scope) const {
  return *reinterpret_cast<T *>(GenericList::back(scope));
}

template <typename T>
FORCE_INLINE void List<T>::push_front(const DerefScope &scope, const T &data) {
  auto *new_data_ptr = GenericList::new_front(scope);
  memcpy(new_data_ptr, &data, sizeof(data));
}

template <typename T>
FORCE_INLINE void List<T>::push_back(const DerefScope &scope, const T &data) {
  auto *new_data_ptr = GenericList::new_back(scope);
  memcpy(new_data_ptr, &data, sizeof(data));
}

template <typename T>
FORCE_INLINE void List<T>::pop_front(const DerefScope &scope) {
  auto data_ptr = GenericList::pop_front(scope);
  reinterpret_cast<T *>(data_ptr)->~T();
}

template <typename T>
FORCE_INLINE void List<T>::pop_back(const DerefScope &scope) {
  auto data_ptr = GenericList::pop_back(scope);
  reinterpret_cast<T *>(data_ptr)->~T();
}

template <typename T>
template <bool Reverse>
FORCE_INLINE void List<T>::insert(const DerefScope &scope,
                                  IteratorImpl<Reverse> *iter, const T &data) {
  auto *new_data_ptr = GenericList::insert(scope, iter);
  memcpy(new_data_ptr, &data, sizeof(data));
}

template <typename T>
template <bool Reverse>
FORCE_INLINE List<T>::template IteratorImpl<Reverse>
List<T>::erase(const DerefScope &scope, const IteratorImpl<Reverse> &iter) {
  uint8_t *data_ptr;
  auto ret = IteratorImpl<Reverse>(GenericList::erase(scope, iter, &data_ptr));
  reinterpret_cast<T *>(data_ptr)->~T();
  return ret;
}

} // namespace far_memory
