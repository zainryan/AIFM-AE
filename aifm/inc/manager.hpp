#pragma once

#include "sync.h"
#include "thread.h"

#include "array.hpp"
#include "cb.hpp"
#include "concurrent_hopscotch.hpp"
#include "dataframe_vector.hpp"
#include "device.hpp"
#include "ds_info.hpp"
#include "list.hpp"
#include "obj_locker.hpp"
#include "parallel.hpp"
#include "pointer.hpp"
#include "queue.hpp"
#include "region.hpp"
#include "stack.hpp"
#include "stats.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <queue>
#include <utility>
#include <vector>

namespace far_memory {
extern bool gc_master_active;

// A GCTask is an interval of (to be GCed) local region.
using GCTask = std::pair<uint64_t, uint64_t>;

class GCParallelizer : public Parallelizer<GCTask> {
public:
  std::vector<Region> *from_regions_;
  GCParallelizer(uint32_t num_slaves, uint32_t task_queues_depth,
                 std::vector<Region> *from_regions);
  void master_fn();
};

class GCParallelMarker : public GCParallelizer {
  void slave_fn(uint32_t tid);

public:
  GCParallelMarker(uint32_t num_slaves, uint32_t task_queues_depth,
                   std::vector<Region> *from_regions);
};

class GCParallelWriteBacker : public GCParallelizer {
  void slave_fn(uint32_t tid);

public:
  GCParallelWriteBacker(uint32_t num_slaves, uint32_t task_queues_depth,
                        std::vector<Region> *from_regions);
};

class FarMemManager {
private:
  constexpr static double kFreeCacheAlmostEmptyThresh = 0.03;
  constexpr static double kFreeCacheLowThresh = 0.12;
  constexpr static double kFreeCacheHighThresh = 0.22;
  constexpr static uint8_t kGCSlaveThreadTaskQueueDepth = 8;
  constexpr static uint32_t kMaxNumRegionsPerGCRound = 128;
  constexpr static double kMaxRatioRegionsPerGCRound = 0.1;
  constexpr static double kMinRatioRegionsPerGCRound = 0.03;

  class RegionManager {
  private:
    constexpr static double kPickRegionMaxRetryTimes = 3;

    std::unique_ptr<uint8_t> local_cache_ptr_;
    CircularBuffer<Region, false> free_regions_;
    CircularBuffer<Region, false> used_regions_;
    CircularBuffer<Region, false> nt_used_regions_;
    rt::Spin region_spin_;
    Region core_local_free_regions_[helpers::kNumCPUs];
    Region core_local_free_nt_regions_[helpers::kNumCPUs];
    friend class FarMemTest;

  public:
    RegionManager(uint64_t size, bool is_local);
    void push_free_region(Region &region);
    std::optional<Region> pop_used_region();
    bool try_refill_core_local_free_region(bool nt, Region *full_region);
    Region &core_local_free_region(bool nt);
    double get_free_region_ratio() const;
    uint32_t get_num_regions() const;
  };

  RegionManager cache_region_manager_;
  RegionManager far_mem_region_manager_;
  std::atomic<uint32_t> pending_gcs_{0};
  bool gc_master_spawned_;
  std::unique_ptr<FarMemDevice> device_ptr_;
  rt::CondVar mutator_cache_condvar_;
  rt::CondVar mutator_far_mem_condvar_;
  rt::Spin gc_lock_;
  GCParallelMarker parallel_marker_;
  GCParallelWriteBacker parallel_write_backer_;
  std::vector<Region> from_regions_{kMaxNumRegionsPerGCRound};
  int ksched_fd_;
  std::queue<uint8_t> available_ds_ids_;
  static ObjLocker obj_locker_;

  friend class FarMemTest;
  friend class FarMemManagerFactory;
  friend class GenericUniquePtr;
  friend class FarMemPtrMeta;
  friend class GenericArray;
  friend class GCParallelWriteBacker;
  friend class DerefScope;

  FarMemManager(uint64_t cache_size, uint64_t far_mem_size,
                uint32_t num_gc_threads, FarMemDevice *device);
  bool is_free_cache_almost_empty() const;
  bool is_free_cache_low() const;
  bool is_free_cache_high() const;
  std::optional<Region> pop_cache_used_region();
  void push_cache_free_region(Region &region);
  void swap_in(bool nt, FarMemPtrMeta *meta);
  void swap_out(FarMemPtrMeta *meta, Object obj);
  void launch_gc_master();
  void gc_cache();
  void gc_far_mem();
  uint64_t allocate_local_object(bool nt, uint16_t object_size);
  std::optional<uint64_t> allocate_local_object_nb(bool nt,
                                                   uint16_t object_size);
  uint64_t allocate_remote_object(bool nt, uint16_t object_size);
  void mutator_wait_for_gc_far_mem();
  void pick_from_regions();
  void mark_fm_ptrs(auto *preempt_guard);
  void wait_mutators_observation();
  void write_back_regions();
  void gc_check();
  void start_prioritizing(Status status);
  void stop_prioritizing();
  uint8_t allocate_ds_id();
  void free_ds_id(uint8_t ds_id);

public:
  using Notifier = std::function<void(Object)>;

  uint32_t num_gc_threads_;
  Notifier notifiers_[kMaxNumDSIDs];

  ~FarMemManager();
  FarMemDevice *get_device() const { return device_ptr_.get(); }
  double get_free_mem_ratio() const;
  bool allocate_generic_unique_ptr_nb(
      GenericUniquePtr *ptr, uint8_t ds_id, uint16_t item_size,
      std::optional<uint8_t> optional_id_len = {},
      std::optional<const uint8_t *> optional_id = {});
  GenericUniquePtr
  allocate_generic_unique_ptr(uint8_t ds_id, uint16_t item_size,
                              std::optional<uint8_t> optional_id_len = {},
                              std::optional<const uint8_t *> optional_id = {});
  bool reallocate_generic_unique_ptr_nb(const DerefScope &scope,
                                        GenericUniquePtr *,
                                        uint16_t new_item_size,
                                        const uint8_t *data_buf);
  template <typename T> UniquePtr<T> allocate_unique_ptr();
  template <typename T> UniquePtr<T> allocate_unique_ptr(const T &t);
  template <typename T, uint64_t... Dims> Array<T, Dims...> allocate_array();
  template <typename T, uint64_t... Dims>
  Array<T, Dims...> *allocate_array_heap();
  GenericConcurrentHopscotch
  allocate_concurrent_hopscotch(uint32_t local_num_entries_shift,
                                uint32_t remote_num_entries_shift,
                                uint64_t remote_data_size);
  GenericConcurrentHopscotch *
  allocate_concurrent_hopscotch_heap(uint32_t local_num_entries_shift,
                                     uint32_t remote_num_entries_shift,
                                     uint64_t remote_data_size);
  template <typename T> DataFrameVector<T> allocate_dataframe_vector();
  template <typename T> DataFrameVector<T> *allocate_dataframe_vector_heap();
  template <typename T>
  List<T> allocate_list(const DerefScope &scope, bool enable_merge = false);
  template <typename T> Queue<T> allocate_queue(const DerefScope &scope);
  template <typename T> Stack<T> allocate_stack(const DerefScope &scope);
  void register_notifier(uint8_t ds_id, Notifier notifier);
  void read_object(uint8_t ds_id, uint8_t obj_id_len, const uint8_t *obj_id,
                   uint16_t *data_len, uint8_t *data_buf);
  bool remove_object(uint64_t ds_id, uint8_t obj_id_len, const uint8_t *obj_id);
  void construct(uint8_t ds_type, uint8_t ds_id, uint32_t param_len,
                 uint8_t *params);
  void deconstruct(uint8_t ds_id);
  void mutator_wait_for_gc_cache();
  static void lock_object(uint8_t obj_id_len, const uint8_t *obj_id);
  static void unlock_object(uint8_t obj_id_len, const uint8_t *obj_id);
};

class FarMemManagerFactory {
private:
  constexpr static uint32_t kDefaultNumGCThreads = 10;

  static FarMemManager *ptr_;
  friend class FarMemManager;

public:
  static FarMemManager *build(uint64_t cache_size,
                              std::optional<uint32_t> optional_num_gc_threads,
                              FarMemDevice *device);
  static FarMemManager *get();
};

FORCE_INLINE Region &
FarMemManager::RegionManager::core_local_free_region(bool nt) {
  assert(!preempt_enabled());
  auto core_num = get_core_num();
  return nt ? core_local_free_nt_regions_[core_num]
            : core_local_free_regions_[core_num];
}

FORCE_INLINE double
FarMemManager::RegionManager::get_free_region_ratio() const {
  return static_cast<double>(free_regions_.size()) / get_num_regions();
}

FORCE_INLINE uint32_t FarMemManager::RegionManager::get_num_regions() const {
  return free_regions_.capacity();
}

FORCE_INLINE double FarMemManager::get_free_mem_ratio() const {
  return cache_region_manager_.get_free_region_ratio();
}

FORCE_INLINE bool FarMemManager::is_free_cache_low() const {
  return get_free_mem_ratio() <= kFreeCacheLowThresh;
}

FORCE_INLINE bool FarMemManager::is_free_cache_almost_empty() const {
  return get_free_mem_ratio() <= kFreeCacheAlmostEmptyThresh;
}

FORCE_INLINE bool FarMemManager::is_free_cache_high() const {
  return get_free_mem_ratio() >= kFreeCacheHighThresh;
}

FORCE_INLINE void FarMemManager::push_cache_free_region(Region &region) {
  cache_region_manager_.push_free_region(region);
}

FORCE_INLINE std::optional<Region> FarMemManager::pop_cache_used_region() {
  return cache_region_manager_.pop_used_region();
}

template <typename T>
FORCE_INLINE UniquePtr<T> FarMemManager::allocate_unique_ptr() {
  static_assert(sizeof(T) <= Object::kMaxObjectDataSize);
  auto object_size = Object::kHeaderSize + sizeof(T) + kVanillaPtrObjectIDSize;
  auto local_object_addr = allocate_local_object(false, object_size);
  auto remote_object_addr = allocate_remote_object(false, object_size);
  Object(local_object_addr, kVanillaPtrDSID, static_cast<uint16_t>(sizeof(T)),
         static_cast<uint8_t>(sizeof(remote_object_addr)),
         reinterpret_cast<const uint8_t *>(&remote_object_addr));
  auto ptr = UniquePtr<T>(local_object_addr);
  Region::atomic_inc_ref_cnt(local_object_addr, -1);
  return ptr;
}

template <typename T>
FORCE_INLINE UniquePtr<T> FarMemManager::allocate_unique_ptr(const T &t) {
  auto ptr = allocate_unique_ptr<T>();
  memcpy(ptr.get_object().get_data_addr(), &t, sizeof(T));
  return ptr;
}

template <typename T, uint64_t... Dims>
FORCE_INLINE Array<T, Dims...> FarMemManager::allocate_array() {
  return Array<T, Dims...>(this);
}

template <typename T, uint64_t... Dims>
FORCE_INLINE Array<T, Dims...> *FarMemManager::allocate_array_heap() {
  return new Array<T, Dims...>(this);
}

template <typename T>
FORCE_INLINE List<T> FarMemManager::allocate_list(const DerefScope &scope,
                                                  bool enable_merge) {
  return List<T>(scope, enable_merge);
}

template <typename T>
FORCE_INLINE Queue<T> FarMemManager::allocate_queue(const DerefScope &scope) {
  return Queue<T>(scope);
}

template <typename T>
FORCE_INLINE Stack<T> FarMemManager::allocate_stack(const DerefScope &scope) {
  return Stack<T>(scope);
}

template <typename T>
FORCE_INLINE DataFrameVector<T> FarMemManager::allocate_dataframe_vector() {
  return DataFrameVector<T>(allocate_ds_id());
}

template <typename T>
FORCE_INLINE DataFrameVector<T> *
FarMemManager::allocate_dataframe_vector_heap() {
  return new DataFrameVector<T>(allocate_ds_id());
}

FORCE_INLINE FarMemManager *FarMemManagerFactory::get() { return ptr_; }

FORCE_INLINE void FarMemManager::register_notifier(uint8_t ds_id,
                                                   Notifier notifier) {
  notifiers_[ds_id] = notifier;
}

FORCE_INLINE void FarMemManager::read_object(uint8_t ds_id, uint8_t obj_id_len,
                                             const uint8_t *obj_id,
                                             uint16_t *data_len,
                                             uint8_t *data_buf) {
  device_ptr_->read_object(ds_id, obj_id_len, obj_id, data_len, data_buf);
}

FORCE_INLINE bool FarMemManager::remove_object(uint64_t ds_id,
                                               uint8_t obj_id_len,
                                               const uint8_t *obj_id) {
  return device_ptr_->remove_object(ds_id, obj_id_len, obj_id);
}

FORCE_INLINE void FarMemManager::construct(uint8_t ds_type, uint8_t ds_id,
                                           uint32_t param_len,
                                           uint8_t *params) {
  device_ptr_->construct(ds_type, ds_id, param_len, params);
}

FORCE_INLINE void FarMemManager::deconstruct(uint8_t ds_id) {
  free_ds_id(ds_id);
  device_ptr_->deconstruct(ds_id);
}

FORCE_INLINE uint64_t get_obj_id_fragment(uint8_t obj_id_len,
                                          const uint8_t *obj_id) {
  uint64_t obj_id_fragment;
  if (likely(obj_id_len >= 8)) {
    obj_id_fragment = *reinterpret_cast<const uint64_t *>(obj_id);
  } else {
    obj_id_fragment = 0;
    for (uint32_t i = 0; i < obj_id_len; i++) {
      *(reinterpret_cast<uint8_t *>(&obj_id_fragment) + i) = obj_id[i];
    }
  }
  return obj_id_fragment;
}

FORCE_INLINE void FarMemManager::lock_object(uint8_t obj_id_len,
                                             const uint8_t *obj_id) {
  // So far we only use at most 8 bytes of obj_id in locker.
  auto obj_id_fragment = get_obj_id_fragment(obj_id_len, obj_id);
  while (!obj_locker_.try_insert(obj_id_fragment))
    ;
}

FORCE_INLINE void FarMemManager::unlock_object(uint8_t obj_id_len,
                                               const uint8_t *obj_id) {
  auto obj_id_fragment = get_obj_id_fragment(obj_id_len, obj_id);
  obj_locker_.remove(obj_id_fragment);
}

FORCE_INLINE void FarMemManager::gc_check() {
  if (unlikely(is_free_cache_low())) {
    Stats::add_free_mem_ratio_record();
    ACCESS_ONCE(almost_empty) = is_free_cache_almost_empty();
#ifndef STW_GC
    launch_gc_master();
#endif
  }
}

} // namespace far_memory
