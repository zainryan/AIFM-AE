#pragma once

#include "sync.h"

#include "cb.hpp"
#include "deref_scope.hpp"
#include "hash.hpp"
#include "helpers.hpp"
#include "pointer.hpp"
#include "shared_pool.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>

namespace far_memory {

class GenericConcurrentHopscotch {
private:
  struct BucketEntry {
    constexpr static uint64_t kBusyPtr = FarMemPtrMeta::kNull + 1;

    uint32_t bitmap;
    rt::Spin spin;
    uint64_t timestamp;
    GenericUniquePtr ptr;

    BucketEntry();
  };
  static_assert(sizeof(BucketEntry) == 24);

#pragma pack(push, 1)
  struct NotifierMeta {
    uint64_t anchor_addr : 48;
    uint8_t offset;
  };
#pragma pack(pop)
  static_assert(sizeof(NotifierMeta) == 7);

  constexpr static uint32_t kNeighborhood = 32;
  constexpr static uint32_t kMaxRetries = 2;
  constexpr static uint32_t kNotifierStashSize = 1024;

  const uint32_t kHashMask_;
  const uint32_t kNumEntries_;
  std::unique_ptr<uint8_t> buckets_mem_;
  BucketEntry *buckets_;
  uint8_t ds_id_;
  CircularBuffer<NotifierMeta, /* Sync = */ true, kNotifierStashSize>
      notifier_stash_;

  friend class FarMemTest;
  friend class FarMemManager;

  GenericConcurrentHopscotch(uint8_t ds_id, uint32_t local_num_entries_shift,
                             uint32_t remote_num_entries_shift,
                             uint64_t remote_data_size);
  bool __get(uint8_t key_len, const uint8_t *key, uint16_t *val_len,
             uint8_t *val);
  void forward_get(uint8_t key_len, const uint8_t *key, uint16_t *val_len,
                   uint8_t *val);
  void _get(const DerefScope &scope, uint8_t key_len, const uint8_t *key,
            uint16_t *val_len, uint8_t *val, bool *forwarded);
  bool _put(uint8_t key_len, const uint8_t *key, uint16_t val_len,
            const uint8_t *val, bool remove_remote);
  bool _remove(uint8_t key_len, const uint8_t *key);
  void process_notifier_stash();
  void do_notifier(NotifierMeta meta);
  void notifier(Object object);

public:
  constexpr static uint32_t kMetadataSize = sizeof(NotifierMeta);

  ~GenericConcurrentHopscotch();
  void get(const DerefScope &scope, uint8_t key_len, const uint8_t *key,
           uint16_t *val_len, uint8_t *val);
  void get_tp(uint8_t key_len, const uint8_t *key, uint16_t *val_len,
              uint8_t *val);
  bool put(const DerefScope &scope, uint8_t key_len, const uint8_t *key,
           uint16_t val_len, const uint8_t *val);
  bool put_tp(uint8_t key_len, const uint8_t *key, uint16_t val_len,
              const uint8_t *val);
  bool remove(const DerefScope &scope, uint8_t key_len, const uint8_t *key);
  bool remove_tp(uint8_t key_len, const uint8_t *key);
};

template <typename K, typename V>
class ConcurrentHopscotch : public GenericConcurrentHopscotch {
private:
  CachelineAligned(int64_t) per_core_size_[helpers::kNumCPUs];

  std::optional<V> _find(const K &key);
  void _insert(const K &key, const V &value);
  bool _erase(const K &key);

public:
  ConcurrentHopscotch(uint32_t index_num_kv);
  bool empty() const;
  uint64_t size() const;
  std::optional<V> find(const DerefScope &scope, const K &key);
  std::optional<V> find_tp(const K &key);
  void insert(const DerefScope &scope, const K &key, const V &value);
  void insert_tp(const K &key, const V &value);
  bool erase(const DerefScope &scope, const K &key);
  bool erase_tp(const K &key);
};

FORCE_INLINE GenericConcurrentHopscotch::BucketEntry::BucketEntry() {
  bitmap = timestamp = 0;
  ptr.nullify();
}

FORCE_INLINE void
GenericConcurrentHopscotch::_get(const DerefScope &scope, uint8_t key_len,
                                 const uint8_t *key, uint16_t *val_len,
                                 uint8_t *val, bool *forwarded) {
  bool miss = __get(key_len, key, val_len, val);
  if (very_unlikely(miss)) {

    if (forwarded) {
      *forwarded = true;
    }
    forward_get(key_len, key, val_len, val);
  }
}

FORCE_INLINE void GenericConcurrentHopscotch::get(const DerefScope &scope,
                                                  uint8_t key_len,
                                                  const uint8_t *key,
                                                  uint16_t *val_len,
                                                  uint8_t *val) {
  _get(scope, key_len, key, val_len, val, nullptr);
}

FORCE_INLINE void GenericConcurrentHopscotch::get_tp(uint8_t key_len,
                                                     const uint8_t *key,
                                                     uint16_t *val_len,
                                                     uint8_t *val) {
  DerefScope scope;
  get(scope, key_len, key, val_len, val);
}

FORCE_INLINE bool GenericConcurrentHopscotch::put(const DerefScope &scope,
                                                  uint8_t key_len,
                                                  const uint8_t *key,
                                                  uint16_t val_len,
                                                  const uint8_t *val) {
  return _put(key_len, key, val_len, val, true);
}

FORCE_INLINE bool GenericConcurrentHopscotch::put_tp(uint8_t key_len,
                                                     const uint8_t *key,
                                                     uint16_t val_len,
                                                     const uint8_t *val) {
  DerefScope scope;
  return put(scope, key_len, key, val_len, val);
}

FORCE_INLINE bool GenericConcurrentHopscotch::remove(const DerefScope &scope,
                                                     uint8_t key_len,
                                                     const uint8_t *key) {
  return _remove(key_len, key);
}

FORCE_INLINE bool GenericConcurrentHopscotch::remove_tp(uint8_t key_len,
                                                        const uint8_t *key) {
  DerefScope scope;
  return remove(scope, key_len, key);
}

FORCE_INLINE void GenericConcurrentHopscotch::process_notifier_stash() {
  if (unlikely(notifier_stash_.size())) {
    NotifierMeta meta;
    while (notifier_stash_.pop_front(&meta)) {
      do_notifier(meta);
    }
  }
}

FORCE_INLINE void GenericConcurrentHopscotch::notifier(Object object) {
  process_notifier_stash();
  auto *meta = reinterpret_cast<const NotifierMeta *>(object.get_obj_id() -
                                                      sizeof(NotifierMeta));
  do_notifier(*meta);
}

FORCE_INLINE bool GenericConcurrentHopscotch::__get(uint8_t key_len,
                                                    const uint8_t *key,
                                                    uint16_t *val_len,
                                                    uint8_t *val) {
  uint32_t hash = hash_32(reinterpret_cast<const void *>(key), key_len);
  uint32_t bucket_idx = hash & kHashMask_;
  auto *bucket = buckets_ + bucket_idx;
  uint64_t timestamp;
  uint32_t retry_counter = 0;

  auto get_once = [&]<bool Lock>() -> bool {
    retry:
      if constexpr (Lock) {
        while (unlikely(!bucket->spin.TryLockWp())) {
          thread_yield();
        }
      }
      auto spin_guard = helpers::finally([&]() {
        if constexpr (Lock) {
          bucket->spin.UnlockWp();
        }
      });
      timestamp = load_acquire(&(bucket->timestamp));
      uint32_t bitmap = bucket->bitmap;
      while (bitmap) {
        auto offset = helpers::bsf_32(bitmap);
        auto &ptr = buckets_[bucket_idx + offset].ptr;
        if (likely(!ptr.is_null())) {
          auto *obj_val_ptr = ptr._deref<false, false>();
          if (unlikely(!obj_val_ptr)) {
            spin_guard.reset();
            process_notifier_stash();
            thread_yield();
            goto retry;
          }
          auto obj = Object(reinterpret_cast<uint64_t>(obj_val_ptr) -
                            Object::kHeaderSize);
          if (obj.get_obj_id_len() == key_len) {
            auto obj_data_len = obj.get_data_len();
            if (strncmp(reinterpret_cast<const char *>(obj_val_ptr) +
                            obj_data_len,
                        reinterpret_cast<const char *>(key), key_len) == 0) {
              *val_len = obj_data_len - sizeof(NotifierMeta);
              memcpy(val, obj_val_ptr, *val_len);
              return true;
            }
          }
        }
        bitmap ^= (1 << offset);
      }
      return false;
  };

  // Fast path.
  do {
    if (get_once.template operator()<false>()) {
      return false;
    }
  } while (timestamp != ACCESS_ONCE(bucket->timestamp) &&
           retry_counter++ < kMaxRetries);

  // Slow path.
  if (timestamp != ACCESS_ONCE(bucket->timestamp)) {
    if (get_once.template operator()<true>()) {
      return false;
    }
  }
  return true;
}

template <typename K, typename V>
FORCE_INLINE
ConcurrentHopscotch<K, V>::ConcurrentHopscotch(uint32_t index_num_kv)
    : GenericConcurrentHopscotch(helpers::bsr_64(index_num_kv - 1) + 1) {
  memset(per_core_size_, 0, sizeof(per_core_size_));
}

template <typename K, typename V>
FORCE_INLINE std::optional<V> ConcurrentHopscotch<K, V>::_find(const K &key) {
  uint16_t val_len;
  V val;
  _get(sizeof(key), &key, &val_len, &val);
  if (val_len == 0) {
    return std::nullopt;
  } else {
    return val;
  }
}

template <typename K, typename V>
FORCE_INLINE void ConcurrentHopscotch<K, V>::_insert(const K &key,
                                                     const V &val) {
  bool key_existed = put(sizeof(key), &key, sizeof(val), &val);
  if (!key_existed) {
    preempt_disable();
    per_core_size_[get_core_num()].data++;
    preempt_enable();
  }
}

template <typename K, typename V>
FORCE_INLINE bool ConcurrentHopscotch<K, V>::_erase(const K &key) {
  bool key_existed = remove(sizeof(key), &key);
  if (key_existed) {
    preempt_disable();
    per_core_size_[get_core_num()].data--;
    preempt_enable();
  }
  return key_existed;
}

template <typename K, typename V>
FORCE_INLINE bool ConcurrentHopscotch<K, V>::empty() const {
  return size() == 0;
}

template <typename K, typename V>
FORCE_INLINE uint64_t ConcurrentHopscotch<K, V>::size() const {
  int64_t sum = 0;
  FOR_ALL_SOCKET0_CORES(i) { sum += per_core_size_[i].data; }
  return sum;
}

template <typename K, typename V>
FORCE_INLINE std::optional<V>
ConcurrentHopscotch<K, V>::find(const DerefScope &scope, const K &key) {
  return _find(key);
}

template <typename K, typename V>
FORCE_INLINE std::optional<V> ConcurrentHopscotch<K, V>::find_tp(const K &key) {
  DerefScope scope;
  return _find(key);
}

template <typename K, typename V>
FORCE_INLINE void ConcurrentHopscotch<K, V>::insert(const DerefScope &scope,
                                                    const K &key,
                                                    const V &val) {
  _insert(key, val);
}

template <typename K, typename V>
FORCE_INLINE void ConcurrentHopscotch<K, V>::insert_tp(const K &key,
                                                       const V &val) {
  DerefScope scope;
  _insert(key, val);
}

template <typename K, typename V>
FORCE_INLINE bool ConcurrentHopscotch<K, V>::erase(const DerefScope &scope,
                                                   const K &key) {
  return _erase(key);
}

template <typename K, typename V>
FORCE_INLINE bool ConcurrentHopscotch<K, V>::erase_tp(const K &key) {
  DerefScope scope;
  return _erase(key);
}

} // namespace far_memory
