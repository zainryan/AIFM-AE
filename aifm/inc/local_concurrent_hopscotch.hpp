#pragma once

#include "sync.h"

#include "cb.hpp"
#include "helpers.hpp"
#include "slab.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>

namespace far_memory {

class LocalGenericConcurrentHopscotch {
private:
#pragma pack(push, 1)
  struct KVDataHeader {
    uint8_t key_len;
    uint16_t val_len;
  };
#pragma pack(pop)
  static_assert(sizeof(KVDataHeader) == 3);

#pragma pack(push, 1)
  struct BucketEntry {
    constexpr static uint64_t kBusyPtr = 0x1;

    uint32_t bitmap;
    rt::Spin spin;
    uint64_t timestamp;
    KVDataHeader *ptr;

    BucketEntry();
  };
#pragma pack(pop)
  static_assert(sizeof(BucketEntry) == 24);

  constexpr static uint32_t kNeighborhood = 32;
  constexpr static uint32_t kMaxRetries = 2;

  const uint32_t kHashMask_;
  const uint32_t kNumEntries_;
  std::unique_ptr<uint8_t> buckets_mem_;
  uint64_t slab_base_addr_;
  Slab slab_;
  BucketEntry *buckets_;
  friend class FarMemTest;

  void do_remove(BucketEntry *bucket, BucketEntry *entry);

public:
  LocalGenericConcurrentHopscotch(uint32_t num_entries_shift,
                                  uint64_t data_size);
  ~LocalGenericConcurrentHopscotch();
  void get(uint8_t key_len, const uint8_t *key, uint16_t *val_len, uint8_t *val,
           bool remove = false);
  bool put(uint8_t key_len, const uint8_t *key, uint16_t val_len,
           const uint8_t *val);
  bool remove(uint8_t key_len, const uint8_t *key);
};

template <typename K, typename V>
class LocalConcurrentHopscotch : public LocalGenericConcurrentHopscotch {
private:
  constexpr static uint64_t kKVDataSize =
      sizeof(K) + sizeof(V) + sizeof(KVDataHeader);
  CachelineAligned(int64_t) per_core_size_[helpers::kNumCPUs];

public:
  LocalConcurrentHopscotch(uint32_t index_num_kv, uint64_t data_num_kv);
  bool empty() const;
  uint64_t size() const;
  std::optional<V> find(const K &key);
  void insert(const K &key, const V &val);
  bool erase(const K &key);
};

template <typename K, typename V>
FORCE_INLINE
LocalConcurrentHopscotch<K, V>::LocalConcurrentHopscotch(uint32_t index_num_kv,
                                                         uint64_t data_num_kv)
    : LocalGenericConcurrentHopscotch(helpers::bsr_64(index_num_kv - 1) + 1,
                                      (data_num_kv - 1) / kKVDataSize + 1) {
  memset(per_core_size_, 0, sizeof(per_core_size_));
}

template <typename K, typename V>
FORCE_INLINE bool LocalConcurrentHopscotch<K, V>::empty() const {
  return size() == 0;
}

template <typename K, typename V>
FORCE_INLINE uint64_t LocalConcurrentHopscotch<K, V>::size() const {
  int64_t sum = 0;
  FOR_ALL_SOCKET0_CORES(i) { sum += per_core_size_[i].data; }
  return sum;
}

template <typename K, typename V>
FORCE_INLINE std::optional<V>
LocalConcurrentHopscotch<K, V>::find(const K &key) {
  uint16_t val_len;
  V val;
  get(sizeof(key), &key, &val_len, &val);
  if (val_len == 0) {
    return std::nullopt;
  } else {
    return val;
  }
}

template <typename K, typename V>
FORCE_INLINE void LocalConcurrentHopscotch<K, V>::insert(const K &key,
                                                         const V &val) {
  bool key_existed = put(sizeof(key), &key, sizeof(val), &val);
  if (!key_existed) {
    preempt_disable();
    per_core_size_[get_core_num()].data++;
    preempt_enable();
  }
}

template <typename K, typename V>
FORCE_INLINE bool LocalConcurrentHopscotch<K, V>::erase(const K &key) {
  bool key_existed = remove(sizeof(key), &key);
  if (key_existed) {
    preempt_disable();
    per_core_size_[get_core_num()].data--;
    preempt_enable();
  }
  return key_existed;
}

FORCE_INLINE LocalGenericConcurrentHopscotch::BucketEntry::BucketEntry() {
  bitmap = timestamp = 0;
  ptr = nullptr;
}

} // namespace far_memory
