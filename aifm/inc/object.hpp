#pragma once

#include "helpers.hpp"

#include <limits>

namespace far_memory {

// Forward declaration.
template <typename T> class UniquePtr;

class Object {
  //
  // Format:
  // |<------------------ header ------------------>|
  // |ptr_addr(6B)|data_len(2B)|ds_id(1B)|id_len(1B)|data|object_ID|
  //
  //      ptr_addr: points to the corresponding far-mem pointer. During GC,
  //                the collector uses the field to jump from Region to far-mem
  //                pointer, and marks the far-mem pointer as absent from local
  //                cache.
  //      data_len: the length of object data.
  //         ds_id: the data structure ID.
  //        id_len: the length of object ID.
  //          data: object data.
  //     object_ID: the unique object ID which is used by the remote side to
  //                locate the object during swapping in and swapping out.
private:
  constexpr static uint32_t kPtrAddrPos = 0;
  constexpr static uint32_t kPtrAddrSize = 6;
  constexpr static uint32_t kDataLenPos = 6;
  constexpr static uint32_t kDSIDPos = 8;
  constexpr static uint32_t kIDLenPos = 9;
  // It stores the address of the object (which is stored in the local region).
  uint64_t addr_;

public:
  constexpr static uint32_t kDSIDSize = 1;
  constexpr static uint32_t kIDLenSize = 1;
  constexpr static uint32_t kDataLenSize = 2;
  constexpr static uint32_t kHeaderSize =
      kPtrAddrSize + kDataLenSize + kDSIDSize + kIDLenSize;
  constexpr static uint16_t kMaxObjectSize =
      std::numeric_limits<uint16_t>::max();
  constexpr static uint16_t kMaxObjectIDSize = (1 << (8 * kIDLenSize)) - 1;
  constexpr static uint16_t kMaxObjectDataSize =
      kMaxObjectSize - kHeaderSize - kMaxObjectIDSize;

  Object();
  // Create a reference to the object at address addr.
  Object(uint64_t addr);
  // Initialize the object at address addr. Field ptr_addr is written by
  // far-mem pointer.
  Object(uint64_t addr, uint8_t ds_id, uint16_t data_len, uint8_t id_len,
         const uint8_t *id);
  void init(uint8_t ds_id, uint16_t data_len, uint8_t id_len,
            const uint8_t *id);
  uint64_t get_addr() const;
  uint16_t get_data_len() const;
  uint8_t get_obj_id_len() const;
  void set_ptr_addr(uint64_t address);
  const uint8_t *get_obj_id() const;
  uint64_t get_ptr_addr() const;
  uint64_t get_data_addr() const;
  void set_ds_id(uint8_t ds_id);
  uint8_t get_ds_id() const;
  void set_data_len(uint16_t data_len);
  void set_object_id(const uint8_t *id, uint8_t id_len);
  void set_obj_id_len(uint8_t id_len);
  uint16_t size() const;
  bool is_freed() const;
  void free();
};

FORCE_INLINE Object::Object() {}

FORCE_INLINE Object::Object(uint64_t addr) : addr_(addr) {}

FORCE_INLINE Object::Object(uint64_t addr, uint8_t ds_id, uint16_t data_len,
                            uint8_t id_len, const uint8_t *id)
    : Object(addr) {
  init(ds_id, data_len, id_len, id);
}

FORCE_INLINE void Object::init(uint8_t ds_id, uint16_t data_len, uint8_t id_len,
                               const uint8_t *id) {
  set_ds_id(ds_id);
  set_data_len(data_len);
  set_obj_id_len(id_len);
  set_object_id(id, id_len);
}

FORCE_INLINE void Object::set_ds_id(uint8_t ds_id) {
  auto *ptr = reinterpret_cast<uint8_t *>(addr_ + kDSIDPos);
  *ptr = ds_id;
}

FORCE_INLINE uint8_t Object::get_ds_id() const {
  auto *ptr = reinterpret_cast<uint8_t *>(addr_ + kDSIDPos);
  return *ptr;
}

FORCE_INLINE void Object::set_obj_id_len(uint8_t id_len) {
  auto *ptr = reinterpret_cast<uint8_t *>(addr_ + kIDLenPos);
  *ptr = id_len;
}

FORCE_INLINE uint8_t Object::get_obj_id_len() const {
  auto *ptr = reinterpret_cast<uint8_t *>(addr_ + kIDLenPos);
  return *ptr;
}

FORCE_INLINE uint64_t Object::get_addr() const { return addr_; }

FORCE_INLINE bool Object::is_freed() const {
  return (*reinterpret_cast<uint8_t *>(addr_ + kPtrAddrPos + kPtrAddrSize -
                                       1)) == 0xFF;
}

FORCE_INLINE void Object::free() {
  *reinterpret_cast<uint8_t *>(addr_ + kPtrAddrPos + kPtrAddrSize - 1) = 0xFF;
}

FORCE_INLINE void Object::set_data_len(uint16_t data_len) {
  auto *ptr = reinterpret_cast<uint16_t *>(addr_ + kDataLenPos);
  *ptr = data_len;
}

FORCE_INLINE uint16_t Object::get_data_len() const {
  auto *ptr = reinterpret_cast<uint16_t *>(addr_ + kDataLenPos);
  return *ptr;
}

FORCE_INLINE void Object::set_object_id(const uint8_t *id, uint8_t id_len) {
  auto offset = kHeaderSize + get_data_len();
  auto *ptr = reinterpret_cast<void *>(addr_ + offset);
  memcpy(ptr, id, id_len);
}

FORCE_INLINE const uint8_t *Object::get_obj_id() const {
  auto offset = kHeaderSize + get_data_len();
  auto *ptr = reinterpret_cast<const uint8_t *>(addr_ + offset);
  return ptr;
}

FORCE_INLINE void Object::set_ptr_addr(uint64_t address) {
  helpers::small_memcpy<kPtrAddrSize>(
      reinterpret_cast<void *>(addr_ + kPtrAddrPos), &address);
}

FORCE_INLINE uint64_t Object::get_ptr_addr() const {
  uint64_t address = 0;
  helpers::small_memcpy<kPtrAddrSize>(
      &address, reinterpret_cast<void *>(addr_ + kPtrAddrPos));
  return address;
}

FORCE_INLINE uint64_t Object::get_data_addr() const {
  return addr_ + kHeaderSize;
}

FORCE_INLINE uint16_t Object::size() const {
  return kHeaderSize + get_data_len() + get_obj_id_len();
}

} // namespace far_memory
