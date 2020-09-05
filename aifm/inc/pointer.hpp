#pragma once

#include "deref_scope.hpp"
#include "helpers.hpp"
#include "object.hpp"
#include "region.hpp"

#include <memory>
#include <optional>
#include <type_traits>

namespace far_memory {

// Format:
//  I) |XXXXXXX !H(1b)|  0   S(1b)!D(1b)00000|E(1b)|  Object Data Addr(47b)  |
// II) |   DS_ID(8b)  |!P(1b)S(1b)| Object Size(16b) |      ObjectID(38b)    |
//
//                  D: dirty bit.
//                  P: present.
//                  H: hot bits.
//                  S: shared bits, meaning the pointer is a UniquePtr or a
//                     SharedPtr.
//                  E: The pointed data is being evacuated.
//   Object Data Addr: the address of the referenced object's data (which are
//                     stored in regions).
//              DS_ID: data structure ID.
//        Object Size: the size of the pointed object.
//          Object ID: universal object ID (used when swapping in).

class FarMemPtrMeta {
private:
  constexpr static uint32_t kSize = 8;
  constexpr static uint32_t kEvacuationPos = 2;
  constexpr static uint32_t kObjectIDBitPos = 26;
  constexpr static uint32_t kObjectIDBitSize = 38;
  constexpr static uint32_t kObjectDataAddrPos = 2;
  constexpr static uint32_t kObjectDataAddrSize = 6;
  constexpr static uint32_t kDirtyClear = 0x400U;
  constexpr static uint32_t kPresentClear = 0x100U;
  constexpr static uint32_t kHotClear = 0x80U;
  constexpr static uint32_t kEvacuationSet = 0x10000U;
  constexpr static uint32_t kObjIDLenPosShift = 9;
  constexpr static uint32_t kObjectDataAddrBitPos = 17;
  constexpr static uint32_t kObjectSizeBitPos = 10;
  constexpr static uint32_t kHotPos = 0;
  constexpr static uint32_t kPresentPos = 1;
  constexpr static uint32_t kHotThresh = 2;
  constexpr static uint32_t kDSIDPos = 0;
  constexpr static uint32_t kSharedBitPos = 9;

  uint8_t metadata_[kSize];
  friend class FarMemManager;
  friend class GenericFarMemPtr;
  friend class GenericUniquePtr;

  FarMemPtrMeta();
  void init(bool shared, uint64_t object_addr);

public:
  constexpr static uint64_t kNull = kPresentClear;
  constexpr static uint64_t kNullMask =
      ((~static_cast<uint64_t>(0)) << (8 * kPresentPos));

  FarMemPtrMeta(const FarMemPtrMeta &other);
  FarMemPtrMeta(bool shared, uint64_t object_addr);
  bool is_dirty() const;
  void set_dirty();
  void clear_dirty();
  bool is_hot() const;
  bool is_nt() const;
  void clear_hot();
  void set_hot();
  bool is_present() const;
  void set_present(uint64_t object_addr);
  void set_evacuation();
  bool is_evacuation() const;
  bool is_shared() const;
  void set_shared();
  uint64_t get_object_id() const;
  uint64_t get_object_data_addr() const;
  void set_object_data_addr(uint64_t new_local_object_addr);
  uint64_t get_object_addr() const;
  uint16_t get_object_size() const;
  Object object();
  uint8_t get_ds_id() const;
  bool is_null() const;
  void nullify();
  uint64_t to_uint64_t() const;
  void from_uint64_t(uint64_t val);
  void mutator_copy(uint64_t new_local_object_addr);
  void gc_copy(uint64_t new_local_object_addr);
  void gc_wb(uint8_t ds_id, uint16_t object_size, uint64_t obj_id);
  static FarMemPtrMeta *from_object(const Object &object);
};

class GenericFarMemPtr {
private:
  FarMemPtrMeta meta_;

protected:
  GenericFarMemPtr();
  GenericFarMemPtr(bool shared, uint64_t object_addr);
  void init(bool shared, uint64_t object_addr);
  Object object();
  FarMemPtrMeta &meta();

public:
  void nullify();
  bool is_null() const;
};

class GenericUniquePtr : public GenericFarMemPtr {
protected:
  friend class FarMemTest;
  friend class FarMemManager;
  template <typename Induce, typename Infer> friend class Prefetcher;

  void init(uint64_t object_addr);
  bool mutator_migrate_object();
  auto pin(void **pinned_raw_ptr = nullptr);
  void _free();

public:
  GenericUniquePtr();
  ~GenericUniquePtr();
  GenericUniquePtr(uint64_t object_addr);
  GenericUniquePtr(GenericUniquePtr &&other);
  GenericUniquePtr &operator=(GenericUniquePtr &&other);
  NOT_COPYABLE(GenericUniquePtr);
  template <bool Mut, bool Nt> void *_deref();
  template <bool Nt = false> const void *deref(const DerefScope &scope);
  template <bool Nt = false> void *deref_mut(const DerefScope &scope);
  void free();
  void release();
  void swap_in(bool nt);
  void move(GenericUniquePtr &other, uint64_t reset_value);
};

template <typename T> class UniquePtr : public GenericUniquePtr {
private:
  friend class FarMemManager;
  template <typename R, uint64_t... Dims> friend class Array;

  UniquePtr(uint64_t local_object_addr);

public:
  UniquePtr();
  ~UniquePtr();
  UniquePtr(UniquePtr &&other);
  UniquePtr &operator=(UniquePtr &&other);
  NOT_COPYABLE(UniquePtr);
  template <bool Nt = false> const T *deref(const DerefScope &scope);
  template <bool Nt = false> T *deref_mut(const DerefScope &scope);
  template <bool Nt = false> T read();
  template <bool Nt = false, typename U> void write(U &&u);
  void free();
};

FORCE_INLINE FarMemPtrMeta::FarMemPtrMeta() {
  // GC design assumes the pointer is always aligned so that R/W operations
  // will be atomic.
  assert(reinterpret_cast<uint64_t>(this) % sizeof(FarMemPtrMeta) == 0);
  nullify();
}

FORCE_INLINE FarMemPtrMeta::FarMemPtrMeta(const FarMemPtrMeta &other) {
  static_assert(kSize == sizeof(uint64_t));
  *reinterpret_cast<uint64_t *>(metadata_) =
      *reinterpret_cast<const uint64_t *>(other.metadata_);
}

FORCE_INLINE
FarMemPtrMeta::FarMemPtrMeta(bool shared, uint64_t object_addr) {
  memset(this, 0, sizeof(*this));
  init(shared, object_addr);
}

FORCE_INLINE
void FarMemPtrMeta::init(bool shared, uint64_t object_addr) {
  if (shared) {
    set_shared();
  }
  set_present(object_addr);
  set_dirty();
}

FORCE_INLINE bool FarMemPtrMeta::is_present() const {
  return !((ACCESS_ONCE(*reinterpret_cast<const uint16_t *>(metadata_))) &
           kPresentClear);
}

FORCE_INLINE bool FarMemPtrMeta::is_null() const {
  return (to_uint64_t() & kNullMask) == kNull;
}

FORCE_INLINE void FarMemPtrMeta::nullify() { from_uint64_t(kNull); }

FORCE_INLINE uint64_t FarMemPtrMeta::get_object_data_addr() const {
  return to_uint64_t() >> kObjectDataAddrBitPos;
}

FORCE_INLINE uint64_t FarMemPtrMeta::get_object_addr() const {
  return get_object_data_addr() - Object::kHeaderSize;
}

FORCE_INLINE uint16_t FarMemPtrMeta::get_object_size() const {
  assert(!is_present());
  return to_uint64_t() >> kObjectSizeBitPos;
}

FORCE_INLINE Object FarMemPtrMeta::object() {
  return Object(get_object_addr());
}

FORCE_INLINE FarMemPtrMeta *FarMemPtrMeta::from_object(const Object &object) {
  return reinterpret_cast<FarMemPtrMeta *>(object.get_ptr_addr());
}

FORCE_INLINE uint64_t FarMemPtrMeta::get_object_id() const {
  return to_uint64_t() >> kObjectIDBitPos;
}

FORCE_INLINE bool FarMemPtrMeta::is_dirty() const {
  return !((*reinterpret_cast<const uint16_t *>(metadata_)) & kDirtyClear);
}

FORCE_INLINE void FarMemPtrMeta::set_dirty() {
  *reinterpret_cast<uint16_t *>(metadata_) &= (~kDirtyClear);
}

FORCE_INLINE void FarMemPtrMeta::clear_dirty() {
  *reinterpret_cast<uint16_t *>(metadata_) |= kDirtyClear;
}

FORCE_INLINE bool FarMemPtrMeta::is_hot() const {
  return !((*reinterpret_cast<const uint16_t *>(metadata_)) & kHotClear);
}

FORCE_INLINE void FarMemPtrMeta::clear_hot() {
  metadata_[kHotPos] = (kHotClear >> (8 * kHotPos)) + (kHotThresh - 1);
}

FORCE_INLINE bool FarMemPtrMeta::is_nt() const {
  auto obj_data_addr = get_object_data_addr();
  if (unlikely(!obj_data_addr)) {
    // The pointer has being freed.
    return false;
  }
  return Region::is_nt(obj_data_addr & (~(Region::kSize - 1)));
}

FORCE_INLINE void FarMemPtrMeta::set_hot() {
  *reinterpret_cast<uint16_t *>(metadata_) &= (~kHotClear);
}

FORCE_INLINE uint64_t FarMemPtrMeta::to_uint64_t() const {
  return ACCESS_ONCE(*reinterpret_cast<const uint64_t *>(metadata_));
}

FORCE_INLINE void FarMemPtrMeta::from_uint64_t(uint64_t val) {
  ACCESS_ONCE(*reinterpret_cast<uint64_t *>(metadata_)) = val;
}

FORCE_INLINE void FarMemPtrMeta::set_evacuation() {
  metadata_[kEvacuationPos] |= 1;
}

FORCE_INLINE bool FarMemPtrMeta::is_evacuation() const {
  return metadata_[kEvacuationPos] & 1;
}

FORCE_INLINE bool FarMemPtrMeta::is_shared() const {
  return to_uint64_t() & (1 << kSharedBitPos);
}

FORCE_INLINE void FarMemPtrMeta::set_shared() {
  from_uint64_t(to_uint64_t() | (1 < kSharedBitPos));
}

FORCE_INLINE GenericFarMemPtr::GenericFarMemPtr() {}

FORCE_INLINE GenericFarMemPtr::GenericFarMemPtr(bool shared,
                                                uint64_t object_addr)
    : meta_(shared, object_addr) {}

FORCE_INLINE void GenericFarMemPtr::init(bool shared, uint64_t object_addr) {
  meta_.init(shared, object_addr);
}

FORCE_INLINE Object GenericFarMemPtr::object() { return meta_.object(); }

FORCE_INLINE FarMemPtrMeta &GenericFarMemPtr::meta() { return meta_; }

FORCE_INLINE void GenericFarMemPtr::nullify() { meta_.nullify(); }

FORCE_INLINE bool GenericFarMemPtr::is_null() const { return meta_.is_null(); }

FORCE_INLINE uint8_t FarMemPtrMeta::get_ds_id() const {
  assert(!is_present());
  return metadata_[kDSIDPos];
}

FORCE_INLINE GenericUniquePtr::GenericUniquePtr() {}

FORCE_INLINE GenericUniquePtr::~GenericUniquePtr() {
  if (!meta().is_null()) {
    free();
  }
}

FORCE_INLINE GenericUniquePtr::GenericUniquePtr(uint64_t object_addr)
    : GenericFarMemPtr(/* shared = */ false, object_addr) {}

FORCE_INLINE void GenericUniquePtr::init(uint64_t object_addr) {
  GenericFarMemPtr::init(/* shared =  */ false, object_addr);
}

FORCE_INLINE auto GenericUniquePtr::pin(void **pinned_raw_ptr) {
  bool in_scope = DerefScope::is_in_deref_scope();
  if (!in_scope) {
    DerefScope::enter_deref_scope();
  }
  auto *derefed_ptr = _deref</*Mut = */ false, /* Nt = */ false>();
  if (pinned_raw_ptr) {
    *pinned_raw_ptr = derefed_ptr;
  }
  return helpers::finally([=]() {
    if (!in_scope) {
      DerefScope::exit_deref_scope();
    }
  });
}

template <bool Mut, bool Nt> FORCE_INLINE void *GenericUniquePtr::_deref() {
retry:
  // 1) movq.
  auto metadata = meta().to_uint64_t();
  auto exceptions = (FarMemPtrMeta::kHotClear | FarMemPtrMeta::kPresentClear |
                     FarMemPtrMeta::kEvacuationSet);
  if constexpr (Mut) {
    exceptions |= FarMemPtrMeta::kDirtyClear;
  }
  // 2) test. 3) jne. They got macro-fused into a single uop.
  if (very_unlikely(metadata & exceptions)) {
    // Slow path.
    if (very_unlikely(metadata & (FarMemPtrMeta::kPresentClear |
                                  FarMemPtrMeta::kEvacuationSet))) {
      if (metadata & FarMemPtrMeta::kPresentClear) {
        if (meta().is_null()) {
          // In this case, _deref() returns nullptr.
          return nullptr;
        }
        swap_in(Nt);
        // Just swapped in, need to update metadata (for the obj data addr).
        metadata = meta().to_uint64_t();
      } else {
        if (!mutator_migrate_object()) {
          // GC or another thread wins the race. They may still need a while to
          // finish migrating the object. Yielding itself rather than busy
          // retrying now.
          thread_yield();
        }
      }
      goto retry;
    }
    if constexpr (Mut) {
      // set P and D.
      __asm__("movb $0, %0"
              : "=m"(meta().metadata_[FarMemPtrMeta::kPresentPos]));
    }
    meta().metadata_[FarMemPtrMeta::kHotPos]--;
  }

  // 4) shrq.
  return reinterpret_cast<void *>(metadata >>
                                  FarMemPtrMeta::kObjectDataAddrBitPos);
}

template <bool Nt>
FORCE_INLINE const void *GenericUniquePtr::deref(const DerefScope &scope) {
  return reinterpret_cast<const void *>(
      GenericUniquePtr::_deref</* Mut = */ false, Nt>());
}

template <bool Nt>
FORCE_INLINE void *GenericUniquePtr::deref_mut(const DerefScope &scope) {
  return GenericUniquePtr::_deref</* Mut = */ true, Nt>();
}

FORCE_INLINE GenericUniquePtr::GenericUniquePtr(GenericUniquePtr &&other) {
  *this = std::move(other);
}

template <typename T>
FORCE_INLINE UniquePtr<T>::UniquePtr(uint64_t object_addr)
    : GenericUniquePtr(object_addr) {}

template <typename T>
FORCE_INLINE UniquePtr<T>::UniquePtr() : GenericUniquePtr() {
  meta().nullify();
}

template <typename T> FORCE_INLINE UniquePtr<T>::~UniquePtr() {
  if (!meta().is_null()) {
    free();
  }
}

template <typename T>
template <bool Nt>
FORCE_INLINE const T *UniquePtr<T>::deref(const DerefScope &scope) {
  return reinterpret_cast<const T *>(GenericUniquePtr::deref<Nt>(scope));
}

template <typename T>
template <bool Nt>
FORCE_INLINE T *UniquePtr<T>::deref_mut(const DerefScope &scope) {
  return reinterpret_cast<T *>(GenericUniquePtr::deref_mut<Nt>(scope));
}

template <typename T> FORCE_INLINE UniquePtr<T>::UniquePtr(UniquePtr &&other) {
  *this = std::move(other);
}

template <typename T>
FORCE_INLINE UniquePtr<T> &UniquePtr<T>::operator=(UniquePtr &&other) {
  GenericUniquePtr::operator=(std::move(other));
  return *this;
}

template <typename T> template <bool Nt> FORCE_INLINE T UniquePtr<T>::read() {
  DerefScope scope;
  return *(deref<Nt>(scope));
}

template <typename T>
template <bool Nt, typename U>
FORCE_INLINE void UniquePtr<T>::write(U &&u) {
  static_assert(std::is_same<std::decay_t<U>, std::decay_t<T>>::value,
                "U must be the same as T");
  DerefScope scope;
  *(deref_mut<Nt>(scope)) = u;
}

template <typename T> FORCE_INLINE void UniquePtr<T>::free() {
  T *raw_ptr;
  auto pin_guard = pin(reinterpret_cast<void **>(&raw_ptr));
  raw_ptr->~T();
  _free();
}

} // namespace far_memory
