#include "pointer.hpp"
#include "deref_scope.hpp"
#include "helpers.hpp"
#include "manager.hpp"

#include <cstdint>

namespace far_memory {

void FarMemPtrMeta::set_present(uint64_t object_addr) {
  static_assert(kSize == sizeof(uint64_t));
  auto obj = Object(object_addr);
  obj.set_ptr_addr(reinterpret_cast<uint64_t>(this));
  wmb();
  uint64_t new_metadata =
      ((object_addr + Object::kHeaderSize) << kObjectDataAddrBitPos) |
      ((kDirtyClear | kHotClear) + ((kHotThresh - 1) << (8 * kHotPos)));
  from_uint64_t(new_metadata);
}

void FarMemPtrMeta::mutator_copy(uint64_t new_local_object_addr) {
  constexpr auto obj_data_addr_mask = (1ULL << kObjectDataAddrBitPos) - 1;
  // Clear the evacuation bit.
  constexpr auto evacuation_mask = (~(1ULL << (8 * kEvacuationPos)));
  constexpr auto mask = (obj_data_addr_mask & evacuation_mask);
  auto masked_old_meta = (to_uint64_t() & mask);
  from_uint64_t(masked_old_meta | ((new_local_object_addr + Object::kHeaderSize)
                                   << kObjectDataAddrBitPos));
}

void FarMemPtrMeta::gc_copy(uint64_t new_local_object_addr) {
  uint64_t old_metadata = to_uint64_t();
  assert((old_metadata & kPresentClear) == 0);
  auto new_local_object_data_addr = new_local_object_addr + Object::kHeaderSize;
  auto new_metadata = (new_local_object_data_addr << kObjectDataAddrBitPos) |
                      (kHotClear + ((kHotThresh - 1) << (8 * kHotPos))) |
                      (old_metadata & (0xFF << (8 * kPresentPos)));
  from_uint64_t(new_metadata);
}

void FarMemPtrMeta::gc_wb(uint8_t ds_id, uint16_t object_size,
                          uint64_t obj_id) {
  assert(obj_id < (1ULL << kObjectIDBitSize));
  auto new_metadata =
      (obj_id << kObjectIDBitPos) |
      (static_cast<uint64_t>(object_size) << kObjectSizeBitPos) |
      kPresentClear | ds_id;
  from_uint64_t(new_metadata);
}

void GenericUniquePtr::release() { meta().nullify(); }

void GenericUniquePtr::swap_in(bool nt) {
  FarMemManagerFactory::get()->swap_in(nt, &meta());
}

bool GenericUniquePtr::mutator_migrate_object() {
  auto *manager = FarMemManagerFactory::get();

  auto object = meta().object();
  rmb();
  if (unlikely(!meta().is_present())) {
    return false;
  }

  auto obj_id_len = object.get_obj_id_len();
  auto *obj_id = object.get_obj_id();
  FarMemManager::lock_object(obj_id_len, obj_id);
  auto guard = helpers::finally(
      [&]() { FarMemManager::unlock_object(obj_id_len, obj_id); });

  if (unlikely(!meta().is_present() || !meta().is_evacuation())) {
    return false;
  }

  bool nt = meta().is_nt();
  auto object_size = object.size();

  auto optional_new_local_object_addr =
      manager->allocate_local_object_nb(nt, object_size);
  if (!optional_new_local_object_addr) {
    return false;
  }
  auto new_local_object_addr = *optional_new_local_object_addr;
  memcpy(reinterpret_cast<void *>(new_local_object_addr),
         reinterpret_cast<void *>(object.get_addr()), object_size);
  Region::atomic_inc_ref_cnt(new_local_object_addr, -1);
  meta().mutator_copy(new_local_object_addr);
  return true;
}

GenericUniquePtr &GenericUniquePtr::operator=(GenericUniquePtr &&other) {
  move(other, FarMemPtrMeta::kNull);
  return *this;
}

void GenericUniquePtr::move(GenericUniquePtr &other, uint64_t reset_value) {
retry:
  uint8_t other_obj_id_len = sizeof(uint64_t);
  const uint8_t *other_obj_id_ptr;
  uint64_t other_obj_id;
  bool other_present = other.meta().is_present();
  Object other_object;
  if (other_present) {
    other_object = other.object();
    other_obj_id_ptr = other_object.get_obj_id();
  } else {
    other_obj_id = other.meta().get_object_id();
    other_obj_id_ptr = reinterpret_cast<const uint8_t *>(&other_obj_id);
  }
  FarMemManager::lock_object(other_obj_id_len, other_obj_id_ptr);
  auto guard = helpers::finally([&]() {
    FarMemManager::unlock_object(other_obj_id_len, other_obj_id_ptr);
  });

  if (unlikely(other.meta().is_present() != other_present)) {
    goto retry;
  }

  meta() = other.meta();
  wmb();
  if (other_present) {
    other_object.set_ptr_addr(reinterpret_cast<uint64_t>(this));
  }
  __builtin_memcpy(reinterpret_cast<uint64_t *>(&other.meta()), &reset_value,
                   sizeof(reset_value));
}

void GenericUniquePtr::_free() {
  assert(!meta().is_null());
  assert(meta().is_present());

  auto obj = object();
  auto obj_id_len = obj.get_obj_id_len();
  auto *obj_id = obj.get_obj_id();
  FarMemManager::lock_object(obj_id_len, obj_id);
  auto guard = helpers::finally(
      [&]() { FarMemManager::unlock_object(obj_id_len, obj_id); });

  object().free();
  meta().nullify();
}

void GenericUniquePtr::free() {
  auto pin_guard = pin();
  _free();
}

} // namespace far_memory
