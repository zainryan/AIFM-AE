#include "dataframe_vector.hpp"
#include "ds_info.hpp"
#include "manager.hpp"

namespace far_memory {
GenericDataFrameVector::GenericDataFrameVector(const uint32_t kChunkSize,
                                               const uint32_t kChunkNumEntries,
                                               uint8_t ds_id, uint8_t dt_id)
    : kChunkSize_(kChunkSize), kChunkNumEntries_(kChunkNumEntries),
      device_(FarMemManagerFactory::get()->get_device()), ds_id_(ds_id) {
  FarMemManagerFactory::get()->construct(kDataFrameVectorDSType, ds_id,
                                         sizeof(dt_id), &dt_id);
  expand(1);
  // DataFrameVector essentially stores a std::vector of GenericUniquePtrs, so
  // it does not need a notifier.
}

void GenericDataFrameVector::expand(uint64_t num) {
  auto old_chunk_ptrs_size = chunk_ptrs_.size();
  uint64_t new_capacity = (old_chunk_ptrs_size + num) * kChunkNumEntries_;
  uint16_t output_len;
  device_->compute(ds_id_, OpCode::Resize, sizeof(new_capacity),
                   reinterpret_cast<const uint8_t *>(&new_capacity),
                   &output_len, nullptr);
  assert(!output_len); // No output.

  for (uint64_t i = 0; i < num; i++) {
    uint64_t obj_id = i + old_chunk_ptrs_size;
    chunk_ptrs_.emplace_back();
    while (
        unlikely(!FarMemManagerFactory::get()->allocate_generic_unique_ptr_nb(
            &chunk_ptrs_.back(), ds_id_, kChunkSize_, sizeof(obj_id),
            reinterpret_cast<uint8_t *>(&obj_id)))) {
      FarMemManagerFactory::get()->mutator_wait_for_gc_cache();
    }
  }
}

} // namespace far_memory
