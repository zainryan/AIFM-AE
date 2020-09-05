extern "C" {
#include <base/assert.h>
#include <base/compiler.h>
#include <base/stddef.h>
}

#include "dataframe_types.hpp"
#include "dataframe_vector.hpp"
#include "server_dataframe_vector.hpp"

#include <cstring>

namespace far_memory {

template <typename T> ServerDataFrameVector<T>::ServerDataFrameVector() {
  memset(ref_cnts_, 0, sizeof(ref_cnts_));
}

template <typename T> ServerDataFrameVector<T>::~ServerDataFrameVector() {}

template <typename T> void ServerDataFrameVector<T>::enter_barrier() {
  while (unlikely(ACCESS_ONCE(global_wait_))) {
    thread_yield();
  }
  preempt_disable();
  auto core_num = get_core_num();
  ACCESS_ONCE(ref_cnts_[core_num].data)++;
  preempt_enable();
}

template <typename T> void ServerDataFrameVector<T>::exit_barrier() {
  preempt_disable();
  auto core_num = get_core_num();
  ACCESS_ONCE(ref_cnts_[core_num].data)--;
  preempt_enable();
}

template <typename T>
void ServerDataFrameVector<T>::disable_and_wait_barrier() {
  global_wait_ = true;
  barrier();
retry:
  int32_t sum = 0;
  FOR_ALL_SOCKET0_CORES(i) { sum += ACCESS_ONCE(ref_cnts_[i].data); }
  if (sum) {
    thread_yield();
    goto retry;
  }
}

template <typename T> void ServerDataFrameVector<T>::enable_barrier() {
  store_release(&global_wait_, false);
}

template <typename T>
void ServerDataFrameVector<T>::read_object(uint8_t obj_id_len,
                                           const uint8_t *obj_id,
                                           uint16_t *data_len,
                                           uint8_t *data_buf) {
  enter_barrier();
  helpers::finally([&]() { exit_barrier(); });
  uint64_t index;
  assert(obj_id_len == sizeof(index));
  index = *((uint64_t *)obj_id);
  auto chunk_size = DataFrameVector<T>::kRealChunkSize;
  *data_len = chunk_size;
  __builtin_memcpy(data_buf, (uint8_t *)vec_.data() + index * chunk_size,
                   chunk_size);
}

template <typename T>
void ServerDataFrameVector<T>::write_object(uint8_t obj_id_len,
                                            const uint8_t *obj_id,
                                            uint16_t data_len,
                                            const uint8_t *data_buf) {
  enter_barrier();
  helpers::finally([&]() { exit_barrier(); });
  uint64_t index;
  assert(obj_id_len == sizeof(index));
  index = *((uint64_t *)obj_id);
  auto chunk_size = DataFrameVector<T>::kRealChunkSize;
  assert(data_len == chunk_size);
  __builtin_memcpy((uint8_t *)vec_.data() + index * chunk_size, data_buf,
                   chunk_size);
}

template <typename T>
bool ServerDataFrameVector<T>::remove_object(uint8_t obj_id_len,
                                             const uint8_t *obj_id) {
  // This should never be called.
  BUG();
}

template <typename T>
void ServerDataFrameVector<T>::compute_resize(uint16_t input_len,
                                              const uint8_t *input_buf,
                                              uint16_t *output_len,
                                              uint8_t *output_buf) {
  disable_and_wait_barrier();
  helpers::finally([&]() { enable_barrier(); });
  uint64_t new_capacity;
  assert(input_len == sizeof(new_capacity));
  new_capacity = *((uint64_t *)input_buf);
  assert(new_capacity > vec_.size());
  vec_.resize(new_capacity);
  *output_len = 0;
}

template <typename T>
void ServerDataFrameVector<T>::compute(uint8_t opcode, uint16_t input_len,
                                       const uint8_t *input_buf,
                                       uint16_t *output_len,
                                       uint8_t *output_buf) {
  if (opcode == GenericDataFrameVector::OpCode::Resize) {
    compute_resize(input_len, input_buf, output_len, output_buf);
  } else {
    BUG();
  }
}

ServerDS *ServerDataFrameVectorFactory::build(uint32_t param_len,
                                              uint8_t *params) {
  uint8_t dt_id;
  BUG_ON(param_len != sizeof(dt_id));
  dt_id = *params;

  if (dt_id == DataFrameTypeID::Char) {
    return new ServerDataFrameVector<char>();
  }
  if (dt_id == DataFrameTypeID::Short) {
    return new ServerDataFrameVector<short>();
  }
  if (dt_id == DataFrameTypeID::Int) {
    return new ServerDataFrameVector<int>();
  }
  if (dt_id == DataFrameTypeID::Long) {
    return new ServerDataFrameVector<long>();
  }
  if (dt_id == DataFrameTypeID::LongLong) {
    return new ServerDataFrameVector<long long>();
  }
  if (dt_id == DataFrameTypeID::Float) {
    return new ServerDataFrameVector<float>();
  }
  if (dt_id == DataFrameTypeID::Double) {
    return new ServerDataFrameVector<double>();
  }
  BUG();
}
} // namespace far_memory
