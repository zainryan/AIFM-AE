#pragma once

#include "helpers.hpp"
#include "server.hpp"

#include <cstring>
#include <memory>
#include <vector>

namespace far_memory {

template <typename T> class ServerDataFrameVector : public ServerDS {
private:
  std::vector<T> vec_;
  bool global_wait_ = false;
  CachelineAligned(int32_t) ref_cnts_[helpers::kNumCPUs];
  friend class ServerDataFrameVectorFactory;

  void compute_resize(uint16_t input_len, const uint8_t *input_buf,
                      uint16_t *output_len, uint8_t *output_buf);
  void enter_barrier();
  void exit_barrier();
  void disable_and_wait_barrier();
  void enable_barrier();

public:
  ServerDataFrameVector();
  ~ServerDataFrameVector();
  void read_object(uint8_t obj_id_len, const uint8_t *obj_id,
                   uint16_t *data_len, uint8_t *data_buf);
  void write_object(uint8_t obj_id_len, const uint8_t *obj_id,
                    uint16_t data_len, const uint8_t *data_buf);
  bool remove_object(uint8_t obj_id_len, const uint8_t *obj_id);
  void compute(uint8_t opcode, uint16_t input_len, const uint8_t *input_buf,
               uint16_t *output_len, uint8_t *output_buf);
};

class ServerDataFrameVectorFactory : public ServerDSFactory {
public:
  ServerDS *build(uint32_t param_len, uint8_t *params);
};

}; // namespace far_memory
