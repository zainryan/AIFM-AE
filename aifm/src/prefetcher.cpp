#include "prefetcher.hpp"
#include "manager.hpp"

namespace far_memory {
uint32_t get_prefetch_win_size(uint32_t object_data_size) {
  return FarMemManagerFactory::get()->get_device()->get_prefetch_win_size() /
         object_data_size;
}
} // namespace far_memory
