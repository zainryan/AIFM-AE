#pragma once

#include "sync.h"

#include <map>
#include <memory>
#include <optional>

namespace far_memory {

constexpr uint32_t kNumMaps = 1024;

struct LockEntry {
  std::unique_ptr<rt::CondVar> cond;

  LockEntry() {}
};

class ObjLocker {
private:
  std::map<uint64_t, LockEntry> maps_[kNumMaps];
  rt::Spin spins_[kNumMaps];

public:
  uint32_t hash_func(uint64_t obj_id);
  bool try_insert(uint64_t obj_id);
  void remove(uint64_t obj_id);
};
}; // namespace far_memory
