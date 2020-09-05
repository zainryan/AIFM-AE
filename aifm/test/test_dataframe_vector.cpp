extern "C" {
#include <runtime/runtime.h>
}

#include "dataframe_vector.hpp"
#include "deref_scope.hpp"
#include "device.hpp"
#include "helpers.hpp"
#include "manager.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

using namespace far_memory;
using namespace std;

constexpr uint64_t kCacheSize = 512 * Region::kSize;
constexpr uint64_t kFarMemSize = (1ULL << 33); // 8 GB.
constexpr uint64_t kWorkSetSize = 1 << 30;
constexpr uint64_t kNumGCThreads = 12;
constexpr uint64_t kNumEntries = 64 << 20; // 64 million entries.

namespace far_memory {
class FarMemTest {
private:
public:
  void do_work(FarMemManager *manager) {
    auto dataframe_vector = manager->allocate_dataframe_vector<long long>();
    for (uint64_t i = 0; i < kNumEntries; i++) {
      DerefScope scope;
      dataframe_vector.push_back(scope, static_cast<long long>(i));
    }

    for (uint64_t i = 0; i < kNumEntries; i++) {
      DerefScope scope;
      TEST_ASSERT(dataframe_vector.at(scope, i) == static_cast<long long>(i));
    }

    {
      DerefScope scope;
      TEST_ASSERT(dataframe_vector.front(scope) == 0);
      TEST_ASSERT(dataframe_vector.back(scope) == kNumEntries - 1);
    }
    TEST_ASSERT(!dataframe_vector.empty());
    TEST_ASSERT(dataframe_vector.size() == kNumEntries);
    for (uint64_t i = 0; i < kNumEntries; i++) {
      DerefScope scope;
      dataframe_vector.pop_back(scope);
      TEST_ASSERT(dataframe_vector.size() == kNumEntries - 1 - i);
    }
    TEST_ASSERT(dataframe_vector.empty());

    dataframe_vector.reserve(kNumEntries * 2);
    TEST_ASSERT(dataframe_vector.capacity() >= kNumEntries * 2);

    cout << "Passed" << endl;
    return;
  }
};
} // namespace far_memory

void _main(void *arg) {
  auto manager = std::unique_ptr<FarMemManager>(FarMemManagerFactory::build(
      kCacheSize, kNumGCThreads, new FakeDevice(kFarMemSize)));
  FarMemTest test;
  test.do_work(manager.get());
}

int main(int argc, char *argv[]) {
  int ret;

  if (argc < 2) {
    std::cerr << "usage: [cfg_file]" << std::endl;
    return -EINVAL;
  }

  ret = runtime_init(argv[1], _main, NULL);
  if (ret) {
    std::cerr << "failed to start runtime" << std::endl;
    return ret;
  }

  return 0;
}
