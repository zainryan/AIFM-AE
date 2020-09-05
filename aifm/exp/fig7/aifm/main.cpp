extern "C" {
#include <runtime/runtime.h>
}
#include "thread.h"

#include "deref_scope.hpp"
#include "device.hpp"
#include "helpers.hpp"
#include "manager.hpp"
#include "stats.hpp"
#include "zipf.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>

using namespace far_memory;
using namespace std;

#define ACCESS_ONCE(x) (*(volatile typeof(x) *)&(x))

std::atomic_flag flag;
std::unique_ptr<std::mt19937> generators[helpers::kNumCPUs];
__thread uint32_t per_core_req_idx = 0;
std::unique_ptr<FarMemManager> manager;

namespace far_memory {
class FarMemTest {
private:
  constexpr static uint64_t kCacheSize = 2048 * Region::kSize;
  constexpr static uint64_t kFarMemSize = (1ULL << 30);
  constexpr static uint32_t kNumGCThreads = 100;
  constexpr static uint32_t kKeyLen = 12;
  constexpr static uint32_t kValueLen = 4;
  constexpr static uint32_t kLocalHashTableNumEntriesShift =
      27; // 128 M entries
  constexpr static uint32_t kRemoteHashTableNumEntriesShift =
      27; // 128 M entries
  constexpr static uint64_t kRemoteHashTableDataSize = (4ULL << 30); // 4 GB
  constexpr static uint32_t kNumKVPairs = 1 << 27;
  constexpr static uint32_t kNumItersPerScope = 64;
  constexpr static uint32_t kNumMutatorThreads = 400;
  constexpr static uint32_t kReqSeqLenPerCore = kNumKVPairs;
  constexpr static uint32_t kNumConnections = 650;
  constexpr static uint32_t kMonitorPerIter = 262144;
  constexpr static uint32_t kMinMonitorIntervalUs = 10 * 1000 * 1000;
  constexpr static uint32_t kMaxRunningUs = 200 * 1000 * 1000; // 200 seconds
  constexpr static double kZipfParamS = 0;

  struct Key {
    char data[kKeyLen];
  };

  struct Value {
    char data[kValueLen + sizeof(GenericConcurrentHopscotch::NotifierMeta)];
  };

  struct alignas(64) Cnt {
    uint64_t c;
  };

  alignas(helpers::kHugepageSize) Key all_gen_keys[kNumKVPairs];
  uint32_t all_zipf_key_indices[helpers::kNumCPUs][kReqSeqLenPerCore];
  Cnt cnts[kNumMutatorThreads];
  std::vector<double> mops_vec;

  uint64_t prev_sum_cnts = 0;
  uint64_t prev_us = 0;
  uint64_t running_us = 0;

  inline void append_uint32_to_char_array(uint32_t n, uint32_t suffix_len,
                                          char *array) {
    uint32_t len = 0;
    while (n) {
      auto digit = n % 10;
      array[len++] = digit + '0';
      n = n / 10;
    }
    while (len < suffix_len) {
      array[len++] = '0';
    }
    std::reverse(array, array + suffix_len);
  }

  inline void random_string(char *data, uint32_t len) {
    preempt_disable();
    auto guard = helpers::finally([&]() { preempt_enable(); });
    auto &generator = *generators[get_core_num()];
    std::uniform_int_distribution<int> distribution('a', 'z' + 1);
    for (uint32_t i = 0; i < len; i++) {
      data[i] = char(distribution(generator));
    }
  }

  inline void random_key(char *data, uint32_t tid) {
    auto tid_len = helpers::static_log(10, kNumMutatorThreads);
    random_string(data, kKeyLen - tid_len);
    append_uint32_to_char_array(tid, tid_len, data + kKeyLen - tid_len);
  }

  void prepare(GenericConcurrentHopscotch *hopscotch_ptr) {
    for (uint32_t i = 0; i < helpers::kNumCPUs; i++) {
      std::random_device rd;
      generators[i].reset(new std::mt19937(rd()));
    }

    std::vector<rt::Thread> threads;
    for (uint32_t tid = 0; tid < kNumMutatorThreads; tid++) {
      threads.emplace_back(rt::Thread([&, tid]() {
        auto num_kv_pairs = kNumKVPairs / kNumMutatorThreads;
        if (tid == kNumMutatorThreads - 1) {
          num_kv_pairs += kNumKVPairs % kNumMutatorThreads;
        }
        auto deref_scope_ptr = std::make_unique<DerefScope>();
        auto *thread_gen_keys =
            &all_gen_keys[tid * (kNumKVPairs / kNumMutatorThreads)];
        Key key;
        Value val;
        for (uint32_t i = 0; i < num_kv_pairs; i++) {
          if (unlikely(i % kNumItersPerScope == 0)) {
            deref_scope_ptr.reset();
            deref_scope_ptr.reset(new DerefScope());
          }
          random_key(key.data, tid);
          random_string(val.data, kValueLen);
          hopscotch_ptr->put(*deref_scope_ptr, kKeyLen,
                             (const uint8_t *)key.data, kValueLen,
                             (const uint8_t *)val.data);
          thread_gen_keys[i] = key;
        }
      }));
    }
    for (auto &thread : threads) {
      thread.Join();
    }
    preempt_disable();
    zipf_table_distribution<> zipf(kNumKVPairs, kZipfParamS);
    auto &generator = generators[get_core_num()];
    for (uint32_t i = 0; i < kReqSeqLenPerCore; i++) {
      auto idx = zipf(*generator);
      BUG_ON(idx >= kNumKVPairs);
      all_zipf_key_indices[0][i] = idx;
    }
    for (uint32_t k = 1; k < helpers::kNumCPUs; k++) {
      memcpy(all_zipf_key_indices[k], all_zipf_key_indices[0],
             sizeof(uint32_t) * kReqSeqLenPerCore);
    }
    preempt_enable();
  }

  void monitor_perf() {
    if (!flag.test_and_set()) {
      auto us = microtime();
      if (us - prev_us > kMinMonitorIntervalUs) {
        uint64_t sum_cnts = 0;
        for (uint32_t i = 0; i < kNumMutatorThreads; i++) {
          sum_cnts += ACCESS_ONCE(cnts[i].c);
        }
        us = microtime();
        auto mops = (double)(sum_cnts - prev_sum_cnts) / (us - prev_us);
        mops_vec.push_back(mops);
        running_us += (us - prev_us);
        if (running_us >= kMaxRunningUs) {
          std::vector<double> last_5_mops(
              mops_vec.end() - std::min(static_cast<int>(mops_vec.size()), 5),
              mops_vec.end());
          std::cout << "mops = "
                    << std::accumulate(last_5_mops.begin(), last_5_mops.end(),
                                       0.0) /
                           last_5_mops.size()
                    << std::endl;
          std::cout << "Done. Force exiting..." << std::endl;
          exit(0);
        }
        prev_us = us;
        prev_sum_cnts = sum_cnts;
      }
      flag.clear();
    }
  }

  void bench_get(GenericConcurrentHopscotch *hopscotch_ptr) {
    prev_us = microtime();
    std::vector<rt::Thread> threads;
    for (uint32_t tid = 0; tid < kNumMutatorThreads; tid++) {
      threads.emplace_back(rt::Thread([&, tid]() {
        auto deref_scope_ptr = std::make_unique<DerefScope>();
        uint32_t cnt = 0;
        while (1) {
          if (unlikely(cnt++ % kNumItersPerScope == 0)) {
            deref_scope_ptr.reset();
            deref_scope_ptr.reset(new DerefScope());
          }
          preempt_disable();
          if (unlikely(cnt % kMonitorPerIter == 0)) {
            monitor_perf();
          }
          auto key_idx =
              all_zipf_key_indices[get_core_num()][per_core_req_idx++];
          if (unlikely(per_core_req_idx == kReqSeqLenPerCore)) {
            per_core_req_idx = 0;
          }
          preempt_enable();
          auto &key = all_gen_keys[key_idx];
          uint16_t val_len;
          Value val;
          hopscotch_ptr->get(*deref_scope_ptr, kKeyLen,
                             (const uint8_t *)key.data, &val_len,
                             (uint8_t *)val.data);
          ACCESS_ONCE(cnts[tid].c)++;
          DONT_OPTIMIZE(val);
        }
      }));
    }
    for (auto &thread : threads) {
      thread.Join();
    }
  }

public:
  void run(netaddr raddr) {
    BUG_ON(madvise(all_gen_keys, sizeof(all_gen_keys), MADV_HUGEPAGE) != 0);
    manager.reset(FarMemManagerFactory::build(
        kCacheSize, kNumGCThreads,
        new TCPDevice(raddr, kNumConnections, kFarMemSize)));
    auto hopscotch = std::unique_ptr<GenericConcurrentHopscotch>(
        manager->allocate_concurrent_hopscotch_heap(
            kLocalHashTableNumEntriesShift, kRemoteHashTableNumEntriesShift,
            kRemoteHashTableDataSize));
    std::cout << "Prepare..." << std::endl;
    prepare(hopscotch.get());
    std::cout << "Get..." << std::endl;
    bench_get(hopscotch.get());
    hopscotch.reset();
    manager.reset();
  }
};
} // namespace far_memory

int argc;
FarMemTest test;
void my_main(void *arg) {
  char **argv = (char **)arg;
  std::string ip_addr_port(argv[1]);
  test.run(helpers::str_to_netaddr(ip_addr_port));
}

int main(int _argc, char *argv[]) {
  int ret;

  if (_argc < 3) {
    std::cerr << "usage: [cfg_file] [ip_addr:port]" << std::endl;
    return -EINVAL;
  }

  char conf_path[strlen(argv[1]) + 1];
  strcpy(conf_path, argv[1]);
  for (int i = 2; i < _argc; i++) {
    argv[i - 1] = argv[i];
  }
  argc = _argc - 1;

  ret = runtime_init(conf_path, my_main, argv);
  if (ret) {
    std::cerr << "failed to start runtime" << std::endl;
    return ret;
  }

  return 0;
}
