extern "C" {
#include <runtime/runtime.h>
}

#include "array.hpp"
#include "device.hpp"
#include "x264.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>

using namespace far_memory;
using namespace std;

constexpr uint64_t kCacheSize = (1ULL << 30);
constexpr uint64_t kFarMemSize = (20ULL << 30);
constexpr uint32_t kNumGCThreads = 4;
constexpr uint64_t kNumEntries = (16ULL << 30);
constexpr uint32_t kChunkSize = (1 << 16);

uint8_t raw[kChunkSize];
uint8_t another_raw[kChunkSize];

void do_work(void *arg) {
  init_fm();

  int64_t fm_idx = 0;
  while (fm_idx < (int64_t)kNumEntries) {
    for (uint32_t i = 0; i < kChunkSize; i++) {
      raw[i] = fm_idx + i;
    }
    dram_to_fm(raw, fm_idx, kChunkSize);
    fm_idx += kChunkSize;
  }
  flush_cache();
  fm_idx = 0;

  auto start = chrono::steady_clock::now();
  decltype(start) end;
  while (fm_idx < (int64_t)kNumEntries) {
    fm_to_dram(raw, fm_idx, kChunkSize);
    fm_idx += kChunkSize;
  }
  end = chrono::steady_clock::now();
  cout << "Elapsed time in microseconds : "
       << chrono::duration_cast<chrono::microseconds>(end - start).count()
       << " µs" << endl;

  fm_idx = 0;
  while (fm_idx < (int64_t)kNumEntries) {
    fm_to_dram(raw, fm_idx, kChunkSize);
    for (uint32_t i = 0; i < kChunkSize; i++) {
      if (raw[i] != (uint8_t)(fm_idx + i)) {
        goto fail;
      }
    }
    fm_idx += kChunkSize;
  }

  cout << "Passed" << endl;
  return;

fail:
  cout << "Failed" << endl;
  return;
}

int main(int argc, char *argv[]) {
  int ret;

  if (argc < 2) {
    std::cerr << "usage: [cfg_file]" << std::endl;
    return -EINVAL;
  }

  ret = runtime_init(argv[1], do_work, NULL);
  if (ret) {
    std::cerr << "failed to start runtime" << std::endl;
    return ret;
  }

  return 0;
}
