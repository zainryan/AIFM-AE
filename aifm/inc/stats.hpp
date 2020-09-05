#pragma once

extern "C" {
#include <runtime/runtime.h>
}

#include "helpers.hpp"

#include <cstdint>

namespace far_memory {

struct alignas(64) Cacheline {
  uint8_t data[64];
};

class Stats {
private:
  static bool enable_swap_;
#ifdef MONITOR_FREE_MEM_RATIO
  static std::vector<std::pair<uint64_t, double>>
      free_mem_ratio_records_[helpers::kNumCPUs];
#endif

#ifdef MONITOR_READ_OBJECT_CYCLES
  static unsigned read_object_cycles_high_start_;
  static unsigned read_object_cycles_low_start_;
  static unsigned read_object_cycles_high_end_;
  static unsigned read_object_cycles_low_end_;
#endif

#ifdef MONITOR_WRITE_OBJECT_CYCLES
  static unsigned write_object_cycles_high_start_;
  static unsigned write_object_cycles_low_start_;
  static unsigned write_object_cycles_high_end_;
  static unsigned write_object_cycles_low_end_;
#endif

  static void _add_free_mem_ratio_record();

public:
  static void enable_swap() { enable_swap_ = true; }
  static void disable_swap() { enable_swap_ = false; }
  static void clear_free_mem_ratio_records();
  static void print_free_mem_ratio_records();

#define ADD_STAT(type, x, enable_flag)                                         \
private:                                                                       \
  static type x##_;                                                            \
                                                                               \
public:                                                                        \
  FORCE_INLINE static void inc_##x(type num) {                                 \
    if (enable_flag) {                                                         \
      ACCESS_ONCE(x##_) += num;                                                \
    }                                                                          \
  }                                                                            \
  FORCE_INLINE static type get_##x() { return ACCESS_ONCE(x##_); }

#define ADD_PER_CORE_STAT(type, x, enable_flag)                                \
private:                                                                       \
  static Cacheline x##_[helpers::kNumCPUs];                                    \
                                                                               \
public:                                                                        \
  FORCE_INLINE static void inc_##x(type num) {                                 \
    if (enable_flag) {                                                         \
      preempt_disable();                                                       \
      ACCESS_ONCE(*((type *)(x##_ + get_core_num()))) += num;                  \
      preempt_enable();                                                        \
    }                                                                          \
  }                                                                            \
  FORCE_INLINE static type get_##x() {                                         \
    type sum = 0;                                                              \
    FOR_ALL_SOCKET0_CORES(i) { sum += ACCESS_ONCE(*((type *)(x##_ + i))); }    \
    return sum;                                                                \
  }

  FORCE_INLINE static uint64_t get_schedule_us() {
    uint64_t sum = 0;
    FOR_ALL_SOCKET0_CORES(i) { sum += ACCESS_ONCE(duration_schedule_us[i].c); }
    return sum;
  }

  FORCE_INLINE static uint64_t get_softirq_us() {
    uint64_t sum = 0;
    FOR_ALL_SOCKET0_CORES(i) { sum += ACCESS_ONCE(duration_softirq_us[i].c); }
    return sum;
  }

  FORCE_INLINE static uint64_t get_gc_us() {
    uint64_t sum = 0;
    FOR_ALL_SOCKET0_CORES(i) { sum += ACCESS_ONCE(duration_gc_us[i].c); }
    return sum;
  }

  FORCE_INLINE static uint64_t get_tcp_rw_bytes() {
    return get_tcp_tx_bytes() + get_tcp_rx_bytes();
  }

  FORCE_INLINE static void add_free_mem_ratio_record() {
#ifdef MONITOR_FREE_MEM_RATIO
    _add_free_mem_ratio_record();
#endif
  }

  FORCE_INLINE static void start_measure_read_object_cycles() {
#ifdef MONITOR_READ_OBJECT_CYCLES
    helpers::timer_start(&read_object_cycles_high_start_,
                         &read_object_cycles_low_start_);
#endif
  }

  FORCE_INLINE static void finish_measure_read_object_cycles() {
#ifdef MONITOR_READ_OBJECT_CYCLES
    helpers::timer_end(&read_object_cycles_high_end_,
                       &read_object_cycles_low_end_);
#endif
  }

  FORCE_INLINE static void reset_measure_read_object_cycles() {
#ifdef MONITOR_READ_OBJECT_CYCLES
    read_object_cycles_high_start_ = read_object_cycles_low_start_ =
        read_object_cycles_high_end_ = read_object_cycles_low_end_;
#endif
  }

  FORCE_INLINE static uint64_t get_elapsed_read_object_cycles() {
#ifdef MONITOR_READ_OBJECT_CYCLES
    return helpers::get_elapsed_cycles(
        read_object_cycles_high_start_, read_object_cycles_low_start_,
        read_object_cycles_high_end_, read_object_cycles_low_end_);
#else
    return 0;
#endif
  }

  FORCE_INLINE static void start_measure_write_object_cycles() {
#ifdef MONITOR_WRITE_OBJECT_CYCLES
    helpers::timer_start(&write_object_cycles_high_start_,
                         &write_object_cycles_low_start_);
#endif
  }

  FORCE_INLINE static void finish_measure_write_object_cycles() {
#ifdef MONITOR_WRITE_OBJECT_CYCLES
    helpers::timer_end(&write_object_cycles_high_end_,
                       &write_object_cycles_low_end_);
#endif
  }

  FORCE_INLINE static void reset_measure_write_object_cycles() {
#ifdef MONITOR_WRITE_OBJECT_CYCLES
    write_object_cycles_high_start_ = write_object_cycles_low_start_ =
        write_object_cycles_high_end_ = write_object_cycles_low_end_;
#endif
  }

  FORCE_INLINE static uint64_t get_elapsed_write_object_cycles() {
#ifdef MONITOR_WRITE_OBJECT_CYCLES
    return helpers::get_elapsed_cycles(
        write_object_cycles_high_start_, write_object_cycles_low_start_,
        write_object_cycles_high_end_, write_object_cycles_low_end_);
#else
    return 0;
#endif
  }
};
} // namespace far_memory
