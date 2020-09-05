#pragma once

extern "C" {
#include <base/compiler.h>
#include <base/limits.h>
#include <runtime/rcu.h>
#include <runtime/thread.h>
}

#include "helpers.hpp"

#include <memory>
#include <type_traits>
#include <vector>

// Only count InScopeV0 and InScopeV1. OutofScope and GC is not counted.
extern __thread int num_threads_on_status[3]; // Defined at runtime/ksched.c
extern int *num_threads_on_status_ptrs[NCPU]; // Defined at runtime/ksched.c

namespace far_memory {

enum Status { OutofScope = 0, InScopeV0, InScopeV1, GC };
static_assert(GC == GC_STATUS); // GC_STATUS is defined and used by Shenango
                                // runtime for performance accounting.

extern bool almost_empty;
extern bool gc_master_active;
extern Status expected_status;

class DerefScope {
private:
  NOT_COPYABLE(DerefScope);
  NOT_MOVEABLE(DerefScope);
  static void mutator_wait_for_gc_cache();
  static int32_t get_num_threads(Status status);
  static bool is_status_expected();

  friend class FarMemManager;
  friend class GenericConcurrentHopscotch;

public:
  DerefScope();
  ~DerefScope();
  static void enter_deref_scope();
  static void exit_deref_scope();
  static bool is_in_deref_scope();
};

FORCE_INLINE DerefScope::DerefScope() { enter_deref_scope(); }

FORCE_INLINE DerefScope::~DerefScope() { exit_deref_scope(); }

FORCE_INLINE void DerefScope::enter_deref_scope() {
  // Disallow nested DerefScopes.
  assert(!is_in_deref_scope());
  // This is necessary for the low-speed far memory, since the swapout speed
  // will always be lower than mutator's dereferencing speed.
  // 1st (mov), [2nd (test) + 3rd (jne)] (1 macro ops) fast-path instructions.
  if (very_unlikely(ACCESS_ONCE(almost_empty))) {
    mutator_wait_for_gc_cache();
  }
  // 4rd (mov) fast-path instruction.
  auto snapshot = ACCESS_ONCE(expected_status);
  // 5th (mov) fast-path instruction.
  set_self_th_status(snapshot);
  // Intel SDM Vol.3 Sec.6.6 confirms that interrupt can only be observed
  // at the instruction boundary. So we can use a single instruction to
  // update the per-CPU counter without disabling the preemption.
  // 6th fast-path instruction.
  __asm__("incl %0" : "=m"(num_threads_on_status[snapshot]));
  barrier();
}

FORCE_INLINE void DerefScope::exit_deref_scope() {
  barrier();
  assert(get_self_th_status() != OutofScope);
  // 1st (mov) fast-path instruction.
  auto old_th_status = get_self_th_status();
  // 2nd fast-path instruction.
  __asm__("decl %0" : "=m"(num_threads_on_status[old_th_status]));
  // 3rd (mov) fast-path instruction.
  set_self_th_status(OutofScope);
  // 4th (cmp) + 5th (jne) fast-path instructions -> 1 macro op.
  if (very_unlikely(old_th_status != expected_status)) {
    if (likely(gc_master_active)) {
      // It used to be the culprit thread but now it has exited the scope so it
      // no longer blocks GC. Yield itself for other prioritized threads.
      barrier();
      thread_yield();
    }
  }
}

FORCE_INLINE bool DerefScope::is_in_deref_scope() {
  return get_self_th_status() != OutofScope;
}

FORCE_INLINE bool DerefScope::is_status_expected() {
  return get_self_th_status() == ACCESS_ONCE(expected_status);
}

FORCE_INLINE int32_t DerefScope::get_num_threads(Status status) {
  int32_t sum = 0;
  for (uint8_t i = 0; i < helpers::kNumCPUs; i++) {
    int *ptr = ACCESS_ONCE(num_threads_on_status_ptrs[i]);
    if (ptr) {
      sum += ACCESS_ONCE(*(ptr + (int)status));
    }
  }
  assert(sum >= 0);
  return sum;
}

static Status flip_status(Status status) {
  assert(status != OutofScope);
  return (status == InScopeV0) ? InScopeV1 : InScopeV0;
}

} // namespace far_memory
