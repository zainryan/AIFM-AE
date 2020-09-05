#pragma once

#include "helpers.hpp"
#include "object.hpp"

#include <cstdint>
#include <memory>
#include <optional>

namespace far_memory {

class Region {
  // Format:
  // |ref_cnt(4B)|Nt(1B)|objects|
  //
  //    ref_cnt: The region can only be GCed when the ref_cnt goes to 0.
  //         Nt: is this region a non-temporal?
  //    objects: objects stored within the region.
public:
  constexpr static uint32_t kRefCntPos = 0;
  constexpr static uint32_t kRefCntSize = 4;
  constexpr static uint32_t kNtPos = 4;
  constexpr static uint32_t kNtSize = 1;
  constexpr static uint32_t kObjectPos = 5;
  constexpr static uint64_t kShift = 20;
  constexpr static uint64_t kSize = (1 << kShift);
  constexpr static uint8_t kGCParallelism = 2;
  constexpr static int32_t kInvalidIdx = -1;
  constexpr static uint32_t kHeaderSize = kRefCntSize + kNtSize;

  static_assert(kSize <= helpers::kHugepageSize);
  static_assert(helpers::kHugepageSize % kSize == 0);

private:
  uint32_t first_free_byte_idx_ = kObjectPos;
  uint8_t *buf_ptr_ = nullptr;
  int32_t region_idx_ = kInvalidIdx;
  uint8_t num_boundaries_ = 0;
  uint32_t gc_boundaries_[kGCParallelism];

  void update_boundaries(bool force);

public:
  Region();
  Region(uint32_t idx, bool is_local, bool nt, uint8_t *buf_ptr);
  NOT_COPYABLE(Region);
  Region(Region &&other);
  Region &operator=(Region &&other);
  ~Region();
  std::optional<uint64_t> allocate_object(uint16_t object_size);
  bool is_invalid() const;
  void invalidate();
  void reset();
  bool is_local() const;
  bool is_nt() const;
  void set_nt();
  void clear_nt();
  uint32_t get_ref_cnt() const;
  void clear_ref_cnt();
  bool is_gcable() const;
  uint8_t get_num_boundaries() const;
  std::pair<uint64_t, uint64_t> get_boundary(uint8_t idx) const;
  void atomic_inc_ref_cnt(int32_t delta);
  static bool is_nt(uint64_t buf_ptr_addr);
  static void atomic_inc_ref_cnt(uint64_t object_addr, int32_t delta);
};

FORCE_INLINE Region::Region() {}

FORCE_INLINE Region::~Region() {}

FORCE_INLINE Region::Region(Region &&other) { *this = std::move(other); }

FORCE_INLINE bool Region::is_invalid() const {
  return region_idx_ == kInvalidIdx;
}

FORCE_INLINE void Region::invalidate() { region_idx_ = kInvalidIdx; }

FORCE_INLINE void Region::reset() {
  first_free_byte_idx_ = kObjectPos;
  num_boundaries_ = 0;
  clear_nt();
}

FORCE_INLINE bool Region::is_local() const { return buf_ptr_; }

FORCE_INLINE void Region::update_boundaries(bool force) {
  if (force || unlikely(first_free_byte_idx_ >
                        kSize / kGCParallelism * (num_boundaries_ + 1))) {
    gc_boundaries_[num_boundaries_++] = first_free_byte_idx_;
  }
}

FORCE_INLINE uint8_t Region::get_num_boundaries() const {
  return num_boundaries_;
}

FORCE_INLINE std::pair<uint64_t, uint64_t>
Region::get_boundary(uint8_t idx) const {
  assert(idx < num_boundaries_);
  auto left_offset = (idx == 0) ? kHeaderSize : gc_boundaries_[idx - 1];
  auto right_offset = gc_boundaries_[idx];
  return std::make_pair(reinterpret_cast<uint64_t>(buf_ptr_) + left_offset,
                        reinterpret_cast<uint64_t>(buf_ptr_) + right_offset);
}

FORCE_INLINE uint32_t Region::get_ref_cnt() const {
  return ACCESS_ONCE(*reinterpret_cast<uint32_t *>(buf_ptr_ + kRefCntPos));
}

FORCE_INLINE void Region::clear_ref_cnt() {
  ACCESS_ONCE(*reinterpret_cast<uint32_t *>(buf_ptr_ + kRefCntPos)) = 0;
}

FORCE_INLINE void Region::atomic_inc_ref_cnt(uint64_t object_addr,
                                             int32_t delta) {
  auto region_addr = (object_addr) & (~(Region::kSize - 1));
  auto ref_cnt_ptr = (reinterpret_cast<uint8_t *>(region_addr)) + kRefCntPos;
  __atomic_add_fetch(reinterpret_cast<int32_t *>(ref_cnt_ptr), delta,
                     __ATOMIC_SEQ_CST);
}

FORCE_INLINE void Region::atomic_inc_ref_cnt(int32_t delta) {
  auto ref_cnt_ptr = buf_ptr_ + kRefCntPos;
  __atomic_add_fetch(reinterpret_cast<int32_t *>(ref_cnt_ptr), delta,
                     __ATOMIC_SEQ_CST);
}

FORCE_INLINE bool Region::is_gcable() const { return get_ref_cnt() == 0; }

FORCE_INLINE bool Region::is_nt() const {
  return ACCESS_ONCE(*reinterpret_cast<uint8_t *>(buf_ptr_ + kNtPos));
}

FORCE_INLINE bool Region::is_nt(uint64_t buf_ptr_addr) {
  return ACCESS_ONCE(*reinterpret_cast<uint8_t *>(buf_ptr_addr + kNtPos));
}

FORCE_INLINE void Region::set_nt() {
  ACCESS_ONCE(*reinterpret_cast<uint8_t *>(buf_ptr_ + kNtPos)) = 1;
}

FORCE_INLINE void Region::clear_nt() {
  ACCESS_ONCE(*reinterpret_cast<uint8_t *>(buf_ptr_ + kNtPos)) = 0;
}

} // namespace far_memory
