#ifndef SIMDJSON_ARM64_SIMD_BITMASK_H
#define SIMDJSON_ARM64_SIMD_BITMASK_H

#include "simdjson/common_defs.h"
#include "simdjson/portability.h"
#include "arm64/architecture.h"
#include "arm64/bitmask_array.h"

#ifdef IS_ARM64

namespace simdjson::arm64 {

simd_bitmask splat_u8 (uint8_t value) { return vmovq_n_u8(value); }
simd_bitmask splat_u16(uint16_t value) { return vmovq_n_u16(value); }
simd_bitmask splat_u32(uint32_t value) { return vmovq_n_u32(value); }
simd_bitmask splat_u64(uint64_t value) { return vmovq_n_u64(value); }

struct simd_bitmask {
  uint64x2_t bitmask;

  really_inline simd_bitmask() { }
  really_inline simd_bitmask(uint64x2_t _bitmask) : bitmask(_bitmask) { }
  really_inline operator uint64x2_t() const { return this->bitmask; }

  really_inline simd_bitmask(uint16_t b0, uint16_t b1, uint16_t b2, uint16_t b3, uint16_t b4, uint16_t b5, uint16_t b6, uint16_t b7)
    : simd_bitmask(vld1q_u16({b0, b1, b2, b3, b4, b5, b6, b7})) { }
  really_inline simd_bitmask(uint8x16_t i0, uint8x16_t i1, uint8x16_t i2, uint8x16_t i3, uint8x16_t i4, uint8x16_t i5, uint8x16_t i6, uint8x16_t i7) {
    const uint8x16_t bit_mask = {0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80,
                                 0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80};
    uint8x16_t sum = vpaddq_u8(
      vpaddq_u8(
        vpaddq_u8(
          vandq_u8(i0, bit_mask),
          vandq_u8(i1, bit_mask)
        ),
        vpaddq_u8(
          vandq_u8(i2, bit_mask),
          vandq_u8(i3, bit_mask)
        )
      ),
      vpaddq_u8(
        vpaddq_u8(
          vandq_u8(i4, bit_mask),
          vandq_u8(i5, bit_mask)
        ),
        vpaddq_u8(
          vandq_u8(i6, bit_mask),
          vandq_u8(i7, bit_mask)
        ),
      ),
    );
    return vreinterpretq_u64_u8(sum);
  }

  really_inline simd_bitmask(bitmask_array b) {
    vst1q_u64(b.bitmasks, this->bitmask);
  }
  really_inline simd_bitmask(uint64_t b0, uint64_t b1) : simd_bitmask(bitmask_array(b0,b1)) { }

  really_inline bitmask_array to_array() const {
    bitmask_array result;
    vld1q_u64(result.bitmasks, this->bitmask);
    return result;
  }

  // Bitwise operations
  really_inline simd_bitmask operator |(const simd_bitmask &other) const {
    return vorq_u64(*this, other);
  }
  really_inline simd_bitmask operator &(const simd_bitmask &other) const {
    return vandq_u64(*this, other);
  }
  really_inline simd_bitmask operator ^(const simd_bitmask &other) const {
    return vxorq_u64(*this, other);
  }
  really_inline simd_bitmask ornot(const simd_bitmask &other) const {
    return vnorq_u64(*this, other);
  }
  really_inline simd_bitmask operator ~() const {
    return vnegq_u64(*this);
  }
  really_inline simd_bitmask andnot(const simd_bitmask &other) const {
    return *this & ~other;
  }
  really_inline simd_bitmask operator |=(const simd_bitmask other) {
    return (*this = *this | other);
  }
  really_inline simd_bitmask operator &=(const simd_bitmask other) {
    return (*this = *this & other);
  }
  really_inline simd_bitmask operator ^=(const simd_bitmask other) {
    return (*this = *this ^ other);
  }

  really_inline simd_bitmask prev(bool &carry) const {
    return this->to_array().prev(carry);
  }
  really_inline simd_bitmask after_series_starting_with(simd_bitmask starting_with, bool &carry) const {
    return this->to_array().after_series_starting_with(starting_with.to_array(), carry);
  }
}

} // namespace simdjson::arm64

#endif // IS_ARM64
#endif // SIMDJSON_ARM64_SIMD_BITMASK_H
