#ifndef SIMDJSON_ARM64_SIMD_INPUT_H
#define SIMDJSON_ARM64_SIMD_INPUT_H

#include "simdjson/common_defs.h"
#include "simdjson/portability.h"
#include "arm64/architecture.h"
#include "arm64/simd_bitmask.h"

#ifdef IS_ARM64

namespace simdjson::arm64 {

struct simd_input64 {
  // Number of SIMD registers necessary to store 64 bytes
  uint8x16_t chunks[64/SIMD_BYTE_WIDTH];

  really_inline simd_input64(const uint8_t *ptr) {
    this->chunks[0] = vld1q_u8(ptr + 0*SIMD_BYTE_WIDTH);
    this->chunks[1] = vld1q_u8(ptr + 1*SIMD_BYTE_WIDTH);
    this->chunks[2] = vld1q_u8(ptr + 2*SIMD_BYTE_WIDTH);
    this->chunks[3] = vld1q_u8(ptr + 3*SIMD_BYTE_WIDTH);
  }

  really_inline simd_input64(uint8x16_t chunk0, uint8x16_t chunk1, uint8x16_t chunk2, uint8x16_t chunk3) {
    this->chunks[0] = chunk0;
    this->chunks[1] = chunk1;
    this->chunks[2] = chunk2;
    this->chunks[3] = chunk3;
  }

  really_inline operator uint64_t() {
    const uint8x16_t bit_mask = {0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80,
                                0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80};
    uint8x16_t sum = vpaddq_u8(
      vpaddq_u8(
        vandq_u8(this->chunks[0], bit_mask),
        vandq_u8(this->chunks[1], bit_mask)
      ),
      vpaddq_u8(
        vandq_u8(this->chunks[2], bit_mask),
        vandq_u8(this->chunks[3], bit_mask)
      )
    );
    uint8x16_t combined = vpaddq_u8(sum, sum);
    return vgetq_lane_u64(vreinterpretq_u64_u8(combined), 0);
  }

  template <typename F>
  really_inline void each(F const& each)
  {
    each(this->chunks[0]);
    each(this->chunks[1]);
    each(this->chunks[2]);
    each(this->chunks[3]);
  }

  template <typename F>
  really_inline simd_input64 map(F const& map) {    
    auto r0 = map(this->chunks[0]);
    auto r1 = map(this->chunks[1]);
    auto r2 = map(this->chunks[2]);
    auto r3 = map(this->chunks[3]);
    return simd_input64(r0,r1,r2,r3);
  }

  template <typename F>
  really_inline simd_input64 map(simd_input64 b, F const& map) {
    auto r0 = map(this->chunks[0], b.chunks[0]);
    auto r1 = map(this->chunks[1], b.chunks[1]);
    auto r2 = map(this->chunks[2], b.chunks[2]);
    auto r3 = map(this->chunks[3], b.chunks[3]);
    return simd_input64(r0,r1,r2,r3);
  }

  template <typename F>
  really_inline uint8x16_t reduce(F const& reduce_pair) {
    uint8x16_t r01 = reduce_pair(this->chunks[0], this->chunks[1]);
    uint8x16_t r23 = reduce_pair(this->chunks[2], this->chunks[3]);
    return reduce_pair(r01, r23);
  }

  really_inline uint64_t eq(uint8_t m) {
    const uint8x16_t mask = vmovq_n_u8(m);
    return this->map( [&](auto a) {
      return vceqq_u8(a, mask);
    });
  }

  really_inline uint64_t lteq(uint8_t m) {
    const uint8x16_t mask = vmovq_n_u8(m);
    return this->map([&](auto a) { return vcleq_u8(a, mask); });
  }

}; // struct simd_input64

struct simd_input {
  simd_input64 chunks[SIMD_WIDTH/64];
  really_inline simd_input(const uint8_t *in_buf) : chunks({
    simd_input64(in_buf+0*64),
    simd_input64(in_buf+1*64)
  }) { }
  really_inline simd_input64 operator[](size_t index) const { return this->chunks[index]; }
};

} // namespace simdjson::arm64

#endif // IS_ARM64
#endif // SIMDJSON_ARM64_SIMD_INPUT_H
