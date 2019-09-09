#ifndef SIMDJSON_WESTMERE_SIMD_INPUT_H
#define SIMDJSON_WESTMERE_SIMD_INPUT_H

#include "simdjson/common_defs.h"
#include "simdjson/portability.h"
#include "westmere/architecture.h"
#include "westmere/simd_bitmask.h"

#ifdef IS_X86_64

TARGET_WESTMERE
namespace simdjson::westmere {

struct simd_input64 {
  // Number of SIMD registers necessary to store 64 bytes
  __m128i chunks[64/SIMD_BYTE_WIDTH];

  really_inline simd_input64(const uint8_t *ptr) {
    this->chunks[0] = _mm_loadu_si128(reinterpret_cast<const __m128i *>(ptr + 0*SIMD_BYTE_WIDTH));
    this->chunks[1] = _mm_loadu_si128(reinterpret_cast<const __m128i *>(ptr + 1*SIMD_BYTE_WIDTH));
    this->chunks[2] = _mm_loadu_si128(reinterpret_cast<const __m128i *>(ptr + 2*SIMD_BYTE_WIDTH));
    this->chunks[3] = _mm_loadu_si128(reinterpret_cast<const __m128i *>(ptr + 3*SIMD_BYTE_WIDTH));
  }

  really_inline simd_input64(__m128i i0, __m128i i1, __m128i i2, __m128i i3)
  {
    this->chunks[0] = i0;
    this->chunks[1] = i1;
    this->chunks[2] = i2;
    this->chunks[3] = i3;
  }

  really_inline operator uint64_t() {
      return
        (static_cast<uint64_t>(static_cast<uint16_t>(_mm_movemask_epi8(this->chunks[0]))) << 0*SIMD_BYTE_WIDTH) |
        (static_cast<uint64_t>(static_cast<uint16_t>(_mm_movemask_epi8(this->chunks[1]))) << 1*SIMD_BYTE_WIDTH) |
        (static_cast<uint64_t>(static_cast<uint16_t>(_mm_movemask_epi8(this->chunks[2]))) << 2*SIMD_BYTE_WIDTH) |
        (static_cast<uint64_t>(static_cast<uint16_t>(_mm_movemask_epi8(this->chunks[3]))) << 3*SIMD_BYTE_WIDTH);
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
  really_inline __m128i reduce(F const& reduce_pair) {
    __m128i r01 = reduce_pair(this->chunks[0], this->chunks[1]);
    __m128i r23 = reduce_pair(this->chunks[2], this->chunks[3]);
    return reduce_pair(r01, r23);
  }

  really_inline uint64_t eq(uint8_t m) {
    const __m128i mask = _mm_set1_epi8(m);
    return this->map([&](auto a) { return _mm_cmpeq_epi8(a, mask); });
  }

  really_inline uint64_t lteq(uint8_t m) {
    const __m128i maxval = _mm_set1_epi8(m);
    return this->map([&](auto a) { return _mm_cmpeq_epi8(_mm_max_epu8(maxval, a), maxval); });
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

} // namespace simdjson::westmere
UNTARGET_REGION // westmere

#endif // IS_X86_64
#endif // SIMDJSON_WESTMERE_SIMD_INPUT_H
