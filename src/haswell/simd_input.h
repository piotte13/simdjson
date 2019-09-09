#ifndef SIMDJSON_HASWELL_SIMD_INPUT_H
#define SIMDJSON_HASWELL_SIMD_INPUT_H

#include "simdjson/common_defs.h"
#include "simdjson/portability.h"
#include "haswell/architecture.h"
#include "haswell/simd_bitmask.h"

#ifdef IS_X86_64

TARGET_HASWELL
namespace simdjson::haswell {

struct simd_input64 {
  // Number of SIMD registers necessary to store 64 bytes
  __m256i chunks[64/SIMD_BYTE_WIDTH];

  really_inline simd_input64(const uint8_t *ptr) {
    this->chunks[0] = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr + 0*SIMD_BYTE_WIDTH));
    this->chunks[1] = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr + 1*SIMD_BYTE_WIDTH));
  }

  really_inline simd_input64(const __m256i chunk0, const __m256i chunk1) {
    this->chunks[0] = chunk0;
    this->chunks[1] = chunk1;
  }

  really_inline operator uint64_t() const {
      return
        (static_cast<uint64_t>(static_cast<uint32_t>(_mm256_movemask_epi8(this->chunks[0]))) << 0*SIMD_BYTE_WIDTH) |
        (static_cast<uint64_t>(static_cast<uint32_t>(_mm256_movemask_epi8(this->chunks[1]))) << 1*SIMD_BYTE_WIDTH);
  }

  template <typename F>
  really_inline void each(F const& each) const {
    each(this->chunks[0]);
    each(this->chunks[1]);
  }

  template <typename F>
  really_inline simd_input64 map(F const& map) const {
    return simd_input64(
      map(this->chunks[0]),
      map(this->chunks[1])
    );
  }

  template <typename F>
  really_inline simd_input64 map(const simd_input64 b, F const& map) const {
    return simd_input64(
      map(this->chunks[0], b.chunks[0]),
      map(this->chunks[1], b.chunks[1])
    );
  }

  template <typename F>
  really_inline __m256i reduce(F const& reduce_pair) const {
    return reduce_pair(this->chunks[0], this->chunks[1]);
  }

  really_inline uint64_t eq(const uint8_t m) const {
    const __m256i mask = _mm256_set1_epi8(m);
    return this->map([&](auto a) { return _mm256_cmpeq_epi8(a, mask); });
  }

  really_inline uint64_t lteq(const uint8_t m) const {
    const __m256i maxval = _mm256_set1_epi8(m);
    return this->map([&](auto a) { return _mm256_cmpeq_epi8(_mm256_max_epu8(maxval, a), maxval); });
  }

}; // struct simd_input64

struct simd_input {
  simd_input64 chunks[SIMD_WIDTH/64];
  really_inline simd_input(const uint8_t *in_buf) : chunks({
    simd_input64(in_buf+0*64),
    simd_input64(in_buf+1*64),
    simd_input64(in_buf+2*64),
    simd_input64(in_buf+3*64)
  }) { }
  really_inline simd_input64 operator[](size_t index) const { return this->chunks[index]; }
};

}
UNTARGET_REGION

#endif // IS_X86_64
#endif // SIMDJSON_HASWELL_SIMD_INPUT_H
