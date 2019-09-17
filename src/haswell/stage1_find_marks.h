#ifndef SIMDJSON_HASWELL_STAGE1_FIND_MARKS_H
#define SIMDJSON_HASWELL_STAGE1_FIND_MARKS_H

#include "simdjson/portability.h"

#ifdef IS_X86_64

#include "haswell/architecture.h"
#include "haswell/simd_input.h"
#include "haswell/simdutf8check.h"
#include "simdjson/stage1_find_marks.h"

TARGET_HASWELL
namespace simdjson::haswell {

really_inline uint64_t compute_quote_mask(const uint64_t quote_bits) {
  // There should be no such thing with a processing supporting avx2
  // but not clmul.
  uint64_t quote_mask = _mm_cvtsi128_si64(_mm_clmulepi64_si128(
      _mm_set_epi64x(0ULL, quote_bits), _mm_set1_epi8(0xFFu), 0));
  return quote_mask;
}

really_inline uint64_t find_whitespace(const simd_input<ARCHITECTURE> in) {
  // clang-format off
  const __m256i white_table = _mm256_setr_epi8(
      32, 100, 100, 100, 17, 100, 113, 2, 100, 9, 10, 112, 100, 13, 100, 100,
      32, 100, 100, 100, 17, 100, 113, 2, 100, 9, 10, 112, 100, 13, 100, 100);
  // clang-format on
  return in.map([&](auto _in) {
    return _mm256_cmpeq_epi8(_in, _mm256_shuffle_epi8(white_table, _in));
  }).to_bitmask();
}

// flatten out values in 'bits' assuming that they are are to have values of idx
// plus their position in the bitvector, and store these indexes at
// base_ptr[base] incrementing base as we go
// will potentially store extra values beyond end of valid bits, so base_ptr
// needs to be large enough to handle this
really_inline void flatten_bits(uint32_t *base_ptr, uint32_t &base, uint32_t idx, uint64_t bits) {
  // In some instances, the next branch is expensive because it is mispredicted.
  // Unfortunately, in other cases,
  // it helps tremendously.
  if (bits == 0)
      return;
  uint32_t cnt = _mm_popcnt_u64(bits);
  uint32_t next_base = base + cnt;
  idx -= 64;
  base_ptr += base;
  {
      base_ptr[0] = idx + trailing_zeroes(bits);
      bits = _blsr_u64(bits);
      base_ptr[1] = idx + trailing_zeroes(bits);
      bits = _blsr_u64(bits);
      base_ptr[2] = idx + trailing_zeroes(bits);
      bits = _blsr_u64(bits);
      base_ptr[3] = idx + trailing_zeroes(bits);
      bits = _blsr_u64(bits);
      base_ptr[4] = idx + trailing_zeroes(bits);
      bits = _blsr_u64(bits);
      base_ptr[5] = idx + trailing_zeroes(bits);
      bits = _blsr_u64(bits);
      base_ptr[6] = idx + trailing_zeroes(bits);
      bits = _blsr_u64(bits);
      base_ptr[7] = idx + trailing_zeroes(bits);
      bits = _blsr_u64(bits);
      base_ptr += 8;
  }
  // We hope that the next branch is easily predicted.
  if (cnt > 8) {
      base_ptr[0] = idx + trailing_zeroes(bits);
      bits = _blsr_u64(bits);
      base_ptr[1] = idx + trailing_zeroes(bits);
      bits = _blsr_u64(bits);
      base_ptr[2] = idx + trailing_zeroes(bits);
      bits = _blsr_u64(bits);
      base_ptr[3] = idx + trailing_zeroes(bits);
      bits = _blsr_u64(bits);
      base_ptr[4] = idx + trailing_zeroes(bits);
      bits = _blsr_u64(bits);
      base_ptr[5] = idx + trailing_zeroes(bits);
      bits = _blsr_u64(bits);
      base_ptr[6] = idx + trailing_zeroes(bits);
      bits = _blsr_u64(bits);
      base_ptr[7] = idx + trailing_zeroes(bits);
      bits = _blsr_u64(bits);
      base_ptr += 8;
  }
  if (cnt > 16) { // unluckly: we rarely get here
      // since it means having one structural or pseudo-structral element
      // every 4 characters (possible with inputs like "","","",...).
      do {
      base_ptr[0] = idx + trailing_zeroes(bits);
      bits = _blsr_u64(bits);
      base_ptr++;
      } while (bits != 0);
  }
  base = next_base;
}

#include "generic/stage1_find_marks.h"

} // namespace haswell
UNTARGET_REGION

TARGET_HASWELL
namespace simdjson {

template <>
int find_structural_bits<Architecture::HASWELL>(const uint8_t *buf, size_t len, simdjson::ParsedJson &pj) {
  return haswell::find_structural_bits(buf, len, pj);
}

} // namespace simdjson
UNTARGET_REGION

#endif // IS_X86_64
#endif // SIMDJSON_HASWELL_STAGE1_FIND_MARKS_H
