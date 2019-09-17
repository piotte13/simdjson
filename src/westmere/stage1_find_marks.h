#ifndef SIMDJSON_WESTMERE_STAGE1_FIND_MARKS_H
#define SIMDJSON_WESTMERE_STAGE1_FIND_MARKS_H

#include "simdjson/portability.h"

#ifdef IS_X86_64

#include "westmere/architecture.h"
#include "westmere/simd_input.h"
#include "westmere/simdutf8check.h"
#include "simdjson/stage1_find_marks.h"

TARGET_WESTMERE
namespace simdjson::westmere {

really_inline uint64_t compute_quote_mask(const uint64_t quote_bits) {
  return _mm_cvtsi128_si64(_mm_clmulepi64_si128(
      _mm_set_epi64x(0ULL, quote_bits), _mm_set1_epi8(0xFFu), 0));
}

really_inline uint64_t find_whitespace(const simd_input<ARCHITECTURE> in) {
  const __m128i white_table = _mm_setr_epi8(32, 100, 100, 100,  17, 100, 113,   2,
                                           100,   9,  10, 112, 100,  13, 100, 100);

  return in.map([&](auto _in) {
    return _mm_cmpeq_epi8(_in, _mm_shuffle_epi8(white_table, _in));
  }).to_bitmask();
}

#include "generic/stage1_find_marks_flatten.h"
#include "generic/stage1_find_marks.h"

} // namespace westmere
UNTARGET_REGION

TARGET_WESTMERE
namespace simdjson {

template <>
int find_structural_bits<Architecture::WESTMERE>(const uint8_t *buf, size_t len, simdjson::ParsedJson &pj) {
  return westmere::find_structural_bits(buf, len, pj);
}

} // namespace simdjson
UNTARGET_REGION

#endif // IS_X86_64
#endif // SIMDJSON_WESTMERE_STAGE1_FIND_MARKS_H
