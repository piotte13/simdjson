#ifndef SIMDJSON_WESTMERE_STAGE1_FIND_MARKS_H
#define SIMDJSON_WESTMERE_STAGE1_FIND_MARKS_H

#include "simdjson/portability.h"

#ifdef IS_X86_64

#include "westmere/bitmask.h"
#include "westmere/simd.h"
#include "simdjson/stage1_find_marks.h"

TARGET_WESTMERE
namespace simdjson::westmere {

using namespace simd;

really_inline void find_whitespace_and_operators(
  const simd8x64<uint8_t> in,
  uint64_t &whitespace, uint64_t &op) {

  // These lookups rely on the fact that anything < 127 will match the lower 4 bits, which is why
  // we can't use the generic lookup_16.
  auto whitespace_table = simd8<uint8_t>::repeat_16(' ', 100, 100, 100, 17, 100, 113, 2, 100, '\t', '\n', 112, 100, '\r', 100, 100);
  auto op_table = simd8<uint8_t>::repeat_16(',', '}', 0, 0, 0xc0u, 0, 0, 0, 0, 0, 0, 0, 0, 0, ':', '{');

  whitespace = in.map([&](simd8<uint8_t> _in) {
    return _in == simd8<uint8_t>(_mm_shuffle_epi8(whitespace_table, _in));
  }).to_bitmask();

  op = in.map([&](simd8<uint8_t> _in) {
    // | 32 handles the fact that { } and [ ] are exactly 32 bytes apart
    return (_in | 32) == simd8<uint8_t>(_mm_shuffle_epi8(op_table, _in-','));
  }).to_bitmask();
}

#include "generic/utf8_zwegner_algorithm.h"
#include "generic/stage1_find_marks.h"

} // namespace westmere
UNTARGET_REGION

TARGET_WESTMERE
namespace simdjson {

template <>
int find_structural_bits<Architecture::WESTMERE>(const uint8_t *buf, size_t len, simdjson::ParsedJson &pj, bool streaming) {
  return westmere::stage1::find_structural_bits<64>(buf, len, pj, streaming);
}

} // namespace simdjson
UNTARGET_REGION

#endif // IS_X86_64
#endif // SIMDJSON_WESTMERE_STAGE1_FIND_MARKS_H
