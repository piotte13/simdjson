#ifndef SIMDJSON_HASWELL_STRINGPARSING_H
#define SIMDJSON_HASWELL_STRINGPARSING_H

#ifdef IS_X86_64

#include "simdjson/common_defs.h"
#include "simdjson/jsoncharutils.h"
#include "simdjson/parsedjson.h"

#ifdef JSON_TEST_STRINGS
void found_string(const uint8_t *buf, const uint8_t *parsed_begin,
                  const uint8_t *parsed_end);
void found_bad_string(const uint8_t *buf);
#endif

TARGET_HASWELL
namespace simdjson::haswell {

// Holds backslashes and quotes locations.
struct bs_and_quote_bits {
  // We store 64 bits so that we can safely shift right 32-36 bits without invoking undefined behavior.
  uint64_t bs_bits;
  uint64_t quote_bits;
  static const int SCAN_WIDTH = sizeof(__m256i);

  really_inline void consume(unsigned int consumed) {
    this->bs_bits >>= consumed;
    this->quote_bits >>= consumed;
  }

  really_inline bool has_backslash_in_string() {
    auto backslashes_before_quotes = ((this->quote_bits - 1) & this->bs_bits);
    return backslashes_before_quotes != 0;
  }

  really_inline bool has_backslash() {
    return this->bs_bits != 0;
  }

  really_inline bool has_quote() {
    return this->quote_bits != 0;
  }

  really_inline unsigned int next_backslash() {
    return trailing_zeroes(this->bs_bits);
  }

  really_inline unsigned int next_quote() {
    return trailing_zeroes(this->quote_bits);
  }
};

really_inline bs_and_quote_bits find_bs_and_quote_bits(const uint8_t *src, uint8_t *dst) {
  // this can read up to 31 bytes beyond the buffer size, but we require
  // SIMDJSON_PADDING of padding
  static_assert(sizeof(__m256i) - 1 <= SIMDJSON_PADDING);
  __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src));
  // store to dest unconditionally - we can overwrite the bits we don't like
  // later
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), v);
  auto quote_mask = _mm256_cmpeq_epi8(v, _mm256_set1_epi8('"'));
  return {
      static_cast<uint32_t>(_mm256_movemask_epi8(
          _mm256_cmpeq_epi8(v, _mm256_set1_epi8('\\')))),     // bs_bits
      static_cast<uint32_t>(_mm256_movemask_epi8(quote_mask)) // quote_bits
  };
}

#include "generic/stringparsing.h"

} // namespace simdjson::haswell
UNTARGET_REGION

#endif // IS_X86_64

#endif
