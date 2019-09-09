#ifndef SIMDJSON_WESTMERE_SIMD_BITMASK_H
#define SIMDJSON_WESTMERE_SIMD_BITMASK_H

#include "simdjson/common_defs.h"
#include "simdjson/portability.h"
#include "westmere/architecture.h"
#include "westmere/bitmask_array.h"

#ifdef IS_X86_64

TARGET_WESTMERE
namespace simdjson::westmere {

struct simd_bitmask {
  __m128i bitmask;

  really_inline simd_bitmask() { }
  really_inline simd_bitmask(const __m128i _bitmask) : bitmask(_bitmask) { }

  really_inline simd_bitmask(const uint16_t b0, const uint16_t b1, const uint16_t b2, const uint16_t b3, const uint16_t b4, const uint16_t b5, const uint16_t b6, const uint16_t b7)
    : simd_bitmask(_mm_setr_epi16(b0, b1, b2, b3, b4, b5, b6, b7)) { }
  really_inline simd_bitmask(const __m128i i0, const __m128i i1, const __m128i i2, const __m128i i3, const __m128i i4, const __m128i i5, const __m128i i6, const __m128i i7)
    : simd_bitmask(
        _mm_movemask_epi8(i0),
        _mm_movemask_epi8(i1),
        _mm_movemask_epi8(i2),
        _mm_movemask_epi8(i3),
        _mm_movemask_epi8(i4),
        _mm_movemask_epi8(i5),
        _mm_movemask_epi8(i6),
        _mm_movemask_epi8(i7)
      ) { }
  really_inline operator __m128i() const { return this->bitmask; }

  really_inline simd_bitmask(bitmask_array b) : simd_bitmask(_mm_loadu_si128(reinterpret_cast<__m128i*>(b.bitmasks))) { }
  really_inline simd_bitmask(const uint64_t b0, const uint64_t b1) : simd_bitmask(bitmask_array(b0,b1)) { }

  really_inline bitmask_array to_array() const {
    bitmask_array result;
    _mm_storeu_si128(reinterpret_cast<__m128i*>(result.bitmasks), this->bitmask);
    return result;
  }

  // Bitwise operations
  really_inline simd_bitmask operator |(const simd_bitmask other) const {
    return _mm_or_si128(*this, other);
  }
  really_inline simd_bitmask operator &(const simd_bitmask other) const {
    return _mm_and_si128(*this, other);
  }
  really_inline simd_bitmask operator ^(const simd_bitmask other) const {
    return _mm_xor_si128(*this, other);
  }
  really_inline simd_bitmask andnot(const simd_bitmask other) const {
    return _mm_andnot_si128(other, *this);
  }
  really_inline simd_bitmask operator ~() const {
    return _mm_and_si128(*this, _mm_set1_epi64((__m64)0UL));
  }
  really_inline simd_bitmask ornot(const simd_bitmask other) {
    return *this | ~other;
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
  really_inline bool any_bits_set(const simd_bitmask bits) const {
    return _mm_testz_si128(*this, bits) > 0;
  }
  really_inline bool any_bits_not_set(const simd_bitmask bits) const {
    return _mm_testnzc_si128(*this, bits) > 0;
  }
  really_inline bool any_bits_set() const {
    return this->any_bits_set(_mm_set1_epi8(0xFF));
  }
  really_inline bool any_bits_not_set() const {
    return this->any_bits_not_set(_mm_set1_epi8(0xFF));
  }

  really_inline simd_bitmask prev(bool &carry) const {
    // Do the main rotation up (left) one bit
    simd_bitmask shifted = _mm_slli_epi64(*this, 1);

    // Grab the carry bits, move them up (left) a spot, and bring in the previous carry
    simd_bitmask carry_out = _mm_srli_epi64(*this, 63);
    simd_bitmask carried = _mm_shuffle_epi32(carry_out, _MM_SHUFFLE(1, 0, 3, 2));
    carried = _mm_insert_epi64(carried, carry, 0);

    // Figure out if there's a carry-out
    carry = this->any_bits_set(_mm_set_epi32(0x80000000UL, 0, 0, 0));

    // Return the shifted and carried bits together
    return shifted | carried;
  }

  really_inline simd_bitmask after_series_starting_with(simd_bitmask starting_with, bool &carry) const {
    return this->to_array().after_series_starting_with(starting_with.to_array(), carry);
  }
};

really_inline simd_bitmask splat_u8 (uint8_t  value) { return _mm_set1_epi8(value); }
really_inline simd_bitmask splat_u16(uint16_t value) { return _mm_set1_epi16(value); }
really_inline simd_bitmask splat_u32(uint32_t value) { return _mm_set1_epi32(value); }
really_inline simd_bitmask splat_u64(uint64_t value) { return _mm_set1_epi64x(value); }

} // namespace simdjson::westmere
UNTARGET_REGION // westmere

#endif // IS_X86_64
#endif // SIMDJSON_WESTMERE_SIMD_BITMASK_H
