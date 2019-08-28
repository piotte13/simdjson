#ifndef SIMDJSON_HASWELL_SIMD_BITMASK_H
#define SIMDJSON_HASWELL_SIMD_BITMASK_H

#include "simdjson/common_defs.h"
#include "simdjson/portability.h"
#include "haswell/architecture.h"
#include "haswell/bitmask_array.h"

#ifdef IS_X86_64

TARGET_HASWELL
namespace simdjson::haswell {

struct simd_bitmask {
  __m256i bitmask;

  really_inline simd_bitmask(__m256i _bitmask) : bitmask(_bitmask) { }
  really_inline operator __m256i() const { return this->bitmask; }

  really_inline simd_bitmask(uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3, uint32_t b4, uint32_t b5, uint32_t b6, uint32_t b7)
    : simd_bitmask(_mm256_setr_epi32(b0, b1, b2, b3, b4, b5, b6, b7)) { }
  really_inline simd_bitmask(__m256i i0, __m256i i1, __m256i i2, __m256i i3, __m256i i4, __m256i i5, __m256i i6, __m256i i7)
    : simd_bitmask(
        _mm256_movemask_epi8(i0),
        _mm256_movemask_epi8(i1),
        _mm256_movemask_epi8(i2),
        _mm256_movemask_epi8(i3),
        _mm256_movemask_epi8(i4),
        _mm256_movemask_epi8(i5),
        _mm256_movemask_epi8(i6),
        _mm256_movemask_epi8(i7)
      ) { }

  really_inline simd_bitmask(bitmask_array b) : simd_bitmask(_mm256_loadu_si256(reinterpret_cast<__m256i*>(b.bitmasks))) { }
  really_inline simd_bitmask(uint64_t b0, uint64_t b1, uint64_t b2, uint64_t b3) : simd_bitmask(bitmask_array(b0,b1,b2,b3)) { }

  really_inline bitmask_array chunks64() const {
    bitmask_array result;
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(result.bitmasks), this->bitmask);
    return result;
  }


  // Bitwise operations
  really_inline simd_bitmask operator |(const simd_bitmask other) const {
    return _mm256_or_si256(*this, other);
  }
  really_inline simd_bitmask operator &(const simd_bitmask other) const {
    return _mm256_and_si256(*this, other);
  }
  really_inline simd_bitmask operator ^(const simd_bitmask other) const {
    return _mm256_xor_si256(*this, other);
  }
  really_inline simd_bitmask andnot(const simd_bitmask other) const {
    return _mm256_andnot_si256(other, *this);
  }
  really_inline simd_bitmask operator ~() const {
    return _mm256_and_si256(*this, _mm256_set1_epi64x(0));
  }
  really_inline simd_bitmask ornot(const simd_bitmask other) const {
    return *this | ~other;
  }
  really_inline bool bits_set(const simd_bitmask bits) const {
    return _mm256_testz_si256(*this, bits) > 0;
  }
  really_inline bool bits_not_set(const simd_bitmask bits) const {
    return _mm256_testnzc_si256(*this, bits) > 0;
  }

  really_inline simd_bitmask prev(bool &carry) const {
    // Do the main rotation forward (left) one bit
    simd_bitmask shifted = _mm256_slli_epi64(*this, 1);

    // Grab the carry bits, move them forward (left) a spot, and bring in the previous carry
    simd_bitmask carry_out = _mm256_srli_epi64(*this, 63);
    simd_bitmask carried = _mm256_permute4x64_epi64(carry_out, _MM_SHUFFLE(2, 1, 0, 3));
    carried = _mm256_insert_epi64(carried, carry, 0);

    // Figure out if there's a carry-out
    carry = (_mm256_extract_epi32(*this, 0) & 0x80000000UL) > 0;

    // Return the shifted and carried bits together
    return shifted | carried;
  }

  // really_inline bitmask_array after_series_starting_with(bitmask_array starting_with, bool &carry) const {
    // // First, add up the slots.
    // simd_bitmask added = _mm256_add_epi64(*this, preceded_by);

    // // We need to know if any of the slots overflowed by checking if added is smaller than original.
    // // Since SSE/AVX only have signed comparisons, we have to subtract MAXINT first :/ XOR high-bit does that.
    // const simd_bitmask high_bit = splat_u64(0x8000000000000000ULL);
    // simd_bitmask overflowed = _mm256_cmpgt_epi64(*this ^ high_bit, added ^ high_bit);
    // simd_bitmask carries = _mm256_permute4x64_epi64(overflowed, _MM_SHUFFLE(2, 1, 0, 3)); // Shuffle overflow forward (left)
    // uint64_t prev_carry = carry * uint64_t(-1);
    // carry = _mm256_extract_epi32(carries, 0) > 0; // Get the new carry
    // carries = _mm256_insert_epi64(carries, prev_carry, 0); // Move the previous carry in
    // carries = _mm256_and_si256(carries, splat_u64(1)); // Set carry values to 1 instead of -1 (FFFF...)

    // // Before we can add in the carries, we need to check if any of the carries will skip its boundaries.
    // uint32_t mask = _mm256_movemask_epi8(carries);
    // if (mask & )
  // }
};

really_inline simd_bitmask splat_u8 (uint8_t  value) { return _mm256_set1_epi8(value); }
really_inline simd_bitmask splat_u16(uint16_t value) { return _mm256_set1_epi16(value); }
really_inline simd_bitmask splat_u32(uint32_t value) { return _mm256_set1_epi32(value); }
really_inline simd_bitmask splat_u64(uint64_t value) { return _mm256_set1_epi64x(value); }

}
UNTARGET_REGION

#endif // IS_X86_64
#endif // SIMDJSON_HASWELL_SIMD_BITMASK_H
