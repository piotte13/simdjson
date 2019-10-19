#ifndef SIMDJSON_WESTMERE_SIMD_INPUT_H
#define SIMDJSON_WESTMERE_SIMD_INPUT_H

#include "simdjson/common_defs.h"
#include "simdjson/portability.h"
#include "simdjson/simdjson.h"

#ifdef IS_X86_64

TARGET_WESTMERE
namespace simdjson::westmere {

// SIMD type, just so we can copy/paste more easily between architectures.
typedef __m128i simd_t;
// Forward-declared so they can be used by splat and friends.
struct simd_m8;
struct simd_u8;
struct simd_i8;
// Output of simd_m8.to_bitmask(). uint32_t for 32-byte SIMD registers, uint16_t for 16-byte.
typedef uint16_t simd_m8_bitmask;

struct simd_base8 {
  simd_t value;

  static const int SIZE = sizeof(value);

  // Zero constructor
  really_inline simd_base8() : value{} {}
  // Conversion from SIMD register
  really_inline simd_base8(const simd_t _value) : value(_value) {}
  // Conversion to const SIMD register
  really_inline operator const simd_t&() const { return this->value; }
  // Conversion to non-const SIMD register
  really_inline operator simd_t&() { return this->value; }
};

// SIMD byte mask type (returned by things like eq and gt)
struct simd_m8: simd_base8 {
  using simd_base8::simd_base8; // Pull in base constructors

  static really_inline simd_m8 splat(bool _value) { return _mm_set1_epi64x(-uint64_t(_value)); }

  really_inline bool any() { return this->to_bitmask(); }

  really_inline simd_m8 operator ||(const simd_m8& other) const { return _mm_or_si128(*this, other); }
  really_inline simd_m8 operator &&(const simd_m8& other) const { return _mm_and_si128(*this, other); }
  really_inline simd_m8 logical_xor(const simd_m8& other) const { return _mm_xor_si128(*this, other); }
  really_inline simd_m8 logical_andnot(const simd_m8& other) const { return _mm_andnot_si128(*this, other); }
  really_inline simd_m8 operator !() const { return this->logical_xor(*this); }

  really_inline simd_m8_bitmask to_bitmask() const { return _mm_movemask_epi8(*this); }
  really_inline bool any() const { return _mm_testz_si128(*this, *this) == 0; }

  really_inline simd_m8 operator==(const simd_m8& other) const { return _mm_cmpeq_epi8(*this, other); }
};

// Signed bytes
struct simd_i8 : simd_base8 {
  using simd_base8::simd_base8; // Pull in base constructors

  // Read from buffer
  static really_inline simd_i8 load(const int8_t *_value) {
    return _mm_loadu_si128(reinterpret_cast<const simd_t *>(_value));
  }

  static really_inline simd_i8 splat(int8_t _value) { return _mm_set1_epi8(_value); }
  static really_inline simd_i8 zero() { return _mm_setzero_si128(); }

  really_inline simd_i8 max(const simd_i8& other) const { return _mm_max_epi8(*this, other); }
  really_inline simd_i8 min(const simd_i8& other) const { return _mm_min_epi8(*this, other); }

  really_inline simd_i8 operator +(const simd_i8& other) const { return _mm_add_epi8(*this, other); }
  really_inline simd_i8 operator -(const simd_i8& other) const { return _mm_sub_epi8(*this, other); }

  really_inline simd_m8 operator==(const simd_i8& other) const { return _mm_cmpeq_epi8(*this, other); }
  really_inline simd_m8 operator >(const simd_i8& other) const { return _mm_cmpgt_epi8(*this, other); }
};

// Unsigned bytes
struct simd_u8: simd_base8 {
  using simd_base8::simd_base8; // Pull in base constructors

  // Read from buffer
  static really_inline simd_u8 load(const uint8_t *_value) {
    return _mm_loadu_si128(reinterpret_cast<const simd_t *>(_value));
  }

  static really_inline simd_u8 splat(uint8_t _value) { return _mm_set1_epi8(_value); }
  static really_inline simd_u8 zero() { return _mm_setzero_si128(); }

  really_inline simd_u8 operator >>(const int count) const { return _mm_srli_epi16(*this, count); }
  really_inline simd_u8 operator <<(const int count) const { return _mm_slli_epi16(*this, count); }

  really_inline simd_u8 operator |(const simd_u8& other) const { return _mm_or_si128(*this, other); }
  really_inline simd_u8 operator &(const simd_u8& other) const { return _mm_and_si128(*this, other); }
  really_inline simd_u8 operator ^(const simd_u8& other) const { return _mm_xor_si128(*this, other); }
  really_inline simd_u8 bit_andnot(const simd_u8& other) const { return _mm_andnot_si128(*this, other); }
  really_inline simd_u8 operator ~() const { return *this ^ *this; }

  really_inline simd_u8 max(const simd_u8& other) const { return _mm_max_epu8(*this, other); }
  really_inline simd_u8 min(const simd_u8& other) const { return _mm_min_epu8(*this, other); }

  really_inline simd_u8 operator +(const simd_u8& other) const { return _mm_add_epi8(*this, other); }
  really_inline simd_u8 operator -(const simd_u8& other) const { return _mm_sub_epi8(*this, other); }
  really_inline simd_u8 saturated_add(const simd_u8& other) const { return _mm_adds_epu8(*this, other); }
  really_inline simd_u8 saturated_sub(const simd_u8& other) const { return _mm_subs_epu8(*this, other); }

  really_inline simd_m8 operator==(const simd_u8& other) const { return _mm_cmpeq_epi8(*this, other); }
  really_inline simd_m8 operator <=(const simd_u8& other) const { return this->max(other) == other; }
};

struct simd_u8x64 {
  const simd_u8 chunks[2];

  really_inline simd_u8x64() : chunks{simd_u8(), simd_u8()} {}

  really_inline simd_u8x64(const simd_t chunk0, const simd_t chunk1) : chunks{chunk0, chunk1} {}

  really_inline simd_u8x64(const uint8_t *ptr) : chunks{simd_u8::load(ptr), simd_u8::load(ptr+32)} {}

  template <typename F>
  really_inline void each(F const& each_chunk) const
  {
    each_chunk(this->chunks[0]);
    each_chunk(this->chunks[1]);
  }

  template <typename F>
  really_inline simd_u8x64 map(F const& map_chunk) const {
    return simd_u8x64(
      map_chunk(this->chunks[0]),
      map_chunk(this->chunks[1])
    );
  }

  template <typename F>
  really_inline simd_u8x64 map(const simd_u8x64 b, F const& map_chunk) const {
    return simd_u8x64(
      map_chunk(this->chunks[0], b.chunks[0]),
      map_chunk(this->chunks[1], b.chunks[1])
    );
  }

  template <typename F>
  really_inline simd_u8 reduce(F const& reduce_pair) const {
    return reduce_pair(this->chunks[0], this->chunks[1]);
  }

  really_inline uint64_t to_bitmask() const {
    uint64_t r_lo = static_cast<uint32_t>(simd_m8(this->chunks[0]).to_bitmask());
    uint64_t r_hi =                       simd_m8(this->chunks[1]).to_bitmask();
    return r_lo | (r_hi << 32);
  }

  really_inline simd_u8x64 bit_or(const uint8_t m) const {
    const simd_u8 mask = simd_u8::splat(m);
    return this->map( [&](auto a) { return a | mask; } );
  }

  really_inline uint64_t eq(const uint8_t m) const {
    const simd_u8 mask = simd_u8::splat(m);
    return this->map( [&](auto a) { return a == mask; } ).to_bitmask();
  }

  really_inline uint64_t lteq(const uint8_t m) const {
    const simd_u8 mask = simd_u8::splat(m);
    return this->map( [&](auto a) { return a <= mask; } ).to_bitmask();
  }

}; // struct simd_u8x64

typedef simd_u8x64 simd_input;

} // namespace simdjson::westmere
UNTARGET_REGION

#endif // IS_X86_64
#endif // SIMDJSON_WESTMERE_SIMD_INPUT_H
