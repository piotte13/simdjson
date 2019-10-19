#ifndef SIMDJSON_HASWELL_SIMD_INPUT_H
#define SIMDJSON_HASWELL_SIMD_INPUT_H

#include "simdjson/common_defs.h"
#include "simdjson/portability.h"
#include "simdjson/simdjson.h"

#ifdef IS_X86_64

TARGET_HASWELL
namespace simdjson::haswell {

// SIMD type, just so we can copy/paste more easily between architectures.
typedef __m256i simd_t;
// Forward-declared so they can be used by splat and friends.
struct simd_m8;
struct simd_u8;
struct simd_i8;
// Output of simd_m8.to_bitmask(). uint32_t for 32-byte SIMD registers, uint16_t for 16-byte.
typedef uint32_t simd_m8_bitmask;

struct simd_base8 {
  simd_t value;

  static const int SIZE = sizeof(value);

  // Zero constructor
  really_inline simd_base8() : value{} {}
  // Conversion from SIMD register
  really_inline simd_base8(const simd_t _value) : value(_value) {}
  really_inline simd_base8(
    uint8_t v0,  uint8_t v1,  uint8_t v2,  uint8_t v3,
    uint8_t v4,  uint8_t v5,  uint8_t v6,  uint8_t v7,
    uint8_t v8,  uint8_t v9,  uint8_t v10, uint8_t v11,
    uint8_t v12, uint8_t v13, uint8_t v14, uint8_t v15,
    uint8_t v16, uint8_t v17, uint8_t v18, uint8_t v19,
    uint8_t v20, uint8_t v21, uint8_t v22, uint8_t v23,
    uint8_t v24, uint8_t v25, uint8_t v26, uint8_t v27,
    uint8_t v28, uint8_t v29, uint8_t v30, uint8_t v31
  ) : value(_mm256_setr_epi8(
    v0, v1, v2, v3, v4, v5, v6, v7,
    v8, v9, v10,v11,v12,v13,v14,v15,
    v16,v17,v18,v19,v20,v21,v22,v23,
    v24,v25,v26,v27,v28,v29,v30,v31
  )) {}
  // Conversion to const SIMD register
  really_inline operator const simd_t&() const { return this->value; }
  // Conversion to non-const SIMD register
  really_inline operator simd_t&() { return this->value; }
};

// SIMD byte mask type (returned by things like eq and gt)
struct simd_m8: simd_base8 {
  static really_inline simd_m8 splat(bool _value) { return _mm256_set1_epi64x(-uint64_t(_value)); }

  really_inline simd_m8() : simd_base8() {}
  really_inline simd_m8(simd_t _value) : simd_base8(_value) {}
  really_inline simd_m8(bool _value) : simd_base8(splat(_value)) {}

  really_inline simd_m8 operator||(const simd_m8 other) const { return _mm256_or_si256(*this, other); }
  really_inline simd_m8 operator&&(const simd_m8 other) const { return _mm256_and_si256(*this, other); }
  really_inline simd_m8 logical_xor(const simd_m8 other) const { return _mm256_xor_si256(*this, other); }
  really_inline simd_m8 logical_andnot(const simd_m8 other) const { return _mm256_andnot_si256(*this, other); }
  really_inline simd_m8 operator!() const { return this->logical_xor(*this); }
  really_inline simd_m8 operator|(const simd_m8 other) const { return *this || other; }
  really_inline simd_m8 operator&(const simd_m8 other) const { return *this && other; }
  really_inline simd_m8& operator|=(const simd_m8 other) { *this = *this | other; return *this; }
  really_inline simd_m8& operator&=(const simd_m8 other) { *this = *this & other; return *this; }

  really_inline simd_m8_bitmask to_bitmask() const { return _mm256_movemask_epi8(*this); }

  really_inline simd_m8 operator==(const simd_m8 other) const { return _mm256_cmpeq_epi8(*this, other); }

  really_inline bool any() const { return _mm256_testz_si256(*this, *this) == 0; }
};

// Signed bytes
struct simd_i8 : simd_base8 {
  static really_inline simd_i8 splat(int8_t _value) { return _mm256_set1_epi8(_value); }
  static really_inline simd_i8 zero() { return _mm256_setzero_si256(); }

  really_inline simd_i8() : simd_base8() {}
  really_inline simd_i8(simd_t _value) : simd_base8(_value) {}
  really_inline simd_i8(int8_t _value) : simd_base8(splat(_value)) {}
  really_inline simd_i8(
    int8_t v0,  int8_t v1,  int8_t v2,  int8_t v3,
    int8_t v4,  int8_t v5,  int8_t v6,  int8_t v7,
    int8_t v8,  int8_t v9,  int8_t v10, int8_t v11,
    int8_t v12, int8_t v13, int8_t v14, int8_t v15,
    int8_t v16, int8_t v17, int8_t v18, int8_t v19,
    int8_t v20, int8_t v21, int8_t v22, int8_t v23,
    int8_t v24, int8_t v25, int8_t v26, int8_t v27,
    int8_t v28, int8_t v29, int8_t v30, int8_t v31
  ) : simd_base8(
    v0, v1, v2, v3, v4, v5, v6, v7,
    v8, v9, v10,v11,v12,v13,v14,v15,
    v16,v17,v18,v19,v20,v21,v22,v23,
    v24,v25,v26,v27,v28,v29,v30,v31
  ) {}

  // Read from buffer
  static really_inline simd_i8 load(const int8_t *_value) {
    return _mm256_loadu_si256(reinterpret_cast<const simd_t *>(_value));
  }

  really_inline simd_i8 max(const simd_i8 other) const { return _mm256_max_epi8(*this, other); }
  really_inline simd_i8 min(const simd_i8 other) const { return _mm256_min_epi8(*this, other); }

  really_inline simd_i8 operator+(const simd_i8 other) const { return _mm256_add_epi8(*this, other); }
  really_inline simd_i8 operator-(const simd_i8 other) const { return _mm256_sub_epi8(*this, other); }
  really_inline simd_i8& operator+=(const simd_i8 other) { *this = *this + other; return *this; }
  really_inline simd_i8& operator-=(const simd_i8 other) { *this = *this - other; return *this; }

  really_inline simd_m8 operator==(const simd_i8 other) const { return _mm256_cmpeq_epi8(*this, other); }
  really_inline simd_m8 operator>(const simd_i8 other) const { return _mm256_cmpgt_epi8(*this, other); }

  // Perform a lookup of the lower 4 bits
  really_inline simd_i8 lookup4(
      int8_t lookup0,  int8_t lookup1,  int8_t lookup2,  int8_t lookup3,
      int8_t lookup4,  int8_t lookup5,  int8_t lookup6,  int8_t lookup7,
      int8_t lookup8,  int8_t lookup9,  int8_t lookup10, int8_t lookup11,
      int8_t lookup12, int8_t lookup13, int8_t lookup14, int8_t lookup15) const {

    simd_i8 lookup_table(
      lookup0, lookup1, lookup2,  lookup3,  lookup4,  lookup5,  lookup6,  lookup7,
      lookup8, lookup9, lookup10, lookup11, lookup12, lookup13, lookup14, lookup15,
      lookup0, lookup1, lookup2,  lookup3,  lookup4,  lookup5,  lookup6,  lookup7,
      lookup8, lookup9, lookup10, lookup11, lookup12, lookup13, lookup14, lookup15
    );
    return _mm256_shuffle_epi8(lookup_table, *this);
  }

  really_inline simd_i8 prev(simd_i8& prev_chunk) const {
    return _mm256_alignr_epi8(*this, _mm256_permute2x128_si256(prev_chunk, *this, 0x21), 15);
  }
  really_inline simd_i8 prev2(const simd_i8 prev_chunk) const {
    return _mm256_alignr_epi8(*this, _mm256_permute2x128_si256(prev_chunk, *this, 0x21), 14);
  }
};

// Unsigned bytes
struct simd_u8: simd_base8 {
  static really_inline simd_u8 splat(uint8_t _value) { return _mm256_set1_epi8(_value); }
  static really_inline simd_u8 zero() { return _mm256_setzero_si256(); }

  really_inline simd_u8() : simd_base8() {}
  really_inline simd_u8(simd_t _value) : simd_base8(_value) {}
  really_inline simd_u8(uint8_t _value) : simd_base8(splat(_value)) {}
  really_inline simd_u8(
    uint8_t v0,  uint8_t v1,  uint8_t v2,  uint8_t v3,
    uint8_t v4,  uint8_t v5,  uint8_t v6,  uint8_t v7,
    uint8_t v8,  uint8_t v9,  uint8_t v10, uint8_t v11,
    uint8_t v12, uint8_t v13, uint8_t v14, uint8_t v15,
    uint8_t v16, uint8_t v17, uint8_t v18, uint8_t v19,
    uint8_t v20, uint8_t v21, uint8_t v22, uint8_t v23,
    uint8_t v24, uint8_t v25, uint8_t v26, uint8_t v27,
    uint8_t v28, uint8_t v29, uint8_t v30, uint8_t v31
  ) : simd_base8(
    v0, v1, v2, v3, v4, v5, v6, v7,
    v8, v9, v10,v11,v12,v13,v14,v15,
    v16,v17,v18,v19,v20,v21,v22,v23,
    v24,v25,v26,v27,v28,v29,v30,v31
  ) {}

  // Read from buffer
  static really_inline simd_u8 load(const uint8_t *_value) {
    return _mm256_loadu_si256(reinterpret_cast<const simd_t *>(_value));
  }

  really_inline simd_u8 operator>>(const int count) const { return _mm256_srli_epi16(*this, count); }
  really_inline simd_u8 operator<<(const int count) const { return _mm256_slli_epi16(*this, count); }

  really_inline simd_u8 operator|(const simd_u8 other) const { return _mm256_or_si256(*this, other); }
  really_inline simd_u8 operator&(const simd_u8 other) const { return _mm256_and_si256(*this, other); }
  really_inline simd_u8 operator^(const simd_u8 other) const { return _mm256_xor_si256(*this, other); }
  really_inline simd_u8 bit_andnot(const simd_u8 other) const { return _mm256_andnot_si256(*this, other); }
  really_inline simd_u8 operator~() const { return *this ^ *this; }
  really_inline simd_u8& operator|=(const simd_u8 other) { *this = *this | other; return *this; }
  really_inline simd_u8& operator&=(const simd_u8 other) { *this = *this & other; return *this; }
  really_inline simd_u8& operator^=(const simd_u8 other) { *this = *this ^ other; return *this; }

  really_inline simd_u8 max(const simd_u8 other) const { return _mm256_max_epu8(*this, other); }
  really_inline simd_u8 min(const simd_u8 other) const { return _mm256_min_epu8(*this, other); }

  really_inline simd_u8 operator+(const simd_u8 other) const { return _mm256_add_epi8(*this, other); }
  really_inline simd_u8 operator-(const simd_u8 other) const { return _mm256_sub_epi8(*this, other); }
  really_inline simd_u8& operator+=(const simd_u8 other) { *this = *this + other; return *this; }
  really_inline simd_u8& operator-=(const simd_u8 other) { *this = *this - other; return *this; }
  really_inline simd_u8 saturated_add(const simd_u8 other) const { return _mm256_adds_epu8(*this, other); }
  really_inline simd_u8 saturated_sub(const simd_u8 other) const { return _mm256_subs_epu8(*this, other); }

  really_inline simd_m8 operator==(const simd_u8 other) const { return _mm256_cmpeq_epi8(*this, other); }
  really_inline simd_m8 operator<=(const simd_u8 other) const { return this->max(other) == other; }

  really_inline bool any_bits_set(simd_u8 bits) const { return _mm256_testz_si256(*this, bits); }
  really_inline bool any_bits_set() const { return !_mm256_testz_si256(*this, *this); }

  really_inline simd_u8 prev(simd_u8& prev_chunk) const {
    return _mm256_alignr_epi8(*this, _mm256_permute2x128_si256(prev_chunk, *this, 0x21), 15);
  }
  really_inline simd_u8 prev2(const simd_u8 prev_chunk) const {
    return _mm256_alignr_epi8(*this, _mm256_permute2x128_si256(prev_chunk, *this, 0x21), 14);
  }

  // Perform a lookup of the lower 4 bits
  really_inline simd_u8 lookup4(
      uint8_t lookup0,  uint8_t lookup1,  uint8_t lookup2,  uint8_t lookup3,
      uint8_t lookup4,  uint8_t lookup5,  uint8_t lookup6,  uint8_t lookup7,
      uint8_t lookup8,  uint8_t lookup9,  uint8_t lookup10, uint8_t lookup11,
      uint8_t lookup12, uint8_t lookup13, uint8_t lookup14, uint8_t lookup15) const {

    simd_u8 lookup_table(
      lookup0, lookup1, lookup2,  lookup3,  lookup4,  lookup5,  lookup6,  lookup7,
      lookup8, lookup9, lookup10, lookup11, lookup12, lookup13, lookup14, lookup15,
      lookup0, lookup1, lookup2,  lookup3,  lookup4,  lookup5,  lookup6,  lookup7,
      lookup8, lookup9, lookup10, lookup11, lookup12, lookup13, lookup14, lookup15
    );
    return _mm256_shuffle_epi8(lookup_table, *this);
  }
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

} // namespace simdjson::haswell
UNTARGET_REGION

#endif // IS_X86_64
#endif // SIMDJSON_HASWELL_SIMD_INPUT_H
