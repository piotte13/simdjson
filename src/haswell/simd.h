#ifndef SIMDJSON_HASWELL_SIMD_H
#define SIMDJSON_HASWELL_SIMD_H

#include "simdjson/common_defs.h"
#include "simdjson/portability.h"

#ifdef IS_X86_64

TARGET_HASWELL
namespace simdjson::haswell::simd {

  // SIMD type, just so we can copy/paste more easily between architectures.
  typedef __m256i simd_t;
  // Output of m8.to_bitmask(). uint32_t for 32-byte SIMD registers, uint16_t for 16-byte.
  typedef uint32_t m8_bitmask;

  // Forward-declared so they can be used by splat and friends.
  struct m8;
  struct u8;
  struct i8;

  template<typename Child, typename Element>
  struct base {
    simd_t value;

    // Zero constructor
    really_inline base() : value{simd_t()} {}

    // Conversion from SIMD register
    really_inline base(const simd_t _value) : value(_value) {}

    // Conversion to SIMD register
    really_inline operator const simd_t&() const { return this->value; }
    really_inline operator simd_t&() { return this->value; }

    // Bit operations
    really_inline Child operator|(const Child other) const { return _mm256_or_si256(*this, other); }
    really_inline Child operator&(const Child other) const { return _mm256_and_si256(*this, other); }
    really_inline Child operator^(const Child other) const { return _mm256_xor_si256(*this, other); }
    really_inline Child bit_andnot(const Child other) const { return _mm256_andnot_si256(*this, other); }
    really_inline Child operator~() const { return this ^ 0xFFu; }
    really_inline Child& operator|=(const Child other) { auto this_cast = (Child*)this; *this_cast = *this_cast | other; return *this_cast; }
    really_inline Child& operator&=(const Child other) { auto this_cast = (Child*)this; *this_cast = *this_cast & other; return *this_cast; }
    really_inline Child& operator^=(const Child other) { auto this_cast = (Child*)this; *this_cast = *this_cast ^ other; return *this_cast; }
  };

  template<typename Child, typename Element, typename Mask=m8>
  struct base8: base<Child,Element> {
    really_inline base8() : base<Child,Element>() {}
    really_inline base8(const simd_t _value) : base<Child,Element>(_value) {}

    really_inline Mask operator==(const Child other) const { return _mm256_cmpeq_epi8(*this, other); }

    static const int SIZE = sizeof(base<Child,Element>::value);

    really_inline Child prev(const Child prev_chunk) const {
      return _mm256_alignr_epi8(*this, _mm256_permute2x128_si256(prev_chunk, *this, 0x21), 15);
    }
    really_inline Child prev2(const Child prev_chunk) const {
      return _mm256_alignr_epi8(*this, _mm256_permute2x128_si256(prev_chunk, *this, 0x21), 14);
    }
  };

  // SIMD byte mask type (returned by things like eq and gt)
  struct m8: base8<m8, bool> {
    static really_inline m8 splat(bool _value) { return _mm256_set1_epi8(-(!!_value)); }

    really_inline m8() : base8() {}
    really_inline m8(const simd_t _value) : base8<m8,bool>(_value) {}
    // Splat constructor
    really_inline m8(bool _value) : base8<m8,bool>(splat(_value)) {}

    really_inline m8_bitmask to_bitmask() const { return _mm256_movemask_epi8(*this); }
    really_inline bool any() const { return _mm256_testz_si256(*this, *this) == 0; }
  };

  template<typename Child, typename Element>
  struct simd_numeric8: base8<Child, Element> {
    static really_inline Child splat(Element _value) { return _mm256_set1_epi8(_value); }
    static really_inline Child zero() { return _mm256_setzero_si256(); }
    static really_inline Child load(const Element* values) {
      return _mm256_loadu_si256(reinterpret_cast<const simd_t *>(values));
    }

    really_inline simd_numeric8() : base8<Child,Element>() {}
    really_inline simd_numeric8(const simd_t _value) : base8<Child,Element>(_value) {}
    // Splat constructor
    really_inline simd_numeric8(Element _value) : base8<Child,Element>(splat(_value)) {}
    // Element array constructor
    really_inline simd_numeric8(const Element* values) : base8<Child,Element>(load(values)) {}

    // Addition/subtraction are the same for signed and unsigned
    really_inline Child operator+(const Child other) const { return _mm256_add_epi8(*this, other); }
    really_inline Child operator-(const Child other) const { return _mm256_sub_epi8(*this, other); }
    really_inline Child& operator+=(const Child other) { *this = *this + other; return *this; }
    really_inline Child& operator-=(const Child other) { *this = *this - other; return *this; }

    // Perform a lookup of the lower 4 bits
    really_inline Child lookup4(
        Element lookup0,  Element lookup1,  Element lookup2,  Element lookup3,
        Element lookup4,  Element lookup5,  Element lookup6,  Element lookup7,
        Element lookup8,  Element lookup9,  Element lookup10, Element lookup11,
        Element lookup12, Element lookup13, Element lookup14, Element lookup15) const {

      Child lookup_table(
        lookup0, lookup1, lookup2,  lookup3,  lookup4,  lookup5,  lookup6,  lookup7,
        lookup8, lookup9, lookup10, lookup11, lookup12, lookup13, lookup14, lookup15,
        lookup0, lookup1, lookup2,  lookup3,  lookup4,  lookup5,  lookup6,  lookup7,
        lookup8, lookup9, lookup10, lookup11, lookup12, lookup13, lookup14, lookup15
      );
      return _mm256_shuffle_epi8(lookup_table, *this);
    }
  };

  // Signed bytes
  struct i8 : simd_numeric8<i8, int8_t> {
    really_inline i8() : simd_numeric8<i8,int8_t>() {}
    really_inline i8(const simd_t _value) : simd_numeric8<i8,int8_t>(_value) {}
    really_inline i8(int8_t _value) : simd_numeric8<i8,int8_t>(_value) {}
    really_inline i8(const int8_t* values) : simd_numeric8<i8,int8_t>(values) {}
    // Member-by-member initialization
    really_inline i8(
      int8_t v0,  int8_t v1,  int8_t v2,  int8_t v3, int8_t v4,  int8_t v5,  int8_t v6,  int8_t v7,
      int8_t v8,  int8_t v9,  int8_t v10, int8_t v11, int8_t v12, int8_t v13, int8_t v14, int8_t v15,
      int8_t v16, int8_t v17, int8_t v18, int8_t v19, int8_t v20, int8_t v21, int8_t v22, int8_t v23,
      int8_t v24, int8_t v25, int8_t v26, int8_t v27, int8_t v28, int8_t v29, int8_t v30, int8_t v31
    ) : simd_numeric8<i8,int8_t>(_mm256_setr_epi8(
      v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,v11,v12,v13,v14,v15,
      v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,v29,v30,v31
    )) {}

    // Order-sensitive comparisons
    really_inline i8 max(const i8 other) const { return _mm256_max_epi8(*this, other); }
    really_inline i8 min(const i8 other) const { return _mm256_min_epi8(*this, other); }
    really_inline m8 operator>(const i8 other) const { return _mm256_cmpgt_epi8(*this, other); }
  };

  // Unsigned bytes
  struct u8: simd_numeric8<u8, uint8_t> {
    really_inline u8() : simd_numeric8<u8,uint8_t>() {}
    really_inline u8(const simd_t _value) : simd_numeric8<u8,uint8_t>(_value) {}
    really_inline u8(uint8_t _value) : simd_numeric8<u8,uint8_t>(_value) {}
    really_inline u8(const uint8_t* values) : simd_numeric8<u8,uint8_t>(values) {}
    // Member-by-member initialization
    really_inline u8(
      uint8_t v0,  uint8_t v1,  uint8_t v2,  uint8_t v3, uint8_t v4,  uint8_t v5,  uint8_t v6,  uint8_t v7,
      uint8_t v8,  uint8_t v9,  uint8_t v10, uint8_t v11, uint8_t v12, uint8_t v13, uint8_t v14, uint8_t v15,
      uint8_t v16, uint8_t v17, uint8_t v18, uint8_t v19, uint8_t v20, uint8_t v21, uint8_t v22, uint8_t v23,
      uint8_t v24, uint8_t v25, uint8_t v26, uint8_t v27, uint8_t v28, uint8_t v29, uint8_t v30, uint8_t v31
    ) : simd_numeric8<u8,uint8_t>(_mm256_setr_epi8(
      v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,v11,v12,v13,v14,v15,
      v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,v29,v30,v31
    )) {}

    // Saturated math
    really_inline u8 saturated_add(const u8 other) const { return _mm256_adds_epu8(*this, other); }
    really_inline u8 saturated_sub(const u8 other) const { return _mm256_subs_epu8(*this, other); }

    // Order-specific operations
    really_inline u8 max(const u8 other) const { return _mm256_max_epu8(*this, other); }
    really_inline u8 min(const u8 other) const { return _mm256_min_epu8(*this, other); }
    really_inline m8 operator<=(const u8 other) const { return this->max(other) == other; }

    // Bit-specific operations
    really_inline bool any_bits_set(u8 bits) const { return _mm256_testz_si256(*this, bits); }
    really_inline bool any_bits_set() const { return !_mm256_testz_si256(*this, *this); }
    really_inline u8 operator>>(const int count) const { return _mm256_srli_epi16(*this, count); }
    really_inline u8 operator<<(const int count) const { return _mm256_slli_epi16(*this, count); }
    really_inline u8& operator>>=(const int count) { *this = *this >> count; return *this; }
    really_inline u8& operator<<=(const int count) { *this = *this << count; return *this; }
  };

  struct u8x64 {
    const u8 chunks[2];

    really_inline u8x64() : chunks{u8(), u8()} {}

    really_inline u8x64(const simd_t chunk0, const simd_t chunk1) : chunks{chunk0, chunk1} {}

    really_inline u8x64(const uint8_t *ptr) : chunks{u8::load(ptr), u8::load(ptr+32)} {}

    template <typename F>
    really_inline void each(F const& each_chunk) const
    {
      each_chunk(this->chunks[0]);
      each_chunk(this->chunks[1]);
    }

    template <typename F>
    really_inline u8x64 map(F const& map_chunk) const {
      return u8x64(
        map_chunk(this->chunks[0]),
        map_chunk(this->chunks[1])
      );
    }

    template <typename F>
    really_inline u8x64 map(const u8x64 b, F const& map_chunk) const {
      return u8x64(
        map_chunk(this->chunks[0], b.chunks[0]),
        map_chunk(this->chunks[1], b.chunks[1])
      );
    }

    template <typename F>
    really_inline u8 reduce(F const& reduce_pair) const {
      return reduce_pair(this->chunks[0], this->chunks[1]);
    }

    really_inline uint64_t to_bitmask() const {
      uint64_t r_lo = static_cast<uint32_t>(m8(this->chunks[0]).to_bitmask());
      uint64_t r_hi =                       m8(this->chunks[1]).to_bitmask();
      return r_lo | (r_hi << 32);
    }

    really_inline u8x64 bit_or(const uint8_t m) const {
      const u8 mask = u8::splat(m);
      return this->map( [&](auto a) { return a | mask; } );
    }

    really_inline uint64_t eq(const uint8_t m) const {
      const u8 mask = u8::splat(m);
      return this->map( [&](auto a) { return a == mask; } ).to_bitmask();
    }

    really_inline uint64_t lteq(const uint8_t m) const {
      const u8 mask = u8::splat(m);
      return this->map( [&](auto a) { return a <= mask; } ).to_bitmask();
    }

  }; // struct u8x64

} // namespace simdjson::haswell::simd
UNTARGET_REGION

#endif // IS_X86_64
#endif // SIMDJSON_HASWELL_SIMD_H
