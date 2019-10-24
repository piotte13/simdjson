#ifndef SIMDJSON_ARM64_SIMD_H
#define SIMDJSON_ARM64_SIMD_H

#include "simdjson/common_defs.h"
#include "simdjson/portability.h"
#include "simdjson/simdjson.h"

#ifdef IS_ARM64

namespace simdjson::arm64::simd {

  really_inline simd_u8_bitmask neon_movemask(simd_t input) {
    const simd_t bit_mask = {0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80,
                                0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80};
    simd_t minput = vandq_u8(input, bit_mask);
    simd_t tmp = vpaddq_u8(minput, minput);
    tmp = vpaddq_u8(tmp, tmp);
    tmp = vpaddq_u8(tmp, tmp);
    return vgetq_lane_u16(vreinterpretq_u16_u8(tmp), 0);
  }

  really_inline uint64_t neon_movemask_bulk(simd_t p0, simd_t p1,
                                            simd_t p2, simd_t p3) {
    const simd_t bit_mask = {0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80,
                            0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80};
    simd_t t0 = vandq_u8(p0, bit_mask);
    simd_t t1 = vandq_u8(p1, bit_mask);
    simd_t t2 = vandq_u8(p2, bit_mask);
    simd_t t3 = vandq_u8(p3, bit_mask);
    simd_t sum0 = vpaddq_u8(t0, t1);
    simd_t sum1 = vpaddq_u8(t2, t3);
    sum0 = vpaddq_u8(sum0, sum1);
    sum0 = vpaddq_u8(sum0, sum0);
    return vgetq_lane_u64(vreinterpretq_u64_u8(sum0), 0);
  }

  // SIMD type, just so we can copy/paste more easily between architectures.
  typedef uint8x16_t simd_t;
  typedef int8x16_t simd_i;
  // Output of m8.to_bitmask(). uint32_t for 32-byte SIMD registers, uint16_t for 16-byte.
  typedef uint32_t m8_bitmask;

  // Forward-declared so they can be used by splat and friends.
  struct m8;
  struct u8;
  struct i8;

  template<typename Child, typename Element, typename Mask=m8>
  struct base_u8 {
    simd_t value;

    // Zero constructor
    really_inline base_u8() : value{simd_t()} {}

    // Conversion from SIMD register
    really_inline base_u8(const simd_t _value) : value(_value) {}

    // Conversion to SIMD register
    really_inline operator const simd_t&() const { return this->value; }
    really_inline operator simd_t&() { return this->value; }

    // Bit operations
    really_inline Child operator|(const Child other) const { return vorrq_u8(*this, other); }
    really_inline Child operator&(const Child other) const { return vandq_u8(*this, other); }
    really_inline Child operator^(const Child other) const { return veorq_u8(*this, other); }
    really_inline Child bit_andnot(const Child other) const { return vbicq_u8(*this, other); }
    really_inline Child operator~() const { return this ^ 0xFFu; }
    really_inline Child& operator|=(const Child other) { auto this_cast = (Child*)this; *this_cast = *this_cast | other; return *this_cast; }
    really_inline Child& operator&=(const Child other) { auto this_cast = (Child*)this; *this_cast = *this_cast & other; return *this_cast; }
    really_inline Child& operator^=(const Child other) { auto this_cast = (Child*)this; *this_cast = *this_cast ^ other; return *this_cast; }

    really_inline Mask operator==(const Child other) const { return vceqq_u8(*this, other); }

    static const int SIZE = sizeof(base<Child,Element>::value);

    really_inline Child prev(const Child prev_chunk) const {
      return vextq_u8(*this, prev_chunk, 16 - 1);
    }
    really_inline Child prev2(const Child prev_chunk) const {
      return vextq_u8(*this, prev_chunk, 16 - 2);
    }
  };

  // SIMD byte mask type (returned by things like eq and gt)
  struct m8: base_u8<m8, bool> {
    static really_inline m8 splat(bool _value) { return vmovq_n_u8(-(!!_value)); }

    really_inline m8() : base_u8<m8, bool>() {}
    really_inline m8(const simd_t _value) : base_u8<m8,bool>(_value) {}
    // Splat constructor
    really_inline m8(bool _value) : base_u8<m8,bool>(splat(_value)) {}

    // Conversion to SIMD register
    really_inline operator const simd_t&() const { return this->value; }
    really_inline operator simd_t&() { return this->value; }

    really_inline m8_bitmask to_bitmask() const { return neon_movemask(*this); }
    really_inline bool any() const { return vmaxvq_u32(*this) == 0; }
  };

  // Unsigned bytes
  struct u8: base_u8<u8, uint8_t> {
    static really_inline simd_t splat(uint8_t _value) { return vmovq_n_u8(_value); }

    static really_inline simd_t zero() { return vdup_n_u8(0); }
    static really_inline simd_t load(const uint8_t* values) { return vld1q_u8(values); }

    really_inline u8() : base_u8<u8,uint8_t>() {}
    really_inline u8(const simd_t _value) : base_u8<u8,uint8_t>(_value) {}
    really_inline u8(uint8_t _value) : base_u8<u8,uint8_t>(splat(_value)) {}
    really_inline u8(const uint8_t* values) : base_u8<u8,uint8_t>(values) {}
    really_inline explicit u8(const simd_i _value): base_u8<u8,uint8_t>(vreinterpretq_u8_s8(_value)) {}

    // Member-by-member initialization
    really_inline u8(
      uint8_t v0,  uint8_t v1,  uint8_t v2,  uint8_t v3, uint8_t v4,  uint8_t v5,  uint8_t v6,  uint8_t v7,
      uint8_t v8,  uint8_t v9,  uint8_t v10, uint8_t v11, uint8_t v12, uint8_t v13, uint8_t v14, uint8_t v15,
      uint8_t v16, uint8_t v17, uint8_t v18, uint8_t v19, uint8_t v20, uint8_t v21, uint8_t v22, uint8_t v23,
      uint8_t v24, uint8_t v25, uint8_t v26, uint8_t v27, uint8_t v28, uint8_t v29, uint8_t v30, uint8_t v31
    ) : base_u8<u8,uint8_t>(simd_t{
      v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,v11,v12,v13,v14,v15,
      v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,v29,v30,v31
    }) {}

    // Saturated math
    really_inline u8 saturating_add(const u8 other) const { return vqaddq_u8(*this, other); }
    really_inline u8 saturating_sub(const u8 other) const { return vqsubq_u8(*this, other); }

    // Addition/subtraction are the same for signed and unsigned
    really_inline Child operator+(const Child other) const { return vaddq_u8(*this, other); }
    really_inline Child operator-(const Child other) const { return vsubq_u8(*this, other); }
    really_inline Child& operator+=(const Child other) { *this = *this + other; return *this; }
    really_inline Child& operator-=(const Child other) { *this = *this - other; return *this; }

    // Order-specific operations
    really_inline u8 max(const u8 other) const { return vmaxq_u8(*this, other); }
    really_inline u8 min(const u8 other) const { return vminq_u8(*this, other); }
    really_inline m8 operator<=(const u8 other) const { return vcleq_u8(*this, other); }

    // Bit-specific operations
    really_inline bool any_bits_set() const { return !vmaxvq_u8(*this); }
    really_inline bool any_bits_set(u8 bits) const { return (*this & bits).any_bits_set(); }
    really_inline u8 operator>>(const int count) const { return vshrq_n_u8(*this, count); }
    really_inline u8 operator<<(const int count) const { return vshlq_n_u8(*this, count); }
    really_inline u8& operator>>=(const int count) { *this = *this >> count; return *this; }
    really_inline u8& operator<<=(const int count) { *this = *this << count; return *this; }

    // Perform a lookup of the lower 4 bits
    really_inline u8 lookup4(
        uint8_t lookup0,  uint8_t lookup1,  uint8_t lookup2,  uint8_t lookup3,
        uint8_t lookup4,  uint8_t lookup5,  uint8_t lookup6,  uint8_t lookup7,
        uint8_t lookup8,  uint8_t lookup9,  uint8_t lookup10, uint8_t lookup11,
        uint8_t lookup12, uint8_t lookup13, uint8_t lookup14, uint8_t lookup15) const {

      u8 lookup_table(
        lookup0, lookup1, lookup2,  lookup3,  lookup4,  lookup5,  lookup6,  lookup7,
        lookup8, lookup9, lookup10, lookup11, lookup12, lookup13, lookup14, lookup15,
        lookup0, lookup1, lookup2,  lookup3,  lookup4,  lookup5,  lookup6,  lookup7,
        lookup8, lookup9, lookup10, lookup11, lookup12, lookup13, lookup14, lookup15
      );
      return vqtbl1q_u8(lookup_table, *this);
    }

    // Perform a lookup of the lower 4 bits
    really_inline i8 lookup4(
        int8_t lookup0,  int8_t lookup1,  int8_t lookup2,  int8_t lookup3,
        int8_t lookup4,  int8_t lookup5,  int8_t lookup6,  int8_t lookup7,
        int8_t lookup8,  int8_t lookup9,  int8_t lookup10, int8_t lookup11,
        int8_t lookup12, int8_t lookup13, int8_t lookup14, int8_t lookup15) const {

      i8 lookup_table(
        lookup0, lookup1, lookup2,  lookup3,  lookup4,  lookup5,  lookup6,  lookup7,
        lookup8, lookup9, lookup10, lookup11, lookup12, lookup13, lookup14, lookup15,
        lookup0, lookup1, lookup2,  lookup3,  lookup4,  lookup5,  lookup6,  lookup7,
        lookup8, lookup9, lookup10, lookup11, lookup12, lookup13, lookup14, lookup15
      );
      return vqtbl1q_s8(lookup_table, *this);
    }
  };

  // Signed bytes
  struct i8 {
    simd_i value;

    static really_inline i8 splat(int8_t _value) { return vmovq_n_s8(_value); }
    static really_inline i8 zero() { return vdup_n_s8(0); }
    static really_inline i8 load(const int8_t* values) { return vld1q_s8(values); }

    really_inline i8() : value{} {}
    really_inline i8(const simd_i _value) : value{_value} {}
    really_inline i8(int8_t _value) : value{splat(_value)} {}
    really_inline i8(const int8_t* values) : value{load(values)} {}
    really_inline explicit i8(const simd_t other) { vreinterpretq_s8_u8(other); }
    // Member-by-member initialization
    really_inline i8(
      int8_t v0,  int8_t v1,  int8_t v2,  int8_t v3, int8_t v4,  int8_t v5,  int8_t v6,  int8_t v7,
      int8_t v8,  int8_t v9,  int8_t v10, int8_t v11, int8_t v12, int8_t v13, int8_t v14, int8_t v15,
      int8_t v16, int8_t v17, int8_t v18, int8_t v19, int8_t v20, int8_t v21, int8_t v22, int8_t v23,
      int8_t v24, int8_t v25, int8_t v26, int8_t v27, int8_t v28, int8_t v29, int8_t v30, int8_t v31
    ) : value{
      v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,v11,v12,v13,v14,v15,
      v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,v29,v30,v31
    } {}

    // Conversion to SIMD register
    really_inline operator const simd_t&() const { return this->value; }
    really_inline operator simd_t&() { return this->value; }

    // Addition/subtraction are the same for signed and unsigned
    really_inline Child operator+(const Child other) const { return vaddq_u8(*this, other); }
    really_inline Child operator-(const Child other) const { return vsubq_u8(*this, other); }
    really_inline Child& operator+=(const Child other) { *this = *this + other; return *this; }
    really_inline Child& operator-=(const Child other) { *this = *this - other; return *this; }

    // Order-sensitive comparisons
    really_inline i8 max(const i8 other) const { return vmaxq_s8(*this, other); }
    really_inline i8 min(const i8 other) const { return vminq_s8(*this, other); }
    really_inline m8 operator>(const i8 other) const { return vcgtq_s8(*this, other); }

    really_inline i8 prev(const i8 prev_chunk) const {
      return vextq_u8(*this, prev_chunk, 16 - 1);
    }
    really_inline i8 prev2(const i8 prev_chunk) const {
      return vextq_u8(*this, prev_chunk, 16 - 2);
    }

    // Perform a lookup of the lower 4 bits
    really_inline u8 lookup4(
        uint8_t lookup0,  uint8_t lookup1,  uint8_t lookup2,  uint8_t lookup3,
        uint8_t lookup4,  uint8_t lookup5,  uint8_t lookup6,  uint8_t lookup7,
        uint8_t lookup8,  uint8_t lookup9,  uint8_t lookup10, uint8_t lookup11,
        uint8_t lookup12, uint8_t lookup13, uint8_t lookup14, uint8_t lookup15) const {

      return u8(*this).lookup4(
        lookup0, lookup1, lookup2,  lookup3,  lookup4,  lookup5,  lookup6,  lookup7,
        lookup8, lookup9, lookup10, lookup11, lookup12, lookup13, lookup14, lookup15,
        lookup0, lookup1, lookup2,  lookup3,  lookup4,  lookup5,  lookup6,  lookup7,
        lookup8, lookup9, lookup10, lookup11, lookup12, lookup13, lookup14, lookup15
      );
    }

    // Perform a lookup of the lower 4 bits
    really_inline i8 lookup4(
        int8_t lookup0,  int8_t lookup1,  int8_t lookup2,  int8_t lookup3,
        int8_t lookup4,  int8_t lookup5,  int8_t lookup6,  int8_t lookup7,
        int8_t lookup8,  int8_t lookup9,  int8_t lookup10, int8_t lookup11,
        int8_t lookup12, int8_t lookup13, int8_t lookup14, int8_t lookup15) const {

      return u8(*this).lookup4(
        lookup0, lookup1, lookup2,  lookup3,  lookup4,  lookup5,  lookup6,  lookup7,
        lookup8, lookup9, lookup10, lookup11, lookup12, lookup13, lookup14, lookup15,
        lookup0, lookup1, lookup2,  lookup3,  lookup4,  lookup5,  lookup6,  lookup7,
        lookup8, lookup9, lookup10, lookup11, lookup12, lookup13, lookup14, lookup15
      );
    }
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

} // namespace simdjson::arm64::simd

#endif // IS_ARM64
#endif // SIMDJSON_ARM64_SIMD_H
