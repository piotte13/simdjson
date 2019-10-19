#ifndef SIMDJSON_ARM64_SIMD_INPUT_H
#define SIMDJSON_ARM64_SIMD_INPUT_H

#include "simdjson/common_defs.h"
#include "simdjson/portability.h"
#include "simdjson/simdjson.h"

#ifdef IS_ARM64

namespace simdjson::arm64 {

// SIMD type, just so we can copy/paste more easily between architectures.
typedef uint8x16_t simd_t;
// Output of simd_u8.to_bitmask(). uint32_t for 32-byte SIMD registers, uint16_t for 16-byte.
typedef uint16_t simd_u8_bitmask;

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

really_inline simd_t splat(uint8_t value)  { return vmovq_n_u8(value); }
really_inline simd_t splat( int8_t value)  { return vmovq_n_u8(value); }
really_inline simd_t splat(uint16_t value) { return vmovq_n_u16(value); }
really_inline simd_t splat( int16_t value) { return vmovq_n_u16(value); }
really_inline simd_t splat(uint32_t value) { return vmovq_n_u32(value); }
really_inline simd_t splat( int32_t value) { return vmovq_n_u32(value); }
really_inline simd_t splat(uint64_t value) { return vmovq_n_u64(value); }
really_inline simd_t splat( int64_t value) { return vmovq_n_u64(value); }
really_inline simd_t splat(bool value) { return vmovq_n_u64(uint64_t(0)-!value); }

struct simd_m8 {
  simd_t value;

  // Constructors
  really_inline simd_m8() : value(simd_t()) {}

  // Conversion from SIMD register
  really_inline simd_m8(const simd_t _value) : value(_value) {}
  // Conversion to const SIMD register
  really_inline operator const simd_t&() const { return this->value; }
  // Conversion to non-const SIMD register
  really_inline operator simd_t&() { return this->value; }

  really_inline simd_m8 operator ||(const simd_m8& other) const { return vorrq_u8(*this, other); }
  really_inline simd_m8 operator &&(const simd_m8& other) const { return vandq_u8(*this, other); }
  really_inline simd_m8 logical_xor(const simd_m8& other) const { return veorq_u8(*this, other); }
  really_inline simd_m8 andnot(const simd_m8& other) const { return vbicq_u8(*this, other); }
  really_inline simd_m8 operator !() const { return this->logical_xor(splat(u8'\xFF')); }

  really_inline simd_u8_bitmask to_bitmask() const { return neon_movemask(*this); }
  really_inline bool any() const { return this->to_bitmask(); }
  really_inline bool all() const { return (this->to_bitmask()+1) == 0; }
};

struct simd_u8 {
  simd_t value;

  // Constructors
  really_inline simd_u8() : value(simd_t()) {}
  really_inline simd_u8(const uint8_t *_value) : value(vld1q_u8(_value)) { }

  // Conversion from SIMD register
  really_inline simd_u8(const simd_t _value) : value(_value) {}
  // Conversion to const SIMD register
  really_inline operator const simd_t&() const { return this->value; }
  // Conversion to non-const SIMD register
  really_inline operator simd_t&() { return this->value; }

  really_inline simd_u8 operator |(const simd_t& other) const { return vorrq_u8(*this, other); }
  really_inline simd_u8 operator &(const simd_t& other) const { return vandq_u8(*this, other); }
  really_inline simd_u8 operator ^(const simd_t& other) const { return veorq_u8(*this, other); }
  really_inline simd_u8 bit_andnot(const simd_t& other) const { return vbicq_u8(*this, other); }
  really_inline simd_u8 operator ~() const { return *this ^ splat(u8'\xFF'); }

  really_inline simd_u8 max(const simd_t& other) const { return vmaxq_u8(*this, other); }
  really_inline simd_u8 min(const simd_t& other) const { return vminq_u8(*this, other); }

  really_inline simd_m8 eq(const simd_t& other) const { return vceqq_u8(*this, other); }
  really_inline simd_m8 lteq(const simd_t& other) const { return vcleq_u8(*this, other); }
};

struct simd_u8x64 {
  const simd_u8 chunks[4];

  really_inline simd_u8x64()
      : chunks { simd_u8(), simd_u8(), simd_u8(), simd_u8() } {}

  really_inline simd_u8x64(const simd_t chunk0, const simd_t chunk1, const simd_t chunk2, const simd_t chunk3)
      : chunks{chunk0, chunk1, chunk2, chunk3} {}

  really_inline simd_u8x64(const uint8_t *ptr)
      : chunks{simd_u8(ptr), simd_u8(ptr+16), simd_u8(ptr+32), simd_u8(ptr+48)} {}

  template <typename F>
  really_inline void each(F const& each_chunk) const {
    each_chunk(this->chunks[0]);
    each_chunk(this->chunks[1]);
    each_chunk(this->chunks[2]);
    each_chunk(this->chunks[3]);
  }

  template <typename F>
  really_inline simd_u8x64 map(F const& map_chunk) const {
    return simd_u8x64(
      map_chunk(this->chunks[0]),
      map_chunk(this->chunks[1]),
      map_chunk(this->chunks[2]),
      map_chunk(this->chunks[3])
    );
  }

  template <typename F>
  really_inline simd_u8x64 map(simd_u8x64 b, F const& map_chunk) const {
    return simd_u8x64(
      map_chunk(this->chunks[0], b.chunks[0]),
      map_chunk(this->chunks[1], b.chunks[1]),
      map_chunk(this->chunks[2], b.chunks[2]),
      map_chunk(this->chunks[3], b.chunks[3])
    );
  }

  template <typename F>
  really_inline simd_t reduce(F const& reduce_pair) const {
    simd_t r01 = reduce_pair(this->chunks[0], this->chunks[1]);
    simd_t r23 = reduce_pair(this->chunks[2], this->chunks[3]);
    return reduce_pair(r01, r23);
  }

  really_inline uint64_t to_bitmask() const {
    return neon_movemask_bulk(this->chunks[0], this->chunks[1], this->chunks[2], this->chunks[3]);
  }

  really_inline simd_u8x64 bit_or(const uint8_t m) const {
    const simd_t mask = splat(m);
    return this->map( [&](auto a) { return a | mask; });
  }

  really_inline uint64_t eq(const uint8_t m) const {
    const simd_t mask = splat(m);
    return this->map( [&](auto a) { return a.eq(mask); }).to_bitmask();
  }

  really_inline uint64_t lteq(const uint8_t m) const {
    const simd_t mask = splat(m);
    return this->map( [&](auto a) { return a.le(mask); }).to_bitmask();
  }

}; // struct simd_u8x64

} // namespace simdjson::arm64

#endif // IS_ARM64
#endif // SIMDJSON_ARM64_SIMD_INPUT_H
