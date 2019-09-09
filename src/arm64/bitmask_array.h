#ifndef SIMDJSON_ARM64_BITMASK_ARRAY_H
#define SIMDJSON_ARM64_BITMASK_ARRAY_H

#include "simdjson/common_defs.h"
#include "simdjson/portability.h"
#include "arm64/architecture.h"

#ifdef IS_ARM64

namespace simdjson::arm64 {

struct bitmask_array;

template<typename F>
really_inline void each64(F const& each) {
  each(0);
  each(1);
}

template<typename R=bitmask_array, typename F>
really_inline R map64(F const& map) {
  auto r0 = map(0);
  auto r1 = map(1);
  return R(r0, r1);
}

struct bitmask_array {
  uint64_t bitmasks[SIMD_WIDTH/64];
  really_inline bitmask_array(
    const uint64_t m0, const uint64_t m1
  ) : bitmasks{
    m0,m1
  } { }
  really_inline bitmask_array() : bitmask_array(0,0) {}
  really_inline bitmask_array(
      const uint16_t m0, const uint16_t m1, const uint16_t m2, const uint16_t m3,
      const uint16_t m4, const uint16_t m5, const uint16_t m6, const uint16_t m7
  ) : bitmasks{
    m0|(static_cast<uint64_t>(m1)<<16)|(static_cast<uint64_t>(m2)<<32)|(static_cast<uint64_t>(m3)<<48),
    m4|(static_cast<uint64_t>(m5)<<16)|(static_cast<uint64_t>(m6)<<32)|(static_cast<uint64_t>(m7)<<48)
  } { }

  really_inline uint64_t operator[](const size_t index) const { return this->bitmasks[index]; }
  really_inline uint64_t& operator[](const size_t index) { return this->bitmasks[index]; }
  static constexpr void assert_is_chunks64() { }

  template<typename F>
  really_inline bitmask_array each(F const &f) const {
    return map64([&](size_t i) { return f(this->bitmasks[i]); });
  }
  template<typename F>
  really_inline bitmask_array map(F const &f) const {
    return map64([&](size_t i) { return f(this->bitmasks[i]); });
  }
  template<typename F, typename B>
  really_inline bitmask_array map(B b, F const &f) const {
    B::assert_is_chunks64();
    return map64([&](size_t i) { return f(this->bitmasks[i], b[i]); });
  }

  really_inline bitmask_array prev(bool &carry) const {
    // Naive algorithm
    return this->map([&](uint64_t bitmask) {
      bool prev_carry = carry;
      carry = (bitmask & 0x8000000000000000ULL) != 0;
      return (bitmask << 1) | prev_carry;
    });
  }

  really_inline bitmask_array after_series_starting_with(const bitmask_array starting_with, bool &carry) const {
    // Naive algorithm
    return this->map(starting_with, [&](uint64_t series_bitmask, uint64_t starting_with_bitmask) {
      uint64_t result;
      carry = add_overflow(series_bitmask, starting_with_bitmask | carry, &result);
      result &= ~series_bitmask;
      return result;
    });
  }

} // namespace simdjson::arm64

#endif // IS_ARM64
#endif // SIMDJSON_ARM64_BITMASK_ARRAY_H
