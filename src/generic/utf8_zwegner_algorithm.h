/*
 * legal utf-8 byte sequence
 * http://www.unicode.org/versions/Unicode6.0.0/ch03.pdf - page 94
 *
 *  Code Points        1st       2s       3s       4s
 * U+0000..U+007F     00..7F
 * U+0080..U+07FF     C2..DF   80..BF
 * U+0800..U+0FFF     E0       A0..BF   80..BF
 * U+1000..U+CFFF     E1..EC   80..BF   80..BF
 * U+D000..U+D7FF     ED       80..9F   80..BF
 * U+E000..U+FFFF     EE..EF   80..BF   80..BF
 * U+10000..U+3FFFF   F0       90..BF   80..BF   80..BF
 * U+40000..U+FFFFF   F1..F3   80..BF   80..BF   80..BF
 * U+100000..U+10FFFF F4       80..8F   80..BF   80..BF
 *
 */

// all byte values must be no larger than 0xF4

using namespace simd;

// NOTE (@jkeiser): this uses simd8<bool> instead of the v_* morphisms, which we've been optimizing
// in simdjson. Should yield identical intrinsics.
using vmask_t = simd8<bool>::bitmask_t;
using vmask2_t = simd8<bool>::bitmask2_t;
using vec_t = simd8<uint8_t>;

struct utf8_checker {
  vec_t has_error;
  vec_t prev_bytes;
  vmask2_t last_cont;
  vmask_t cont_error;

  // NOTE (@jkeiser): I added constants showing what error each flag was, so that the relationship
  // between the tables and the errors was clearer.
  static const uint8_t OVERLONG_2  = 0x01; // 1100000_         ________         Could have been encoded in 1 byte
  static const uint8_t OVERLONG_3  = 0x02; // 11100000         100_____         Could have been encoded in 2 bytes
  static const uint8_t SURROGATE   = 0x04; // 11101010         101_____         Surrogate pairs
  // NOTE (@jkeiser): to make the tables smaller, I removed second bytes starting with 11 since missing continuations will be detected elsewhere
  static const uint8_t TOO_LARGE   = 0x08; // 11110100         (1001|101_)____ > U+10FFFF
  static const uint8_t TOO_LARGE_2 = 0x10; // 1111(0101..1111) ________       > U+10FFFF
  // NOTE (@jkeiser): I added validation of overlong 4-byte encodings. No performance impact, though.
  static const uint8_t OVERLONG_4  = 0x20; // 11110000         1000____         Could have been encoded in 3 bytes

  // check whether the current bytes are valid UTF-8
  // at the end of the function, previous gets updated
  really_inline void check_utf8_bytes(vec_t bytes, int high) {
    // Count: 14 simd ops, 4 simd constants, 3 movemask, 15 64-bit ops
    vec_t shifted_bytes = bytes.prev<1>(this->prev_bytes);
    // Which bytes are required to be continuation bytes
    vmask2_t req = this->last_cont;

    // Compute the continuation byte mask by finding bytes that start with
    // 11x, 111x, and 1111. For each of these prefixes, we get a bitmask
    // and shift it forward by 1, 2, or 3. This loop should be unrolled by
    // the compiler, and the (n == 1) branch inside eliminated.
    //
    // NOTE (@jkeiser): I unrolled the for(i=1..3) loop because I don't trust compiler unrolling
    // anymore. This should be exactly equivalent and yield the same optimizations (and also lets
    // us rearrange statements if we so desire).

    vmask_t set = high & bytes.shl<1>().high_bits_to_bitmask();
    // Mark continuation bytes: those that have the high bit set but
    // not the next one
    vmask_t cont = high ^ set;
    // We add the shifted mask here instead of ORing it, which would
    // be the more natural operation, so that this line can be done
    // with one lea. While adding could give a different result due
    // to carries, this will only happen for invalid UTF-8 sequences,
    // and in a way that won't cause it to pass validation. Reasoning:
    // Any bits for required continuation bytes come after the bits
    // for their leader bytes, and are all contiguous. For a carry to
    // happen, two of these bit sequences would have to overlap. If
    // this is the case, there is a leader byte before the second set
    // of required continuation bytes (and thus before the bit that
    // will be cleared by a carry). This leader byte will not be
    // in the continuation mask, despite being required. QEDish.
    req += vmask2_t(set) << 1;
    set &= bytes.shl<2>().high_bits_to_bitmask();
    req += vmask2_t(set) << 2;
    set &= bytes.shl<3>().high_bits_to_bitmask();
    req += vmask2_t(set) << 3;

    // Check that continuation bytes match. We must cast req from vmask2_t
    // (which holds the carry mask in the upper half) to vmask_t, which
    // zeroes out the upper bits
    //
    // NOTE (@jkeiser): I turned the if() statement here into this->has_error for performance in
    // success cases: instead of spending time testing the result and introducing a branch (which
    // can affect performance even if it's easily predictable), we test once at the end.
    // The ^ is equivalent to !=, however, leaving a 1 where the bits are different and 0 where they
    // are the same.
    this->cont_error |= cont ^ vmask_t(req);

    // Look up error masks for three consecutive nibbles. We need to
    // AND with 0x0F for each one, because vpshufb has the neat
    // "feature" that negative values in an index byte will result in 
    // a zero.
    //
    // NOTE (@jkeiser): I removed the & 0x0F here because shr<4> already does that.
    vec_t e_1 = shifted_bytes.shr<4>().lookup_16<uint8_t>(
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,

        OVERLONG_2,   // [1100]000_         ________        Could have been encoded in 1 byte
        0,
        OVERLONG_3 |  // [1110]0000         100_____        Could have been encoded in 2 bytes
          SURROGATE,  // [1110]1010         101_____        Surrogate pairs
        OVERLONG_4 |  // [1111]0000         1000____        Could have been encoded in 3 bytes
          TOO_LARGE | // [1111]0100         (1001|101_)____ > U+10FFFF
          TOO_LARGE_2 // [1111](0101..1111) ________        > U+10FFFF
    );

    // NOTE (@jkeiser): I removed the & 0x0F here because lookup_lower_4() ignores the upper nibble
    vec_t e_2 = shifted_bytes.lookup_lower_4_bits<uint8_t>(
      OVERLONG_2 |                                        // 1100[000_]       ________        Could have been encoded in 1 byte
        OVERLONG_3 |                                      // 1110[0000]       100_____        Could have been encoded in 2 bytes
        OVERLONG_4,                                       // 1111[0000]       1000____        Could have been encoded in 3 bytes
      OVERLONG_2,                                         // 1100[000_]       ________        Could have been encoded in 1 byte
      0, 0,

      TOO_LARGE,                                          // 1111[0100]       (1001|101_)____ > U+10FFFF
      TOO_LARGE_2, TOO_LARGE_2, TOO_LARGE_2,              // 1111[0101..1111] ________        > U+10FFFF
      
      TOO_LARGE_2, TOO_LARGE_2, TOO_LARGE_2, TOO_LARGE_2, // 1111[0101..1111] ________        > U+10FFFF

      TOO_LARGE_2,                                        // 1111[0101..1111] ________        > U+10FFFF
      TOO_LARGE_2 |                                       // 1111[0101..1111] ________        > U+10FFFF
        SURROGATE,                                        // 1110[1010]       101_____        Surrogate pairs
      TOO_LARGE_2, TOO_LARGE_2                            // 1111[0101..1111] ________        > U+10FFFF
    );

    // Errors that apply no matter what the third byte is
    const uint8_t CARRY = OVERLONG_2 | // 1100000_         [____]____        Could have been encoded in 1 byte
                          TOO_LARGE_2; // 1111(0101..1111) [____]____        > U+10FFFF
    // NOTE (@jkeiser): I removed the & 0x0F here because shr<4> already does that. 
    vec_t e_3 = bytes.shr<4>().lookup_16<uint8_t>(
      CARRY, CARRY, CARRY, CARRY,

      CARRY, CARRY, CARRY, CARRY,

      CARRY | OVERLONG_3  // 11100000       [100_]____       Could have been encoded in 2 bytes
            | OVERLONG_4, // 11110000       [1000]____       Could have been encoded in 3 bytes
      CARRY | OVERLONG_3  // 11100000       [100_]____       Could have been encoded in 2 bytes
            | TOO_LARGE,  // 11110100       [1001|101_]____  > U+10FFFF
      CARRY | SURROGATE   // 11101010       [101_]____       Surrogate pairs
            | TOO_LARGE,  // 11110100       [1001|101_]____  > U+10FFFF
      CARRY | SURROGATE   // 11101010       [101_]____       Surrogate pairs
            | TOO_LARGE,  // 11110100       [1001..1111]____ > U+10FFFF

      CARRY, CARRY, CARRY, CARRY
    );

    // Check if any bits are set in all three error masks
    //
    // NOTE (@jkeiser): I turned the if() statement here into this->has_error for performance in
    // success cases: instead of spending time testing the result and introducing a branch (which
    // can affect performance even if it's easily predictable), we test once at the end.
    this->has_error |= e_1 & e_2 & e_3;

    // Save continuation bits and input bytes for the next round
    this->prev_bytes = bytes;
    this->last_cont = req >> sizeof(vec_t);
  }

  really_inline void check_next_input(vec_t bytes) {
    // NOTE (@jkeiser): this ascii fast path is different from the one in simdjson: it triggers less
    // (after two consecutive ascii blocks) but also does less work (the simdjson one will check
    // the previous block's last 4 bytes to see if there are continuations there). It'll be worth
    // checking which of these approaches works better for ASCII files, and which works better for
    // mixed utf-8 / ascii files like twitter.json, and figuring out what to do from there.
    // TODO (@jkeiser): To fit the models together, we check for ASCII on every SIMD block rather
    // than every 64 bytes, as other simdjson checkers do; the reuse of "high" actually makes that
    // more difficult because you have to mask or shift. Come back and fix that if feasible.
    vmask_t high = bytes.high_bits_to_bitmask();
    if (unlikely(high | this->last_cont)) {
      // TODO (@jkeiser): To work with simdjson's caller model, I moved the calculation of
      // shifted_bytes inside check_utf8_bytes. I believe this adds an extra instruction to the hot
      // path, which is undesirable, though 2 register accesses vs. 1 memory access might be a wash.
      // Come back and try the other way.
      this->check_utf8_bytes(bytes, high);
    }
  }

  really_inline void check_next_input(simd8x64<uint8_t> in) {
    in.each([&](auto bytes) { this->check_next_input(bytes); });
  }

  really_inline ErrorValues errors() {
    return this->has_error.any_bits_set_anywhere() ? simdjson::UTF8_ERROR : simdjson::SUCCESS;
  }
}; // struct utf8_checker
