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

struct utf8_checker {
  simd8<uint8_t> has_error;
  simd8<uint8_t> prev_bytes;
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

  // really_inline void check_cont(const simd8<uint8_t> bytes, const vmask_t bit_7) {
  //   // Compute the continuation byte mask by finding bytes that start with
  //   // 11x, 111x, and 1111. For each of these prefixes, we get a bitmask
  //   // and shift it forward by 1, 2, or 3. This loop should be unrolled by
  //   // the compiler, and the (n == 1) branch inside eliminated.
  //   //
  //   // NOTE (@jkeiser): I unrolled the for(i=1..3) loop because I don't trust compiler unrolling
  //   // anymore. This should be exactly equivalent and yield the same optimizations (and also lets
  //   // us rearrange statements if we so desire).

  //   // We add the shifted mask here instead of ORing it, which would
  //   // be the more natural operation, so that this line can be done
  //   // with one lea. While adding could give a different result due
  //   // to carries, this will only happen for invalid UTF-8 sequences,
  //   // and in a way that won't cause it to pass validation. Reasoning:
  //   // Any bits for required continuation bytes come after the bits
  //   // for their leader bytes, and are all contiguous. For a carry to
  //   // happen, two of these bit sequences would have to overlap. If
  //   // this is the case, there is a leader byte before the second set
  //   // of required continuation bytes (and thus before the bit that
  //   // will be cleared by a carry). This leader byte will not be
  //   // in the continuation mask, despite being required. QEDish.
  //   // Which bytes are required to be continuation bytes
  //   vmask2_t cont_required = this->last_cont;

  //   // 2-byte lead: 11______
  //   const vmask_t bit_6 = bytes.get_bit<6>();
  //   const vmask_t lead_2_plus = bit_7 & bit_6;       // 11______
  //   cont_required += vmask2_t(lead_2_plus) << 1;

  //   // 3-byte lead: 111_____
  //   const vmask_t bit_5 = bytes.get_bit<5>();
  //   const vmask_t lead_3_plus = lead_2_plus & bit_5; // 111_____
  //   cont_required += vmask2_t(lead_3_plus) << 2;

  //   // 4-byte lead: 1111____
  //   const vmask_t bit_4 = bytes.get_bit<4>();
  //   const vmask_t lead_4_plus = lead_3_plus & bit_4;
  //   cont_required += vmask2_t(lead_4_plus) << 3;

  //   const vmask_t cont = bit_7 ^ lead_2_plus;        // 10______ TODO &~ bit_6 might be fine, and involve less data dependency

  //   // Check that continuation bytes match. We must cast req from vmask2_t
  //   // (which holds the carry mask in the upper half) to vmask_t, which
  //   // zeroes out the upper bits
  //   //
  //   // NOTE (@jkeiser): I turned the if() statement here into this->has_error for performance in
  //   // success cases: instead of spending time testing the result and introducing a branch (which
  //   // can affect performance even if it's easily predictable), we test once at the end.
  //   // The ^ is equivalent to !=, however, leaving a 1 where the bits are different and 0 where they
  //   // are the same.
  //   this->cont_error |= cont ^ vmask_t(cont_required);
  // }

  // check whether the current bytes are valid UTF-8
  // at the end of the function, previous gets updated
  really_inline void check_utf8_bytes(const simd8<uint8_t> bytes, const vmask_t bit_7) {
    // Count: 14 simd ops, 4 simd constants, 3 movemask, 15 64-bit ops
    const simd8<uint8_t> shifted_bytes = bytes.prev<1>(this->prev_bytes);

    // Compute the continuation byte mask by finding bytes that start with
    // 11x, 111x, and 1111. For each of these prefixes, we get a bitmask
    // and shift it forward by 1, 2, or 3. This loop should be unrolled by
    // the compiler, and the (n == 1) branch inside eliminated.
    //
    // NOTE (@jkeiser): I unrolled the for(i=1..3) loop because I don't trust compiler unrolling
    // anymore. This should be exactly equivalent and yield the same optimizations (and also lets
    // us rearrange statements if we so desire).

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
    // Which bytes are required to be continuation bytes
    vmask2_t cont_required = this->last_cont;

    // 2-byte lead: 11______
    const vmask_t bit_6 = bytes.get_bit<6>();
    const vmask_t lead_2_plus = bit_7 & bit_6;       // 11______
    cont_required += vmask2_t(lead_2_plus) << 1;

    // 3-byte lead: 111_____
    const vmask_t bit_5 = bytes.get_bit<5>();
    const vmask_t lead_3_plus = lead_2_plus & bit_5; // 111_____
    cont_required += vmask2_t(lead_3_plus) << 2;

    // 4-byte lead: 1111____
    const vmask_t bit_4 = bytes.get_bit<4>();
    const vmask_t lead_4_plus = lead_3_plus & bit_4;
    cont_required += vmask2_t(lead_4_plus) << 3;

    const vmask_t cont = bit_7 ^ lead_2_plus;        // 10______ TODO &~ bit_6 might be fine, and involve less data dependency

    // Check that continuation bytes match. We must cast req from vmask2_t
    // (which holds the carry mask in the upper half) to vmask_t, which
    // zeroes out the upper bits
    //
    // NOTE (@jkeiser): I turned the if() statement here into this->has_error for performance in
    // success cases: instead of spending time testing the result and introducing a branch (which
    // can affect performance even if it's easily predictable), we test once at the end.
    // The ^ is equivalent to !=, however, leaving a 1 where the bits are different and 0 where they
    // are the same.
    this->cont_error |= cont ^ vmask_t(cont_required);

    // Look up error masks for three consecutive nibbles. We need to
    // AND with 0x0F for each one, because vpshufb has the neat
    // "feature" that negative values in an index byte will result in 
    // a zero.
    //
    simd8<uint8_t> nibble_1_error = shifted_bytes.shr<4>().lookup_16<uint8_t>(
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

    simd8<uint8_t> nibble_2_error = (shifted_bytes & 0x0F).lookup_16<uint8_t>(
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
    simd8<uint8_t> nibble_3_error = bytes.shr<4>().lookup_16<uint8_t>(
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
    this->has_error |= nibble_1_error & nibble_2_error & nibble_3_error;

    // Save continuation bits and input bytes for the next round
    this->prev_bytes = bytes;
    this->last_cont = cont_required >> sizeof(simd8<uint8_t>);
  }

  really_inline void check_next_input(simd8<uint8_t> bytes) {
    vmask_t bit_7 = bytes.get_bit<7>();
    if (unlikely(bit_7)) {
      // TODO (@jkeiser): To work with simdjson's caller model, I moved the calculation of
      // shifted_bytes inside check_utf8_bytes. I believe this adds an extra instruction to the hot
      // path (saving prev_bytes), which is undesirable, though 2 register accesses vs. 1 memory
      // access might be a wash. Come back and try the other way.
      this->check_utf8_bytes(bytes, bit_7);
    } else {
      this->cont_error |= this->last_cont;
    }
  }

  really_inline void check_next_input(simd8x64<uint8_t> in) {
    in.each([&](auto bytes) { this->check_next_input(bytes); });
  }

  really_inline ErrorValues errors() {
    return (this->has_error.any_bits_set_anywhere() | this->cont_error) ? simdjson::UTF8_ERROR : simdjson::SUCCESS;
  }
}; // struct utf8_checker
