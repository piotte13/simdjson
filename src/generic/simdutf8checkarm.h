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

struct processed_utf_bytes {
  simd8<uint8_t> raw_bytes;
  simd8<int8_t> high_nibbles;
  simd8<int8_t> carried_continuations;
};

static const int8_t _nibbles[] = {
    1, 1, 1, 1, 1, 1, 1, 1, // 0xxx (ASCII)
    0, 0, 0, 0,             // 10xx (continuation)
    2, 2,                   // 110x
    3,                      // 1110
    4,                      // 1111, next should be 0 (not checked here)
};

static const int8_t _initial_mins[] = {
    -128,         -128, -128, -128, -128, -128,
    -128,         -128, -128, -128, -128, -128, // 10xx => false
    (int8_t)0xC2, -128,                         // 110x
    (int8_t)0xE1,                               // 1110
    (int8_t)0xF1,
};

static const int8_t _second_mins[] = {
    -128,         -128, -128, -128, -128, -128,
    -128,         -128, -128, -128, -128, -128, // 10xx => false
    127,          127,                          // 110x => true
    (int8_t)0xA0,                               // 1110
    (int8_t)0x90,
};

struct utf8_checker {
  simd8<uint8_t> has_error;
  processed_utf_bytes previous;

  // all byte values must be no larger than 0xF4
  really_inline void check_smaller_than_0xF4(simd8<uint8_t> current_bytes) {
    // unsigned, saturates to 0 below max
    // this->has_error |= current_bytes.saturating_sub(0xF4u);
    // unsigned, saturates to 0 below max
    this->has_error |= vqsubq_u8(current_bytes, vdupq_n_u8(0xF4));
  }

  really_inline simd8<int8_t> continuation_lengths(simd8<int8_t> high_nibbles) {
    // return high_nibbles.lookup4<int8_t>(
    //   1, 1, 1, 1, 1, 1, 1, 1, // 0xxx (ASCII)
    //   0, 0, 0, 0,             // 10xx (continuation)
    //   2, 2,                   // 110x
    //   3,                      // 1110
    //   4);                     // 1111, next should be 0 (not checked here)
    return vqtbl1q_s8(vld1q_s8(_nibbles), vreinterpretq_u8_s8(high_nibbles));
  }

  really_inline simd8<int8_t> carry_continuations(simd8<int8_t> initial_lengths) {
    // simd8<int8_t> prev_carried_continuations = initial_lengths.prev(this->previous.carried_continuations);
    // simd8<int8_t> right1 = simd8<int8_t>(simd8<uint8_t>(prev_carried_continuations).saturating_sub(1));
    // simd8<int8_t> sum = initial_lengths + right1;

    // simd8<int8_t> prev2_carried_continuations = sum.prev<2>(this->previous.carried_continuations);
    // simd8<int8_t> right2 = simd8<int8_t>(simd8<uint8_t>(prev2_carried_continuations).saturating_sub(2));
    // return sum + right2;
    int8x16_t right1 = vreinterpretq_s8_u8(vqsubq_u8(
        vreinterpretq_u8_s8(vextq_s8(this->previous.carried_continuations, initial_lengths, 16 - 1)),
        vdupq_n_u8(1)));
    int8x16_t sum = vaddq_s8(initial_lengths, right1);

    int8x16_t right2 = vreinterpretq_s8_u8(
        vqsubq_u8(vreinterpretq_u8_s8(vextq_s8(this->previous.carried_continuations, sum, 16 - 2)),
                  vdupq_n_u8(2)));
    return vaddq_s8(sum, right2);
  }

  really_inline void check_continuations(simd8<int8_t> initial_lengths, simd8<int8_t> carries) {
    // overlap || underlap
    // carry > length && length > 0 || !(carry > length) && !(length > 0)
    // (carries > length) == (lengths > 0)
    // (carries > current) == (current > 0)
    // this->has_error |= simd8<uint8_t>(
    //   (carries > initial_lengths) == (initial_lengths > simd8<int8_t>::zero()));
    uint8x16_t overunder = vceqq_u8(vcgtq_s8(carries, initial_lengths),
                                    vcgtq_s8(initial_lengths, vdupq_n_s8(0)));

    this->has_error |= overunder;
  }

  really_inline void check_carried_continuations() {
    // static const int8_t last_1[32] = {
    //   9, 9, 9, 9, 9, 9, 9, 9,
    //   9, 9, 9, 9, 9, 9, 9, 9,
    //   9, 9, 9, 9, 9, 9, 9, 9,
    //   9, 9, 9, 9, 9, 9, 9, 1
    // };
    // this->has_error |= simd8<uint8_t>(this->previous.carried_continuations > simd8<int8_t>(last_1 + 32 - sizeof(simd8<int8_t>)));
      // All bytes are ascii. Therefore the byte that was just before must be
      // ascii too. We only check the byte that was just before simd_input. Nines
      // are arbitrary values.
      const int8x16_t verror =
          (int8x16_t){9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1};
      this->has_error |= vcgtq_s8(this->previous.carried_continuations, verror);
  }

  // when 0xED is found, next byte must be no larger than 0x9F
  // when 0xF4 is found, next byte must be no larger than 0x8F
  // next byte must be continuation, ie sign bit is set, so signed < is ok
  really_inline void check_first_continuation_max(simd8<uint8_t> current_bytes,
                                                  simd8<uint8_t> off1_current_bytes) {
    simd8<bool> prev_ED = off1_current_bytes == 0xEDu;
    simd8<bool> prev_F4 = off1_current_bytes == 0xF4u;
    // Check if ED is followed by A0 or greater
    simd8<bool> ED_too_large = (simd8<int8_t>(current_bytes) > simd8<int8_t>::splat(0x9Fu)) & prev_ED;
    // Check if F4 is followed by 90 or greater
    simd8<bool> F4_too_large = (simd8<int8_t>(current_bytes) > simd8<int8_t>::splat(0x8Fu)) & prev_F4;
    // These will also error if ED or F4 is followed by ASCII, but that's an error anyway
    this->has_error |= simd8<uint8_t>(ED_too_large | F4_too_large);
  }

  // map off1_hibits => error condition
  // hibits     off1    cur
  // C       => < C2 && true
  // E       => < E1 && < A0
  // F       => < F1 && < 90
  // else      false && false
  really_inline void check_overlong(simd8<uint8_t> current_bytes,
                                    simd8<uint8_t> off1_current_bytes,
                                    simd8<int8_t> high_nibbles) {
    // simd8<int8_t> off1_high_nibbles = high_nibbles.prev(this->previous.high_nibbles);

    // // Two-byte characters must start with at least C2
    // // Three-byte characters must start with at least E1
    // // Four-byte characters must start with at least F1
    // simd8<int8_t> initial_mins = off1_high_nibbles.lookup4<int8_t>(
    //   -128, -128, -128, -128, -128, -128, -128, -128, // 0xxx -> false
    //   -128, -128, -128, -128,                         // 10xx -> false
    //   0xC2, -128,                                     // 1100 -> C2
    //   0xE1,                                           // 1110
    //   0xF1                                            // 1111
    // );
    // simd8<bool> initial_under = initial_mins > simd8<int8_t>(off1_current_bytes);

    // // Two-byte characters starting with at least C2 are always OK
    // // Three-byte characters starting with at least E1 must be followed by at least A0
    // // Four-byte characters starting with at least F1 must be followed by at least 90
    // simd8<int8_t> second_mins = off1_high_nibbles.lookup4<int8_t>(
    //   -128, -128, -128, -128, -128, -128, -128, -128, -128, // 0xxx => false
    //   -128, -128, -128,                                     // 10xx => false
    //   127, 127,                                             // 110x => true
    //   0xA0,                                                 // 1110
    //   0x90                                                  // 1111
    // );
    // simd8<bool> second_under = second_mins > simd8<int8_t>(current_bytes);
    // this->has_error |= simd8<uint8_t>(initial_under & second_under);
    int8x16_t off1_high_nibbles = vextq_s8(this->previous.high_nibbles, high_nibbles, 16 - 1);
    int8x16_t initial_mins =
        vqtbl1q_s8(vld1q_s8(_initial_mins), vreinterpretq_u8_s8(off1_high_nibbles));

    uint8x16_t initial_under = vcgtq_s8(initial_mins, vreinterpretq_s8_u8(off1_current_bytes));

    int8x16_t second_mins = vqtbl1q_s8(vld1q_s8(_second_mins), vreinterpretq_u8_s8(off1_high_nibbles));
    uint8x16_t second_under = vcgtq_s8(second_mins, vreinterpretq_s8_u8(current_bytes));
    this->has_error |= vandq_u8(initial_under, second_under);
  }

  really_inline void count_nibbles(simd8<uint8_t> bytes, struct processed_utf_bytes *answer) {
    answer->raw_bytes = bytes;
    // answer->high_nibbles = simd8<int8_t>(bytes.shr<4>() & 0x0F);
    answer->high_nibbles = vreinterpretq_s8_u8(vshrq_n_u8(bytes, 4));
  }

  // check whether the current bytes are valid UTF-8
  // at the end of the function, previous gets updated
  really_inline void check_utf8_bytes(simd8<uint8_t> current_bytes) {
    struct processed_utf_bytes pb {};
    this->count_nibbles(current_bytes, &pb);

    this->check_smaller_than_0xF4(current_bytes);

    simd8<int8_t> initial_lengths = this->continuation_lengths(pb.high_nibbles);

    pb.carried_continuations = this->carry_continuations(initial_lengths);

    this->check_continuations(initial_lengths, pb.carried_continuations);

    // simd8<uint8_t> off1_current_bytes = pb.raw_bytes.prev(this->previous.raw_bytes);
    simd8<uint8_t> off1_current_bytes = vextq_u8(this->previous.raw_bytes, pb.raw_bytes, 16 - 1);
    this->check_first_continuation_max(current_bytes, off1_current_bytes);

    this->check_overlong(current_bytes, off1_current_bytes, pb.high_nibbles);
    this->previous = pb;
  }

  really_inline void check_next_input(simd8<uint8_t> in) {
    if (likely(in.any_bits_set(0x80u))) {
      this->check_carried_continuations();
    } else {
      this->check_utf8_bytes(in);
    }
  }

  really_inline void check_next_input(simd8x64<uint8_t> in) {
    simd8<uint8_t> bits = in.reduce([&](auto a, auto b) { return a | b; });
    if (likely(bits.any_bits_set(0x80u))) {
      // it is ascii, we just check carried continuations.
      this->check_carried_continuations();
    } else {
      // it is not ascii so we have to do heavy work
      in.each([&](auto _in) { this->check_utf8_bytes(_in); });
    }
  }

  really_inline ErrorValues errors() {
    // return this->has_error.any_bits_set() ? simdjson::UTF8_ERROR : simdjson::SUCCESS;
    uint64x2_t v64 = vreinterpretq_u64_u8(this->has_error);
    uint32x2_t v32 = vqmovn_u64(v64);
    uint64x1_t result = vreinterpret_u64_u32(v32);
    return vget_lane_u64(result, 0) != 0 ? simdjson::UTF8_ERROR
                                        : simdjson::SUCCESS;

  }
}; // struct utf8_checker
