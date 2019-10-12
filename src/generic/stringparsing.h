// This file contains the common code every implementation uses
// It is intended to be included multiple times and compiled multiple times
// We assume the file in which it is include already includes
// "stringparsing.h" (this simplifies amalgation)

// begin copypasta
// These chars yield themselves: " \ /
// b -> backspace, f -> formfeed, n -> newline, r -> cr, t -> horizontal tab
// u not handled in this table as it's complex
static const uint8_t escape_map[256] = {
    0, 0, 0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0,    0, 0,    0, // 0x0.
    0, 0, 0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0,    0, 0,    0,
    0, 0, 0x22, 0, 0,    0, 0,    0, 0, 0, 0, 0, 0,    0, 0,    0x2f,
    0, 0, 0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0,    0, 0,    0,

    0, 0, 0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0,    0, 0,    0, // 0x4.
    0, 0, 0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0x5c, 0, 0,    0, // 0x5.
    0, 0, 0x08, 0, 0,    0, 0x0c, 0, 0, 0, 0, 0, 0,    0, 0x0a, 0, // 0x6.
    0, 0, 0x0d, 0, 0x09, 0, 0,    0, 0, 0, 0, 0, 0,    0, 0,    0, // 0x7.

    0, 0, 0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0,    0, 0,    0,
    0, 0, 0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0,    0, 0,    0,
    0, 0, 0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0,    0, 0,    0,
    0, 0, 0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0,    0, 0,    0,

    0, 0, 0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0,    0, 0,    0,
    0, 0, 0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0,    0, 0,    0,
    0, 0, 0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0,    0, 0,    0,
    0, 0, 0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0,    0, 0,    0,
};

// handle a unicode codepoint
// write appropriate values into dest
// src will advance 6 bytes or 12 bytes
// dest will advance a variable amount (return via pointer)
// return true if the unicode codepoint was valid

// We work in little-endian then swap at write time
really_inline void handle_unicode_codepoint(const uint8_t *& src, uint8_t *&dst, bool &has_error) {
  // hex_to_u32_nocheck fills high 16 bits of the return value with 1s if the
  // conversion isn't valid; we defer the check for this to inside the
  // multilingual plane check
  uint32_t code_point = hex_to_u32_nocheck(src); // \u...
  src += 6;
  // check for low surrogate for characters outside the Basic
  // Multilingual Plane.
  if (code_point >= 0xd800 && code_point < 0xdc00) {
    if ((src[0] != '\\') || (src[1] != 'u')) { // XXXX\u
      has_error = true;
    }
    uint32_t code_point_2 = hex_to_u32_nocheck(&src[2]); // XXXX\u...

    // if the first code point is invalid we will get here, as we will go past
    // the check for being outside the Basic Multilingual plane. If we don't
    // find a \u immediately afterwards we fail out anyhow, but if we do,
    // this check catches both the case of the first code point being invalid
    // or the second code point being invalid.
    if ((code_point | code_point_2) >> 16) {
      has_error = true;
    }

    code_point = (((code_point - 0xd800) << 10) | (code_point_2 - 0xdc00)) + 0x10000;
    src += 6;
  }
  size_t offset = codepoint_to_utf8(code_point, dst);
  dst += offset;
  if (offset == 0) {
    has_error = true;
  }
}

really_inline void parse_backslash(const uint8_t *&src, uint8_t *&dst, bs_and_quote_bits &scanned_bits, bool &has_error) {
  unsigned int bs_dist = scanned_bits.next_backslash();
  src += bs_dist;
  dst += bs_dist; // We've already copied in any non-backslash bits

  // Read the escape character (the n in \n).
  uint8_t escape_char = src[1];
  src += 2;

  // Handle \u separately; it's the only > 1 char escape.
  // I, I took the branch less traveled by. And that has made all the difference.
  if (escape_char == 'u') {
    handle_unicode_codepoint(src, dst, has_error);

  } else {
    // Write out the translated escape character. e.g. \n -> 0x0A
    uint8_t escape_result = escape_map[escape_char];
    has_error = has_error || (escape_result == 0u); // Error if it's an unrecognized escape character.
    dst[0] = escape_result;
    dst++;
  }

  // Copy everything starting after the \x to dst to "fill in the blanks". Use a constant number of
  // bytes; we'll fill in the rest later.
  scanned_bits = find_bs_and_quote_bits(src, dst);
}

WARN_UNUSED
really_inline bool parse_string(UNUSED const uint8_t *buf,
                                UNUSED size_t len, ParsedJson &pj,
                                UNUSED const uint32_t depth,
                                UNUSED uint32_t offset) {
  pj.write_tape(pj.current_string_buf_loc - pj.string_buf, '"');
  uint8_t *dst = pj.current_string_buf_loc + sizeof(uint32_t);
  const uint8_t *const start_of_string = dst;

  // Process the string in blocks, stopping when we find a quote.
  const uint8_t *src = &buf[offset + 1]; /* we know that buf at offset is a " */
  bool has_error = false;
  bs_and_quote_bits scanned_bits;
  do {
    scanned_bits = find_bs_and_quote_bits(src, dst);
    while (scanned_bits.has_backslash_in_string()) {
      parse_backslash(src, dst, scanned_bits, has_error);
    }
    src += bs_and_quote_bits::SCAN_WIDTH;
  } while (!scanned_bits.has_quote());
  dst += scanned_bits.next_quote(); // we've already copied over everything to dst.

  // Write out the length of the string at the *start* of the string.
  uint32_t str_length = dst - start_of_string;
  memcpy(pj.current_string_buf_loc, &str_length, sizeof(uint32_t));
  /*****************************
    * Above, check for overflow in case someone has a crazy string
    * (>=4GB?)                 _
    * But only add the overflow check when the document itself exceeds
    * 4GB
    * Currently unneeded because we refuse to parse docs larger or equal
    * to 4GB.
    ****************************/

  /* NULL termination is still handy if you expect all your strings to be NULL terminated? */
  /* It comes at a small cost */
  dst[0] = '\0';
  dst++;

  // Advance the string tape now that we've written the whole string.
  pj.current_string_buf_loc = dst;

  return !has_error;
}
