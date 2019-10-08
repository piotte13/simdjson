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
WARN_UNUSED
really_inline bool handle_unicode_codepoint(const uint8_t *&src, uint8_t *&dst) {
  // hex_to_u32_nocheck fills high 16 bits of the return value with 1s if the
  // conversion isn't valid; we defer the check for this to inside the
  // multilingual plane check
  src++; // skip the u
  uint32_t code_point = hex_to_u32_nocheck(src);
  src += 4;
  // check for low surrogate for characters outside the Basic
  // Multilingual Plane.
  if (code_point >= 0xd800 && code_point < 0xdc00) {
    if ((src[0] != '\\') || src[1] != 'u') {
      return false;
    }
    src += 2; // skip \u
    uint32_t code_point_2 = hex_to_u32_nocheck(src);
    src += 4;

    // if the first code point is invalid we will get here, as we will go past
    // the check for being outside the Basic Multilingual plane. If we don't
    // find a \u immediately afterwards we fail out anyhow, but if we do,
    // this check catches both the case of the first code point being invalid
    // or the second code point being invalid.
    if ((code_point | code_point_2) >> 16) {
      return false;
    }

    code_point = (((code_point - 0xd800) << 10) | (code_point_2 - 0xdc00)) + 0x10000;
  }
  size_t offset = codepoint_to_utf8(code_point, dst);
  dst += offset;
  return offset > 0;
}

WARN_UNUSED really_inline bool parse_string(UNUSED const uint8_t *buf,
                                            UNUSED size_t len, ParsedJson &pj,
                                            UNUSED const uint32_t depth,
                                            UNUSED uint32_t offset) {
  pj.write_tape(pj.current_string_buf_loc - pj.string_buf, '"');
  utf8_checker utf8;
  const uint8_t *src = &buf[offset + 1]; /* we know that buf at offset is a " */
  uint8_t *dst = pj.current_string_buf_loc + sizeof(uint32_t);
  const uint8_t *start_of_string = dst;
  const uint8_t *scan_start = src;
  scanned_string scanned = scan_string(src, buf+len, utf8);
  while (1) {
    if (((scanned.bs_bits - 1) & scanned.quote_bits) != 0) {

      /* we encountered quotes first. Move dst to point to quotes and exit */

      /* find out where the quote is... */
      uint32_t quote_dist = take_trailing_one(scanned.quote_bits) - (src-scan_start);
      memcpy(dst, src, quote_dist);
      src += quote_dist + 1; // skip past the quote
      dst += quote_dist;

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

      /* NULL termination is still handy if you expect all your strings to
       * be NULL terminated? */
      /* It comes at a small cost */
      dst[quote_dist] = '\0';
      dst++;

      /* we advance the point, accounting for the fact that we have a NULL
       * termination         */
      pj.current_string_buf_loc = dst;
      return utf8.has_any_errors();
    }
    if (((scanned.quote_bits - 1) & scanned.bs_bits) != 0) {
      /* we encountered backslash first. Handle backslash. First find where it is ... */
      uint32_t bs_dist = take_trailing_one(scanned.bs_bits);
      // the next character is escaped; eliminate it from bits so we don't process it.
      scanned.quote_bits &= ~(uint32_t(1) << (bs_dist+1));
      scanned.bs_bits &= ~(uint32_t(1) << (bs_dist+1));

      bs_dist -= (src-scan_start);
      memcpy(dst, src, bs_dist);
      src += bs_dist + 1; // Skip past the backslash
      dst += bs_dist;

      uint8_t escape_char = src[0];
      if (escape_char == 'u') {
        // handle_unicode_codepoint updates src and dst
        if (!handle_unicode_codepoint(src, dst)) {
          return false;
        }
      } else {
        /* simple 1:1 conversion. Will eat bs_dist+2 characters in input and
         * write bs_dist+1 characters to output
         * note this may reach beyond the part of the buffer we've actually
         * seen. I think this is ok */
        uint8_t escape_result = escape_map[escape_char];
        if (escape_result == 0u) {
          return false; /* bogus escape value is an error */
        }
        dst[0] = escape_result;
        src++;
        dst++;
      }

    } else {
      /* they are the same. Since they can't co-occur, it means we
       * encountered neither. */
      // Copy all bytes and jump to the next scan position.
      size_t consumed = src - scan_start;
      if (consumed < scanned.bytes_scanned()) {
        size_t remaining = scanned.bytes_scanned() - consumed;
        memcpy(dst, src, remaining);
        src += remaining;
        dst += remaining;
      }
      scan_start = src;
      scanned = scan_string(src, buf+len, utf8);
    }
  }
  /* can't be reached */
  return true;
}
