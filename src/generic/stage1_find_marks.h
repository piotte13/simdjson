// This file contains the common code every implementation uses in stage1
// It is intended to be included multiple times and compiled multiple times
// We assume the file in which it is included already includes
// "simdjson/stage1_find_marks.h" (this simplifies amalgation)

// Debugging aids
void print_input(const uint8_t* buf) {
  printf("\n");
  printf("%20s %.64s\n", "input", buf);
}
void print_bitmask(const char* name, uint64_t bitmask, uint64_t overflow=0) {
  printf("%20.20s ", name);
  for (int i=0;i<64;i++) {
    if (bitmask & 1) {
      printf("X");
    } else {
      printf(" ");
    }
    bitmask >>= 1;
  }
  if (overflow) {
    printf("X");
  } else {
    printf(" ");
  }
  printf("\n");
}
#define PRINT_BITMASK(name) print_bitmask(#name, name);

// return a bitvector indicating where we have characters that end an odd-length
// sequence of backslashes (and thus change the behavior of the next character
// to follow). A even-length sequence of backslashes, and, for that matter, the
// largest even-length prefix of our odd-length sequence of backslashes, simply
// modify the behavior of the backslashes themselves.
// We also update the prev_iter_ends_odd_backslash reference parameter to
// indicate whether we end an iteration on an odd-length sequence of
// backslashes, which modifies our subsequent search for odd-length
// sequences of backslashes in an obvious way.
really_inline uint64_t follows_odd_sequence_of(const uint64_t match, uint64_t &overflow) {
  const uint64_t even_bits = 0x5555555555555555ULL;
  const uint64_t odd_bits = ~even_bits;
  uint64_t start_edges = match & ~(match << 1);
  /* flip lowest if we have an odd-length run at the end of the prior
   * iteration */
  uint64_t even_start_mask = even_bits ^ overflow;
  uint64_t even_starts = start_edges & even_start_mask;
  uint64_t odd_starts = start_edges & ~even_start_mask;
  uint64_t even_carries = match + even_starts;

  uint64_t odd_carries;
  /* must record the carry-out of our odd-carries out of bit 63; this
   * indicates whether the sense of any edge going to the next iteration
   * should be flipped */
  bool new_overflow = add_overflow(match, odd_starts, &odd_carries);

  odd_carries |= overflow; /* push in bit zero as a
                              * potential end if we had an
                              * odd-numbered run at the
                              * end of the previous
                              * iteration */
  overflow = new_overflow ? 0x1ULL : 0x0ULL;
  uint64_t even_carry_ends = even_carries & ~match;
  uint64_t odd_carry_ends = odd_carries & ~match;
  uint64_t even_start_odd_end = even_carry_ends & odd_bits;
  uint64_t odd_start_even_end = odd_carry_ends & even_bits;
  uint64_t odd_ends = even_start_odd_end | odd_start_even_end;
  return odd_ends;
}

//
// Check if the current character immediately follows a matching character.
//
// For example, this checks for quotes with backslashes in front of them:
//
//     const uint64_t backslashed_quote = in.eq('"') & immediately_follows(in.eq('\'), prev_backslash);
//
really_inline uint64_t follows(const uint64_t match, uint64_t &overflow) {
  const uint64_t result = match << 1 | overflow;
  overflow = match >> 63;
  return result;
}

//
// Check if the current character follows a matching character, with possible "filler" between.
// For example, this checks for empty curly braces, e.g. 
//
//     in.eq('}') & follows(in.eq('['), in.eq(' '), prev_empty_array) // { <whitespace>* }
//
really_inline uint64_t follows(const uint64_t match, const uint64_t filler, uint64_t &overflow ) {
  uint64_t follows_match = follows(match, overflow);
  uint64_t result;
  overflow |= add_overflow(follows_match, filler, &result);
  return result;
}

//
// Detect missing values and operators.
//
// 1. Find missing values: [<operator> <whitespace>*] <operator>
//
//    e.g. {"a": }
//
//    <start of file> is treated like [ or { by initializing prev_desires_value to 1. This will catch
//    ":" and "," at the start of a file.
//
//    <end of file> is treated like ] or } by calling detect_errors_on_eof(). "[1," will leave
//    prev_follows_separator = true and detect_errors_on_eof() will mark that as an error.
//
//    NOTE: Unbalanced {} and [] are handled in stage 2, which includes {<eof> and [<eof>
//
// 2. Find missing operators: <value> <whitespace>+ <value>
//
//    e.g. "hello" "world"
//    e.g. {} 123
//    e.g. [] {}
//
//    All characters except operators and whitespace are primitives. (string, number, true, false,
//    null and even invalid characters: invalid literal characters will be handled in stage 2.)
//
//    This will treat characters inside strings as invalid literals; any errors *inside* strings
//    will be masked away later.
//
//    Rule 1: value must come after an operator.
//
really_inline uint64_t detect_value_sequence_errors(
    const uint64_t open,
    const uint64_t close,
    const uint64_t separator,
    const uint64_t start_primitive,
    const uint64_t whitespace,
    uint64_t &prev_value_required,
    uint64_t &prev_value_allowed) {
  const uint64_t value_required = follows(separator, whitespace, prev_value_required);
  const uint64_t value_allowed = follows(open | separator, whitespace, prev_value_allowed);
  return (close & value_required) |          // } or ] after , or :
         (separator & value_allowed) |       // , or : without a value in front of it
         (start_primitive & ~value_allowed); // value after another value, } or ]
}

really_inline ErrorValues detect_errors_on_eof(
    const uint64_t idx,
    uint64_t &unescaped_chars_error,
    const uint64_t prev_in_string,
    uint64_t &value_sequence_error,
    const uint64_t prev_value_required) {
  const uint64_t eof_error_position = (idx % 64);
  value_sequence_error |= prev_value_required << eof_error_position;

  if (unescaped_chars_error) {
    return UNESCAPED_CHARS;
  }
  if (prev_in_string) {
    return UNCLOSED_STRING;
  }
  if (value_sequence_error) {
    return UNEXPECTED_ERROR;
  }
  return SUCCESS;
}

//
// Return a mask of all string characters plus end quotes.
//
// prev_escaped is overflow saying whether the next character is escaped.
// prev_in_string is overflow saying whether we're still in a string.
//
// Backslash sequences outside of quotes will be detected in stage 2.
//
really_inline uint64_t find_in_string(const simd_input<ARCHITECTURE> in, uint64_t &prev_escaped, uint64_t &prev_in_string) {
  const uint64_t backslash = in.eq('\\');
  const uint64_t escaped = follows_odd_sequence_of(backslash, prev_escaped);
  const uint64_t quote = in.eq('"') & ~escaped;
  // compute_quote_mask returns start quote plus string contents.
  const uint64_t in_string = compute_quote_mask(quote) ^ prev_in_string;
  /* right shift of a signed value expected to be well-defined and standard
   * compliant as of C++20,
   * John Regher from Utah U. says this is fine code */
  prev_in_string = static_cast<uint64_t>(static_cast<int64_t>(in_string) >> 63);
  // Use ^ to turn the beginning quote off, and the end quote on.
  return in_string ^ quote;
}

really_inline uint64_t invalid_string_bytes(const simd_input<ARCHITECTURE> in, const uint64_t quote_mask) {
  /* All Unicode characters may be placed within the
   * quotation marks, except for the characters that MUST be escaped:
   * quotation mark, reverse solidus, and the control characters (U+0000
   * through U+001F).
   * https://tools.ietf.org/html/rfc8259 */
  const uint64_t unescaped = in.lteq(0x1F);
  return quote_mask & unescaped;
}

//
// Determine which characters are *structural*:
// - braces: [] and {}
// - the start of primitives (123, true, false, null)
// - the start of invalid non-whitespace (+, &, ture, UTF-8)
//
// Also detects value sequence errors:
// - two values with no separator between ("hello" "world")
// - separators with no values ([1,] [1,,]and [,2])
//
// This method will find all of the above whether it is in a string or not.
//
// To reduce dependency on the expensive "what is in a string" computation, this method treats the
// contents of a string the same as content outside. Errors and structurals inside the string or on
// the trailing quote will need to be removed later when the correct string information is known.
//
really_inline uint64_t find_structurals(
    const simd_input<ARCHITECTURE> in,
    uint64_t &prev_value_required,
    uint64_t &prev_value_allowed,
    uint64_t &prev_primitive,
    uint64_t &value_sequence_error) {
  // These use SIMD so let's kick them off before running the regular 64-bit stuff ...
  uint64_t whitespace = find_whitespace(in);

  // Get operators, {} [] , and :
  // For braces, we take advantage of a feature of ASCII: [] = 5B and 5D, and {} = 7B and 7D.
  // Thus, turning a brace into a curly is just OR 0x20, and then we can compare to { or }.
  const simd_input<ARCHITECTURE> to_curly = in.bit_or(0x20);
  const uint64_t open = to_curly.eq('{');             // [ and {
  const uint64_t close = to_curly.eq('}');            // ] and }
  const uint64_t colon = in.eq(':');
  const uint64_t separator = colon | in.eq(',');

  // Detect the start of a run of primitive characters. Includes numbers, booleans, and strings (").
  // Everything except whitespace, braces, colon and comma.
  const uint64_t primitive = ~(open | close | separator | whitespace);
  const uint64_t follows_primitive = follows(primitive, prev_primitive);
  const uint64_t start_primitive = primitive & ~follows_primitive;

  // Detect errors in the value sequence now, so we don't have to keep all this information around.
  // All the caller needs is errors and structurals.
  value_sequence_error = detect_value_sequence_errors(
    open, close, separator, start_primitive, whitespace,
    prev_value_required, prev_value_allowed
  );

  // Return final structurals
  return open | close | colon | start_primitive;
}

// Find structural bits in a 64-byte chunk.
really_inline void find_structural_bits_64(
    const uint8_t *buf, const size_t idx, uint32_t *&base_ptr, uint32_t &base,
    uint64_t &prev_escaped, uint64_t &prev_in_string,
    uint64_t &prev_value_required, uint64_t &prev_value_allowed, uint64_t &prev_primitive,
    uint64_t &structurals,
    uint64_t &unescaped_chars_error,
    uint64_t &value_sequence_error,
    utf8_checker<ARCHITECTURE> &utf8_state) {
  // print_input(buf);
  // Validate UTF-8
  const simd_input<ARCHITECTURE> in(buf);
  utf8_state.check_next_input(in);

  // Detect values in strings
  const uint64_t in_string = find_in_string(in, prev_escaped, prev_in_string);
  unescaped_chars_error |= invalid_string_bytes(in, in_string);

  /* take the previous iterations structural bits, not our current
   * iteration, and flatten */
  flatten_bits(base_ptr, base, idx, structurals);

  // find_structurals doesn't use in_string; we filter that out here.
  uint64_t local_value_sequence_error;
  structurals = find_structurals(in, prev_value_required, prev_value_allowed, prev_primitive, local_value_sequence_error);
  // PRINT_BITMASK(structurals)

  structurals &= ~in_string;
  value_sequence_error |= local_value_sequence_error & ~in_string;
}

int find_structural_bits(const uint8_t *buf, size_t len, simdjson::ParsedJson &pj) {
  if (len > pj.byte_capacity) {
    std::cerr << "Your ParsedJson object only supports documents up to "
              << pj.byte_capacity << " bytes but you are trying to process "
              << len << " bytes" << std::endl;
    return simdjson::CAPACITY;
  }
  uint32_t *base_ptr = pj.structural_indexes;
  uint32_t base = 0;
  utf8_checker<ARCHITECTURE> utf8_state;

  /* we have padded the input out to 64 byte multiple with the remainder
   * being zeros persistent state across loop does the last iteration end
   * with an odd-length sequence of backslashes? */

  // Whether the first character of the next iteration is escaped.
  uint64_t prev_escaped = 0;
  // Whether the last iteration was still inside a string (all 1's = true, all 0's = false).
  uint64_t prev_in_string = 0;
  // Whether the last character of the previous iteration is a primitive value character
  // (anything except whitespace, braces, comma or colon).
  uint64_t prev_primitive = 0;
  // Whether the last iteration had an operator (comma or colon) that requires a value.
  uint64_t prev_value_required = 0;
  // Whether the last iteration had an operator (open brace, comma or colon) that *allows* a value.
  uint64_t prev_value_allowed = 1;
  // Mask of structural characters from the last iteration.
  // Kept around for performance reasons, so we can call flatten_bits to soak up some unused
  // CPU capacity while the next iteration is busy with an expensive clmul in compute_quote_mask.
  uint64_t structurals = 0;

  size_t lenminus64 = len < 64 ? 0 : len - 64;
  size_t idx = 0;
  // Errors with unescaped characters in strings (ASCII codepoints < 0x20)
  uint64_t unescaped_chars_error = 0;
  // Errors with missing values or missing operators between values
  uint64_t value_sequence_error = 0;

  for (; idx < lenminus64; idx += 64) {
    find_structural_bits_64(&buf[idx], idx, base_ptr, base,
                            prev_escaped, prev_in_string,
                            prev_value_required, prev_value_allowed, prev_primitive,
                            structurals,
                            unescaped_chars_error, value_sequence_error,
                            utf8_state);
  }
  /* If we have a final chunk of less than 64 bytes, pad it to 64 with
   * spaces  before processing it (otherwise, we risk invalidating the UTF-8
   * checks). */
  if (idx < len) {
    uint8_t tmp_buf[64];
    memset(tmp_buf, 0x20, 64);
    memcpy(tmp_buf, buf + idx, len - idx);
    find_structural_bits_64(&tmp_buf[0], idx, base_ptr, base,
                            prev_escaped, prev_in_string,
                            prev_value_required, prev_value_allowed, prev_primitive,
                            structurals,
                            unescaped_chars_error, value_sequence_error,
                            utf8_state);
    idx += 64;
  }

  // finally, flatten out the remaining structurals from the last iteration
  flatten_bits(base_ptr, base, idx, structurals);

  // Check for errors on eof
  simdjson::ErrorValues error = detect_errors_on_eof(
    len,
    unescaped_chars_error, prev_in_string,
    value_sequence_error, prev_value_required
  );
  if (error != simdjson::SUCCESS) {
    return error;
  }

  pj.n_structural_indexes = base;
  /* a valid JSON file cannot have zero structural indexes - we should have
   * found something */
  if (pj.n_structural_indexes == 0u) {
    return simdjson::EMPTY;
  }
  if (base_ptr[pj.n_structural_indexes - 1] > len) {
    return simdjson::UNEXPECTED_ERROR;
  }
  if (len != base_ptr[pj.n_structural_indexes - 1]) {
    /* the string might not be NULL terminated, but we add a virtual NULL
     * ending character. */
    base_ptr[pj.n_structural_indexes++] = len;
  }
  /* make it safe to dereference one beyond this array */
  base_ptr[pj.n_structural_indexes] = 0;
  return utf8_state.errors();
}
