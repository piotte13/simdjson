// This file contains the common code every implementation uses in stage1
// It is intended to be included multiple times and compiled multiple times
// We assume the file in which it is included already includes
// "simdjson/stage1_find_marks.h" (this simplifies amalgation)

namespace stage1 {

class bit_indexer {
public:
  uint32_t *tail;

  bit_indexer(uint32_t *index_buf) : tail(index_buf) {}

  // flatten out values in 'bits' assuming that they are are to have values of idx
  // plus their position in the bitvector, and store these indexes at
  // base_ptr[base] incrementing base as we go
  // will potentially store extra values beyond end of valid bits, so base_ptr
  // needs to be large enough to handle this
  really_inline void write_indexes(uint32_t idx, uint64_t bits) {
    // In some instances, the next branch is expensive because it is mispredicted.
    // Unfortunately, in other cases,
    // it helps tremendously.
    if (bits == 0)
        return;
    uint32_t cnt = hamming(bits);

    // Do the first 8 all together
    for (int i=0; i<8; i++) {
      this->tail[i] = idx + trailing_zeroes(bits);
      bits = clear_lowest_bit(bits);
    }

    // Do the next 8 all together (we hope in most cases it won't happen at all
    // and the branch is easily predicted).
    if (unlikely(cnt > 8)) {
      for (int i=8; i<16; i++) {
        this->tail[i] = idx + trailing_zeroes(bits);
        bits = clear_lowest_bit(bits);
      }

      // Most files don't have 16+ structurals per block, so we take several basically guaranteed
      // branch mispredictions here. 16+ structurals per block means either punctuation ({} [] , :)
      // or the start of a value ("abc" true 123) every four characters.
      if (unlikely(cnt > 16)) {
        uint32_t i = 16;
        do {
          this->tail[i] = idx + trailing_zeroes(bits);
          bits = clear_lowest_bit(bits);
          i++;
        } while (i < cnt);
      }
    }

    this->tail += cnt;
  }
};

class json_structural_scanner {
public:
  ParsedJson& pj;
  bit_indexer structural_indexes;
  // Errors with unescaped characters in strings (ASCII codepoints < 0x20)
  uint64_t unescaped_chars_error = 0;

  // Whether the first character of the next iteration is escaped.
  uint64_t prev_escaped = 0ULL;
  // Whether the last iteration was still inside a string (all 1's = true, all 0's = false).
  uint64_t prev_in_string = 0ULL;
  // Whether the last character of the previous iteration is a primitive value character
  // (anything except whitespace, braces, comma or colon).
  uint64_t prev_primitive = 0ULL;
  // Mask of structural characters from the last iteration.
  // Kept around for performance reasons, so we can call flatten_bits to soak up some unused
  // CPU capacity while the next iteration is busy with an expensive clmul in compute_quote_mask.
  uint64_t prev_structurals = 0;

  json_structural_scanner(ParsedJson &_pj) : pj{_pj}, structural_indexes{_pj.structural_indexes} {}

  //
  // Return a mask of all string characters plus end quotes.
  //
  // prev_escaped is overflow saying whether the next character is escaped.
  // prev_in_string is overflow saying whether we're still in a string.
  //
  // Backslash sequences outside of quotes will be detected in stage 2.
  //
  really_inline uint64_t find_strings(const simd::simd8x64<uint8_t> in);

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
  really_inline uint64_t find_potential_structurals(const simd::simd8x64<uint8_t> in);

  //
  // Find the important bits of JSON in a STEP_SIZE-byte chunk, and add them to structural_indexes.
  //
  template<size_t STEP_SIZE>
  really_inline void scan_step(const uint8_t *orig_buf, const size_t idx, utf8_checker &utf8_checker, const uint8_t* remainder_buf, bool is_last);

  template<size_t STEP_SIZE>
  really_inline void scan_first(const uint8_t *buf, utf8_checker &utf8_checker) {
    this->scan_step<STEP_SIZE>(buf, 0, utf8_checker, NULL, false);
  }

  template<size_t STEP_SIZE>
  really_inline void scan_next(const uint8_t *buf, const size_t idx, utf8_checker &utf8_checker) {
    this->scan_step<STEP_SIZE>(buf, idx, utf8_checker, NULL, false);
  }

  template<size_t STEP_SIZE>
  really_inline ErrorValues scan_remainder(const uint8_t *buf, const size_t idx, const size_t len, utf8_checker &utf8_checker) {
    uint8_t end_buf[STEP_SIZE];
    memset(end_buf, 0x20, STEP_SIZE);
    memcpy(end_buf, &buf[idx], len - idx);
    this->scan_step<STEP_SIZE>(buf, idx, utf8_checker, end_buf, true);
    return this->scan_eof(idx+STEP_SIZE, len, utf8_checker);
  }

  template<size_t STEP_SIZE>
  really_inline ErrorValues scan_last(const uint8_t *buf, const size_t idx, const size_t len, utf8_checker &utf8_checker) {
    return this->scan_remainder<STEP_SIZE>(buf, idx, len, utf8_checker);
  }

  template<size_t STEP_SIZE>
  really_inline ErrorValues scan_single(const uint8_t *buf, const size_t len, utf8_checker &utf8_checker) {
    return this->scan_remainder<STEP_SIZE>(buf, 0, len, utf8_checker);
  }

  //
  // Finish the scan and return any errors.
  //
  // This may detect errors as well, such as unclosed string and certain UTF-8 errors.
  //
  really_inline ErrorValues scan_eof(const size_t idx, const size_t len, utf8_checker &utf8_checker);

  //
  // Parse the entire input in STEP_SIZE-byte chunks.
  //
  template<size_t STEP_SIZE>
  really_inline ErrorValues scan(const uint8_t *buf, const size_t len, utf8_checker &utf8_checker);
};

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
really_inline uint64_t follows(const uint64_t match, const uint64_t filler, uint64_t &overflow) {
  uint64_t follows_match = follows(match, overflow);
  uint64_t result;
  overflow |= add_overflow(follows_match, filler, &result);
  return result;
}

//
// Return a mask of all string characters plus end quotes.
//
// prev_escaped is overflow saying whether the next character is escaped.
// prev_in_string is overflow saying whether we're still in a string.
//
// Backslash sequences outside of quotes will be detected in stage 2.
//
really_inline uint64_t json_structural_scanner::find_strings(const simd::simd8x64<uint8_t> in) {
  const uint64_t backslash = in.eq('\\');
  const uint64_t escaped = follows_odd_sequence_of(backslash, prev_escaped);
  const uint64_t quote = in.eq('"') & ~escaped;
  // prefix_xor flips on bits inside the string (and flips off the end quote).
  const uint64_t in_string = prefix_xor(quote) ^ prev_in_string;
  /* right shift of a signed value expected to be well-defined and standard
  * compliant as of C++20,
  * John Regher from Utah U. says this is fine code */
  prev_in_string = static_cast<uint64_t>(static_cast<int64_t>(in_string) >> 63);
  // Use ^ to turn the beginning quote off, and the end quote on.
  return in_string ^ quote;
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
really_inline uint64_t json_structural_scanner::find_potential_structurals(const simd::simd8x64<uint8_t> in) {
  // These use SIMD so let's kick them off before running the regular 64-bit stuff ...
  uint64_t whitespace, op;
  find_whitespace_and_operators(in, whitespace, op);

  // Detect the start of a run of primitive characters. Includes numbers, booleans, and strings (").
  // Everything except whitespace, braces, colon and comma.
  const uint64_t primitive = ~(op | whitespace);
  const uint64_t follows_primitive = follows(primitive, prev_primitive);
  const uint64_t start_primitive = primitive & ~follows_primitive;

  // Return final structurals
  return op | start_primitive;
}

//
// Find the important bits of JSON in a 128-byte chunk, and add them to structural_indexes.
//
// PERF NOTES:
// We pipe 2 inputs through these stages:
// 1. Load JSON into registers. This takes a long time and is highly parallelizable, so we load
//    2 inputs' worth at once so that by the time step 2 is looking for them input, it's available.
// 2. Scan the JSON for critical data: strings, primitives and operators. This is the critical path.
//    The output of step 1 depends entirely on this information. These functions don't quite use
//    up enough CPU: the second half of the functions is highly serial, only using 1 execution core
//    at a time. The second input's scans has some dependency on the first ones finishing it, but
//    they can make a lot of progress before they need that information.
// 3. Step 1 doesn't use enough capacity, so we run some extra stuff while we're waiting for that
//    to finish: utf-8 checks and generating the output from the last iteration.
// 
// The reason we run 2 inputs at a time, is steps 2 and 3 are *still* not enough to soak up all
// available capacity with just one input. Running 2 at a time seems to give the CPU a good enough
// workout.
//
template<>
really_inline void json_structural_scanner::scan_step<128>(const uint8_t *orig_buf, const size_t idx, utf8_checker &utf8_checker, const uint8_t* remainder_buf, bool is_last) {
  //
  // Load up all 128 bytes into SIMD registers
  //
  const uint8_t* buf = is_last ? remainder_buf : &orig_buf[idx];
  simd::simd8x64<uint8_t> in_1(&buf[0]);
  simd::simd8x64<uint8_t> in_2(&buf[64]);

  //
  // Find the strings and potential structurals (operators / primitives).
  //
  // This will include false structurals that are *inside* strings--we'll filter strings out
  // before we return.
  //
  uint64_t string_1 = this->find_strings(in_1);
  uint64_t structurals_1 = this->find_potential_structurals(in_1);
  uint64_t string_2 = this->find_strings(in_2);
  uint64_t structurals_2 = this->find_potential_structurals(in_2);

  //
  // Do miscellaneous work while the processor is busy calculating strings and structurals.
  //
  // After that, weed out structurals that are inside strings and find invalid string characters.
  //
  uint64_t unescaped_1 = in_1.lteq(0x1F);
  // NOTE: we're absolutely relying on the these if()s getting optimized away, because 0 is only ever
  // passed explicitly.
  if (idx == 0) {
    utf8_checker.check_first(in_1, buf);
  } else {
    if (is_last) {
      utf8_checker.check_last(in_1, &orig_buf[idx-sizeof(simd8<uint8_t>)], remainder_buf);
    } else {
      // Only pass the buffer is there is something behind it (it's there to get the previous bytes!)
      utf8_checker.check_next(in_1, buf);
    }
    this->structural_indexes.write_indexes(idx-64, this->prev_structurals); // Output *last* iteration's structurals to ParsedJson
  }
  this->prev_structurals = structurals_1 & ~string_1;
  this->unescaped_chars_error |= unescaped_1 & string_1;

  uint64_t unescaped_2 = in_2.lteq(0x1F);
  utf8_checker.check_next(in_2, buf+64);
  this->structural_indexes.write_indexes(idx, this->prev_structurals); // Output *last* iteration's structurals to ParsedJson
  this->prev_structurals = structurals_2 & ~string_2;
  this->unescaped_chars_error |= unescaped_2 & string_2;
}

//
// Find the important bits of JSON in a 64-byte chunk, and add them to structural_indexes.
//
template<>
really_inline void json_structural_scanner::scan_step<64>(const uint8_t *orig_buf, const size_t idx, utf8_checker &utf8_checker, const uint8_t* remainder_buf, bool is_last) {
  //
  // Load up bytes into SIMD registers
  //
  const uint8_t* buf = is_last ? remainder_buf : &orig_buf[idx];
  simd::simd8x64<uint8_t> in_1(&buf[0]);

  //
  // Find the strings and potential structurals (operators / primitives).
  //
  // This will include false structurals that are *inside* strings--we'll filter strings out
  // before we return.
  //
  uint64_t string_1 = this->find_strings(in_1);
  uint64_t structurals_1 = this->find_potential_structurals(in_1);

  //
  // Do miscellaneous work while the processor is busy calculating strings and structurals.
  //
  // After that, weed out structurals that are inside strings and find invalid string characters.
  //
  uint64_t unescaped_1 = in_1.lteq(0x1F);
  // NOTE: we're absolutely relying on the these if()s getting optimized away, because 0 is only ever
  // passed explicitly.
  if (idx == 0) {
    utf8_checker.check_first(in_1, buf);
  } else {
    if (is_last) {
      utf8_checker.check_last(in_1, &orig_buf[idx-sizeof(simd8<uint8_t>)], remainder_buf);
    } else {
      // Only pass the buffer is there is something behind it (it's there to get the previous bytes!)
      utf8_checker.check_next(in_1, &buf[idx]);
    }
    this->structural_indexes.write_indexes(idx-64, this->prev_structurals); // Output *last* iteration's structurals to ParsedJson
  }
  this->prev_structurals = structurals_1 & ~string_1;
  this->unescaped_chars_error |= unescaped_1 & string_1;
}

// Handle EOF (there are still errors to check)
really_inline ErrorValues json_structural_scanner::scan_eof(const size_t idx, const size_t len, utf8_checker &utf8_checker) {
  // Flatten out the remaining structurals from the last iteration
  this->structural_indexes.write_indexes(idx-64, this->prev_structurals);

  // The UTF-8 checker needs to look at the last set of bytes to be sure there's no error.
  if (this->prev_in_string) {
    return UNCLOSED_STRING;
  }
  if (this->unescaped_chars_error) {
    return UNESCAPED_CHARS;
  }

  pj.n_structural_indexes = this->structural_indexes.tail - pj.structural_indexes;
  /* a valid JSON file cannot have zero structural indexes - we should have
   * found something */
  if (unlikely(pj.n_structural_indexes == 0u)) {
    return EMPTY;
  }
  if (unlikely(pj.structural_indexes[pj.n_structural_indexes - 1] > len)) {
    return UNEXPECTED_ERROR;
  }
  if (len != pj.structural_indexes[pj.n_structural_indexes - 1]) {
    /* the string might not be NULL terminated, but we add a virtual NULL
     * ending character. */
    pj.structural_indexes[pj.n_structural_indexes++] = len;
  }
  /* make it safe to dereference one beyond this array */
  pj.structural_indexes[pj.n_structural_indexes] = 0;
  return utf8_checker.check_eof();
}

template<size_t STEP_SIZE>
really_inline ErrorValues json_structural_scanner::scan(const uint8_t *buf, const size_t len, utf8_checker &utf8_checker) {
  if (unlikely(len > pj.byte_capacity)) {
    std::cerr << "Your ParsedJson object only supports documents up to "
              << pj.byte_capacity << " bytes but you are trying to process "
              << len << " bytes" << std::endl;
    return simdjson::CAPACITY;
  }

  // If we have more than one step, we loop.
  if (likely(len > STEP_SIZE)) {

    // Having more than STEP_SIZE bytes means having at least one chunk at the beginning.
    this->scan_first<STEP_SIZE>(buf, utf8_checker);

    // Loop until all except the last chunk.
    size_t idx;
    for (idx = STEP_SIZE; idx < (len - STEP_SIZE); idx += STEP_SIZE) {
      this->scan_next<STEP_SIZE>(buf, idx, utf8_checker);
    }

    // The last step will be between 1 and STEP_SIZE bytes.
    return this->scan_last<STEP_SIZE>(buf, idx, len, utf8_checker);

  } else {
    return this->scan_single<STEP_SIZE>(buf, len, utf8_checker);
  }
}

template<size_t STEP_SIZE>
int find_structural_bits(const uint8_t *buf, size_t len, simdjson::ParsedJson &pj, UNUSED bool streaming) {
  // TODO I imagine this breaks streaming ... but I can't reason about what errors streaming does
  // and does not want to return.
  utf8_checker utf8_checker{};
  json_structural_scanner scanner(pj);
  return scanner.scan<STEP_SIZE>(buf, len, utf8_checker);
}

} // namespace stage1