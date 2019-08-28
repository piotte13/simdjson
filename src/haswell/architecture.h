#ifndef SIMDJSON_HASWELL_ARCHITECTURE_H
#define SIMDJSON_HASWELL_ARCHITECTURE_H

#include "simdjson/common_defs.h"
#include "simdjson/portability.h"

#ifdef IS_X86_64

#include "simdjson/simdjson.h"


namespace simdjson::haswell {

static const Architecture ARCHITECTURE = Architecture::HASWELL;
typedef __m256i simd_u64;
typedef __m256i simd_u8;
typedef uint32_t short_bitmask_t;
static const size_t SIMD_BYTE_WIDTH = sizeof(simd_u8);
static const size_t SIMD_WIDTH = SIMD_BYTE_WIDTH*8;

} // namespace simdjson::haswell


#endif // IS_X86_64

#endif // SIMDJSON_HASWELL_ARCHITECTURE_H
