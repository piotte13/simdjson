#ifndef SIMDJSON_ARM64_ARCHITECTURE_H
#define SIMDJSON_ARM64_ARCHITECTURE_H

#include "simdjson/common_defs.h"
#include "simdjson/portability.h"

#ifdef IS_ARM64

#include "simdjson/simdjson.h"

namespace simdjson::arm64 {

static const Architecture ARCHITECTURE = Architecture::ARM64;
typedef uint64x2_t simd_u64;
typedef uint8x16_t simd_u8;
typedef uint16_t short_bitmask_t;
static const size_t SIMD_BYTE_WIDTH = sizeof(simd_u8);
static const size_t SIMD_WIDTH = SIMD_BYTE_WIDTH*8;

} // namespace simdjson::arm64

#endif // IS_ARM64

#endif // SIMDJSON_ARM64_ARCHITECTURE_H
