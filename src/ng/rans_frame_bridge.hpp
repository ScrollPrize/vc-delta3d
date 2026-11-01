#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
// Owning handoff from the one upstream encoder: no oversized caller buffer,
// intermediate copy, or duplicate entropy implementation.
std::vector<std::byte> delta_rans_frame(const uint8_t*,size_t);
