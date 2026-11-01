#pragma once
#include <cstdint>
#include <span>
#include <vector>
std::vector<std::vector<uint8_t>> ng_cabac_candidates(const uint8_t*, bool verify);
bool ng_cabac_decode(std::span<const uint8_t>, uint8_t*);
extern "C" int codec_cabac_matrix(const uint8_t*, uint64_t* sizes);
