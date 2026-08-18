#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace vc_delta3d {

// Compression for three-dimensional uint8 and uint16 scalar fields. A small
// D3D1 header is followed by an entropy-coded frame of the delta-filtered
// voxels. The
// filter stores each element as the difference from its predecessor along a
// per-chunk subset of the z, y, x axes (mod 2^8/2^16), which roughly halves
// the compressed size of scroll CT data while the filter itself runs at
// memory bandwidth.
//
// Which axes to difference is chosen per chunk (rANS uint8 payloads only;
// everything else stays full zyx): each pass cancels smooth structure along
// its axis but doubles white-noise variance, so on noise-dominated level-0
// scroll data the third pass usually costs more than it saves and a two-axis
// filter wins (occasionally one axis, on sparse or very clean chunks). The
// encoder probes all eight subsets with order-0 histograms of the Lorenzo
// corner residuals over every 4th z-slice, keeps the lowest-entropy one, and
// records it in the header (~14% smaller level-0 chunks than fixed zyx at
// identical fidelity).
//
// Entropy coding uses order-0 rANS with 12-bit tables, eight interleaved
// 64-bit states, and 32-bit renormalization. It codes sharply peaked
// residual distributions at the order-0 entropy floor.
//
// Optionally the voxels are quantized before filtering (near-lossless mode):
// each value is snapped to the nearest multiple of quantBinWidth, so the
// per-voxel error is bounded by quantBinWidth/2 and masked zeros stay
// exactly zero. Width 3 (max error +-1) shrinks scroll CT chunks by a
// further ~20-25% over lossless. The width is recorded in the header so
// recompression can tell what an existing payload already carries;
// quantization is idempotent, so re-encoding at the same width is lossless.
//
// D3D1 layout (all integers little-endian):
//   0..3   magic 'D' '3' 'D' '1'
//   4      format version (1)
//   5      element size in bytes (1 or 2)
//   6      quantization bin width (0 and 1 both mean lossless)
//   7      bits 0-3: rANS marker (1). Bit 7 set means bits 4-6 record the
//          delta-axis mask (bit 4 x, bit 5 y, bit 6 z). When bit 7 is clear,
//          the filter is full zyx.
//   8..19  chunk dims as three uint32: z, y, x (element counts)
//   20..   256 uint16 symbol frequencies summing to 4096, then eight uint64
//          initial rANS states, then the rANS byte stream (decoded back to
//          front by construction)

// Zarr codec identifier.
inline constexpr const char* kCodecId = "vc-delta3d";

// Four-byte identifier at the start of an encoded payload. The magic-aware
// decode entry points allow an embedding application to use the same payload
// layout inside an alternate envelope without rewriting the input buffer.
using WireMagic = std::array<std::byte, 4>;
inline constexpr WireMagic kWireMagic{
    std::byte{'D'}, std::byte{'3'}, std::byte{'D'}, std::byte{'1'}};

// Common near-lossless quantization widths. Width 1 is lossless; width 2k+1
// bounds the per-voxel error by +-k.
inline constexpr int kQuantizationLossless = 1;
inline constexpr int kQuantizationMaxError1 = 3;
inline constexpr int kQuantizationMaxError2 = 5;

// Compresses one decoded chunk of shapeZYX elements of elemSize bytes.
// quantBinWidth 1 is lossless; larger widths quantize first (see above).
// Throws std::invalid_argument unless input.size() equals z*y*x*elemSize
// with elemSize 1 or 2, and quantBinWidth is in [1, 255].
std::vector<std::byte> compress(std::span<const std::byte> input,
                                std::array<int, 3> shapeZYX,
                                std::size_t elemSize,
                                int quantBinWidth = kQuantizationLossless);

// In-place near-lossless quantization as applied by compress: snaps
// each element to the nearest multiple of quantBinWidth (clamped to the
// dtype max; 0 stays 0). Width 1 is a no-op. Exposed so recompression
// tools can compute the expected decoded bytes for verification.
void quantize(std::span<std::byte> data,
              std::size_t elemSize,
              int quantBinWidth);

// Quantization bin width recorded in a D3D1 payload (>= 1), or std::nullopt
// if input is not a D3D1 payload. Payloads written before quantization
// existed report 1 (lossless).
std::optional<int> quantization(std::span<const std::byte> input);

// Delta-axis mask recorded in a D3D1 payload (bit 0 x, bit 1 y, bit 2 z), or
// std::nullopt if input is not a D3D1 payload or predates per-chunk filter
// selection (such payloads are always full zyx). Recompression uses this to
// tell whether re-encoding a payload could still shrink it.
std::optional<int> deltaMask(std::span<const std::byte> input);

// Decompresses a D3D1 payload whose decoded size must equal expectedSize.
// Returns std::nullopt on any error or size mismatch.
std::optional<std::vector<std::byte>> decompress(
    std::span<const std::byte> input,
    std::size_t expectedSize);

// Decompresses directly into caller-owned output storage. Returns false on
// invalid input or size mismatch.
bool decompressInto(std::span<const std::byte> input,
                    std::span<std::byte> output);

// Variants of decompress/decompressInto that validate expectedMagic instead
// of kWireMagic. The input is decoded in place without copying it.
std::optional<std::vector<std::byte>> decompressWithMagic(
    std::span<const std::byte> input,
    std::size_t expectedSize,
    const WireMagic& expectedMagic);

bool decompressIntoWithMagic(std::span<const std::byte> input,
                             std::span<std::byte> output,
                             const WireMagic& expectedMagic);

} // namespace vc_delta3d
