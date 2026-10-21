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
// Instead of the delta filter, uint8 chunks can use a per-chunk linear
// predictor (LPC, Predictor::Lpc): the encoder least-squares fits 62 weights
// and an intercept over every causal voxel in the 5x5x5 box around a voxel,
// and codes the prediction residuals with the same rANS frame. On scroll CT
// it is ~18-25% smaller than the delta filter at full resolution, but encodes
// and decodes several times slower. The encoder also builds the delta frame
// and keeps whichever payload is smaller, so Lpc is never worse than Delta.
//
// D3D1 layout (all integers little-endian):
//   0..3   magic 'D' '3' 'D' '1'
//   4      format version (1)
//   5      element size in bytes (1 or 2)
//   6      quantization bin width (0 and 1 both mean lossless)
//   7      bits 0-3: predictor marker, 1 = delta filter, 2 = LPC.
//          Delta: bit 7 set means bits 4-6 record the delta-axis mask (bit 4
//          x, bit 5 y, bit 6 z). When bit 7 is clear, the filter is full zyx.
//          LPC: bits 4-7 are zero.
//   8..19  chunk dims as three uint32: z, y, x (element counts)
//   20..   LPC only: the 129-byte parameter block below
//   then   256 uint16 symbol frequencies summing to 4096, then eight uint64
//          initial rANS states, then the rANS byte stream (decoded back to
//          front by construction)
//
// LPC parameter block (element size must be 1):
//   0      stencil id (1: the 62-tap causal box of radius 2)
//   1..124 62 int16 weights in 1/4096 units, in tap order: dz = -2..0, then
//          dy = -2..2, then dx = -2..2, keeping the causal offsets (dz < 0,
//          or dz = 0 and dy < 0, or dz = 0, dy = 0 and dx < 0)
//   125..  int32 bias in 1/4096 units, |bias| <= 2^30
//
// LPC residuals are r = (v - p) mod 256 in raster order (x fastest). Voxels
// with z >= 2, 2 <= y < Y-2 and 2 <= x < X-2 are predicted as
//   p = (bias + sum_i w_i * v[z+dz_i][y+dy_i][x+dx_i]) >> 12
// (int32 arithmetic, arithmetic shift, no clamp). All other voxels use the
// clamped Lorenzo prediction p = clamp(W + N + U - NW - WU - NU + NWU, 0,
// 255), reading neighbors outside the chunk as 0.

// Predictor applied before entropy coding (header byte 7, bits 0-3).
enum class Predictor { Delta = 1, Lpc = 2 };

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

// Smallest chunk edge for which Predictor::Lpc tries the linear predictor;
// smaller chunks, and uint16 chunks, always use the delta filter.
inline constexpr int kLpcMinDim = 16;

// Compresses one decoded chunk of shapeZYX elements of elemSize bytes.
// quantBinWidth 1 is lossless; larger widths quantize first (see above).
// Throws std::invalid_argument unless input.size() equals z*y*x*elemSize
// with elemSize 1 or 2, and quantBinWidth is in [1, 255].
std::vector<std::byte> compress(std::span<const std::byte> input,
                                std::array<int, 3> shapeZYX,
                                std::size_t elemSize,
                                int quantBinWidth = kQuantizationLossless,
                                Predictor predictor = Predictor::Delta);

// Predictor recorded in a D3D1 payload, or std::nullopt if input is not a
// D3D1 payload.
std::optional<Predictor> predictor(std::span<const std::byte> input);

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
