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
// the compressed size of scroll CT data compared to zstd on raw bytes while
// the filter itself runs at memory bandwidth.
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
// Two entropy codecs exist, recorded in header byte 7:
//   Zstd (0): a plain zstd frame. Level 1 both compresses better and runs
//     ~2x faster than level 3 on this data: the delta residuals are
//     noise-like, so the greedy match search at levels 2-12 emits
//     statistically-coincidental short matches whose offsets cost more than
//     huffman-coded literals; level 1 finds almost no matches and lands near
//     the order-0 entropy of the residuals.
//   Rans (1): order-0 rANS (12-bit tables, eight interleaved 64-bit states,
//     32-bit renormalization). zstd's huffman-coded literals cannot spend
//     fractional bits per symbol, which costs ~19% on the sharply peaked
//     residual distributions that quantization produces; rANS codes them at
//     the order-0 entropy floor (~18% smaller chunks at width 3 on dense
//     scroll data, ~9-10% across mixed pyramid levels) with encode faster
//     than and decode comparable to the zstd path. This is the default.
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
//   7      bits 0-3: entropy codec id (0 zstd, 1 rANS; pre-codec payloads
//          carry 0). Bit 7 set means bits 4-6 record the delta-axis mask
//          (bit 4 x, bit 5 y, bit 6 z). Payloads written before per-chunk
//          filter selection have bits 4-7 clear and are always full zyx.
//   8..19  chunk dims as three uint32: z, y, x (element counts)
//   20..   codec 0: zstd frame of the filtered payload
//          codec 1: 256 uint16 symbol frequencies summing to 4096, then
//                   eight uint64 initial rANS states, then the rANS byte
//                   stream (decoded back to front by construction)
inline constexpr int kDefaultZstdLevel = 1;

// Entropy codec for the filtered payload (D3D1 header byte 7).
enum class EntropyCodec : unsigned char { Zstd = 0, Rans = 1 };
inline constexpr EntropyCodec kDefaultEntropyCodec = EntropyCodec::Rans;

// Zarr codec identifier.
inline constexpr const char* kCodecId = "vc-delta3d";

// Common near-lossless quantization widths. Width 1 is lossless; width 2k+1
// bounds the per-voxel error by +-k.
inline constexpr int kQuantizationLossless = 1;
inline constexpr int kQuantizationMaxError1 = 3;
inline constexpr int kQuantizationMaxError2 = 5;

// Compresses one decoded chunk of shapeZYX elements of elemSize bytes.
// quantBinWidth 1 is lossless; larger widths quantize first (see above).
// `level` only applies to the Zstd codec and is ignored for Rans.
// Throws std::invalid_argument unless input.size() equals z*y*x*elemSize
// with elemSize 1 or 2, and quantBinWidth is in [1, 255].
std::vector<std::byte> compress(std::span<const std::byte> input,
                                std::array<int, 3> shapeZYX,
                                std::size_t elemSize,
                                int level = kDefaultZstdLevel,
                                int quantBinWidth = kQuantizationLossless,
                                EntropyCodec codec = kDefaultEntropyCodec);

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

// Entropy codec recorded in a D3D1 payload, or std::nullopt if input is not
// a D3D1 payload or names an unknown codec.
std::optional<EntropyCodec> entropyCodec(std::span<const std::byte> input);

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

} // namespace vc_delta3d
