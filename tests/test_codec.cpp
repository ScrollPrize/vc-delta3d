// VC-Delta3D codec coverage: D3D1 delta and LPC + rANS and context-model
// (Ctx) roundtrips and corrupt/mismatched payload handling.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <vc_delta3d/codec.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

std::vector<std::byte> smoothVolume(std::array<int, 3> shape, std::size_t elemSize)
{
    // Correlated ramp + noise: representative of CT data and sensitive to
    // axis-order bugs in the filter (values differ along every axis).
    std::vector<std::byte> bytes(
        static_cast<std::size_t>(shape[0]) * shape[1] * shape[2] * elemSize);
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> noise(-3, 3);
    std::size_t i = 0;
    for (int z = 0; z < shape[0]; ++z)
        for (int y = 0; y < shape[1]; ++y)
            for (int x = 0; x < shape[2]; ++x) {
                const int v = 7 * z + 3 * y + 5 * x + noise(rng);
                if (elemSize == 1) {
                    bytes[i++] = static_cast<std::byte>(v & 0xFF);
                } else {
                    const auto value = static_cast<std::uint16_t>(v * 111);
                    std::memcpy(bytes.data() + i, &value, 2);
                    i += 2;
                }
            }
    return bytes;
}

std::span<const std::byte> asSpan(const std::vector<std::byte>& v)
{
    return {v.data(), v.size()};
}

} // namespace

TEST_CASE("D3D1 roundtrip uint8")
{
    const std::array<int, 3> shape{8, 6, 10};
    const auto input = smoothVolume(shape, 1);
    const auto compressed = vc_delta3d::compress(asSpan(input), shape, 1);

    REQUIRE(compressed.size() > 20);
    CHECK(static_cast<char>(compressed[0]) == 'D');
    CHECK(static_cast<char>(compressed[1]) == '3');
    CHECK(static_cast<char>(compressed[2]) == 'D');
    CHECK(static_cast<char>(compressed[3]) == '1');

    const auto decoded = vc_delta3d::decompress(asSpan(compressed), input.size());
    REQUIRE(decoded.has_value());
    CHECK(*decoded == input);
}

TEST_CASE("D3D1 roundtrip uint16")
{
    const std::array<int, 3> shape{5, 7, 9};
    const auto input = smoothVolume(shape, 2);
    const auto compressed = vc_delta3d::compress(asSpan(input), shape, 2);
    const auto decoded = vc_delta3d::decompress(asSpan(compressed), input.size());
    REQUIRE(decoded.has_value());
    CHECK(*decoded == input);
}

TEST_CASE("D3D1 roundtrip single-voxel and flat shapes")
{
    for (const auto shape : {std::array<int, 3>{1, 1, 1},
                             std::array<int, 3>{1, 1, 16},
                             std::array<int, 3>{16, 1, 1},
                             std::array<int, 3>{1, 16, 1}}) {
        const auto input = smoothVolume(shape, 1);
        const auto compressed = vc_delta3d::compress(asSpan(input), shape, 1);
        const auto decoded = vc_delta3d::decompress(asSpan(compressed), input.size());
        REQUIRE(decoded.has_value());
        CHECK(*decoded == input);
    }
}

TEST_CASE("Mismatched shape or element size throws")
{
    const std::array<int, 3> shape{4, 4, 4};
    const auto input = smoothVolume(shape, 1);
    CHECK_THROWS_AS(vc_delta3d::compress(asSpan(input), {4, 4, 5}, 1),
                    std::invalid_argument);
    CHECK_THROWS_AS(vc_delta3d::compress(asSpan(input), shape, 4),
                    std::invalid_argument);
    CHECK_THROWS_AS(vc_delta3d::compress(asSpan(input), {0, 0, 0}, 1),
                    std::invalid_argument);
}

TEST_CASE("Near-lossless quantization bounds the per-voxel error")
{
    const std::array<int, 3> shape{8, 6, 10};
    for (const std::size_t elemSize : {std::size_t{1}, std::size_t{2}}) {
        const auto input = smoothVolume(shape, elemSize);
        for (const int width : {vc_delta3d::kQuantizationMaxError1, vc_delta3d::kQuantizationMaxError2}) {
            const auto compressed =
                vc_delta3d::compress(asSpan(input), shape, elemSize, width);
            CHECK(vc_delta3d::quantization(asSpan(compressed)) == width);

            const auto decoded = vc_delta3d::decompress(asSpan(compressed), input.size());
            REQUIRE(decoded.has_value());
            const int maxErr = width / 2;
            for (std::size_t i = 0; i < input.size(); i += elemSize) {
                int orig = 0, got = 0;
                std::memcpy(&orig, input.data() + i, elemSize);
                std::memcpy(&got, decoded->data() + i, elemSize);
                CHECK(std::abs(orig - got) <= maxErr);
                if (orig == 0)
                    CHECK(got == 0); // masked background must stay exact
            }

            // Idempotent: re-encoding the quantized voxels at the same
            // width must reproduce them exactly.
            const auto again =
                vc_delta3d::compress(asSpan(*decoded), shape, elemSize, width);
            const auto decodedAgain = vc_delta3d::decompress(asSpan(again), input.size());
            REQUIRE(decodedAgain.has_value());
            CHECK(*decodedAgain == *decoded);
        }
    }
}

TEST_CASE("Quantization width is reported for all payload generations")
{
    const std::array<int, 3> shape{4, 4, 4};
    const auto input = smoothVolume(shape, 1);

    auto lossless = vc_delta3d::compress(asSpan(input), shape, 1);
    CHECK(vc_delta3d::quantization(asSpan(lossless)) == vc_delta3d::kQuantizationLossless);

    // Payloads written before quantization existed carry 0 in byte 6 and
    // must read back as lossless.
    lossless[6] = std::byte{0};
    CHECK(vc_delta3d::quantization(asSpan(lossless)) == vc_delta3d::kQuantizationLossless);
    CHECK(vc_delta3d::decompress(asSpan(lossless), input.size()).has_value());

    const std::vector<std::byte> garbage(64, std::byte{0xAB});
    CHECK_FALSE(vc_delta3d::quantization(asSpan(garbage)).has_value());

    CHECK_THROWS_AS(
        vc_delta3d::compress(asSpan(input), shape, 1, 0),
        std::invalid_argument);
    CHECK_THROWS_AS(
        vc_delta3d::compress(asSpan(input), shape, 1, 256),
        std::invalid_argument);
}

TEST_CASE("rANS marker and delta mask are recorded")
{
    const std::array<int, 3> shape{8, 6, 10};
    for (const std::size_t elemSize : {std::size_t{1}, std::size_t{2}}) {
        const auto input = smoothVolume(shape, elemSize);
        const auto encoded =
            vc_delta3d::compress(asSpan(input), shape, elemSize);
        CHECK((static_cast<int>(encoded[7]) & 0x0F) == 1);
        CHECK((static_cast<int>(encoded[7]) & 0x80) == 0x80);
        CHECK(vc_delta3d::deltaMask(asSpan(encoded)).has_value());
        const auto decoded =
            vc_delta3d::decompress(asSpan(encoded), input.size());
        REQUIRE(decoded.has_value());
        CHECK(*decoded == input);
    }
}

TEST_CASE("Per-chunk delta mask roundtrips on directional data")
{
    // Patterns whose lowest-entropy filter differs (smooth along one axis,
    // noisy along the others, pure noise, near-constant): whatever mask the
    // probe picks, decode must reproduce the input exactly and the payload
    // must record a mask.
    const std::array<int, 3> shape{16, 12, 14};
    const std::size_t n = 16 * 12 * 14;
    std::mt19937 rng(7);
    for (int pattern = 0; pattern < 4; ++pattern) {
        std::vector<std::byte> input(n);
        std::size_t i = 0;
        for (int z = 0; z < shape[0]; ++z)
            for (int y = 0; y < shape[1]; ++y)
                for (int x = 0; x < shape[2]; ++x) {
                    int v = 0;
                    switch (pattern) {
                    case 0: v = 16 * z + static_cast<int>(rng() % 3); break;
                    case 1: v = 16 * x + static_cast<int>(rng() % 3); break;
                    case 2: v = static_cast<int>(rng() % 256); break;
                    case 3: v = 100; break;
                    }
                    input[i++] = static_cast<std::byte>(v & 0xFF);
                }
        const auto compressed = vc_delta3d::compress(asSpan(input), shape, 1);
        const auto mask = vc_delta3d::deltaMask(asSpan(compressed));
        REQUIRE(mask.has_value());
        CHECK(*mask >= 0);
        CHECK(*mask <= 7);
        const auto decoded = vc_delta3d::decompress(asSpan(compressed), n);
        REQUIRE(decoded.has_value());
        CHECK(*decoded == input);
    }
}

TEST_CASE("rANS payloads without a mask decode as full zyx")
{
    // uint16 payloads never probe, so their filter is the fixed zyx cascade;
    // clearing the header's mask bits reconstructs a payload written before
    // per-chunk selection existed and it must still decode.
    const std::array<int, 3> shape{6, 5, 7};
    const auto input = smoothVolume(shape, 2);
    auto compressed = vc_delta3d::compress(asSpan(input), shape, 2);
    REQUIRE(vc_delta3d::deltaMask(asSpan(compressed)) == 7);

    compressed[7] = std::byte{1};
    CHECK_FALSE(vc_delta3d::deltaMask(asSpan(compressed)).has_value());
    const auto decoded = vc_delta3d::decompress(asSpan(compressed), input.size());
    REQUIRE(decoded.has_value());
    CHECK(*decoded == input);

    // Mask bits without the flag bit are not a valid codec byte.
    compressed[7] = std::byte{0x51};
    CHECK_FALSE(vc_delta3d::decompress(asSpan(compressed), input.size()).has_value());
}

TEST_CASE("rANS handles degenerate and incompressible payloads")
{
    // Constant chunk: a single symbol owns the whole frequency table.
    const std::array<int, 3> shape{16, 16, 16};
    std::vector<std::byte> zeros(16 * 16 * 16, std::byte{0});
    const auto czero = vc_delta3d::compress(asSpan(zeros), shape, 1);
    const auto dzero = vc_delta3d::decompress(asSpan(czero), zeros.size());
    REQUIRE(dzero.has_value());
    CHECK(*dzero == zeros);

    // Uniform random bytes: near-incompressible, exercises the worst-case
    // output bound and full 256-symbol tables.
    std::vector<std::byte> noise(16 * 16 * 16);
    std::mt19937 rng(99);
    for (auto& b : noise) b = static_cast<std::byte>(rng() & 0xFF);
    const auto cnoise = vc_delta3d::compress(asSpan(noise), shape, 1);
    const auto dnoise = vc_delta3d::decompress(asSpan(cnoise), noise.size());
    REQUIRE(dnoise.has_value());
    CHECK(*dnoise == noise);
}

TEST_CASE("Corrupt rANS payloads and invalid markers return nullopt")
{
    const std::array<int, 3> shape{8, 8, 8};
    const auto input = smoothVolume(shape, 1);
    const auto good = vc_delta3d::compress(asSpan(input), shape, 1);
    REQUIRE(vc_delta3d::decompress(asSpan(good), input.size()).has_value());

    // Truncated stream.
    std::vector<std::byte> truncated(good.begin(),
                                     good.begin() + good.size() - 5);
    CHECK_FALSE(vc_delta3d::decompress(asSpan(truncated), input.size()).has_value());

    // Trailing garbage: the stream must be consumed exactly.
    auto padded = good;
    padded.insert(padded.end(), 8, std::byte{0x55});
    CHECK_FALSE(vc_delta3d::decompress(asSpan(padded), input.size()).has_value());

    // Frequency table not summing to 4096.
    auto badTable = good;
    badTable[20] = static_cast<std::byte>(
        static_cast<unsigned char>(badTable[20]) ^ 0x01);
    CHECK_FALSE(vc_delta3d::decompress(asSpan(badTable), input.size()).has_value());

    // Invalid entropy marker.
    auto badMarker = good;
    badMarker[7] = std::byte{0x7E};
    CHECK_FALSE(vc_delta3d::decompress(asSpan(badMarker), input.size()).has_value());
}

TEST_CASE("Corrupt and mismatched payloads return nullopt")
{
    const std::array<int, 3> shape{4, 4, 4};
    const auto input = smoothVolume(shape, 1);
    const auto compressed = vc_delta3d::compress(asSpan(input), shape, 1);

    // Wrong expected size (header dims disagree).
    CHECK_FALSE(vc_delta3d::decompress(asSpan(compressed), input.size() + 1).has_value());

    // Truncated rANS frame.
    std::vector<std::byte> truncated(compressed.begin(),
                                     compressed.begin() + compressed.size() / 2);
    CHECK_FALSE(vc_delta3d::decompress(asSpan(truncated), input.size()).has_value());

    // Garbage bytes without a D3D1 header.
    std::vector<std::byte> garbage(64, std::byte{0xAB});
    CHECK_FALSE(vc_delta3d::decompress(asSpan(garbage), input.size()).has_value());
}

TEST_CASE("Alternate wire magic decodes without rewriting the input")
{
    const std::array<int, 3> shape{4, 5, 6};
    const auto input = smoothVolume(shape, 1);
    auto encoded = vc_delta3d::compress(asSpan(input), shape, 1);
    constexpr vc_delta3d::WireMagic alternateMagic{
        std::byte{'A'}, std::byte{'L'}, std::byte{'T'}, std::byte{'1'}};
    std::copy(alternateMagic.begin(), alternateMagic.end(), encoded.begin());

    CHECK_FALSE(
        vc_delta3d::decompress(asSpan(encoded), input.size()).has_value());

    std::vector<std::byte> output(input.size());
    CHECK(vc_delta3d::decompressIntoWithMagic(
        asSpan(encoded), output, alternateMagic));
    CHECK(output == input);

    const auto allocated = vc_delta3d::decompressWithMagic(
        asSpan(encoded), input.size(), alternateMagic);
    REQUIRE(allocated.has_value());
    CHECK(*allocated == input);
}

namespace {

// Scroll-like uint8 chunk: smooth structure along all axes, noise, and a
// masked-out (zero) region, so the LPC fit has something to predict and the
// Lorenzo border, interior and masked voxels are all exercised.
std::vector<std::byte> scrollLikeVolume(std::array<int, 3> shape, unsigned seed)
{
    std::vector<std::byte> bytes(
        static_cast<std::size_t>(shape[0]) * shape[1] * shape[2]);
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, 4.0);
    std::size_t i = 0;
    for (int z = 0; z < shape[0]; ++z)
        for (int y = 0; y < shape[1]; ++y)
            for (int x = 0; x < shape[2]; ++x) {
                double v = 120 + 60 * std::sin(0.21 * x + 0.13 * y) *
                                     std::cos(0.17 * z) +
                           noise(rng);
                if (x + y > shape[1] + shape[2] - 12)
                    v = 0;  // masked corner
                bytes[i++] = static_cast<std::byte>(std::clamp(
                    static_cast<int>(std::lround(v)), 0, 255));
            }
    return bytes;
}

} // namespace

TEST_CASE("LPC roundtrips and beats the delta filter on smooth data")
{
    // At 16^3 the 129-byte parameter block and the Lorenzo border (half the
    // voxels) cost more than the predictor saves; from ~20^3 LPC wins.
    for (const auto shape : {std::array<int, 3>{20, 20, 20},
                             std::array<int, 3>{17, 33, 20},
                             std::array<int, 3>{64, 64, 64},
                             std::array<int, 3>{20, 18, 130},
                             std::array<int, 3>{40, 40, 18}}) {  // scalar rows
        const auto input = scrollLikeVolume(shape, 3);
        const auto delta = vc_delta3d::compress(asSpan(input), shape, 1);
        const auto lpc = vc_delta3d::compress(
            asSpan(input), shape, 1, vc_delta3d::kQuantizationLossless,
            vc_delta3d::Predictor::Lpc);
        CHECK(vc_delta3d::predictor(asSpan(lpc)) == vc_delta3d::Predictor::Lpc);
        CHECK(lpc.size() < delta.size());
        CHECK((static_cast<int>(lpc[7]) & 0xFF) == 2);
        CHECK(static_cast<int>(lpc[20]) == 2);  // stencil id
        CHECK_FALSE(vc_delta3d::deltaMask(asSpan(lpc)).has_value());
        const auto decoded = vc_delta3d::decompress(asSpan(lpc), input.size());
        REQUIRE(decoded.has_value());
        CHECK(*decoded == input);
    }
}

TEST_CASE("LPC never produces a larger payload than the delta filter")
{
    // Noise, constant and all-zero chunks give LPC nothing to gain; the
    // encoder must fall back to the delta payload, which still decodes.
    const std::array<int, 3> shape{16, 16, 16};
    const std::size_t n = 16 * 16 * 16;
    std::mt19937 rng(5);
    std::vector<std::byte> noise(n), constant(n, std::byte{77}),
        zeros(n, std::byte{0});
    for (auto& b : noise) b = static_cast<std::byte>(rng() & 0xFF);
    for (const auto* input : {&noise, &constant, &zeros}) {
        const auto delta = vc_delta3d::compress(asSpan(*input), shape, 1);
        const auto lpc = vc_delta3d::compress(
            asSpan(*input), shape, 1, vc_delta3d::kQuantizationLossless,
            vc_delta3d::Predictor::Lpc);
        CHECK(lpc.size() <= delta.size());
        const auto decoded = vc_delta3d::decompress(asSpan(lpc), n);
        REQUIRE(decoded.has_value());
        CHECK(*decoded == *input);
    }
}

TEST_CASE("LPC falls back to the delta filter where it does not apply")
{
    // uint16 chunks and chunks below kLpcMinDim along any axis.
    const std::array<int, 3> small{15, 32, 32};
    const auto input8 = scrollLikeVolume(small, 4);
    const auto c8 = vc_delta3d::compress(
        asSpan(input8), small, 1, vc_delta3d::kQuantizationLossless,
        vc_delta3d::Predictor::Lpc);
    CHECK(vc_delta3d::predictor(asSpan(c8)) == vc_delta3d::Predictor::Delta);

    const std::array<int, 3> shape16{16, 16, 16};
    const auto input16 = smoothVolume(shape16, 2);
    const auto c16 = vc_delta3d::compress(
        asSpan(input16), shape16, 2, vc_delta3d::kQuantizationLossless,
        vc_delta3d::Predictor::Lpc);
    CHECK(vc_delta3d::predictor(asSpan(c16)) == vc_delta3d::Predictor::Delta);
    const auto decoded = vc_delta3d::decompress(asSpan(c16), input16.size());
    REQUIRE(decoded.has_value());
    CHECK(*decoded == input16);
}

TEST_CASE("LPC with near-lossless quantization bounds the error")
{
    const std::array<int, 3> shape{24, 24, 24};
    const auto input = scrollLikeVolume(shape, 6);
    const int width = vc_delta3d::kQuantizationMaxError1;
    const auto compressed = vc_delta3d::compress(
        asSpan(input), shape, 1, width, vc_delta3d::Predictor::Lpc);
    CHECK(vc_delta3d::quantization(asSpan(compressed)) == width);
    const auto decoded = vc_delta3d::decompress(asSpan(compressed), input.size());
    REQUIRE(decoded.has_value());
    auto expected = input;
    vc_delta3d::quantize(expected, 1, width);
    CHECK(*decoded == expected);
}

TEST_CASE("Corrupt LPC payloads return nullopt")
{
    const std::array<int, 3> shape{24, 24, 24};
    const auto input = scrollLikeVolume(shape, 8);
    const auto good = vc_delta3d::compress(
        asSpan(input), shape, 1, vc_delta3d::kQuantizationLossless,
        vc_delta3d::Predictor::Lpc);
    REQUIRE(vc_delta3d::predictor(asSpan(good)) == vc_delta3d::Predictor::Lpc);
    REQUIRE(vc_delta3d::decompress(asSpan(good), input.size()).has_value());

    // Unknown stencil ids, including 1 (the pre-release 62-tap layout).
    for (const auto id : {std::byte{1}, std::byte{9}}) {
        auto badStencil = good;
        badStencil[20] = id;
        CHECK_FALSE(vc_delta3d::decompress(asSpan(badStencil), input.size()).has_value());
    }

    // Tap mask bits beyond the 62 taps.
    auto badMask = good;
    badMask[20 + 8] |= std::byte{0x40};
    CHECK_FALSE(vc_delta3d::decompress(asSpan(badMask), input.size()).has_value());

    // LPC marker with stray high bits, and an LPC payload claiming uint16.
    auto badMarker = good;
    badMarker[7] = std::byte{0x82};
    CHECK_FALSE(vc_delta3d::decompress(asSpan(badMarker), input.size()).has_value());
    auto badElem = good;
    badElem[5] = std::byte{2};
    CHECK_FALSE(vc_delta3d::decompress(asSpan(badElem), 2 * input.size()).has_value());

    std::vector<std::byte> truncated(good.begin(), good.begin() + 100);
    CHECK_FALSE(vc_delta3d::decompress(asSpan(truncated), input.size()).has_value());
}

namespace {

constexpr std::array<int, 3> kCtxShape{128, 128, 128};
constexpr std::size_t kCtxVoxels = std::size_t{128} * 128 * 128;

std::vector<std::byte> compressCtx(const std::vector<std::byte>& input,
                                   std::array<int, 3> shape = kCtxShape,
                                   std::size_t elemSize = 1,
                                   int quant = vc_delta3d::kQuantizationLossless)
{
    return vc_delta3d::compress(asSpan(input), shape, elemSize, quant,
                                vc_delta3d::Predictor::Ctx);
}

} // namespace

TEST_CASE("Ctx roundtrips 128^3 chunks and beats LPC on smooth data")
{
    const auto input = scrollLikeVolume(kCtxShape, 11);
    const auto ctx = compressCtx(input);
    const auto lpc = vc_delta3d::compress(
        asSpan(input), kCtxShape, 1, vc_delta3d::kQuantizationLossless,
        vc_delta3d::Predictor::Lpc);
    CHECK(vc_delta3d::predictor(asSpan(ctx)) == vc_delta3d::Predictor::Ctx);
    CHECK((static_cast<int>(ctx[7]) & 0xFF) == 3);
    CHECK_FALSE(vc_delta3d::deltaMask(asSpan(ctx)).has_value());
    // The volcodec-ng stream follows the D3D1 header: "VCNG", revision 2, mode 6.
    CHECK(std::memcmp(ctx.data() + 20, "VCNG", 4) == 0);
    CHECK(static_cast<int>(ctx[24]) == 2);
    CHECK(static_cast<int>(ctx[25]) == 6);
    CHECK(ctx.size() < lpc.size());
    const auto decoded = vc_delta3d::decompress(asSpan(ctx), input.size());
    REQUIRE(decoded.has_value());
    CHECK(*decoded == input);
    CHECK(compressCtx(input) == ctx);  // deterministic
}

TEST_CASE("Ctx stores constant and incompressible chunks compactly")
{
    std::mt19937 rng(12);
    std::vector<std::byte> noise(kCtxVoxels), constant(kCtxVoxels, std::byte{91});
    for (auto& b : noise) b = static_cast<std::byte>(rng() & 0xFF);
    const auto c = compressCtx(constant);
    CHECK(c.size() == 20 + 12 + 1);  // mode 3: one byte
    const auto r = compressCtx(noise);
    CHECK(r.size() == 20 + 12 + kCtxVoxels);  // mode 2: raw
    for (const auto* pair : {&c, &r}) {
        const auto& input = pair == &c ? constant : noise;
        const auto decoded = vc_delta3d::decompress(asSpan(*pair), kCtxVoxels);
        REQUIRE(decoded.has_value());
        CHECK(*decoded == input);
    }
}

TEST_CASE("Ctx falls back to LPC where it does not apply")
{
    // Any shape other than 128^3 gets the Lpc behavior, uint16 the delta filter.
    for (const auto shape : {std::array<int, 3>{64, 64, 64},
                             std::array<int, 3>{128, 128, 127},
                             std::array<int, 3>{129, 128, 128}}) {
        CAPTURE(shape[0]); CAPTURE(shape[2]);
        const auto input = scrollLikeVolume(shape, 13);
        const auto ctx = compressCtx(input, shape);
        const auto lpc = vc_delta3d::compress(
            asSpan(input), shape, 1, vc_delta3d::kQuantizationLossless,
            vc_delta3d::Predictor::Lpc);
        CHECK(ctx == lpc);
        CHECK(vc_delta3d::predictor(asSpan(ctx)) == vc_delta3d::Predictor::Lpc);
    }
    const auto input16 = smoothVolume(kCtxShape, 2);
    const auto c16 = compressCtx(input16, kCtxShape, 2);
    CHECK(vc_delta3d::predictor(asSpan(c16)) == vc_delta3d::Predictor::Delta);
    const auto decoded = vc_delta3d::decompress(asSpan(c16), input16.size());
    REQUIRE(decoded.has_value());
    CHECK(*decoded == input16);
}

TEST_CASE("Ctx with near-lossless quantization takes the Lpc path")
{
    // On quantized voxels the context model codes worse than the delta filter
    // (multiples of the bin width defeat its fitted predictor), so a quantized
    // Ctx chunk is exactly the Lpc payload, which bounds the error as usual.
    const auto input = scrollLikeVolume(kCtxShape, 14);
    const int width = vc_delta3d::kQuantizationMaxError1;
    const auto compressed = compressCtx(input, kCtxShape, 1, width);
    CHECK(compressed == vc_delta3d::compress(asSpan(input), kCtxShape, 1, width,
                                             vc_delta3d::Predictor::Lpc));
    CHECK(vc_delta3d::quantization(asSpan(compressed)) == width);
    CHECK(compressed.size() < compressCtx(input).size());
    const auto decoded = vc_delta3d::decompress(asSpan(compressed), input.size());
    REQUIRE(decoded.has_value());
    auto expected = input;
    vc_delta3d::quantize(expected, 1, width);
    CHECK(*decoded == expected);
}

TEST_CASE("Ctx applies to every channel of a container")
{
    const int channels = 2;
    std::vector<std::byte> input;
    for (int c = 0; c < channels; ++c) {
        const auto field = scrollLikeVolume(kCtxShape, 15 + c);
        input.insert(input.end(), field.begin(), field.end());
    }
    const auto packed = vc_delta3d::compressChannels(
        asSpan(input), kCtxShape, channels, vc_delta3d::ChannelLayout::First, 1,
        vc_delta3d::kQuantizationLossless, vc_delta3d::Predictor::Ctx);
    std::size_t offset = 24 + 4 * channels;
    for (int c = 0; c < channels; ++c) {
        const auto single = compressCtx(std::vector<std::byte>(
            input.begin() + c * kCtxVoxels, input.begin() + (c + 1) * kCtxVoxels));
        CHECK(vc_delta3d::predictor(asSpan(single)) == vc_delta3d::Predictor::Ctx);
        REQUIRE(offset + single.size() <= packed.size());
        CHECK(std::equal(single.begin(), single.end(), packed.begin() + offset));
        offset += single.size();
    }
    CHECK(offset == packed.size());
    const auto decoded = vc_delta3d::decompress(asSpan(packed), input.size());
    REQUIRE(decoded.has_value());
    CHECK(*decoded == input);
}

TEST_CASE("Corrupt Ctx payloads return nullopt")
{
    const auto input = scrollLikeVolume(kCtxShape, 16);
    const auto good = compressCtx(input);
    REQUIRE(vc_delta3d::decompress(asSpan(good), input.size()).has_value());

    // Ctx marker with stray high bits, a uint16 Ctx payload, dims other than 128^3.
    auto badMarker = good;
    badMarker[7] = std::byte{0x83};
    CHECK_FALSE(vc_delta3d::decompress(asSpan(badMarker), input.size()).has_value());
    auto badElem = good;
    badElem[5] = std::byte{2};
    CHECK_FALSE(vc_delta3d::decompress(asSpan(badElem), 2 * input.size()).has_value());
    auto badDims = good;
    badDims[16] = std::byte{64};
    CHECK_FALSE(vc_delta3d::decompress(asSpan(badDims), input.size() / 2).has_value());

    // A damaged volcodec-ng header, and truncated streams.
    auto badMagic = good;
    badMagic[20] = std::byte{'X'};
    CHECK_FALSE(vc_delta3d::decompress(asSpan(badMagic), input.size()).has_value());
    auto badMode = good;
    badMode[25] = std::byte{4};
    CHECK_FALSE(vc_delta3d::decompress(asSpan(badMode), input.size()).has_value());
    for (const std::size_t keep : {std::size_t{20}, std::size_t{31}, std::size_t{40},
                                   good.size() / 2, good.size() - 1}) {
        CAPTURE(keep);
        std::vector<std::byte> truncated(good.begin(), good.begin() + keep);
        CHECK_FALSE(vc_delta3d::decompress(asSpan(truncated), input.size()).has_value());
    }

    // Flipped bits in the stream must not crash the decoder (whether it
    // notices depends on where they land).
    std::mt19937 rng(17);
    for (int i = 0; i < 200; ++i) {
        auto flipped = good;
        const std::size_t at = 26 + rng() % (good.size() - 26);
        flipped[at] ^= static_cast<std::byte>(1u << (rng() % 8));
        (void)vc_delta3d::decompress(asSpan(flipped), input.size());
    }
}


namespace {

// `channels` distinct smooth fields, channels first (c, z, y, x) or last (z, y, x, c).
std::vector<std::byte> channelVolume(std::array<int, 3> shape, int channels,
                                     std::size_t elemSize, bool last)
{
    const std::size_t n = static_cast<std::size_t>(shape[0]) * shape[1] * shape[2];
    std::vector<std::byte> out(n * channels * elemSize);
    for (int c = 0; c < channels; ++c) {
        auto field = smoothVolume(shape, elemSize);
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t dst = last ? (i * channels + c) : (c * n + i);
            for (std::size_t b = 0; b < elemSize; ++b)
                out[dst * elemSize + b] = static_cast<std::byte>(
                    static_cast<unsigned>(field[i * elemSize + b]) + (b == 0 ? 17u * c : 0u));
        }
    }
    return out;
}

} // namespace

TEST_CASE("Multi-channel containers roundtrip in both layouts")
{
    const std::array<int, 3> shape{6, 5, 7};
    for (std::size_t elemSize : {std::size_t{1}, std::size_t{2}})
        for (int channels : {1, 3, 8})
            for (bool last : {false, true}) {
                CAPTURE(elemSize); CAPTURE(channels); CAPTURE(last);
                const auto layout = last ? vc_delta3d::ChannelLayout::Last
                                         : vc_delta3d::ChannelLayout::First;
                const auto input = channelVolume(shape, channels, elemSize, last);
                const auto packed = vc_delta3d::compressChannels(asSpan(input), shape, channels,
                                                                 layout, elemSize);
                CHECK(static_cast<int>(packed[4]) == 2);
                CHECK(vc_delta3d::channelCount(asSpan(packed)) == channels);
                CHECK(vc_delta3d::quantization(asSpan(packed)) == 1);
                CHECK_FALSE(vc_delta3d::deltaMask(asSpan(packed)).has_value());
                const auto decoded = vc_delta3d::decompress(asSpan(packed), input.size());
                REQUIRE(decoded.has_value());
                CHECK(*decoded == input);
            }
}

TEST_CASE("A container holds one single-field payload per channel")
{
    const std::array<int, 3> shape{6, 5, 7};
    const int channels = 4;
    const auto input = channelVolume(shape, channels, 1, false);
    const auto packed = vc_delta3d::compressChannels(asSpan(input), shape, channels,
                                                     vc_delta3d::ChannelLayout::First, 1);
    const std::size_t field = 6 * 5 * 7;
    std::size_t expected = 24 + 4 * channels;
    for (int c = 0; c < channels; ++c)
        expected += vc_delta3d::compress({input.data() + c * field, field}, shape, 1).size();
    CHECK(packed.size() == expected);
    const auto single = vc_delta3d::compress({input.data(), field}, shape, 1);
    CHECK(static_cast<int>(single[4]) == 1);
    CHECK(vc_delta3d::channelCount(asSpan(single)) == 1);
}

TEST_CASE("LPC applies to every channel of a container")
{
    const std::array<int, 3> shape{20, 24, 22};
    const int channels = 3;
    std::vector<std::byte> input;
    for (int c = 0; c < channels; ++c) {
        const auto field = scrollLikeVolume(shape, 3 + c);
        input.insert(input.end(), field.begin(), field.end());
    }
    const auto first = vc_delta3d::ChannelLayout::First;
    const auto delta = vc_delta3d::compressChannels(asSpan(input), shape, channels, first, 1);
    const auto lpc = vc_delta3d::compressChannels(
        asSpan(input), shape, channels, first, 1, vc_delta3d::kQuantizationLossless,
        vc_delta3d::Predictor::Lpc);
    CHECK(lpc.size() < delta.size());
    // Each channel payload is exactly what compress writes for that field.
    const std::size_t field = input.size() / channels;
    std::size_t offset = 24 + 4 * channels;
    for (int c = 0; c < channels; ++c) {
        const auto single = vc_delta3d::compress(
            {input.data() + c * field, field}, shape, 1,
            vc_delta3d::kQuantizationLossless, vc_delta3d::Predictor::Lpc);
        CHECK(vc_delta3d::predictor(asSpan(single)) == vc_delta3d::Predictor::Lpc);
        REQUIRE(offset + single.size() <= lpc.size());
        CHECK(std::equal(single.begin(), single.end(), lpc.begin() + offset));
        offset += single.size();
    }
    CHECK(offset == lpc.size());
    const auto decoded = vc_delta3d::decompress(asSpan(lpc), input.size());
    REQUIRE(decoded.has_value());
    CHECK(*decoded == input);
}

TEST_CASE("Signed values roundtrip as their bytes")
{
    const std::array<int, 3> shape{5, 6, 7};
    std::vector<std::int8_t> values(5 * 6 * 7);
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = static_cast<std::int8_t>(static_cast<int>(i % 23) - 11);
    const std::span<const std::byte> bytes(reinterpret_cast<const std::byte*>(values.data()), values.size());
    const auto packed = vc_delta3d::compressChannels(bytes, {5, 6, 7}, 1, vc_delta3d::ChannelLayout::Last, 1);
    const auto decoded = vc_delta3d::decompress(asSpan(packed), values.size());
    REQUIRE(decoded.has_value());
    CHECK(std::memcmp(decoded->data(), values.data(), values.size()) == 0);
    (void)shape;
}

TEST_CASE("Corrupt or mismatched containers fail to decode")
{
    const std::array<int, 3> shape{4, 5, 6};
    const int channels = 3;
    const auto input = channelVolume(shape, channels, 1, true);
    const auto packed = vc_delta3d::compressChannels(asSpan(input), shape, channels,
                                                     vc_delta3d::ChannelLayout::Last, 1);
    CHECK_FALSE(vc_delta3d::decompress(asSpan(packed), input.size() - 1).has_value());
    auto truncated = packed;
    truncated.resize(packed.size() - 5);
    CHECK_FALSE(vc_delta3d::decompress(asSpan(truncated), input.size()).has_value());
    auto badCount = packed;
    badCount[20] = std::byte{9};
    CHECK_FALSE(vc_delta3d::decompress(asSpan(badCount), input.size()).has_value());
    auto badTable = packed;
    badTable[24] = std::byte{0xFF}; badTable[25] = std::byte{0xFF}; badTable[26] = std::byte{0xFF};
    CHECK_FALSE(vc_delta3d::decompress(asSpan(badTable), input.size()).has_value());
    auto badLayout = packed;
    badLayout[7] = std::byte{2};
    CHECK_FALSE(vc_delta3d::decompress(asSpan(badLayout), input.size()).has_value());
    auto badDims = packed;  // container says 4x5x6, but claim 4x5x3 with twice the channels
    badDims[16] = std::byte{3}; badDims[20] = std::byte{6};
    CHECK_FALSE(vc_delta3d::decompress(asSpan(badDims), input.size()).has_value());
    auto overflow = packed;  // dims whose product wraps a 64-bit size
    for (int i = 8; i < 20; ++i) overflow[i] = std::byte{0xFF};
    CHECK_FALSE(vc_delta3d::decompress(asSpan(overflow), input.size()).has_value());
}

TEST_CASE("Invalid channel arguments throw")
{
    const std::vector<std::byte> bytes(2 * 3 * 4 * 2);
    CHECK_THROWS_AS(vc_delta3d::compressChannels(asSpan(bytes), {2, 3, 4}, 0,
                                                 vc_delta3d::ChannelLayout::First, 1),
                    std::invalid_argument);
    CHECK_THROWS_AS(vc_delta3d::compressChannels(asSpan(bytes), {2, 3, 4}, 3,
                                                 vc_delta3d::ChannelLayout::First, 1),
                    std::invalid_argument);
    CHECK_THROWS_AS(vc_delta3d::compressChannels(asSpan(bytes), {2, 3, 4}, 2,
                                                 vc_delta3d::ChannelLayout::First, 3),
                    std::invalid_argument);
}
