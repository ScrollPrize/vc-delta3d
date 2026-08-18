// VC-Delta3D codec coverage: D3D1 delta-zyx+zstd roundtrips and
// corrupt/mismatched payload handling.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <vc_delta3d/codec.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include <zstd.h>

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

TEST_CASE("D3D1 zstd fixture remains decodable")
{
    // Version-1 payload for uint8 values [0, 1, ..., 7] with shape 2x2x2.
    // Keep this fixed fixture independent of the encoder so wire-format
    // compatibility cannot accidentally regress with a matching code change.
    constexpr unsigned char encodedBytes[] = {
        0x44, 0x33, 0x44, 0x31, 0x01, 0x01, 0x01, 0x00,
        0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
        0x02, 0x00, 0x00, 0x00, 0x28, 0xb5, 0x2f, 0xfd,
        0x20, 0x08, 0x41, 0x00, 0x00, 0x00, 0x01, 0x02,
        0x00, 0x04, 0x00, 0x00, 0x00,
    };
    const auto encoded = std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(encodedBytes), sizeof(encodedBytes));
    const auto decoded = vc_delta3d::decompress(encoded, 8);
    REQUIRE(decoded.has_value());
    for (std::size_t i = 0; i < decoded->size(); ++i)
        CHECK((*decoded)[i] == static_cast<std::byte>(i));
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

TEST_CASE("Filter reduces compressed size on correlated data")
{
    const std::array<int, 3> shape{32, 32, 32};
    const auto input = smoothVolume(shape, 1);
    const auto filtered = vc_delta3d::compress(asSpan(input), shape, 1);

    std::vector<std::byte> unfiltered(ZSTD_compressBound(input.size()));
    const auto rc = ZSTD_compress(
        unfiltered.data(), unfiltered.size(), input.data(), input.size(), 3);
    REQUIRE_FALSE(ZSTD_isError(rc));
    CHECK(filtered.size() < rc);
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

TEST_CASE("Plain zstd frames without a D3D1 header are rejected")
{
    const std::array<int, 3> shape{4, 4, 4};
    const auto input = smoothVolume(shape, 1);
    std::vector<std::byte> frame(ZSTD_compressBound(input.size()));
    const auto rc = ZSTD_compress(
        frame.data(), frame.size(), input.data(), input.size(), 3);
    REQUIRE_FALSE(ZSTD_isError(rc));
    frame.resize(rc);
    CHECK_FALSE(vc_delta3d::decompress(asSpan(frame), input.size()).has_value());
}

TEST_CASE("Near-lossless quantization bounds the per-voxel error")
{
    const std::array<int, 3> shape{8, 6, 10};
    for (const std::size_t elemSize : {std::size_t{1}, std::size_t{2}}) {
        const auto input = smoothVolume(shape, elemSize);
        for (const int width : {vc_delta3d::kQuantizationMaxError1, vc_delta3d::kQuantizationMaxError2}) {
            const auto compressed =
                vc_delta3d::compress(asSpan(input), shape, elemSize,
                                  vc_delta3d::kDefaultZstdLevel, width);
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
                vc_delta3d::compress(asSpan(*decoded), shape, elemSize,
                                  vc_delta3d::kDefaultZstdLevel, width);
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
        vc_delta3d::compress(asSpan(input), shape, 1, vc_delta3d::kDefaultZstdLevel, 0),
        std::invalid_argument);
    CHECK_THROWS_AS(
        vc_delta3d::compress(asSpan(input), shape, 1, vc_delta3d::kDefaultZstdLevel, 256),
        std::invalid_argument);
}

TEST_CASE("Codec id is recorded and both codecs roundtrip identically")
{
    const std::array<int, 3> shape{8, 6, 10};
    for (const std::size_t elemSize : {std::size_t{1}, std::size_t{2}}) {
        const auto input = smoothVolume(shape, elemSize);
        const auto rans = vc_delta3d::compress(asSpan(input), shape, elemSize,
                                            vc_delta3d::kDefaultZstdLevel,
                                            vc_delta3d::kQuantizationLossless,
                                            vc_delta3d::EntropyCodec::Rans);
        const auto zstd = vc_delta3d::compress(asSpan(input), shape, elemSize,
                                            vc_delta3d::kDefaultZstdLevel,
                                            vc_delta3d::kQuantizationLossless,
                                            vc_delta3d::EntropyCodec::Zstd);
        CHECK(vc_delta3d::entropyCodec(asSpan(rans)) == vc_delta3d::EntropyCodec::Rans);
        CHECK(vc_delta3d::entropyCodec(asSpan(zstd)) == vc_delta3d::EntropyCodec::Zstd);
        // rANS payloads record their delta-axis mask (flag bit 7); zstd
        // payloads keep the plain codec byte for Python compatibility.
        CHECK((static_cast<int>(rans[7]) & 0x0F) == 1);
        CHECK((static_cast<int>(rans[7]) & 0x80) == 0x80);
        CHECK(static_cast<int>(zstd[7]) == 0);
        CHECK(vc_delta3d::deltaMask(asSpan(rans)).has_value());
        CHECK_FALSE(vc_delta3d::deltaMask(asSpan(zstd)).has_value());

        const auto fromRans = vc_delta3d::decompress(asSpan(rans), input.size());
        const auto fromZstd = vc_delta3d::decompress(asSpan(zstd), input.size());
        REQUIRE(fromRans.has_value());
        REQUIRE(fromZstd.has_value());
        CHECK(*fromRans == input);
        CHECK(*fromZstd == input);
    }

    // Default codec for new payloads is rANS.
    const auto input = smoothVolume(shape, 1);
    const auto def = vc_delta3d::compress(asSpan(input), shape, 1);
    CHECK(vc_delta3d::entropyCodec(asSpan(def)) == vc_delta3d::kDefaultEntropyCodec);
    CHECK(vc_delta3d::kDefaultEntropyCodec == vc_delta3d::EntropyCodec::Rans);
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
    CHECK(vc_delta3d::entropyCodec(asSpan(compressed)) == vc_delta3d::EntropyCodec::Rans);
    const auto decoded = vc_delta3d::decompress(asSpan(compressed), input.size());
    REQUIRE(decoded.has_value());
    CHECK(*decoded == input);

    // Mask bits without the flag bit are not a valid codec byte.
    compressed[7] = std::byte{0x51};
    CHECK_FALSE(vc_delta3d::entropyCodec(asSpan(compressed)).has_value());
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

TEST_CASE("Corrupt rANS payloads and unknown codec ids return nullopt")
{
    const std::array<int, 3> shape{8, 8, 8};
    const auto input = smoothVolume(shape, 1);
    const auto good = vc_delta3d::compress(asSpan(input), shape, 1,
                                        vc_delta3d::kDefaultZstdLevel,
                                        vc_delta3d::kQuantizationLossless,
                                        vc_delta3d::EntropyCodec::Rans);
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

    // Unknown codec id.
    auto badCodec = good;
    badCodec[7] = std::byte{0x7E};
    CHECK_FALSE(vc_delta3d::entropyCodec(asSpan(badCodec)).has_value());
    CHECK_FALSE(vc_delta3d::decompress(asSpan(badCodec), input.size()).has_value());
}

TEST_CASE("Corrupt and mismatched payloads return nullopt")
{
    const std::array<int, 3> shape{4, 4, 4};
    const auto input = smoothVolume(shape, 1);
    const auto compressed = vc_delta3d::compress(asSpan(input), shape, 1);

    // Wrong expected size (header dims disagree).
    CHECK_FALSE(vc_delta3d::decompress(asSpan(compressed), input.size() + 1).has_value());

    // Truncated zstd frame.
    std::vector<std::byte> truncated(compressed.begin(),
                                     compressed.begin() + compressed.size() / 2);
    CHECK_FALSE(vc_delta3d::decompress(asSpan(truncated), input.size()).has_value());

    // Garbage bytes (neither D3D1 nor zstd).
    std::vector<std::byte> garbage(64, std::byte{0xAB});
    CHECK_FALSE(vc_delta3d::decompress(asSpan(garbage), input.size()).has_value());
}
