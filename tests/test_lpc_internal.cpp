// LPC internals against a scalar transcription of the predictor spec in
// codec.hpp. Encoder and decoder share the SIMD sum, so a roundtrip alone
// cannot catch a SIMD path that deviates from the spec consistently (which
// would make payloads decode differently across platforms).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../src/codec.cpp"

#include <random>

namespace {

using vc_delta3d::LpcGeometry;
using vc_delta3d::LpcParams;

// Prediction of voxel (z, y, x) exactly as codec.hpp specifies it.
int specPredict(const std::vector<std::uint8_t>& v, const LpcGeometry& g,
                const LpcParams& p, std::size_t z, std::size_t y,
                std::size_t x)
{
    auto at = [&](long zz, long yy, long xx) -> int {
        if (zz < 0 || yy < 0 || xx < 0)
            return 0;
        return v[(static_cast<std::size_t>(zz) * g.y + yy) * g.x + xx];
    };
    const long Z = static_cast<long>(z), Y = static_cast<long>(y),
               X = static_cast<long>(x);
    if (z >= 2 && y >= 2 && y + 2 < g.y && x >= 2 && x + 2 < g.x) {
        std::int64_t s = p.bias;
        for (int t = 0; t < vc_delta3d::kLpcTaps; ++t) {
            const auto o = vc_delta3d::kLpcStencil[t];
            s += static_cast<std::int64_t>(p.w[t]) *
                 at(Z + o.dz, Y + o.dy, X + o.dx);
        }
        REQUIRE(s >= std::numeric_limits<std::int32_t>::min());
        REQUIRE(s <= std::numeric_limits<std::int32_t>::max());
        return static_cast<int>(s >> 12);
    }
    const int p3 = at(Z, Y, X - 1) + at(Z, Y - 1, X) + at(Z - 1, Y, X) -
                   at(Z, Y - 1, X - 1) - at(Z - 1, Y, X - 1) -
                   at(Z - 1, Y - 1, X) + at(Z - 1, Y - 1, X - 1);
    return std::clamp(p3, 0, 255);
}

LpcParams randomParams(std::mt19937& rng, bool extreme)
{
    LpcParams p;
    std::uniform_int_distribution<int> w(extreme ? -32768 : -3000,
                                         extreme ? 32767 : 3000);
    for (auto& wt : p.w)
        wt = static_cast<std::int16_t>(extreme ? (rng() & 1 ? 32767 : -32768)
                                               : w(rng));
    p.bias = extreme ? (rng() & 1 ? (1 << 30) : -(1 << 30))
                     : std::uniform_int_distribution<int>(-1 << 20, 1 << 20)(rng);
    return p;
}

} // namespace

TEST_CASE("SIMD accumulate matches the scalar sum")
{
    std::mt19937 rng(11);
    std::vector<std::uint8_t> buf(4096);
    for (auto& b : buf)
        b = static_cast<std::uint8_t>(rng());
    std::array<std::ptrdiff_t, vc_delta3d::kLpcTaps> off;
    for (auto& o : off)
        o = std::uniform_int_distribution<int>(-600, 600)(rng);
    for (const bool extreme : {false, true})
        for (std::size_t n = 1; n <= 140; ++n)
            for (const int taps : {vc_delta3d::kLpcPrevTaps, vc_delta3d::kLpcTaps}) {
                const auto p = randomParams(rng, extreme);
                std::vector<std::int32_t> acc(n);
                const std::uint8_t* base = buf.data() + 1200;
                vc_delta3d::lpcAccumulate(base, off.data(), p.w.data(), taps,
                                          p.bias, acc.data(), n);
                for (std::size_t i = 0; i < n; ++i) {
                    std::int64_t s = p.bias;
                    for (int t = 0; t < taps; ++t)
                        s += static_cast<std::int64_t>(p.w[t]) * base[off[t] + i];
                    // Extreme parameters may exceed int32; the format only
                    // promises exactness within the documented bounds,
                    // where both must agree.
                    if (s >= std::numeric_limits<std::int32_t>::min() &&
                        s <= std::numeric_limits<std::int32_t>::max())
                        REQUIRE(acc[i] == s);
                }
            }
}

TEST_CASE("Residuals and reconstruction follow the spec")
{
    std::mt19937 rng(12);
    for (const auto shape : {std::array<std::size_t, 3>{6, 7, 9},
                             std::array<std::size_t, 3>{9, 12, 18},
                             std::array<std::size_t, 3>{7, 9, 37},
                             std::array<std::size_t, 3>{5, 6, 130}})
        for (const bool extreme : {false, true}) {
            const LpcGeometry g(shape[0], shape[1], shape[2]);
            std::vector<std::uint8_t> v(g.z * g.y * g.x);
            for (auto& b : v)
                b = static_cast<std::uint8_t>(rng());
            auto p = randomParams(rng, extreme);
            if (extreme) {
                // Largest magnitudes the bounds allow: bias 2^30 plus full
                // positive weights on 255-valued voxels stays below 2^31.
                p.bias = 1 << 30;
                for (auto& wt : p.w)
                    wt = 32767;
                std::fill(v.begin(), v.end(), std::uint8_t{255});
            }
            std::vector<std::uint8_t> res(v.size());
            vc_delta3d::lpcResiduals(v.data(), res.data(), g, p);
            for (std::size_t z = 0; z < g.z; ++z)
                for (std::size_t y = 0; y < g.y; ++y)
                    for (std::size_t x = 0; x < g.x; ++x) {
                        const std::size_t i = (z * g.y + y) * g.x + x;
                        REQUIRE(res[i] == static_cast<std::uint8_t>(
                                              v[i] - specPredict(v, g, p, z, y, x)));
                    }
            vc_delta3d::lpcReconstruct(res.data(), g, p);
            CHECK(res == v);
        }
}
