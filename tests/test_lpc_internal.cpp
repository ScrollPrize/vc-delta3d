// LPC internals against a scalar transcription of the predictor spec in
// codec.hpp. Encoder and decoder share the SIMD sum, so a roundtrip alone
// cannot catch a SIMD path that deviates from the spec consistently (which
// would make payloads decode differently across platforms).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../src/codec.cpp"

#include <functional>
#include <random>

namespace {

using vc_delta3d::LpcGeometry;
using vc_delta3d::LpcParams;
using vc_delta3d::LpcTapList;

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
            if (!((p.mask >> t) & 1))
                continue;
            const auto o = vc_delta3d::kLpcStencil[t];
            s += static_cast<std::int64_t>(p.w[t]) * at(Z + o.dz, Y + o.dy, X + o.dx);
        }
        return static_cast<int>((((s % 65536) + 65536) % 65536) >> 8);
    }
    const int p3 = at(Z, Y, X - 1) + at(Z, Y - 1, X) + at(Z - 1, Y, X) -
                   at(Z, Y - 1, X - 1) - at(Z - 1, Y, X - 1) -
                   at(Z - 1, Y - 1, X) + at(Z - 1, Y - 1, X - 1);
    return std::clamp(p3, 0, 255);
}

LpcParams randomParams(std::mt19937& rng, int density)
{
    LpcParams p;
    for (int t = 0; t < vc_delta3d::kLpcTaps; ++t)
        if (static_cast<int>(rng() % 64) < density) {
            p.mask |= std::uint64_t{1} << t;
            p.w[t] = static_cast<std::int16_t>(rng());
        }
    p.bias = static_cast<std::uint16_t>(rng());
    return p;
}

using Kernel = std::function<void(const std::uint8_t*, const LpcTapList&,
                                  std::uint16_t, std::uint16_t*, std::size_t)>;

// Every kernel this build and CPU can run, with its minimum row length.
std::vector<std::pair<Kernel, std::size_t>> kernels()
{
    std::vector<std::pair<Kernel, std::size_t>> k{
        {vc_delta3d::lpcAccumulateScalar, 1}, {vc_delta3d::lpcAccumulate, 1}};
#if defined(VC_DELTA3D_LPC_SSE2)
    k.push_back({vc_delta3d::lpcAccumulateSse2, 16});
#endif
#if defined(VC_DELTA3D_LPC_NEON)
    k.push_back({vc_delta3d::lpcAccumulateNeon, 16});
#endif
#if defined(VC_DELTA3D_LPC_AVX2)
    if (vc_delta3d::cpuHasAvx2())
        k.push_back({vc_delta3d::lpcAccumulateAvx2, 32});
#endif
    return k;
}

} // namespace

TEST_CASE("Every LPC kernel matches the scalar sum mod 2^16")
{
    std::mt19937 rng(11);
    std::vector<std::uint8_t> buf(4096);
    for (auto& b : buf)
        b = static_cast<std::uint8_t>(rng());
    const auto ks = kernels();
    MESSAGE(ks.size() << " kernels");
    for (std::size_t n = 1; n <= 160; ++n)
        for (const int taps : {0, 1, 7, 40, 62}) {
            LpcTapList list;
            list.n = taps;
            for (int t = 0; t < taps; ++t) {
                list.off[t] = std::uniform_int_distribution<int>(-600, 600)(rng);
                list.w[t] = static_cast<std::uint16_t>(rng());
            }
            vc_delta3d::lpcPairTaps(list);
            const auto bias = static_cast<std::uint16_t>(rng());
            const std::uint8_t* base = buf.data() + 1200;
            std::vector<std::uint16_t> expected(n);
            for (std::size_t i = 0; i < n; ++i) {
                std::uint64_t s = bias;
                for (int t = 0; t < taps; ++t)
                    s += std::uint64_t{list.w[t]} * base[list.off[t] + i];
                expected[i] = static_cast<std::uint16_t>(s);
            }
            for (const auto& [kernel, minN] : ks) {
                if (n < minN)
                    continue;
                std::vector<std::uint16_t> acc(n);
                kernel(base, list, bias, acc.data(), n);
                REQUIRE(acc == expected);
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
        for (const int density : {0, 20, 41, 64}) {
            const LpcGeometry g(shape[0], shape[1], shape[2]);
            std::vector<std::uint8_t> v(g.z * g.y * g.x);
            for (auto& b : v)
                b = static_cast<std::uint8_t>(rng());
            const auto p = randomParams(rng, density);
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

TEST_CASE("Parameter block roundtrips and selection keeps 40 taps")
{
    std::mt19937 rng(13);
    const auto p = randomParams(rng, 30);
    std::vector<std::byte> block(vc_delta3d::lpcParamBytes(p.mask));
    vc_delta3d::writeLpcParams(block.data(), p);
    const auto back = vc_delta3d::readLpcParams(block);
    REQUIRE(back.has_value());
    CHECK(back->mask == p.mask);
    CHECK(back->w == p.w);
    CHECK(back->bias == p.bias);
    CHECK_FALSE(vc_delta3d::readLpcParams(std::span(block).first(block.size() - 1)).has_value());

    // A smooth, noisy chunk: the fit keeps exactly kLpcSelectedTaps taps.
    const LpcGeometry g(24, 24, 24);
    std::vector<std::uint8_t> v(g.z * g.y * g.x);
    std::size_t i = 0;
    for (std::size_t z = 0; z < g.z; ++z)
        for (std::size_t y = 0; y < g.y; ++y)
            for (std::size_t x = 0; x < g.x; ++x)
                v[i++] = static_cast<std::uint8_t>(80 + 3 * x + 2 * y + z + rng() % 7);
    LpcParams fit;
    REQUIRE(vc_delta3d::lpcFit(v.data(), g, 1, fit));
    CHECK(std::popcount(fit.mask) == vc_delta3d::kLpcSelectedTaps);
}
