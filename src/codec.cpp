#include <vc_delta3d/codec.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

#if defined(_MSC_VER)
#include <intrin.h>   // __umulh / _udiv128 for the 128-bit rANS reciprocal math
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace vc_delta3d {

namespace {

constexpr std::size_t kHeaderSize = 20;
constexpr unsigned char kFormatVersion = 1;

// Delta-axis masks: bit 0 differences along x, bit 1 along y, bit 2 along z.
// kDeltaMaskZyx (all three passes) is what every payload used before
// per-chunk selection and what non-selected payloads still use.
constexpr unsigned kDeltaMaskZyx = 7;

// In-place backward difference along each axis in `mask`, z then y then x.
// Iterating each pass from the end keeps every subtraction reading
// not-yet-filtered values; the element-wise arithmetic wraps mod 2^8/2^16,
// so the inverse is exact. The passes commute, so the order is only a
// convention shared with deltaUnfilter.
template <typename T>
void deltaFilter(T* data, std::size_t z, std::size_t y, std::size_t x,
                 unsigned mask)
{
    const std::size_t slice = y * x;
    const std::size_t n = z * slice;
    if (mask & 4)
        for (std::size_t i = n; i-- > slice;)
            data[i] = static_cast<T>(data[i] - data[i - slice]);
    if (mask & 2)
        for (std::size_t s = 0; s < z; ++s) {
            T* p = data + s * slice;
            for (std::size_t i = slice; i-- > x;)
                p[i] = static_cast<T>(p[i] - p[i - x]);
        }
    if (mask & 1)
        for (std::size_t r = 0; r < z * y; ++r) {
            T* p = data + r * x;
            for (std::size_t i = x; i-- > 1;)
                p[i] = static_cast<T>(p[i] - p[i - 1]);
        }
}

// Inclusive prefix sum over one row of uint8 (log-step shift-adds per
// 16-byte vector, then a running carry across vectors). Wrap-around adds
// match the scalar path exactly.
#if defined(__ARM_NEON)
#define VC_DELTA3D_PREFIX_U8 1
inline void prefixSumRowU8(std::uint8_t* p, std::size_t x)
{
    const uint8x16_t zero = vdupq_n_u8(0);
    uint8x16_t carry = zero;
    std::size_t i = 0;
    for (; i + 16 <= x; i += 16) {
        uint8x16_t v = vld1q_u8(p + i);
        v = vaddq_u8(v, vextq_u8(zero, v, 15));
        v = vaddq_u8(v, vextq_u8(zero, v, 14));
        v = vaddq_u8(v, vextq_u8(zero, v, 12));
        v = vaddq_u8(v, vextq_u8(zero, v, 8));
        v = vaddq_u8(v, carry);
        vst1q_u8(p + i, v);
        carry = vdupq_laneq_u8(v, 15);
    }
    std::uint8_t c = vgetq_lane_u8(carry, 0);
    for (; i < x; ++i) {
        c = static_cast<std::uint8_t>(p[i] + c);
        p[i] = c;
    }
}
#elif defined(__SSE2__)
#define VC_DELTA3D_PREFIX_U8 1
inline void prefixSumRowU8(std::uint8_t* p, std::size_t x)
{
    __m128i carry = _mm_setzero_si128();
    std::size_t i = 0;
    for (; i + 16 <= x; i += 16) {
        __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + i));
        v = _mm_add_epi8(v, _mm_slli_si128(v, 1));
        v = _mm_add_epi8(v, _mm_slli_si128(v, 2));
        v = _mm_add_epi8(v, _mm_slli_si128(v, 4));
        v = _mm_add_epi8(v, _mm_slli_si128(v, 8));
        v = _mm_add_epi8(v, carry);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(p + i), v);
        carry = _mm_set1_epi8(
            static_cast<char>(_mm_extract_epi16(v, 7) >> 8));
    }
    auto c = static_cast<std::uint8_t>(_mm_cvtsi128_si32(carry));
    for (; i < x; ++i) {
        c = static_cast<std::uint8_t>(p[i] + c);
        p[i] = c;
    }
}
#endif

// Inverse: forward prefix sums along x, then y, then z.
template <typename T>
void deltaUnfilter(T* data, std::size_t z, std::size_t y, std::size_t x,
                   unsigned mask)
{
    const std::size_t slice = y * x;
    const std::size_t n = z * slice;
    if (mask & 1)
        for (std::size_t r = 0; r < z * y; ++r) {
            T* p = data + r * x;
#if defined(VC_DELTA3D_PREFIX_U8)
            if constexpr (sizeof(T) == 1) {
                prefixSumRowU8(reinterpret_cast<std::uint8_t*>(p), x);
                continue;
            }
#endif
            for (std::size_t i = 1; i < x; ++i)
                p[i] = static_cast<T>(p[i] + p[i - 1]);
        }
    if (mask & 2)
        for (std::size_t s = 0; s < z; ++s) {
            T* p = data + s * slice;
            for (std::size_t i = x; i < slice; ++i)
                p[i] = static_cast<T>(p[i] + p[i - x]);
        }
    if (mask & 4)
        for (std::size_t i = slice; i < n; ++i)
            data[i] = static_cast<T>(data[i] + data[i - slice]);
}

// Picks the delta-axis mask with the lowest order-0 residual entropy. Each
// difference pass cancels smooth structure but doubles white-noise variance,
// so on noise-dominated chunks fewer passes often beat the full cascade; the
// winner varies chunk to chunk, hence measuring instead of guessing. One
// pass over every 4th z-slice histograms all eight subsets at once via the
// Lorenzo-corner identity (the residual of any axis subset is the
// alternating sum over the corresponding neighbor corners); entropy of the
// subsampled histogram tracks the full-chunk rANS size closely enough that
// this recovers nearly all of the exhaustive gain. Requires all dims >= 2.
unsigned selectDeltaMask(const std::uint8_t* q,
                         std::size_t z,
                         std::size_t y,
                         std::size_t x)
{
    std::uint32_t h[8][256] = {};
    std::uint64_t n = 0;
    const std::size_t slice = y * x;
    for (std::size_t zz = 1; zz < z; zz += 4) {
        for (std::size_t yy = 1; yy < y; yy += 2) {
            const std::uint8_t* p = q + zz * slice + yy * x;
            const std::uint8_t* py = p - x;
            const std::uint8_t* pz = p - slice;
            const std::uint8_t* pyz = pz - x;
            for (std::size_t xx = 1; xx < x; ++xx) {
                const int v = p[xx];
                const int vx = p[xx - 1], vy = py[xx], vz = pz[xx];
                const int vxy = py[xx - 1], vxz = pz[xx - 1], vyz = pyz[xx];
                const int vxyz = pyz[xx - 1];
                h[0][v]++;
                h[1][static_cast<std::uint8_t>(v - vx)]++;
                h[2][static_cast<std::uint8_t>(v - vy)]++;
                h[3][static_cast<std::uint8_t>(v - vx - vy + vxy)]++;
                h[4][static_cast<std::uint8_t>(v - vz)]++;
                h[5][static_cast<std::uint8_t>(v - vx - vz + vxz)]++;
                h[6][static_cast<std::uint8_t>(v - vy - vz + vyz)]++;
                h[7][static_cast<std::uint8_t>(v - vx - vy - vz + vxy +
                                               vxz + vyz - vxyz)]++;
                ++n;
            }
        }
    }
    unsigned best = kDeltaMaskZyx;
    double bestBits = std::numeric_limits<double>::max();
    for (unsigned m = 0; m < 8; ++m) {
        double bits = 0;
        for (int s = 0; s < 256; ++s)
            if (h[m][s]) {
                const double p = static_cast<double>(h[m][s]) / n;
                bits -= p * std::log2(p);
            }
        if (bits < bestBits) {
            bestBits = bits;
            best = m;
        }
    }
    return best;
}

// Snap to the nearest multiple of `width` (bins centered on multiples, so 0
// stays exactly 0 and masked background survives untouched). Idempotent:
// bin centers map to themselves.
template <typename T>
void quantizeValues(T* data, std::size_t n, int width)
{
    constexpr int maxVal = std::numeric_limits<T>::max();
    const int half = width / 2;
    if constexpr (sizeof(T) == 1) {
        T lut[256];
        for (int v = 0; v < 256; ++v)
            lut[v] = static_cast<T>(std::min((v + half) / width * width, maxVal));
        for (std::size_t i = 0; i < n; ++i)
            data[i] = lut[data[i]];
    } else {
        for (std::size_t i = 0; i < n; ++i)
            data[i] = static_cast<T>(
                std::min((data[i] + half) / width * width, maxVal));
    }
}

// ---- order-0 rANS (ryg's rans64 construction): 12-bit frequency tables,
// eight interleaved 64-bit states, 32-bit renormalization. Encoding walks
// the symbols back to front and emits renorm words back to front, so the
// decoder streams strictly forward. Division-free encode via 64-bit
// reciprocals (Alverson); exact for all 64-bit states. ----

constexpr int kRansScaleBits = 12;
constexpr std::uint32_t kRansM = 1u << kRansScaleBits;   // total frequency
constexpr std::uint64_t kRansL = 1ull << 31;             // renorm lower bound
constexpr std::size_t kRansLanes = 8;
constexpr std::size_t kRansTableBytes = 256 * 2;         // uint16 per symbol
constexpr std::size_t kRansStateBytes = kRansLanes * 8;

struct RansEncSym {
    std::uint64_t rcp;   // reciprocal so q = mulhi(x, rcp) >> shift = x/freq
    std::uint64_t bias;  // cum, or cum + M - 1 for freq 1 (rcp = 2^64 - 1)
    std::uint64_t xmax;  // renorm threshold ((L>>12)<<32) * freq
    std::uint32_t cmpl;  // M - freq
    std::uint32_t shift;
};

inline std::uint64_t ransMulHi(std::uint64_t a, std::uint64_t b)
{
#if defined(_MSC_VER)
    return __umulh(a, b);   // high 64 bits of the 64x64 product
#else
    return static_cast<std::uint64_t>(
        (static_cast<unsigned __int128>(a) * b) >> 64);
#endif
}

// Histogram -> frequencies summing to exactly kRansM, every present symbol
// nonzero. Rounding drift is settled against the most frequent symbol.
void ransNormalize(const std::uint64_t hist[256],
                   std::uint64_t total,
                   std::uint32_t freq[256])
{
    std::uint64_t assigned = 0;
    for (int s = 0; s < 256; ++s) {
        freq[s] = hist[s]
            ? static_cast<std::uint32_t>(
                  std::max<std::uint64_t>(1, (hist[s] * kRansM) / total))
            : 0;
        assigned += freq[s];
    }
    while (assigned != kRansM) {
        int s = 0;
        for (int i = 1; i < 256; ++i)
            if (freq[i] > freq[s]) s = i;
        if (assigned > kRansM) {
            const auto cut = std::min<std::uint64_t>(freq[s] - 1, assigned - kRansM);
            freq[s] -= static_cast<std::uint32_t>(cut);
            assigned -= cut;
        } else {
            freq[s] += static_cast<std::uint32_t>(kRansM - assigned);
            assigned = kRansM;
        }
    }
}

void ransBuildEncTable(const std::uint32_t freq[256], RansEncSym enc[256])
{
    std::uint32_t cum = 0;
    for (int s = 0; s < 256; ++s) {
        const std::uint32_t f = freq[s];
        RansEncSym& e = enc[s];
        e.cmpl = kRansM - f;
        e.xmax = ((kRansL >> kRansScaleBits) << 32) * f;
        if (f == 0) {
            e = {};
        } else if (f == 1) {
            // mulhi(x, 2^64 - 1) = x - 1; the off-by-(M-1) folds into bias
            e.rcp = ~0ull;
            e.shift = 0;
            e.bias = static_cast<std::uint64_t>(cum) + kRansM - 1;
        } else {
            std::uint32_t k = 1;
            while ((1u << k) < f) ++k;  // k = ceil(log2 f)
            e.shift = k - 1;
            // num = 2^(k+63) + (f-1): high 64 bits = 2^(k-1), low 64 bits =
            // (f-1). The quotient num/f provably fits in 64 bits (f > 2^(k-1)),
            // so the 128/64 -> 64 division is well-defined on both paths.
#if defined(_MSC_VER)
            std::uint64_t ransRcpRem;
            e.rcp = _udiv128(std::uint64_t{1} << (k - 1),
                             static_cast<std::uint64_t>(f) - 1, f, &ransRcpRem);
#else
            const auto num =
                ((static_cast<unsigned __int128>(1) << (k + 63)) + f - 1);
            e.rcp = static_cast<std::uint64_t>(num / f);
#endif
            e.bias = cum;
        }
        cum += f;
    }
}

// Compresses `input` into [headerReserve][freq table][states+stream].
std::vector<std::byte> ransCompressFrame(std::span<const std::byte> input,
                                         std::size_t headerReserve)
{
    const std::size_t n = input.size();
    const auto* in = reinterpret_cast<const std::uint8_t*>(input.data());

    std::uint64_t h4[4][256] = {};
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        h4[0][in[i]]++;
        h4[1][in[i + 1]]++;
        h4[2][in[i + 2]]++;
        h4[3][in[i + 3]]++;
    }
    for (; i < n; ++i) h4[0][in[i]]++;
    std::uint64_t hist[256];
    for (int s = 0; s < 256; ++s)
        hist[s] = h4[0][s] + h4[1][s] + h4[2][s] + h4[3][s];

    std::uint32_t freq[256];
    ransNormalize(hist, n, freq);
    RansEncSym enc[256];
    ransBuildEncTable(freq, enc);

    // Worst case is ~1.5 bytes/symbol (12-bit code ceiling); 2n is safe.
    const std::size_t streamCap = 2 * n + kRansStateBytes + 64;
    std::vector<std::byte> output(headerReserve + kRansTableBytes + streamCap);
    auto* base = reinterpret_cast<std::uint8_t*>(output.data());

    std::uint8_t* table = base + headerReserve;
    for (int s = 0; s < 256; ++s) {
        table[2 * s] = static_cast<std::uint8_t>(freq[s] & 0xFF);
        table[2 * s + 1] = static_cast<std::uint8_t>(freq[s] >> 8);
    }

    std::uint8_t* end = base + output.size();
    std::uint8_t* p = end;
    std::uint64_t x[kRansLanes];
    std::fill(std::begin(x), std::end(x), kRansL);
    // Branchless renorm: speculatively store each lane's low word below the
    // claimed stream and only advance past it when the state actually
    // overflows; the buffer's slack keeps the speculative store in bounds.
    std::size_t j = n;
    while (j & (kRansLanes - 1)) {
        --j;
        const RansEncSym& e = enc[in[j]];
        std::uint64_t& s = x[j & (kRansLanes - 1)];
        const auto w = static_cast<std::uint32_t>(s);
        std::memcpy(p - 4, &w, 4);
        const bool renorm = s >= e.xmax;
        p -= renorm ? 4 : 0;
        s = renorm ? s >> 32 : s;
        const std::uint64_t q = ransMulHi(s, e.rcp) >> e.shift;
        s = s + e.bias + q * e.cmpl;
    }
    // Blocks of eight: renorm decisions depend only on each lane's own
    // state, so only the cheap store-offset adds serialize across lanes.
    while (j) {
        j -= kRansLanes;
        const std::uint8_t* sym = in + j;
        std::size_t o = 0;
        for (std::size_t k = kRansLanes; k-- > 0;) {
            const RansEncSym& e = enc[sym[k]];
            std::uint64_t s = x[k];
            const auto w = static_cast<std::uint32_t>(s);
            std::memcpy(p - o - 4, &w, 4);
            const bool renorm = s >= e.xmax;
            o += renorm ? 4 : 0;
            s = renorm ? s >> 32 : s;
            const std::uint64_t q = ransMulHi(s, e.rcp) >> e.shift;
            x[k] = s + e.bias + q * e.cmpl;
        }
        p -= o;
    }
    for (std::size_t k = kRansLanes; k-- > 0;) {
        p -= 8;
        std::memcpy(p, &x[k], 8);
    }

    const std::size_t streamSize = static_cast<std::size_t>(end - p);
    std::memmove(base + headerReserve + kRansTableBytes, p, streamSize);
    output.resize(headerReserve + kRansTableBytes + streamSize);
    return output;
}

// Decodes a rANS frame produced by ransCompressFrame. Returns false on any
// inconsistency (bad table, short stream, or states/stream not ending
// exactly where encoding started them).
bool ransDecompressFrame(std::span<const std::byte> frame,
                         std::byte* outBytes,
                         std::size_t n)
{
    if (frame.size() < kRansTableBytes + kRansStateBytes)
        return false;
    const auto* table = reinterpret_cast<const std::uint8_t*>(frame.data());

    std::uint32_t freq[256];
    std::uint32_t cum[256];
    std::uint32_t total = 0;
    for (int s = 0; s < 256; ++s) {
        freq[s] = static_cast<std::uint32_t>(table[2 * s]) |
                  (static_cast<std::uint32_t>(table[2 * s + 1]) << 8);
        cum[s] = total;
        total += freq[s];
    }
    if (total != kRansM)
        return false;
    // Fused per-slot entry: one L1 load yields symbol, freq-1 and
    // slot - cum (each fits 12/8 bits within a uint32).
    static thread_local std::uint32_t slotEntry[kRansM];
    for (int s = 0; s < 256; ++s)
        for (std::uint32_t j = 0; j < freq[s]; ++j)
            slotEntry[cum[s] + j] =
                ((freq[s] - 1) << 20) | (j << 8) | static_cast<std::uint32_t>(s);

    const auto* p = reinterpret_cast<const std::uint8_t*>(frame.data()) +
                    kRansTableBytes;
    const auto* end = reinterpret_cast<const std::uint8_t*>(frame.data()) +
                      frame.size();
    std::uint64_t x[kRansLanes];
    for (std::size_t k = 0; k < kRansLanes; ++k) {
        std::memcpy(&x[k], p, 8);
        p += 8;
    }

    auto* out = reinterpret_cast<std::uint8_t*>(outBytes);
    std::size_t i = 0;
    // Fast path: while >= 32 stream bytes remain, all eight lanes can renorm
    // branchlessly (each consumes at most 4 bytes per symbol).
    for (; i + kRansLanes <= n && end - p >= 32; i += kRansLanes) {
        // Stage 1: symbol resolve + state advance, independent per lane.
        std::uint64_t t[kRansLanes];
        for (std::size_t k = 0; k < kRansLanes; ++k) {
            const auto slot = static_cast<std::uint32_t>(x[k]) & (kRansM - 1);
            const std::uint32_t e = slotEntry[slot];
            out[i + k] = static_cast<std::uint8_t>(e);
            t[k] = (static_cast<std::uint64_t>(e >> 20) + 1) *
                       (x[k] >> kRansScaleBits) +
                   ((e >> 8) & 0xFFF);
        }
        // Stage 2: branchless renorm; only the cheap offset adds serialize.
        std::size_t o = 0;
        for (std::size_t k = 0; k < kRansLanes; ++k) {
            std::uint32_t w;
            std::memcpy(&w, p + o, 4);
            const bool renorm = t[k] < kRansL;
            x[k] = renorm ? (t[k] << 32) | w : t[k];
            o += renorm ? 4 : 0;
        }
        p += o;
    }
    for (; i < n; ++i) {
        std::uint64_t& s = x[i & (kRansLanes - 1)];
        const auto slot = static_cast<std::uint32_t>(s) & (kRansM - 1);
        const std::uint32_t e = slotEntry[slot];
        out[i] = static_cast<std::uint8_t>(e);
        s = (static_cast<std::uint64_t>(e >> 20) + 1) * (s >> kRansScaleBits) +
            ((e >> 8) & 0xFFF);
        if (s < kRansL) {
            std::uint32_t w = 0;
            if (p + 4 > end)
                return false;
            std::memcpy(&w, p, 4);
            p += 4;
            s = (s << 32) | w;
        }
    }

    // Encoding started every lane at kRansL and the decoder must consume the
    // stream exactly; anything else means a corrupt payload.
    if (p != end)
        return false;
    for (std::size_t k = 0; k < kRansLanes; ++k)
        if (x[k] != kRansL)
            return false;
    return true;
}

void writeU32(std::byte* out, std::uint32_t value)
{
    out[0] = static_cast<std::byte>(value & 0xFF);
    out[1] = static_cast<std::byte>((value >> 8) & 0xFF);
    out[2] = static_cast<std::byte>((value >> 16) & 0xFF);
    out[3] = static_cast<std::byte>((value >> 24) & 0xFF);
}

std::uint32_t readU32(const std::byte* in)
{
    return static_cast<std::uint32_t>(in[0]) |
           (static_cast<std::uint32_t>(in[1]) << 8) |
           (static_cast<std::uint32_t>(in[2]) << 16) |
           (static_cast<std::uint32_t>(in[3]) << 24);
}

bool hasHeader(std::span<const std::byte> input,
               const WireMagic& expectedMagic)
{
    return input.size() > kHeaderSize &&
           std::equal(expectedMagic.begin(), expectedMagic.end(),
                      input.begin()) &&
           static_cast<unsigned char>(input[4]) == kFormatVersion;
}

// ---- LPC: per-chunk linear prediction over the 62 causal voxels of the
// radius-2 box (see the layout notes in codec.hpp). Predictions use int32
// sums of int16 weights times uint8 voxels; with |w| <= 2^15 and |bias| <=
// 2^30 a sum stays below 2^31, so encoder and decoder agree exactly. ----

constexpr int kLpcTaps = 62;
constexpr std::size_t kLpcRadius = 2;
constexpr int kLpcShift = 12;
constexpr unsigned char kLpcStencilBox2 = 1;
constexpr std::size_t kLpcParamBytes = 1 + 2 * kLpcTaps + 4;
constexpr std::int32_t kLpcMaxBias = 1 << 30;

struct LpcOffset {
    int dz, dy, dx;
};

// Causal offsets in tap order. The two current-row taps come last (dx = -2,
// then dx = -1), so the first kLpcTaps - 2 taps only read earlier rows.
constexpr std::array<LpcOffset, kLpcTaps> lpcStencil()
{
    constexpr int r = static_cast<int>(kLpcRadius);
    std::array<LpcOffset, kLpcTaps> taps{};
    int n = 0;
    for (int dz = -r; dz <= 0; ++dz)
        for (int dy = -r; dy <= r; ++dy)
            for (int dx = -r; dx <= r; ++dx)
                if (dz < 0 || (dz == 0 && (dy < 0 || (dy == 0 && dx < 0))))
                    taps[n++] = {dz, dy, dx};
    return taps;
}
constexpr auto kLpcStencil = lpcStencil();
constexpr int kLpcPrevTaps = kLpcTaps - 2;
static_assert(kLpcStencil[kLpcTaps - 2].dz == 0 &&
              kLpcStencil[kLpcTaps - 2].dy == 0 &&
              kLpcStencil[kLpcTaps - 2].dx == -2);
static_assert(kLpcStencil[kLpcTaps - 1].dx == -1);

struct LpcParams {
    std::array<std::int16_t, kLpcTaps> w;
    std::int32_t bias;
};

struct LpcGeometry {
    std::size_t z, y, x;
    std::array<std::ptrdiff_t, kLpcTaps> off;  // flat offsets of the taps

    LpcGeometry(std::size_t z_, std::size_t y_, std::size_t x_)
        : z(z_), y(y_), x(x_)
    {
        const auto sy = static_cast<std::ptrdiff_t>(y);
        const auto sx = static_cast<std::ptrdiff_t>(x);
        for (int t = 0; t < kLpcTaps; ++t)
            off[t] = (kLpcStencil[t].dz * sy + kLpcStencil[t].dy) * sx +
                     kLpcStencil[t].dx;
    }

    // Rows whose voxels x in [2, X-2) use the linear predictor.
    bool interiorRow(std::size_t zz, std::size_t yy) const
    {
        return zz >= kLpcRadius && yy >= kLpcRadius && yy + kLpcRadius < y &&
               x > 2 * kLpcRadius;
    }
};

// Clamped 3-D Lorenzo prediction for voxel xx of a row. n, u and nu are the
// rows above, behind and above-behind (an all-zero row outside the chunk);
// row[xx - 1] must already hold its final value.
inline int lorenzoPredict(const std::uint8_t* row, const std::uint8_t* n,
                          const std::uint8_t* u, const std::uint8_t* nu,
                          std::size_t xx)
{
    int p = n[xx] + u[xx] - nu[xx];
    if (xx > 0)
        p += row[xx - 1] - n[xx - 1] - u[xx - 1] + nu[xx - 1];
    return std::clamp(p, 0, 255);
}

// acc[i] = bias + sum over taps t < nTaps of w[t] * base[off[t] + i], for i
// in [0, n). Integer sums are exact in any order, so the SIMD paths only
// regroup the work: blocks of 16 voxels keep their four int32x4
// accumulators in registers across all taps. A short final block overlaps
// the previous one (recomputing a few sums) instead of falling back to
// scalar code. nTaps must be even.
#if defined(__aarch64__)
#define VC_DELTA3D_LPC_SIMD 1
inline void lpcAccumulateBlock(const std::uint8_t* base,
                               const std::ptrdiff_t* off,
                               const std::int16_t* w, int nTaps,
                               std::int32_t bias, std::int32_t* acc)
{
    int32x4_t a0 = vdupq_n_s32(bias), a1 = a0, a2 = a0, a3 = a0;
    for (int t = 0; t < nTaps; ++t) {
        const uint8x16_t v = vld1q_u8(base + off[t]);
        const int16x8_t lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v)));
        const int16x8_t hi = vreinterpretq_s16_u16(vmovl_high_u8(v));
        a0 = vmlal_n_s16(a0, vget_low_s16(lo), w[t]);
        a1 = vmlal_high_n_s16(a1, lo, w[t]);
        a2 = vmlal_n_s16(a2, vget_low_s16(hi), w[t]);
        a3 = vmlal_high_n_s16(a3, hi, w[t]);
    }
    vst1q_s32(acc, a0);
    vst1q_s32(acc + 4, a1);
    vst1q_s32(acc + 8, a2);
    vst1q_s32(acc + 12, a3);
}
#elif defined(__SSE2__)
#define VC_DELTA3D_LPC_SIMD 1
// Two taps per pmaddwd: voxels of taps t and t+1 are interleaved as int16
// pairs and multiplied by the matching (w[t], w[t+1]) pair.
inline void lpcAccumulateBlock(const std::uint8_t* base,
                               const std::ptrdiff_t* off,
                               const std::int16_t* w, int nTaps,
                               std::int32_t bias, std::int32_t* acc)
{
    const __m128i zero = _mm_setzero_si128();
    __m128i a0 = _mm_set1_epi32(bias), a1 = a0, a2 = a0, a3 = a0;
    for (int t = 0; t < nTaps; t += 2) {
        const __m128i va = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(base + off[t]));
        const __m128i vb = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(base + off[t + 1]));
        const __m128i wab = _mm_set1_epi32(static_cast<std::int32_t>(
            (static_cast<std::uint32_t>(static_cast<std::uint16_t>(w[t + 1])) << 16) |
            static_cast<std::uint16_t>(w[t])));
        const __m128i aLo = _mm_unpacklo_epi8(va, zero);
        const __m128i aHi = _mm_unpackhi_epi8(va, zero);
        const __m128i bLo = _mm_unpacklo_epi8(vb, zero);
        const __m128i bHi = _mm_unpackhi_epi8(vb, zero);
        a0 = _mm_add_epi32(a0, _mm_madd_epi16(_mm_unpacklo_epi16(aLo, bLo), wab));
        a1 = _mm_add_epi32(a1, _mm_madd_epi16(_mm_unpackhi_epi16(aLo, bLo), wab));
        a2 = _mm_add_epi32(a2, _mm_madd_epi16(_mm_unpacklo_epi16(aHi, bHi), wab));
        a3 = _mm_add_epi32(a3, _mm_madd_epi16(_mm_unpackhi_epi16(aHi, bHi), wab));
    }
    _mm_storeu_si128(reinterpret_cast<__m128i*>(acc), a0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(acc + 4), a1);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(acc + 8), a2);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(acc + 12), a3);
}
#endif

void lpcAccumulate(const std::uint8_t* base,
                   const std::ptrdiff_t* off,
                   const std::int16_t* w,
                   int nTaps,
                   std::int32_t bias,
                   std::int32_t* acc,
                   std::size_t n)
{
#if defined(VC_DELTA3D_LPC_SIMD)
    constexpr std::size_t kBlock = 16;
    if (n >= kBlock) {
        std::array<std::ptrdiff_t, kLpcTaps> o;
        std::copy(off, off + nTaps, o.begin());
        std::size_t i = 0;
        for (; i + kBlock <= n; i += kBlock) {
            lpcAccumulateBlock(base, o.data(), w, nTaps, bias, acc + i);
            for (int t = 0; t < nTaps; ++t)
                o[t] += kBlock;
        }
        if (i < n) {
            const std::size_t back = kBlock - (n - i);
            for (int t = 0; t < nTaps; ++t)
                o[t] -= static_cast<std::ptrdiff_t>(back);
            lpcAccumulateBlock(base, o.data(), w, nTaps, bias,
                               acc + n - kBlock);
        }
        return;
    }
#endif
    for (std::size_t i = 0; i < n; ++i)
        acc[i] = bias;
    for (int t = 0; t < nTaps; ++t) {
        const std::uint8_t* s = base + off[t];
        const std::int32_t wt = w[t];
        for (std::size_t i = 0; i < n; ++i)
            acc[i] += wt * s[i];
    }
}

// Pointers to the Lorenzo neighbor rows of row (zz, yy).
struct LorenzoRows {
    const std::uint8_t *n, *u, *nu;
};
LorenzoRows lorenzoRows(const std::uint8_t* v, const LpcGeometry& g,
                        std::size_t zz, std::size_t yy,
                        const std::uint8_t* zeros)
{
    const std::uint8_t* row = v + (zz * g.y + yy) * g.x;
    const std::size_t slice = g.y * g.x;
    return {yy > 0 ? row - g.x : zeros, zz > 0 ? row - slice : zeros,
            yy > 0 && zz > 0 ? row - slice - g.x : zeros};
}

// Encoder: residuals of src under the LPC predictor (src is left intact).
void lpcResiduals(const std::uint8_t* src, std::uint8_t* res,
                  const LpcGeometry& g, const LpcParams& p)
{
    std::vector<std::uint8_t> zeros(g.x, 0);
    std::vector<std::int32_t> acc(g.x);
    for (std::size_t zz = 0; zz < g.z; ++zz)
        for (std::size_t yy = 0; yy < g.y; ++yy) {
            const std::size_t r = (zz * g.y + yy) * g.x;
            const std::uint8_t* row = src + r;
            const auto l = lorenzoRows(src, g, zz, yy, zeros.data());
            // Voxels [lo, hi) use the linear predictor, the rest Lorenzo.
            std::size_t lo = g.x, hi = g.x;
            if (g.interiorRow(zz, yy)) {
                lo = kLpcRadius;
                hi = g.x - kLpcRadius;
                lpcAccumulate(row + lo, g.off.data(), p.w.data(), kLpcTaps,
                              p.bias, acc.data(), hi - lo);
                for (std::size_t xx = lo; xx < hi; ++xx)
                    res[r + xx] = static_cast<std::uint8_t>(
                        row[xx] - (acc[xx - lo] >> kLpcShift));
            }
            for (std::size_t xx = 0; xx < g.x; ++xx)
                if (xx < lo || xx >= hi)
                    res[r + xx] = static_cast<std::uint8_t>(
                        row[xx] - lorenzoPredict(row, l.n, l.u, l.nu, xx));
        }
}

// Decoder: v holds residuals on entry and voxels on return. Rows are
// reconstructed in raster order; within an interior row the 60 taps on
// earlier rows are summed vectorized, leaving only the two current-row taps
// in the serial scan.
void lpcReconstruct(std::uint8_t* v, const LpcGeometry& g, const LpcParams& p)
{
    std::vector<std::uint8_t> zeros(g.x, 0);
    std::vector<std::int32_t> acc(g.x);
    const std::int32_t wWW = p.w[kLpcTaps - 2];
    const std::int32_t wW = p.w[kLpcTaps - 1];
    for (std::size_t zz = 0; zz < g.z; ++zz)
        for (std::size_t yy = 0; yy < g.y; ++yy) {
            std::uint8_t* row = v + (zz * g.y + yy) * g.x;
            const auto l = lorenzoRows(v, g, zz, yy, zeros.data());
            std::size_t lo = g.x, hi = g.x;
            if (g.interiorRow(zz, yy)) {
                lo = kLpcRadius;
                hi = g.x - kLpcRadius;
            }
            for (std::size_t xx = 0; xx < lo; ++xx)
                row[xx] = static_cast<std::uint8_t>(
                    row[xx] + lorenzoPredict(row, l.n, l.u, l.nu, xx));
            if (lo < hi) {
                lpcAccumulate(row + lo, g.off.data(), p.w.data(), kLpcPrevTaps,
                              p.bias, acc.data(), hi - lo);
                for (std::size_t xx = lo; xx < hi; ++xx) {
                    const std::int32_t s =
                        acc[xx - lo] + wWW * row[xx - 2] + wW * row[xx - 1];
                    row[xx] =
                        static_cast<std::uint8_t>(row[xx] + (s >> kLpcShift));
                }
            }
            for (std::size_t xx = hi; xx < g.x; ++xx)
                row[xx] = static_cast<std::uint8_t>(
                    row[xx] + lorenzoPredict(row, l.n, l.u, l.nu, xx));
        }
}

// Sum over i < n of a[i] * b[i] for uint8 rows (n < 66052, so no overflow).
inline std::uint32_t dotU8(const std::uint8_t* a, const std::uint8_t* b,
                           std::size_t n)
{
    std::uint32_t s = 0;
    for (std::size_t i = 0; i < n; ++i)
        s += static_cast<std::uint32_t>(a[i]) * b[i];
    return s;
}

// Least-squares fit of the LPC weights and intercept on the interior voxels
// of every `stride`-th row along z and y. The normal equations are
// accumulated exactly in integers, as dot products between the sampled row
// segments at each tap offset, so the fit does not depend on the platform.
// Returns false if the quantized weights do not fit the format.
bool lpcFit(const std::uint8_t* v, const LpcGeometry& g, std::size_t stride,
            LpcParams& out)
{
    constexpr int kVec = kLpcTaps + 1;  // taps, then the target voxel
    constexpr int k = kLpcTaps + 1;     // unknowns: weights, then intercept
    std::array<std::ptrdiff_t, kVec> off{};
    std::copy(g.off.begin(), g.off.end(), off.begin());
    off[kLpcTaps] = 0;

    std::vector<std::uint64_t> dot(kVec * kVec, 0);
    std::array<std::uint64_t, kVec> sum{};
    std::uint64_t count = 0;
    const std::size_t n = g.x - 2 * kLpcRadius;
    for (std::size_t zz = kLpcRadius; zz < g.z; zz += stride)
        for (std::size_t yy = kLpcRadius; yy + kLpcRadius < g.y; yy += stride) {
            const std::uint8_t* base = v + (zz * g.y + yy) * g.x + kLpcRadius;
            for (int a = 0; a < kVec; ++a) {
                const std::uint8_t* pa = base + off[a];
                std::uint32_t s = 0;
                for (std::size_t i = 0; i < n; ++i)
                    s += pa[i];
                sum[a] += s;
                for (int b = a; b < kVec; ++b)
                    dot[a * kVec + b] += dotU8(pa, base + off[b], n);
            }
            count += n;
        }
    if (count == 0)
        return false;

    // Augmented normal equations [G | rhs] for u = (taps..., 1) -> target.
    std::vector<double> m(static_cast<std::size_t>(k) * (k + 1));
    auto at = [&](int r, int c) -> double& { return m[r * (k + 1) + c]; };
    for (int a = 0; a < kLpcTaps; ++a) {
        for (int b = a; b < kLpcTaps; ++b)
            at(a, b) = at(b, a) = static_cast<double>(dot[a * kVec + b]);
        at(a, kLpcTaps) = at(kLpcTaps, a) = static_cast<double>(sum[a]);
        at(a, k) = static_cast<double>(dot[a * kVec + kLpcTaps]);
    }
    at(kLpcTaps, kLpcTaps) = static_cast<double>(count);
    at(kLpcTaps, k) = static_cast<double>(sum[kLpcTaps]);
    // Ridge keeps degenerate chunks (constant, all zero) solvable.
    for (int a = 0; a < k; ++a)
        at(a, a) += 1e-6 * (1 + at(a, a));

    // Gauss-Jordan elimination with partial pivoting.
    for (int c = 0; c < k; ++c) {
        int piv = c;
        for (int r = c + 1; r < k; ++r)
            if (std::fabs(at(r, c)) > std::fabs(at(piv, c)))
                piv = r;
        if (piv != c)
            for (int cc = 0; cc <= k; ++cc)
                std::swap(at(c, cc), at(piv, cc));
        const double d = at(c, c);
        if (d == 0)
            continue;
        for (int r = 0; r < k; ++r) {
            if (r == c || at(r, c) == 0)
                continue;
            const double f = at(r, c) / d;
            for (int cc = c; cc <= k; ++cc)
                at(r, cc) -= f * at(c, cc);
        }
    }

    constexpr double kScale = 1 << kLpcShift;
    for (int a = 0; a < kLpcTaps; ++a) {
        const double w = at(a, a) != 0 ? at(a, k) / at(a, a) : 0.0;
        const double q = std::nearbyint(w * kScale);
        if (!(q >= std::numeric_limits<std::int16_t>::min() &&
              q <= std::numeric_limits<std::int16_t>::max()))
            return false;
        out.w[a] = static_cast<std::int16_t>(q);
    }
    const double c0 = at(kLpcTaps, kLpcTaps) != 0
                          ? at(kLpcTaps, k) / at(kLpcTaps, kLpcTaps)
                          : 0.0;
    // The rounding half is folded into the bias, so ">> 12" rounds.
    const double bias = std::nearbyint(c0 * kScale) + (1 << (kLpcShift - 1));
    if (!(std::fabs(bias) <= kLpcMaxBias))
        return false;
    out.bias = static_cast<std::int32_t>(bias);
    return true;
}

void writeLpcParams(std::byte* out, const LpcParams& p)
{
    out[0] = static_cast<std::byte>(kLpcStencilBox2);
    for (int t = 0; t < kLpcTaps; ++t) {
        const auto u = static_cast<std::uint16_t>(p.w[t]);
        out[1 + 2 * t] = static_cast<std::byte>(u & 0xFF);
        out[2 + 2 * t] = static_cast<std::byte>(u >> 8);
    }
    writeU32(out + 1 + 2 * kLpcTaps, static_cast<std::uint32_t>(p.bias));
}

std::optional<LpcParams> readLpcParams(std::span<const std::byte> in)
{
    if (in.size() < kLpcParamBytes ||
        static_cast<unsigned char>(in[0]) != kLpcStencilBox2)
        return std::nullopt;
    LpcParams p;
    for (int t = 0; t < kLpcTaps; ++t)
        p.w[t] = static_cast<std::int16_t>(
            static_cast<unsigned>(in[1 + 2 * t]) |
            (static_cast<unsigned>(in[2 + 2 * t]) << 8));
    p.bias = static_cast<std::int32_t>(readU32(in.data() + 1 + 2 * kLpcTaps));
    if (p.bias > kLpcMaxBias || p.bias < -kLpcMaxBias)
        return std::nullopt;
    return p;
}

// Header byte 7: bits 0-3 carry the predictor marker. Delta payloads with
// bit 7 set carry the delta-axis mask in bits 4-6; payloads from before
// per-chunk selection have bits 4-7 clear and imply full zyx, and a clear
// flag with nonzero mask bits is invalid. LPC payloads have bits 4-7 clear.
struct HeaderByte {
    Predictor predictor;
    unsigned mask;
    bool hasMask;
};

std::optional<HeaderByte> parseHeaderByte(unsigned char b)
{
    if ((b & 0x0F) == static_cast<unsigned>(Predictor::Lpc)) {
        if ((b & 0xF0) != 0)
            return std::nullopt;
        return HeaderByte{Predictor::Lpc, 0, false};
    }
    const bool hasMask = (b & 0x80) != 0;
    if ((b & 0x0F) != static_cast<unsigned>(Predictor::Delta) ||
        (!hasMask && (b & 0x70) != 0))
        return std::nullopt;
    HeaderByte out;
    out.predictor = Predictor::Delta;
    out.hasMask = hasMask;
    out.mask = hasMask ? (b >> 4) & 7u : kDeltaMaskZyx;
    return out;
}

void writeHeader(std::vector<std::byte>& output, std::size_t elemSize,
                 int quantBinWidth, unsigned predictorByte,
                 std::size_t z, std::size_t y, std::size_t x)
{
    std::copy(kWireMagic.begin(), kWireMagic.end(), output.begin());
    output[4] = static_cast<std::byte>(kFormatVersion);
    output[5] = static_cast<std::byte>(elemSize);
    output[6] = static_cast<std::byte>(quantBinWidth);
    output[7] = static_cast<std::byte>(predictorByte);
    writeU32(output.data() + 8, static_cast<std::uint32_t>(z));
    writeU32(output.data() + 12, static_cast<std::uint32_t>(y));
    writeU32(output.data() + 16, static_cast<std::uint32_t>(x));
}

// Row sampling stride of the LPC fit along z and y: the largest stride up to
// 8 that still leaves kLpcFitSamples interior voxels (falling back to every
// row). On 128^3 scroll chunks stride 8 fits as well as stride 4 (2.332 vs
// 2.331 bits/voxel) at a quarter of the cost.
constexpr std::size_t kLpcFitSamples = 16384;
std::size_t lpcFitStride(const LpcGeometry& g)
{
    for (std::size_t s = 8; s > 1; --s) {
        const std::size_t rows = ((g.z - kLpcRadius + s - 1) / s) *
                                 ((g.y - 2 * kLpcRadius + s - 1) / s);
        if (rows * (g.x - 2 * kLpcRadius) >= kLpcFitSamples)
            return s;
    }
    return 1;
}

// LPC payload for a quantized uint8 chunk, or nothing if the fitted weights
// do not fit the format.
std::optional<std::vector<std::byte>> compressLpc(
    const std::uint8_t* data, std::size_t z, std::size_t y, std::size_t x,
    int quantBinWidth)
{
    const LpcGeometry g(z, y, x);
    LpcParams params;
    if (!lpcFit(data, g, lpcFitStride(g), params))
        return std::nullopt;
    std::vector<std::uint8_t> residuals(z * y * x);
    lpcResiduals(data, residuals.data(), g, params);
    auto output = ransCompressFrame(
        {reinterpret_cast<const std::byte*>(residuals.data()), residuals.size()},
        kHeaderSize + kLpcParamBytes);
    writeHeader(output, 1, quantBinWidth,
                static_cast<unsigned>(Predictor::Lpc), z, y, x);
    writeLpcParams(output.data() + kHeaderSize, params);
    return output;
}

} // namespace

std::vector<std::byte> compress(std::span<const std::byte> input,
                                std::array<int, 3> shapeZYX,
                                std::size_t elemSize,
                                int quantBinWidth,
                                Predictor predictor)
{
    const bool shapeValid =
        (elemSize == 1 || elemSize == 2) &&
        shapeZYX[0] > 0 && shapeZYX[1] > 0 && shapeZYX[2] > 0 &&
        static_cast<std::size_t>(shapeZYX[0]) *
                static_cast<std::size_t>(shapeZYX[1]) *
                static_cast<std::size_t>(shapeZYX[2]) * elemSize ==
            input.size();
    if (!shapeValid) {
        throw std::invalid_argument(
            "compress: chunk shape/element size does not match payload");
    }
    if (quantBinWidth < 1 || quantBinWidth > 255) {
        throw std::invalid_argument(
            "compress: quantization bin width must be in [1, 255]");
    }

    const auto z = static_cast<std::size_t>(shapeZYX[0]);
    const auto y = static_cast<std::size_t>(shapeZYX[1]);
    const auto x = static_cast<std::size_t>(shapeZYX[2]);

    std::vector<std::byte> filtered(input.begin(), input.end());
    quantize({filtered.data(), filtered.size()}, elemSize, quantBinWidth);

    std::optional<std::vector<std::byte>> lpc;
    if (predictor == Predictor::Lpc && elemSize == 1 &&
        std::min({z, y, x}) >= static_cast<std::size_t>(kLpcMinDim))
        lpc = compressLpc(reinterpret_cast<const std::uint8_t*>(filtered.data()),
                          z, y, x, quantBinWidth);

    // The uint16 probe isn't implemented (16-bit volumes are rare in the
    // streaming path). Degenerate dims skip the probe; the extra passes are
    // no-ops there anyway.
    unsigned mask = kDeltaMaskZyx;
    if (elemSize == 1 && z >= 2 && y >= 2 && x >= 2)
        mask = selectDeltaMask(
            reinterpret_cast<const std::uint8_t*>(filtered.data()), z, y, x);
    if (elemSize == 1)
        deltaFilter(reinterpret_cast<std::uint8_t*>(filtered.data()), z, y, x,
                    mask);
    else
        deltaFilter(reinterpret_cast<std::uint16_t*>(filtered.data()), z, y, x,
                    mask);

    const std::span<const std::byte> filteredSpan(filtered.data(),
                                                  filtered.size());
    auto output = ransCompressFrame(filteredSpan, kHeaderSize);
    writeHeader(output, elemSize, quantBinWidth, 0x80u | (mask << 4) | 1u,
                z, y, x);
    if (lpc && lpc->size() < output.size())
        return std::move(*lpc);
    return output;
}

std::optional<Predictor> predictor(std::span<const std::byte> input)
{
    if (!hasHeader(input, kWireMagic))
        return std::nullopt;
    const auto parsed = parseHeaderByte(static_cast<unsigned char>(input[7]));
    if (!parsed)
        return std::nullopt;
    return parsed->predictor;
}

void quantize(std::span<std::byte> data,
              std::size_t elemSize,
              int quantBinWidth)
{
    if (quantBinWidth <= 1)
        return;
    if (elemSize == 1)
        quantizeValues(reinterpret_cast<std::uint8_t*>(data.data()),
                       data.size(), quantBinWidth);
    else
        quantizeValues(reinterpret_cast<std::uint16_t*>(data.data()),
                       data.size() / 2, quantBinWidth);
}

std::optional<int> quantization(std::span<const std::byte> input)
{
    if (!hasHeader(input, kWireMagic))
        return std::nullopt;
    return std::max(1, static_cast<int>(input[6]));
}

std::optional<int> deltaMask(std::span<const std::byte> input)
{
    if (!hasHeader(input, kWireMagic))
        return std::nullopt;
    const auto parsed = parseHeaderByte(static_cast<unsigned char>(input[7]));
    if (!parsed || !parsed->hasMask)
        return std::nullopt;
    return static_cast<int>(parsed->mask);
}

bool decompressInto(std::span<const std::byte> input,
                    std::span<std::byte> output)
{
    return decompressIntoWithMagic(input, output, kWireMagic);
}

bool decompressIntoWithMagic(std::span<const std::byte> input,
                             std::span<std::byte> output,
                             const WireMagic& expectedMagic)
{
    if (!hasHeader(input, expectedMagic))
        return false;

    const auto elemSize = static_cast<std::size_t>(input[5]);
    const std::size_t z = readU32(input.data() + 8);
    const std::size_t y = readU32(input.data() + 12);
    const std::size_t x = readU32(input.data() + 16);
    if ((elemSize != 1 && elemSize != 2) || z == 0 || y == 0 || x == 0 ||
        z * y * x * elemSize != output.size())
        return false;
    const auto parsed = parseHeaderByte(static_cast<unsigned char>(input[7]));
    if (!parsed)
        return false;

    if (parsed->predictor == Predictor::Lpc) {
        if (elemSize != 1)
            return false;
        const auto params = readLpcParams(input.subspan(kHeaderSize));
        if (!params ||
            !ransDecompressFrame(input.subspan(kHeaderSize + kLpcParamBytes),
                                 output.data(), output.size()))
            return false;
        lpcReconstruct(reinterpret_cast<std::uint8_t*>(output.data()),
                       LpcGeometry(z, y, x), *params);
        return true;
    }

    if (!ransDecompressFrame(input.subspan(kHeaderSize), output.data(),
                             output.size()))
        return false;

    if (elemSize == 1)
        deltaUnfilter(reinterpret_cast<std::uint8_t*>(output.data()), z, y, x,
                      parsed->mask);
    else
        deltaUnfilter(reinterpret_cast<std::uint16_t*>(output.data()), z, y, x,
                      parsed->mask);
    return true;
}

std::optional<std::vector<std::byte>> decompress(
    std::span<const std::byte> input,
    std::size_t expectedSize)
{
    return decompressWithMagic(input, expectedSize, kWireMagic);
}

std::optional<std::vector<std::byte>> decompressWithMagic(
    std::span<const std::byte> input,
    std::size_t expectedSize,
    const WireMagic& expectedMagic)
{
    std::vector<std::byte> output(expectedSize);
    if (!decompressIntoWithMagic(input, output, expectedMagic))
        return std::nullopt;
    return output;
}

} // namespace vc_delta3d
