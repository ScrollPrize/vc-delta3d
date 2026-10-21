#include <vc_delta3d/codec.hpp>

#include <algorithm>
#include <bit>
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
#elif defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#endif
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>  // AVX2 LPC path, selected at run time
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

// ---- LPC: per-chunk linear prediction from a per-chunk subset of the 62
// causal voxels of the radius-2 box (see the layout notes in codec.hpp).
// Weights are in 1/256 units and only bits 8-15 of a weighted sum form the
// prediction (mod 256), so every sum is taken mod 2^16: encoder and decoder
// agree exactly for any weights and bias, and the SIMD paths run on 16-bit
// lanes. ----

constexpr int kLpcTaps = 62;
constexpr std::size_t kLpcRadius = 2;
constexpr int kLpcShift = 8;
constexpr unsigned char kLpcStencilBox2 = 2;
// Stencil id, tap mask and bias; each selected tap adds a 2-byte weight.
constexpr std::size_t kLpcFixedParamBytes = 1 + 8 + 2;
// Taps the encoder keeps per chunk. Greedy selection of 40 out of 62 costs
// ~0.15% size on 2.4um scroll chunks and saves a third of the decode sum.
constexpr int kLpcSelectedTaps = 40;

struct LpcOffset {
    int dz, dy, dx;
};

// Causal offsets in tap order. The two current-row taps come last (dx = -2,
// then dx = -1); all others only read earlier rows.
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
constexpr int kLpcTapWW = kLpcTaps - 2;
constexpr int kLpcTapW = kLpcTaps - 1;
static_assert(kLpcStencil[kLpcTapWW].dz == 0 && kLpcStencil[kLpcTapWW].dy == 0 &&
              kLpcStencil[kLpcTapWW].dx == -2);
static_assert(kLpcStencil[kLpcTapW].dx == -1);

struct LpcParams {
    std::uint64_t mask = 0;                  // bit t set: tap t is used
    std::array<std::int16_t, kLpcTaps> w{};  // 1/256 units, 0 for unused taps
    std::uint16_t bias = 0;                  // 1/256 units, rounding included
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

// The selected taps of a chunk as offset/weight lists. With rowTaps false,
// the two current-row taps are kept apart (wWW, wW) for the decoder's serial
// scan and the list only reads earlier rows.
struct LpcTapList {
    std::array<std::ptrdiff_t, kLpcTaps> off{};
    std::array<std::uint16_t, kLpcTaps> w{};
    int n = 0;
    std::uint16_t wWW = 0, wW = 0;
};

LpcTapList lpcTapList(const LpcParams& p, const LpcGeometry& g, bool rowTaps)
{
    LpcTapList list;
    for (int t = 0; t < kLpcTaps; ++t) {
        if (!((p.mask >> t) & 1))
            continue;
        const auto w = static_cast<std::uint16_t>(p.w[t]);
        if (!rowTaps && t == kLpcTapWW) {
            list.wWW = w;
        } else if (!rowTaps && t == kLpcTapW) {
            list.wW = w;
        } else {
            list.off[list.n] = g.off[t];
            list.w[list.n] = w;
            ++list.n;
        }
    }
    return list;
}

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

// acc[i] = bias + sum over the taps of w[t] * base[off[t] + i] (mod 2^16),
// for i in [0, n). The sum is exact in any order, so the SIMD paths only
// regroup it: fixed-width blocks keep their 16-bit accumulators in registers
// across all taps, and a short final block overlaps the previous one
// (recomputing a few sums) instead of falling back to scalar code.
void lpcAccumulateScalar(const std::uint8_t* base, const LpcTapList& taps,
                         std::uint16_t bias, std::uint16_t* acc, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
        acc[i] = bias;
    for (int t = 0; t < taps.n; ++t) {
        const std::uint8_t* s = base + taps.off[t];
        const unsigned w = taps.w[t];
        for (std::size_t i = 0; i < n; ++i)
            acc[i] = static_cast<std::uint16_t>(acc[i] + w * s[i]);
    }
}

// Runs block(base + i, acc + i) over [0, n) in steps of kBlock (n >= kBlock).
template <std::size_t kBlock, typename Block>
inline void lpcBlocks(const std::uint8_t* base, std::uint16_t* acc,
                      std::size_t n, Block block)
{
    std::size_t i = 0;
    for (; i + kBlock <= n; i += kBlock)
        block(base + i, acc + i);
    if (i < n)
        block(base + n - kBlock, acc + n - kBlock);
}

#if defined(__aarch64__)
#define VC_DELTA3D_LPC_NEON 1
void lpcAccumulateNeon(const std::uint8_t* base, const LpcTapList& taps,
                       std::uint16_t bias, std::uint16_t* acc, std::size_t n)
{
    lpcBlocks<16>(base, acc, n, [&](const std::uint8_t* b, std::uint16_t* a) {
        uint16x8_t a0 = vdupq_n_u16(bias), a1 = a0;
        for (int t = 0; t < taps.n; ++t) {
            const uint8x16_t v = vld1q_u8(b + taps.off[t]);
            a0 = vmlaq_n_u16(a0, vmovl_u8(vget_low_u8(v)), taps.w[t]);
            a1 = vmlaq_n_u16(a1, vmovl_high_u8(v), taps.w[t]);
        }
        vst1q_u16(a, a0);
        vst1q_u16(a + 8, a1);
    });
}
#endif

#if defined(__SSE2__) || defined(_M_X64)
#define VC_DELTA3D_LPC_SSE2 1
void lpcAccumulateSse2(const std::uint8_t* base, const LpcTapList& taps,
                       std::uint16_t bias, std::uint16_t* acc, std::size_t n)
{
    const __m128i zero = _mm_setzero_si128();
    lpcBlocks<16>(base, acc, n, [&](const std::uint8_t* b, std::uint16_t* a) {
        __m128i a0 = _mm_set1_epi16(static_cast<short>(bias)), a1 = a0;
        for (int t = 0; t < taps.n; ++t) {
            const __m128i v = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(b + taps.off[t]));
            const __m128i w = _mm_set1_epi16(static_cast<short>(taps.w[t]));
            a0 = _mm_add_epi16(a0, _mm_mullo_epi16(_mm_unpacklo_epi8(v, zero), w));
            a1 = _mm_add_epi16(a1, _mm_mullo_epi16(_mm_unpackhi_epi8(v, zero), w));
        }
        _mm_storeu_si128(reinterpret_cast<__m128i*>(a), a0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(a + 8), a1);
    });
}
#endif

// AVX2 is chosen at run time (GCC/Clang), so baseline x86-64 builds such as
// the published wheels still use it where the CPU has it. Define
// VC_DELTA3D_DISABLE_AVX2 to compare against the SSE2 path.
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__)) && !defined(VC_DELTA3D_DISABLE_AVX2)
#define VC_DELTA3D_LPC_AVX2 1
__attribute__((target("avx2"))) void lpcAccumulateAvx2(
    const std::uint8_t* base, const LpcTapList& taps, std::uint16_t bias,
    std::uint16_t* acc, std::size_t n)
{
    __m256i w[kLpcTaps];
    for (int t = 0; t < taps.n; ++t)
        w[t] = _mm256_set1_epi16(static_cast<short>(taps.w[t]));
    lpcBlocks<32>(base, acc, n, [&](const std::uint8_t* b, std::uint16_t* a)
                                    __attribute__((target("avx2"))) {
        __m256i a0 = _mm256_set1_epi16(static_cast<short>(bias)), a1 = a0;
        for (int t = 0; t < taps.n; ++t) {
            const __m256i v = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(b + taps.off[t]));
            a0 = _mm256_add_epi16(a0, _mm256_mullo_epi16(
                _mm256_cvtepu8_epi16(_mm256_castsi256_si128(v)), w[t]));
            a1 = _mm256_add_epi16(a1, _mm256_mullo_epi16(
                _mm256_cvtepu8_epi16(_mm256_extracti128_si256(v, 1)), w[t]));
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a), a0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + 16), a1);
    });
}

bool cpuHasAvx2()
{
    static const bool has = __builtin_cpu_supports("avx2");
    return has;
}
#endif

void lpcAccumulate(const std::uint8_t* base, const LpcTapList& taps,
                   std::uint16_t bias, std::uint16_t* acc, std::size_t n)
{
#if defined(VC_DELTA3D_LPC_AVX2)
    if (n >= 32 && cpuHasAvx2())
        return lpcAccumulateAvx2(base, taps, bias, acc, n);
#endif
#if defined(VC_DELTA3D_LPC_SSE2)
    if (n >= 16)
        return lpcAccumulateSse2(base, taps, bias, acc, n);
#endif
#if defined(VC_DELTA3D_LPC_NEON)
    if (n >= 16)
        return lpcAccumulateNeon(base, taps, bias, acc, n);
#endif
    lpcAccumulateScalar(base, taps, bias, acc, n);
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
    const LpcTapList taps = lpcTapList(p, g, true);
    std::vector<std::uint8_t> zeros(g.x, 0);
    std::vector<std::uint16_t> acc(g.x);
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
                lpcAccumulate(row + lo, taps, p.bias, acc.data(), hi - lo);
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
// reconstructed in raster order; within an interior row the taps on earlier
// rows are summed vectorized, leaving only the two current-row taps in the
// serial scan.
void lpcReconstruct(std::uint8_t* v, const LpcGeometry& g, const LpcParams& p)
{
    const LpcTapList taps = lpcTapList(p, g, false);
    std::vector<std::uint8_t> zeros(g.x, 0);
    std::vector<std::uint16_t> acc(g.x);
    const unsigned wWW = taps.wWW, wW = taps.wW;
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
                lpcAccumulate(row + lo, taps, p.bias, acc.data(), hi - lo);
                for (std::size_t xx = lo; xx < hi; ++xx) {
                    const auto s = static_cast<std::uint16_t>(
                        acc[xx - lo] + wWW * row[xx - 2] + wW * row[xx - 1]);
                    row[xx] =
                        static_cast<std::uint8_t>(row[xx] + (s >> kLpcShift));
                }
            }
            for (std::size_t xx = hi; xx < g.x; ++xx)
                row[xx] = static_cast<std::uint8_t>(
                    row[xx] + lorenzoPredict(row, l.n, l.u, l.nu, xx));
        }
}

// ---- LPC fit (encoder only; any weights decode correctly). ----

// Sum over i < n of a[i] * b[i] for uint8 rows (n < 66052, so no overflow).
inline std::uint32_t dotU8(const std::uint8_t* a, const std::uint8_t* b,
                           std::size_t n)
{
    std::uint32_t s = 0;
    for (std::size_t i = 0; i < n; ++i)
        s += static_cast<std::uint32_t>(a[i]) * b[i];
    return s;
}

// Normal equations of the least-squares fit, unknowns = 62 tap weights then
// the intercept (index kLpcTaps).
constexpr int kLpcUnknowns = kLpcTaps + 1;
struct LpcNormal {
    std::array<double, kLpcUnknowns * kLpcUnknowns> a;
    std::array<double, kLpcUnknowns> b;
    double at(int r, int c) const { return a[r * kLpcUnknowns + c]; }
};

// Accumulates the normal equations over the interior voxels of every
// `stride`-th row along z and y, exactly in integers (dot products between
// the sampled row segments at each tap offset), so the fit does not depend on
// the platform. Returns false if there are no interior voxels.
bool lpcNormalEquations(const std::uint8_t* v, const LpcGeometry& g,
                        std::size_t stride, LpcNormal& ne)
{
    constexpr int kVec = kLpcTaps + 1;  // taps, then the target voxel
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

    constexpr int k = kLpcUnknowns;
    for (int a = 0; a < kLpcTaps; ++a) {
        for (int b = a; b < kLpcTaps; ++b)
            ne.a[a * k + b] = ne.a[b * k + a] =
                static_cast<double>(dot[a * kVec + b]);
        ne.a[a * k + kLpcTaps] = ne.a[kLpcTaps * k + a] =
            static_cast<double>(sum[a]);
        ne.b[a] = static_cast<double>(dot[a * kVec + kLpcTaps]);
    }
    ne.a[kLpcTaps * k + kLpcTaps] = static_cast<double>(count);
    ne.b[kLpcTaps] = static_cast<double>(sum[kLpcTaps]);
    // Ridge keeps degenerate chunks (constant, all zero) positive definite.
    for (int a = 0; a < k; ++a)
        ne.a[a * k + a] += 1e-6 * (1 + ne.a[a * k + a]);
    return true;
}

// Solves the normal equations restricted to the unknowns `idx` with the
// right-hand side rhs (Cholesky). Returns false unless positive definite.
bool lpcSolve(const LpcNormal& ne, const std::vector<int>& idx,
              const std::vector<double>& rhs, std::vector<double>& x)
{
    const std::size_t k = idx.size();
    std::vector<double> l(k * k, 0.0);
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t j = 0; j <= i; ++j) {
            double s = ne.at(idx[i], idx[j]);
            for (std::size_t m = 0; m < j; ++m)
                s -= l[i * k + m] * l[j * k + m];
            if (i == j) {
                if (!(s > 0))
                    return false;
                l[i * k + i] = std::sqrt(s);
            } else {
                l[i * k + j] = s / l[j * k + j];
            }
        }
    x.assign(k, 0.0);
    for (std::size_t i = 0; i < k; ++i) {
        double s = rhs[i];
        for (std::size_t m = 0; m < i; ++m)
            s -= l[i * k + m] * x[m];
        x[i] = s / l[i * k + i];
    }
    for (std::size_t i = k; i-- > 0;) {
        double s = x[i];
        for (std::size_t m = i + 1; m < k; ++m)
            s -= l[m * k + i] * x[m];
        x[i] = s / l[i * k + i];
    }
    return true;
}

// Greedy forward selection: starting from the intercept alone, repeatedly
// add the tap that lowers the least-squares error most, until `count` taps
// are in. Each candidate's error reduction comes from the Schur complement
// against the running Cholesky factor, so a step costs O(62 k^2).
std::uint64_t lpcSelectTaps(const LpcNormal& ne, int count)
{
    constexpr int k = kLpcUnknowns;
    std::vector<double> l(k * k, 0.0);  // Cholesky factor, selection order
    std::vector<int> sel;
    std::vector<double> z;              // l^-1 b over the selection
    std::vector<double> u(k), bestU(k);
    sel.push_back(kLpcTaps);
    l[0] = std::sqrt(ne.at(kLpcTaps, kLpcTaps));
    z.push_back(ne.b[kLpcTaps] / l[0]);
    std::uint64_t mask = 0;
    for (int step = 0; step < count; ++step) {
        const std::size_t n = sel.size();
        int best = -1;
        double bestGain = -1, bestD = 0, bestNum = 0;
        for (int t = 0; t < kLpcTaps; ++t) {
            if ((mask >> t) & 1)
                continue;
            double d = ne.at(t, t), num = ne.b[t];
            for (std::size_t i = 0; i < n; ++i) {
                double s = ne.at(sel[i], t);
                for (std::size_t m = 0; m < i; ++m)
                    s -= l[i * k + m] * u[m];
                u[i] = s / l[i * k + i];
                d -= u[i] * u[i];
                num -= u[i] * z[i];
            }
            if (!(d > 0))
                continue;
            const double gain = num * num / d;
            if (gain > bestGain) {
                best = t;
                bestGain = gain;
                bestD = d;
                bestNum = num;
                std::copy(u.begin(), u.begin() + n, bestU.begin());
            }
        }
        if (best < 0)
            break;
        std::copy(bestU.begin(), bestU.begin() + n, l.begin() + n * k);
        l[n * k + n] = std::sqrt(bestD);
        z.push_back(bestNum / l[n * k + n]);
        sel.push_back(best);
        mask |= std::uint64_t{1} << best;
    }
    return mask;
}

// Weights for the selected taps at 1/256 steps, rounded sequentially (Babai
// style): round the largest free weight, refit the others with it fixed,
// repeat; the intercept is fitted last. Rounding all weights at once costs
// 1.5-5% size on scroll chunks, this ~0.05-0.1%.
bool lpcQuantize(const LpcNormal& ne, std::uint64_t mask, LpcParams& out)
{
    constexpr double kScale = 1 << kLpcShift;
    std::vector<int> free;
    for (int t = 0; t < kLpcTaps; ++t)
        if ((mask >> t) & 1)
            free.push_back(t);
    free.push_back(kLpcTaps);
    std::vector<int> fixed;
    std::array<double, kLpcTaps> value{};  // rounded weights, as reals
    std::vector<double> rhs, x;
    out = LpcParams{};
    out.mask = mask;
    while (true) {
        rhs.assign(free.size(), 0.0);
        for (std::size_t i = 0; i < free.size(); ++i) {
            rhs[i] = ne.b[free[i]];
            for (const int f : fixed)
                rhs[i] -= ne.at(free[i], f) * value[f];
        }
        if (!lpcSolve(ne, free, rhs, x))
            return false;
        if (free.size() == 1)
            break;
        std::size_t pick = 0;
        for (std::size_t i = 1; i + 1 < free.size(); ++i)
            if (std::fabs(x[i]) > std::fabs(x[pick]))
                pick = i;
        const double q = std::clamp(std::nearbyint(x[pick] * kScale), -32768.0, 32767.0);
        if (!std::isfinite(q))
            return false;
        const int t = free[pick];
        value[t] = q / kScale;
        out.w[t] = static_cast<std::int16_t>(q);
        fixed.push_back(t);
        free.erase(free.begin() + static_cast<std::ptrdiff_t>(pick));
    }
    // The rounding half is folded into the bias, so ">> 8" rounds.
    const double bias = std::nearbyint(x[0] * kScale) + (1 << (kLpcShift - 1));
    if (!std::isfinite(bias) || std::fabs(bias) > 1e15)
        return false;
    out.bias = static_cast<std::uint16_t>(static_cast<std::int64_t>(bias));
    return true;
}

bool lpcFit(const std::uint8_t* v, const LpcGeometry& g, std::size_t stride,
            LpcParams& out)
{
    LpcNormal ne;
    if (!lpcNormalEquations(v, g, stride, ne))
        return false;
    return lpcQuantize(ne, lpcSelectTaps(ne, kLpcSelectedTaps), out);
}

std::size_t lpcParamBytes(std::uint64_t mask)
{
    return kLpcFixedParamBytes + 2 * static_cast<std::size_t>(std::popcount(mask));
}

void writeLpcParams(std::byte* out, const LpcParams& p)
{
    out[0] = static_cast<std::byte>(kLpcStencilBox2);
    for (int i = 0; i < 8; ++i)
        out[1 + i] = static_cast<std::byte>((p.mask >> (8 * i)) & 0xFF);
    std::byte* o = out + 9;
    for (int t = 0; t < kLpcTaps; ++t)
        if ((p.mask >> t) & 1) {
            const auto u = static_cast<std::uint16_t>(p.w[t]);
            *o++ = static_cast<std::byte>(u & 0xFF);
            *o++ = static_cast<std::byte>(u >> 8);
        }
    o[0] = static_cast<std::byte>(p.bias & 0xFF);
    o[1] = static_cast<std::byte>(p.bias >> 8);
}

std::optional<LpcParams> readLpcParams(std::span<const std::byte> in)
{
    if (in.size() < kLpcFixedParamBytes ||
        static_cast<unsigned char>(in[0]) != kLpcStencilBox2)
        return std::nullopt;
    LpcParams p;
    for (int i = 0; i < 8; ++i)
        p.mask |= static_cast<std::uint64_t>(in[1 + i]) << (8 * i);
    if ((p.mask >> kLpcTaps) != 0 || in.size() < lpcParamBytes(p.mask))
        return std::nullopt;
    const std::byte* o = in.data() + 9;
    auto u16 = [](const std::byte* b) {
        return static_cast<std::uint16_t>(static_cast<unsigned>(b[0]) |
                                          (static_cast<unsigned>(b[1]) << 8));
    };
    for (int t = 0; t < kLpcTaps; ++t)
        if ((p.mask >> t) & 1) {
            p.w[t] = static_cast<std::int16_t>(u16(o));
            o += 2;
        }
    p.bias = u16(o);
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

// LPC payload for a quantized uint8 chunk, or nothing if the fit failed.
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
        kHeaderSize + lpcParamBytes(params.mask));
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
            !ransDecompressFrame(input.subspan(kHeaderSize + lpcParamBytes(params->mask)),
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
