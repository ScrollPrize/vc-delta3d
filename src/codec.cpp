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

// Header byte 7: bits 0-3 carry the rANS marker; bit 7 set means bits 4-6
// carry the
// delta-axis mask. Payloads from before per-chunk selection have bits 4-7
// clear and imply full zyx; a clear flag with nonzero mask bits is invalid.
struct HeaderByte {
    unsigned mask;
    bool hasMask;
};

std::optional<HeaderByte> parseHeaderByte(unsigned char b)
{
    const bool hasMask = (b & 0x80) != 0;
    if ((b & 0x0F) != 1 || (!hasMask && (b & 0x70) != 0))
        return std::nullopt;
    HeaderByte out;
    out.hasMask = hasMask;
    out.mask = hasMask ? (b >> 4) & 7u : kDeltaMaskZyx;
    return out;
}

} // namespace

std::vector<std::byte> compress(std::span<const std::byte> input,
                                std::array<int, 3> shapeZYX,
                                std::size_t elemSize,
                                int quantBinWidth)
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
    std::copy(kWireMagic.begin(), kWireMagic.end(), output.begin());
    output[4] = static_cast<std::byte>(kFormatVersion);
    output[5] = static_cast<std::byte>(elemSize);
    output[6] = static_cast<std::byte>(quantBinWidth);
    output[7] = static_cast<std::byte>(0x80u | (mask << 4) | 1u);
    writeU32(output.data() + 8, static_cast<std::uint32_t>(z));
    writeU32(output.data() + 12, static_cast<std::uint32_t>(y));
    writeU32(output.data() + 16, static_cast<std::uint32_t>(x));
    return output;
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
