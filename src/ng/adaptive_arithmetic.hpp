#pragma once
// Extracted unchanged arithmetic engine from fiber-comp eb7f68b65:
// volume-cartographer/python/vc/compression/vcz2_adaptive_arithmetic.inc.
#include <cstdint>
#include <span>
#include <vector>
#include <stdexcept>
#include <utility>
#include <algorithm>
#include <array>
namespace ng_cabac {
#if defined(__GNUC__) || defined(__clang__)
#define NG_ARITH_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define NG_ARITH_INLINE __forceinline
#else
#define NG_ARITH_INLINE inline
#endif
inline std::uint64_t binary_quotient(std::uint64_t numerator,std::uint32_t total) {
#if defined(__SIZEOF_INT128__)
    // numerator <= 2^32*32768. With a ceiling reciprocal at precision48,
    // the approximate quotient is exact or one too high; correct it exactly.
    static const auto inverse=[](){std::array<std::uint64_t,32769> t{};
        for(unsigned d=1;d<t.size();d++)t[d]=((std::uint64_t(1)<<48)+d-1)/d;
        return t;}();
    auto q=std::uint64_t((__uint128_t(numerator)*inverse[total])>>48);
    return q-(q*total>numerator);
#else
    return numerator/total;
#endif
}
struct AuBits {
    std::vector<std::uint8_t> bytes;
    std::uint32_t logical = 0;
};

class AuBitWriter {
public:
    void write(unsigned bit)
    {
        if ((bits_ & 7u) == 0) bytes_.push_back(0);
        bytes_.back() |= std::uint8_t((bit & 1u) << (bits_ & 7u));
        ++bits_;
    }
    AuBits finish() && { return {std::move(bytes_), bits_}; }
private:
    std::vector<std::uint8_t> bytes_;
    std::uint32_t bits_ = 0;
};

class AuEncoder {
public:
    void encode(std::uint32_t cumulative, std::uint32_t frequency, std::uint32_t total)
    {
        if (!frequency || total > 32768 || cumulative + frequency > total) throw std::runtime_error("VC6U invalid arithmetic interval");
        const std::uint64_t range = std::uint64_t(high_) - low_ + 1;
        const auto oldLow = low_;
        high_ = std::uint32_t(std::uint64_t(oldLow) + (range * (cumulative + frequency)) / total - 1);
        low_ = std::uint32_t(std::uint64_t(oldLow) + (range * cumulative) / total);
        renormalize();
    }
    AuBits finish()
    {
        ++pending_;
        emitFollow(low_ < kQ1 ? 0 : 1);
        return std::move(writer_).finish();
    }
private:
    static constexpr std::uint32_t kQ1 = 0x40000000u, kHalf = 0x80000000u, kQ3 = 0xc0000000u;
    std::uint32_t low_ = 0, high_ = 0xffffffffu;
    std::uint64_t pending_ = 0;
    AuBitWriter writer_;

    void emitFollow(unsigned bit)
    {
        writer_.write(bit);
        while (pending_) { writer_.write(bit ^ 1u); --pending_; }
    }
    void renormalize()
    {
        for (;;) {
            if (high_ < kHalf) emitFollow(0);
            else if (low_ >= kHalf) { emitFollow(1); low_ -= kHalf; high_ -= kHalf; }
            else if (low_ >= kQ1 && high_ < kQ3) { ++pending_; low_ -= kQ1; high_ -= kQ1; }
            else break;
            low_ <<= 1;
            high_ = (high_ << 1) | 1u;
        }
    }
};

class AuDecoder {
public:
    AuDecoder(std::span<const std::uint8_t> bytes, std::uint32_t logical) : bytes_(bytes), logical_(logical)
    {
        if (bytes.size() != (std::uint64_t(logical) + 7) / 8) throw std::runtime_error("VC6U arithmetic extent");
        if (logical % 8 && !bytes.empty() && (bytes.back() & std::uint8_t(0xffu << (logical % 8))))
            throw std::runtime_error("VC6U arithmetic padding");
        for (unsigned i = 0; i < 32; ++i) code_ = (code_ << 1) | readOrZero();
    }
    std::uint32_t target(std::uint32_t total) const
    {
        if (!total || total > 32768) throw std::runtime_error("VC6U invalid arithmetic total");
        const std::uint64_t range = std::uint64_t(high_) - low_ + 1;
        const auto scaled = ((std::uint64_t(code_ - low_) + 1) * total - 1) / range;
        if (scaled >= total) throw std::runtime_error("VC6U arithmetic target");
        return static_cast<std::uint32_t>(scaled);
    }
    void accept(std::uint32_t cumulative, std::uint32_t frequency, std::uint32_t total)
    {
        if (!frequency || total > 32768 || cumulative + frequency > total) throw std::runtime_error("VC6U invalid arithmetic interval");
        const std::uint64_t range = std::uint64_t(high_) - low_ + 1;
        const auto oldLow = low_;
        high_ = std::uint32_t(std::uint64_t(oldLow) + (range * (cumulative + frequency)) / total - 1);
        low_ = std::uint32_t(std::uint64_t(oldLow) + (range * cumulative) / total);
        renormalize();
    }
    std::uint64_t physicalReads() const { return reads_; }
    // Exact specialization of target(total) followed by accept() for two
    // symbols. Same interval and renormalization, one division instead of the
    // generic target division plus both endpoint divisions.
    NG_ARITH_INLINE unsigned binary(std::uint32_t zero,std::uint32_t one)
    {
        const auto total=zero+one;
        if(!zero || !one || total>32768) throw std::runtime_error("VC6U binary interval");
        const std::uint64_t range=std::uint64_t(high_)-low_+1;
        const auto split=std::uint32_t(std::uint64_t(low_)+binary_quotient(range*zero,total));
        const unsigned bit=code_>=split;
        if(bit)low_=split;else high_=split-1;
        renormalize();
        return bit;
    }
private:
    static constexpr std::uint32_t kQ1 = 0x40000000u, kHalf = 0x80000000u, kQ3 = 0xc0000000u;
    std::span<const std::uint8_t> bytes_;
    std::uint32_t logical_ = 0, position_ = 0, low_ = 0, high_ = 0xffffffffu, code_ = 0;
    std::uint64_t reads_ = 0;

    unsigned readOrZero()
    {
        ++reads_;
        if (position_ >= logical_) return 0;
        const auto bit = (bytes_[position_ / 8] >> (position_ & 7u)) & 1u;
        ++position_;
        return bit;
    }
    void renormalize()
    {
        for (;;) {
            if (high_ < kHalf) {}
            else if (low_ >= kHalf) { code_ -= kHalf; low_ -= kHalf; high_ -= kHalf; }
            else if (low_ >= kQ1 && high_ < kQ3) { code_ -= kQ1; low_ -= kQ1; high_ -= kQ1; }
            else break;
            low_ <<= 1;
            high_ = (high_ << 1) | 1u;
            code_ = (code_ << 1) | readOrZero();
        }
    }
};


// Shared medium-adaptation binary model (formerly local to cabac.cpp).
struct Binary {
    uint32_t c[2]={1,1};
    void update(unsigned b) {
        c[b]+=4;
        if(c[0]+c[1]>4096) for(auto& x:c) x=std::max(1u,(x+1)/2);
    }
    NG_ARITH_INLINE unsigned decode(AuDecoder* dec) {
        unsigned bit=dec->binary(c[0],c[1]);update(bit);return bit;
    }
    unsigned code(AuEncoder* enc,AuDecoder* dec,unsigned bit) {
        if(!enc)return decode(dec);
        const unsigned total=c[0]+c[1];
        enc->encode(bit?c[0]:0,c[bit],total);
        update(bit); return bit;
    }
};
}
