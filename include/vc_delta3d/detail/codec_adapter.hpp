#pragma once
/* Source-compatible C++ ownership/error adapter over the C API. */
#include <vc_delta3d/codec.h>
#include <memory>
#include <stdexcept>

namespace vc_delta3d {
namespace detail {
inline const uint8_t *bytes(std::span<const std::byte> s) { return reinterpret_cast<const uint8_t*>(s.data()); }
inline std::optional<vc_frame_info> inspect(std::span<const std::byte> s) {
    vc_frame_info info; if (!vc_frame_inspect(bytes(s),s.size(),&info)) return {}; return info;
}
inline std::vector<std::byte> encode(std::span<const std::byte> input,std::array<int,3> shape,
                              size_t elem,int q,Predictor p,int channels,ChannelLayout layout) {
    auto opt=vc_default_options();
    if (channels<0 || elem>2 || q<1 || q>255) throw std::invalid_argument("invalid encoding options");
    for (unsigned a=0;a<3;++a) {
        if (shape[a]<=0) throw std::invalid_argument("invalid shape");
        opt.shape[a]=unsigned(shape[a]);
    }
    opt.element_size=unsigned(elem); opt.quantization=unsigned(q); opt.preset=vc_preset(p);
    opt.channels=unsigned(channels); opt.layout=unsigned(layout);
    uint8_t *raw;size_t size;
    if (!vc_compress(bytes(input),input.size(),&opt,&raw,&size)) throw std::invalid_argument("invalid input or encoding failed");
    std::unique_ptr<uint8_t,decltype(&vc_free)> owner(raw,&vc_free);
    const auto *first=reinterpret_cast<const std::byte*>(raw); return {first,first+size};
}
}
inline std::vector<std::byte> compress(std::span<const std::byte> input,std::array<int,3> shape,size_t elem,int q,Predictor p) {
    return detail::encode(input,shape,elem,q,p,0,ChannelLayout::First);
}
inline std::vector<std::byte> compressChannels(std::span<const std::byte> input,std::array<int,3> shape,int channels,
                                        ChannelLayout layout,size_t elem,int q,Predictor p) {
    if (channels<1) throw std::invalid_argument("invalid channel count");
    return detail::encode(input,shape,elem,q,p,channels,layout);
}
inline std::optional<Predictor> predictor(std::span<const std::byte> s) {
    auto i=detail::inspect(s); if (!i || !i->legacy_scalar) return {}; return Predictor::Delta;
}
inline std::optional<int> channelCount(std::span<const std::byte> s) {
    auto i=detail::inspect(s); if (!i || i->channels>INT32_MAX) return {}; return int(i->channels);
}
inline std::optional<size_t> decodedSize(std::span<const std::byte> s) {
    auto i=detail::inspect(s); if (!i) return {}; return i->decoded_bytes;
}
inline std::optional<int> quantization(std::span<const std::byte> s) {
    auto i=detail::inspect(s); if (!i) return {}; return int(i->quantization);
}
inline std::optional<int> deltaMask(std::span<const std::byte> s) {
    auto i=detail::inspect(s); if (!i || i->delta_mask<0) return {}; return i->delta_mask;
}
inline void quantize(std::span<std::byte> data,size_t elem,int q) {
    if (q<=1) return;
    if (elem>2 || !vc_quantize(data.data(),data.size(),unsigned(elem),unsigned(q))) throw std::invalid_argument("invalid quantization");
}
inline bool decompressIntoWithMagic(std::span<const std::byte> s,std::span<std::byte> out,const WireMagic& magic) {
    return vc_decompress_with_magic(detail::bytes(s),s.size(),reinterpret_cast<const uint8_t*>(magic.data()),out.data(),out.size())!=0;
}
inline bool decompressInto(std::span<const std::byte> s,std::span<std::byte> out) {
    return vc_decompress(detail::bytes(s),s.size(),out.data(),out.size())!=0;
}
inline std::optional<std::vector<std::byte>> decompressWithMagic(std::span<const std::byte> s,size_t count,const WireMagic& magic) {
    std::vector<std::byte> out(count); if (!decompressIntoWithMagic(s,out,magic)) return {}; return out;
}
inline std::optional<std::vector<std::byte>> decompress(std::span<const std::byte> s,size_t count) {
    return decompressWithMagic(s,count,kWireMagic);
}
}
