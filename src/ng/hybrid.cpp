#include "hybrid.h"
#include "cabac.hpp"
#include <vc_delta3d/codec.hpp>
#include <cstring>
#include <new>
#include <vector>

extern "C" int volcodec_ng_hybrid_encode(const uint8_t* src, void* dst,
    size_t cap, size_t* written, int policy, int* winner) {
    if (!src || !dst || !written || policy < 0 || policy > 3) return -1;
    *written = 0;
    try {
        std::vector<std::byte> best(volcodec_ng_lib_encode_bound());
        size_t n = 0;
        int rc = volcodec_ng_lib_encode(src, best.data(), best.size(), &n);
        if (rc) return rc;
        best.resize(n);
        int selected = 0;
        if (policy) {
            // Upstream LPC already computes the exact minimum of LPC and Delta;
            // do not encode Delta twice when both alternatives are requested.
            auto other = vc_delta3d::compress(
                {reinterpret_cast<const std::byte*>(src), 128*128*128},
                {128,128,128}, 1, 1,
                policy == 2 ? vc_delta3d::Predictor::Lpc : vc_delta3d::Predictor::Delta);
            if (other.size() < best.size()) {
                selected = vc_delta3d::predictor(other) == vc_delta3d::Predictor::Lpc ? 2 : 1;
                best = std::move(other);
            }
        }
        if (policy==3) {
            auto candidates=ng_cabac_candidates(src,false);
            for(size_t i=0;i<candidates.size();++i) if(candidates[i].size()<best.size()) {
                selected=3+int(i);
                best.assign(reinterpret_cast<const std::byte*>(candidates[i].data()),
                            reinterpret_cast<const std::byte*>(candidates[i].data()+candidates[i].size()));
            }
        }
        *written = best.size();
        if (winner) *winner = selected;
        if (cap < best.size()) return -2;
        std::memcpy(dst, best.data(), best.size());
        return 0;
    } catch (const std::bad_alloc&) { return -4; }
      catch (...) { return -3; }
}

extern "C" int volcodec_ng_hybrid_decode(const void* data, size_t n,
    uint8_t* dst, size_t cap) {
    if (!data || !dst || cap < 128*128*128) return -1;
    if (n < 4) return -3;
    if (std::memcmp(data,"NCAB",4)==0)
        return ng_cabac_decode({static_cast<const uint8_t*>(data),n},dst)?0:-3;
    if (std::memcmp(data,"VCNG",4)==0)
        return volcodec_ng_lib_decode(data,n,dst,cap);
    if (std::memcmp(data,"D3D1",4)!=0) return -3;
    try {
        return vc_delta3d::decompressInto(
            {reinterpret_cast<const std::byte*>(data),n},
            {reinterpret_cast<std::byte*>(dst),128*128*128}) ? 0 : -3;
    } catch (...) { return -3; }
}
