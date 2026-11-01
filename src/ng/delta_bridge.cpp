// Compile Delta3D exactly once in the same TU as the bridge, sharing its
// private filter/rANS helpers rather than maintaining a second implementation.
#include "codec.cpp"
#include "residual_bridge.h"
#include "rans_frame_bridge.hpp"
#include <chrono>
#include <cmath>
extern "C" size_t delta_rans_size_lower_bound(const uint64_t*hist,size_t n,uint16_t*out){
    if(!n){std::fill(out,out+256,0);return 0;}
    uint32_t freq[256];vc_delta3d::ransNormalize(hist,n,freq);
    double bits=0;
    for(unsigned s=0;s<256;s++){
        out[s]=uint16_t(freq[s]);
        if(hist[s])bits+=double(hist[s])*(12.-std::log2(double(freq[s])));
    }
    // Every lane starts at L and ends in [L,L*2^32). Its final state can
    // absorb at most 32 ideal bits. Each rANS update's negative log error is
    // bounded by -log2(1-M/L) < 1/262144 bits (also after renormalization).
    // Subtract another bit for floating-point summation, and round DOWN to
    // whole 32-bit renormalization words. Thus this can never overestimate.
    double lower=bits-256.-double(n)/262144.-1.;
    return vc_delta3d::kRansStateBytes+(lower>0?size_t(lower/32.)*4:0);
}
extern "C" unsigned delta_prepare_residuals(const uint8_t* src, uint8_t* residuals) {
    const unsigned mask=vc_delta3d::selectDeltaMask(src,NG_Z,NG_Y,NG_X);
    std::memcpy(residuals,src,NG_N);
    vc_delta3d::deltaFilter(residuals,NG_Z,NG_Y,NG_X,mask);
    return mask;
}
extern "C" void delta_reconstruct_residuals(uint8_t* residuals,unsigned mask) {
    vc_delta3d::deltaUnfilter(residuals,NG_Z,NG_Y,NG_X,mask);
}
extern "C" void delta_apply_mask(uint8_t* v,unsigned mask) {
    vc_delta3d::deltaFilter(v,NG_Z,NG_Y,NG_X,mask);
}
std::vector<std::byte> delta_rans_frame(const uint8_t*v,size_t n){
    return vc_delta3d::ransCompressFrame({reinterpret_cast<const std::byte*>(v),n},0);
}
extern "C" int delta_rans_encode(const uint8_t* v,size_t n,uint8_t* dst,size_t cap,size_t* size) {
    try {
        auto b=delta_rans_frame(v,n);
        if(b.size()>cap)return -1;
        std::memcpy(dst,b.data(),b.size());*size=b.size();return 0;
    } catch(...) {return -1;}
}
extern "C" int delta_rans_decode(const uint8_t* v,size_t n,uint8_t* dst,size_t count) {
    try {return vc_delta3d::ransDecompressFrame({reinterpret_cast<const std::byte*>(v),n},reinterpret_cast<std::byte*>(dst),count)?0:-1;}
    catch(...) {return -1;}
}
// Degenerate frequency tables encode no information. Validate the complete
// original eight-state frame, then fill directly without building a slot LUT.
// General alphabets continue through the one shared upstream implementation.
extern "C" int delta_rans_decode_constant(const uint8_t*v,size_t n,uint8_t*dst,size_t count){
    if(n==vc_delta3d::kRansTableBytes+vc_delta3d::kRansStateBytes){
        int symbol=-1;bool valid=true;
        for(int s=0;s<256;s++){unsigned f=v[2*s]|unsigned(v[2*s+1])<<8;if(f){if(f!=vc_delta3d::kRansM||symbol>=0)valid=false;symbol=s;}}
        if(valid&&symbol>=0){
            for(size_t k=0;k<vc_delta3d::kRansLanes;k++){uint64_t state;std::memcpy(&state,v+vc_delta3d::kRansTableBytes+8*k,8);if(state!=vc_delta3d::kRansL)return -1;}
            std::memset(dst,symbol,count);return 0;
        }
    }
    return delta_rans_decode(v,n,dst,count);
}
// Diagnostic only: remove per-table Python/ctypes overhead from the zero-symbol
// setup probe. Reuse the actual shared decoder and its full table validation.
extern "C" uint64_t delta_rans_table_time(const uint16_t*tables,size_t count,int repeats){
    if(!tables||count>128||repeats<1||repeats>1000)return UINT64_MAX;
    std::vector<std::array<uint8_t,vc_delta3d::kRansTableBytes+vc_delta3d::kRansStateBytes>> frames(count);
    for(size_t i=0;i<count;i++){
        for(int s=0;s<256;s++){unsigned f=tables[i*256+s];frames[i][2*s]=f;frames[i][2*s+1]=f>>8;}
        for(size_t k=0;k<vc_delta3d::kRansLanes;k++){uint64_t v=vc_delta3d::kRansL;std::memcpy(frames[i].data()+vc_delta3d::kRansTableBytes+8*k,&v,8);}
    }
    uint8_t out=0;auto start=std::chrono::steady_clock::now();
    for(int r=0;r<repeats;r++)for(auto&f:frames)if(delta_rans_decode(f.data(),f.size(),&out,0))return UINT64_MAX;
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count()/repeats;
}
extern "C" int delta_fixed_trial(const uint8_t* src,unsigned mask,uint64_t* size) {
    if(mask>7)return -1;
    try {
        std::vector<uint8_t> v(src,src+NG_N),decoded(v.size());
        vc_delta3d::deltaFilter(v.data(),NG_Z,NG_Y,NG_X,mask);
        auto frame=vc_delta3d::ransCompressFrame(
            {reinterpret_cast<const std::byte*>(v.data()),v.size()},vc_delta3d::kHeaderSize);
        vc_delta3d::writeHeader(frame,1,1,0x80u|(mask<<4)|1u,NG_Z,NG_Y,NG_X);
        if(!vc_delta3d::decompressInto(frame,{reinterpret_cast<std::byte*>(decoded.data()),decoded.size()}) ||
            std::memcmp(src,decoded.data(),decoded.size()))return -1;
        *size=frame.size();return 0;
    } catch(...) {return -1;}
}
