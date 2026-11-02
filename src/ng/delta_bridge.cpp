/* Temporary C++ ownership adapter for preset orchestration. */
#include "rans_frame_bridge.hpp"
#include "rans.h"
#include <cstdlib>
#include <memory>
#include <new>
std::vector<std::byte> delta_rans_frame(const uint8_t *data,size_t count) {
    uint8_t *raw; size_t size;
    if (!vc_rans_encode(data,count,0,&raw,&size)) throw std::bad_alloc();
    std::unique_ptr<uint8_t,decltype(&std::free)> owner(raw,&std::free);
    const auto *p=reinterpret_cast<const std::byte*>(raw); return {p,p+size};
}
