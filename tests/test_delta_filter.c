#include "../src/delta_filter.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Independent corner definition, not another copy of the pass implementation. */
static unsigned expected(const uint16_t* src, size_t z, size_t y, size_t x,
                         size_t ny, size_t nx, unsigned mask)
{
    int value = 0;
    for (unsigned subset = 0; subset < 8; ++subset) {
        if (subset & ~mask) continue;
        const unsigned dx = subset & 1, dy = (subset >> 1) & 1, dz = subset >> 2;
        if (x < dx || y < dy || z < dz) continue;
        const unsigned parity = (dx + dy + dz) & 1;
        const unsigned v = src[((z-dz)*ny + y-dy)*nx + x-dx];
        value += parity ? -(int)v : (int)v;
    }
    return (unsigned)value;
}

int main(void)
{
    const size_t shapes[][3] = {{1,1,1}, {1,2,17}, {3,5,31}, {4,7,32},
                                {9,11,33}, {7,19,128}};
    for (size_t k = 0; k < sizeof(shapes)/sizeof(shapes[0]); ++k) {
        const size_t nz=shapes[k][0], ny=shapes[k][1], nx=shapes[k][2], n=nz*ny*nx;
        uint8_t* u8 = malloc(n);
        uint16_t* u16 = malloc(n*sizeof(*u16));
        uint16_t* original = malloc(n*sizeof(*original));
        assert(u8 && u16 && original);
        for (size_t i=0; i<n; ++i) original[i]=(uint16_t)(i*113 + i/17 + 255);
        for (unsigned mask=0; mask<8; ++mask) {
            for (size_t i=0; i<n; ++i) {u8[i]=(uint8_t)original[i]; u16[i]=original[i];}
            vc_delta_filter_u8(u8,nz,ny,nx,mask);
            vc_delta_filter_u16(u16,nz,ny,nx,mask);
            for (size_t z=0; z<nz; ++z) for (size_t y=0; y<ny; ++y) for (size_t x=0; x<nx; ++x) {
                const size_t i=(z*ny+y)*nx+x;
                const unsigned v=expected(original,z,y,x,ny,nx,mask);
                assert(u8[i]==(uint8_t)v && u16[i]==(uint16_t)v);
            }
            vc_delta_unfilter_u8(u8,nz,ny,nx,mask);
            vc_delta_unfilter_u16(u16,nz,ny,nx,mask);
            for (size_t i=0; i<n; ++i) assert(u8[i]==(uint8_t)original[i] && u16[i]==original[i]);
        }
        free(original); free(u16); free(u8);
    }
    uint16_t* u16 = malloc(65536*sizeof(*u16));
    uint8_t u8[256];
    assert(u16);
    for (int width=1; width<=255; ++width) {
        for (unsigned i=0; i<256; ++i) u8[i]=(uint8_t)i;
        for (unsigned i=0; i<65536; ++i) u16[i]=(uint16_t)i;
        vc_delta_quantize_u8(u8,256,width);
        vc_delta_quantize_u16(u16,65536,width);
        for (unsigned i=0; i<65536; ++i) {
            const unsigned q=(i+width/2)/width*width;
            assert(u16[i]==(q>65535?65535:q));
            if (i<256) assert(u8[i]==(q>255?255:q));
        }
    }
    free(u16);
    puts("C delta filters: all masks, SIMD tails, both element widths, all quantizers passed");
    return 0;
}
