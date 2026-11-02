#include <vc_delta3d/codec.h>
#include <vc_delta3d/indexed.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
int main(void) {
    size_t n=32u*32*32;
    uint8_t *src=malloc(2*n),*decoded=malloc(2*n); assert(src&&decoded);
    for(size_t i=0;i<2*n;++i) src[i]=(uint8_t)((i/32+i/1024)*3);
    for(unsigned p=3;p<=6;++p) for(unsigned channels=0;channels<=2;channels+=2) {
        vc_encode_options opt=vc_default_options(); opt.preset=(vc_preset)p;
        opt.shape[0]=opt.shape[1]=opt.shape[2]=32; opt.channels=channels;opt.layout=1;
        size_t count=channels?2*n:n,bytes;uint8_t *frame;
        assert(vc_compress(src,count,&opt,&frame,&bytes));
        vc_frame_info info;assert(vc_frame_inspect(frame,bytes,&info));
        assert(info.decoded_bytes==count && info.channels==(channels?channels:1));
        assert(vc_decompress(frame,bytes,decoded,count));assert(!memcmp(src,decoded,count));
        if(channels) {
            vc_chunk_index index;
            uint32_t hn=vc_indexed_header_size(frame,8);assert(hn);
            assert(vc_index_read(frame,hn,bytes,&index));assert(index.leaf_count==2);
            for(size_t k=0;k<index.leaf_count;++k) {
                vc_indexed_leaf *leaf=&index.leaves[k];
                assert(vc_indexed_leaf_decode(frame+leaf->offset,(size_t)leaf->length,leaf,decoded,n));
                for(size_t i=0;i<n;++i) assert(decoded[i]==src[2*i+leaf->channel]);
            }
            vc_index_free(&index);
        }
        vc_free(frame);
    }
    vc_encode_options opt=vc_default_options();opt.shape[0]=opt.shape[1]=opt.shape[2]=32;
    uint8_t *frame;size_t bytes;opt.preset=(vc_preset)2;
    assert(!vc_compress(src,n,&opt,&frame,&bytes));opt.preset=(vc_preset)7;
    assert(!vc_compress(src,n,&opt,&frame,&bytes));
    free(src);free(decoded);puts("public C API and independent leaf decode passed");return 0;
}
