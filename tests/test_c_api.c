#include <vc_delta3d/codec.h>
#include <vc_delta3d/indexed.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
static size_t varint(uint8_t*p,uint64_t v){size_t n=0;do{p[n++]=(uint8_t)((v&127)|(v>127?128:0));v>>=7;}while(v);return n;}
int ng_set_isa_limit(int);

/* Every preset must write the same bytes with the vector kernels and with the
 * plain C kernels, and each must decode the other's frames. */
static void kernels_agree(void) {
    enum { D=64 };
    const size_t n=(size_t)D*D*D;
    uint8_t *src=malloc(n),*decoded=malloc(n);
    uint32_t state=12345;
    for(unsigned z=0;z<D;++z)for(unsigned y=0;y<D;++y)for(unsigned x=0;x<D;++x){
        state=state*1664525u+1013904223u;
        /* A smooth ramp with noise, a constant slab and an empty corner. */
        unsigned v=2*x+y+3*z/2+(state>>29);
        if(z<8)v=200;
        if(x>=40&&y>=40)v=0;
        src[((size_t)z*D+y)*D+x]=(uint8_t)v;
    }
    vc_encode_options opt=vc_default_options();
    opt.shape[0]=opt.shape[1]=opt.shape[2]=D;
    const vc_preset presets[]={VC_FASTER,VC_FAST,VC_BALANCED,VC_SMALL,VC_SLOWER,VC_MEDIUM,VC_SLOW};
    for(unsigned p=0;p<sizeof(presets)/sizeof(*presets);++p){
        uint8_t *frame[2];size_t bytes[2];
        opt.preset=presets[p];
        for(unsigned k=0;k<2;++k){
            assert(!ng_set_isa_limit(k?128:0));
            assert(vc_compress(src,n,&opt,&frame[k],&bytes[k]));
        }
        assert(bytes[0]==bytes[1]&&!memcmp(frame[0],frame[1],bytes[0]));
        for(unsigned k=0;k<3;++k){
            assert(!ng_set_isa_limit(k==0?0:k==1?256:128));
            memset(decoded,0x5A,n);
            assert(vc_decompress(frame[0],bytes[0],decoded,n)&&!memcmp(src,decoded,n));
        }
        vc_free(frame[0]);vc_free(frame[1]);
    }
    assert(!ng_set_isa_limit(0));
    free(src);free(decoded);
}

int main(void) {
    kernels_agree();
    size_t n=32u*32*32;
    uint8_t *src=malloc(4*n),*decoded=malloc(4*n); assert(src&&decoded);
    /* Empty and nonempty block regions force nonuniform selectors as well as
     * exercising zero-predictor handling with all presets/channel layouts. */
    for(size_t i=0;i<4*n;++i) src[i]=(i%(n)<n/2)?0:(uint8_t)((i/32+i/1024)*3);
    for(unsigned p=3;p<=6;++p) for(unsigned channels=0;channels<=2;channels+=2) {
        vc_encode_options opt=vc_default_options(); opt.preset=(vc_preset)p;
        opt.shape[0]=opt.shape[1]=opt.shape[2]=32; opt.channels=channels;opt.layout=1;
        size_t count=channels?2*n:n,bytes;uint8_t *frame;
        assert(vc_compress(src,count,&opt,&frame,&bytes));
        vc_frame_info info;assert(vc_frame_inspect(frame,bytes,&info));
        assert(info.decoded_bytes==count && info.channels==(channels?channels:1));
        assert(vc_decompress(frame,bytes,decoded,count));assert(!memcmp(src,decoded,count));
        /* Exercise malformed/truncated input with the same public C path,
         * including under ASan/UBSan. Never expose a partial success. */
        assert(!vc_decompress(frame,bytes,decoded,count-1));
        assert(!vc_decompress(frame,bytes,decoded,count+1));
        for(size_t cut=0;cut<32 && cut<bytes;++cut)
            assert(!vc_decompress(frame,cut,decoded,count));
        assert(!vc_decompress(frame,bytes-1,decoded,count));
        uint8_t magic=frame[0];frame[0]^=0x80;
        assert(!vc_decompress(frame,bytes,decoded,count));frame[0]=magic;
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
    /* Noncubic chunks use cubic leaves with the requested preset. */
    const uint32_t shapes[3][3]={{32,32,64},{32,64,32},{64,32,32}};
    vc_partition_node leaf={4,0};
    for(unsigned k=0;k<3;k++){
        assert(!vc_indexed_encode(src,2*n,shapes[k],1,0,VC_FAST,&leaf,1,&frame,&bytes));
        assert(vc_indexed_encode(src,2*n,shapes[k],1,0,VC_FAST,NULL,0,&frame,&bytes));
        assert(vc_decompress(frame,bytes,decoded,2*n)&&!memcmp(src,decoded,2*n));
        vc_chunk_index idx;assert(vc_index_read(frame,bytes,bytes,&idx)&&idx.leaf_count==2);
        vc_indexed_leaf*l=idx.leaves;
        assert(vc_indexed_leaf_decode(frame+l->offset,(size_t)l->length,l,decoded,n));
        assert(!vc_decompress(frame+l->offset,(size_t)l->length,decoded,2*n));
        vc_index_free(&idx);
        vc_free(frame);
    }
    /* Repeated channel + spatial templates, bounded hostile parsing. */
    vc_partition_node repeat[]={{8,2},{7,2},{4,0}};
    uint32_t rs[3]={32,32,64};
    assert(vc_indexed_encode(src,4*n,rs,2,0,VC_FAST,repeat,3,&frame,&bytes));
    assert(vc_decompress(frame,bytes,decoded,4*n)&&!memcmp(src,decoded,4*n));
    uint32_t hn=vc_indexed_header_size(frame,bytes);
    {   /* prefix 8, flags 1, dims 3, channels 1, repeat 2, repeat 2, leaf 1, varint lengths */
        vc_chunk_index idx;assert(vc_index_read(frame,bytes,bytes,&idx)&&idx.leaf_count==4);
        size_t expected=18;uint8_t tmp[10];
        for(size_t i=0;i<4;i++)expected+=varint(tmp,idx.leaves[i].length);
        assert(hn==expected);vc_index_free(&idx);
    }
    for(size_t cut=0;cut<hn;cut++){vc_chunk_index idx;assert(!vc_index_read(frame,cut,bytes,&idx));}
    for(size_t i=0;i<hn;i++)for(unsigned b=0;b<8;b++){
        vc_chunk_index idx;frame[i]^=(uint8_t)(1u<<b);
        if(vc_index_read(frame,hn,bytes,&idx)){assert(idx.leaf_count);vc_index_free(&idx);}
        frame[i]^=(uint8_t)(1u<<b);
    }
    vc_free(frame);
    /* 64-bit lengths and implicit offsets without large allocation. */
    uint8_t ext[32]={'N','G','I','1',0,0,0,0, 0, 0,0,0, 1, 7,1, 8};size_t en=16;
    uint64_t long_leaf=(UINT64_C(1)<<32)+11;
    en+=varint(ext+en,long_leaf);en+=varint(ext+en,4);ext[4]=(uint8_t)en;
    vc_chunk_index large;
    assert(vc_index_read(ext,en,en+long_leaf+4,&large));
    assert(large.leaf_count==2&&large.leaves[1].offset==en+long_leaf);vc_index_free(&large);
    ext[en-1]=5;assert(!vc_index_read(ext,en,en+long_leaf+4,&large));
    memset(src,0,n);
    for(unsigned p=VC_FAST;p<=VC_SMALL;++p){
        opt.preset=(vc_preset)p;
        if(!vc_preset_valid(p)){assert(!vc_compress(src,n,&opt,&frame,&bytes));continue;}
        assert(vc_compress(src,n,&opt,&frame,&bytes));
        assert(vc_decompress(frame,bytes,decoded,n)&&!memcmp(src,decoded,n));
        assert(!vc_decompress(frame,bytes,decoded,n+1));
        vc_free(frame);
    }
    free(src);free(decoded);puts("public C API and independent leaf decode passed");return 0;
}
