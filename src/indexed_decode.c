#include "indexed_internal.h"
uint32_t vc_indexed_header_size(const uint8_t*p,size_t n) {
    if(!p||n<6)return 0;
    int wide=!memcmp(p,"NGX2",4);
    if((!wide&&memcmp(p,"NGC2",4))||(wide&&n<8))return 0;
    uint32_t size=(uint32_t)read_le(p+4,wide?4:2);return size>=(wide?34u:16u)?size:0;
}
void vc_index_free(vc_chunk_index*index) {
    if(index){free(index->leaves);memset(index,0,sizeof(*index));}
}
int vc_index_read(const uint8_t*data,size_t size,uint64_t object,vc_chunk_index*result) {
    if(!result)return 0;memset(result,0,sizeof(*result));
    uint32_t header=vc_indexed_header_size(data,size);if(!header||header>size||header>object)return 0;
    vc_chunk_index index={0};index.header_bytes=header;index.encoded_bytes=object;
    unsigned wide=!memcmp(data,"NGX2",4),width=wide?4:1;
    size_t p=wide?8:6;if(data[p]>1)return 0;index.layout=data[p++];uint32_t v;
    for(unsigned a=0;a<3;a++){
        if(header-p<width)return 0;v=(uint32_t)read_le(data+p,width);p+=width;
        if(v>=UINT32_MAX/32)return 0;index.shape[a]=(v+1)*32;
    }
    if(header-p<width)return 0;v=(uint32_t)read_le(data+p,width);p+=width;
    if(v>=INT32_MAX)return 0;index.channels=v+1;
    if(!voxels(index.shape,&index.decoded_bytes)||index.decoded_bytes>SIZE_MAX/index.channels)return 0;
    index.decoded_bytes*=index.channels;
    node*t=NULL;size_t cap=0,nt=0,needed=1;
    while(needed){
        if(p>=header||!vc_reserve(&t,&cap,nt+1,sizeof(*t)))goto fail;
        unsigned tag=data[p++];if(tag>8)goto fail;uint32_t value=0;
        if(tag<8){
            unsigned width=wide?4:1;if(header-p<width)goto fail;
            uint64_t x=read_le(data+p,width)+1;p+=width;if(tag<3)x*=32;
            if(x>UINT32_MAX)goto fail;value=(uint32_t)x;
        }
        t[nt++]=(node){tag,VC_LEAF_CHOICE,value,0};needed--;
        if(tag<4)needed+=2;else if(tag<8)needed++;
        if(needed>header-p)goto fail;
    }
    if(!finish_tree(t,nt))goto fail;
    width=wide?8:4;if((header-p)%width)goto fail;
    size_t nl=(header-p)/width;
    domain root={0};memcpy(root.shape,index.shape,sizeof(root.shape));root.channels=index.channels;
    if(!expand(t,root,nl,&index.leaves,&index.leaf_count)||index.leaf_count!=nl)goto fail;
    uint64_t next=header;
    for(size_t i=0;i<nl;i++){
        vc_indexed_leaf*l=index.leaves+i;l->offset=next;l->length=read_le(data+p,width);p+=width;
        if(l->length<4||next>object||l->length>object-next)goto fail;next+=l->length;
    }
    if(next!=object)goto fail;free(t);*result=index;return 1;
fail:free(t);vc_index_free(&index);return 0;
}
int vc_indexed_leaf_decode(const uint8_t*src,size_t size,const vc_indexed_leaf*l,uint8_t*out,size_t count) {
    uint64_t n;
    if(!src||!out||!l||size<4||size!=l->length||l->codec!=VC_LEAF_CHOICE||
       !leaf_shape(l->shape)||!voxels(l->shape,&n)||n!=count)return 0;
    return vc_ng_leaf_decode(src,size,l->shape,out,count)==0;
}
int vc_indexed_decode(const uint8_t *src,size_t size,uint8_t *out,size_t count) {
    vc_chunk_index index;
    if(!out || !vc_index_read(src,size,size,&index)) return 0;
    if(index.decoded_bytes!=count) { vc_index_free(&index); return 0; }
    uint8_t *scratch=NULL; size_t capacity=0;
    for(size_t i=0;i<index.leaf_count;++i) {
        const vc_indexed_leaf *l=&index.leaves[i]; uint64_t n; voxels(l->shape,&n);
        if(!index.layout && full(l,index.shape)) {
            if(!vc_indexed_leaf_decode(src+l->offset,(size_t)l->length,l,out+(size_t)l->channel*(size_t)n,(size_t)n)) goto fail;
        } else {
            if(n>capacity) {
                uint8_t *grown=realloc(scratch,(size_t)n); if(!grown) goto fail;
                scratch=grown; capacity=(size_t)n;
            }
            if(!vc_indexed_leaf_decode(src+l->offset,(size_t)l->length,l,scratch,(size_t)n)) goto fail;
            size_t p=0,field=(size_t)(index.decoded_bytes/index.channels);
            for(uint32_t z=0;z<l->shape[0];++z) for(uint32_t y=0;y<l->shape[1];++y) {
                size_t q=((size_t)(l->origin[0]+z)*index.shape[1]+l->origin[1]+y)*index.shape[2]+l->origin[2];
                if(!index.layout) memcpy(out+q+(size_t)l->channel*field,scratch+p,l->shape[2]);
                else for(uint32_t x=0;x<l->shape[2];++x) out[(q+x)*index.channels+l->channel]=scratch[p+x];
                p+=l->shape[2];
            }
        }
    }
    free(scratch); vc_index_free(&index); return 1;
fail:
    free(scratch); vc_index_free(&index); return 0;
}
