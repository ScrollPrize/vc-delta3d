#include "buffer.h"
/* NGC1 range-access header and decode. Parsing needs only the header; each
 * leaf is decoded independently. Iterative tree traversal avoids stack overflow
 * on maliciously deep partitions. No payload is copied during index parsing. */
#include <vc_delta3d/indexed.h>
#include <stdlib.h>
#include <string.h>

int vc_ng_preset_decode(const unsigned char*,size_t,unsigned char*,size_t);
unsigned vc_ng_frame_dimension(const unsigned char*,size_t);
int vc_ng_rect_decode(const unsigned char*,size_t,unsigned,unsigned,unsigned,unsigned char*,size_t);
int vc_ng_preset_encode(const unsigned char*,unsigned,unsigned,unsigned char*,size_t,size_t*);
int vc_ng_rect_encode(const unsigned char*,unsigned,unsigned,unsigned,unsigned char*,size_t,size_t*);
typedef struct { uint32_t origin[3],shape[3],channel,channels; } domain;
static uint64_t read_le(const uint8_t *p,unsigned n) {
    uint64_t value=0; for(unsigned i=0;i<n;++i) value|=(uint64_t)p[i]<<(8*i); return value;
}
static void write_le(uint8_t *p,uint64_t value,unsigned n) {
    for(unsigned i=0;i<n;++i) p[i]=(uint8_t)(value>>(8*i));
}
static int voxels(const uint32_t s[3],uint64_t *n) {
    uint64_t v=1;
    for(unsigned a=0;a<3;++a) {
        if(!s[a] || s[a]%32 || v>UINT64_MAX/s[a]) return 0;
        v*=s[a];
    }
    *n=v; return 1;
}
static int leaf_shape(const uint32_t s[3]) {
    uint64_t n; return voxels(s,&n) && s[2]<=1024 && (uint64_t)s[0]*s[1]<=65536;
}
static int cube(const uint32_t s[3]) { return s[0]==s[1] && s[1]==s[2]; }
static int full(const vc_indexed_leaf *l,const uint32_t s[3]) {
    return !l->origin[0]&&!l->origin[1]&&!l->origin[2]&&!memcmp(l->shape,s,3*sizeof(*s));
}
int vc_indexed_encode(const uint8_t *input,size_t input_size,const uint32_t shape[3],uint32_t channels,
                      unsigned layout,unsigned preset,const vc_partition_node *plan,size_t nodes,
                      uint8_t **output,size_t *output_size) {
    if(!output || !output_size) return 0;
    *output=NULL; *output_size=0;
    uint64_t field;
    if(!input || !shape || !channels || channels>INT32_MAX || layout>1 || preset<3 || preset>6 ||
       (nodes&&!plan) || !voxels(shape,&field) || field>SIZE_MAX/channels || field*channels!=input_size) return 0;
    uint8_t *out=NULL,*scratch=NULL; size_t capacity=0,scratch_cap=0,used=24;
    domain *stack=NULL; size_t stack_cap=0,pending=0,step=0;
    vc_indexed_leaf *leaves=NULL; size_t leaf_cap=0,leaf_count=0;
    if(!vc_reserve(&stack,&stack_cap,1,sizeof(*stack)) || !vc_reserve(&out,&capacity,24,1)) goto fail;
    domain root={0}; memcpy(root.shape,shape,sizeof(root.shape)); root.channels=channels; stack[pending++]=root;
    while(pending) {
        domain d=stack[--pending]; vc_partition_node node={4,0,0};
        if(nodes) {
            if(step>=nodes) goto fail;
            node=plan[step++];
        } else if(d.channels>1) { node.axis=3; node.cut=d.channels/2; }
        else if(!cube(d.shape) || !leaf_shape(d.shape)) {
            unsigned axis=0;
            for(unsigned a=1;a<3;++a) if(d.shape[a]>d.shape[axis]) axis=a;
            node.axis=axis; node.cut=(d.shape[axis]/64)*32;
        }
        if(node.axis<4) {
            if(node.rectangular || !node.cut) goto fail;
            domain right=d; unsigned axis=node.axis; uint32_t cut=node.cut;
            if(axis==3) {
                if(cut>=d.channels) goto fail;
                d.channels=cut; right.channel+=cut; right.channels-=cut;
            } else {
                if(cut%32 || cut>=d.shape[axis]) goto fail;
                d.shape[axis]=cut; right.origin[axis]+=cut; right.shape[axis]-=cut;
            }
            if(used>UINT32_MAX-5 || !vc_reserve(&out,&capacity,used+5,1) ||
               !vc_reserve(&stack,&stack_cap,pending+2,sizeof(*stack))) goto fail;
            out[used++]=(uint8_t)axis; write_le(out+used,cut,4); used+=4;
            stack[pending++]=right; stack[pending++]=d; continue;
        }
        if(node.axis!=4 || node.cut || d.channels!=1 || !leaf_shape(d.shape) || (!node.rectangular&&!cube(d.shape))) goto fail;
        if(used>UINT32_MAX-17 || !vc_reserve(&out,&capacity,used+17,1) ||
           !vc_reserve(&leaves,&leaf_cap,leaf_count+1,sizeof(*leaves))) goto fail;
        vc_indexed_leaf *l=&leaves[leaf_count++];
        l->channel=d.channel; memcpy(l->origin,d.origin,sizeof(l->origin)); memcpy(l->shape,d.shape,sizeof(l->shape));
        l->codec=node.rectangular?VC_LEAF_RECTANGULAR:VC_LEAF_CHOICE;
        out[used++]=(uint8_t)l->codec;
        l->offset=used; /* Temporarily the index record address, not payload offset. */
        memset(out+used,0,16); used+=16;
    }
    if(nodes && step!=nodes) goto fail;
    memcpy(out,"NGC1",4); write_le(out+4,used,4);
    for(unsigned a=0;a<3;++a) write_le(out+8+4*a,shape[a],4);
    write_le(out+20,channels|(layout?UINT32_C(0x80000000):0),4);
    for(size_t i=0;i<leaf_count;++i) {
        const vc_indexed_leaf *l=&leaves[i]; uint64_t vox; if(!voxels(l->shape,&vox) || vox>SIZE_MAX) goto fail;
        size_t n=(size_t)vox; const uint8_t *src;
        if(!layout && full(l,shape)) src=input+(size_t)l->channel*n;
        else {
            if(!vc_reserve(&scratch,&scratch_cap,n,1)) goto fail;
            size_t p=0;
            for(uint32_t z=0;z<l->shape[0];++z) for(uint32_t y=0;y<l->shape[1];++y) {
                size_t q=((size_t)(l->origin[0]+z)*shape[1]+l->origin[1]+y)*shape[2]+l->origin[2];
                if(!layout) memcpy(scratch+p,input+q+(size_t)l->channel*(size_t)field,l->shape[2]);
                else for(uint32_t x=0;x<l->shape[2];++x) scratch[p+x]=input[(q+x)*channels+l->channel];
                p+=l->shape[2];
            }
            src=scratch;
        }
        if(used>SIZE_MAX-131072 || n>(SIZE_MAX-used-131072)/4 || !vc_reserve(&out,&capacity,used+4*n+131072,1)) goto fail;
        size_t bytes=0; int rc;
        if(l->codec==VC_LEAF_RECTANGULAR) rc=vc_ng_rect_encode(src,l->shape[2],l->shape[1],l->shape[0],out+used,capacity-used,&bytes);
        else rc=vc_ng_preset_encode(src,l->shape[0],preset-3,out+used,capacity-used,&bytes);
        if(rc || bytes>capacity-used) goto fail;
        write_le(out+l->offset,used,8); write_le(out+l->offset+8,bytes,8); used+=bytes;
    }
    free(scratch); free(stack); free(leaves); *output=out; *output_size=used; return 1;
fail:
    free(out); free(scratch); free(stack); free(leaves); return 0;
}
uint32_t vc_indexed_header_size(const uint8_t *p,size_t n) {
    if(!p || n<8 || memcmp(p,"NGC1",4)) return 0;
    uint32_t size=(uint32_t)read_le(p+4,4); return size>=41?size:0;
}
void vc_index_free(vc_chunk_index *index) {
    if(index) { free(index->leaves); memset(index,0,sizeof(*index)); }
}
int vc_index_read(const uint8_t *data,size_t size,uint64_t object,vc_chunk_index *result) {
    if(!result) return 0;
    memset(result,0,sizeof(*result));
    uint32_t header=vc_indexed_header_size(data,size);
    if(!header || header>size || header>object) return 0;
    vc_chunk_index index={0}; index.header_bytes=header; index.encoded_bytes=object;
    for(unsigned a=0;a<3;++a) index.shape[a]=(uint32_t)read_le(data+8+4*a,4);
    uint32_t c=(uint32_t)read_le(data+20,4);
    index.layout=c>>31; index.channels=c&0x7fffffff;
    if(!index.channels || !voxels(index.shape,&index.decoded_bytes) ||
       index.decoded_bytes>SIZE_MAX/index.channels) return 0;
    index.decoded_bytes*=index.channels;
    /* Every pending split needs five bytes; every leaf record needs seventeen.
     * These input-derived upper bounds are checked before allocating. */
    size_t stack_cap=(header-24)/5+1, leaf_cap=(header-24)/17;
    if(stack_cap>SIZE_MAX/sizeof(domain) || leaf_cap>SIZE_MAX/sizeof(vc_indexed_leaf)) return 0;
    domain *stack=malloc(stack_cap*sizeof(*stack));
    index.leaves=malloc(leaf_cap*sizeof(*index.leaves));
    if(!stack || !index.leaves) { free(stack); vc_index_free(&index); return 0; }
    domain root={0}; memcpy(root.shape,index.shape,sizeof(root.shape)); root.channels=index.channels;
    size_t pending=1,p=24; stack[0]=root; uint64_t next=header;
    while(pending) {
        domain d=stack[--pending];
        if(p>=header) goto fail;
        unsigned tag=data[p++];
        if(tag<4) {
            if(header-p<4 || pending>stack_cap-2) goto fail;
            uint32_t cut=(uint32_t)read_le(data+p,4); p+=4;
            domain right=d;
            if(tag==3) {
                if(!cut || cut>=d.channels) goto fail;
                d.channels=cut; right.channel+=cut; right.channels-=cut;
            } else {
                if(!cut || cut%32 || cut>=d.shape[tag]) goto fail;
                d.shape[tag]=cut; right.origin[tag]+=cut; right.shape[tag]-=cut;
            }
            stack[pending++]=right; stack[pending++]=d; continue;
        }
        if((tag!=VC_LEAF_CHOICE && tag!=VC_LEAF_RECTANGULAR) || d.channels!=1 ||
           !leaf_shape(d.shape) || (tag==VC_LEAF_CHOICE && !cube(d.shape)) ||
           header-p<16 || index.leaf_count>=leaf_cap) goto fail;
        vc_indexed_leaf *l=&index.leaves[index.leaf_count++];
        l->channel=d.channel; memcpy(l->origin,d.origin,sizeof(l->origin)); memcpy(l->shape,d.shape,sizeof(l->shape));
        l->codec=tag; l->offset=read_le(data+p,8); l->length=read_le(data+p+8,8); p+=16;
        if(l->offset!=next || l->length<4 || next>object || l->length>object-next) goto fail;
        next+=l->length;
    }
    if(p!=header || next!=object) goto fail;
    free(stack); *result=index; return 1;
fail:
    free(stack); vc_index_free(&index); return 0;
}
int vc_indexed_leaf_decode(const uint8_t *src,size_t size,const vc_indexed_leaf *l,uint8_t *out,size_t count) {
    uint64_t n;
    if(!src || !out || !l || size<4 || size!=l->length || !leaf_shape(l->shape) || !voxels(l->shape,&n) || n!=count) return 0;
    if(l->codec==VC_LEAF_RECTANGULAR) {
        if(memcmp(src,"NDB2",4)) return 0;
        return vc_ng_rect_decode(src,size,l->shape[2],l->shape[1],l->shape[0],out,count)==0;
    }
    if(l->codec!=VC_LEAF_CHOICE || memcmp(src,"NCF1",4) || !cube(l->shape) ||
       vc_ng_frame_dimension(src,size)!=l->shape[0]) return 0;
    return vc_ng_preset_decode(src,size,out,count)==0;
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
