/* Frozen-frame import ONLY in the benchmark, never in the production decoder.
 * Compare like-for-like headers with the same new selector payload on both
 * sides; also report the complete original frame size. */
#include "buffer.h"
#include "ng/header.h"
#include "ng/selector.h"
#include "ng/selector_bits.h"
#include "ng/ng_stream.h"
#include "ng/residual_bridge.h"
#include "ng/ng_frame.h"
#include <stdio.h>
#include <zstd.h>
#ifdef HEADER_STATS
/* Benchmark-only schema access; production codec is unchanged. */
#define reader header_reader
#define take header_take
#define get header_get
#include "ng/header.c"
#undef reader
#undef take
#undef get
static size_t header_table_bytes(const uint8_t *p,size_t n){
    arrays a={0};walk w={{p,n,0,1},{0},&a,1,0};size_t bytes=0;
    if(!schema(&w,1))abort();
    for(unsigned k=0;k<a.count;k++)(void)varget(&w.input);
    unsigned tag=header_get(&w.input,1);size_t zn=varget(&w.input);
    const uint8_t*z=header_take(&w.input,zn);
    if(!w.input.ok)abort();
    if((tag&3)==2){uint16_t freq[256];if(ng_read_freq_table(z,zn,freq,&bytes))abort();}
    free(a.words.data);free(w.output.data);return bytes;
}
#endif
#define CHECK(x) do{if(!(x)){fprintf(stderr,"line %d: %s\n",__LINE__,#x);exit(1);}}while(0)
typedef struct{const uint8_t*p;size_t n,i;} reader;
static const uint8_t*take(reader*r,size_t n){CHECK(n<=r->n-r->i);const uint8_t*p=r->p+r->i;r->i+=n;return p;}
static unsigned get(reader*r){const uint8_t*p=take(r,4);return p[0]|(unsigned)p[1]<<8|(unsigned)p[2]<<16|(unsigned)p[3]<<24;}
int main(int argc,char**argv){
    CHECK(argc==2);FILE*f=fopen(argv[1],"rb");CHECK(f&&!fseek(f,0,SEEK_END));long sz=ftell(f);CHECK(sz>=12);rewind(f);
    uint8_t*src=malloc((size_t)sz);CHECK(src&&fread(src,1,(size_t)sz,f)==(size_t)sz);fclose(f);
    CHECK(!memcmp(src,"NCF1",4));reader frame={src,(size_t)sz,4};size_t hn=get(&frame),hz=get(&frame);
    uint8_t*h=malloc(hn);CHECK(h&&ZSTD_decompress(h,hn,take(&frame,hz),hz)==hn);
    reader r={h,hn,0};unsigned d=get(&r),mode=get(&r),nm=get(&r),nc=get(&r);(void)get(&r);
    unsigned edge=mode>>8?mode>>8:4;size_t count=(size_t)(d/edge)*(d/edge)*(d/edge);
    vc_buffer raw={0},models={0};
    for(unsigned i=0;i<nm;i++){size_t len=get(&r);CHECK(vc_buffer_append(&models,take(&r,len),len));}
    const uint8_t*menu=take(&r,nc*12);unsigned single=get(&r);uint8_t*ids=malloc(count);CHECK(ids);
    if(single==254){size_t len=get(&r);CHECK(len==(count*vc_selector_bits(nc)+7)/8);vc_selector_unpack(take(&r,len),count,vc_selector_bits(nc),ids);}
    else if(single==255){size_t len=get(&r);CHECK(vc_ng_stream_decode(take(&r,len),len,ids,count));}
    else if(single==253){size_t len=get(&r);CHECK(vc_selector_decode(take(&frame,len),len,ids,count));}
    else{CHECK(single<nc);memset(ids,single,count);}
    uint8_t*sel=NULL;size_t sn=0;CHECK(vc_selector_encode(ids,count,&sel,&sn));
    unsigned fields[7]={d,d,d,(mode&255)|(edge<<8),nm,nc,(unsigned)sn};
    for(unsigned j=0;j<7;j++)CHECK(vc_buffer_put(&raw,fields[j],4));
    CHECK(vc_buffer_append(&raw,models.data,models.size)&&vc_buffer_append(&raw,menu,nc*12)&&vc_buffer_append(&raw,h+r.i,hn-r.i));
    if(hn>r.i){uint8_t*p=raw.data+28+models.size+nc*12;size_t pn=ng_model_size(p);CHECK(pn>=6);
        memset(p+6,0,pn-6-(4u*p[4]-2));}
    uint8_t*enc=NULL,*dec=NULL;size_t en=0,dn=0;
    CHECK(vc_header_encode(raw.data,raw.size,&enc,&en));
    const char *export_path=getenv("HEADER_EXPORT");
    if(export_path){
        FILE *ef=fopen(export_path,"wb");CHECK(ef&&fwrite(enc,1,en,ef)==en);CHECK(!fclose(ef));
        size_t counts[60]={0};for(size_t i=0;i<count;i++)counts[ids[i]]++;
        unsigned best=0;for(unsigned i=1;i<nc;i++)if(counts[i]>counts[best])best=i;
        fprintf(stderr,"dominant_predictor=%u\n",menu[best*12]);
    }
#ifdef HEADER_STATS
    fprintf(stderr,"header_table_bytes=%zu\n",header_table_bytes(enc,en));
#endif
    CHECK(vc_header_decode(enc,en,&dec,&dn)&&dn==raw.size&&!memcmp(dec,raw.data,dn));
    size_t cap=ZSTD_compressBound(raw.size);uint8_t*z=malloc(cap);CHECK(z);
    size_t zn=ZSTD_compress(z,cap,raw.data,raw.size,3);CHECK(!ZSTD_isError(zn));
    vc_buffer nf={0};CHECK(vc_buffer_append(&nf,"NCF1",4)&&vc_buffer_put(&nf,en,4)&&vc_buffer_append(&nf,enc,en)&&vc_buffer_append(&nf,sel,sn)&&vc_buffer_append(&nf,src+frame.i,(size_t)sz-frame.i));
    size_t volume=(size_t)d*d*d;uint8_t*voxels=malloc(volume);CHECK(voxels);
#ifdef BASELINE
    CHECK(!vc_ng_preset_decode(src,(size_t)sz,voxels,volume));
#else
    CHECK(!vc_ng_preset_decode(nf.data,nf.size,voxels,volume));
#endif
    uint64_t hash=UINT64_C(14695981039346656037);for(size_t i=0;i<volume;i++){hash^=voxels[i];hash*=UINT64_C(1099511628211);}
    printf("{\"dimension\":%u,\"edge\":%u,\"raw_header\":%zu,\"compact_header\":%zu,\"zstd_header\":%zu,\"selector_bytes\":%zu,\"original_header\":%zu,\"original_frame\":%ld,\"new_frame\":%zu,\"voxel_hash\":\"%016llx\"}\n",d,edge,raw.size,en,zn,sn,hz,sz,nf.size,(unsigned long long)hash);
    free(voxels);free(nf.data);
    free(src);free(h);free(models.data);free(raw.data);free(ids);free(sel);free(enc);free(dec);free(z);return 0;
}
