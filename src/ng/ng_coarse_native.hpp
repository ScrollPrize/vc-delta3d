#include "ng_coarse_inputs.h"
namespace coarse_native {
bool smooth(unsigned mode){return mode>=16;}
unsigned scale(unsigned mode){return mode&32?4:2;}
unsigned extras(unsigned mode){if(smooth(mode)){require(mode<256&&(!(mode&192)||!(mode&8)));unsigned base=mode&31;require(base==17||base==18||base==19||base==25||base==26||base==27);return (mode&7)==2?27:(mode&7)==3?54:0;}require(mode<=4);return mode==2?8:mode==3?27:mode==4?54:0;}
Bytes box_mean(const uint8_t*src,unsigned d,unsigned s){
    unsigned c=d/s,count=s*s*s;Bytes out(size_t(c)*c*c);
    for(unsigned z=0;z<c;z++)for(unsigned y=0;y<c;y++)for(unsigned x=0;x<c;x++){
        unsigned sum=0;for(unsigned dz=0;dz<s;dz++)for(unsigned dy=0;dy<s;dy++)for(unsigned dx=0;dx<s;dx++)sum+=src[(size_t(s*z+dz)*d+s*y+dy)*d+s*x+dx];
        out[(size_t(z)*c+y)*c+x]=(sum+count/2)/count;
    }return out;
}
Bytes downsample(const uint8_t*src,unsigned d,bool gaussian){
    unsigned c=d/2;Bytes out(size_t(c)*c*c);
    for(unsigned z=0;z<c;z++)for(unsigned y=0;y<c;y++)for(unsigned x=0;x<c;x++){
        unsigned sum=0;
        if(!gaussian){for(unsigned dz=0;dz<2;dz++)for(unsigned dy=0;dy<2;dy++)for(unsigned dx=0;dx<2;dx++)sum+=src[(size_t(2*z+dz)*d+2*y+dy)*d+2*x+dx];sum=(sum+4)/8;}
        else{for(int dz=-1;dz<=1;dz++)for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
            unsigned xx=std::clamp(int(2*x)+dx,0,int(d)-1),yy=std::clamp(int(2*y)+dy,0,int(d)-1),zz=std::clamp(int(2*z)+dz,0,int(d)-1);
            sum+=src[(size_t(zz)*d+yy)*d+xx]*(dx==0?2:1)*(dy==0?2:1)*(dz==0?2:1);
        }sum=(sum+32)/64;}
        out[(size_t(z)*c+y)*c+x]=uint8_t(sum);
    }return out;
}
Bytes smooth_planes(const Bytes&coarse,unsigned d,unsigned extra,bool gaussian,unsigned base=2){
    size_t n=size_t(d)*d*d;Bytes planes(n*extra),lower;if(extra==54)lower=downsample(coarse.data(),d/base,gaussian);
    for(unsigned k=0;k<extra;k++){
        unsigned step=k>=27?2*base:base,c=d/step,kk=k%27;int off[3];for(unsigned a=0;a<3;a++){off[a]=int(kk%3)-1;kk/=3;}
        const Bytes&values=k>=27?lower:coarse;
        for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++)for(unsigned x=0;x<d;x++){
            unsigned xyz[3]={x,y,z};int q[3];for(unsigned a=0;a<3;a++)q[a]=std::clamp(int((xyz[a]+(gaussian?step/2:0))/step)+off[a],0,int(c)-1);
            planes[k*n+(size_t(z)*d+y)*d+x]=values[(size_t(q[2])*c+q[1])*c+q[0]];
        }
    }return planes;
}
struct Model {Bytes params;std::vector<int32_t>w;unsigned extra=0;bool prefix=false;};
Model fit(const uint8_t*src,unsigned mode,const uint8_t*planes=nullptr){
    Model m;m.extra=extras(mode);m.prefix=mode!=0&&!smooth(mode);m.params.resize(4096);m.w.resize(2*std::max(1u,m.extra));size_t n=0;
    int rc=smooth(mode)?ng_external_native_fit(src,planes,m.extra,m.params.data(),m.params.size(),&n,m.w.data()):ng_coarse_native_fit(src,m.extra,m.prefix,m.params.data(),m.params.size(),&n,m.w.data());require(rc==0);m.params.resize(n);return m;
}
void walk(const Model&m,unsigned d,const uint8_t*src,Bytes&symbols,uint8_t*out,const uint8_t*planes=nullptr,const Bytes*coarse=nullptr,unsigned mode=0){
    std::unique_ptr<void,decltype(&ng_causal_destroy)>state(ng_causal_create(m.params.data(),m.params.size()),ng_causal_destroy);require(state!=nullptr);
    const uint8_t*data=src?src:out;
    for(size_t i=0;i<size_t(d)*d*d;i++){
        unsigned x=i%d,y=i/d%d,z=i/(size_t(d)*d);uint16_t q;uint8_t cls;require(ng_causal_prediction(state.get(),i,&q,&cls)==0);
        bool known=m.prefix&&ng_coarse_site(x,y,z);
        if(!known)for(unsigned k=0;k<m.extra;k++)q=uint16_t(q+int64_t(m.w[(z==0)*m.extra+k])*(planes?planes[k*size_t(d)*d*d+i]:ng_coarse_input(data,d,x,y,z,k,m.extra)));
        unsigned s=scale(mode);
        if((mode&192)&&x%s==s-1&&y%s==s-1&&z%s==s-1){
            int sum=0;for(unsigned dz=0;dz<s;dz++)for(unsigned dy=0;dy<s;dy++)for(unsigned dx=0;dx<s;dx++)if(dx||dy||dz)sum+=data[(size_t(z-dz)*d+y-dy)*d+x-dx];
            int pred=int((*coarse)[(size_t(z/s)*(d/s)+y/s)*(d/s)+x/s])*int(s*s*s)-sum;
            if(mode&128){int lo=std::max(0,pred-int(s*s*s)/2),hi=std::min(255,pred+int(s*s*s)/2-1);require(lo<=hi);int p=geometry_trials::predicted(q);if(p<lo||p>hi)q=uint16_t(std::clamp(p,lo,hi)*256);}
            else q=uint16_t(std::clamp(pred,0,255)*256);
        }
        uint8_t value=src?src[i]:known?out[i]:geometry_trials::reconstruct(symbols[i],q);
        if(src)symbols[i]=known?0:ng_prediction_symbol(value,q);
        if(out)out[i]=value;
        require(ng_causal_accept(state.get(),i,value,known?value:geometry_trials::predicted(q))==0);
    }
}
Bytes encode(const uint8_t*src,unsigned d,unsigned mode,unsigned depth=1){
    require(d>=64&&ng_geometry_valid(d,d,d)&&depth>=1&&depth<=2&&(depth==1||d>=128)&&(!(mode&32)||(depth==1&&d>=128)));unsigned extra=extras(mode);GeometryScope geom(d,d,d);Bytes coarse_frame,planes,coarse;
    if(mode!=0){unsigned c=d/scale(mode);coarse.resize(size_t(c)*c*c);
        if(smooth(mode)){if(mode&8){coarse=downsample(src,d,true);if(mode&32)coarse=downsample(coarse.data(),d/2,true);}else coarse=box_mean(src,d,scale(mode));planes=smooth_planes(coarse,d,extra,bool(mode&8),scale(mode));}
        else for(unsigned z=0;z<c;z++)for(unsigned y=0;y<c;y++)for(unsigned x=0;x<c;x++)coarse[(size_t(z)*c+y)*c+x]=src[(size_t(2*z)*d+2*y)*d+2*x];
        if(depth>1)coarse_frame=encode(coarse.data(),c,mode,depth-1);
        else{coarse_frame.resize(3*coarse.size()+131072);size_t n=0;require(ng_codec_encode(coarse.data(),c,3,coarse_frame.data(),coarse_frame.size(),&n)==0);coarse_frame.resize(n);}}
    auto m=fit(src,mode,planes.data());Bytes symbols(N);walk(m,d,src,symbols,nullptr,planes.empty()?nullptr:planes.data(),&coarse,mode);std::vector<uint16_t>activity(N);require(ng_activity_symbol_values(symbols.data(),activity.data())==0);
    Bytes active;std::vector<uint16_t>aa;for(size_t i=0;i<N;i++)if(!m.prefix||!ng_coarse_site(i%d,i/d%d,i/(size_t(d)*d))){active.push_back(symbols[i]);aa.push_back(activity[i]);}
    auto thr=activity_refine(active,aa,activity_quantiles(aa,16),{}, {},1,true,64,true);auto classes=activity_classify(activity,thr);std::vector<Bytes>streams(16);
    for(size_t i=0;i<N;i++)if(!m.prefix||!ng_coarse_site(i%d,i/d%d,i/(size_t(d)*d)))streams[classes[i]].push_back(symbols[i]);
    Bytes h,body;put32(h,d);put32(h,mode);put32(h,m.params.size());packed_bank::append(h,m.params);for(unsigned k=0;k<2*m.extra;k++)put32(h,uint32_t(m.w[k]));for(auto x:thr)packed_bank::put16(h,x);
    put32(h,coarse_frame.size());packed_bank::append(body,coarse_frame);for(auto&v:streams)small_bank::stream_header(h,body,pack_stream(v,true),v.size());
    auto z=packed_bank::compress(h);Bytes frame={'C','N','G','1'};put32(frame,h.size());put32(frame,z.size());packed_bank::append(frame,z);packed_bank::append(frame,body);return frame;
}
struct Read {unsigned d;bool prefix;std::vector<Bytes>&s;size_t pos[16]={};
    static int get(void*p,size_t i,unsigned c){auto&r=*static_cast<Read*>(p);if(r.prefix&&ng_coarse_site(i%r.d,i/r.d%r.d,i/(size_t(r.d)*r.d)))return 0;if(c>=16||r.pos[c]>=r.s[c].size())return -1;return r.s[c][r.pos[c]++];}
};
void decode(const uint8_t*src,size_t n,uint8_t*out,size_t cap,const uint8_t*external_coarse=nullptr){
    require(n>=12&&!memcmp(src,"CNG1",4));Bytes frame(src,src+n);packed_bank::Reader r{frame};r.skip(4);auto hn=r.u(4),zn=r.u(4);auto hb=packed_bank::decompress(r.take(zn),hn);packed_bank::Reader h{hb};
    unsigned d=h.u(4),mode=h.u(4);require(d>=64&&ng_geometry_valid(d,d,d));GeometryScope geom(d,d,d);require(cap>=N);
    Model m;m.extra=extras(mode);m.prefix=mode!=0&&!smooth(mode);m.params=h.take(h.u(4));m.w.resize(2*m.extra);for(auto&w:m.w)w=int32_t(packed_bank::signed_value(h.u(4),32));
    std::vector<uint16_t>thr(15);for(auto&v:thr)v=h.u(2);auto cf=r.take(h.u(4));
    Bytes planes,coarse;if(mode!=0){unsigned c=d/scale(mode);coarse.resize(size_t(c)*c*c);
        if(external_coarse)std::copy(external_coarse,external_coarse+coarse.size(),coarse.begin());
        else if(cf.size()>=4&&!memcmp(cf.data(),"CNG1",4))decode(cf.data(),cf.size(),coarse.data(),coarse.size());
        else require(ng_codec_decode(cf.data(),cf.size(),coarse.data(),coarse.size())==0);
        if(smooth(mode))planes=smooth_planes(coarse,d,m.extra,bool(mode&8),scale(mode));
        else for(unsigned z=0;z<c;z++)for(unsigned y=0;y<c;y++)for(unsigned x=0;x<c;x++)out[(size_t(2*z)*d+2*y)*d+2*x]=coarse[(size_t(z)*c+y)*c+x];}else require(cf.empty());
    std::vector<Bytes>streams(16);size_t total=0;for(auto&v:streams){v=small_bank::read_stream(h,r,N-total);total+=v.size();}require(total==N-(m.prefix?N/8:0));h.end();r.end();
    Read read{d,m.prefix,streams};Bytes raw(N);require(ng_activity_decode(thr.data(),16,0,Read::get,&read,raw.data())==0);for(unsigned k=0;k<16;k++)require(read.pos[k]==streams[k].size());for(auto&v:raw)v=pyramid(v);walk(m,d,nullptr,raw,out,planes.empty()?nullptr:planes.data(),&coarse,mode);
}
// Native byte accounting: each row is d,total,own header,coarse frame,own
// residual payload. The final Fast leaf is counted as one complete payload.
unsigned sizes(const uint8_t*src,size_t n,unsigned d,uint64_t*out,unsigned cap){
    require(cap>=1);out[0]=d;out[1]=n;out[2]=out[3]=0;out[4]=n;
    if(n<4||memcmp(src,"CNG1",4))return 1;
    require(n>=12);Bytes f(src,src+n);packed_bank::Reader r{f};r.skip(4);auto hn=r.u(4),zn=r.u(4);auto hb=packed_bank::decompress(r.take(zn),hn);packed_bank::Reader h{hb};
    require(h.u(4)==d);unsigned mode=h.u(4);h.take(h.u(4));unsigned extra=extras(mode);h.skip(8*extra+30);size_t cn=h.u(4);auto cf=r.take(cn);
    out[2]=12+zn;out[3]=cn;require(out[2]+cn<=n);out[4]=n-out[2]-cn;
    return cn?1+sizes(cf.data(),cn,d/scale(mode),out+5,cap-1):1;
}
}
extern "C" int ng_coarse_native_encode(const uint8_t*s,unsigned d,unsigned mode,uint8_t*out,size_t cap,size_t*n){try{auto f=coarse_native::encode(s,d,mode);require(f.size()<=cap);std::copy(f.begin(),f.end(),out);*n=f.size();return 0;}catch(const std::exception&e){fprintf(stderr,"coarse native encode: %s\n",e.what());return -1;}}
extern "C" int ng_coarse_native_decode(const uint8_t*s,size_t n,uint8_t*out,size_t cap){try{coarse_native::decode(s,n,out,cap);return 0;}catch(const std::exception&e){fprintf(stderr,"coarse native decode: %s\n",e.what());return -1;}}
extern "C" int ng_coarse_pyramid_encode(const uint8_t*s,unsigned d,unsigned mode,unsigned depth,uint8_t*out,size_t cap,size_t*n){try{auto f=coarse_native::encode(s,d,mode,depth);require(f.size()<=cap);std::copy(f.begin(),f.end(),out);*n=f.size();return 0;}catch(const std::exception&e){fprintf(stderr,"coarse pyramid encode: %s\n",e.what());return -1;}}
extern "C" int ng_coarse_pyramid_sizes(const uint8_t*s,size_t n,unsigned d,uint64_t*out,unsigned cap){try{return coarse_native::sizes(s,n,d,out,cap);}catch(const std::exception&e){fprintf(stderr,"coarse pyramid sizes: %s\n",e.what());return -1;}}
extern "C" int ng_coarse_external_decode(const uint8_t*s,size_t n,const uint8_t*coarse,uint8_t*out,size_t cap){try{require(coarse);coarse_native::decode(s,n,out,cap,coarse);return 0;}catch(const std::exception&e){fprintf(stderr,"coarse external decode: %s\n",e.what());return -1;}}
extern "C" int ng_smoothed_coarse(const uint8_t*s,unsigned d,unsigned gaussian,uint8_t*out,size_t cap){try{require(d>=64&&ng_geometry_valid(d,d,d)&&gaussian<=1);auto c=coarse_native::downsample(s,d,gaussian);require(cap>=c.size());std::copy(c.begin(),c.end(),out);return 0;}catch(const std::exception&e){fprintf(stderr,"smooth coarse: %s\n",e.what());return -1;}}
