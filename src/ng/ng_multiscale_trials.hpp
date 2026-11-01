// Compression research only. Native, independently decodable whole chunks.
// Coarse samples bracket each new point; causal finer neighbors correct the
// interpolant. No side information about the source is used by the decoder.
namespace multiscale_trials {
template<class Consume> void walk(unsigned d,unsigned mode,uint8_t*data,Consume consume){
    require(d>=2&&(d&(d-1))==0&&mode<=3);
    size_t n=size_t(d)*d*d;Bytes known(n);size_t pitch[3]={1,d,size_t(d)*d};
    data[0]=consume(0,0,128*256);known[0]=1;
    for(unsigned step=d/2;step;step/=2)for(unsigned axis=0;axis<3;axis++){
        unsigned stride[3];for(unsigned a=0;a<3;a++)stride[a]=a<=axis?step:2*step;
        for(unsigned z=0;z<d;z+=stride[2])for(unsigned y=0;y<d;y+=stride[1])for(unsigned x=0;x<d;x+=stride[0]){
            unsigned c[3]={x,y,z};if(c[axis]%(2*step)!=step)continue;
            size_t i=(size_t(z)*d+y)*d+x,delta=pitch[axis]*step;
            auto interp=[&](size_t j){
                require(known[j-delta]);int lo=data[j-delta];
                if(c[axis]+step<d){require(known[j+delta]);return (lo+int(data[j+delta])+1)/2;}
                if(c[axis]>=3*step){require(known[j-3*delta]);return std::clamp((3*lo-int(data[j-3*delta])+1)/2,0,255);}
                return lo;
            };
            int base=interp(i),sum=0,count=0;
            // Same refinement stage, previously decoded fine samples. Their
            // coarse brackets are already available on both sides as well.
            if(mode)for(unsigned a=0;a<3;a++)if(a!=axis&&c[a]>=stride[a]){
                size_t j=i-pitch[a]*stride[a];if(known[j]){sum+=int(data[j])-interp(j);count++;}
            }
            int correction=count?sum/count:0;if(mode==1)correction/=2;
            int pred=std::clamp(base+correction,0,255);
            int contrast=std::abs(int(data[i-delta])-base);
            unsigned cls=0;while(contrast>1&&cls<7){contrast>>=1;cls++;}
            // Mode3 additionally separates low/high local correction energy.
            unsigned group=axis*8+cls;if(mode==3&&std::abs(correction)>4)group+=24;
            data[i]=consume(i,group,uint16_t(pred*256));known[i]=1;
        }
    }
    require(std::all_of(known.begin(),known.end(),[](uint8_t x){return x==1;}));
}
Bytes encode(const uint8_t*src,unsigned d,unsigned mode){
    require(ng_geometry_valid(d,d,d));size_t n=size_t(d)*d*d;Bytes data(n);unsigned G=mode==3?48:24;
    std::vector<Bytes>streams(G);
    walk(d,mode,data.data(),[&](size_t i,unsigned g,uint16_t q){streams[g].push_back(ng_prediction_symbol(src[i],q));return src[i];});
    Bytes h,body;put32(h,d);put32(h,mode);
    for(auto&v:streams)small_bank::stream_header(h,body,pack_stream(v,true),v.size());
    auto z=packed_bank::compress(h);Bytes frame={'M','S','P','1'};put32(frame,h.size());put32(frame,z.size());packed_bank::append(frame,z);packed_bank::append(frame,body);return frame;
}
void decode(const uint8_t*src,size_t n,uint8_t*out,size_t cap){
    require(n>=12&&!memcmp(src,"MSP1",4));Bytes frame(src,src+n);packed_bank::Reader r{frame};r.skip(4);auto hn=r.u(4),zn=r.u(4);auto hb=packed_bank::decompress(r.take(zn),hn);packed_bank::Reader h{hb};
    unsigned d=h.u(4),mode=h.u(4);require(mode<=3&&ng_geometry_valid(d,d,d));size_t count=size_t(d)*d*d;require(cap>=count);
    unsigned G=mode==3?48:24;std::vector<Bytes>streams(G);std::vector<size_t>pos(G);size_t total=0;
    for(auto&v:streams){v=small_bank::read_stream(h,r,count-total);total+=v.size();}require(total==count);h.end();r.end();
    walk(d,mode,out,[&](size_t,unsigned g,uint16_t q){require(pos[g]<streams[g].size());return geometry_trials::reconstruct(streams[g][pos[g]++],q);});
    for(unsigned g=0;g<G;g++)require(pos[g]==streams[g].size());
}
}
extern "C" int ng_multiscale_encode(const uint8_t*s,unsigned d,unsigned mode,uint8_t*out,size_t cap,size_t*n){try{auto f=multiscale_trials::encode(s,d,mode);require(f.size()<=cap);std::copy(f.begin(),f.end(),out);*n=f.size();return 0;}catch(const std::exception&e){fprintf(stderr,"multiscale encode: %s\n",e.what());return -1;}}
extern "C" int ng_multiscale_decode(const uint8_t*s,size_t n,uint8_t*out,size_t cap){try{multiscale_trials::decode(s,n,out,cap);return 0;}catch(const std::exception&e){fprintf(stderr,"multiscale decode: %s\n",e.what());return -1;}}
