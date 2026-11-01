// Compression-only joint activity/prediction classifier experiments.
// Shared snapshot framing, causal prediction and rANS implementation live in
// edge_trials/compact_contexts. No predictor fitting or pixel values change.
namespace joint_context_trials {
using namespace edge_trials;
using Hist=std::array<uint32_t,257>;
double cost(const Hist&h){double n=0,c=0;unsigned active=0;for(auto v:h)if(v){n+=v;c-=double(v)*std::log2(double(v));active++;}return n?c+n*std::log2(n)+12*active+24:0;}
struct Fit {
    JointMap map;unsigned A,P,K,B,bins;std::vector<Hist>cells;std::vector<unsigned>cls;std::vector<double>ac;
    Fit(const Trial&t,const std::vector<uint16_t>&q,const std::vector<uint16_t>&activity,unsigned nb,bool gate):K(t.s.thresholds.size()+1),B(hybrid_geometry::banks(t.s)),bins(nb){
        map.gate=gate;map.banks=B;
        // Retain exact old class boundaries and add finer raw-activity slices.
        for(auto v:t.s.thresholds)if(v+1<2560)map.athr.push_back(v+1);
        for(unsigned v=1;v<32;v*=2)map.athr.push_back(v);
        for(unsigned v=32;v<256;v+=16)map.athr.push_back(v);
        for(unsigned v=256;v<2560;v+=64)map.athr.push_back(v);
        std::sort(map.athr.begin(),map.athr.end());map.athr.erase(std::unique(map.athr.begin(),map.athr.end()),map.athr.end());
        map.pthr={1,2,4,8,16,24,32,48,64,96,128,160,192,208,224,232,240,248,252,254,255};
        map.prepare();A=map.athr.size()+1;P=map.pthr.size()+1;map.groups=B*K*bins;require(map.groups<=2048);
        cells.resize(B*A*P);map.ids.resize(cells.size());cls.resize(A);ac.resize(A);
        for(unsigned a=0;a<A;a++){unsigned lo=a?map.athr[a-1]:0,hi=a+1<A?map.athr[a]:2560;
            cls[a]=std::lower_bound(t.s.thresholds.begin(),t.s.thresholds.end(),lo)-t.s.thresholds.begin();
            unsigned clo=cls[a]?t.s.thresholds[cls[a]-1]+1:0,chi=cls[a]+1<K?t.s.thresholds[cls[a]]+1:2560;
            ac[a]=((lo+hi)*0.5-clo)/std::max(1u,chi-clo)-0.5;
            for(unsigned b=0;b<B;b++)for(unsigned p=0;p<P;p++){unsigned value=p?map.pthr[p-1]:0;
                map.ids[(b*A+a)*P+p]=(b*K+cls[a])*bins+prediction_context(value*256,bins==3?24:27);}}
        for(size_t i=0;i<N;i++){unsigned b=t.s.menu[t.s.selectors[geometry_trials::block(i,t.s.d)]].entropy;
            unsigned cell=(b*A+map.ai[activity[i]])*P+map.pi[geometry_trials::predicted(q[i])],sym=t.s.symbols[i];
            if(gate){if(t.source[i]==0)sym=256;else{unsigned zs=ng_prediction_symbol(0,q[i]);require(sym!=zs);sym-=sym>zs;}}
            cells[cell][sym]++;}
    }
    std::vector<Hist>aggregate()const{std::vector<Hist>h(map.groups);for(unsigned c=0;c<cells.size();c++)for(unsigned s=0;s<257;s++)h[map.ids[c]][s]+=cells[c][s];return h;}
    void angled(){
        const std::vector<double>base=bins==3?std::vector<double>{16,240}:std::vector<double>{1,4,16,240,252,255};
        for(unsigned b=0;b<B;b++)for(unsigned k=0;k<K;k++){
            auto intercept=base;std::vector<double>slope(base.size());
            auto assign=[&](){for(unsigned a=0;a<A;a++)if(cls[a]==k)for(unsigned p=0;p<P;p++){
                double v=p?map.pthr[p-1]:0;unsigned g=0;double previous=0;
                for(unsigned j=0;j<base.size();j++){double boundary=std::clamp(intercept[j]+slope[j]*ac[a],previous,256.0);g+=v>=boundary;previous=boundary;}
                map.ids[(b*A+a)*P+p]=(b*K+k)*bins+g;}};
            auto score=[&](){std::vector<Hist>h(bins);for(unsigned a=0;a<A;a++)if(cls[a]==k)for(unsigned p=0;p<P;p++){unsigned c=(b*A+a)*P+p;for(unsigned s=0;s<257;s++)h[map.ids[c]%bins][s]+=cells[c][s];}double v=0;for(auto&x:h)v+=cost(x);return v;};
            double best=score();
            for(unsigned pass=0;pass<2;pass++)for(unsigned j=0;j<base.size();j++)for(unsigned axis=0;axis<2;axis++){
                double&param=axis?slope[j]:intercept[j],old=param,winner=old;
                for(double delta:{-16.,-4.,-1.,1.,4.,16.}){param=old+delta;assign();double v=score();if(v<best){best=v;winner=param;}}
                param=winner;assign();
            }
        }
    }
    void lut(){
        double best=0;for(auto&h:aggregate())best+=cost(h);auto saved=map.ids;
        for(unsigned pass=0;pass<6;pass++){
            auto h=aggregate();std::vector<std::array<float,257>>logp(h.size());
            for(unsigned g=0;g<h.size();g++){double n=128.5;for(auto v:h[g])n+=v;for(unsigned s=0;s<257;s++)logp[g][s]=std::log2(n/(h[g][s]+0.5));}
            auto old=map.ids;
            for(unsigned b=0;b<B;b++)for(unsigned a=0;a<A;a++)for(unsigned p=0;p<P;p++){
                unsigned c=(b*A+a)*P+p;uint64_t n=0;for(auto v:cells[c])n+=v;if(!n)continue;
                unsigned candidates[5]={old[c],a?old[c-P]:old[c],a+1<A?old[c+P]:old[c],p?old[c-1]:old[c],p+1<P?old[c+1]:old[c]};
                double lowest=1e300;unsigned winner=old[c];for(auto g:candidates){double v=0;for(unsigned s=0;s<257;s++)if(cells[c][s])v+=cells[c][s]*logp[g][s];
                    // Weak spatial regularity: favour a small, compressible map.
                    for(unsigned j=1;j<5;j++)v+=2*(g!=candidates[j]);if(v<lowest){lowest=v;winner=g;}}
                map.ids[c]=winner;
            }
            double objective=0;for(auto&x:aggregate())objective+=cost(x);if(objective<best){best=objective;saved=map.ids;}
        }map.ids=std::move(saved);
    }
};
Bytes frame(const Trial&t,JointMap map,const std::vector<uint16_t>&q,const std::vector<uint16_t>&activity){
    std::vector<unsigned>remap(map.groups,UINT_MAX);unsigned groups=0;for(auto&id:map.ids){if(remap[id]==UINT_MAX)remap[id]=groups++;id=remap[id];}map.groups=groups;
    std::vector<Bytes>streams(groups*(map.gate?2:1));
    for(size_t i:wave_positions(t.s)){unsigned b=t.s.menu[t.s.selectors[geometry_trials::block(i,t.s.d)]].entropy,g=map.group(b,activity[i],q[i])*(map.gate?2:1),sym=t.s.symbols[i];
        if(map.gate){bool zero=t.source[i]==0;streams[g].push_back(zero);if(!zero){unsigned zs=ng_prediction_symbol(0,q[i]);streams[g+1].push_back(sym-(sym>zs));}}
        else streams[g].push_back(sym);}
    Bytes h,body;snapshot_header(h,t.s,0,0,0);map.write(h);compact_contexts::write(h,body,streams);require(h.size()<=1048576);auto z=packed_bank::compress(h);Bytes out={'E','G','B','4'};put32(out,h.size());put32(out,z.size());packed_bank::append(out,z);packed_bank::append(out,body);return out;
}
Bytes encode(Trial&t,unsigned kind,unsigned bins,bool gate){
    require(kind<=2&&(bins==3||bins==7));GeometryScope geom(t.s.d,t.s.d,t.s.d);choice_grid::Scope grid(4);
    std::vector<uint16_t>q(N),activity(N);Bytes out(N);PredictionCapture capture{t.s,q};native_prediction_walk(t.s,out.data(),PredictionCapture::get,&capture);require(out==t.source);
    require(ng_activity_symbol_values(t.s.symbols.data(),activity.data())==0);Fit fit(t,q,activity,bins,gate);if(kind==1)fit.angled();if(kind==2)fit.lut();return frame(t,fit.map,q,activity);
}
}
extern "C" int ng_joint_context_encode(void*p,unsigned kind,unsigned bins,unsigned gate,uint8_t*out,size_t cap,size_t*n){try{require(p&&out&&n&&gate<=1);auto frame=joint_context_trials::encode(*static_cast<edge_trials::Trial*>(p),kind,bins,gate);require(frame.size()<=cap);std::copy(frame.begin(),frame.end(),out);*n=frame.size();return 0;}catch(const std::exception&e){fprintf(stderr,"joint context: %s\n",e.what());return -1;}}
