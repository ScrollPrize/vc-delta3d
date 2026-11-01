// Experimental entropy routing. Predictors/residuals are unchanged. All fitting
// is native; decoder routing is integer-only and independent within each row.
namespace entropy_alternatives {
using Hist=std::vector<uint32_t>;
struct Router {
    unsigned kind=0,side=4;
    Bytes map;
    std::array<int32_t,4> weights{};
    learned_difficulty::Model score;
    unsigned cell(unsigned cls,unsigned K,int n,int b)const{
        unsigned c=std::min(3u,cls*4/K);
        if(kind==10)return (c*8+unsigned(n+b)/64)*16+unsigned(n-b+255)/32;
        int score=weights[0]+weights[1]*(n+b)+weights[2]*std::abs(n-b)+weights[3]*int(cls);
        return c*32+unsigned(std::clamp(score/256,0,31));
    }
    unsigned group(size_t i,unsigned d,unsigned cls,unsigned K,unsigned bank,const uint8_t*src)const{
        unsigned x=i%d,y=i/d%d,z=i/(size_t(d)*d);
        if(kind>=16){int16_t f[learned_difficulty::max_features];score.features([&](int dx,int dy,int dz){return int(geometry_trials::value(src,int(x)+dx,int(y)+dy,int(z)+dz,d));},f);return score.group(cls,bank,f);}
        if(kind>=12){unsigned q=d/side,g=map[size_t(bank)*q*q*q+(size_t(z/side)*q+y/side)*q+x/side];return kind>=14?bank*64+g*K+cls:bank*16+g;}
        int n=y?src[i-d]:128,b=z?src[i-size_t(d)*d]:128;
        return bank*16+map[bank*(kind==10?512:128)+cell(cls,K,n,b)];
    }
};
// Histogram KL assignment with deterministic spread initialization. Bank/table
// cost is included in the measured frame, not hidden from compression results.
Bytes cluster(const std::vector<Hist>&hist,unsigned channels=1,unsigned limit=16){
    unsigned count=hist.size(),symbols=256*channels;Bytes ids(count);std::vector<std::pair<double,unsigned>> order;
    // Sparse fitting records avoid repeatedly scanning empty block histograms.
    std::vector<std::vector<std::pair<unsigned,uint32_t>>> sparse(count);
    for(unsigned i=0;i<count;i++){double total=0,m=0;for(unsigned s=0;s<symbols;s++)if(hist[i][s]){total+=hist[i][s];m+=hist[i][s]*std::log2(1.+s%256);sparse[i].emplace_back(s,hist[i][s]);}if(total)order.emplace_back(m/total,i);}
    std::sort(order.begin(),order.end());
    for(unsigned k=0;k<order.size();k++)ids[order[k].second]=std::min(limit-1,unsigned(limit*size_t(k)/order.size()));
    for(unsigned pass=0;pass<4;pass++){
        std::vector<Hist> sums(limit,Hist(symbols));
        for(unsigned i=0;i<count;i++)for(auto [s,v]:sparse[i])sums[ids[i]][s]+=v;
        std::vector<double> costs(limit*symbols);for(unsigned g=0;g<limit;g++)for(unsigned c=0;c<channels;c++){double total=128;for(unsigned s=0;s<256;s++)total+=sums[g][c*256+s];for(unsigned s=0;s<256;s++)costs[g*symbols+c*256+s]=std::log2(total/(sums[g][c*256+s]+.5));}
        for(unsigned i=0;i<count;i++){
            double score[16]={};for(auto [s,v]:sparse[i])for(unsigned g=0;g<limit;g++)score[g]+=v*costs[g*symbols+s];
            ids[i]=std::min_element(score,score+limit)-score;
        }
    }return ids;
}
Router fit(const Snapshot&s,const Bytes&src,unsigned kind){
    Router r;r.kind=kind;r.side=(kind==13||kind==15)?8:4;unsigned B=banks(s),K=s.thresholds.size()+1,d=s.d;
    if(kind>=16){r.score=learned_difficulty::fit(s,src,kind);return r;}
    require(kind<14||K<=16);
    if(kind==11){
        // Fit cheap oblique log-residual-scale regressor, quantize coefficients
        // once; subsequent histograms use exactly the integer decoder rule.
        double a[4][5]={};
        for(size_t i=0;i<src.size();i++){
            auto&c=s.menu[s.selectors[geometry_trials::block(i,d)]];if(c.predictor==choice_frame::zero_predictor)continue;
            int n=i/d%d?src[i-d]:128,b=i/(size_t(d)*d)?src[i-size_t(d)*d]:128;
            double x[]={1.,double(n+b),double(std::abs(n-b)),double(s.classes[i])};double target=4*std::log2(1.+s.symbols[i]);
            for(unsigned j=0;j<4;j++){for(unsigned k=0;k<4;k++)a[j][k]+=x[j]*x[k];a[j][4]+=x[j]*target;}
        }
        for(unsigned j=0;j<4;j++)a[j][j]+=1.;
        for(unsigned j=0;j<4;j++){unsigned p=j;for(unsigned k=j+1;k<4;k++)if(std::abs(a[k][j])>std::abs(a[p][j]))p=k;for(unsigned k=0;k<5;k++)std::swap(a[j][k],a[p][k]);double v=a[j][j];for(unsigned k=j;k<5;k++)a[j][k]/=v;for(unsigned q=0;q<4;q++)if(q!=j){double v=a[q][j];for(unsigned k=j;k<5;k++)a[q][k]-=v*a[j][k];}}
        for(unsigned j=0;j<4;j++)r.weights[j]=int32_t(std::clamp(std::round(a[j][4]*256),-65536.,65536.));
    }
    unsigned cells=kind>=12?(d/r.side)*(d/r.side)*(d/r.side):kind==10?512:128;
    r.map.resize(kind>=12?cells:B*cells);
    for(unsigned bank=0;bank<B;bank++){
        unsigned channels=kind>=14?K:1;std::vector<Hist>hist(cells,Hist(channels*256));
        for(size_t i=0;i<src.size();i++){
            auto&c=s.menu[s.selectors[geometry_trials::block(i,d)]];if(c.entropy!=bank||c.predictor==choice_frame::zero_predictor)continue;
            unsigned x=i%d,y=i/d%d,z=i/(size_t(d)*d),cell;
            if(kind>=12){unsigned q=d/r.side;cell=(z/r.side*q+y/r.side)*q+x/r.side;}
            else cell=r.cell(s.classes[i],K,y?src[i-d]:128,z?src[i-size_t(d)*d]:128);
            ++hist[cell][(kind>=14?s.classes[i]*256:0)+s.symbols[i]];
        }
        auto assigned=cluster(hist,channels,kind>=14?4:16);
        if(kind>=12){
            // One explicit routing map per original entropy bank; selector
            // determines which bank applies at each voxel.
            if(bank==0)r.map.clear();r.map.insert(r.map.end(),assigned.begin(),assigned.end());
        }else std::copy(assigned.begin(),assigned.end(),r.map.begin()+bank*cells);
    }return r;
}
void write(Bytes&h,const Router&r){put32(h,r.kind);if(r.kind>=16){learned_difficulty::write(h,r.score);return;}for(auto w:r.weights)put32(h,uint32_t(w));put32(h,r.map.size());packed_bank::append(h,r.map);}
Router read(packed_bank::Reader&h,unsigned d,unsigned B){Router r;r.kind=h.u(4);require(r.kind>=10&&r.kind<=18);if(r.kind>=16){r.score=learned_difficulty::read(h,B,r.kind);return r;}r.side=(r.kind==13||r.kind==15)?8:4;for(auto&w:r.weights){w=int32_t(packed_bank::signed_value(h.u(4),32));require(w>=-65536&&w<=65536);}unsigned cells=r.kind>=12?(d/r.side)*(d/r.side)*(d/r.side):r.kind==10?512:128;require(h.u(4)==B*cells);r.map=h.take(B*cells);for(auto v:r.map)require(v<(r.kind>=14?4:16));return r;}
}
