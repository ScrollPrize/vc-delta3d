// Independent spatial entropy regions; reuse fitted predictors, residuals,
// difficulty classes, native rANS, frame writer, and partition dynamic program.
namespace region_entropy {
struct Input{unsigned edge=0;std::vector<Bytes> models;std::vector<choice_frame::Choice> menu;Bytes choices,symbols,classes;std::vector<uint16_t> thr;unsigned class_count=16;};
thread_local Input* capture=nullptr;
thread_local Bytes reference;
thread_local uint64_t stats[8]={};
thread_local unsigned seed_policy=0;
void observe(unsigned,const std::vector<Bytes>&models,const std::vector<choice_frame::Choice>&menu,const Bytes&choices,const Bytes&symbols,const std::vector<uint16_t>&thr,const Bytes&classes,unsigned,const Bytes&frame){
    if(capture&&frame==reference)*capture={choice_grid::edge(),models,menu,choices,symbols,classes,thr};
}
    struct Reset{
        unsigned root=adaptive_grid::s2_root,opts=adaptive_grid::s2_options,minimum=adaptive_grid::s2_min;
        bool enabled=choice_frame::enabled;
        ~Reset(){capture=nullptr;reference.clear();choice_frame::observer=nullptr;choice_frame::region_banks.clear();choice_frame::enabled=enabled;adaptive_grid::s2_root=root;adaptive_grid::s2_options=opts;adaptive_grid::s2_min=minimum;}
    };
template<class Pack> Bytes refine(unsigned d,const Input&in,unsigned B,unsigned adaptive,unsigned passes,Bytes best,Pack pack_frame){
        std::fill_n(stats,8,0);stats[0]=stats[1]=best.size();stats[2]=in.edge;
        if(in.symbols.empty())return best;
        require(in.class_count>=2&&in.class_count<=64);unsigned n=d/2;size_t blocks=size_t(n)*n*n;
        Bytes zero(blocks),predictor(blocks);std::vector<uint32_t> block(N);
        for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++)for(unsigned x=0;x<d;x++){
            size_t i=(size_t(z)*d+y)*d+x,b=adaptive_grid::index(x/2,y/2,z/2,n);block[i]=b;
            auto c=in.choices[adaptive_grid::index(x/in.edge,y/in.edge,z/in.edge,d/in.edge)];
            zero[b]=in.menu[c].predictor==choice_frame::zero_predictor;predictor[b]=c;
        }
        bool has_zero=std::any_of(zero.begin(),zero.end(),[](auto v){return v;});unsigned banks=B+has_zero;
        Bytes working(blocks);std::vector<std::pair<double,size_t>> order;unsigned cn=d/8;
        std::vector<double> energy(size_t(cn)*cn*cn),count(energy.size());
        for(size_t i=0;i<N;i++)if(!zero[block[i]]){unsigned x=i%d,y=(i/d)%d,z=i/(size_t(d)*d);auto b=adaptive_grid::index(x/8,y/8,z/8,cn);energy[b]+=in.symbols[i];count[b]++;}
        for(size_t b=0;b<energy.size();b++)if(count[b])order.emplace_back(energy[b]/count[b],b);
        std::sort(order.begin(),order.end());Bytes seed(energy.size());
        for(size_t k=0;k<order.size();k++)seed[order[k].second]=std::min<size_t>(B-1,k*B/order.size());
        for(unsigned z=0;z<n;z++)for(unsigned y=0;y<n;y++)for(unsigned x=0;x<n;x++){
            size_t b=adaptive_grid::index(x,y,z,n);working[b]=zero[b]?B:seed[adaptive_grid::index(x/4,y/4,z/4,cn)];
        }
        if(seed_policy){
            // Preserve the incumbent's predictor-conditioned distributions,
            // then split its most populated bank. This is only initialization:
            // subsequent assignments and merges never consult predictor IDs.
            require(B==4);uint64_t population[3]={};std::vector<double> magnitude(blocks);
            for(size_t b=0;b<blocks;b++)if(!zero[b]){unsigned k=in.menu[predictor[b]].entropy;require(k<3);working[b]=k;population[k]++;}
            for(size_t i=0;i<N;i++)magnitude[block[i]]+=in.symbols[i];
            unsigned largest=std::max_element(population,population+3)-population;order.clear();
            for(size_t b=0;b<blocks;b++)if(!zero[b]&&working[b]==largest)order.emplace_back(magnitude[b],b);
            std::sort(order.begin(),order.end());for(size_t k=order.size()/2;k<order.size();k++)working[order[k].second]=3;
        }
        Bytes best_regions;
        auto pack=[&](const Bytes&assignment){stats[3]++;
            auto f=pack_frame(assignment);
            if(f.size()<best.size()){best=f;best_regions=assignment;stats[4]++;}return f.size();};
        pack(working);
        for(unsigned pass=0;pass<passes;pass++){
            std::vector<three_candidate::Stats> hist(banks,three_candidate::Stats(in.class_count));
            for(size_t i=0;i<N;i++)hist[working[block[i]]].hist[in.classes[i]][in.symbols[i]]++;
            for(auto&s:hist)s.normalized_costs();
            std::vector<std::array<double,8>> cost(blocks);
            for(size_t b=0;b<blocks;b++)for(unsigned k=0;k<8;k++)cost[b][k]=(zero[b]?k==B:k<B)?0.:1e30;
            for(size_t i=0;i<N;i++)if(!zero[block[i]])for(unsigned k=0;k<B;k++)cost[block[i]][k]+=hist[k].cost[in.classes[i]][in.symbols[i]];
            Bytes next=working;size_t winning=std::numeric_limits<size_t>::max();
            if(adaptive){
                choice_grid::Scope fine(2);adaptive_grid::Scope root(32);
                auto tokens=adaptive_grid::serialize(working,d);double counts[256]={};for(auto t:tokens)counts[t]++;
                double prior[256]={};for(unsigned k=0;k<banks;k++)prior[k]=-std::log2((counts[k]+.5)/(tokens.size()+.5*(banks+1)));
                prior[255]=-std::log2((counts[255]+.5)/(tokens.size()+.5*(banks+1)));
                for(double penalty:{.25,1.,4.,16.}){
                    for(unsigned k=0;k<banks;k++)adaptive_grid::s2_cost[k]=penalty*prior[k];adaptive_grid::s2_cost[255]=penalty*prior[255];
                    adaptive_grid::s2_root=32;adaptive_grid::s2_min=2;Bytes trial(blocks);adaptive_grid::select(d,cost,trial);
                    // Only the entropy map is adaptive; preserve predictor grid.
                    choice_grid::Scope original(in.edge);adaptive_grid::Scope fixed(0);size_t bytes=pack(trial);
                    if(bytes<winning){winning=bytes;next=std::move(trial);}
                }
            }else{
                for(unsigned z=0;z<n;z+=4)for(unsigned y=0;y<n;y+=4)for(unsigned x=0;x<n;x+=4){
                    std::array<double,4> sums{};
                    for(unsigned k=z;k<z+4;k++)for(unsigned j=y;j<y+4;j++)for(unsigned i=x;i<x+4;i++){auto b=adaptive_grid::index(i,j,k,n);if(!zero[b])for(unsigned c=0;c<B;c++)sums[c]+=cost[b][c];}
                    auto winner=std::min_element(sums.begin(),sums.begin()+B)-sums.begin();
                    for(unsigned k=z;k<z+4;k++)for(unsigned j=y;j<y+4;j++)for(unsigned i=x;i<x+4;i++){auto b=adaptive_grid::index(i,j,k,n);next[b]=zero[b]?B:winner;}
                }pack(next);
            }
            stats[5]++;if(next==working)break;working=std::move(next);
        }
        if(!best_regions.empty()){
            stats[7]=1+*std::max_element(best_regions.begin(),best_regions.end());
            // Concrete lower-bound witness: homogeneous8³ entropy regions
            // which retain more than one predictor ID inside them.
            for(unsigned z=0;z<n;z+=4)for(unsigned y=0;y<n;y+=4)for(unsigned x=0;x<n;x+=4){
                auto b=adaptive_grid::index(x,y,z,n);bool same=true,mixed=false;
                for(unsigned k=z;k<z+4;k++)for(unsigned j=y;j<y+4;j++)for(unsigned i=x;i<x+4;i++){auto c=adaptive_grid::index(i,j,k,n);same&=best_regions[c]==best_regions[b];mixed|=predictor[c]!=predictor[b];}
                stats[6]+=same&&mixed;
            }
        }
        stats[1]=best.size();return best;
}
}
static int encode_regions_impl(const uint8_t*src,unsigned d,unsigned B,unsigned adaptive,unsigned passes,uint8_t*out,size_t cap,size_t*written,bool force_shared=false){
    three_candidate::LegacySlowScope legacy;
    using namespace region_entropy;
    if(!src||!out||!written||d<32||d>128||d%32||(B!=2&&B!=4)||adaptive>1||passes<1||passes>8||capture||choice_frame::observer||!choice_frame::region_banks.empty())return -1;
    Reset reset;
    try{
        GeometryScope geom(d,d,d);choice_grid::Scope grid(4);adaptive_grid::Scope tree(0);
        adaptive_grid::s2_root=adaptive_grid::s2_options=0;
        require(ng_codec_encode_size_search(src,d,1,3,out,cap,written)==0);
        Bytes best(out,out+*written);reference=best;Input in;capture=&in;choice_frame::observer=observe;
        ng_choice_edge=size_search::selected;require(ng_codec_encode(src,d,1,out,cap,written)==0);
        choice_frame::observer=nullptr;capture=nullptr;require(in.edge&&best==Bytes(out,out+*written));
        std::fill_n(stats,8,0);stats[0]=stats[1]=best.size();stats[2]=in.edge;
        if(in.symbols.empty())return 0;
        auto pack=[&](const Bytes&map){choice_frame::region_banks=map;return choice_frame::write(d,in.models,in.menu,in.choices,in.symbols,in.thr,in.classes);};
        if(force_shared){
            best=pack(Bytes(size_t(d/2)*(d/2)*(d/2)));bool used[60]={};for(auto c:in.choices)used[c]=true;
            stats[6]=std::count(std::begin(used),std::end(used),true);stats[1]=best.size();
        }else best=refine(d,in,B,adaptive,passes,std::move(best),pack);
        require(best.size()<=cap);memcpy(out,best.data(),best.size());*written=best.size();return 0;
    }catch(...){return -1;}
}
extern "C" void ng_region_entropy_stats(uint64_t*out){std::copy_n(region_entropy::stats,8,out);}
extern "C" void ng_region_entropy_seed(unsigned value){region_entropy::seed_policy=value;}
extern "C" int ng_codec_encode_regions(const uint8_t*s,unsigned d,unsigned b,unsigned a,unsigned p,uint8_t*out,size_t cap,size_t*n){return encode_regions_impl(s,d,b,a,p,out,cap,n);}
extern "C" int ng_codec_test_shared_region(const uint8_t*s,unsigned d,uint8_t*out,size_t cap,size_t*n){return encode_regions_impl(s,d,2,0,1,out,cap,n,true);}
