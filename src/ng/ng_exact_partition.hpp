// Compression-first search, sharing the production evaluator and frame writer.
namespace exact_partition {
thread_local bool enabled=false,contexts=false,coordinated=false;
thread_local unsigned budget=64,sweeps=2;
thread_local Bytes reference;
thread_local uint64_t stats[10]={};
thread_local std::vector<uint64_t> accepted;

template<class Evaluate,class Costs,class Thresholds,class Pack>
void run(unsigned d,Bytes&choices,const zero_blocks::Scan&zeros,Bytes&frame,
         std::vector<uint16_t>&thr,Evaluate evaluate,Costs costs,Thresholds thresholds,Pack pack){
    const unsigned n=d/2;
    auto empty=[&](size_t b){return zeros.any()&&zeros.empty[b];};
    auto record=[&](Bytes next){frame=std::move(next);accepted.push_back(frame.size());};
    auto context_search=[&](){
        if(!contexts)return;
        // Preserve all16 difficulty bins. A short map shares arbitrary
        // predictor/size groups; no per-leaf size ID is transmitted.
        for(unsigned pass=0;pass<2;pass++){
            Bytes labels=choices,levels(choices.size());
            if(zeros.any())for(size_t b=0;b<labels.size();b++)if(empty(b))labels[b]=3;
            adaptive_grid::serialize(labels,d,&levels);
            unsigned base=3+unsigned(zeros.any());Bytes initial(5*base),used(5*base);
            for(unsigned c=0;c<initial.size();c++)initial[c]=c/5;
            for(size_t b=0;b<labels.size();b++)used[5*labels[b]+levels[b]]=1;
            if(!choice_frame::context_map.empty())initial=choice_frame::context_map;
            unsigned groups=1+*std::max_element(initial.begin(),initial.end());
            std::vector<Bytes>proposals;
            for(unsigned a=0;a<groups;a++)for(unsigned b=a+1;b<groups;b++){
                auto p=initial;for(auto&v:p)if(v==b)v=a;proposals.push_back(std::move(p));
            }
            if(groups<8)for(unsigned c=0;c<initial.size();c++)if(used[c]){
                unsigned peers=0;for(unsigned j=0;j<used.size();j++)peers+=used[j]&&initial[j]==initial[c];
                if(peers>1){auto p=initial;p[c]=groups;proposals.push_back(std::move(p));}
            }
            // Empty map is ordinary sharing and avoids map-signaling overhead.
            proposals.push_back({});size_t before=frame.size();auto start=choice_frame::context_map,best=start;
            for(auto p:proposals){
                if(!p.empty()){
                    unsigned next=0;int remap[16];std::fill_n(remap,16,-1);
                    for(auto&v:p){if(remap[v]<0)remap[v]=next++;v=remap[v];}
                }
                if(p==start)continue;choice_frame::context_map=p;stats[5]++;
                auto f=pack(thr);if(f.size()<frame.size()){record(std::move(f));best=p;stats[6]++;}
            }
            choice_frame::context_map=std::move(best);
            if(frame.size()==before)break;
        }
    };
    context_search();
    for(unsigned sweep=0;sweep<sweeps&&(budget||coordinated);sweep++){
        size_t before=frame.size();
        if(coordinated){
            // Joint changes can amortize the fixed-to-tree signaling cost.
            // Reuse the existing bottom-up partitioner, including explicit
            // zero leaves so its topology matches the actual writer.
            evaluate(false);auto residual=costs();
            std::vector<std::array<double,4>> c(residual.size());
            Bytes labels=choices;for(size_t b=0;b<c.size();b++){
                for(unsigned k=0;k<3;k++)c[b][k]=empty(b)?1e30:residual[b][k];
                c[b][3]=empty(b)?0.:1e30;if(empty(b))labels[b]=3;
            }
            auto tokens=adaptive_grid::serialize(labels,d);double counts[256]={},base[256]={};
            for(auto t:tokens)counts[t]++;
            for(unsigned k:{0u,1u,2u,3u,255u})base[k]=-std::log2((counts[k]+.5)/(tokens.size()+2.5));
            for(double scale:{.125,.25,.5,1.,2.,4.,8.,16.}){
                for(unsigned k:{0u,1u,2u,3u,255u})adaptive_grid::s2_cost[k]=scale*base[k];
                auto old=choices;auto old_thr=thr;adaptive_grid::select(d,c,choices);
                for(size_t b=0;b<choices.size();b++)if(empty(b))choices[b]=old[b];
                if(choices==old)continue;stats[9]++;evaluate(true);thr=thresholds(true);auto f=pack(thr);
                if(f.size()<frame.size())record(std::move(f));
                else{choices=std::move(old);thr=std::move(old_thr);evaluate(true);}
            }
        }
        if(budget){
        evaluate(false);auto cost=costs();
        struct Patch{unsigned x,y,z,w,kind;double gain;std::array<uint8_t,8> child{};size_t order;};
        std::vector<Patch>patches;
        for(unsigned w: {1u,2u,4u,8u,16u})for(unsigned z=0;z<n;z+=w)for(unsigned y=0;y<n;y+=w)for(unsigned x=0;x<n;x+=w){
            std::array<double,3>sum{};std::array<std::array<double,3>,8>child{};double current=0;unsigned count[3]={};
            for(unsigned k=z;k<z+w;k++)for(unsigned j=y;j<y+w;j++)for(unsigned i=x;i<x+w;i++){
                size_t b=adaptive_grid::index(i,j,k,n);if(empty(b))continue;
                unsigned a=w>1?unsigned(i-x>=w/2)+2*unsigned(j-y>=w/2)+4*unsigned(k-z>=w/2):0;
                current+=cost[b][choices[b]];count[choices[b]]++;
                for(unsigned c=0;c<3;c++){sum[c]+=cost[b][c];child[a][c]+=cost[b][c];}
            }
            unsigned total=count[0]+count[1]+count[2];if(!total)continue;
            for(unsigned c=0;c<3;c++)if(count[c]!=total)patches.push_back({x,y,z,w,c,current-sum[c],{},patches.size()});
            if(w>1){Patch p{x,y,z,w,3,current,{},patches.size()};
                for(unsigned a=0;a<8;a++){p.child[a]=std::min_element(child[a].begin(),child[a].end())-child[a].begin();p.gain-=child[a][p.child[a]];}
                patches.push_back(p);
            }
        }
        size_t limit=std::min(size_t(budget),patches.size());
        std::partial_sort(patches.begin(),patches.begin()+limit,patches.end(),[](auto&a,auto&b){return a.gain!=b.gain?a.gain>b.gain:a.order<b.order;});
        for(size_t t=0;t<limit;t++){
            const auto&p=patches[t];auto old=choices;auto old_thr=thr;
            for(unsigned z=p.z;z<p.z+p.w;z++)for(unsigned y=p.y;y<p.y+p.w;y++)for(unsigned x=p.x;x<p.x+p.w;x++){
                size_t b=adaptive_grid::index(x,y,z,n);if(empty(b))continue;
                unsigned a=p.kind==3?unsigned(x-p.x>=p.w/2)+2*unsigned(y-p.y>=p.w/2)+4*unsigned(z-p.z>=p.w/2):0;
                choices[b]=p.kind==3?p.child[a]:p.kind;
            }
            if(choices==old)continue;stats[3]++;evaluate(true);thr=thresholds(true);auto f=pack(thr);
            if(f.size()<frame.size()){record(std::move(f));stats[4]++;}
            else {choices=std::move(old);thr=std::move(old_thr);evaluate(true);}
        }
        }
        context_search();stats[8]++;
        if(frame.size()==before)break;
    }
    stats[1]=frame.size();stats[7]=choice_frame::context_map.empty()?3+zeros.any():1+*std::max_element(choice_frame::context_map.begin(),choice_frame::context_map.end());
}
}
