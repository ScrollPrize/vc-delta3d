// Experimental octree choice selection. Models and residual reconstruction
// are shared with the existing presets; only the selector partition changes.
namespace adaptive_grid {
thread_local unsigned root=0;
thread_local double node_bits=8;
thread_local uint64_t leaves[6]={};
// S2 is opt-in. Costs are learned from the incumbent tree tokens; exact
// compressed frame size, not this surrogate, decides whether to accept it.
thread_local unsigned s2_root=0,s2_min=2,s2_rounds=2;
thread_local unsigned s2_options=0;
thread_local double s2_cost[256]={};
thread_local uint64_t s2_stats[8]={};
struct Scope { unsigned old=root;double old_bits=node_bits; explicit Scope(unsigned r,double bits=8){root=r;node_bits=bits;} ~Scope(){root=old;node_bits=old_bits;} };
size_t index(unsigned x,unsigned y,unsigned z,unsigned n){return (size_t(z)*n+y)*n+x;}
void fill(Bytes&out,unsigned n,unsigned x,unsigned y,unsigned z,unsigned w,uint8_t c){
    for(unsigned k=z;k<z+w;k++)for(unsigned j=y;j<y+w;j++)
        std::fill_n(out.begin()+index(x,j,k,n),w,c);
}
// Bottom-up sufficient costs, then top-down emit. No repeated voxel scans or
// fitting per node. One leaf byte versus one split byte plus eight children;
// these are conservative pre-compression metadata costs, not an exact oracle.
template<size_t K> void select(unsigned d,const std::vector<std::array<double,K>>&cost,Bytes&out,const std::vector<std::vector<std::array<double,K>>>*by_size=nullptr){
    require(root&&choice_grid::edge()==2&&d%root==0);
    struct Level {unsigned n;std::vector<std::array<double,K>> sums;std::vector<double> best;Bytes pick;};
    std::vector<Level> levels;
    levels.push_back({d/2,cost,{}, {}});
    for(unsigned w=2;;w*=2){
        auto&l=levels.back();l.best.resize(l.sums.size());l.pick.resize(l.sums.size());
        if(by_size){require(levels.size()<=by_size->size());l.sums=by_size->at(levels.size()-1);}
        for(unsigned z=0;z<l.n;z++)for(unsigned y=0;y<l.n;y++)for(unsigned x=0;x<l.n;x++){
            size_t i=index(x,y,z,l.n);auto&s=l.sums[i];unsigned p=std::min_element(s.begin(),s.end())-s.begin();
            if(s2_root)for(unsigned k=0;k<K;k++)if(s[k]+s2_cost[k]<s[p]+s2_cost[p])p=k;
            l.best[i]=s[p]+(s2_root?s2_cost[p]:node_bits);l.pick[i]=p;
            if(s2_root&&w<s2_min)l.best[i]=std::numeric_limits<double>::infinity();
            if(levels.size()>1){auto&child=levels[levels.size()-2];double split=s2_root?s2_cost[255]:node_bits;
                for(unsigned a=0;a<8;a++)split+=child.best[index(2*x+(a&1),2*y+((a>>1)&1),2*z+(a>>2),child.n)];
                if(split<l.best[i]){l.best[i]=split;l.pick[i]=255;}
            }
        }
        if(w==root)break;
        Level next{l.n/2,{}, {}, {}};next.sums.resize(size_t(next.n)*next.n*next.n);
        for(unsigned z=0;z<next.n;z++)for(unsigned y=0;y<next.n;y++)for(unsigned x=0;x<next.n;x++){
            auto&s=next.sums[index(x,y,z,next.n)];
            for(unsigned a=0;a<8;a++){auto&t=l.sums[index(2*x+(a&1),2*y+((a>>1)&1),2*z+(a>>2),l.n)];for(unsigned k=0;k<K;k++)s[k]+=t[k];}
        }levels.push_back(std::move(next));
    }
    auto emit=[&](auto&&self,unsigned level,unsigned x,unsigned y,unsigned z)->void{
        auto&l=levels[level];unsigned p=l.pick[index(x,y,z,l.n)];
        if(p!=255){fill(out,d/2,x<<level,y<<level,z<<level,1u<<level,p);return;}
        require(level>0);for(unsigned a=0;a<8;a++)self(self,level-1,2*x+(a&1),2*y+((a>>1)&1),2*z+(a>>2));
    };
    auto&top=levels.back();for(unsigned z=0;z<top.n;z++)for(unsigned y=0;y<top.n;y++)for(unsigned x=0;x<top.n;x++)emit(emit,levels.size()-1,x,y,z);
}
Bytes serialize(const Bytes&selectors,unsigned d,Bytes*leaf_levels=nullptr){
    require(root&&choice_grid::edge()==2&&d%root==0);std::fill_n(leaves,6,0);Bytes tree;unsigned n=d/2;
    auto visit=[&](auto&&self,unsigned x,unsigned y,unsigned z,unsigned w)->void{
        uint8_t c=selectors[index(x,y,z,n)];bool same=true;
        for(unsigned k=z;k<z+w&&same;k++)for(unsigned j=y;j<y+w&&same;j++)
            for(unsigned i=x;i<x+w;i++)if(selectors[index(i,j,k,n)]!=c){same=false;break;}
        if(same){tree.push_back(c);unsigned l=0;for(unsigned q=w;q>1;q/=2)l++;leaves[l]++;if(leaf_levels)fill(*leaf_levels,n,x,y,z,w,l);return;}
        require(w>1);tree.push_back(255);w/=2;for(unsigned a=0;a<8;a++)self(self,x+(a&1)*w,y+((a>>1)&1)*w,z+(a>>2)*w,w);
    };
    for(unsigned z=0;z<n;z+=root/2)for(unsigned y=0;y<n;y+=root/2)for(unsigned x=0;x<n;x+=root/2)visit(visit,x,y,z,root/2);
    return tree;
}
Bytes deserialize(const Bytes&tree,unsigned d,unsigned r,unsigned nc){
    require((r==8||r==16||r==32)&&d%r==0&&choice_grid::edge()==2);
    Bytes out(choice_grid::count(d));unsigned n=d/2;size_t pos=0;
    auto visit=[&](auto&&self,unsigned x,unsigned y,unsigned z,unsigned w)->void{
        require(pos<tree.size());unsigned c=tree[pos++];
        if(c!=255){require(c<nc);fill(out,n,x,y,z,w,c);return;}
        require(w>1);w/=2;for(unsigned a=0;a<8;a++)self(self,x+(a&1)*w,y+((a>>1)&1)*w,z+(a>>2)*w,w);
    };
    for(unsigned z=0;z<n;z+=r/2)for(unsigned y=0;y<n;y+=r/2)for(unsigned x=0;x<n;x+=r/2)visit(visit,x,y,z,r/2);
    require(pos==tree.size());return out;
}
}
