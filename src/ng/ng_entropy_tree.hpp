// Small per-chunk conditional residual distributions. Integer-only inference.
namespace entropy_tree {
constexpr unsigned features=15,bins=32;
constexpr unsigned row_features=265u|(63u<<9);
using Feature=std::array<uint8_t,features>;
// Only completed Y rows / Z planes. No source in the current X row.
inline int row_feature(const uint8_t*s,unsigned d,int x,int y,int z,unsigned f){
    auto at=[&](int dx,int dy,int dz){return int(geometry_trials::value(s,x+dx,y+dy,z+dz,d));};
    int n=at(0,-1,0),b=at(0,0,-1);
    switch(f){
    case 9:return n+b;
    case 10:{int a[]={n,b,at(-1,-1,0),at(1,-1,0),at(-1,0,-1),at(1,0,-1)};auto mm=std::minmax_element(a,a+6);return *mm.second-*mm.first;}
    case 11:return n-at(-1,-1,0);
    case 12:return b-at(-1,0,-1);
    case 13:return n-at(0,-2,0);
    case 14:return b-at(0,0,-2);
    default:throw std::runtime_error("invalid prior-row feature");
    }
}
inline unsigned magnitude(unsigned v){
    static constexpr unsigned limits[]={0,1,2,3,4,6,8,12,16,24,32,48,64,96,128,255};
    return std::lower_bound(std::begin(limits),std::end(limits),v)-std::begin(limits);
}
inline unsigned signed_bin(int v){return v<0?15-magnitude(unsigned(-v)):16+magnitude(unsigned(v));}
unsigned feature(const uint8_t*s,size_t i,unsigned d,unsigned cls,unsigned predictor,unsigned f){
    if(f==0)return cls;if(f==8)return predictor;
    int x=i%d,y=i/d%d,z=i/(d*d);
    if(f>=9){int v=row_feature(s,d,x,y,z,f);return f==9?unsigned(v)/16:f==10?magnitude(v):signed_bin(v);}
    int w=geometry_trials::value(s,x-1,y,z,d),n=geometry_trials::value(s,x,y-1,z,d),b=geometry_trials::value(s,x,y,z-1,d);
    switch(f){
    case 1:return signed_bin(w-n);
    case 2:return signed_bin(w-b);
    case 3:return signed_bin(n-b);
    case 4:return magnitude(std::max({w,n,b})-std::min({w,n,b}));
    case 5:return magnitude(std::abs(w-int(geometry_trials::value(s,x-2,y,z,d))));
    case 6:return signed_bin(int(ng_directional_base(s,i,11,x,y,z,d))-(w+n+b+1)/3);
    case 7:return unsigned(w+n+b)/24;
    default:throw std::runtime_error("invalid entropy tree feature");
    }
}
struct Node {unsigned feature=255,cut=0,left=0,right=0,leaf=0;};
struct Tree {
    std::vector<Node>nodes{Node{}};
    unsigned classify(const uint8_t*s,size_t i,unsigned d,unsigned cls,unsigned predictor)const{
        unsigned n=0;for(unsigned depth=0;depth<=8;depth++){
            const auto&v=nodes.at(n);if(v.feature==255)return v.leaf;
            n=feature(s,i,d,cls,predictor,v.feature)<=v.cut?v.left:v.right;
        }throw std::runtime_error("entropy tree depth");
    }
    void assign_leaves(){unsigned leaf=0;for(auto&n:nodes)if(n.feature==255)n.leaf=leaf++;}
};
using Forest=std::vector<Tree>;
// Compile quantized predicates into exact integer comparisons once per frame.
// This leaves the serialized tree and every leaf assignment unchanged.
struct CompiledTree {
    struct Node {int cut;uint16_t left,right,leaf;uint8_t feature;};
    std::vector<Node> nodes;
    static int threshold(unsigned f,unsigned c){
        static constexpr int limits[]={0,1,2,3,4,6,8,12,16,24,32,48,64,96,128,255};
        if(f==1||f==2||f==3||f==6||(f>=11&&f<=14)){
            if(c<15)return -limits[14-c]-1;
            if(c==15)return -1;
            return limits[c-16];
        }
        if(f==4||f==5||f==10)return limits[std::min(c,15u)];
        if(f==7)return 24*(c+1)-1;
        if(f==9)return 16*(c+1)-1;
        return c;
    }
    explicit CompiledTree(const Tree&t){
        for(auto&v:t.nodes)nodes.push_back({v.feature==255?0:threshold(v.feature,v.cut),uint16_t(v.left),uint16_t(v.right),uint16_t(v.leaf),uint8_t(v.feature)});
    }
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((always_inline))
#endif
    inline unsigned classify(const uint8_t*s,size_t i,unsigned d,unsigned x,unsigned y,unsigned z,unsigned cls,unsigned predictor)const{
        const int w=x?s[i-1]:128,n=y?s[i-d]:128,b=z?s[i-size_t(d)*d]:128;
        unsigned node=0;
        for(unsigned depth=0;depth<=8;depth++){
            const auto&v=nodes[node];int value;
            switch(v.feature){
            case 255:return v.leaf;
            case 0:value=cls;break;
            case 1:value=w-n;break;
            case 2:value=w-b;break;
            case 3:value=n-b;break;
            case 4:value=std::max({w,n,b})-std::min({w,n,b});break;
            case 5:value=std::abs(w-int(x>1?s[i-2]:128));break;
            case 6:value=int(ng_directional_base(s,i,11,x,y,z,d))-(w+n+b+1)/3;break;
            case 7:value=w+n+b;break;
            case 8:value=predictor;break;
            default:value=row_feature(s,d,x,y,z,v.feature);break;
            }
            node=value<=v.cut?v.left:v.right;
        }
        throw std::runtime_error("compiled tree depth");
    }
};
inline bool compiled_predicates_exact(){
    for(unsigned c=0;c<31;c++){
        for(int v=-255;v<=255;v++)if((signed_bin(v)<=c)!=(v<=CompiledTree::threshold(1,c)))return false;
        for(unsigned v=0;v<=255;v++)if((magnitude(v)<=c)!=(int(v)<=CompiledTree::threshold(4,c)))return false;
        for(unsigned v=0;v<=765;v++)if((v/24<=c)!=(int(v)<=CompiledTree::threshold(7,c)))return false;
    }
    return true;
}
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include "ng_gather.h"
#endif
// Structure-of-arrays execution plan for eight independent wavefront lanes.
// Scalar CompiledTree remains the portable fallback.
struct VectorForest {
    std::vector<int> meta,cuts;
    std::vector<int> packed;
    std::vector<unsigned> roots;
    unsigned used=0,zero_root=0;
    explicit VectorForest(const Forest&trees){
        for(auto&t:trees){unsigned base=meta.size();roots.push_back(base);
            for(auto&v:t.nodes){
                if(v.feature==255){meta.push_back(255);cuts.push_back(v.leaf);}
                else{used|=1u<<v.feature;meta.push_back(v.feature|((base+v.left)<<8)|((base+v.right)<<18));cuts.push_back(CompiledTree::threshold(v.feature,v.cut));}
                if(v.feature==255)packed.push_back(15|(v.leaf<<4));
                else{
                    int cut=CompiledTree::threshold(v.feature,v.cut);require(v.feature<15&&cut>=-256&&cut<=767);
                    packed.push_back(v.feature|((cut+256)<<4)|(v.left<<14)|(v.right<<21));
                }
            }
            unsigned size=(meta.size()+127)&~127u;meta.resize(size,255);cuts.resize(size,0);packed.resize(size,15);
        }
        zero_root=meta.size();meta.push_back(255);cuts.push_back(0);
        packed.push_back(15);
    }
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define NG_FATTR __attribute__((target("avx2")))
#define NG_FNAME make_features
#define NG_FV __m256i
#define NG_FSTEP 8
#define NG_FOP0 _mm256_setzero_si256
#define NG_FOP1 _mm256_set1_epi32
#define NG_FOP2 _mm256_cmpgt_epi32
#define NG_FOP3 _mm256_loadu_si256
#define NG_FOP4 _mm256_cvtepu8_epi32
#define NG_FOP5 _mm_loadl_epi64
#define NG_FOP6 _mm256_blendv_epi8
#define NG_FOP7 _mm256_min_epi32
#define NG_FOP8 _mm256_max_epi32
#define NG_FOP9 _mm256_storeu_si256
#define NG_FOP10 _mm256_sub_epi32
#define NG_FOP11 _mm256_abs_epi32
#define NG_FOP12 _mm256_add_epi32
#define NG_FOP13 _mm256_and_si256
#define NG_FOP14 _mm256_mulhi_epu16
#include "ng_tree_features_vector.inc"
#define NG_FATTR __attribute__((target("avx512f,avx512bw")))
#define NG_FNAME make_features512
#define NG_FV __m512i
#define NG_FSTEP 16
#define NG_FOP0 _mm512_setzero_si512
#define NG_FOP1 _mm512_set1_epi32
#define NG_FOP2(A,B) _mm512_maskz_set1_epi32(_mm512_cmpgt_epi32_mask(A,B),-1)
#define NG_FOP3 _mm512_loadu_si512
#define NG_FOP4 _mm512_cvtepu8_epi32
#define NG_FOP5 _mm_loadu_si128
#define NG_FOP6(A,B,M) _mm512_mask_blend_epi32(_mm512_cmpneq_epi32_mask(M,_mm512_setzero_si512()),A,B)
#define NG_FOP7 _mm512_min_epi32
#define NG_FOP8 _mm512_max_epi32
#define NG_FOP9 _mm512_storeu_si512
#define NG_FOP10 _mm512_sub_epi32
#define NG_FOP11 _mm512_abs_epi32
#define NG_FOP12 _mm512_add_epi32
#define NG_FOP13 _mm512_and_si512
#define NG_FOP14 _mm512_mulhi_epu16
#include "ng_tree_features_vector.inc"
    __attribute__((target("avx2")))
    void classify(unsigned count,const int*features,const int*start,int*leaves)const{
        for(unsigned j=0;j<count;j+=8){
            __m256i node=_mm256_loadu_si256((const __m256i*)(start+j));
            __m256i lane=_mm256_setr_epi32(j,j+1,j+2,j+3,j+4,j+5,j+6,j+7);
            __m256i result=_mm256_setzero_si256();
            for(unsigned depth=0;depth<=8;depth++){
                __m256i info=_mm256_i32gather_epi32(meta.data(),node,4);
                __m256i cut=_mm256_i32gather_epi32(cuts.data(),node,4);
                __m256i f=_mm256_and_si256(info,_mm256_set1_epi32(255));
                __m256i done=_mm256_cmpeq_epi32(f,_mm256_set1_epi32(255));
                result=_mm256_blendv_epi8(result,cut,done);
                if(_mm256_movemask_epi8(done)==-1)break;
                f=_mm256_andnot_si256(done,f);
                __m256i index=_mm256_add_epi32(_mm256_slli_epi32(f,5),lane);
                __m256i value=_mm256_i32gather_epi32(features,index,4);
                __m256i left=_mm256_and_si256(_mm256_srli_epi32(info,8),_mm256_set1_epi32(1023));
                __m256i right=_mm256_srli_epi32(info,18);
                __m256i next=_mm256_blendv_epi8(left,right,_mm256_cmpgt_epi32(value,cut));
                node=_mm256_blendv_epi8(next,node,done);
            }
            _mm256_storeu_si256((__m256i*)(leaves+j),result);
        }
    }
#define NG_TATTR __attribute__((target("avx2")))
#define NG_TNAME classify_packed
#define NG_TV __m256i
#define NG_TSTEP 8
#define NG_TSET _mm256_set1_epi32
#define NG_TLOAD(P) _mm256_loadu_si256((const __m256i*)(P))
#define NG_TSTORE(P,V) _mm256_storeu_si256((__m256i*)(P),V)
#define NG_TGATHER(P,I) _mm256_i32gather_epi32(P,I,4)
#define NG_TAND _mm256_and_si256
#define NG_TANDNOT _mm256_andnot_si256
#define NG_TOR _mm256_or_si256
#define NG_TSRL _mm256_srli_epi32
#define NG_TSLL _mm256_slli_epi32
#define NG_TADD _mm256_add_epi32
#define NG_TSUB _mm256_sub_epi32
#define NG_TEQ _mm256_cmpeq_epi32
#define NG_TGT _mm256_cmpgt_epi32
#define NG_TBLEND _mm256_blendv_epi8
#define NG_TALL(V) (_mm256_movemask_epi8(V)==-1)
#include "ng_tree_packed_vector.inc"
#undef NG_TNAME
#undef NG_TGATHER
#define NG_TNAME classify_packed_software
#define NG_TGATHER(P,I) vcn_lc_gather(P,I)
#include "ng_tree_packed_vector.inc"
#include "ng_tree_packed_undef.inc"
#define NG_TATTR __attribute__((target("avx512f")))
#define NG_TNAME classify_packed512
#define NG_TV __m512i
#define NG_TSTEP 16
#define NG_TSET _mm512_set1_epi32
#define NG_TLOAD(P) _mm512_loadu_si512((const void*)(P))
#define NG_TSTORE(P,V) _mm512_storeu_si512((void*)(P),V)
#define NG_TGATHER(P,I) _mm512_i32gather_epi32(I,P,4)
#define NG_TAND _mm512_and_si512
#define NG_TANDNOT _mm512_andnot_si512
#define NG_TOR _mm512_or_si512
#define NG_TSRL _mm512_srli_epi32
#define NG_TSLL _mm512_slli_epi32
#define NG_TADD _mm512_add_epi32
#define NG_TSUB _mm512_sub_epi32
#define NG_TEQ(A,B) _mm512_maskz_set1_epi32(_mm512_cmpeq_epi32_mask(A,B),-1)
#define NG_TGT(A,B) _mm512_maskz_set1_epi32(_mm512_cmpgt_epi32_mask(A,B),-1)
#define NG_TBLEND(A,B,M) _mm512_mask_blend_epi32(_mm512_cmpneq_epi32_mask(M,_mm512_setzero_si512()),A,B)
#define NG_TALL(V) (_mm512_cmpeq_epi32_mask(V,_mm512_set1_epi32(-1))==65535)
#include "ng_tree_packed_vector.inc"
#include "ng_tree_packed_undef.inc"
#endif
};
/* Exact tree compilation into per-feature allowed-leaf masks. Intersecting
 * these masks selects the same leaf without a serial node-to-node walk. */
struct MaskForest {
    static constexpr unsigned stride=1032;
    unsigned feature_count=9,bank_stride=9*stride;
    std::vector<uint64_t> table;
    std::vector<uint64_t> initial;
    unsigned used=0;
    explicit MaskForest(const Forest&forest){
        for(auto&t:forest)for(auto&n:t.nodes)if(n.feature!=255)feature_count=std::max(feature_count,n.feature+1);
        bank_stride=feature_count*stride;table.assign((forest.size()+1)*bank_stride,1);
        for(unsigned b=0;b<forest.size();b++){
            const auto&t=forest[b];std::vector<uint64_t>desc(t.nodes.size());
            std::vector<unsigned>parents(t.nodes.size());uint64_t seen=0;
            for(auto&n:t.nodes){if(n.feature==255){uint64_t bit=uint64_t(1)<<n.leaf;require(!(seen&bit));seen|=bit;}
                else {++parents[n.left];++parents[n.right];}}
            for(unsigned i=0;i<parents.size();i++)require(parents[i]==unsigned(i!=0));
            for(size_t i=t.nodes.size();i-->0;){auto&n=t.nodes[i];desc[i]=n.feature==255?(uint64_t(1)<<n.leaf):(desc[n.left]|desc[n.right]);}
            std::fill(table.begin()+b*bank_stride,table.begin()+(b+1)*bank_stride,desc[0]);
            for(auto&n:t.nodes)if(n.feature!=255){
                require(n.feature<features);used|=1u<<n.feature;int cut=CompiledTree::threshold(n.feature,n.cut);
                auto*p=table.data()+b*bank_stride+n.feature*stride;
                for(int v=-256;v<=767;v++)p[v+256]&=~(v<=cut?desc[n.right]:desc[n.left]);
            }
        }
        initial.resize((forest.size()+1)*32*13);
        for(unsigned b=0;b<=forest.size();b++)for(unsigned c=0;c<13;c++)for(unsigned k=0;k<32;k++)
            initial[(b*13+c)*32+k]=table[b*bank_stride+k+256]&table[b*bank_stride+8*stride+c+256];
    }
    void classify(unsigned count,const int*features,const int*roots,int*leaves)const{
        for(unsigned j=0;j<count;j++){
            const uint64_t*p=table.data()+(roots[j]/128)*bank_stride;uint64_t mask=~uint64_t(0);
            for(unsigned f=0;f<feature_count;f++)if(used&(1u<<f))mask&=p[f*stride+features[f*32+j]+256];
            if(!used)mask=p[256];leaves[j]=std::countr_zero(mask);
        }
    }
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    __attribute__((target("avx512f,avx512cd")))
    static void leaf_indices(__m512i mask,unsigned count,int*out){
        (void)count; // Tree validity/unique leaves checked once at compilation.
        __m512i index=_mm512_sub_epi64(_mm512_set1_epi64(63),_mm512_lzcnt_epi64(mask));
        _mm256_storeu_si256((__m256i*)out,_mm512_cvtepi64_epi32(index));
    }
    __attribute__((target("avx512f,avx2")))
    void classify512(unsigned count,const int*features,const int*roots,int*leaves,const uint64_t*prepared=nullptr,bool vector_indices=false)const{
        for(unsigned j=0;j<count;j+=8){
            __m256i base=_mm256_mullo_epi32(_mm256_srli_epi32(_mm256_loadu_si256((const __m256i*)(roots+j)),7),_mm256_set1_epi32(bank_stride));
            __m512i mask=prepared?_mm512_loadu_si512(prepared+j):_mm512_set1_epi64(-1);
            for(unsigned f=0;f<feature_count;f++)if((used&(prepared?~257u:~0u))&(1u<<f)){
                __m256i index=_mm256_add_epi32(base,_mm256_add_epi32(_mm256_loadu_si256((const __m256i*)(features+f*32+j)),_mm256_set1_epi32(f*stride+256)));
                mask=_mm512_and_si512(mask,_mm512_i32gather_epi64(index,table.data(),8));
            }
            if(!used)mask=_mm512_i32gather_epi64(_mm256_add_epi32(base,_mm256_set1_epi32(256)),table.data(),8);
            if(vector_indices&&__builtin_cpu_supports("avx512cd"))leaf_indices(mask,std::min(8u,count-j),leaves+j);
            else{uint64_t values[8];_mm512_storeu_si512(values,mask);
                for(unsigned k=0;k<8&&j+k<count;k++)leaves[j+k]=std::countr_zero(values[k]);}
        }
    }
#endif
};
struct Split {double gain=0;unsigned feature=0,cut=0;};
// Exact empirical entropy for screening split points, with a fixed 256-byte
// table/node charge per extra leaf and >=256 samples per child. Final output
// is measured with actual normalized rANS tables and all model metadata.
Split best_split(const std::vector<uint32_t>&ids,const std::vector<Feature>&x,const Bytes&sy,const std::vector<double>&nlogn,unsigned mask){
    if(ids.size()<512)return {};
    using Hist=std::array<uint32_t,256>;
    std::vector<Hist>hist(features*bins);Hist total{};
    for(auto i:ids){++total[sy[i]];for(unsigned f=0;f<features;f++)if(mask&(1u<<f))++hist[f*bins+x[i][f]][sy[i]];}
    double parent=nlogn[ids.size()];for(auto v:total)parent-=nlogn[v];Split best;
    for(unsigned f=0;f<features;f++){
        if(!(mask&(1u<<f)))continue;
        Hist left{};uint32_t ln=0;
        for(unsigned b=0;b<bins-1;b++){
            for(unsigned s=0;s<256;s++){left[s]+=hist[f*bins+b][s];ln+=hist[f*bins+b][s];}
            auto rn=ids.size()-ln;if(ln<256||rn<256)continue;
            double cost=nlogn[ln]+nlogn[rn]+2048.;
            for(unsigned s=0;s<256;s++)cost-=nlogn[left[s]]+nlogn[total[s]-left[s]];
            double gain=parent-cost;if(gain>best.gain)best={gain,f,b};
        }
    }return best;
}
std::array<Tree,3> fit(const std::vector<uint32_t>&ids,const std::vector<Feature>&x,const Bytes&sy,const std::vector<double>&nlogn,unsigned mask=511){
    struct Leaf{unsigned node,depth;std::vector<uint32_t>ids;Split split;};
    Tree tree;std::vector<Leaf>leaves;leaves.push_back({0,0,ids,best_split(ids,x,sy,nlogn,mask)});
    std::array<Tree,3>saved;
    for(unsigned stage=0;stage<3;stage++){
        unsigned limit=16u<<stage;
        while(leaves.size()<limit){
            auto it=std::max_element(leaves.begin(),leaves.end(),[](auto&a,auto&b){return a.split.gain<b.split.gain;});
            if(it==leaves.end()||it->split.gain<=0)break;
            Leaf parent=std::move(*it);leaves.erase(it);auto sp=parent.split;
            unsigned l=tree.nodes.size(),r=l+1;tree.nodes[parent.node]={sp.feature,sp.cut,l,r,0};tree.nodes.push_back({});tree.nodes.push_back({});
            std::vector<uint32_t>li,ri;li.reserve(parent.ids.size());ri.reserve(parent.ids.size());
            for(auto i:parent.ids)(x[i][sp.feature]<=sp.cut?li:ri).push_back(i);
            unsigned depth=parent.depth+1;
            auto ls=depth<8?best_split(li,x,sy,nlogn,mask):Split{},rs=depth<8?best_split(ri,x,sy,nlogn,mask):Split{};
            leaves.push_back({l,depth,std::move(li),ls});leaves.push_back({r,depth,std::move(ri),rs});
        }
        tree.assign_leaves();saved[stage]=tree;
    }return saved;
}
void write(Bytes&h,const Forest&forest){
    for(auto&t:forest){put32(h,t.nodes.size());for(auto&n:t.nodes){h.push_back(n.feature);h.push_back(n.cut);packed_bank::put16(h,n.left);packed_bank::put16(h,n.right);packed_bank::put16(h,n.leaf);}}
}
Forest read(packed_bank::Reader&h,unsigned banks,unsigned limit){
    Forest forest;
    for(unsigned b=0;b<banks;b++){
        unsigned count=h.u(4);require(count>=1&&count<=2*limit-1);Tree t;t.nodes.clear();
        for(unsigned i=0;i<count;i++){
            Node n;n.feature=h.u(1);n.cut=h.u(1);n.left=h.u(2);n.right=h.u(2);n.leaf=h.u(2);
            require(n.leaf<limit);if(n.feature!=255)require(n.feature<features&&n.cut<bins-1&&n.left>i&&n.right>i&&n.left<count&&n.right<count);
            t.nodes.push_back(n);
        }
        std::vector<unsigned>depth(count);
        for(unsigned i=0;i<count;i++){
            require(depth[i]<=8);auto&v=t.nodes[i];if(v.feature!=255){depth[v.left]=std::max(depth[v.left],depth[i]+1);depth[v.right]=std::max(depth[v.right],depth[i]+1);}
        }
        forest.push_back(std::move(t));
    }return forest;
}
}
