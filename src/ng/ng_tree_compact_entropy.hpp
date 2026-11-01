// Exact two-level representation of the decoded rANS tables. The serialized
// frequencies and entropy payload are unchanged.
struct TreeCompactEntropy {
    std::vector<uint8_t> symbols;
    std::vector<uint32_t> parameters;
    explicit TreeCompactEntropy(const ng_context_rans&c):symbols((size_t(c.K)<<c.d.tb)+3),parameters(size_t(c.K)*256){
        const unsigned M=1u<<c.d.tb;
        for(unsigned g=0;g<c.K;g++)for(unsigned slot=0;slot<M;slot++){
            uint32_t e=c.d.T[size_t(g)*M+slot];unsigned s=e&255,fm=(e>>8)&4095,offset=e>>20;
            symbols[size_t(g)*M+slot]=s;parameters[g*256+s]=fm|((slot-offset)<<12);
        }
    }
};
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define VCN_LC_RL (1u<<15)
#define NG_EATTR __attribute__((target("avx512f,avx512bw,avx512vl,popcnt"),always_inline))
#define NG_ENAME context_entropy_compact_impl
#define NG_EUNROLL 1
#define NG_EEXTRA ,const TreeCompactEntropy*compact
#define NG_ELOOKUP(INDEX,C,S) tree_compact_entry(compact,INDEX,C,S,tb,kM)
__attribute__((target("avx2"),always_inline)) static inline __m256i tree_compact_entry(const TreeCompactEntropy*c,__m256i index,__m256i bank,__m256i state,int tb,__m256i mask){
    auto sy=_mm256_and_si256(_mm256_i32gather_epi32((const int*)c->symbols.data(),index,1),_mm256_set1_epi32(255));
    auto b=_mm256_srl_epi32(bank,_mm_cvtsi32_si128(tb-8));
    auto p=_mm256_i32gather_epi32((const int*)c->parameters.data(),_mm256_add_epi32(b,sy),4);
    auto offset=_mm256_sub_epi32(_mm256_and_si256(state,mask),_mm256_srli_epi32(p,12));
    return _mm256_or_si256(sy,_mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(p,_mm256_set1_epi32(4095)),8),_mm256_slli_epi32(offset,20)));
}
#include "ng_entropy_vector.h"
#undef VCN_LC_RL
__attribute__((target("avx512f,avx512bw,avx512vl,popcnt"))) static int context_entropy_compact(void*p,const uint8_t*groups,uint8_t*out,const TreeCompactEntropy*c){
    auto*d=static_cast<ng_context_rans*>(p);context_entropy_compact_impl(32,groups,&d->d,out,1,c);return d->d.over?-1:0;
}
#endif
