/* Experimental dense spatial stencil. Every tap is visited, including zeros;
 * SIMD is across voxels, offsets are derived from the rectangular geometry.
 * Same modulo-65536 arithmetic as the selected-tap kernels. In-row taps still
 * belong to causal reconstruction, not this previous-row preparation. */
#define NG_DENSE_TAPS(BODY) do { \
  int k=pl->dense_r[0]; \
  for(int dz=-2;dz<=0;dz++){ \
    int r=pl->dense_r[-dz];if(r<0)continue; \
    for(int dy=-r;dy<=(dz?r:-1);dy++)for(int dx=-r;dx<=r;dx++){ \
      const uint8_t*s=row+(ptrdiff_t)dz*VCN_LC_SP+(ptrdiff_t)dy*VCN_LC_RP+dx; \
      uint16_t w=pl->dense_w[k++];BODY; \
    } \
  } \
} while(0)
/* The default 3/3/1 geometry has79 previous-row taps. Expand its offsets at
 * compile time; four independent accumulation chains hide multiply latency.
 * No selected-tap indices or zero-weight tests appear in this path. */
#define NG_DENSE_ROW3(Z,Y) NG_DENSE_TAP(Z,Y,-1) NG_DENSE_TAP(Z,Y,0) NG_DENSE_TAP(Z,Y,1)
#define NG_DENSE_ROW7(Z,Y) NG_DENSE_TAP(Z,Y,-3) NG_DENSE_TAP(Z,Y,-2) NG_DENSE_ROW3(Z,Y) NG_DENSE_TAP(Z,Y,2) NG_DENSE_TAP(Z,Y,3)
#define NG_DENSE_FIXED \
  NG_DENSE_ROW3(-2,-1) NG_DENSE_ROW3(-2,0) NG_DENSE_ROW3(-2,1) \
  NG_DENSE_ROW7(-1,-3) NG_DENSE_ROW7(-1,-2) NG_DENSE_ROW7(-1,-1) NG_DENSE_ROW7(-1,0) NG_DENSE_ROW7(-1,1) NG_DENSE_ROW7(-1,2) NG_DENSE_ROW7(-1,3) \
  NG_DENSE_ROW7(0,-3) NG_DENSE_ROW7(0,-2) NG_DENSE_ROW7(0,-1)
static void ng_dense_pre_scalar(const uint8_t*row,const vcn_lc_plist*pl,uint16_t bias,uint16_t*pre,int count){
  for(int x=0;x<count;x++)pre[x]=bias;
  NG_DENSE_TAPS(for(int x=0;x<count;x++)pre[x]=(uint16_t)(pre[x]+(uint32_t)w*s[x]));
}
#if VCN_HAVE_AVX2
__attribute__((target("avx2")))
static void ng_dense_pre_256(const uint8_t*row,const vcn_lc_plist*pl,uint16_t bias,uint16_t*pre,unsigned needed){
  for(int x=0;x<VCN_LC_D;x+=16){
    if(!(needed&(1u<<(x>>5)))){memset(pre+x,0,32);continue;}
    __m256i a[4]={_mm256_set1_epi16((short)bias),_mm256_setzero_si256(),_mm256_setzero_si256(),_mm256_setzero_si256()};
    if(pl->dense_r[0]==3&&pl->dense_r[1]==3&&pl->dense_r[2]==1){
      int k=3;
#define NG_DENSE_TAP(Z,Y,X) do { \
      __m256i v=_mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(row+x+(ptrdiff_t)(Z)*VCN_LC_SP+(ptrdiff_t)(Y)*VCN_LC_RP+(X)))); \
      a[k&3]=_mm256_add_epi16(a[k&3],_mm256_mullo_epi16(v,_mm256_set1_epi16((short)pl->dense_w[k])));k++; \
    } while(0);
      NG_DENSE_FIXED
#undef NG_DENSE_TAP
    }else NG_DENSE_TAPS(a[0]=_mm256_add_epi16(a[0],_mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(s+x))),_mm256_set1_epi16((short)w))));
    _mm256_storeu_si256((__m256i*)(pre+x),_mm256_add_epi16(_mm256_add_epi16(a[0],a[1]),_mm256_add_epi16(a[2],a[3])));
  }
}
#if (defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 11) || (defined(__clang__) && __clang_major__ >= 12)
#define NG_HAVE_DENSE_WIDE 1
__attribute__((target("avx512f,avx512bw")))
static void ng_dense_pre_512(const uint8_t*row,const vcn_lc_plist*pl,uint16_t bias,uint16_t*pre,unsigned needed){
  for(int x=0;x<VCN_LC_D;x+=32){
    if(!(needed&(1u<<(x>>5)))){memset(pre+x,0,64);continue;}
    __m512i a[4]={_mm512_set1_epi16((short)bias),_mm512_setzero_si512(),_mm512_setzero_si512(),_mm512_setzero_si512()};
    if(pl->dense_r[0]==3&&pl->dense_r[1]==3&&pl->dense_r[2]==1){
      int k=3;
#define NG_DENSE_TAP(Z,Y,X) do { \
      __m512i v=_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(row+x+(ptrdiff_t)(Z)*VCN_LC_SP+(ptrdiff_t)(Y)*VCN_LC_RP+(X)))); \
      a[k&3]=_mm512_add_epi16(a[k&3],_mm512_mullo_epi16(v,_mm512_set1_epi16((short)pl->dense_w[k])));k++; \
    } while(0);
      NG_DENSE_FIXED
#undef NG_DENSE_TAP
    }else NG_DENSE_TAPS(a[0]=_mm512_add_epi16(a[0],_mm512_mullo_epi16(_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(s+x))),_mm512_set1_epi16((short)w))));
    _mm512_storeu_si512(pre+x,_mm512_add_epi16(_mm512_add_epi16(a[0],a[1]),_mm512_add_epi16(a[2],a[3])));
  }
}
#endif
#endif
static void ng_dense_pre_dispatch(const uint8_t*row,const vcn_lc_plist*pl,uint16_t bias,uint16_t*pre,unsigned needed){
#if VCN_HAVE_AVX2
#ifdef NG_HAVE_DENSE_WIDE
  if(ng_dense_pre!=3&&ng_use_avx512()&&__builtin_cpu_supports("avx512bw")){ng_dense_pre_512(row,pl,bias,pre,needed);return;}
#endif
  if(ng_dense_pre!=3&&vcn_use_avx2()){ng_dense_pre_256(row,pl,bias,pre,needed);return;}
#endif
  for(int x=0;x<VCN_LC_D;x+=32){
    if(needed&(1u<<(x>>5)))ng_dense_pre_scalar(row+x,pl,bias,pre+x,32);
    else memset(pre+x,0,64);
  }
}
#undef NG_DENSE_TAPS
#undef NG_DENSE_FIXED
#undef NG_DENSE_ROW7
#undef NG_DENSE_ROW3
