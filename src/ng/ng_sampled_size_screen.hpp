// Cheap fixed-size evidence only. No NG fits, rANS or Zstd calls. Sampled
// directional residuals are proxies, not substitutes for the final encoder.
// Every sample uses ORIGINAL-volume causal neighbours (no stitched borders).
extern "C" int ng_codec_size_features(const uint8_t*src,unsigned d,unsigned preset,
                                     unsigned samples,unsigned seed,double*out){
    try{
        require(src&&out&&d>=32&&d%32==0&&(preset==1||preset==3)&&samples>=1&&samples<=8);
        constexpr unsigned side=32,M=12,K=4;constexpr size_t S=side*side*side,B=16*16*16;
        unsigned nr=d/side;require(samples<=size_t(nr)*nr*nr);
        std::vector<unsigned>regions(size_t(nr)*nr*nr);std::iota(regions.begin(),regions.end(),0);
        uint32_t state=seed?seed:1;
        for(size_t i=regions.size();i>1;i--){state^=state<<13;state^=state>>17;state^=state<<5;std::swap(regions[i-1],regions[state%i]);}
        std::vector<Bytes>sym(M,Bytes(samples*S));Bytes cls(samples*S);
        std::vector<uint8_t>empty(samples*B,1);
        std::array<std::array<uint32_t,256>,M*K>hist{};
        for(unsigned q=0;q<samples;q++){
            unsigned r=regions[q],ox=(r%nr)*side,oy=(r/nr%nr)*side,oz=(r/(nr*nr))*side;
            for(unsigned z=0;z<side;z++)for(unsigned y=0;y<side;y++)for(unsigned x=0;x<side;x++){
                unsigned xx=x+ox,yy=y+oy,zz=z+oz;size_t i=(size_t(zz)*d+yy)*d+xx,j=q*S+(z*side+y)*side+x;
                int v=src[i],a=xx?src[i-1]:v,b=yy?src[i-d]:v,c=zz?src[i-size_t(d)*d]:v;
                unsigned cl=std::min(3,(std::abs(a-b)+std::abs(a-c)+std::abs(b-c))/16);cls[j]=cl;
                if(v)empty[q*B+((z/2)*16+y/2)*16+x/2]=0;
                for(unsigned m=0;m<M;m++){
                    auto s=pyramid(uint8_t(v-ng_directional_base(src,i,m,xx,yy,zz,d)));
                    sym[m][j]=s;hist[m*K+cl][s]++;
                }
            }
        }
        std::array<std::array<double,256>,M*K>cost;
        for(unsigned k=0;k<M*K;k++){
            double total=128;for(auto n:hist[k])total+=n;
            for(unsigned s=0;s<256;s++)cost[k][s]=std::log2(total/(hist[k][s]+.5));
        }
        using C=std::array<double,M>;std::vector<C>sums(samples*B);
        for(unsigned q=0;q<samples;q++)for(unsigned z=0;z<side;z++)for(unsigned y=0;y<side;y++)for(unsigned x=0;x<side;x++){
            size_t j=q*S+(z*side+y)*side+x,b=q*B+((z/2)*16+y/2)*16+x/2;
            for(unsigned m=0;m<M;m++)sums[b][m]+=cost[m*K+cls[j]][sym[m][j]];
        }
        C totals{};for(size_t b=0;b<sums.size();b++)if(!empty[b])for(unsigned m=0;m<M;m++)totals[m]+=sums[b][m];
        std::array<unsigned,M>rank;std::iota(rank.begin(),rank.end(),0);
        std::stable_sort(rank.begin(),rank.end(),[&](unsigned a,unsigned b){return totals[a]<totals[b];});
        double scale=double(d)*d*d/(samples*S);unsigned nn=16;
        for(unsigned level=0,e=2;e<=32;level++,e*=2){
            size_t per=size_t(nn)*nn*nn;Bytes chosen(sums.size());std::array<size_t,M+1>counts{};
            double residual=0,transitions=0,zeros=0,unequal=0;
            for(size_t b=0;b<sums.size();b++){
                unsigned id=rank[0];
                if(empty[b]){id=M;zeros++;}
                else {
                    // Fast is uniform. Slow evidence permits three globally
                    // strong bare predictors; real Slow still fits NG later.
                    if(preset==1)for(unsigned k=1;k<3;k++)if(sums[b][rank[k]]<sums[b][id])id=rank[k];
                    residual+=sums[b][id];
                }
                chosen[b]=id;counts[id]++;
                size_t local=b%per;unsigned x=local%nn,y=local/nn%nn,z=local/(nn*nn);
                for(size_t step:{size_t(1),size_t(nn),size_t(nn)*nn}){
                    bool valid=step==1?x>0:step==nn?y>0:z>0;
                    if(valid){transitions++;unequal+=chosen[b]!=chosen[b-step];}
                }
            }
            double entropy=0;for(auto n:counts)if(n)entropy-=n*std::log2(double(n)/chosen.size());
            double* f=out+level*7;
            f[0]=residual*scale/8;f[1]=entropy*scale/8;
            f[2]=unequal/std::max(1.,transitions)*chosen.size()*scale/8;
            f[3]=zeros*e*e*e*scale;f[4]=chosen.size()*scale;
            f[5]=std::log2(double(e)/4)*double(d)*d*d;
            f[6]=totals[rank[0]]*scale/8;
            if(e==32)break;
            unsigned next=nn/2;size_t np=size_t(next)*next*next;
            std::vector<C>merged(samples*np);std::vector<uint8_t>ze(samples*np,1);
            for(unsigned q=0;q<samples;q++)for(unsigned z=0;z<nn;z++)for(unsigned y=0;y<nn;y++)for(unsigned x=0;x<nn;x++){
                size_t a=q*per+(z*nn+y)*nn+x,b=q*np+((z/2)*next+y/2)*next+x/2;
                for(unsigned m=0;m<M;m++)merged[b][m]+=sums[a][m];ze[b]&=empty[a];
            }
            sums=std::move(merged);empty=std::move(ze);nn=next;
        }
        return 0;
    }catch(...){return -1;}
}
