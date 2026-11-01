    NG_BULK_ATTR static int NG_BULK_NAME(void*p,ng_route_wave_data*w){
        try{
            auto&a=*static_cast<Decoder*>(p);unsigned d=w->width,n=w->lanes,used=a.vector?a.vector->used:0;
            auto tick=std::chrono::steady_clock::now();
            a.bulk_groups.resize(d*n);a.bulk_symbols.resize(d*n);
            bool wide=false,simd=false;
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
            simd=vector_enabled&&__builtin_cpu_supports("avx2");
            wide=simd&&!ng_disable_avx512&&__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512cd");
#endif
            if(a.alternative.kind>=16){
                a.prepare_learned(w);
                for(unsigned i=0;i<n*d;i++)a.bulk_groups[i]=a.learned_zero[i]?a.learned_banks[i]*a.G:a.alternative.score.score_group(a.learned_scores[i],a.learned_banks[i]);
            }else if(a.alternative.kind){
                for(unsigned j=0;j<n;j++){
                    unsigned y=w->ys[j],z=w->zs[j];size_t offset=w->offsets[j];
                    const uint8_t*select=a.s.selectors.data()+(size_t(z/ng_choice_edge)*(d/ng_choice_edge)+y/ng_choice_edge)*(d/ng_choice_edge);
                    unsigned shift=a.alternative.side==8?3:2,q=d>>shift;
                    size_t block_row=(size_t(z>>shift)*q+(y>>shift))*q;
                    const uint16_t*activity=w->activity+j*d;
                    for(unsigned bx=0;bx<d/ng_choice_edge;bx++){
                        const auto&c=a.s.menu[select[bx]];
                        if(c.predictor==choice_frame::zero_predictor||a.alternative.kind>=12){
                            unsigned group=c.entropy*a.G;
                            if(c.predictor!=choice_frame::zero_predictor){
                                for(unsigned x=ng_choice_edge*bx;x<ng_choice_edge*(bx+1);x++){
                                    unsigned bank=a.alternative.map[size_t(c.entropy)*q*q*q+block_row+(x>>shift)];
                                    a.bulk_groups[j*d+x]=group+(a.alternative.kind>=14?bank*a.K+w->class_lut[activity[x]]:bank);
                                }
                                continue;
                            }
                            memset(a.bulk_groups.data()+j*d+bx*ng_choice_edge,group,ng_choice_edge);continue;
                        }
                        const uint8_t*map=a.alternative.map.data()+c.entropy*(a.alternative.kind==10?512:128);
                        for(unsigned x=ng_choice_edge*bx;x<ng_choice_edge*(bx+1);x++){
                            int north=y?a.out[offset-d+x]:128,back=z?a.out[offset-size_t(d)*d+x]:128;
                            unsigned cls=w->class_lut[activity[x]],coarse=a.coarse_classes[cls],cell;
                            if(a.alternative.kind==10)cell=((coarse*8+unsigned(north+back)/64)*16)+unsigned(north-back+255)/32;
                            else{const auto&v=a.alternative.weights;int score=v[0]+v[1]*(north+back)+v[2]*std::abs(north-back)+v[3]*int(cls);cell=coarse*32+(unsigned(std::clamp(score,0,31*256))>>8);}
                            a.bulk_groups[j*d+x]=c.entropy*16+map[cell];
                        }
                    }
                }
            }else for(unsigned j=0;j<n;j++){
                int y=w->ys[j],z=w->zs[j];size_t offset=w->offsets[j];
                // Padding matches the existing scalar feature definition (128).
                std::vector<int> neighbors(4*(d+2),128);
                int*nr=neighbors.data()+1,*br=nr+d+2,*nn=br+d+2,*bb=nn+d+2;
                for(unsigned x=0;x<d;x++){
                    if(y)nr[x]=a.out[offset-d+x];if(z)br[x]=a.out[offset-size_t(d)*d+x];
                    if((used&(1u<<13))&&y>1)nn[x]=a.out[offset-2*d+x];if((used&(1u<<14))&&z>1)bb[x]=a.out[offset-2*size_t(d)*d+x];
                }
                const uint8_t*select=a.s.selectors.data()+(size_t(z/ng_choice_edge)*(d/ng_choice_edge)+y/ng_choice_edge)*(d/ng_choice_edge);
                const uint16_t*activity=w->activity+j*d;const uint8_t*lut=w->class_lut;
                for(unsigned x0=0;x0<d;x0+=32){
                    alignas(64) int f[15*32],roots[32],leaves[32];unsigned bank[32];bool zero[32];
                    for(unsigned k=0;k<32;k++)f[k]=lut[activity[x0+k]];
                    for(unsigned block=0;block<32/ng_choice_edge;block++){
                        unsigned selected=select[x0/ng_choice_edge+block];const auto&c=a.s.menu[selected];
                        for(unsigned k=block*ng_choice_edge;k<ng_choice_edge*(block+1);k++){
                            zero[k]=c.predictor==choice_frame::zero_predictor;bank[k]=c.entropy*a.G;
                            roots[k]=zero[k]?a.vector->zero_root:a.vector->roots[c.entropy];f[8*32+k]=c.predictor;
                        }
                    }
                    if(used&(1u<<3))for(unsigned k=0;k<32;k++)f[3*32+k]=nr[x0+k]-br[x0+k];
                    if(used&(1u<<9))for(unsigned k=0;k<32;k++)f[9*32+k]=nr[x0+k]+br[x0+k];
                    if(used&(1u<<10))for(unsigned k=0;k<32;k++){int x=x0+k;f[10*32+k]=std::max({nr[x],br[x],nr[x-1],nr[x+1],br[x-1],br[x+1]})-std::min({nr[x],br[x],nr[x-1],nr[x+1],br[x-1],br[x+1]});}
                    if(used&(1u<<11))for(unsigned k=0;k<32;k++){int x=x0+k;f[11*32+k]=nr[x]-nr[x-1];}
                    if(used&(1u<<12))for(unsigned k=0;k<32;k++){int x=x0+k;f[12*32+k]=br[x]-br[x-1];}
                    if(used&(1u<<13))for(unsigned k=0;k<32;k++)f[13*32+k]=nr[x0+k]-nn[x0+k];
                    if(used&(1u<<14))for(unsigned k=0;k<32;k++)f[14*32+k]=br[x0+k]-bb[x0+k];
                    if(!a.row_lookup.empty())
                    for(unsigned k=0;k<32;k++){
                        unsigned x=x0+k,selected=select[x/ng_choice_edge];leaves[k]=zero[k]?0:a.row_lookup[(selected*a.K+f[k])*32+a.difference_bins[nr[x]-br[x]+255]];
                    }
                    if(a.row_lookup.empty()){
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
                        if(wide)a.masks->classify512(32,f,roots,leaves,nullptr,true);else
#endif
                        a.masks->classify(32,f,roots,leaves);
                    }
                    for(unsigned k=0;k<32;k++)a.bulk_groups[a.row_entropy?j*d+x0+k:(x0+k)*n+j]=uint8_t(bank[k]+leaves[k]);
                }
            }
            if(profile_enabled){auto now=std::chrono::steady_clock::now();bulk_profile[0]+=std::chrono::duration<double>(now-tick).count();tick=now;}
            if(a.entropy)require((a.row_entropy?ng_context_rans_read_rows(a.entropy.get(),n*d,a.bulk_groups.data(),a.bulk_symbols.data()):ng_context_rans_read_wave(a.entropy.get(),n,d,a.bulk_groups.data(),a.bulk_symbols.data()))==0);
            else for(unsigned i=0;i<d*n;i++){auto&c=a.cursors[a.bulk_groups[i]];require(c.next!=c.end);a.bulk_symbols[i]=*c.next++;}
            if(profile_enabled){auto now=std::chrono::steady_clock::now();bulk_profile[1]+=std::chrono::duration<double>(now-tick).count();tick=now;}
            for(unsigned j=0;j<n;j++){
                uint8_t*dst=a.out+w->offsets[j],*err=w->errors+w->error_offsets[j];const uint8_t*syms=a.bulk_symbols.data()+(a.row_entropy?j*d:j);
                if(a.row_entropy){for(unsigned x=0;x<d;x++){unsigned sy=syms[x];dst[x]=uint8_t((sy>>1)^-(sy&1));err[x]=(sy+1)>>1;}}
                else for(unsigned x=0;x<d;x++){unsigned sy=syms[x*n];dst[x]=uint8_t((sy>>1)^-(sy&1));err[x]=(sy+1)>>1;}
                const uint8_t*select=a.s.selectors.data()+(size_t(w->zs[j]/ng_choice_edge)*(d/ng_choice_edge)+w->ys[j]/ng_choice_edge)*(d/ng_choice_edge);
                for(unsigned bx=0;bx<d/ng_choice_edge;bx++)if(a.s.menu[select[bx]].predictor==choice_frame::zero_predictor)
                    for(unsigned k=0;k<ng_choice_edge;k++)require(dst[ng_choice_edge*bx+k]==0);
                if(w->raw)memcpy(w->raw+w->offsets[j],dst,d);
            }
            if(profile_enabled)bulk_profile[2]+=std::chrono::duration<double>(std::chrono::steady_clock::now()-tick).count();
            return 0;
        }catch(...){return -1;}
    }
