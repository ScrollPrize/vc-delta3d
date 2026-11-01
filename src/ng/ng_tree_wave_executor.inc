    NG_WAVE_ATTR static int NG_WAVE_NAME(void*p,ng_route_wave_data*w){
        try{
            auto&a=*static_cast<Decoder*>(p);auto*cols=static_cast<ng_bank_wave_column*>(w->columns);
            const unsigned pipeline=NG_WAVE_PIPELINE;
#ifdef NG_WAVE_PROFILE
            const bool profiling=true;
#else
            const bool profiling=NG_WAVE_STATIC_PATH?false:profile_enabled;
#endif
            double*timings=wave_parts;
            const unsigned replay=replay_mode;
            const unsigned d=w->width,n=NG_WAVE_STATIC_PATH?32:w->lanes,t=w->ys[0]+w->delta*w->zs[0];
            bool simd=false,wide=false;
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
            wide=NG_WAVE_STATIC_WIDE||(!ng_disable_avx512&&__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512bw")&&__builtin_cpu_supports("avx512vl")&&__builtin_cpu_supports("popcnt"));
            simd=NG_WAVE_STATIC_PATH||(vector_enabled&&a.vector&&!(a.vector->used&~511u)&&__builtin_cpu_supports("avx2"));
#endif
            const bool prepared=simd&&pipeline>=4;
            const bool compact=NG_WAVE_STATIC_PATH||w->compact_step;
            ng_context_rans local_entropy;void*entropy=a.entropy.get();
            if(NG_WAVE_STATIC_PATH&&entropy){local_entropy=*static_cast<ng_context_rans*>(entropy);entropy=&local_entropy;}
            const unsigned feature_stride=pipeline==12?2:9,predictor_slot=pipeline==12?1:8;
            if(prepared)prepare_tree_columns(a,a.s,*a.vector,a.masks.get(),w,pipeline,a.G);
            uint8_t*here_base=w->history+(t%w->history_depth)*w->history_plane+w->zs[0]+1;
            const uint8_t*north_base=w->history+((t-1+w->history_depth)%w->history_depth)*w->history_plane+w->zs[0]+1;
            const uint8_t*back_base=w->history+((t-w->delta+w->history_depth)%w->history_depth)*w->history_plane+w->zs[0];
            const uint8_t*nb_base=w->history+((t-w->delta-1+w->history_depth)%w->history_depth)*w->history_plane+w->zs[0];
            const size_t stride=w->history_stride;
            alignas(64) int scratch_features[9*32]={};
            alignas(64) uint16_t local_previous[3][32];
            uint16_t(*previous)[32]=pipeline>=15?local_previous:w->previous;
            if(pipeline>=15)memcpy(previous,w->previous,sizeof(local_previous));
            if(a.alternative.kind==17){a.prepare_learned(w);a.routed_errors.resize(d*32);}
            for(unsigned x=0;x<d;x++){
                const bool zero_column=prepared&&pipeline>=16&&a.routed_allzero[x/choice_grid::edge()];
                const bool measured=profiling&&((x&15)==3);auto tick=std::chrono::steady_clock::time_point{};
                if(measured)tick=std::chrono::steady_clock::now();
                auto mark=[&](unsigned k){if(measured){auto now=std::chrono::steady_clock::now();timings[k]+=16*std::chrono::duration<double>(now-tick).count();tick=now;}};
                size_t indices[32];unsigned classes[32];uint8_t symbols[32]={};
                uint8_t*here=here_base+size_t(x)*stride;
                const uint8_t*north=north_base+size_t(x)*stride;
                const uint8_t*back=back_base+size_t(x)*stride;
                const uint8_t*nb=nb_base+size_t(x)*stride;
                const uint8_t*neighbors[8]={here-(x?w->history_stride:0),here-(x>1?2*w->history_stride:0),north,back,north-(x?w->history_stride:0),back-(x?w->history_stride:0),nb,nb-(x?w->history_stride:0)};
                mark(0);
                if(a.alternative.kind==17){
                    uint8_t groups[32];
                    for(unsigned j=0;j<n;j++){
                        unsigned i=j*d+x,bank=a.learned_banks[i];groups[j]=bank*a.G;
                        if(!a.learned_zero[i]){auto&m=a.alternative.score;const auto&weights=m.current_weights[bank];int score=a.learned_scores[i];
                            for(unsigned k=0;k<3;k++){
                                int c=weights[2*k],b=weights[2*k+1];int v=x>k?int(previous[k][j]):128;
                                score+=c*(v-128)+b*std::abs(v-int(a.learned_north[i]));}
                            groups[j]=m.score_group(score,bank);
                        }
                    }
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
                    if(n==32&&wide)require(context_entropy_wide(entropy,groups,symbols)==0);else
#endif
                    require(ng_context_rans_read(entropy,n,groups,symbols,2)==0);
                    for(unsigned j=0;j<n;j++)if(a.learned_zero[j*d+x])require(symbols[j]==0);
                }else if(prepared){
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
                    int*features=pipeline==12?scratch_features:a.routed_features.data()+x*9*32;int leaves[32];
                    if(pipeline==12){memcpy(features,a.routed_features.data()+x*2*32,128);memcpy(features+8*32,a.routed_features.data()+(x*2+1)*32,128);}
                    if(replay==2){size_t start=replay_cursor;require(start+n<=replay_leaves.size());const uint8_t*cached=replay_leaves.data()+start;for(unsigned j=0;j<n;j++)leaves[j]=cached[j];replay_cursor=start+n;}
                    else if(zero_column){std::fill_n(leaves,32,0);if(replay==1)replay_leaves.insert(replay_leaves.end(),n,0);}
                    else{
                    if(pipeline>=14&&wide)a.vector->make_features512(n,x,w->ys,w->zs,neighbors,features);
                    else a.vector->make_features(n,x,w->ys,w->zs,neighbors,features);mark(2);
                    if(a.masks&&pipeline>=8&&wide)a.masks->classify512(n,features,a.routed_roots.data()+x*32,leaves,(pipeline==11||pipeline==12)?a.routed_masks.data()+x*32:nullptr,pipeline>=14);
                    else if(a.masks)a.masks->classify(n,features,a.routed_roots.data()+x*32,leaves);
                    else if(pipeline==6)a.vector->classify_packed_software(n,features,a.routed_roots.data()+x*32,leaves);
                    else if(wide)a.vector->classify_packed512(n,features,a.routed_roots.data()+x*32,leaves);
                    else a.vector->classify_packed(n,features,a.routed_roots.data()+x*32,leaves);
                    mark(3);
                    if(replay==1)for(unsigned j=0;j<n;j++)replay_leaves.push_back(leaves[j]);
                    }
                    if(a.entropy){uint8_t groups[32];
                        if((acceleration&64)&&wide&&n==32)context_groups(a.routed_groups.data()+x*32,leaves,groups);
                        else for(unsigned j=0;j<n;j++)groups[j]=a.routed_groups[x*32+j]+leaves[j];
                        if(pipeline>=13&&n==32&&wide&&a.compact_entropy)require(context_entropy_compact(entropy,groups,symbols,a.compact_entropy.get())==0);
                        else if(pipeline>=13&&n==32&&wide)require(context_entropy_wide(entropy,groups,symbols)==0);
                        else require(ng_context_rans_read(entropy,n,groups,symbols,2)==0);}
                    else if(pipeline>=18){
                        uint8_t groups[32];for(unsigned j=0;j<n;j++)groups[j]=a.routed_groups[x*32+j]+leaves[j];
                        unsigned j=0;
                        for(;j+8<=n;j+=8){
                            uint64_t ids;memcpy(&ids,groups+j,8);
                            if(ids==uint64_t(groups[j])*UINT64_C(0x0101010101010101)){
                                auto&cursor=a.cursors[groups[j]];require(cursor.end-cursor.next>=8);memcpy(symbols+j,cursor.next,8);cursor.next+=8;
                            }else for(unsigned k=j;k<j+8;k++){
                                auto&cursor=a.cursors[groups[k]];require(cursor.next!=cursor.end);symbols[k]=*cursor.next++;
                            }
                        }
                        for(;j<n;j++){auto&cursor=a.cursors[groups[j]];require(cursor.next!=cursor.end);symbols[j]=*cursor.next++;}
                    }else for(unsigned j=0;j<n;j++){
                        auto&cursor=a.cursors[a.routed_groups[x*32+j]+leaves[j]];
                        if(cursor.next==cursor.end)throw std::runtime_error("exhausted prepared entropy leaf");symbols[j]=*cursor.next++;
                    }
                    for(unsigned j=0;j<n;j++)if(a.routed_zeros[x*32+j]&&symbols[j])throw std::runtime_error("nonzero prepared zero");
                    mark(4);
#endif
                }else{
                    for(unsigned j=0;j<n;j++){indices[j]=w->offsets[j]+x;classes[j]=w->class_lut[w->activity[j*d+x]];}
                    if(batch(p,n,indices,classes,x,w->ys,w->zs,neighbors,symbols))return -1;
                    mark(1);
                }
                if(zero_column){
                    memset(a.routed_errors.data()+x*32,0,32);memset(w->values+x*32,0,32);memset(here,0,32);
                    // A2-wide zero choice does not erase the third causal tap.
                    // Advance history per voxel for experimental grids; the
                    // existing4-wide shortcut is safe once its block ends.
                    if(choice_grid::edge()!=4){
                        memcpy(previous[2],previous[1],32*sizeof(uint16_t));
                        memcpy(previous[1],previous[0],32*sizeof(uint16_t));
                        memset(previous[0],0,32*sizeof(uint16_t));
                    }else if(!(x&3))memset(previous,0,3*32*sizeof(uint16_t));
                    if(w->raw)for(unsigned j=0;j<n;j++)w->raw[w->offsets[j]+x]=0;
                    continue;
                }
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
                if(pipeline>=20&&compact&&(prepared||a.alternative.kind==17)&&wide&&!w->raw){
                    ng_route_compact_symbols(static_cast<ng_bank_compact_column*>(w->columns)+x,w->block_weights[x/ng_choice_edge],previous,w->values+x*32,symbols,a.routed_errors.data()+x*32);
                    memcpy(here,w->values+x*32,32);mark(6);continue;
                }
#endif
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
                if(prepared&&pipeline>=9&&wide){
                    ng_route_symbols512(symbols,cols+x,compact?static_cast<ng_bank_compact_column*>(w->columns)+x:nullptr,a.routed_errors.data()+x*32);
                    if(w->raw)for(unsigned j=0;j<n;j++)w->raw[w->offsets[j]+x]=uint8_t((symbols[j]>>1)^-(symbols[j]&1));
                }else
#endif
                for(unsigned j=0;j<n;j++){
                    unsigned s=symbols[j];uint8_t residual=uint8_t((s>>1)^-(s&1));
                    if(compact){auto&c=static_cast<ng_bank_compact_column*>(w->columns)[x];c.flags[j]=(c.flags[j]&255)|(unsigned(residual)<<8);c.residual[j]=residual;}
                    else cols[x].residual[j]=residual;
                    if(prepared)a.routed_errors[x*32+j]=(s+1)>>1;
                    else w->errors[w->error_offsets[j]+x]=(s+1)>>1;
                    if(w->raw)w->raw[w->offsets[j]+x]=residual;
                }
                mark(5);
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
                if(compact&&pipeline>=13&&wide)ng_route_compact_wide(static_cast<ng_bank_compact_column*>(w->columns)+x,w->block_weights[x/ng_choice_edge],previous,w->values+x*32);else
#endif
                if(compact)w->compact_step(static_cast<ng_bank_compact_column*>(w->columns)+x,w->block_weights[x/ng_choice_edge],previous,w->values+x*32);else
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
                if(simd&&pipeline>=3&&wide)ng_bank_wave_avx512(1,cols+x,previous,w->values+x*32);
                else if(simd)ng_bank_wave_avx2(1,cols+x,previous,w->values+x*32);else
#endif
                ng_bank_wave_c(1,cols+x,previous,w->values+x*32);
                mark(6);
                memcpy(here,w->values+x*32,32);
                if(!simd&&a.alternative.kind!=17)for(unsigned j=0;j<n;j++)a.out[w->offsets[j]+x]=w->values[x*32+j];
                mark(7);
            }
            if(prepared||(a.alternative.kind==17&&compact&&wide&&!w->raw&&pipeline>=20))for(unsigned j=0;j<n;j++)for(unsigned x=0;x<d;x++)w->errors[w->error_offsets[j]+x]=a.routed_errors[x*32+j];
            if(pipeline>=15)memcpy(w->previous,previous,sizeof(local_previous));
            if(NG_WAVE_STATIC_PATH&&a.entropy)*static_cast<ng_context_rans*>(a.entropy.get())=local_entropy;
            return 0;
        }catch(...){return -1;}
    }
