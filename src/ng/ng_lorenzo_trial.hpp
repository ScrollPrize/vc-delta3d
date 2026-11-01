/* Included after shared activity helpers. Full native NG (two spatial banks,
 * class biases and fractional-prediction sign mapping), plus optional Lorenzo
 * and a separately fitted native NG correction relative to Lorenzo. */
extern "C" int ng_activity_refined_encode(const uint8_t*,int,unsigned,uint8_t*,size_t,size_t*);
extern "C" int ng_activity_refined_decode(const uint8_t*,size_t,uint8_t*,size_t);
static int ng_lorenzo_encode_impl(const uint8_t*src,unsigned d,unsigned options,uint8_t*out,size_t cap,size_t*written,uint64_t*stats,const Bytes* cached_incumbent){
    try{
        require(src&&out&&written&&ng_geometry_valid(d,d,d)&&options<512&&(!(options&2)||(options&1))&&((options&48)!=48));
        GeometryScope geometry(d,d,d);const int cfg[6]={3,3,1,16,40,128};
        const size_t voxels=N;BlockHandoff handoff;auto original=encode(src,cfg,504,0,false,false,&handoff);
        Bytes baseline={'N','D','L','1'};put32(baseline,d);put32(baseline,d);put32(baseline,d);baseline.insert(baseline.end(),original.begin(),original.end());
        unsigned P=((options&2)?10:(options&1)?9:8)+!!(options&64)+!!(options&128),nb=d/4;size_t blocks=size_t(nb)*nb*nb;
        unsigned clamp=(options&128)?P-1:99,direct=(options&64)?P-1-!!(options&128):99;int32_t feature_weights[2]={0,0};
        Bytes incumbent,old_choices,old_seed,old_payload;std::vector<uint16_t>old_thr;
        if(options&256){
            size_t bytes=0;
            if(cached_incumbent){incumbent=*cached_incumbent;bytes=incumbent.size();}
            else {incumbent.resize(N*3+131072);
                require(ng_activity_refined_encode(src,d,13,incumbent.data(),incumbent.size(),&bytes)==0);incumbent.resize(bytes);}
            size_t p=8;old_choices.resize(blocks);
            if(incumbent[2]=='C'){auto m=get32(incumbent,p);std::fill(old_choices.begin(),old_choices.end(),m);require(get32(incumbent,p)==16);}
            else if(incumbent[2]=='B'){require(get32(incumbent,p)==4&&get32(incumbent,p)==16&&get32(incumbent,p)==blocks);auto n=get32(incumbent,p);unpack_stream(incumbent.data()+p,n,old_choices,true);p+=n;}
            else{require(incumbent[2]=='D');get32(incumbent,p);require(get32(incumbent,p)==1);auto n=get32(incumbent,p);unpack_stream(incumbent.data()+p,n,old_choices,true);p+=n;}
            old_payload.assign(incumbent.begin()+p,incumbent.end());
            old_thr.resize(15);for(auto&t:old_thr){require(p+2<=bytes);t=incumbent[p]|unsigned(incumbent[p+1])<<8;p+=2;}
        }
        Bytes params(65536),params2,classes(N),symbols(N),raw(N),lorenzo(N);
        std::vector<uint16_t>base(N),base2,activity(N);std::vector<uint32_t>block(N);
        std::vector<Bytes>candidates(P,Bytes(N));std::vector<const uint8_t*>ptr(P);
        std::vector<Bytes>fixed(12,Bytes(N));bank_fixed_candidates(src,fixed,d);
        if(options&(2|48))for(size_t i=0;i<voxels;i++)lorenzo[i]=uint8_t(src[i]-unpyramid(fixed[11][i]));
        for(unsigned m=1;m<std::min(P,9u);m++)candidates[m]=std::move(fixed[m==8?11:m]);
        if(options&64)candidates[direct]=std::move(fixed[0]);
        for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++)for(unsigned x=0;x<d;x++){
            size_t i=(size_t(z)*d+y)*d+x;block[i]=((z/4)*nb+y/4)*nb+x/4;
        }
        size_t pn=0;
        if(options&48)require(ng_feature_prepare(src,cfg,lorenzo.data(),(options&16)?3:1,feature_weights,params.data(),params.size(),&pn,candidates[0].data(),classes.data(),base.data())==0);
        else require(ng_ex_prepare(src,cfg,0,0,params.data(),params.size(),&pn,candidates[0].data(),classes.data(),base.data())==0);
        params.resize(pn);if(!(options&48))require(params==handoff.params);remove_class_bias(base,params,classes,16);
        if(options&2){params2.resize(65536);base2.resize(N);Bytes cl2(N);size_t n2=0;
            require(ng_base_prepare(src,cfg,lorenzo.data(),params2.data(),params2.size(),&n2,candidates[9].data(),cl2.data(),base2.data())==0);
            params2.resize(n2);remove_class_bias(base2,params2,cl2,16);}
        for(unsigned m=0;m<P;m++)ptr[m]=candidates[m].data();
        if(options&128)require(ng_multi_clamp_symbols(params.data(),params.size(),params2.empty()?nullptr:params2.data(),params2.size(),handoff.choices.data(),src,
            base.data(),base2.empty()?nullptr:base2.data(),ptr.data(),symbols.data(),classes.data(),activity.data(),candidates[0].data(),(options&2)?candidates[9].data():nullptr,clamp,candidates[clamp].data())==0);
        std::vector<uint16_t>native_thr(15);size_t tp=params.size()-62;
        for(auto&t:native_thr){t=params[tp]|unsigned(params[tp+1])<<8;tp+=2;}
        std::vector<Bytes>seeds{handoff.choices,choose_blocks(ptr.data(),P,classes.data(),16,4)};
        // Only import representable palettes. A no-Lorenzo ablation cannot
        // reproduce a Lorenzo block; keep the complete incumbent nevertheless.
        if(!old_choices.empty()&&(options&64)&&((options&1)||std::find(old_choices.begin(),old_choices.end(),8)==old_choices.end())){
            old_seed=old_choices;for(auto&m:old_seed)if(m==0)m=direct;seeds.push_back(old_seed);
        }
        Bytes best_trial,best_choices;
        for(auto sel:seeds)for(unsigned pass=0;pass<((options&4)?3u:1u);pass++){
            require(ng_multi_clamp_symbols(params.data(),params.size(),params2.empty()?nullptr:params2.data(),params2.size(),sel.data(),src,
                base.data(),base2.empty()?nullptr:base2.data(),ptr.data(),symbols.data(),classes.data(),activity.data(),candidates[0].data(),(options&2)?candidates[9].data():nullptr,clamp,(options&128)?candidates[clamp].data():nullptr)==0);
            for(size_t i=0;i<voxels;i++)raw[i]=unpyramid(symbols[i]);
            ActivityHistogram trial_histogram(raw,activity);
            auto disk_choices=sel;Bytes bits;
            if(options&128){unsigned count=0;for(auto&m:disk_choices)if(m==0||m==clamp){if(count%8==0)bits.push_back(0);if(m==clamp){bits.back()|=1u<<(count%8);m=0;}count++;}}
            auto selector=pack_stream(disk_choices,true);
            const size_t prefix_size=16+selector.size()+bits.size()+4+params.size()+
                ((options&(48|64|128))?(4+((options&48)?8:0)):0)+((options&2)?4+params2.size():0);
            auto evaluate=[&](const std::vector<uint16_t>&thr,size_t selection_score=SIZE_MAX){
                bool audit=!old_seed.empty()&&sel==old_seed&&thr==old_thr;
                if(!(options&4))selection_score=0;
                if(encoder_bound_enabled&&!audit&&selection_score!=SIZE_MAX&&!best_trial.empty()){
                    size_t bound=prefix_size+trial_histogram.lower_bound(thr);
                    if(encoder_work_enabled)encoder_work[10]++;
                    if(bound>=selection_score&&bound>=best_trial.size()){
                        if(encoder_work_enabled)encoder_work[11]++;
                        return SIZE_MAX;
                    }
                }
                auto entropy_classes=activity_classify(activity,thr);
                auto payload=activity_stream_payload(raw,thr,entropy_classes);
                if(!old_seed.empty()&&sel==old_seed&&thr==old_thr)require(payload==old_payload);
                Bytes f={'N','A','L',uint8_t(options&(48|64|128)?'2':'1')};put32(f,d);put32(f,P);
                if(options&(48|64|128)){put32(f,options&(1|2|48|64|128));if(options&48)for(auto w:feature_weights)put32(f,uint32_t(w));}
                put32(f,selector.size());f.insert(f.end(),selector.begin(),selector.end());
                f.insert(f.end(),bits.begin(),bits.end());
                put32(f,params.size());f.insert(f.end(),params.begin(),params.end());
                if(options&2){put32(f,params2.size());f.insert(f.end(),params2.begin(),params2.end());}
                f.insert(f.end(),payload.begin(),payload.end());size_t size=f.size();
                if(best_trial.empty()||size<best_trial.size()){best_trial=std::move(f);best_choices=sel;}return size;
            };
            auto thr=native_thr;size_t score=evaluate(thr);
            if(!old_thr.empty()){auto s=evaluate(old_thr,score);if(s<score){score=s;thr=old_thr;}}
            if(options&4){auto quantiles=activity_quantiles(activity,16);auto qs=evaluate(quantiles,score);if(qs<score){score=qs;thr=quantiles;}
                auto tuned=activity_refine(raw,activity,thr,block,sel,1);if(evaluate(tuned,pass==2?0:score)<score)thr=tuned;}
            if((options&4)&&pass<2)activity_reselect(symbols,activity_classify(activity,thr),block,ptr,sel,1);
        }
        bool improved=best_trial.size()<baseline.size();
        if(stats){std::fill(stats,stats+15,0);stats[0]=baseline.size();stats[1]=best_trial.size();stats[2]=improved;for(auto m:best_choices)stats[3+m]++;}
        auto*best=(improved||(options&8))?&best_trial:&baseline;
        if(!(options&8)&&!incumbent.empty()&&incumbent.size()<best->size())best=&incumbent;
        require(best->size()<=cap);std::memcpy(out,best->data(),best->size());*written=best->size();return 0;
    }catch(...){return -1;}
}
extern "C" int ng_lorenzo_encode(const uint8_t*src,unsigned d,unsigned options,uint8_t*out,size_t cap,size_t*written,uint64_t*stats){
    return ng_lorenzo_encode_impl(src,d,options,out,cap,written,stats,nullptr);
}
extern "C" int ng_lorenzo_decode(const uint8_t*src,size_t n,uint8_t*out,size_t cap){
    if(src&&n>=4&&!std::memcmp(src,"NDL1",4))return ng_leaf_decode(src,n,out,cap,0);
    if(src&&n>=4&&(!std::memcmp(src,"NAD1",4)||!std::memcmp(src,"NAB1",4)||!std::memcmp(src,"NAC1",4)))return ng_activity_refined_decode(src,n,out,cap);
    try{
        require(src&&out&&n>=16&&(!std::memcmp(src,"NAL1",4)||!std::memcmp(src,"NAL2",4)));Bytes frame(src,src+n);size_t pos=4;
        unsigned d=get32(frame,pos),P=get32(frame,pos),options=P==10?3:P==9?1:0;int32_t feature_weights[2]={0,0};
        if(src[3]=='2'){options=get32(frame,pos);require(!(options&~(1|2|48|64|128))&&(options&48)!=48&&(!(options&2)||(options&1)));
            if(options&48)for(auto&w:feature_weights){uint32_t v=get32(frame,pos);w=v<=INT32_MAX?int32_t(v):int32_t(int64_t(v)-4294967296LL);}}
        unsigned sn=get32(frame,pos),clamp=(options&128)?P-1:99,direct=(options&64)?P-1-!!(options&128):99;
        require(ng_geometry_valid(d,d,d)&&P==((options&2)?10u:(options&1)?9u:8u)+!!(options&64)+!!(options&128)&&cap>=size_t(d)*d*d&&sn<=n-pos);
        GeometryScope geometry(d,d,d);unsigned nb=d/4;Bytes choices(size_t(nb)*nb*nb);unpack_stream(src+pos,sn,choices,true);pos+=sn;
        for(auto m:choices)require(m<P);
        if(options&128){unsigned count=0;for(auto&m:choices){require(m!=clamp);if(m==0){require(pos+count/8<n);if(src[pos+count/8]&(1u<<(count%8)))m=clamp;count++;}}pos+=(count+7)/8;}
        using State=std::unique_ptr<void,decltype(&ng_causal_destroy)>;State state(nullptr,ng_causal_destroy),state2(nullptr,ng_causal_destroy);
        auto read_state=[&](State&s){auto pn=get32(frame,pos);require(pn<=n-pos);s.reset(ng_causal_create(src+pos,pn));require(bool(s));pos+=pn;};
        read_state(state);if(options&2)read_state(state2);
        Streams streams;auto thr=activity_parse(frame,pos,16,1,streams);
        require(ng_activity_decode(thr.data(),16,0,Streams::read,&streams,out)==0);
        for(size_t c=0;c<streams.pos.size();c++)require(streams.pos[c]==streams.symbols[c].size());
        for(unsigned z=0;z<d;z++)for(unsigned y=0;y<d;y++)for(unsigned x=0;x<d;x++){
            size_t i=(size_t(z)*d+y)*d+x;unsigned m=choices[((z/4)*nb+y/4)*nb+x/4];uint16_t q=0,q2=0;uint8_t cl=0;
            require(ng_causal_prediction(state.get(),i,&q,&cl)==0);
            if(options&48)q=uint16_t(q+(int64_t)feature_weights[z==0]*activity_prediction(out,i,8,x,y,z,d));
            if(m==clamp)q=ng_clamp_prediction(q,out,i,x,y,z,d);
            if(state2)require(ng_causal_prediction(state2.get(),i,&q2,&cl)==0);
            uint8_t residual=out[i],pred;
            if(m==direct)pred=0;
            else if(m==0||m==clamp||(m==9&&(options&2))){if(m==9&&(options&2))q=uint16_t(q2+256u*activity_prediction(out,i,8,x,y,z,d));
                pred=uint8_t(uint16_t(q+128u)>>8);if(q&128)residual=uint8_t(-residual);}
            else pred=activity_prediction(out,i,m,x,y,z,d);
            out[i]=uint8_t(pred+residual);
            require(ng_causal_accept(state.get(),i,out[i],pred)==0);
            if(state2)require(ng_causal_accept(state2.get(),i,out[i],pred)==0);
        }
        return 0;
    }catch(...){return -1;}
}
