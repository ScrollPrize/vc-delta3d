// Same entropy-region optimizer as Slow; only the retained context and frame
// writer differ. RC3 keeps its entire learned forest and all64 leaf labels.
static int rc_encode_regions_impl(const uint8_t*src,unsigned d,unsigned config,unsigned banks,uint8_t*out,size_t cap,size_t*n,bool force=false){
    if(!src||!out||!n||config>2||d<32||d>128||d%32||banks<2||banks>(config==2?3u:4u)||!choice_frame::region_banks.empty())return -1;
    region_entropy::Reset reset;
    try{
        GeometryScope geom(d,d,d);choice_grid::Scope grid(4);adaptive_grid::Scope fixed(0);
        adaptive_grid::s2_root=adaptive_grid::s2_options=0;
        require(ng_codec_encode_size_search(src,d,8+config,3,out,cap,n)==0);
        Bytes baseline(out,out+*n);ng_choice_edge=size_search::selected;
        std::unique_ptr<hybrid_geometry::Trial> t(hybrid_geometry::create(src,d,1,config?4:0));
        require(hybrid_geometry::encode(*t,config==2?7:2)==baseline);
        auto&s=t->s;region_entropy::Input in{s.choice_edge,s.models,s.menu,s.selectors,s.symbols,s.classes,s.thresholds};
        if(config==2){
            in.class_count=64;in.classes.assign(N,0);
            for(size_t i=0;i<N;i++){
                auto&c=s.menu[s.selectors[geometry_trials::block(i,d)]];
                if(c.predictor!=choice_frame::zero_predictor)in.classes[i]=t->trees[2][c.entropy].classify(src,i,d,s.classes[i],c.predictor);
            }
            t->cached_leaf=in.classes;
            require(hybrid_geometry::encode(*t,7)==baseline); // cached routing is byte-exact
        }
        auto pack=[&](const Bytes&map){
            if(config==2){s.regions=map;return hybrid_geometry::encode(*t,7);}
            choice_frame::region_banks=map;return choice_frame::write(d,in.models,in.menu,in.choices,in.symbols,in.thr,in.classes);
        };
        Bytes result;
        if(force){Bytes map(size_t(d/2)*(d/2)*(d/2));for(size_t i=0;i<map.size();i++)map[i]=(i%(d/2))/4%banks;result=pack(map);}
        else result=region_entropy::refine(d,in,banks,1,4,std::move(baseline),pack);
        require(result.size()<=cap);std::copy(result.begin(),result.end(),out);*n=result.size();return 0;
    }catch(const std::exception&e){fprintf(stderr,"RC regions: %s\n",e.what());return -1;}catch(...){return -1;}
}
extern "C" int ng_rc_encode_regions(const uint8_t*s,unsigned d,unsigned c,unsigned b,uint8_t*out,size_t cap,size_t*n){return rc_encode_regions_impl(s,d,c,b,out,cap,n);}
extern "C" int ng_rc_test_region_frame(const uint8_t*s,unsigned d,uint8_t*out,size_t cap,size_t*n){return rc_encode_regions_impl(s,d,2,2,out,cap,n,true);}
