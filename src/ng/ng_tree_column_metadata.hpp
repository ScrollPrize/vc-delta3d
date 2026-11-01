// Shared preparation for ordinary and register-resident tree executors.
struct TreeColumnMetadata {
    std::vector<int> routed_features,routed_roots;
    std::vector<uint64_t> routed_masks;
    std::vector<unsigned> routed_groups;
    Bytes routed_zeros,routed_errors,routed_allzero;
};
static void prepare_tree_columns(TreeColumnMetadata&a,const Snapshot&s,const entropy_tree::VectorForest&v,const entropy_tree::MaskForest*m,ng_route_wave_data*w,unsigned pipeline,unsigned G){
    const unsigned d=w->width,n=w->lanes,feature_stride=pipeline==12?2:9,predictor_slot=pipeline==12?1:8;

                a.routed_features.resize(d*feature_stride*32);a.routed_roots.assign(d*32,v.zero_root);
                if((pipeline==11||pipeline==12))a.routed_masks.resize(d*32,1);
                a.routed_groups.resize(d*32);a.routed_zeros.resize(d*32);a.routed_errors.resize(d*32);
                if(pipeline>=16)a.routed_allzero.assign(d/choice_grid::edge(),1);
                for(unsigned j=0;j<n;j++)for(unsigned bx=0;bx<d/choice_grid::edge();bx++){
                    auto&c=s.menu[s.selectors[((size_t(w->zs[j]/choice_grid::edge())*(d/choice_grid::edge())+w->ys[j]/choice_grid::edge())*(d/choice_grid::edge()))+bx]];
                    bool zero=c.predictor==choice_frame::zero_predictor;
                    if(pipeline>=16&&!zero)a.routed_allzero[bx]=0;
                    for(unsigned x=bx*choice_grid::edge();x<(bx+1)*choice_grid::edge();x++){
                        a.routed_roots[x*32+j]=zero?v.zero_root:v.roots[c.entropy];
                        a.routed_groups[x*32+j]=entropy_bank(s,c.entropy,x,w->ys[j],w->zs[j])*G;a.routed_zeros[x*32+j]=zero;
                        a.routed_features[(x*feature_stride)*32+j]=w->class_lut[w->activity[j*d+x]];
                        a.routed_features[(x*feature_stride+predictor_slot)*32+j]=c.predictor;
                        if((pipeline==11||pipeline==12))a.routed_masks[x*32+j]=m->initial[((zero?v.roots.size():c.entropy)*13+c.predictor)*32+a.routed_features[(x*feature_stride)*32+j]];
                    }
                }

}
