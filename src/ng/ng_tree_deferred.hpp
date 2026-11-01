// Own the prepared metadata while independent groups of the same t wave wait
// for co-execution. Native publication is shared with the normal scheduler.
struct DeferredTreeWave {
    ng_route_wave_data w{};
    std::vector<ng_bank_compact_column>columns;
    std::vector<std::array<std::array<uint16_t,32>,4>>weights;
    std::vector<uint16_t>activity;
    Bytes values;
    size_t offsets[32]{},error_offsets[32]{};int ys[32]{},zs[32]{};
    alignas(64) uint16_t previous[3][32]{};
    TreeColumnMetadata metadata;
    std::unique_ptr<void,decltype(&ng_context_rans_destroy)>entropy{nullptr,ng_context_rans_destroy};
    void capture(ng_route_wave_data*source,std::unique_ptr<void,decltype(&ng_context_rans_destroy)>&e){
        w=*source;columns.resize(w.width);weights.resize(w.width/ng_choice_edge);activity.resize(w.width*32);values.resize(w.width*32);
        memcpy(columns.data(),w.columns,columns.size()*sizeof(columns[0]));memcpy(weights.data(),w.block_weights,weights.size()*sizeof(weights[0]));memcpy(activity.data(),w.activity,activity.size()*2);
        memcpy(previous,w.previous,sizeof(previous));memcpy(offsets,w.offsets,sizeof(offsets));memcpy(error_offsets,w.error_offsets,sizeof(error_offsets));memcpy(ys,w.ys,sizeof(ys));memcpy(zs,w.zs,sizeof(zs));
        for(unsigned x=0;x<w.width;x++)for(unsigned j=w.lanes;j<32;j++){auto&c=columns[x];c.pre[j]=c.constant[j]=c.flags[j]=c.bounds[j]=c.residual[j]=0;for(unsigned k=0;k<4;k++)weights[x/ng_choice_edge][k][j]=0;}
        w.columns=columns.data();w.block_weights=reinterpret_cast<const uint16_t(*)[4][32]>(weights.data());w.activity=activity.data();w.values=values.data();w.previous=previous;w.offsets=offsets;w.error_offsets=error_offsets;w.ys=ys;w.zs=zs;
        entropy.swap(e);
    }
};
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
template<size_t...I> __attribute__((target("avx512f,avx512bw,avx512vl,avx512cd,avx2,popcnt")))
static void tree_interleave(const Snapshot&s,const entropy_tree::VectorForest&v,const entropy_tree::MaskForest&m,DeferredTreeWave*const*q,unsigned G,std::index_sequence<I...>){
    std::array<TreeRegisterWave,sizeof...(I)>states={TreeRegisterWave(s,v,m,&q[I]->w,static_cast<ng_context_rans*>(q[I]->entropy.get()),nullptr,std::move(q[I]->metadata),G)...};
    for(unsigned x=0;x<q[0]->w.width;x++)(states[I].template step<true>(),...);
    (states[I].template finish<true>(),...);
    ((q[I]->metadata=std::move(static_cast<TreeColumnMetadata&>(states[I]))),...);
}
#endif
