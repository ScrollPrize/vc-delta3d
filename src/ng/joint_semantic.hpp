#pragma once
// Included inside the experiment's anonymous namespace; shares its byte mapping
// and the existing arithmetic engine. No independent prediction implementation.
struct JointSemantic {
    int C;
    explicit JointSemantic(int classes):C(classes){}
    int context(const uint8_t*cls)const{return std::max(cls[0],cls[1])*C/16;}
    static int width(unsigned symbol){unsigned m=(symbol+1)/2;int w=0;while(m){w++;m>>=1;}return w;}
    static void emit_tags(const std::array<uint16_t,2>&s,int c,std::vector<Bytes>&streams){
        unsigned tags=0;for(int k=0;k<2;k++){
            unsigned w=width(s[k]),tag=w==8?15:w?2*w-1+(s[k]&1):0;tags|=tag<<(4*k);
            if(w>1&&w<8)streams[4+k*6+w-2].push_back(uint8_t((s[k]+1)/2-(1u<<(w-1))));
        }streams[c].push_back(uint8_t(tags));
    }
    void emit(const std::array<uint16_t,2>&s,const uint8_t*cls,std::vector<Bytes>&streams)const{
        int c=context(cls),pattern=int(s[0]!=0)+2*int(s[1]!=0);streams[c].push_back(uint8_t(pattern));if(!pattern)return;
        int w[]={width(s[0]),width(s[1])};int sub=c*3+pattern-1;
        streams[C+sub].push_back(uint8_t(w[0]*9+w[1]));
        unsigned signs=0;bool sign_used=false;
        for(int a=0;a<2;a++)if(w[a]&&w[a]<8){signs|=(s[a]&1)<<a;sign_used=true;}
        if(sign_used)streams[4*C+sub].push_back(uint8_t(signs));
        for(int a=0;a<2;a++)if(w[a]>1&&w[a]<8)streams[7*C+a*6+w[a]-2].push_back(uint8_t((s[a]+1)/2-(1<<(w[a]-1))));
    }
    template<bool Lut,class Get> JOINT_INLINE std::array<uint16_t,2> read_impl(const uint8_t*cls,Get get)const{
        int c=context(cls),pattern=get(c);need(pattern<4);if(!pattern)return {};
        int sub=c*3+pattern-1,joint=get(C+sub);need(joint<=80);
        static const auto widths=[](){std::array<std::array<int,2>,81> t{};for(int i=0;i<81;i++)t[i]={i/9,i%9};return t;}();
        auto ww=Lut?widths[joint]:std::array<int,2>{joint/9,joint%9};int w[]={ww[0],ww[1]};need(((w[0]!=0)+2*(w[1]!=0))==pattern);
        unsigned signs=0;if((w[0]&&w[0]<8)||(w[1]&&w[1]<8))signs=get(4*C+sub);need(signs<4);
        std::array<uint16_t,2>s{};
        for(int a=0;a<2;a++)if(w[a]){if(w[a]==8){s[a]=255;continue;}unsigned mag=1u<<(w[a]-1);if(w[a]>1){unsigned r=get(7*C+a*6+w[a]-2);need(r<mag);mag+=r;}s[a]=uint16_t(2*mag-((signs>>a)&1));}
        return s;
    }
    JOINT_INLINE std::array<uint16_t,2> read(const uint8_t*cls,const std::vector<Bytes>&streams,std::vector<size_t>&used)const{
        return read_impl<false>(cls,[&](int k){need(used[k]<streams[k].size());return unsigned(streams[k][used[k]++]);});
    }
    JOINT_INLINE std::array<uint16_t,2> read_pointers(const uint8_t*cls,const uint8_t**cur,const uint8_t*const*end)const{
        return read_impl<true>(cls,[&](int k){need(cur[k]!=end[k]);return unsigned(*cur[k]++);});
    }
    static void validate_tag_ranges(const std::vector<Bytes>&streams){
        need(streams.size()==16);
        for(int k=0;k<12;k++){unsigned bits=0;for(auto v:streams[4+k])bits|=v;need(bits<(1u<<(k%6+1)));}
    }
    static void validate_tag_streams(const std::vector<Bytes>&streams){
        need(streams.size()==16);std::array<size_t,256> hist{};std::array<size_t,12> expected{};
        for(int c=0;c<4;c++)for(auto tag:streams[c])hist[tag]++;
        for(unsigned pair=0;pair<256;pair++)for(int c=0;c<2;c++){
            unsigned tag=(pair>>(4*c))&15,w=(tag+1)/2;
            if(w>1&&w<8)expected[c*6+w-2]+=hist[pair];
        }
        for(int k=0;k<12;k++)need(streams[4+k].size()==expected[k]);
        validate_tag_ranges(streams);
    }
    struct Tag{uint16_t base,limit;int remainder;};
    static JOINT_INLINE const Tag& tag_info(unsigned tag){
        static constexpr Tag t[]={{0,0,-1},{2,0,-1},{1,0,-1},{4,2,0},{3,2,0},{8,4,1},{7,4,1},{16,8,2},{15,8,2},{32,16,3},{31,16,3},{64,32,4},{63,32,4},{128,64,5},{127,64,5},{255,0,-1}};
        return t[tag];
    }
    template<bool Prechecked> static JOINT_INLINE std::array<uint16_t,2> assemble_tags(unsigned tags,const uint8_t**cur,const uint8_t*const*end){
        std::array<uint16_t,2>s{};
        for(int k=0;k<2;k++){
            const auto&a=tag_info((tags>>(4*k))&15);unsigned r=0;
            if(a.remainder>=0){int stream=k*6+a.remainder;
                if constexpr(!Prechecked)need(cur[stream]!=end[stream]);
                r=*cur[stream]++;
                if constexpr(!Prechecked)need(r<a.limit);
            }
            s[k]=a.base+2*r;
        }
        return s;
    }
    template<bool Lookup,bool Prechecked=false> JOINT_INLINE std::array<uint16_t,2> read_tags(int c,const uint8_t**cur,const uint8_t*const*end)const{
        auto get=[&](int k){need(cur[k]!=end[k]);return unsigned(*cur[k]++);};
        unsigned tags=get(c);std::array<uint16_t,2>s{};
        if constexpr(Lookup){
            return assemble_tags<Prechecked>(tags,cur+4,end+4);
        }else for(int k=0;k<2;k++){
            unsigned tag=(tags>>(4*k))&15;
            if(tag==15)s[k]=255;
            else if(tag){unsigned w=(tag+1)/2,mag=1u<<(w-1);
                unsigned r=w>1?get(4+k*6+w-2):0;need(r<mag);s[k]=2*(mag+r)-(1-(tag&1));}
        }
        return s;
    }
    // Streams are independent by role: stage an entire row without changing
    // the consumption order WITHIN any stream. Final arithmetic is branchless.
    void read_row(int n,const uint8_t*a,const uint8_t*b,const uint8_t**cur,const uint8_t*const*end,std::array<uint16_t,2>*out)const{
        int sub[256],w[2][256],mag[2][256],signs[256];need(n<=256);
        auto get=[&](int k){need(cur[k]!=end[k]);return unsigned(*cur[k]++);};
        for(int i=0;i<n;i++){
            int c=std::max(a[i],b[i])*C/16,p=get(c);need(p<4);sub[i]=p?c*3+p-1:-1;
        }
        for(int i=0;i<n;i++){
            int j=sub[i]<0?0:int(get(C+sub[i]));need(j<=80);w[0][i]=j/9;w[1][i]=j%9;
            need(((w[0][i]!=0)+2*(w[1][i]!=0))==(sub[i]<0?0:sub[i]%3+1));
        }
        for(int i=0;i<n;i++){
            bool used=(w[0][i]>0&&w[0][i]<8)||(w[1][i]>0&&w[1][i]<8);
            signs[i]=used?get(4*C+sub[i]):0;need(signs[i]<4);
        }
        for(int c=0;c<2;c++)for(int i=0;i<n;i++){
            int ww=w[c][i];unsigned m=ww?1u<<(ww-1):0;
            if(ww>1&&ww<8){unsigned r=get(7*C+c*6+ww-2);need(r<m);m+=r;}mag[c][i]=m;
        }
        for(int i=0;i<n;i++)for(int c=0;c<2;c++)out[i][c]=w[c][i]==8?255:2*mag[c][i]-(w[c][i]?((signs[i]>>c)&1):0);
    }
};
struct JointSemanticBinary {
    int C,dimensions;std::array<std::vector<ng_cabac::Binary>,4> models;
    explicit JointSemanticBinary(int classes,int dims=2):C(classes),dimensions(dims){for(auto&v:models)v.resize(C*256);}
    template<bool Decode> struct Bins {
        JointSemanticBinary& parent;int context;ng_cabac::AuEncoder*enc;ng_cabac::AuDecoder*const*dec;
        JOINT_INLINE unsigned operator()(int role,int index,unsigned b)const{
            auto&m=parent.models[role][context*256+index];
            if constexpr(Decode)return m.decode(dec[role]);
            else return m.code(enc+role,nullptr,b);
        }
    };
    template<bool Decode>
    std::array<uint16_t,2> code_impl(const uint8_t*cls,ng_cabac::AuEncoder*enc,ng_cabac::AuDecoder*const*dec,std::array<uint16_t,2>in,int context){
        int c=context>=0?context:std::max(cls[0],cls[1])*C/16;need(c<C);
        // c is checked once; grammar-generated indices are <=126 (not stream
        // supplied). No need for a vector bounds check for every binary bin.
        const Bins<Decode> bit{*this,c,enc,dec};
        unsigned pattern=unsigned(in[0]!=0)+2*unsigned(in[1]!=0);
        unsigned lo=bit(0,0,pattern&1),hi=dimensions==2?bit(0,1+lo,pattern>>1):0;pattern=lo+2*hi;
        int w[2]={};std::array<uint16_t,2>out{};unsigned first_sign=0;
        for(int a=0;a<2;a++)if(pattern&(1u<<a)){
            int target=JointSemantic::width(in[a]);w[a]=1;
            int base=a?8+8*w[0]:0;
            while(w[a]<8&&bit(1,base+w[a]-1,target>w[a]))w[a]++;
            if(w[a]==8){out[a]=255;if(!a)first_sign=1;continue;}
            unsigned mag=1u<<(w[a]-1),input_mag=(in[a]+1)/2;
            for(int b=w[a]-2;b>=0;b--)mag|=bit(3,a*64+w[a]*7+b,(input_mag>>b)&1)<<b;
            unsigned sign=bit(2,a*64+pattern*8+w[a]-1+(a?32*first_sign:0),in[a]&1);
            if(!a)first_sign=sign;out[a]=uint16_t(2*mag-sign);
        }
        return out;
    }
    std::array<uint16_t,2> code(const uint8_t*cls,ng_cabac::AuEncoder*enc,ng_cabac::AuDecoder*const*dec,std::array<uint16_t,2>in={},int context=-1){
        return enc?code_impl<false>(cls,enc,dec,in,context):code_impl<true>(cls,enc,dec,in,context);
    }
};
