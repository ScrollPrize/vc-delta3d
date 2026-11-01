// Experimental selector geometry. Legacy mode words remain byte-identical.
namespace choice_grid {
inline unsigned edge(){return ng_choice_edge;}
inline size_t volume(){return size_t(edge())*edge()*edge();}
inline size_t count(unsigned d){return size_t(d/edge())*(d/edge())*(d/edge());}
inline size_t block(size_t i,unsigned d){unsigned e=edge(),n=d/e;return ((i/(d*d)/e)*n+(i/d%d)/e)*n+(i%d)/e;}
struct Scope {
    unsigned old=ng_choice_edge;
    explicit Scope(unsigned e){require(e==2||e==4||e==8||e==16||e==32);ng_choice_edge=e;}
    ~Scope(){ng_choice_edge=old;}
};
inline unsigned mode_word(unsigned mode){return mode|(edge()==4?0:edge()<<8);}
inline unsigned wire_edge(unsigned word){return word>>8?word>>8:4;}
}
