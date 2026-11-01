            NG_FV lo=NG_FOP7(w,NG_FOP7(north,back)),hi=NG_FOP8(w,NG_FOP8(north,back));
            NG_FOP9((NG_FV*)(f+32+j),NG_FOP10(w,north));
            NG_FOP9((NG_FV*)(f+64+j),NG_FOP10(w,back));
            NG_FOP9((NG_FV*)(f+96+j),NG_FOP10(north,back));
            NG_FOP9((NG_FV*)(f+128+j),NG_FOP10(hi,lo));
            NG_FOP9((NG_FV*)(f+160+j),NG_FOP11(NG_FOP10(w,ww)));
            NG_FV sum=NG_FOP12(w,NG_FOP12(north,back));
            NG_FOP9((NG_FV*)(f+224+j),sum);
            if(used&(1u<<6)){
                NG_FV yz=NG_FOP13(my,mz);
                NG_FV p=NG_FOP12(NG_FOP13(north,my),NG_FOP13(back,mz));
                p=NG_FOP10(p,NG_FOP13(yz,NG_FNEIGHBOR(6)));
                if(x){
                    p=NG_FOP12(p,w);
                    p=NG_FOP10(p,NG_FOP13(my,NG_FNEIGHBOR(4)));
                    p=NG_FOP10(p,NG_FOP13(mz,NG_FNEIGHBOR(5)));
                    p=NG_FOP12(p,NG_FOP13(yz,NG_FNEIGHBOR(7)));
                    p=NG_FOP6(p,NG_FOP7(hi,NG_FOP8(lo,p)),yz);
                }
                p=NG_FOP13(p,NG_FOP1(255));
                // Exact floor((sum+1)/3) for the complete uint8 neighbor range.
                NG_FV mean=NG_FOP14(NG_FOP12(sum,NG_FOP1(1)),NG_FOP1(21846));
                NG_FOP9((NG_FV*)(f+192+j),NG_FOP10(p,mean));
            }
