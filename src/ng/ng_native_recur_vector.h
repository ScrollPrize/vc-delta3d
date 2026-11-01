/* Shared exact original-NG recurrence,16/32 independent rows. */
NG_NATTR static inline void NG_NNAME(int nl, const uint16_t (*restrict A)[VCN_LC_NL],
                                                            const uint8_t (*restrict SY)[VCN_LC_NL], vcn_lc_lanes *L,
                                                            uint8_t (*restrict V)[VCN_LC_NL],const uint8_t (*restrict DM)[VCN_LC_NL]) {
  const int nh = NG_NHALVES(nl);
  NG_NV w0[NG_NMAX], w1[NG_NMAX], w2[NG_NMAX], W0[NG_NMAX], W1[NG_NMAX], W2[NG_NMAX];
  for (int h = 0; h < NG_NMAX; h++) {
    w0[h] = NG_NLOAD((const NG_NV *)(L->w[0] + NG_NSTEP * h)), w1[h] = NG_NLOAD((const NG_NV *)(L->w[1] + NG_NSTEP * h));
    w2[h] = NG_NLOAD((const NG_NV *)(L->w[2] + NG_NSTEP * h));
    W0[h] = NG_NLOAD((const NG_NV *)(L->W[0] + NG_NSTEP * h)), W1[h] = NG_NLOAD((const NG_NV *)(L->W[1] + NG_NSTEP * h));
    W2[h] = NG_NLOAD((const NG_NV *)(L->W[2] + NG_NSTEP * h));
  }
  const NG_NV k128 = NG_NSET(128), k1 = NG_NSET(1), kB = NG_NSET(255), k0 = NG_NZERO();
  for (int x = 0; x < VCN_LC_D; x++) {
    NG_NV v[NG_NMAX];
    for (int h = 0; h < nh; h++) {
      NG_NV P = NG_NADD(NG_NADD(NG_NLOAD((const NG_NV *)(A[x] + NG_NSTEP * h)), NG_NMUL(W0[h], w0[h])),
                                         NG_NADD(NG_NMUL(W1[h], w1[h]), NG_NMUL(W2[h], w2[h])));
      if(DM) {
        const NG_NV dm=NG_NBYTES(DM[x]+NG_NSTEP*h);
        const NG_NV dx=NG_NEQ(dm,NG_NSET(2));
        const NG_NV dp=NG_NADD(NG_NLOAD((const NG_NV*)(A[x]+NG_NSTEP*h)),
            NG_NAND(dx,NG_NSLL(W0[h],8)));
        P=NG_NBLEND(P,dp,NG_NGT(dm,k0));
      }
      const NG_NV s = NG_NBYTES(SY[x]+NG_NSTEP*h);
      const NG_NV es = NG_NXOR(NG_NSRL(s, 1), NG_NSUB(k0, NG_NAND(s, k1)));
      const NG_NV fm = NG_NSRA(NG_NSLL(P, 8), 15);
      const NG_NV vv = NG_NAND(NG_NADD(NG_NSRL(NG_NADD(P, k128), 8),
                                                           NG_NSUB(NG_NXOR(es, fm), fm)),
                                          kB);
      W2[h] = W1[h], W1[h] = W0[h], W0[h] = vv;
      v[h] = vv;
    }
    NG_NOUTPUT(V[x],v,nh);
  }
  for (int h = 0; h < NG_NMAX; h++) {
    NG_NSTORE((NG_NV *)(L->W[0] + NG_NSTEP * h), W0[h]), NG_NSTORE((NG_NV *)(L->W[1] + NG_NSTEP * h), W1[h]);
    NG_NSTORE((NG_NV *)(L->W[2] + NG_NSTEP * h), W2[h]);
  }
}

#undef NG_NATTR
#undef NG_NNAME
#undef NG_NV
#undef NG_NHALVES
#undef NG_NMAX
#undef NG_NSTEP
#undef NG_NBYTES
#undef NG_NOUTPUT
#undef NG_NLOAD
#undef NG_NADD
#undef NG_NMUL
#undef NG_NSET
#undef NG_NZERO
#undef NG_NEQ
#undef NG_NAND
#undef NG_NSLL
#undef NG_NBLEND
#undef NG_NGT
#undef NG_NXOR
#undef NG_NSRL
#undef NG_NSUB
#undef NG_NSRA
#undef NG_NSTORE
