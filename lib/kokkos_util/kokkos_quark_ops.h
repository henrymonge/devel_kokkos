// -*- C++ -*-

/*! \file
 * \Define quark contractions
 */

#include "kokkos_defs.h"

#ifndef KOKKOS_QUARK_OPS_H
#define KOKKOS_QUARK_OPS_H

using namespace QDP;

KOKKOS_INLINE_FUNCTION void kokkos_quarkContractXX(auto cm_out,auto cm_init, auto cm1, auto cm2){

  // Permutations: +(0,1,2)+(1,2,0)+(2,0,1)-(1,0,2)-(0,2,1)-(2,1,0)

  // k1 = 0, k2 = 0
  // d(0,0) = eps^{i1,j1,0}\epsilon^{i2,j2,0} Q1^{i1,i2} Q2^{j1,j2}
  //       +(1,2,0),-(2,1,0)    +(1,2,0),-(2,1,0)
  cm_out(0,0) = cm_init(0,0) + cm1(1,1)*cm2(2,2)
              - cm1(1,2)*cm2(2,1)
              - cm1(2,1)*cm2(1,2)
              + cm1(2,2)*cm2(1,1);
  // k1 = 1, k2 = 0
  // d(0,1) = eps^{i1,j1,1}\epsilon^{i2,j2,0} Q1^{i1,i2} Q2^{j1,j2}
  //       +(2,0,1),-(0,2,1)    +(1,2,0),-(2,1,0)    
  cm_out(0,1) = cm_init(0,1) + cm1(2,1)*cm2(0,2)
              - cm1(2,2)*cm2(0,1)
              - cm1(0,1)*cm2(2,2)
              + cm1(0,2)*cm2(2,1);

  // k1 = 2, k2 = 0
  // d(0,2) = eps^{i1,j1,2}\epsilon^{i2,j2,0} Q1^{i1,i2} Q2^{j1,j2}
  //       +(0,1,2),-(1,0,2)    +(1,2,0),-(2,1,0)    
  cm_out(0,2) = cm_init(0,2) + cm1(0,1)*cm2(1,2)
              - cm1(0,2)*cm2(1,1)
              - cm1(1,1)*cm2(0,2)
              + cm1(1,2)*cm2(0,1);

  // k1 = 0, k2 = 1
  // d(1,0) = eps^{i1,j1,0}\epsilon^{i2,j2,0} Q1^{i1,i2} Q2^{j1,j2}
  //       +(1,2,0),-(2,1,0)    +(2,0,1),-(0,2,1)
  cm_out(1,0) = cm_init(1,0) + cm1(1,2)*cm2(2,0)
              - cm1(1,0)*cm2(2,2)
              - cm1(2,2)*cm2(1,0)
              + cm1(2,0)*cm2(1,2);

  // k1 = 1, k2 = 1
  // d(1,1) = eps^{i1,j1,1}\epsilon^{i2,j2,0} Q1^{i1,i2} Q2^{j1,j2}
  //       +(2,0,1),-(0,2,1)    +(2,0,1),-(0,2,1)
  cm_out(1,1) = cm_init(1,1) + cm1(2,2)*cm2(0,0)
              - cm1(2,0)*cm2(0,2)
              - cm1(0,2)*cm2(2,0)
              + cm1(0,0)*cm2(2,2);

  // k1 = 2, k2 = 1
  // d(1,1) = eps^{i1,j1,2}\epsilon^{i2,j2,0} Q1^{i1,i2} Q2^{j1,j2}
  //       +(0,1,2),-(1,0,2)    +(2,0,1),-(0,2,1)
  cm_out(1,2) = cm_init(1,2) + cm1(0,2)*cm2(1,0)
              - cm1(0,0)*cm2(1,2)
              - cm1(1,2)*cm2(0,0)
              + cm1(1,0)*cm2(0,2);

  // k1 = 0, k2 = 2
  // d(2,0) = eps^{i1,j1,0}\epsilon^{i2,j2,0} Q1^{i1,i2} Q2^{j1,j2}
  //       +(1,2,0),-(2,1,0)    +(0,1,2),-(1,0,2)
  cm_out(2,0) = cm_init(2,0) + cm1(1,0)*cm2(2,1)
              - cm1(1,1)*cm2(2,0)
              - cm1(2,0)*cm2(1,1)
              + cm1(2,1)*cm2(1,0);

  // k1 = 1, k2 = 2
  // d(2,1) = eps^{i1,j1,1}\epsilon^{i2,j2,0} Q1^{i1,i2} Q2^{j1,j2}
  //       +(2,0,1),-(0,2,1)    +(0,1,2),-(1,0,2)
  cm_out(2,1) = cm_init(2,1) + cm1(2,0)*cm2(0,1)
              - cm1(2,1)*cm2(0,0)
              - cm1(0,0)*cm2(2,1)
              + cm1(0,1)*cm2(2,0);

  // k1 = 2, k2 = 2
  // d(2,2) = eps^{i1,j1,2}\epsilon^{i2,j2,0} Q1^{i1,i2} Q2^{j1,j2}
  //       +(0,1,2),-(1,0,2)    +(0,1,2),-(1,0,2)
  cm_out(2,2) = cm_init(2,2) + cm1(0,0)*cm2(1,1)
              - cm1(0,1)*cm2(1,0)
              - cm1(1,0)*cm2(0,1)
              + cm1(1,1)*cm2(0,0);

}

KOKKOS_INLINE_FUNCTION void kokkos_quarkContract13(auto prop_out, auto prop1, auto prop2){


  //prop_out must be zero
  for(int j=0; j < 4; ++j){
    for(int i=0; i < 4; ++i){
     auto sout = Kokkos::subview(prop_out,i,j,Kokkos::ALL,Kokkos::ALL);
     for (int c1=0;c1<3;c1++){
       for (int c2=0;c2<3;c2++){
            sout(c1,c2) = 0;
       }
     }

      for(int k=0; k < 4; ++k){
         auto s1 = Kokkos::subview(prop1,k,i,Kokkos::ALL,Kokkos::ALL);
         auto s2 = Kokkos::subview(prop2,k,j,Kokkos::ALL,Kokkos::ALL);
         kokkos_quarkContractXX(sout,sout,s1,s2);
      } 
    }
  }

}

#endif
