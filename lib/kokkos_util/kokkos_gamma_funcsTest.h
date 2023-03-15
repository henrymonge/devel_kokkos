// -*- C++ -*-

/*! \file
 * \Kokkos math operations utilities
 */
#include "kokkos_util/kokkos_defs.h"


#ifndef KOKKOS_GAMMA_FUNCSTEST_H
#define KOKKOS_GAMMA_FUNCSTEST_H

using namespace QDP;

/*
KOKKOS_INLINE_FUNCTION void phases_to_View(auto view_phases, SftMom phases){
    const QDP::Subset& sub = QDP::all;
    int numSites = sub.siteTable().size();

    for (int mom_num=0; mom_num < num_mom; ++mom_num){
      //Initialize color matrices  
      Kokkos::parallel_for( "Site loop",Kokkos::RangePolicy<Kokkos::HostSpace::execution_space>(0,numSites), KOKKOS_LAMBDA ( int n ) {
        int qdp_index = sub.siteTable()[n];
        view_phases(n,mom_num) = Kokkos::complex<double>(phases[num_mom].elem(qdp_index).elem().elem().real(),
                                                     phases[num_mom].elem(qdp_index).elem().elem().imag());
      });
      Kokkos::fence();

    }    

}
*/

/*
  multi2d<DComplex>
  SftMom::sft(const LatticeComplex& cf) const
  {
    multi2d<DComplex> hsum(num_mom, sft_set.numSubsets()) ;

    QDPIO::cout << "Using 0 this for sft\n";
    for (int mom_num=0; mom_num < num_mom; ++mom_num)
      hsum[mom_num] = sumMulti(phases[mom_num]*cf, sft_set) ;

    return hsum ;
  }

*/

/*

  for(int k=0; k < s1.size(); ++k)
    for(int i=0; i < ss.numSubsets(); ++i)
      dest(k,i) = sum(s1[k],ss[i]);

*/

/*
KOKKOS_INLINE_FUNCTION void kokkos_sft(auto cf, Set sft_set, auto view_phases, auto hsum){

  for (int mom_num=0; mom_num < num_mom; ++mom_num){
      //loop over sft_set
    for (int k = 0; k < sft_set.numSubsets();k++){
      int numSites = sub.siteTable().size();
      Kokkos::parallel_reduce("Loop1", N, KOKKOS_LAMBDA (const int& i, double& lsum ) {

        lsum += 1.0*i;
        },result);
    } 
    //for(int i = 0; i < sft_set.numSubsets();i ++ ){
     
     //hsum[mom_num] = sumMulti(phases[mom_num]*cf, sft_set) ;
     
   // }
  }
}
*/

/*

struct kokkos_sft_Reduction {
  using value_type = float[];
  using size_type = View<float*>::size_type;

  size_type value_count;

  View<float*> X_;
  View<float*> Y_;

  ColumnSums (const View<float*>& X,View<float*>& Y) :
    value_count (X.extent(1)), // # columns in X
    X_ (X), Y_(Y)
  {}

  KOKKOS_INLINE_FUNCTION void
  operator() (const size_type i, value_type sum) const {
      sum[Y(i)] += X_(i);
  }

  KOKKOS_INLINE_FUNCTION void
  join (value_type dst,
        const value_type src) const {
    for (size_type j = 0; j < value_count; ++j) {
      dst[j] += src[j];
    }
  }

  KOKKOS_INLINE_FUNCTION void init (value_type sum) const {
    for (size_type j = 0; j < value_count; ++j) {
      sum[j] = 0.0;
    }
  }
};


*/
/*
struct kokkos_sft_Reduction {
  using value_type = float[];
  using size_type = View_corr_type::size_type;

  size_type value_count;

  View_corr_type corr_;
  View_LatticeComplex1d phases_;
  View_int_1d idx_map_;
  kokkos_sft_Reduction (const int sets, const View_corr_type& corr,View_LatticeComplex1d& phases,
                        View_int_1d& idx_map) : value_count (sets), // # columns in X
                        corr_(corr), phases_(phases), idx_map_(idx_map)
      {}
      KOKKOS_INLINE_FUNCTION void
      operator() (const size_type i, value_type sum) const {
          sum[idx_map_(i)] += (phases_(i,0)*corr_(i)).real();
      }

      KOKKOS_INLINE_FUNCTION void
      join (value_type dst,
            const value_type src) const {
        for (size_type j = 0; j < value_count; ++j) {
          dst[j] += src[j];
        }
      }

      KOKKOS_INLINE_FUNCTION void init (value_type sum) const {
        for (size_type j = 0; j < value_count; ++j) {
          sum[j] = 0.0;
        }
      }
};
*/



KOKKOS_INLINE_FUNCTION Kokkos::complex<double> timesMinusIK(const Kokkos::complex<double>& s1)
{
  Kokkos::complex<double> d;
  //zero_rep(d.real());
  d.real()=s1.imag();
  //d.imag() = -s1.elem();
  d.imag() = -s1.real();
  return d;
}

KOKKOS_INLINE_FUNCTION void kokkos_multiply_view_prop(auto factor, auto in_view, int n)
{
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            for(int c1=0;c1<3;c1++){
               for(int c2=0;c2<3;c2++){
                    in_view(n,i,j,c1,c2)=factor*in_view(n,i,j,c1,c2);
               }
            }
        }
    }
}

/*
KOKKOS_INLINE_FUNCTION Kokkos::complex<double> QDPtoKokkosComplex(const QDP::RComplex<double> c1)
{
  Kokkos::complex<double> d;
  d.real()=c1.real();
  d.imag() = c1.imag();
  return d;
}
*/

/*
KOKKOS_INLINE_FUNCTION void kokkos_sft( auto corr, View_int_2d d_sft_sets, View_corr_type k_b_prop, auto d_phases){
  int sets = d_sft_sets.extent(0);
 
  typedef Kokkos::TeamPolicy<>               team_policy;
  typedef Kokkos::TeamPolicy<>::member_type  member_type;


  Kokkos::parallel_for( "kokkos sft", team_policy( sets, Kokkos::AUTO ), KOKKOS_LAMBDA ( const member_type &teamMember) {
      const int j = teamMember.league_rank();
      int numSites = d_sft_sets.extent(1);
      Kokkos::complex<double> result=0;
      Kokkos::Sum<Kokkos::complex<double>> complexSum(result);

      Kokkos::parallel_reduce( Kokkos::TeamThreadRange( teamMember, numSites), [&] ( const int i, Kokkos::complex<double> &innerUpdate ) {
        innerUpdate += d_phases(d_sft_sets(j,i),0)*k_b_prop(d_sft_sets(j,i));
      },complexSum);
      corr(j)=result;
    });
}
*/

multi2d<DComplex> kokkos_sft(multi1d<bool> doSet,View_int_2d d_sft_sets, View_corr_type k_b_prop, auto d_phases){

      multi2d<DComplex> global_hsum;
      global_hsum.resize(d_phases.extent(0),d_sft_sets.extent(0));

      for(int i=0;i < d_phases.extent(0);i++){
        for(int j=0;j < d_sft_sets.extent(0);j++){
          global_hsum[i][j]=0;
          if(doSet[j]){
              Kokkos::complex<double> tmpSum=0;
              Kokkos::parallel_reduce(  "Site loop",range_policy(0,d_sft_sets.extent(1)), KOKKOS_LAMBDA ( const int k, Kokkos::complex<double> &innerUpdate ) {
                innerUpdate += d_phases(i,d_sft_sets(j,k))*k_b_prop(d_sft_sets(j,k));
              },tmpSum);
              global_hsum[i][j]=tmpSum.real();
          }
        }
      }
      QDPInternal::globalSumArray(global_hsum);
  
  return global_hsum;
}




KOKKOS_INLINE_FUNCTION void kokkos_sft_multi( View_int_2d d_sft_sets, View_corr_type k_b_prop, auto d_phases){

  int sets = d_sft_sets.extent(0);

  View_corr_type corr("myCorr",sets);
  typedef Kokkos::TeamPolicy<>               team_policy;
  typedef Kokkos::TeamPolicy<>::member_type  member_type;


  Kokkos::parallel_for( "kokkos sft", team_policy( sets, Kokkos::AUTO,d_phases.extent(0) ), KOKKOS_LAMBDA ( const member_type &teamMember) {
      const int j = teamMember.league_rank();
      int numSites = d_sft_sets.extent(1);

      Kokkos::parallel_for(Kokkos::TeamThreadRange( teamMember, Kokkos::AUTO ), KOKKOS_LAMBDA ( const int k) {    

          Kokkos::complex<double> result=0;
          Kokkos::Sum<Kokkos::complex<double>> complexSum(result);
    
          Kokkos::parallel_reduce( Kokkos::TeamThreadRange( teamMember, numSites), [&] ( const int i, Kokkos::complex<double> &innerUpdate ) {
            innerUpdate += d_phases(k,d_sft_sets(j,i))*k_b_prop(d_sft_sets(j,i));
          },complexSum);
          corr(j)=result;
     });
  });

}


/*
void kokkos_sft(View_corr_array_type summedCorrs, const View_corr_type& corr,View_LatticeComplex1d& phases,
                        View_int_2d d_sft_sets, View_int_1d& idx_map){
  
   int sets = d_sft_sets.extent(0);
   int moms=4;  
  Kokkos::parallel_for( "Set loop",range_policy(0,sets), KOKKOS_LAMBDA ( int nSite ) {
    int numSites = d_sft_sets.extent(1);
     Kokkos::parallel_for( "Mom loop",range_policy(0,moms), KOKKOS_LAMBDA ( int iSite ) {
        Kokkos::parallel_for( "Site loop",range_policy(0,numSites), KOKKOS_LAMBDA ( int mSite ) {
            int index = d_sft_sets(nSite,mSite);
            summedCorrs(nSite,iSite)+=phases(0,mSite)*corr(mSite);
          });
     });
       
  });
  Kokkos::fence();
  
}
*/

KOKKOS_INLINE_FUNCTION Kokkos::complex<double> timesIK(const Kokkos::complex<double>& s1)
{
  Kokkos::complex<double>  d;

  //zero_rep(d.real());
  d.real()=-s1.imag();
  d.imag() = s1.real();
  return d;
}


KOKKOS_INLINE_FUNCTION auto PropDotGammaSpinTest(auto prop, auto smat){

        Kokkos::complex<double> temp=0;
    for (int i=0; i<4;i++){ 
      for (int j=0; j<4;j++){
        temp=prop(i,j)*smat(i,j);
      }
    }
}

//psi_sq = real(trace(adj(anti_quark_prop) * Gamma(n1) * quark_prop_1 * Gamma(n)));


KOKKOS_INLINE_FUNCTION int kokkos_local_site(auto coord, auto latt_size){

	int order = 0;

	// In the 4D Case: t+Lt(x + Lx(y + Ly*z)
	// essentially  starting from i = dim[Nd-2]
	//  order =  latt_size[i-1]*(coord[i])
	//   and need to wrap i-1 around to Nd-1 when it gets below 0
	for(int mmu=latt_size.size()-2; mmu >= 0; --mmu) {
		int wrapmu = (mmu-1) % latt_size.size();
		if ( wrapmu < 0 ) wrapmu += latt_size.size();
		order = latt_size[wrapmu]*(coord[mmu] + order);
	}

    order += coord[ latt_size.size()-1 ];

    return order;

}

//returns pro adj
KOKKOS_INLINE_FUNCTION void adj_kokkos_prop(auto prop,auto adj_prop){

    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            for(int c1=0;c1<3;c1++){
               for(int c2=0;c2<3;c2++){                
                    adj_prop(i,j,c1,c2)=prop(j,i,c2,c1);
                    adj_prop(i,j,c1,c2).imag()=-adj_prop(i,j,c1,c2).imag();
               }
            }
        }
    }
}

//returns pro adj
KOKKOS_INLINE_FUNCTION void adj_kokkos_prop(auto adj_prop, auto prop,int n){

    auto prop_in = Kokkos::subview( prop, n,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    auto prop_out = Kokkos::subview( adj_prop, n,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);

    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            for(int c1=0;c1<3;c1++){
               for(int c2=0;c2<3;c2++){
                    prop_out(i,j,c1,c2)=prop_in(j,i,c2,c1);
                    prop_out(i,j,c1,c2).imag()=-prop_out(i,j,c1,c2).imag();
               }
            }
        }
    }
}


KOKKOS_INLINE_FUNCTION void kokkos_prop1_dot_prop2(auto prop_out,auto prop1,auto prop2, int n){

    auto p1 = Kokkos::subview( prop1, n,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    auto p2 = Kokkos::subview( prop2, n,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    auto result = Kokkos::subview( prop_out, n,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);

    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            for(int c1=0;c1<3;c1++){
                for(int c2=0;c2<3;c2++){
                    result(i,j,c1,c2)=0;
                    for(int k=0;k<4;k++){
                        for(int c3=0;c3<3;c3++){
                            result(i,j,c1,c2)+=p1(i,k,c1,c3)*p2(k,j,c3,c2);
                        }
                    }
                }
            }
        }
    }

}


KOKKOS_INLINE_FUNCTION Kokkos::complex<double> kokkos_trace_prop1_dot_prop2(auto prop1,auto prop2, int n){

    auto p1 = Kokkos::subview( prop1, n,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    auto p2 = Kokkos::subview( prop2, n,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    Kokkos::complex<double> trace = 0;

    for(int i=0;i<4;i++){
        for(int c1=0;c1<3;c1++){
            for(int k=0;k<4;k++){
                for(int c3=0;c3<3;c3++){
                    trace+=p1(i,k,c1,c3)*p2(k,i,c3,c1);
                }
            }
        }
    }
    return trace;
}

KOKKOS_INLINE_FUNCTION void kokkos_traceColor_prop1_dot_prop2(auto spinProp,auto prop1, auto prop2){

    for(int i=0;i<4;i++){
       for(int j=0;j<4;j++){
            spinProp(i,j)=0;
            for(int c1=0;c1<3;c1++){
                for(int k=0;k<4;k++){
                    for(int c3=0;c3<3;c3++){
                        spinProp(i,j)+=prop1(i,k,c1,c3)*prop2(k,j,c3,c1);
                    }
                }
            }
        }
    }
}

KOKKOS_INLINE_FUNCTION Kokkos::complex<double> kokkos_prop_trace(auto prop, int n){

    Kokkos::complex<double> trace=0;
    auto p = Kokkos::subview( prop, n,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);

    for(int i=0;i<4;i++){
        for(int c1=0;c1<3;c1++){
            trace+=p(i,i,c1,c1);
        }
    }
    return trace;
}


KOKKOS_INLINE_FUNCTION Kokkos::complex<double> kokkos_site_prop_trace(auto prop){

    Kokkos::complex<double> trace=0;
    for(int i=0;i<4;i++){
        for(int c1=0;c1<3;c1++){
            trace+=prop(i,i,c1,c1);
        }
    }
    return trace;
}


KOKKOS_INLINE_FUNCTION void kokkos_SpinMatrixProduct(auto sout, auto sm1, auto sm2){
   
    for (int i=0; i<4;i++){
      for (int j=0; j<4;j++){
        sout(i,j)=0;
        for (int k=0; k<4;k++){
          sout(i,j)+=sm1(i,k)*sm2(k,j);
        }
      }
    }
}

KOKKOS_INLINE_FUNCTION Kokkos::complex<double> kokkos_spinMatrix_trace(auto sMat){

    Kokkos::complex<double> trace=0;
    for(int i=0;i<4;i++){
       trace+=sMat(i,i);  
    }
    return trace;
}

KOKKOS_INLINE_FUNCTION void kokkos_SitePropDotSpinMatrix(int nSite, auto prop_out, auto prop, auto sm){
 
    for (int i=0; i<3;i++){
      for (int j=0; j<3;j++){
        auto p_out=Kokkos::subview(prop_out,nSite,Kokkos::ALL,Kokkos::ALL,i,j);
        auto p_in=Kokkos::subview(prop,nSite,Kokkos::ALL,Kokkos::ALL,i,j);
        kokkos_SpinMatrixProduct(p_out, p_in, sm);
      }
    }
}

KOKKOS_INLINE_FUNCTION void kokkos_SpinMatrixDotSiteProp(int nSite,auto prop_out, auto sm, auto prop){

    for (int i=0; i<3;i++){
      for (int j=0; j<3;j++){
        auto p_out=Kokkos::subview(prop_out,nSite,Kokkos::ALL,Kokkos::ALL,i,j);
        auto p_in=Kokkos::subview(prop,nSite,Kokkos::ALL,Kokkos::ALL,i,j);
        kokkos_SpinMatrixProduct(p_out, sm,p_in);
      }
    }
}

KOKKOS_INLINE_FUNCTION void kokkos_propSpinTrace(auto cmat, auto prop){

    for (int c1=0; c1<3;c1++){
      for (int c2=0; c2<3;c2++){
        cmat(c1,c2)=0;
        for(int i=0; i < 4; i++ ){
           cmat(c1,c2) += prop(i,i,c1,c2);
        }
      }
    }
}


KOKKOS_INLINE_FUNCTION void kokkos_ColorMatrixProduct(auto cm_out, auto cm1, auto cm2){

    for (int i=0; i<3;i++){
      for (int j=0; j<3;j++){
        cm_out(i,j)=0;
        for (int k=0; k<3;k++){
        cm_out(i,j)+=cm1(i,k)*cm2(k,j);
        }
      }
    }
}

KOKKOS_INLINE_FUNCTION void kokkos_PropColorMatrixProduct(auto prop_out, auto prop_in, auto cm){

    for (int i=0; i<4;i++){
      for (int j=0; j<4;j++){
        for(int c1=0; c1<3;c1++){
          for(int c2=0; c2 <3; c2++){
             prop_out(i,j,c1,c2)=0;
             for (int c3=0; c3<3;c3++){
                prop_out(i,j,c1,c2)+=prop_in(i,j,c1,c3)*cm(c3,c2);
             }
          }
        } 
      }
    }
}


//KOKKOS_INLINE_FUNCTION void gammaDotKokkosProp(int gamma_value,View_prop_type r,View_prop_type d, int n, int c1, int c2){
KOKKOS_INLINE_FUNCTION void KokkosPropDotGamma(View_prop_type r,int gamma_value,View_Latt_spin_matrix_type d, int n, int c1, int c2){
 
  auto sub_r = Kokkos::subview( r, n,Kokkos::ALL,Kokkos::ALL,c1,c2);
 
  switch(gamma_value) {
    case 0:
        for(int i=0; i < 4; ++i){ 
         d(n,0,i) = sub_r(0,i);
         d(n,1,i) = sub_r(1,i);
         d(n,2,i) = sub_r(2,i);
         d(n,3,i) = sub_r(3,i);
        }
      break;
    case 1:
      for(int i=0; i < 4; ++i){
        d(n,0,i) = timesIK(sub_r(3,i));
        d(n,1,i) = timesIK(sub_r(2,i));
        d(n,2,i) = timesMinusIK(sub_r(1,i));
        d(n,3,i) = timesMinusIK(sub_r(0,i));
      }      
      break;
    case 2:
      for(int i=0; i < 4; ++i){
        d(n,0,i) = -sub_r(3,i);
        d(n,1,i) = sub_r(2,i);
        d(n,2,i) = sub_r(1,i);
        d(n,3,i) = -sub_r(0,i);
        }
      break;
    case 3:
      for(int i=0; i < 4; ++i){
        d(n,0,i) = timesMinusIK(sub_r(0,i));
        d(n,1,i) = timesIK(sub_r(1,i));
        d(n,2,i) = timesMinusIK(sub_r(2,i));
       d(n,3,i) = timesIK(sub_r(3,i));
      }
      break;
    case 4:
      for(int i=0; i < 4; ++i){
        d(n,0,i) = timesIK(sub_r(2,i));
        d(n,1,i) = timesMinusIK(sub_r(3,i));
        d(n,2,i) = timesMinusIK(sub_r(0,i));
        d(n,3,i) = timesIK(sub_r(1,i));
      }
      break;
    case 5:
      for(int i=0; i < 4; ++i){
        d(n,0,i) = -sub_r(1,i);
        d(n,1,i) = sub_r(0,i);
       d(n,2,i) = -sub_r(3,i);
       d(n,3,i) = sub_r(2,i);
      }
      break;
    case 6:
      for(int i=0; i < 4; ++i){
        d(n,0,i) = timesMinusIK(sub_r(1,i));
        d(n,1,i) = timesMinusIK(sub_r(0,i));
        d(n,2,i) = timesMinusIK(sub_r(3,i));
        d(n,3,i) = timesMinusIK(sub_r(2,i));
      }
      break;
    case 7:
      for(int i=0; i < 4; ++i){
        d(n,0,i) = sub_r(2,i);
        d(n,1,i) = sub_r(3,i);
       d(n,2,i) = -sub_r(0,i);
       d(n,3,i) = -sub_r(1,i);
      }
      break;
    case 8:
      for(int i=0; i < 4; ++i){
        d(n,0,i) = sub_r(2,i);
        d(n,1,i) = sub_r(3,i);
        d(n,2,i) = sub_r(0,i);
        d(n,3,i) = sub_r(1,i);
      }
      break;
    case 9:
      for(int i=0; i < 4; ++i){
        d(n,0,i) = timesIK(sub_r(1,i));
        d(n,1,i) = timesIK(sub_r(0,i));
        d(n,2,i) = timesMinusIK(sub_r(3,i));
        d(n,3,i) = timesMinusIK(sub_r(2,i));
      }
      break;
    case 10:
      for(int i=0; i < 4; ++i){
        d(n,0,i) = -sub_r(1,i);
        d(n,1,i) = sub_r(0,i);
        d(n,2,i) = sub_r(3,i);
        d(n,3,i) = -sub_r(2,i);
      }
      break;
    case 11:
      for(int i=0; i < 4; ++i){
        d(n,0,i) = timesMinusIK(sub_r(2,i));
        d(n,1,i) = timesIK(sub_r(3,i));
        d(n,2,i) = timesMinusIK(sub_r(0,i));
        d(n,3,i) = timesIK(sub_r(1,i));
      }
      break;
    case 12:
      for(int i=0; i < 4; ++i){
        d(n,0,i) = timesIK(sub_r(0,i));
        d(n,1,i) = timesMinusIK(sub_r(1,i));
        d(n,2,i) = timesMinusIK(sub_r(2,i));
        d(n,3,i) = timesIK(sub_r(3,i));
      }
      break;
    case 13:
      for(int i=0; i < 4; ++i){
        d(n,0,i) = -sub_r(3,i);
        d(n,1,i) = sub_r(2,i);
        d(n,2,i) = -sub_r(1,i);
        d(n,3,i) = sub_r(0,i);
      }
      break;
    case 14:
      for(int i=0; i < 4; ++i){
        d(n,0,i) = timesMinusIK(sub_r(3,i));
        d(n,1,i) = timesMinusIK(sub_r(2,i));
        d(n,2,i) = timesMinusIK(sub_r(1,i));
        d(n,3,i) = timesMinusIK(sub_r(0,i));
      }
      break;
    case 15:
      for(int i=0; i < 4; ++i){
        d(n,0,i) = sub_r(0,i);
        d(n,1,i) = sub_r(1,i);
        d(n,2,i) = -sub_r(2,i);
        d(n,3,i) = -sub_r(3,i);
     }
      break;

    default:
      break;
     //QDPIO::cerr << __func__ << ": Invalid Gamma matrix value " << std::endl;
  }

}

///Multiplies kokkos spin matrix views sout = gamma_matrix * s
KOKKOS_INLINE_FUNCTION void KokkosGammaDotSmat(auto sout, int  gamma_value, auto s){

  switch(gamma_value) {
    case 0:
        for(int i=0; i < 4; ++i){
         sout(0,i) = s(0,i);
         sout(1,i) = s(1,i);
         sout(2,i) = s(2,i);
         sout(3,i) = s(3,i);
        }
      break;
    case 1:
      for(int i=0; i < 4; ++i){
        sout(0,i) = timesIK(s(3,i));
        sout(1,i) = timesIK(s(2,i));
        sout(2,i) = timesMinusIK(s(1,i));
        sout(3,i) = timesMinusIK(s(0,i));
      }
      break;
    case 2:
      for(int i=0; i < 4; ++i){
        sout(0,i) = -s(3,i);
        sout(1,i) = s(2,i);
        sout(2,i) = s(1,i);
        sout(3,i) = -s(0,i);
        }
      break;
    case 3:
      for(int i=0; i < 4; ++i){
        sout(0,i) = timesMinusIK(s(0,i));
        sout(1,i) = timesIK(s(1,i));
        sout(2,i) = timesMinusIK(s(2,i));
       sout(3,i) = timesIK(s(3,i));
      }
      break;
    case 4:
      for(int i=0; i < 4; ++i){
        sout(0,i) = timesIK(s(2,i));
        sout(1,i) = timesMinusIK(s(3,i));
        sout(2,i) = timesMinusIK(s(0,i));
        sout(3,i) = timesIK(s(1,i));
      }
      break;
    case 5:
      for(int i=0; i < 4; ++i){
        sout(0,i) = -s(1,i);
        sout(1,i) = s(0,i);
       sout(2,i) = -s(3,i);
       sout(3,i) = s(2,i);
      }
      break;
    case 6:
      for(int i=0; i < 4; ++i){
        sout(0,i) = timesMinusIK(s(1,i));
        sout(1,i) = timesMinusIK(s(0,i));
        sout(2,i) = timesMinusIK(s(3,i));
        sout(3,i) = timesMinusIK(s(2,i));
      }
      break;
    case 7:
      for(int i=0; i < 4; ++i){
        sout(0,i) = s(2,i);
        sout(1,i) = s(3,i);
       sout(2,i) = -s(0,i);
       sout(3,i) = -s(1,i);
      }
      break;
    case 8:
      for(int i=0; i < 4; ++i){
        sout(0,i) = s(2,i);
        sout(1,i) = s(3,i);
        sout(2,i) = s(0,i);
        sout(3,i) = s(1,i);
      }
      break;
    case 9:
      for(int i=0; i < 4; ++i){
        sout(0,i) = timesIK(s(1,i));
        sout(1,i) = timesIK(s(0,i));
        sout(2,i) = timesMinusIK(s(3,i));
        sout(3,i) = timesMinusIK(s(2,i));
      }
      break;
    case 10:
      for(int i=0; i < 4; ++i){
        sout(0,i) = -s(1,i);
        sout(1,i) = s(0,i);
        sout(2,i) = s(3,i);
        sout(3,i) = -s(2,i);
      }
      break;
    case 11:
      for(int i=0; i < 4; ++i){
        sout(0,i) = timesMinusIK(s(2,i));
        sout(1,i) = timesIK(s(3,i));
        sout(2,i) = timesMinusIK(s(0,i));
        sout(3,i) = timesIK(s(1,i));
      }
      break;
    case 12:
      for(int i=0; i < 4; ++i){
        sout(0,i) = timesIK(s(0,i));
        sout(1,i) = timesMinusIK(s(1,i));
        sout(2,i) = timesMinusIK(s(2,i));
        sout(3,i) = timesIK(s(3,i));
      }
      break;
    case 13:
      for(int i=0; i < 4; ++i){
        sout(0,i) = -s(3,i);
        sout(1,i) = s(2,i);
        sout(2,i) = -s(1,i);
        sout(3,i) = s(0,i);
      }
      break;
    case 14:
      for(int i=0; i < 4; ++i){
        sout(0,i) = timesMinusIK(s(3,i));
        sout(1,i) = timesMinusIK(s(2,i));
        sout(2,i) = timesMinusIK(s(1,i));
        sout(3,i) = timesMinusIK(s(0,i));
      }
      break;
    case 15:
      for(int i=0; i < 4; ++i){
        sout(0,i) = s(0,i);
        sout(1,i) = s(1,i);
        sout(2,i) = -s(2,i);
        sout(3,i) = -s(3,i);
     }
      break;

    default:
      //QDPIO::cerr << __func__ << ": Invalid Gamma matrix value " << std::endl;
      break;
  }

}
//KokkosPropDotGamma(d_quark_prop1_shft,m,s1,n,k,c3, c2);
KOKKOS_INLINE_FUNCTION void KokkosGammaDotProp(View_prop_shifted_type r,int gamma_value,View_Latt_spin_matrix_type d, int n, int k, int c1, int c2){

  auto sub_r = Kokkos::subview( r, n, k,Kokkos::ALL,Kokkos::ALL,c1,c2);

  switch(gamma_value) {
    case 0:
        for(int i=0; i < 4; ++i){
         d(n,0,i) = sub_r(0,i);
         d(n,1,i) = sub_r(1,i);
         d(n,2,i) = sub_r(2,i);
         d(n,3,i) = sub_r(3,i);
        }
      break;
    case 1:
      for(int i=0; i < 4; ++i){
        d(n,0,i) = timesIK(sub_r(3,i));
        d(n,1,i) = timesIK(sub_r(2,i));
        d(n,2,i) = timesMinusIK(sub_r(1,i));
        d(n,3,i) = timesMinusIK(sub_r(0,i));
      }
      break;
    case 2:
      for(int i=0; i < 4; ++i){
        d(n,0,i) = -sub_r(3,i);
        d(n,1,i) = sub_r(2,i);
        d(n,2,i) = sub_r(1,i);
        d(n,3,i) = -sub_r(0,i);
        }
      break;
    case 3:
      for(int i=0; i < 4; ++i){
        d(n,0,i) = timesMinusIK(sub_r(0,i));
        d(n,1,i) = timesIK(sub_r(1,i));
        d(n,2,i) = timesMinusIK(sub_r(2,i));
       d(n,3,i) = timesIK(sub_r(3,i));
      }
      break;
    case 4:
      for(int i=0; i < 4; ++i){
        d(n,0,i) = timesIK(sub_r(2,i));
        d(n,1,i) = timesMinusIK(sub_r(3,i));
        d(n,2,i) = timesMinusIK(sub_r(0,i));
        d(n,3,i) = timesIK(sub_r(1,i));
      }
      break;
    case 5:
      for(int i=0; i < 4; ++i){
        d(n,0,i) = -sub_r(1,i);
        d(n,1,i) = sub_r(0,i);
       d(n,2,i) = -sub_r(3,i);
       d(n,3,i) = sub_r(2,i);
      }
      break;
    case 6:
      for(int i=0; i < 4; ++i){
        d(n,0,i) = timesMinusIK(sub_r(1,i));
        d(n,1,i) = timesMinusIK(sub_r(0,i));
        d(n,2,i) = timesMinusIK(sub_r(3,i));
        d(n,3,i) = timesMinusIK(sub_r(2,i));
      }
      break;
    case 7:
      for(int i=0; i < 4; ++i){
        d(n,0,i) = sub_r(2,i);
        d(n,1,i) = sub_r(3,i);
       d(n,2,i) = -sub_r(0,i);
       d(n,3,i) = -sub_r(1,i);
      }
      break;
    case 8:
      for(int i=0; i < 4; ++i){
        d(n,0,i) = sub_r(2,i);
        d(n,1,i) = sub_r(3,i);
        d(n,2,i) = sub_r(0,i);
        d(n,3,i) = sub_r(1,i);
      }
      break;
    case 9:
      for(int i=0; i < 4; ++i){
        d(n,0,i) = timesIK(sub_r(1,i));
        d(n,1,i) = timesIK(sub_r(0,i));
        d(n,2,i) = timesMinusIK(sub_r(3,i));
        d(n,3,i) = timesMinusIK(sub_r(2,i));
      }
      break;
    case 10:
      for(int i=0; i < 4; ++i){
        d(n,0,i) = -sub_r(1,i);
        d(n,1,i) = sub_r(0,i);
        d(n,2,i) = sub_r(3,i);
        d(n,3,i) = -sub_r(2,i);
      }
      break;
    case 11:
      for(int i=0; i < 4; ++i){
        d(n,0,i) = timesMinusIK(sub_r(2,i));
        d(n,1,i) = timesIK(sub_r(3,i));
        d(n,2,i) = timesMinusIK(sub_r(0,i));
        d(n,3,i) = timesIK(sub_r(1,i));
      }
      break;
    case 12:
      for(int i=0; i < 4; ++i){
        d(n,0,i) = timesIK(sub_r(0,i));
        d(n,1,i) = timesMinusIK(sub_r(1,i));
        d(n,2,i) = timesMinusIK(sub_r(2,i));
        d(n,3,i) = timesIK(sub_r(3,i));
      }
      break;
    case 13:
      for(int i=0; i < 4; ++i){
        d(n,0,i) = -sub_r(3,i);
        d(n,1,i) = sub_r(2,i);
        d(n,2,i) = -sub_r(1,i);
        d(n,3,i) = sub_r(0,i);
      }
      break;
    case 14:
      for(int i=0; i < 4; ++i){
        d(n,0,i) = timesMinusIK(sub_r(3,i));
        d(n,1,i) = timesMinusIK(sub_r(2,i));
        d(n,2,i) = timesMinusIK(sub_r(1,i));
        d(n,3,i) = timesMinusIK(sub_r(0,i));
      }
      break;
    case 15:
      for(int i=0; i < 4; ++i){
        d(n,0,i) = sub_r(0,i);
        d(n,1,i) = sub_r(1,i);
        d(n,2,i) = -sub_r(2,i);
        d(n,3,i) = -sub_r(3,i);
     }
      break;

    default:
      break;
     //QDPIO::cerr << __func__ << ": Invalid Gamma matrix value " << std::endl;
  }

}

/*
KOKKOS_INLINE_FUNCTION void KokkosPropDotGamma(auto prop_out, auto prop, int gamma_value){    
    for(int c1=0;c1<3;c1++){
        for(int c2=0;c2<3;c2++){
            auto spin_out=Kokkos::subview(prop_out,Kokkos::ALL,Kokkos::ALL,c1,c2);
            auto spin_in=Kokkos::subview(prop,Kokkos::ALL,Kokkos::ALL,c1,c2);
            KokkosSmatDotGamma(spin_out, spin_in, gamma_value);
        }
    }
}

*/
///Multiplies kokkos spin matrix views sout = s * gamma_matrix
KOKKOS_INLINE_FUNCTION void KokkosSmatDotGamma(auto sout, auto s, int gamma_value){

  switch(gamma_value) {
    case 0:
      for(int i=0; i < 4; ++i){
        sout(i,0) =  s(i,0);
        sout(i,1) =  s(i,1);
        sout(i,2) =  s(i,2);
        sout(i,3) =  s(i,3);
      }
      break;
    case 1:
      for(int i=0; i < 4; ++i){
        sout(i,0) = timesMinusIK(s(i,3));
        sout(i,1) = timesMinusIK(s(i,2));
        sout(i,2) = timesIK(s(i,1));
        sout(i,3) = timesIK(s(i,0));
      }
      break;
    case 2:
      for(int i=0; i < 4; ++i){
        sout(i,0) = -s(i,3);
        sout(i,1) =  s(i,2);
        sout(i,2) =  s(i,1);
        sout(i,3) = -s(i,0);
      }
      break;
    case 3:
      for(int i=0; i < 4; ++i){
        sout(i,0) = timesMinusIK(s(i,0));
        sout(i,1) = timesIK(s(i,1));
        sout(i,2) = timesMinusIK(s(i,2));
        sout(i,3) = timesIK(s(i,3));
      }
      break;
    case 4:
      for(int i=0; i < 4; ++i){
        sout(i,0) = timesMinusIK(s(i,2));
        sout(i,1) = timesIK(s(i,3));
        sout(i,2) = timesIK(s(i,0));
        sout(i,3) = timesMinusIK(s(i,1));
     }
      break;
    case 5:
      for(int i=0; i < 4; ++i){
        sout(i,0) =  s(i,1);
        sout(i,1) = -s(i,0);
        sout(i,2) =  s(i,3);
        sout(i,3) = -s(i,2);
      }
      break;
    case 6:
      for(int i=0; i < 4; ++i){
        sout(i,0) = timesMinusIK(s(i,1));
        sout(i,1) = timesMinusIK(s(i,0));
        sout(i,2) = timesMinusIK(s(i,3));
        sout(i,3) = timesMinusIK(s(i,2));
      }
      break;
    case 7:
      for(int i=0; i < 4; ++i){
        sout(i,0) = -s(i,2);
        sout(i,1) = -s(i,3);
        sout(i,2) =  s(i,0);
        sout(i,3) =  s(i,1);
      }
      break;
    case 8:
      for(int i=0; i < 4; ++i){
        sout(i,0) =  s(i,2);
        sout(i,1) =  s(i,3);
        sout(i,2) =  s(i,0);
        sout(i,3) =  s(i,1);
      }
      break;
    case 9:
      for(int i=0; i < 4; ++i){
        sout(i,0) = timesIK(s(i,1));
        sout(i,1) = timesIK(s(i,0));
        sout(i,2) = timesMinusIK(s(i,3));
        sout(i,3) = timesMinusIK(s(i,2));
      }
      break;
    case 10:
      for(int i=0; i < 4; ++i){
        sout(i,0) =  s(i,1);
        sout(i,1) = -s(i,0);
        sout(i,2) = -s(i,3);
        sout(i,3) =  s(i,2);
      }
      break;
    case 11:
      for(int i=0; i < 4; ++i){
        sout(i,0) = timesMinusIK(s(i,2));
        sout(i,1) = timesIK(s(i,3));
        sout(i,2) = timesMinusIK(s(i,0));
        sout(i,3) = timesIK(s(i,1));
      }
      break;
    case 12:
      for(int i=0; i < 4; ++i){
        sout(i,0) = timesIK(s(i,0));
        sout(i,1) = timesMinusIK(s(i,1));
        sout(i,2) = timesMinusIK(s(i,2));
        sout(i,3) = timesIK(s(i,3));
      }
      break;
    case 13:
      for(int i=0; i < 4; ++i){
        sout(i,0) =  s(i,3);
        sout(i,1) = -s(i,2);
        sout(i,2) =  s(i,1);
        sout(i,3) = -s(i,0);
      }
      break;
    case 14:
      for(int i=0; i < 4; ++i){
        sout(i,0) = timesMinusIK(s(i,3));
        sout(i,1) = timesMinusIK(s(i,2));
        sout(i,2) = timesMinusIK(s(i,1));
        sout(i,3) = timesMinusIK(s(i,0));
      }
      break;
    case 15:
      for(int i=0; i < 4; ++i){
        sout(i,0) =  s(i,0);
        sout(i,1) =  s(i,1);
        sout(i,2) = -s(i,2);
        sout(i,3) = -s(i,3);
      }
      break;
    default:
      break;
      //QDPIO::cerr << __func__ << ": Invalid Gamma matrix value " << std::endl;
  }
}


KOKKOS_INLINE_FUNCTION void kokkos_PropDotGamma(auto prop_out, auto prop_in, int gamma_value, int n){

    for (int c1=0; c1<3;c1++){
      for (int c2=0; c2<3;c2++){
        auto p_out_spin = Kokkos::subview( prop_out,n, Kokkos::ALL,Kokkos::ALL,c1,c2);
        auto p_in_spin = Kokkos::subview( prop_in,n, Kokkos::ALL,Kokkos::ALL,c1,c2);
        KokkosSmatDotGamma(p_out_spin, p_in_spin, gamma_value);
       }
      }
}

KOKKOS_INLINE_FUNCTION void kokkos_GammaDotProp(auto prop_out, int gamma_value, auto prop_in, int n){

    for (int c1=0; c1<3;c1++){
      for (int c2=0; c2<3;c2++){
        auto p_out_spin = Kokkos::subview( prop_out,n, Kokkos::ALL,Kokkos::ALL,c1,c2);
        auto p_in_spin = Kokkos::subview( prop_in,n, Kokkos::ALL,Kokkos::ALL,c1,c2);
        KokkosGammaDotSmat(p_out_spin, gamma_value, p_in_spin);  
       }
      }
}

KOKKOS_INLINE_FUNCTION void kokkos_ColorMatrixDotProp(auto prop_out, auto cmatrix, auto prop_in, int n){

    for (int i=0; i<4;i++){
      for (int j=0; j<4;j++){
        auto p_out_color = Kokkos::subview( prop_out,n,i,j, Kokkos::ALL,Kokkos::ALL);
        auto p_in_color = Kokkos::subview( prop_in,n,i,j, Kokkos::ALL,Kokkos::ALL);
        auto cm = Kokkos::subview( cmatrix,n, Kokkos::ALL,Kokkos::ALL);
        kokkos_ColorMatrixProduct(p_out_color, cm, p_in_color );
        }
      }
}


KOKKOS_INLINE_FUNCTION void KokkosPropDotGamma(int gamma_value, View_prop_shifted_type l, View_Latt_spin_matrix_type d, int n, int k, int c1, int c2){

  auto sub_l = Kokkos::subview( l, n, k, Kokkos::ALL,Kokkos::ALL,c1,c2);

  switch(gamma_value) {
    case 0:
      for(int i=0; i < 4; ++i){
        d(n,i,0) =  sub_l(i,0);
        d(n,i,1) =  sub_l(i,1);
        d(n,i,2) =  sub_l(i,2);
        d(n,i,3) =  sub_l(i,3);
      }
      break;
    case 1:
      for(int i=0; i < 4; ++i){
        d(n,i,0) = timesMinusIK(sub_l(i,3));
        d(n,i,1) = timesMinusIK(sub_l(i,2));
        d(n,i,2) = timesIK(sub_l(i,1));
        d(n,i,3) = timesIK(sub_l(i,0));
      }
      break;
    case 2:
      for(int i=0; i < 4; ++i){
        d(n,i,0) = -sub_l(i,3);
        d(n,i,1) =  sub_l(i,2);
        d(n,i,2) =  sub_l(i,1);
        d(n,i,3) = -sub_l(i,0);
      }
      break;
    case 3:
      for(int i=0; i < 4; ++i){
        d(n,i,0) = timesMinusIK(sub_l(i,0));
        d(n,i,1) = timesIK(sub_l(i,1));
        d(n,i,2) = timesMinusIK(sub_l(i,2));
        d(n,i,3) = timesIK(sub_l(i,3));
      }
      break;
    case 4:
      for(int i=0; i < 4; ++i){
        d(n,i,0) = timesMinusIK(sub_l(i,2));
        d(n,i,1) = timesIK(sub_l(i,3));
        d(n,i,2) = timesIK(sub_l(i,0));
        d(n,i,3) = timesMinusIK(sub_l(i,1));
     }
      break;
    case 5:
      for(int i=0; i < 4; ++i){
        d(n,i,0) =  sub_l(i,1);
        d(n,i,1) = -sub_l(i,0);
        d(n,i,2) =  sub_l(i,3);
        d(n,i,3) = -sub_l(i,2);
      }
      break;
    case 6:
      for(int i=0; i < 4; ++i){
        d(n,i,0) = timesMinusIK(sub_l(i,1));
        d(n,i,1) = timesMinusIK(sub_l(i,0));
        d(n,i,2) = timesMinusIK(sub_l(i,3));
        d(n,i,3) = timesMinusIK(sub_l(i,2));
      }
      break;
    case 7:
      for(int i=0; i < 4; ++i){
        d(n,i,0) = -sub_l(i,2);
        d(n,i,1) = -sub_l(i,3);
        d(n,i,2) =  sub_l(i,0);
        d(n,i,3) =  sub_l(i,1);
      }
      break;
    case 8:
      for(int i=0; i < 4; ++i){
        d(n,i,0) =  sub_l(i,2);
        d(n,i,1) =  sub_l(i,3);
        d(n,i,2) =  sub_l(i,0);
        d(n,i,3) =  sub_l(i,1);
      }
      break;
    case 9:
      for(int i=0; i < 4; ++i){
        d(n,i,0) = timesIK(sub_l(i,1));
        d(n,i,1) = timesIK(sub_l(i,0));
        d(n,i,2) = timesMinusIK(sub_l(i,3));
        d(n,i,3) = timesMinusIK(sub_l(i,2));
      }
      break;
    case 10:
      for(int i=0; i < 4; ++i){
        d(n,i,0) =  sub_l(i,1);
        d(n,i,1) = -sub_l(i,0);
        d(n,i,2) = -sub_l(i,3);
        d(n,i,3) =  sub_l(i,2);
      }
      break;
    case 11:
      for(int i=0; i < 4; ++i){
        d(n,i,0) = timesMinusIK(sub_l(i,2));
        d(n,i,1) = timesIK(sub_l(i,3));
        d(n,i,2) = timesMinusIK(sub_l(i,0));
        d(n,i,3) = timesIK(sub_l(i,1));
      }
      break;
    case 12:
      for(int i=0; i < 4; ++i){
        d(n,i,0) = timesIK(sub_l(i,0));
        d(n,i,1) = timesMinusIK(sub_l(i,1));
        d(n,i,2) = timesMinusIK(sub_l(i,2));
        d(n,i,3) = timesIK(sub_l(i,3));
      }
      break;
    case 13:
      for(int i=0; i < 4; ++i){
        d(n,i,0) =  sub_l(i,3);
        d(n,i,1) = -sub_l(i,2);
        d(n,i,2) =  sub_l(i,1);
        d(n,i,3) = -sub_l(i,0);
      }
      break;
    case 14:
      for(int i=0; i < 4; ++i){
        d(n,i,0) = timesMinusIK(sub_l(i,3));
        d(n,i,1) = timesMinusIK(sub_l(i,2));
        d(n,i,2) = timesMinusIK(sub_l(i,1));
        d(n,i,3) = timesMinusIK(sub_l(i,0));
      }
      break;
    case 15:
      for(int i=0; i < 4; ++i){
        d(n,i,0) =  sub_l(i,0);
        d(n,i,1) =  sub_l(i,1);
        d(n,i,2) = -sub_l(i,2);
        d(n,i,3) = -sub_l(i,3);
      }
      break;
    default:
      break;
      //QDPIO::cerr << __func__ << ": Invalid Gamma matrix value " << std::endl;
  }

}

#endif
