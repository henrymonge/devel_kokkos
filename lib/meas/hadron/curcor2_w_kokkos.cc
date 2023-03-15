/*! \file
 *  \brief Mesonic current correlators
 */

#include "chromabase.h"
#include "util/ft/sftmom.h"
#include "meas/hadron/mesons_w.h"
#include "qdp_util.h"                 // part of QDP++, for crtesn()
#include <Kokkos_Core.hpp>
#include "qdp.h"
#include "meas/hadron/kokkos_gamma_funcsTest.h"
#include "meas/hadron/kokkos_defs.h"
#include <chrono>
#include <time.h>
#include <fstream>
#include <stdio.h>

namespace Chroma {

//! Construct current correlators 
/*!
 * \ingroup hadron
 *
 * This routine is specific to Wilson fermions!
 *
 *  The two propagators can be identical or different.

 * This includes the "rho_1--rho_2" correlators used for O(a) improvement

 * For use with "rotated" propagators we added the possibility of also
 * computing the local std::vector current, when no_vec_cur = 4. In this
 * case the 3 local currents come last.

 * \param u               gauge field ( Read )
 * \param quark_prop_1    first quark propagator ( Read )
 * \param quark_prop_2    second (anti-) quark propagator ( Read )
 * \param phases          fourier transform phase factors ( Read )
 * \param t0              timeslice coordinate of the source ( Read )
 * \param no_vec_cur      number of std::vector current types, 3 or 4 ( Read )
 * \param xml             namelist file object ( Read )
 * \param xml_group       std::string used for writing xml data ( Read )
 *
 *         ____
 *         \
 * cc(t) =  >  < m(t_source, 0) c(t + t_source, x) >
 *         /                    
 *         ----
 *           x
 */


void curcor2(const multi1d<LatticeColorMatrix>& u, 
	     const LatticePropagator& quark_prop_1, 
	     const LatticePropagator& quark_prop_2, 
	     const SftMom& phases,
	     int t0,
	     int no_vec_cur,
	     XMLWriter& xml,
	     const std::string& xml_group)
{
  START_CODE();


  QDPIO::cout << "\n\n***************\nUsing Kokkos Currents Code GPU\n***************\n\n";

  double time;
  StopWatch swatch;
  swatch.reset();
  swatch.start();


  //For CPU
  //using MemSpace=Kokkos::HostSpace; 

  //For GPU
  //using MemSpace=Kokkos::Experimental::HIP;
  using MemSpace=Kokkos::OpenMP;
  using ExecSpace = MemSpace::execution_space;
  using range_policy = Kokkos::RangePolicy<ExecSpace>;

  //const QDP::Subset& sub = QDP::rb[0];
  const QDP::Subset& sub = QDP::all;

  std::cout.precision(6);

  // Allocate y, x vectors and Matrix A on device.
  int numSites = sub.siteTable().size();
  int nodeNumber=Layout::nodeNumber();

  multi1d<int> LattDims = Layout::lattSize();

  View_LatticeComplex1d view_chi_sqN("view_chi_sqN", numSites,4);
  View_LatticeComplex1d view_psi_sqN("view_psi_sqN", numSites,4);

  int G5 = Ns*Ns-1;
  LatticePropagator anti_quark_prop =  Gamma(G5) * quark_prop_2 * Gamma(G5);
  LatticePropagator adj_anti_quark_prop =  adj(anti_quark_prop);
  LatticePropagator tmp_prop2;
  LatticePropagator tmp_prop1;

  View_prop_type d_adj_anti_quark_prop("d_adj_anti-quark", numSites);
  View_prop_type d_anti_quark_prop("d_anti_quark_prop", numSites);
  View_prop_type d_quark_prop1("d_quark", numSites);


  //Declare shifted props
  View_prop_shifted_type  d_quark_prop1_shft("d_quark_prop1_shft", numSites,4);
  View_prop_shifted_type  d_adj_anti_quark_prop_shft("d_adj_anti_quark_prop_shft", numSites,4);
  View_prop_shifted_type  d_anti_quark_prop_shft("d_anti_quark_prop_shft", numSites,4);


  // temp propagators for intermediate states
  View_prop_type d_tmp_propN("d_tmp_propN", numSites);
  View_prop_type d_tmp_propN2("d_tmp_propN2", numSites);
  View_prop_type d_tmp_propN3("d_tmp_propN3", numSites);
  View_prop_type d_tmp_propN4("d_tmp_propN4", numSites);

  View_color_matrix_arr_type d_u("d_u",numSites,Nd);


  // Create host mirrors of device views.
  View_prop_type::HostMirror h_adj_anti_quark_prop = Kokkos::create_mirror_view( d_adj_anti_quark_prop );
  View_prop_type::HostMirror h_anti_quark_prop = Kokkos::create_mirror_view( d_anti_quark_prop );
  View_prop_type::HostMirror h_quark_prop1 = Kokkos::create_mirror_view( d_quark_prop1 );


  View_prop_shifted_type::HostMirror h_quark_prop1_shft = Kokkos::create_mirror_view( d_quark_prop1_shft );
  View_prop_shifted_type::HostMirror h_adj_anti_quark_prop_shft = Kokkos::create_mirror_view( d_adj_anti_quark_prop_shft );
  View_prop_shifted_type::HostMirror h_anti_quark_prop_shft = Kokkos::create_mirror_view( d_anti_quark_prop_shft );
  View_color_matrix_arr_type::HostMirror h_u = Kokkos::create_mirror_view( d_u );


 //Copy props to device
  //Initialize color matrices  
  Kokkos::parallel_for( "Site loop",Kokkos::RangePolicy<Kokkos::HostSpace::execution_space>(0,numSites), KOKKOS_LAMBDA ( int n ) {
    int qdp_index = sub.siteTable()[n];

   for(int k=0; k<Nd; k++){
      for(int i=0;i <3;i++){
        for(int j=0; j< 3; j++){
          h_u(n,k,i,j) = Kokkos::complex<double>(u[k].elem(qdp_index).elem().elem(i,j).real(),
                                                 u[k].elem(qdp_index).elem().elem(i,j).imag());
       }
      }
    }
    
  }); 
  Kokkos::fence();


  //Initialize kokkos props
  for(int k=0;k<4;k++){
   tmp_prop2 = shift(quark_prop_1, FORWARD, k);
   tmp_prop1 = shift(anti_quark_prop, FORWARD, k);
    Kokkos::parallel_for( "init shift props",Kokkos::RangePolicy<Kokkos::HostSpace::execution_space>(0,numSites), KOKKOS_LAMBDA ( int n ) {
      int qdp_index = sub.siteTable()[n];

      for ( int i = 0; i  < 4; ++i ) {
        for ( int j = 0; j < 4; ++j ) {
          for ( int c1 = 0; c1  < 3; ++c1) {
            for  ( int c2 = 0; c2  < 3; ++c2 ) {
              h_quark_prop1_shft(n,k,i,j,c1,c2)=QDPtoKokkosComplex(tmp_prop2.elem(qdp_index).elem(i,j).elem(c1,c2));
              h_anti_quark_prop_shft(n,k,i,j,c1,c2)=QDPtoKokkosComplex(tmp_prop1.elem(qdp_index).elem(i,j).elem(c1,c2));
              if(k==0){
                h_adj_anti_quark_prop(n,i,j,c1,c2)=QDPtoKokkosComplex( adj_anti_quark_prop.elem(qdp_index).elem(i,j).elem(c1,c2));
                h_quark_prop1(n,i,j,c1,c2)=QDPtoKokkosComplex(quark_prop_1.elem(qdp_index).elem(i,j).elem(c1,c2));
                h_anti_quark_prop(n,i,j,c1,c2)=QDPtoKokkosComplex(anti_quark_prop.elem(qdp_index).elem(i,j).elem(c1,c2));
              }
            }
          }
        }
      }
    });
  }

  /*
  ********INDEXING TEST*********

  const int nodeSites = Layout::sitesOnNode();

  int lsindex= Layout::linearSiteIndex(98304);

   QDPIO::cout<<"lsindex = "<<lsindex<<"\n";
   QDPIO::cout<<"Number of nodes= "<<Layout::numNodes()<< "   nodeSites=  "<< nodeSites<<"\n"; 
  for(int n=0;n<numSites;n++){
     QDP::multi1d<int> coords=QDP::crtesn(n,Layout::lattSize());
    int qdp_index = sub.siteTable()[n];
    //std::cout<< qdp_index <<"/"<<n << "   (" <<coords[0]<<","<< coords[1]<<","<coords[2]<<","<<coords[3]<<")\n";
    QDPIO::cout<< qdp_index <<"/"<<n << "   ("  << coords[0] <<","<< coords[1]<<","<<coords[2]<<","<<coords[3]<<")\n";;

  }

  ********INDEXING TEST**********/

  /*
  for(int linear=0; linear < nodeSites; ++linear)
  {
    multi1d<int> coord = Layout::siteCoords(Layout::nodeNumber(), linear);

    int node   = Layout::nodeNumber(coord);
    int lin    = Layout::linearSiteIndex(coord);
    int icolor = func(coord);

    std::cout << " coord="<<coord<<" node="<<node<<" linear="<<linear<<" col="<<icolor << std::endl;

  }
  */
  // Deep copy host views to device views.
  Kokkos::deep_copy( d_adj_anti_quark_prop, h_adj_anti_quark_prop);
  Kokkos::deep_copy( d_quark_prop1, h_quark_prop1 );
  Kokkos::deep_copy( d_anti_quark_prop, h_anti_quark_prop );

  Kokkos::deep_copy( d_u, h_u );


  Kokkos::deep_copy( d_quark_prop1_shft, h_quark_prop1_shft);
  Kokkos::deep_copy( d_anti_quark_prop_shft, h_anti_quark_prop_shft);

  Kokkos::deep_copy(d_adj_anti_quark_prop_shft, h_adj_anti_quark_prop_shft);
 
  int j_decay = phases.getDir();
  //Parallel for to make the contractions

  int kv = -1;
  int kcv = Nd-2;
  for(int k = 0; k < 1; ++k)
  {
    if( k != j_decay ){   
      int m = 1 << k;
      kv = kv + 1;
      kcv = kcv + 1;

      auto quark_prop1_shft_k = Kokkos::subview( d_quark_prop1_shft,Kokkos::ALL ,k,Kokkos::ALL,Kokkos::ALL,
                                                                      Kokkos::ALL, Kokkos::ALL);
      auto d_anti_quark_prop_shft_k=Kokkos::subview(d_anti_quark_prop_shft,Kokkos::ALL,k,Kokkos::ALL,Kokkos::ALL,
                                                                                           Kokkos::ALL,Kokkos::ALL);
      Kokkos::parallel_for( "Site loop",range_policy(0,numSites), KOKKOS_LAMBDA ( int n ) {
        int gamma_value=m;
        auto u_k=Kokkos::subview(d_u,Kokkos::ALL,k,Kokkos::ALL,Kokkos::ALL);

        kokkos_PropDotGamma(d_tmp_propN,quark_prop1_shft_k ,gamma_value,n);
        kokkos_ColorMatrixDotProp(d_tmp_propN2, u_k, d_tmp_propN, n);


        //kokkos_prop1_dot_prop2(d_tmp_propN3,d_adj_anti_quark_prop,d_tmp_propN2, n);        
        //view_chi_sqN(n,k)=-kokkos_prop_trace(d_tmp_propN3,n);
        view_chi_sqN(n,k)=- kokkos_trace_prop1_dot_prop2(d_adj_anti_quark_prop,d_tmp_propN2, n);



        kokkos_GammaDotProp(d_tmp_propN,gamma_value,d_tmp_propN2,n);
        //kokkos_prop1_dot_prop2(d_tmp_propN4,d_adj_anti_quark_prop,d_tmp_propN, n);
        //view_psi_sqN(n,k)=kokkos_prop_trace(d_tmp_propN4,n);

        view_psi_sqN(n,k)=kokkos_trace_prop1_dot_prop2(d_adj_anti_quark_prop,d_tmp_propN, n);


        kokkos_PropDotGamma(d_tmp_propN,d_anti_quark_prop_shft_k,gamma_value,n);
        kokkos_ColorMatrixDotProp(d_tmp_propN2, u_k, d_tmp_propN, n);
        adj_kokkos_prop(d_tmp_propN4,d_tmp_propN2,n);
        //kokkos_prop1_dot_prop2(d_tmp_propN3,d_tmp_propN4,d_quark_prop1, n);        
        //view_chi_sqN(n,k)+=kokkos_prop_trace(d_tmp_propN3,n);
        view_chi_sqN(n,k)+=kokkos_trace_prop1_dot_prop2(d_tmp_propN4,d_quark_prop1, n);

        kokkos_GammaDotProp(d_tmp_propN,gamma_value,d_tmp_propN2,n);
        adj_kokkos_prop(d_tmp_propN3,d_tmp_propN,n);
        //kokkos_prop1_dot_prop2(d_tmp_propN4,d_tmp_propN3,d_quark_prop1, n);
        //view_psi_sqN(n,k)+=kokkos_prop_trace(d_tmp_propN4,n);
        view_psi_sqN(n,k)+=kokkos_trace_prop1_dot_prop2(d_tmp_propN3,d_quark_prop1, n);
    
        view_chi_sqN(n,k)+=view_psi_sqN(n,k);

      });

    }
  }
  Kokkos::fence();

  std::ofstream out("out.py",std::ios_base::app);

  auto *coutbuf = std::cout.rdbuf();
  std::cout.rdbuf(out.rdbuf());

  if (nodeNumber==0){
    std::cout <<"\n\n*************ZERO*************************";
    for ( int n = 0; n < 1; ++n ) {
        std::cout <<"\nh_view_chi_sqN(0,0)_kokkos= "<< view_chi_sqN(n,0).real()<<"\n";
        std::cout <<"\nh_view_psi_sqN(0,0)_kokkos= "<< view_psi_sqN(n,0).real() << "\n";
    }
    std::cout <<"\n\n*************ZERO*************************";

 }
  std::cout.rdbuf(coutbuf);
  

  //******************END KOKKOS CODE**************

  if ( no_vec_cur < 2 || no_vec_cur > 4 )
    QDP_error_exit("no_vec_cur must be 2 or 3 or 4", no_vec_cur);

  // Initial group
  push(xml, xml_group);

  write(xml, "num_vec_cur", no_vec_cur);

  // Vector currents
  {
    /* Construct the 2*(Nd-1) non-local std::vector-current to rho correlators */
    int kv = -1;
    int kcv = Nd-2;

    for(int k = 0; k < Nd; ++k)
    {
      if( k != j_decay )
      {
	int n = 1 << k;
	kv = kv + 1;
	kcv = kcv + 1;

      }
    }


    /* Construct the O(a) improved std::vector-current to rho correlators,
       if desired */

    if ( no_vec_cur >= 3 )
    {
      kv = 2*Nd-3;
      int jd = 1 << j_decay;

      for(int k = 0; k < Nd; ++k)
      {
	if( k != j_decay )
	{
	  int n = 1 << k;
	  kv = kv + 1;
	  int n1 = n ^ jd;


      Kokkos::parallel_for( "Site loop",range_policy(0,numSites), KOKKOS_LAMBDA ( int nSite ) {
        view_psi_sqN(nSite,k)=0;
        adj_kokkos_prop(d_tmp_propN,d_anti_quark_prop,nSite);
        kokkos_GammaDotProp(d_tmp_propN2,n1,d_quark_prop1,nSite);
        kokkos_PropDotGamma(d_tmp_propN3,d_tmp_propN2,n,nSite);
        //kokkos_prop1_dot_prop2(d_tmp_propN4,d_tmp_propN,d_tmp_propN3, nSite);
        //view_psi_sqN(nSite,k)+=kokkos_prop_trace(d_tmp_propN4,nSite);
        view_psi_sqN(nSite,k)+=kokkos_trace_prop1_dot_prop2(d_tmp_propN,d_tmp_propN3, nSite);       
 
      });
 
    if( k==0 && nodeNumber==0){
      int qdp_index = sub.siteTable()[0];
      std::cout <<"\n\n**************ONE************************";  
      std::cout <<"\nview_psi_sqN(n,k)_kokkos= "<< view_psi_sqN(0,k);
      std::cout <<"\n**************ONE************************\n\n";
    }

	}
      }
    }

       
    /* Construct the local std::vector-current to rho correlators, if desired */

    if ( no_vec_cur >= 4 )
    {
      kv = 3*Nd-4;

      for(int k = 0; k < Nd; ++k)
      {
	if( k != j_decay )
	{
	  int n = 1 << k;
	  kv = kv + 1;


       Kokkos::parallel_for( "Site loop",range_policy(0,numSites), KOKKOS_LAMBDA ( int nSite ) {
        view_psi_sqN(nSite,k)=0;
        adj_kokkos_prop(d_tmp_propN,d_anti_quark_prop,nSite);
        kokkos_GammaDotProp(d_tmp_propN2,n,d_quark_prop1,nSite);
        kokkos_PropDotGamma(d_tmp_propN3,d_tmp_propN2,n,nSite);
        //kokkos_prop1_dot_prop2(d_tmp_propN4,d_tmp_propN,d_tmp_propN3, nSite);
        //view_psi_sqN(nSite,k)+=kokkos_prop_trace(d_tmp_propN4,nSite);
        view_psi_sqN(nSite,k)+=kokkos_trace_prop1_dot_prop2(d_tmp_propN,d_tmp_propN3, nSite);

      });

     if( k==0 && nodeNumber==0){
      int qdp_index = sub.siteTable()[0];
      std::cout <<"\n\n*************SECOND*************************";
      std::cout <<"\nview_psi_sqN(n,k)_kokkos= "<< view_psi_sqN(0,k);
      std::cout <<"\n***************SECOND***********************\n\n";
    }

	}

      }
    }
  }


  //
  // Axial currents
  //
  {
    /* Construct the 2 axial-current to pion correlators */
    int n = G5 ^ (1 << j_decay);

      //#######################################
      auto quark_prop1_shft_jd = Kokkos::subview( d_quark_prop1_shft,Kokkos::ALL ,j_decay,Kokkos::ALL,Kokkos::ALL,
                                                                      Kokkos::ALL, Kokkos::ALL);

      auto d_anti_quark_prop_shft_jd=Kokkos::subview(d_anti_quark_prop_shft,Kokkos::ALL,j_decay,Kokkos::ALL,Kokkos::ALL,
                                                                                           Kokkos::ALL,Kokkos::ALL);

      int gamma_value=n;
      auto u_jd=Kokkos::subview(d_u,Kokkos::ALL,j_decay,Kokkos::ALL,Kokkos::ALL);

      Kokkos::parallel_for( "Site loop",range_policy(0,numSites), KOKKOS_LAMBDA ( int nSite ) {
        kokkos_PropDotGamma(d_tmp_propN,quark_prop1_shft_jd,G5,nSite);
        kokkos_ColorMatrixDotProp(d_tmp_propN2, u_jd, d_tmp_propN, nSite);
        kokkos_PropDotGamma(d_tmp_propN3,d_adj_anti_quark_prop,gamma_value,nSite);
        //kokkos_prop1_dot_prop2(d_tmp_propN4,d_tmp_propN3,d_tmp_propN2, nSite);        
        //view_chi_sqN(nSite,j_decay)=kokkos_prop_trace(d_tmp_propN4,nSite);
        if(nSite==0){
            QDPIO::cout << "new = " << kokkos_trace_prop1_dot_prop2(d_tmp_propN3,d_tmp_propN2, nSite).real()<<"\n";
        }


        view_chi_sqN(nSite,j_decay)=kokkos_trace_prop1_dot_prop2(d_tmp_propN3,d_tmp_propN2, nSite);



        kokkos_PropDotGamma(d_tmp_propN,d_quark_prop1,G5,nSite);
        kokkos_PropDotGamma(d_tmp_propN2,d_adj_anti_quark_prop,gamma_value,nSite);
        //kokkos_prop1_dot_prop2(d_tmp_propN4,d_tmp_propN2,d_tmp_propN, nSite);
        //view_psi_sqN(nSite,j_decay)=kokkos_prop_trace(d_tmp_propN4,nSite);
        view_psi_sqN(nSite,j_decay)=kokkos_trace_prop1_dot_prop2(d_tmp_propN2,d_tmp_propN, nSite);


        kokkos_PropDotGamma(d_tmp_propN,d_anti_quark_prop_shft_jd,G5,nSite);
        kokkos_ColorMatrixDotProp(d_tmp_propN2, u_jd, d_tmp_propN, nSite);
        kokkos_GammaDotProp(d_tmp_propN3,n,d_tmp_propN2,nSite);
        adj_kokkos_prop(d_tmp_propN4,d_tmp_propN3,nSite);
        //kokkos_prop1_dot_prop2(d_tmp_propN,d_tmp_propN4,d_quark_prop1, nSite);        
        //view_chi_sqN(nSite,j_decay)-=kokkos_prop_trace(d_tmp_propN,nSite);
        view_chi_sqN(nSite,j_decay)-=kokkos_trace_prop1_dot_prop2(d_tmp_propN4,d_quark_prop1, nSite); 


     });

    /*
    Kokkos::parallel_for( "Site loop",range_policy(0,numSites), KOKKOS_LAMBDA ( int nSite ) {
        int qdp_index = sub.siteTable()[nSite];
        chi_sq.elem(qdp_index).elem() = view_chi_sqN(nSite,j_decay).real();
    });
    */

    if(nodeNumber==0){
      int qdp_index = sub.siteTable()[0];
      std::cout <<"\n\n*************THIRD*************************";
      std::cout <<"\nview_psi_sqN(n,k)_kokkos= "<< view_psi_sqN(0,j_decay);
      std::cout <<"\nview_chi_sqN(n,k)_kokkos= "<< view_chi_sqN(0,j_decay);
      std::cout <<"\n***************THIRD***********************\n\n";
      std::cout.rdbuf(coutbuf);
    }

  }


  swatch.stop();
  time=swatch.getTimeInSeconds();

  QDPIO::cout << "Time Contractions = " << time << " secs\n";


  pop(xml);  // xml_group
              
  END_CODE();
}

}  // end namespace Chroma
