//  $Id: mesons2_w.cc,v 1.1 2006-07-10 19:53:37 edwards Exp $
//  $Log: mesons2_w.cc,v $
//  Revision 1.1  2006-07-10 19:53:37  edwards
//  A complex version.
//
//  Revision 3.0  2006/04/03 04:59:00  edwards
//  Major overhaul of fermion and gauge action interface. Basically,
//  all fermacts and gaugeacts now carry out  <T,P,Q>  template parameters. These are
//  the fermion type, the "P" - conjugate momenta, and "Q" - generalized coordinates
//  in the sense of Hamilton's equations. The fermbc's have been rationalized to never
//  be over multi1d<T>. The "createState" within the FermionAction is now fixed meaning
//  the "u" fields are now from the coordinate type. There are now "ConnectState" that
//  derive into FermState<T,P,Q> and GaugeState<P,Q>.
//
//  Revision 2.1  2005/11/20 18:29:00  edwards
//  Added parenthesis
//
//  Revision 2.0  2005/09/25 21:04:35  edwards
//  Moved to version 2.0
//
//  Revision 1.21  2005/02/28 03:35:25  edwards
//  Fixed doxygen comments.
//
//  Revision 1.20  2005/01/14 18:42:36  edwards
//  Converted all lib files to be in chroma namespace.
//
//  Revision 1.19  2004/07/28 02:38:04  edwards
//  Changed {START,END}_CODE("foo") to {START,END}_CODE().
//
//  Revision 1.18  2004/02/11 12:51:34  bjoo
//  Stripped out Read() and Write()
//
//  Revision 1.17  2004/02/03 20:47:24  edwards
//  Small code tweaks.
//
//  Revision 1.16  2003/10/01 03:01:39  edwards
//  Removed extraneous include.
//
//  Revision 1.15  2003/09/29 21:31:36  edwards
//  Tiny cosmetic change.
//
//  Revision 1.14  2003/06/24 03:25:27  edwards
//  Changed from nml to xml.
//
//  Revision 1.13  2003/04/02 22:28:22  edwards
//  Changed proto.h to qdp_util.h
//
//  Revision 1.12  2003/04/01 03:27:26  edwards
//  Added const to sftmom.
//
//  Revision 1.11  2003/04/01 02:38:26  edwards
//  Added doxygen comments.
//
//  Revision 1.10  2003/03/14 21:51:54  flemingg
//  Changes the way in which the nml data is output to match what's done
//  in szin.
//
//  Revision 1.9  2003/03/14 17:16:13  flemingg
//  Variant 1 is now working with SftMom::sft().  In arbitrary units,
//  the relative performance seems to be: V1) 7.5, V2) 10, V3) 100.
//
//  Revision 1.8  2003/03/14 05:14:32  flemingg
//  rewrite of mesons_w.cc to use the new SftMom class.  mesons_w.cc still
//  needs to be cleaned up once the best strategy is resolved.  But for now,
//  the library and test program compiles and runs.
//
//  Revision 1.7  2003/03/06 03:38:35  edwards
//  Added start/end_code.
//
//  Revision 1.6  2003/03/06 02:07:12  flemingg
//  Changed the MomList class to eliminate an unneeded class member.
//
//  Revision 1.5  2003/03/06 00:30:14  flemingg
//  Complete rewrite of lib/meas/hadron/mesons_w.cc, including a simple test
//  program in mainprogs/tests built with 'make check' and various other
//  changes to autoconf/make files to support this rewrite.
//

#include "chromabase.h"
#include "util/ft/sftmom.h"
#include "meas/hadron/mesons_w.h"
#include "kokkos_util/kokkos_gamma_funcsTest.h"
#include <chrono>
#include <time.h>
#include <fstream>
#include <stdio.h>

void propToFile(std::string propName, View_prop_gamma_type prop, int n, int g){

  std::cout <<propName<<"=np.zeros(shape=(32,16,4,4,3,3))\n";
  for ( int i = 0; i < 4; ++i ) {
    for ( int j = 0; j < 4; ++j ) {
      for ( int k = 0; k < 3; ++k ) {
       for ( int l = 0; l < 3; ++l ) {
            std::cout <<propName<<"["<<n<<","<<g<<","<<i<<","<<j<<","<<k<<","<<l<<"]=";
            std::cout << prop(n,g,i,j,k,l).real() << "\n";
       }
      }
    }
  }
}

void propToFile(std::string propName, View_prop_type prop, int n){
  std::cout <<propName<<"=np.zeros(shape=(32,4,4,3,3))\n";
  for ( int i = 0; i < 4; ++i ) {
    for ( int j = 0; j < 4; ++j ) {
      for ( int k = 0; k < 3; ++k ) {
       for ( int l = 0; l < 3; ++l ) {
            std::cout <<propName<<"["<<n<<","<<i<<","<<j<<","<<k<<","<<l<<"]=";
            std::cout << prop(n,i,j,k,l).real() << "\n";
       }
      }
    }
  } 
}

void propToFileQDP(std::string propName, LatticePropagator temp1, int n){
  std::cout <<propName<<"=np.zeros(shape=(32,4,4,3,3))\n";
  for ( int i = 0; i < 4; ++i ) {
    for ( int j = 0; j < 4; ++j ) {
      for ( int k = 0; k < 3; ++k ) {
       for ( int l = 0; l < 3; ++l ) {
            std::cout <<propName<<"["<<n<<","<<i<<","<<j<<","<<k<<","<<l<<"]=";
            std::cout << temp1.elem(n).elem(i,j).elem(k,l).real() << "\n";
       }
      }
    }
  } 
}


namespace Chroma {

//! Meson 2-pt functions
/* This routine is specific to Wilson fermions!
 *
 * Construct meson propagators and writes in COMPLEX 
 *
 * The two propagators can be identical or different.
 *
 * \param quark_prop_1  first quark propagator ( Read )
 * \param quark_prop_2  second (anti-) quark propagator ( Read )
 * \param t0            timeslice coordinate of the source ( Read )
 * \param phases        object holds list of momenta and Fourier phases ( Read )
 * \param xml           xml file object ( Write )
 * \param xml_group     std::string used for writing xml data ( Read )
 *
 *        ____
 *        \
 * m(t) =  >  < m(t_source, 0) m(t + t_source, x) >
 *        /
 *        ----
 *          x
 */

void mesons2(const LatticePropagator& quark_prop_1,
	     const LatticePropagator& quark_prop_2,
	     const SftMom& phases,
	     int t0,
	     XMLWriter& xml,
	     const std::string& xml_group)
{
  START_CODE();

  
  // Length of lattice in decay direction
  int length = phases.numSubsets();
  // Construct the anti-quark propagator from quark_prop_2
  int G5 = Ns*Ns-1;
  LatticePropagator anti_quark_prop =  Gamma(G5) * quark_prop_2 * Gamma(G5);


  QDPIO::cout << "\n\n***************\nUsing Kokkos Contractions GPU\n***************\n\n";



  double time;
  StopWatch swatch;
  swatch.reset();
  swatch.start();


  //For CPU
  //using MemSpace=Kokkos::HostSpace; 

  //For GPU
  using MemSpace=Kokkos::Experimental::HIP;

  using ExecSpace = MemSpace::execution_space;
  using range_policy = Kokkos::RangePolicy<ExecSpace>;

  const QDP::Subset& sub = QDP::rb[0];

  // Allocate y, x vectors and Matrix A on device.
  int numSites = sub.siteTable().size();

  View_corrs_type  view_corr("corr", numSites);

  LatticePropagator adj_anti_quark_prop =  adj(anti_quark_prop);
 
  View_prop_type d_adj_anti_quark_prop("d_adj_anti-quark", numSites);
  View_prop_type d_quark_prop1("d_quark", numSites);
  View_Latt_spin_matrix_type s1("s1",numSites);
  View_Latt_spin_matrix_type s2("s2",numSites);



  // Create host mirrors of device views.
  View_prop_type::HostMirror h_adj_anti_quark_prop = Kokkos::create_mirror_view( d_adj_anti_quark_prop );
  View_prop_type::HostMirror h_quark_prop1 = Kokkos::create_mirror_view( d_quark_prop1 );
  View_corrs_type::HostMirror h_view_corr = Kokkos::create_mirror_view( view_corr );

  // This variant uses the function SftMom::sft() to do all the work
  // computing the Fourier transform of the meson correlation function
  // inside the class SftMom where all the of the Fourier phases and
  // momenta are stored.  It's primary disadvantage is that it
  // requires more memory because it does all of the Fourier transforms
  // at the same time.
   
 
 // Loop over gamma matrix insertions

  //Initialize kokkos props
  //Kokkos::parallel_for( "Site loop",range_policy(0,numSites), KOKKOS_LAMBDA ( int n ) {

  Kokkos::parallel_for( "Site loop",Kokkos::RangePolicy<Kokkos::HostSpace::execution_space>(0,numSites), KOKKOS_LAMBDA ( int n ) {
    int qdp_index = sub.siteTable()[n];
    for ( int i = 0; i  < 4; ++i ) {
      for ( int j = 0; j < 4; ++j ) {
        for ( int c1 = 0; c1  < 3; ++c1) {
          for  ( int c2 = 0; c2  < 3; ++c2 ) {
             h_adj_anti_quark_prop(n,i,j,c1,c2)=Kokkos::complex<double>( adj_anti_quark_prop.elem(qdp_index).elem(i,j).elem(c1,c2).real(),adj_anti_quark_prop.elem(qdp_index).elem(i,j).elem(c1,c2).imag());
             h_quark_prop1(n,i,j,c1,c2)=Kokkos::complex<double>( quark_prop_1.elem(qdp_index).elem(i,j).elem(c1,c2).real(),quark_prop_1.elem(qdp_index).elem(i,j).elem(c1,c2).imag());
          }
        }
      }
    }
  });


  // Deep copy host views to device views.
  Kokkos::deep_copy( d_adj_anti_quark_prop, h_adj_anti_quark_prop);
  Kokkos::deep_copy( d_quark_prop1, h_quark_prop1 );
    
  //Parallel for to make the contractions
  Kokkos::parallel_for( "Site loop",range_policy(0,numSites), KOKKOS_LAMBDA ( int n ) {
  // KokkosPropDotGamma(View_prop_type r,int gamma_value,View_prop_gamma_type d, int n, int c1, int c2)
    for ( int c1 = 0; c1  < 3; ++c1 ) { 
          for ( int c2 = 0; c2 < 3; ++c2 ) {
            for (int gamma_value = 0; gamma_value < 16; gamma_value++){
                KokkosPropDotGamma(d_adj_anti_quark_prop,gamma_value,s1,n, c1, c2);
                KokkosPropDotGamma(d_quark_prop1,gamma_value,s2,n, c2, c1);
                for ( int i = 0; i  < 4; ++i ) {
                for  ( int j = 0; j  < 4; ++j ) {
                   view_corr(n,gamma_value) += s1(n,i,j)*s2(n,j,i);               
                }
              }
          }
      }
    }
  });

  Kokkos::fence();
  
  swatch.stop();
  time=swatch.getTimeInSeconds();

  Kokkos::deep_copy( h_view_corr, view_corr );

  QDPIO::cout << "Time Contractions = " << time << " secs\n";
  /*
  std::ofstream out("out.py",std::ios_base::app);

  auto *coutbuf = std::cout.rdbuf();
  std::cout.rdbuf(out.rdbuf());
  std::cout << "import numpy as np\n";


  std::cout <<"C(n,0)_Kokkos= ";
  for ( int n = 0; n < 9; ++n ) {
      std::cout <<h_view_corr(n,0).real()<<" ";
  }

  std::cout.rdbuf(coutbuf); 
  */
  END_CODE();
}

}  // end namespace Chroma
