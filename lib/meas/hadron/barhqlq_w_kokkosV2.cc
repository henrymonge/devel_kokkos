/*! \file
 *  \brief Heavy-light baryon 2-pt functions
 */


#include "meas/hadron/barhqlq_w_kokkos.h"
#include "meas/hadron/barspinmat_w.h"
#include "kokkos_util/kokkos_quark_ops.h"
#include "kokkos_util/kokkos_gamma_funcsTest.h"


namespace Chroma 
{

  //! Baryon 2pt contractions
  /*! \ingroup hadron */
 

   namespace  Baryon2PtContractions
  {
    //! Cascade 2-pt
    /*! \ingroup hadron */
     //KOKKOS_INLINE_FUNCTION
     void kokkos_xi2pt(auto k_b_prop, auto quark_propagator_1,
                                   auto quark_propagator_2,
                                   View_spin_matrix_type T, View_spin_matrix_type sp)
    {
#if QDP_NC == 3

      const QDP::Subset& sub = QDP::all;
      int numSites = sub.siteTable().size();

      //auxiliary views
      View_prop_type tmp1("tmp1",numSites);
      View_prop_type tmp2("tmp2",numSites); 
      View_prop_type di_quark("di_quark",numSites);
      View_Latt_spin_matrix_type stmp1("stmp1",numSites);
      View_Latt_spin_matrix_type stmp2("stmp2",numSites);
      View_Latt_color_matrix_type ctmp("ctmp",numSites);

      Kokkos::parallel_for( "Site loop",range_policy(0,numSites), KOKKOS_LAMBDA ( int nSite ) {
        auto q_prop_1 = Kokkos::subview(quark_propagator_1,nSite,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto q_prop_2 = Kokkos::subview(quark_propagator_2,nSite,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL); 
        auto sub_di_quark = Kokkos::subview(di_quark,nSite,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto sub_tmp1 = Kokkos::subview(tmp1,nSite,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto sub_tmp2 = Kokkos::subview(tmp2,nSite,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto sub_stmp1 = Kokkos::subview(stmp1,nSite,Kokkos::ALL,Kokkos::ALL);
        auto sub_stmp2 = Kokkos::subview(stmp2,nSite,Kokkos::ALL,Kokkos::ALL);
        auto sub_ctmp = Kokkos::subview(ctmp,nSite,Kokkos::ALL,Kokkos::ALL);

        // di_quark = quarkContract13(quark_propagator_1 * sp,sp * quark_propagator_2);
        kokkos_SitePropDotSpinMatrix(nSite,tmp1, quark_propagator_1, sp);        
        kokkos_SpinMatrixDotSiteProp(nSite,tmp2, sp, quark_propagator_2);
        kokkos_quarkContract13(sub_di_quark, sub_tmp1, sub_tmp2);
      
        //trace(T * traceColor(quark_propagator_1 * di_quark))
        kokkos_traceColor_prop1_dot_prop2(sub_stmp2,q_prop_1, sub_di_quark);
        kokkos_SpinMatrixProduct(sub_stmp1,T,sub_stmp2);        
        k_b_prop(nSite) = kokkos_spinMatrix_trace(sub_stmp1);


        //trace(T * traceColor(quark_propagator_1 * traceSpin(di_quark))
        kokkos_propSpinTrace(sub_ctmp, sub_di_quark);
        kokkos_PropColorMatrixProduct(sub_tmp2, q_prop_1, sub_ctmp); 
        kokkos_SpinMatrixDotSiteProp(nSite,tmp1,T,tmp2);
        k_b_prop(nSite) += kokkos_site_prop_trace(sub_tmp1);
        
     }); //End kokkos parallel_for
    Kokkos::fence();
       //Tag tClock.stop();
       //double time = timer.seconds();
       //QDPIO::cout << "Total in kokkos_xi2pt ins conts = " << tClock.getTimeInSeconds() << " secs" << std::endl;


#endif
    }


    //! Sigma 2-pt
    //! \ingroup hadron //
    KOKKOS_INLINE_FUNCTION void kokkos_sigma2pt(auto k_b_prop, auto quark_propagator_1,
                                   auto quark_propagator_2,
                                   View_spin_matrix_type T, View_spin_matrix_type sp)
    {

#if QDP_NC == 3
      const QDP::Subset& sub = QDP::all;
      int numSites = sub.siteTable().size();

      int nodeNumber=Layout::nodeNumber();     
  
      //auxiliary views
      View_prop_type tmp1("tmp1",numSites);
      View_prop_type tmp2("tmp2",numSites); 
      View_prop_type di_quark("di_quark",numSites);
      View_Latt_spin_matrix_type stmp1("stmp1",numSites);
      View_Latt_spin_matrix_type stmp2("stmp2",numSites);
      View_Latt_color_matrix_type ctmp("ctmp",numSites);

      Kokkos::parallel_for( "Site loop",range_policy(0,numSites), KOKKOS_LAMBDA ( int nSite ) {
        auto q_prop_1 = Kokkos::subview(quark_propagator_1,nSite,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto q_prop_2 = Kokkos::subview(quark_propagator_2,nSite,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL); 
        auto sub_di_quark = Kokkos::subview(di_quark,nSite,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto sub_tmp1 = Kokkos::subview(tmp1,nSite,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto sub_tmp2 = Kokkos::subview(tmp2,nSite,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto sub_stmp1 = Kokkos::subview(stmp1,nSite,Kokkos::ALL,Kokkos::ALL);
        auto sub_stmp2 = Kokkos::subview(stmp2,nSite,Kokkos::ALL,Kokkos::ALL);
        auto sub_ctmp = Kokkos::subview(ctmp,nSite,Kokkos::ALL,Kokkos::ALL);

        kokkos_SitePropDotSpinMatrix(nSite,tmp1, quark_propagator_1, sp);
        kokkos_SpinMatrixDotSiteProp(nSite,tmp2, sp, quark_propagator_2);
        kokkos_quarkContract13(sub_di_quark, sub_tmp1, sub_tmp2);
      
        //trace(T * traceColor(quark_propagator_2 * di_quark)))
        kokkos_traceColor_prop1_dot_prop2(sub_stmp2,q_prop_2, sub_di_quark);
        kokkos_SpinMatrixProduct(sub_stmp1,T,sub_stmp2);        
        k_b_prop(nSite) = kokkos_spinMatrix_trace(sub_stmp1);


       //trace(T * traceColor(quark_propagator_2 * traceSpin(di_quark)))
        kokkos_propSpinTrace(sub_ctmp, sub_di_quark);
        kokkos_PropColorMatrixProduct(sub_tmp2, q_prop_2, sub_ctmp); 
        kokkos_SpinMatrixDotSiteProp(nSite,tmp1,T,tmp2);
        k_b_prop(nSite) += kokkos_site_prop_trace(sub_tmp1);

     }); //End kokkos parallel_for
    Kokkos::fence();
#endif
    }


    //! Lambda 2-pt
    //! \ingroup hadron //

    KOKKOS_INLINE_FUNCTION void  kokkos_lambda2pt(auto k_b_prop, auto quark_propagator_1,
                                                  auto quark_propagator_2,
                                                  View_spin_matrix_type T, View_spin_matrix_type sp)
  
    {
#if QDP_NC == 3

      // WARNING: I'm not convinced the original SZIN version (or this version) is correct!
      const QDP::Subset& sub = QDP::all;
      int numSites = sub.siteTable().size();

      //auxiliary views
      View_prop_type tmp1("tmp1",numSites);
      View_prop_type tmp2("tmp2",numSites); 
      View_prop_type di_quark("di_quark",numSites);
      View_Latt_spin_matrix_type stmp1("stmp1",numSites);
      View_Latt_spin_matrix_type stmp2("stmp2",numSites);
      View_Latt_color_matrix_type ctmp("ctmp",numSites);

      Kokkos::parallel_for( "Site loop",range_policy(0,numSites), KOKKOS_LAMBDA ( int nSite ) {
        auto q_prop_1 = Kokkos::subview(quark_propagator_1,nSite,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto q_prop_2 = Kokkos::subview(quark_propagator_2,nSite,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL); 
        auto sub_di_quark = Kokkos::subview(di_quark,nSite,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto sub_tmp1 = Kokkos::subview(tmp1,nSite,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto sub_tmp2 = Kokkos::subview(tmp2,nSite,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto sub_stmp1 = Kokkos::subview(stmp1,nSite,Kokkos::ALL,Kokkos::ALL);
        auto sub_stmp2 = Kokkos::subview(stmp2,nSite,Kokkos::ALL,Kokkos::ALL);
        auto sub_ctmp = Kokkos::subview(ctmp,nSite,Kokkos::ALL,Kokkos::ALL);


        kokkos_SitePropDotSpinMatrix(nSite,tmp1, quark_propagator_2, sp);
        kokkos_SpinMatrixDotSiteProp(nSite,tmp2, sp, quark_propagator_2);
        kokkos_quarkContract13(sub_di_quark, sub_tmp1, sub_tmp2);


        //b_prop  = trace(T * traceColor(quark_propagator_1 * traceSpin(di_quark)))
        kokkos_propSpinTrace(sub_ctmp, sub_di_quark);
        kokkos_PropColorMatrixProduct(sub_tmp2, q_prop_1, sub_ctmp);
        kokkos_SpinMatrixDotSiteProp(nSite,tmp1,T,tmp2);
        k_b_prop(nSite) = kokkos_site_prop_trace(sub_tmp1);

        //trace(T * traceColor(quark_propagator_1 * di_quark))
        kokkos_traceColor_prop1_dot_prop2(sub_stmp2,q_prop_1, sub_di_quark);
        kokkos_SpinMatrixProduct(sub_stmp1,T,sub_stmp2);
        k_b_prop(nSite) += kokkos_spinMatrix_trace(sub_stmp1);

        //di_quark = quarkContract13(quark_propagator_2 * sp,sp * quark_propagator_1);
        kokkos_SitePropDotSpinMatrix(nSite,tmp1, quark_propagator_2, sp);
        kokkos_SpinMatrixDotSiteProp(nSite,tmp2, sp, quark_propagator_1);
        kokkos_quarkContract13(sub_di_quark, sub_tmp1, sub_tmp2);
        
        //b_prop += trace(T * traceColor(quark_propagator_2 * di_quark));
        kokkos_traceColor_prop1_dot_prop2(sub_stmp2,q_prop_2, sub_di_quark);
        kokkos_SpinMatrixProduct(sub_stmp1,T,sub_stmp2);
        k_b_prop(nSite) += kokkos_spinMatrix_trace(sub_stmp1);

     }); //End kokkos parallel_for

    Kokkos::fence();


#endif
    }

    //! Lambda 2-pt
    //! \ingroup hadron //
    KOKKOS_INLINE_FUNCTION void  kokkos_lambdaNaive2pt(auto k_b_prop, auto quark_propagator_1,
                                                  auto quark_propagator_2,
                                                  View_spin_matrix_type T, View_spin_matrix_type sp)
   {
#if QDP_NC == 3

      const QDP::Subset& sub = QDP::all;
      int numSites = sub.siteTable().size();

      //auxiliary views
      View_prop_type tmp1("tmp1",numSites);
      View_prop_type tmp2("tmp2",numSites); 
      View_prop_type di_quark("di_quark",numSites);
      View_Latt_color_matrix_type ctmp("ctmp",numSites);

      Kokkos::parallel_for( "Site loop",range_policy(0,numSites), KOKKOS_LAMBDA ( int nSite ) {
        auto q_prop_1 = Kokkos::subview(quark_propagator_1,nSite,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto q_prop_2 = Kokkos::subview(quark_propagator_2,nSite,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL); 
        auto sub_di_quark = Kokkos::subview(di_quark,nSite,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto sub_tmp1 = Kokkos::subview(tmp1,nSite,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto sub_tmp2 = Kokkos::subview(tmp2,nSite,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto sub_ctmp = Kokkos::subview(ctmp,nSite,Kokkos::ALL,Kokkos::ALL);

         //di_quark = quarkContract13(quark_propagator_2 * sp,sp * quark_propagator_2);
        kokkos_SitePropDotSpinMatrix(nSite,tmp1, quark_propagator_2, sp);
        kokkos_SpinMatrixDotSiteProp(nSite,tmp2, sp, quark_propagator_2);
        kokkos_quarkContract13(sub_di_quark, sub_tmp1, sub_tmp2);

        //LatticeComplex(trace(T * traceColor(quark_propagator_1 * traceSpin(di_quark))));
        kokkos_propSpinTrace(sub_ctmp, sub_di_quark);
        kokkos_PropColorMatrixProduct(sub_tmp2, q_prop_1, sub_ctmp);
        kokkos_SpinMatrixDotSiteProp(nSite,tmp1,T,tmp2);
        k_b_prop(nSite) = kokkos_site_prop_trace(sub_tmp1);

      }); //End kokkos parallel_for
    Kokkos::fence();
#endif
    }

    //! Delta 2-pt
    //! \ingroup hadron //

    KOKKOS_INLINE_FUNCTION void  kokkos_sigmast2pt(auto k_b_prop, auto quark_propagator_1,
                                                  auto quark_propagator_2,
                                                  View_spin_matrix_type T, View_spin_matrix_type sp)
    {
#if QDP_NC == 3

      const QDP::Subset& sub = QDP::all;
      int numSites = sub.siteTable().size();

      //auxiliary views
      View_prop_type tmp1("tmp1",numSites);
      View_prop_type tmp2("tmp2",numSites); 
      View_prop_type di_quark("di_quark",numSites);
      View_Latt_spin_matrix_type stmp1("stmp1",numSites);
      View_Latt_spin_matrix_type stmp2("stmp2",numSites);
      View_Latt_color_matrix_type ctmp("ctmp",numSites);

      Kokkos::parallel_for( "Site loop",range_policy(0,numSites), KOKKOS_LAMBDA ( int nSite ) {
        auto q_prop_1 = Kokkos::subview(quark_propagator_1,nSite,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto q_prop_2 = Kokkos::subview(quark_propagator_2,nSite,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL); 
        auto sub_di_quark = Kokkos::subview(di_quark,nSite,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto sub_tmp1 = Kokkos::subview(tmp1,nSite,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto sub_tmp2 = Kokkos::subview(tmp2,nSite,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto sub_stmp1 = Kokkos::subview(stmp1,nSite,Kokkos::ALL,Kokkos::ALL);
        auto sub_stmp2 = Kokkos::subview(stmp2,nSite,Kokkos::ALL,Kokkos::ALL);
        auto sub_ctmp = Kokkos::subview(ctmp,nSite,Kokkos::ALL,Kokkos::ALL);

        //di_quark = quarkContract13(quark_propagator_1 * sp,sp * quark_propagator_2);
        kokkos_SitePropDotSpinMatrix(nSite,tmp1, quark_propagator_1, sp);
        kokkos_SpinMatrixDotSiteProp(nSite,tmp2, sp, quark_propagator_2);
        kokkos_quarkContract13(sub_di_quark, sub_tmp1, sub_tmp2);

        //b_prop  = trace(T * traceColor(quark_propagator_2 * di_quark))
        kokkos_traceColor_prop1_dot_prop2(sub_stmp2,q_prop_2, sub_di_quark);
        kokkos_SpinMatrixProduct(sub_stmp1,T,sub_stmp2);
        k_b_prop(nSite) = kokkos_spinMatrix_trace(sub_stmp1);

        //b_prop += trace(T * traceColor(quark_propagator_2 * traceSpin(di_quark)));
        kokkos_propSpinTrace(sub_ctmp, sub_di_quark);
        kokkos_PropColorMatrixProduct(sub_tmp2, q_prop_2, sub_ctmp);
        kokkos_SpinMatrixDotSiteProp(nSite,tmp1,T,tmp2);
        k_b_prop(nSite) += kokkos_site_prop_trace(sub_tmp1);

        //di_quark = quarkContract13(quark_propagator_2 * sp,sp * quark_propagator_1);
        kokkos_SitePropDotSpinMatrix(nSite,tmp1, quark_propagator_2, sp);
        kokkos_SpinMatrixDotSiteProp(nSite,tmp2, sp, quark_propagator_1);
        kokkos_quarkContract13(sub_di_quark, sub_tmp1, sub_tmp2);

       //b_prop += trace(T * traceColor(quark_propagator_2 * di_quark));
        kokkos_traceColor_prop1_dot_prop2(sub_stmp2,q_prop_2, sub_di_quark);
        kokkos_SpinMatrixProduct(sub_stmp1,T,sub_stmp2);
        k_b_prop(nSite) += kokkos_spinMatrix_trace(sub_stmp1);

        //di_quark = quarkContract13(quark_propagator_2 * sp,sp * quark_propagator_2);
        kokkos_SitePropDotSpinMatrix(nSite,tmp1, quark_propagator_2, sp);
        kokkos_SpinMatrixDotSiteProp(nSite,tmp2, sp, quark_propagator_2);
        kokkos_quarkContract13(sub_di_quark, sub_tmp1, sub_tmp2);
     
        //b_prop += trace(T * traceColor(quark_propagator_1 * di_quark));
        kokkos_traceColor_prop1_dot_prop2(sub_stmp2,q_prop_1, sub_di_quark);
        kokkos_SpinMatrixProduct(sub_stmp1,T,sub_stmp2);
        k_b_prop(nSite) += kokkos_spinMatrix_trace(sub_stmp1);
       
        // b_prop *= 2;
        k_b_prop(nSite) *= 2;

        //trace(T * traceColor(quark_propagator_1 * traceSpin(di_quark)));
        kokkos_propSpinTrace(sub_ctmp, sub_di_quark);
        kokkos_PropColorMatrixProduct(sub_tmp2, q_prop_1, sub_ctmp);
        kokkos_SpinMatrixDotSiteProp(nSite,tmp1,T,tmp2);
        k_b_prop(nSite) += kokkos_site_prop_trace(sub_tmp1);

      }); //End kokkos parallel_for
    Kokkos::fence(); 
#endif
    }

  }  // namespace  Baryon2PtContractions
  

  //! Heavy-light baryon 2-pt functions
  /*!
   * \ingroup hadron
   *
   * This routine is specific to Wilson fermions! 
   *
   * Construct baryon propagators for the Proton and the Delta^+ with
   * degenerate "u" and "d" quarks, as well as the Lambda for, in
   * addition, a degenerate "s" quark. For these degenerate quarks, the
   * Lambda is degenerate with the Proton, but we keep it for compatibility
   * with the sister routine that treats non-degenerate quarks.

   * The routine optionally computes time-charge reversed baryons and adds them
   * in for increased statistics.

   * \param propagator_1   "s" quark propagator ( Read )
   * \param propagator_2   "u" quark propagator ( Read )
   * \param t0             cartesian coordinates of the source ( Read )
   * \param bc_spec        boundary condition for spectroscopy ( Read )
   * \param time_rev       add in time reversed contribution if true ( Read )
   * \param phases         object holds list of momenta and Fourier phases ( Read )
   * \param xml            xml file object ( Read )
   * \param xml_group      group name for xml data ( Read )
   *
   */


  void barhqlq(View_prop_type propagator_1,
           View_prop_type propagator_2,
           View_LatticeComplex1d d_phases, multi1d<bool> doSet,View_int_2d d_sft_sets,
           int t0, int bc_spec, bool time_rev,
           XMLWriter& xml,
           const std::string& xml_group)
  {
    START_CODE();
   
  QDPIO::cout << "\n\n***************\nUsing Kokkos barhqlq Code GPU\n***************\n\n";

  double time;
  StopWatch swatch;
  swatch.reset();
  swatch.start();

    if ( Ns != 4 || Nc != 3 )		/* Code is specific to Ns=4 and Nc=3. */
      return;

  
   const QDP::Subset& sub = QDP::all;
   int numSites = sub.siteTable().size();

    multi3d<DComplex> bardisp1;
    multi3d<DComplex> bardisp2;


  swatch.reset();
  swatch.start();
    
    barhqlq(propagator_1, propagator_2, d_phases, doSet, d_sft_sets, bardisp1);


  swatch.stop();
  time=swatch.getTimeInSeconds();

  QDPIO::cout << "Kokkos barhqlq contraction time  = " << time << " secs\n";


  
  swatch.reset();
  swatch.start();

    // Possibly add in a time-reversed contribution
    bool time_revP = (bc_spec*bc_spec == 1) ? time_rev : false;

    
    if (time_revP)
    {
      QDPIO::cout << "\n\n************\nDoing time reverse\n************\n\n";
      // Time-charge reverse the quark propagators //
      // S_{CT} = gamma_5 gamma_4 = gamma_1 gamma_2 gamma_3 = Gamma(7) //

      View_prop_type d_tmp_prop("d_tmp_prop", numSites);
      View_prop_type d_q1_tmp("d_q1_tmp", numSites);
      View_prop_type d_q2_tmp("d_q2_tmp", numSites); 

      //LatticePropagator q1_tmp = - (Gamma(7) * propagator_1 * Gamma(7));
      //LatticePropagator q2_tmp = - (Gamma(7) * propagator_2 * Gamma(7));

      //barhqlq(q1_tmp, q2_tmp, phases, bardisp2);

      //NEED TO ADD THE SIGN CHANGE
      //
      Kokkos::parallel_for( "Site loop",range_policy(0,numSites), KOKKOS_LAMBDA ( int nSite ) {
        kokkos_GammaDotProp(d_tmp_prop,7,propagator_1,nSite);
        kokkos_PropDotGamma(d_q1_tmp,d_tmp_prop,7,nSite);

        kokkos_GammaDotProp(d_tmp_prop,7,propagator_2,nSite);
        kokkos_PropDotGamma(d_q2_tmp,d_tmp_prop,7,nSite);
        kokkos_multiply_view_prop(-1, d_q1_tmp, nSite);
        kokkos_multiply_view_prop(-1, d_q2_tmp, nSite);
      });
    Kokkos::fence();
      barhqlq(d_q1_tmp, d_q2_tmp, d_phases, doSet, d_sft_sets, bardisp2);
      //barhqlq(d_q1_tmp, d_q2_tmp, phases, bardisp2);
    }    
  /*
  swatch.reset();
  swatch.start();
  
    int num_baryons = bardisp1.size3();
    int num_mom = bardisp1.size2();
    int length  = bardisp1.size1();

    // Loop over baryons
    XMLArrayWriter xml_bar(xml,num_baryons);
    push(xml_bar, xml_group);
   
    for(int baryons = 0; baryons < num_baryons; ++baryons)
    {
      push(xml_bar);     // next array element
      write(xml_bar, "baryon_num", baryons);

      // Loop over sink momenta
      XMLArrayWriter xml_sink_mom(xml_bar,num_mom);
      push(xml_sink_mom, "momenta");

      for(int sink_mom_num = 0; sink_mom_num < num_mom; ++sink_mom_num)
      {
	push(xml_sink_mom);
	write(xml_sink_mom, "sink_mom_num", sink_mom_num) ;
	//write(xml_sink_mom, "sink_mom", phases.numToMom(sink_mom_num)) ;

	multi1d<Complex> barprop(length);

	// forward //
	for(int t = 0; t < length; ++t)
	{
	  int t_eff = (t - t0 + length) % length;
	    
	  if ( bc_spec < 0 && (t_eff+t0) >= length)
	    barprop[t_eff] = -bardisp1[baryons][sink_mom_num][t];
	  else
	    barprop[t_eff] =  bardisp1[baryons][sink_mom_num][t];
	}


	if (time_revP)
	{
	  // backward 
	  for(int t = 0; t < length; ++t)
	  {
	    int t_eff = (length - t + t0) % length;
	
	    if ( bc_spec < 0 && (t_eff-t0) > 0)
	    {
	      barprop[t_eff] -= bardisp2[baryons][sink_mom_num][t];
	      barprop[t_eff] *= 0.5;
	    }
	    else
	    {
	      barprop[t_eff] += bardisp2[baryons][sink_mom_num][t];
	      barprop[t_eff] *= 0.5;
	    }
	  }
	}
    
    
	write(xml_sink_mom, "barprop", barprop);
	pop(xml_sink_mom);
      } // end for(sink_mom_num)
 
      pop(xml_sink_mom);
      pop(xml_bar);
    } // end for(gamma_value)

    pop(xml_bar);
  
  swatch.stop();
  time=swatch.getTimeInSeconds();
  */
  //QDPIO::cout << "Time after barqh Contractions = " << time << " secs\n";
  
    END_CODE();
  }

  //! Heavy-light baryon 2-pt functions
  /*!
   * \ingroup hadron
   *
   * This routine is specific to Wilson fermions! 
   *
   *###########################################################################
   * WARNING: No symmetrization over the spatial part of the wave functions   #
   *          is performed. Therefore, if this routine is called with         #
   *          "shell-sink" quark propagators of different widths the          #
   *          resulting octet baryons may have admixters of excited           #
   *          decouplet baryons with mixed symmetric spatial wave functions,  #
   *          and vice-versa!!!                                               #
   *###########################################################################

   * Construct heavy-light baryon propagators with two "u" quarks and
   * one separate "s" quark for the Sigma^+, the Lambda and the Sigma^{*+}.
   * In the Lambda we take the "u" and "d" quark as degenerate!

   * The routine also computes time-charge reversed baryons and adds them
   * in for increased statistics.

   * \param quark_propagator_1   "s" quark propagator ( Read )
   * \param quark_propagator_2   "u" quark propagator ( Read )
   * \param barprop              baryon propagator ( Modify )
   * \param phases               object holds list of momenta and Fourier phases ( Read )
   *
   *        ____
   *        \
   * b(t) =  >  < b(t_source, 0) b(t + t_source, x) >
   *        /                    
   *        ----
   *          x

   * For the Sigma^+ we take

   * |S_1, s_z=1/2> = (s C gamma_5 u) "u_up"

   * for the Lambda

   * |L_1, s_z=1/2> = 2*(u C gamma_5 d) "s_up" + (s C gamma_5 d) "u_up"
   *                  + (u C gamma_5 s) "d_up"

   * and for the Sigma^{*+}

   * |S*_1, s_z=3/2> = 2*(s C gamma_- u) "u_up" + (u C gamma_- u) "s_up".

   * We have put "q_up" in quotes, since this is meant in the Dirac basis,
   * not in the 'DeGrand-Rossi' chiral basis used in the program!
   * In gamma_- we ignore a factor sqrt(2).

   * For all baryons we compute a 'B_2' that differs from the 'B_1' above
   * by insertion of a gamma_4 between C and the gamma_{5,-}.
   * And finally, we also compute the non-relativistic baryons, 'B_3',
   * which up to a factor 1/2 are just the difference B_1 - B_2, as can
   * be seen by projecting to the "upper" components in the Dirac basis,
   * achieved by (1 + gamma_4)/2 q, for quark q.

   * The Sigma^+_k is baryon 3*(k-1), the Lambda_k is baryon 3*(k-1)+1
   * and the Sigma^{*+}_k is baryon 3*(k-1)+2.

   * We are using a chiral basis for the Dirac matrices (gamma_5 diagonal).
   * Therefore a spin-up quark in the Dirac basis corresponds to
   * 1/sqrt(2) * ( - q_1 - q_3 ) in this chiral basis. We shall neglect
   * the sign and the 1/sqrt(2) here.
   * The projection on "spin_up" is done with the projector "T". 
   */

  void barhqlq(View_prop_type quark_propagator_1,
           View_prop_type quark_propagator_2,
           View_LatticeComplex1d d_phases, multi1d<bool> doSet,View_int_2d d_sft_sets,
           multi3d<DComplex>& barprop)
  {
    START_CODE();


    double total_time=0;
	StopWatch tClock;
     tClock.reset();
     tClock.start();


    //QDPIO::cout << "Now running kokkos barhqlq code\n";
    int nodeNumber=Layout::nodeNumber();
    const QDP::Subset& sub = QDP::all;
    int numSites = sub.siteTable().size();
    // Length of lattice in decay direction

    //int length = phases.numSubsets() ;
    int length = d_phases.extent(1);

    if ( Ns != 4 || Nc != 3 )		/* Code is specific to Ns=4 and Nc=3. */
      return;

    // Setup the return stuff
    const int num_baryons = 17;
    ///int num_mom = phases.numMom();
    int num_mom = d_phases.extent(0);

    barprop.resize(num_baryons,num_mom,length);

     
    // T_mixed = (1 + \Sigma_3)*(1 + gamma_4) / 2 
    //         = (1 + Gamma(8) - i G(3) - i G(11)) / 2
    SpinMatrix T_mixed = BaryonSpinMats::Tmixed();

    // T_unpol = (1/2)(1 + gamma_4)
    SpinMatrix T_unpol = BaryonSpinMats::Tunpol();

    // C gamma_5 = Gamma(5)
    SpinMatrix Cg5 = BaryonSpinMats::Cg5();

    // C gamma_5 gamma_4 = - Gamma(13)
    SpinMatrix Cg5g4 = BaryonSpinMats::Cg5g4();

    // C g_5 NR = (1/2)*C gamma_5 * ( 1 + g_4 )
    SpinMatrix Cg5NR = BaryonSpinMats::Cg5NR();

    SpinMatrix BarSMat = BaryonSpinMats::Cgm();
   
    //Convert init kokkos spin matrices
    View_spin_matrix_type d_T_mixed("d_T_mixed");    
    View_spin_matrix_type d_T_unpol("d_T_unpol");
    View_spin_matrix_type d_Cg5("d_Cg5");
    View_spin_matrix_type d_Cg5g4("d_Cg5g4");
    View_spin_matrix_type d_Cg5NR("d_Cg5NR");
    View_spin_matrix_type d_BarSMat("d_BarSMat");

    View_spin_matrix_type::HostMirror h_T_mixed = Kokkos::create_mirror_view( d_T_mixed );
    View_spin_matrix_type::HostMirror h_T_unpol = Kokkos::create_mirror_view( d_T_unpol );
    View_spin_matrix_type::HostMirror h_Cg5 = Kokkos::create_mirror_view( d_Cg5 );
    View_spin_matrix_type::HostMirror h_Cg5g4 = Kokkos::create_mirror_view( d_Cg5g4 );
    View_spin_matrix_type::HostMirror h_Cg5NR = Kokkos::create_mirror_view( d_Cg5NR );
    View_spin_matrix_type::HostMirror h_BarSMat = Kokkos::create_mirror_view( d_BarSMat );

    QDPSpMatrixToKokkosSpMatrix(h_T_mixed,T_mixed);
    QDPSpMatrixToKokkosSpMatrix(h_T_unpol,T_unpol);
    QDPSpMatrixToKokkosSpMatrix(h_Cg5,Cg5);
    QDPSpMatrixToKokkosSpMatrix(h_Cg5g4,Cg5g4);
    QDPSpMatrixToKokkosSpMatrix(h_Cg5NR,Cg5NR);
    QDPSpMatrixToKokkosSpMatrix(h_BarSMat,BarSMat);
    
    Kokkos::deep_copy( d_T_mixed, h_T_mixed);
    Kokkos::deep_copy( d_T_unpol, h_T_unpol);
    Kokkos::deep_copy( d_Cg5, h_Cg5);
    Kokkos::deep_copy( d_Cg5g4, h_Cg5g4);
    Kokkos::deep_copy( d_Cg5NR, h_Cg5NR);
    Kokkos::deep_copy( d_BarSMat, h_BarSMat);

    
    SpinMatrix  Cg4m = BaryonSpinMats::Cg4m();
    View_spin_matrix_type d_Cg4m("d_Cg4m");
    View_spin_matrix_type::HostMirror h_Cg4m = Kokkos::create_mirror_view( d_Cg4m );
    QDPSpMatrixToKokkosSpMatrix(h_Cg4m,Cg4m);
    Kokkos::deep_copy( d_Cg4m, h_Cg4m);

    SpinMatrix  CgmNR = BaryonSpinMats::CgmNR();
    View_spin_matrix_type d_CgmNR("d_CgmNR");
    View_spin_matrix_type::HostMirror h_CgmNR = Kokkos::create_mirror_view( d_CgmNR );
    QDPSpMatrixToKokkosSpMatrix(h_CgmNR,CgmNR);
    Kokkos::deep_copy( d_CgmNR, h_CgmNR);
 
    SpinMatrix  TmixedNegPar = BaryonSpinMats::TmixedNegPar();
    View_spin_matrix_type d_TmixedNegPar("d_TmixedNegPar");
    View_spin_matrix_type::HostMirror h_TmixedNegPar = Kokkos::create_mirror_view( d_TmixedNegPar );
    QDPSpMatrixToKokkosSpMatrix(h_TmixedNegPar,TmixedNegPar);
    Kokkos::deep_copy( d_TmixedNegPar, h_TmixedNegPar);
    

    SpinMatrix Cg5NRnegPar = BaryonSpinMats::Cg5NRnegPar();
    View_spin_matrix_type d_Cg5NRnegPar("d_Cg5NRnegPar");
    View_spin_matrix_type::HostMirror h_Cg5NRnegPar = Kokkos::create_mirror_view( d_Cg5NRnegPar );
    QDPSpMatrixToKokkosSpMatrix(h_Cg5NRnegPar,Cg5NRnegPar);
    Kokkos::deep_copy( d_Cg5NRnegPar, h_Cg5NRnegPar);

    //View_corr_type k_b_prop("k_b_prop",numSites);
    //View_corr_type::HostMirror h_k_b_prop = Kokkos::create_mirror_view( k_b_prop );
    //View_corr_array_type d_k_b_prop("k_b_prop",num_baryons,numSites);
    View_corr_array_type d_k_b_prop("d_k_b_prop",numSites,num_baryons);

   
     tClock.stop();
     QDPIO::cout << "Total data movement time = "<< tClock.getTimeInSeconds() << " secs" << std::endl;
     tClock.reset();
     tClock.start();

     //total_time =tClock.getTimeInSeconds();
    //Printing out values to compare
    //auto *coutbuf = std::cout.rdbuf();
    //std::ofstream out("bars.py",std::ios_base::app);
    //std::cout.rdbuf(out.rdbuf());

      //auxiliary views
      
      View_prop_type tmp2("tmp2",numSites);
      View_prop_type tmp1("tmp1",numSites);
 
      View_prop_type di_quark_d_Cg5("di_quark_d_Cg5",numSites);
      View_prop_type di_quark_d_BarSMat("di_quark_d_BarSMat",numSites);
      View_prop_type di_quark_d_Cg5g("di_quark_d_Cg5g",numSites);
      View_prop_type di_quark_d_Cg4m("di_quark_d_Cg4m",numSites);
      View_prop_type di_quark_d_Cg5NR("di_quark_d_Cg5NR",numSites);
      View_prop_type di_quark_d_CgmNR("di_quark_d_CgmNR",numSites);
      View_prop_type di_quark_d_Cg5NRnegPar("di_quark_d_Cg5NRnegPar",numSites);
      View_prop_type di_quark("di_quark",numSites);


      View_Latt_spin_matrix_type stmp1("stmp1",numSites);
      View_Latt_spin_matrix_type stmp2("stmp2",numSites);
      View_Latt_color_matrix_type ctmp("ctmp",numSites);
      //View_LatticeComplex_2d d_hsum("d_hsum",numSites,num_mom,length);      
 
      //multi3d<DComplex> kokkos_3hsum;

      Kokkos::complex<double> tmpSum=0;

      Kokkos::parallel_for( "Site loop",range_policy(0,numSites), KOKKOS_LAMBDA ( int nSite ) {
        auto q_prop_1 = Kokkos::subview(quark_propagator_1,nSite,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto q_prop_2 = Kokkos::subview(quark_propagator_2,nSite,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL); 
        auto sub_di_quark = Kokkos::subview(di_quark,nSite,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto sub_tmp1 = Kokkos::subview(tmp1,nSite,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto sub_tmp2 = Kokkos::subview(tmp2,nSite,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto sub_stmp1 = Kokkos::subview(stmp1,nSite,Kokkos::ALL,Kokkos::ALL);
        auto sub_stmp2 = Kokkos::subview(stmp2,nSite,Kokkos::ALL,Kokkos::ALL);
        auto sub_ctmp = Kokkos::subview(ctmp,nSite,Kokkos::ALL,Kokkos::ALL);

       // Loop over baryons
        for(int baryons = 0; baryons < num_baryons; ++baryons)  
        { 
          auto k_b_prop = Kokkos::subview(d_k_b_prop,Kokkos::ALL,baryons);
          switch (baryons)
          {
          case 0:
             kokkos_sigma2pt(nSite,k_b_prop,q_prop_1, q_prop_2,
                         d_T_mixed, d_Cg5, sub_di_quark,sub_tmp1,sub_tmp2,sub_stmp1,sub_stmp2,sub_ctmp);
             break;
        
          case 1:
             kokkos_lambda2pt(nSite,k_b_prop,q_prop_1, q_prop_2, 
                              d_T_mixed, d_Cg5, sub_di_quark,sub_tmp1,sub_tmp2,sub_stmp1,sub_stmp2,sub_ctmp);
             break;
         
          case 2:
             kokkos_sigmast2pt(nSite,k_b_prop,q_prop_1, q_prop_2,
                                    d_T_mixed, d_BarSMat, sub_di_quark,sub_tmp1,sub_tmp2,sub_stmp1,sub_stmp2,sub_ctmp);    
             break;

          case 3:
             kokkos_sigma2pt(nSite,k_b_prop,q_prop_1, q_prop_2,
                                    d_T_mixed, d_Cg5g4, sub_di_quark,sub_tmp1,sub_tmp2,sub_stmp1,sub_stmp2,sub_ctmp);   
             break;
          
          case 4:
             kokkos_lambda2pt(nSite,k_b_prop,q_prop_1, q_prop_2, 
                              d_T_mixed, d_Cg5g4, sub_di_quark,sub_tmp1,sub_tmp2,sub_stmp1,sub_stmp2,sub_ctmp);    
             break;

          case 5:
             kokkos_sigmast2pt(nSite,k_b_prop,q_prop_1, q_prop_2, 
                               d_T_mixed, d_Cg4m, sub_di_quark,sub_tmp1,sub_tmp2,sub_stmp1,sub_stmp2,sub_ctmp);
             break;
         
          case 6:
             kokkos_sigma2pt(nSite,k_b_prop,q_prop_1, q_prop_2, 
                             d_T_mixed, d_Cg5NR, sub_di_quark,sub_tmp1,sub_tmp2,sub_stmp1,sub_stmp2,sub_ctmp);
             break;

          case 7:
             kokkos_lambda2pt(nSite,k_b_prop,q_prop_1, q_prop_2, 
                              d_T_mixed, d_Cg5NR, sub_di_quark,sub_tmp1,sub_tmp2,sub_stmp1,sub_stmp2,sub_ctmp);
             break;
       
          case 8:
             kokkos_sigmast2pt(nSite,k_b_prop,q_prop_1, q_prop_2, 
                               d_T_mixed, d_CgmNR, sub_di_quark,sub_tmp1,sub_tmp2,sub_stmp1,sub_stmp2,sub_ctmp);     
             k_b_prop(nSite) *= 4.0;
             break;

          case 9:
             kokkos_sigma2pt(nSite,k_b_prop,q_prop_1, q_prop_2, 
                            d_T_unpol, d_Cg5, sub_di_quark,sub_tmp1,sub_tmp2,sub_stmp1,sub_stmp2,sub_ctmp);
             break;
        
          case 10:
             kokkos_sigma2pt(nSite,k_b_prop,q_prop_1, q_prop_2, 
                             d_T_unpol, d_Cg5g4, sub_di_quark,sub_tmp1,sub_tmp2,sub_stmp1,sub_stmp2,sub_ctmp);
             break;
     
          case 11:
             kokkos_sigma2pt(nSite,k_b_prop,q_prop_1, q_prop_2, 
                             d_T_unpol, d_Cg5NR, sub_di_quark,sub_tmp1,sub_tmp2,sub_stmp1,sub_stmp2,sub_ctmp);
             break;
        
          case 12:
             kokkos_lambdaNaive2pt(nSite,k_b_prop,q_prop_1, q_prop_2, 
                                   d_T_unpol, d_Cg5, sub_di_quark,sub_tmp1,sub_tmp2,sub_stmp1,sub_stmp2,sub_ctmp);
             break;
          
          case 13:
             kokkos_xi2pt(nSite,k_b_prop,q_prop_1, q_prop_2, 
                              d_T_unpol, d_Cg5, sub_di_quark,sub_tmp1,sub_tmp2,sub_stmp1,sub_stmp2,sub_ctmp);
             break;
         
          case 14:
             kokkos_lambdaNaive2pt(nSite,k_b_prop,q_prop_1, q_prop_2, 
                                   d_T_unpol, d_Cg5, sub_di_quark,sub_tmp1,sub_tmp2,sub_stmp1,sub_stmp2,sub_ctmp);
             break;
          
          case 15:
             kokkos_xi2pt(nSite,k_b_prop,q_prop_1, q_prop_2, 
                              d_T_mixed, d_Cg5, sub_di_quark,sub_tmp1,sub_tmp2,sub_stmp1,sub_stmp2,sub_ctmp);
             break;
        
          case 16:
             kokkos_sigma2pt(nSite,k_b_prop,q_prop_1, q_prop_2, 
                            d_TmixedNegPar, d_Cg5NRnegPar, sub_di_quark,sub_tmp1,sub_tmp2,sub_stmp1,sub_stmp2,sub_ctmp);
             break;
        
          default:
             break;
        }

         }//end baryon loop
         //innerUpdate += 0;
     }); //End kokkos parallel_for
    Kokkos::fence();
    
    tClock.stop();
    total_time+=tClock.getTimeInSeconds();
    QDPIO::cout << "Total in case baryons  = " << tClock.getTimeInSeconds() << " secs" << std::endl;
 
    double time=0;
    total_time=0;
    // Loop over baryons

    View_corr_array_type::HostMirror h_k_b_prop = Kokkos::create_mirror_view( d_k_b_prop );
    Kokkos::deep_copy( h_k_b_prop, d_k_b_prop);


    auto *coutbuf = std::cout.rdbuf();
    std::ofstream out("bars.py",std::ios_base::app);
    std::cout.rdbuf(out.rdbuf());
    for(int baryons = 0; baryons < num_baryons; ++baryons)  
    {
      //StopWatch tClock;
      tClock.reset();
      tClock.start();      
      auto n_k_b_prop = Kokkos::subview(d_k_b_prop,Kokkos::ALL,baryons);
      QDPIO::cout<<"\n"<<baryons <<"  d_k_b_prop(0,"<<baryons<<")  =   "<<h_k_b_prop(0,baryons).real(); 
    //Project onto zero and if desired non-zero momentum
    multi2d<DComplex> kokkos_hsum;
    kokkos_hsum = kokkos_sft(doSet, d_sft_sets, n_k_b_prop,d_phases);
    
    for(int sink_mom_num=0; sink_mom_num < num_mom; ++sink_mom_num){
        QDPIO::cout<<"\n"<<sink_mom_num<<"  Kokkos_hsum vs hsum  =   "<<kokkos_hsum[sink_mom_num][0].elem().elem().elem().real();
        QDPIO::cout<< "\n*************\n";
	    for(int t = 0; t < length; ++t)
    	 {
	       // NOTE: there is NO  1/2  multiplying hsum
	       barprop[baryons][sink_mom_num][t] = kokkos_hsum[sink_mom_num][t];
	     }
      }
      tClock.stop();
      time += tClock.getTimeInSeconds();
      //total_time+=tClock.getTimeInSeconds();
    
    } // end loop over baryons
    
    std::cout.rdbuf(coutbuf);

    //QDPIO::cout << "Total for sft baryons  = " << time << " secs" << std::endl;

    QDPIO::cout << "Total fory baryons old  = " << total_time << " secs" << std::endl;


    
    END_CODE();
  }

}  // end namespace Chroma


