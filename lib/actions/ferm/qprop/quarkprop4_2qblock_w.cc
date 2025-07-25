/*! \file
 *  \brief Full quark propagator solver
 *
 *  Given a complete propagator as a source, this does all the inversions needed
 */

//#include "wilstype_fermact_w.h"
#include "wilstype_fermact_2qblock_w.h"
#include "util/ferm/transf.h"
#include "actions/ferm/qprop/quarkprop4_2qblock_w.h"
#include "actions/ferm/invert/syssolver_linop_2qblock_factory.h"
#include "actions/ferm/invert/syssolver_mdagm_2qblock_factory.h"
#include "actions/ferm/invert/multi_syssolver_linop_2qblock_factory.h"
#include "actions/ferm/invert/multi_syssolver_mdagm_2qblock_factory.h"
#include "actions/ferm/invert/multi_syssolver_mdagm_accumulate_2qblock_factory.h"


#define FOURQ_SOLVE 1


namespace Chroma 
{
  //! Given a complete propagator as a source, this does all the inversions needed
  /*! \ingroup qprop
   *
   * This routine is actually generic to all Wilson-like fermions
   *
   * \param q_sol    quark propagator ( Write )
   * \param q_src    source ( Read )
   * \param RsdCG    CG (or MR) residual used here ( Read )
   * \param MaxCG    maximum number of CG iterations ( Read )
   * \param ncg_had  number of CG iterations ( Write )
   */

  template<typename T>
  void quarkProp4_a(LatticePropagator& q_sol, 
		    XMLWriter& xml_out,
		    const LatticePropagator& q_src,
		    Handle< SystemSolver<T> > qprop,
		    QuarkSpinType quarkSpinType,
		    int& ncg_had)
  {
    START_CODE();

    QDPIO::cout << "Entering quarkProp4 4q - MRHS interface" << std::endl;
    push(xml_out, "QuarkProp4");

    ncg_had = 0;
		{ 
			Double norm_chi;
			Double fact;
			LatticeFermion  chi;
			LatticeFermion  psi;
			// This version loops over all color and spin indices

            psi = zero;
            

            // Extract a fermion source
            // Due to the vaguaries of initializing a std::shared<const T>
            // We go via a temporary.
            LatticeFermion tmp;
            //PropToFerm(q_src, tmp, color_source, spin_source);
            PropToFerm(q_src, tmp, 0, 0);

            // Normalize temporary 
            norm_chi = sqrt(norm2(tmp));
            fact = toDouble(1)/norm_chi;
            tmp *= fact;
        
            // Create the RHS 	
            chi = tmp;

			// Do the MultiRHS solve
			//
			// Convention: In true multiRHS solve only solution 0 will have non-zero
			// n-count for now. That way accumulating ncg_had by adding 0s potentially
			// will work.

            SystemSolverResults_t results = (*qprop)(psi, chi);
			// Accumulate ncg_had and restore solution into solution prop	
			ncg_had = 0;

            // Undo rescale by multiplying by 1/fact = norm_chi[idx]
            psi *= norm_chi; 

            // Insert  solution into propagator
            //FermToProp(*(psi_ptrs[idx]), q_sol, col_idx, spin_idx);
            FermToProp(psi, q_sol, 0,0);

            // Accumulate ncg_had. This will be correct if we follow
            // the convention that true mrhs solvers return only a count
            // in results[0].n_count and keep all others as zero
            // Fake MRHS solvers (which loop over sources) can fill out 
            // an accurate iteration count for each solve. 
            ncg_had += results.n_count;
            push(xml_out,"Qprop");
            write(xml_out, "color_source", 0);
            write(xml_out, "spin_source", 0);
            write(xml_out, "n_count", results.n_count);
            write(xml_out, "resid", results.resid);
            pop(xml_out);

		} // psis, chis etc go away here. 

    pop(xml_out);
    QDPIO::cout << "Exiting quarkProp4" << std::endl;

    END_CODE();
  }



  template<typename T>
  void quarkProp4_a(LatticePropagator& q_sol, 
		    XMLWriter& xml_out,
		    const LatticePropagator& q_src,
		    Handle< SystemSolver<T> > qprop,
		    QuarkSpinType quarkSpinType,
		    int& ncg_had, bool fourq)
  {
    START_CODE();

    QDPIO::cout << "Entering new double quarkProp4 4q - MRHS interface" << std::endl;
    push(xml_out, "QuarkProp4");

    ncg_had = 0;
		{ 
			Double norm_chi;
			Double fact;
			LatticePropagator  chi;
			LatticePropagator  psi = zero;
    
            // Normalize temporary         
            norm_chi = sqrt(norm2(q_src));
            fact = toDouble(1)/norm_chi;
            chi *= fact;
        
            SystemSolverResults_t results = (*qprop)(psi, chi);
			// Accumulate ncg_had and restore solution into solution prop	
			ncg_had = 0;

            // Undo rescale by multiplying by 1/fact = norm_chi[idx]
            psi *= norm_chi; 

            // Accumulate ncg_had. This will be correct if we follow
            // the convention that true mrhs solvers return only a count
            // in results[0].n_count and keep all others as zero
            // Fake MRHS solvers (which loop over sources) can fill out 
            // an accurate iteration count for each solve. 
            ncg_had += results.n_count;
            push(xml_out,"Qprop");
            write(xml_out, "color_source", 0);
            write(xml_out, "spin_source", 0);
            write(xml_out, "n_count", results.n_count);
            write(xml_out, "resid", results.resid);
            pop(xml_out);

		} // psis, chis etc go away here. 

    pop(xml_out);
    QDPIO::cout << "Exiting quarkProp4" << std::endl;

    END_CODE();
  }



  typedef LatticePropagator LP;
  typedef LatticeFermion LF;
  typedef multi1d<LatticeColorMatrix> LCM;

  //! Given a complete propagator as a source, this does all the inversions needed
  /*! \ingroup qprop
   *
   * This routine is actually generic to all Wilson-like fermions
   *
   * \param q_sol    quark propagator ( Write )
   * \param q_src    source ( Read )
   * \param invParam inverter parameters ( Read )
   * \param ncg_had  number of CG iterations ( Write )
   */
#if 0 //FOURQ_SOLVE
  void quarkProp4(LatticePropagator& q_sol,
          XMLWriter& xml_out,
          const LatticePropagator& q_src,
          Handle< SystemSolver<LatticePropagator> > qprop,
          QuarkSpinType quarkSpinType,
          int& ncg_had)
  {
    quarkProp4_a<LatticePropagator>(q_sol, xml_out, q_src, qprop, quarkSpinType, ncg_had, true);
  }
#endif
#if 0
  void quarkProp4(LatticePropagator& q_sol,
          XMLWriter& xml_out,
          const LatticePropagator& q_src,
          Handle< SystemSolver<LF> > qprop,
          QuarkSpinType quarkSpinType,
          int& ncg_had)
  {
    quarkProp4_a<LF>(q_sol, xml_out, q_src, qprop, quarkSpinType, ncg_had);
  }
#endif

  //! Given a complete propagator as a source, this does all the inversions needed
  /*! \ingroup qprop
   *
   * This routine is actually generic to all Wilson-like fermions
   *
   * \param q_sol    quark propagator ( Write )
   * \param q_src    source ( Read )
   * \param invParam inverter parameters ( Read )
   * \param ncg_had  number of CG iterations ( Write )
   */
#if 1
  template<>
  void
  WilsonTypeFermAct2QB<LatticePropagator,LCM,LCM>::quarkProp(
    LatticePropagator& q_sol,
    XMLWriter& xml_out,
    const LatticePropagator& q_src,
    Handle< FermState<LatticePropagator,LCM,LCM> > state,
    const GroupXML_t& invParam,
    QuarkSpinType quarkSpinType,
    int& ncg_had) const
  {
    QDPIO::cout << "In double quarkProp_4q()" << std::endl;
    StopWatch swatch;
    swatch.start();
    Handle< SystemSolver<LatticePropagator> > qprop(this->qprop(state,invParam));
    swatch.stop();
    QDPIO::cout << "Creating qprop took " << swatch.getTimeInSeconds() 
		<< "sec " << std::endl;
    quarkProp4_a<LatticePropagator>(q_sol, xml_out, q_src, qprop, quarkSpinType, ncg_had,true);
  }
#endif

#if 0
  template<>
  void
  WilsonTypeFermAct<LF,LCM,LCM>::quarkProp(
    LatticePropagator& q_sol,
    XMLWriter& xml_out,
    const LatticePropagator& q_src,
    Handle< FermState<LF,LCM,LCM> > state,
    const GroupXML_t& invParam,
    QuarkSpinType quarkSpinType,
    int& ncg_had) const
  {
    QDPIO::cout << "In quarkProp_4q()" << std::endl;
    StopWatch swatch;
    swatch.start();
    Handle< SystemSolver<LF> > qprop(this->qprop(state,invParam));
    swatch.stop();
    QDPIO::cout << "Creating qprop took " << swatch.getTimeInSeconds()
        << "sec " << std::endl;
    quarkProp4_a<LF>(q_sol, xml_out, q_src, qprop, quarkSpinType, ncg_had);
  }
#endif

  //! Given a complete propagator as a source, this does all the inversions needed
  /*! \ingroup qprop
   *
   * This routine is actually generic to all Wilson-like fermions
   *
   * \param q_sol    quark propagator ( Write )
   * \param q_src    source ( Read )
   * \param invParam inverter parameters ( Read )
   * \param ncg_had  number of CG iterations ( Write )
   */
#if 0
  template<>
  void
  WilsonTypeFermAct5D2QB<LatticePropagator,LCM,LCM>::quarkProp(
    LatticePropagator& q_sol,
    XMLWriter& xml_out,
    const LatticePropagator& q_src,
    Handle< FermState<LatticePropagator,LCM,LCM> > state,
    const GroupXML_t& invParam,
    QuarkSpinType quarkSpinType,
    int& ncg_had) const
  {
    Handle< SystemSolver<LatticePropagator> > qprop(this->qprop(state,invParam));
    quarkProp4_a<LatticePropagator>(q_sol, xml_out, q_src, qprop, quarkSpinType, ncg_had, true);
  }

#endif
/*
  template<>
  void 
  WilsonTypeFermAct5D<LF,LCM,LCM>::quarkProp(
    LatticePropagator& q_sol, 
    XMLWriter& xml_out,
    const LatticePropagator& q_src,
    Handle< FermState<LF,LCM,LCM> > state,
    const GroupXML_t& invParam,
    QuarkSpinType quarkSpinType,
    int& ncg_had) const
  {
    Handle< SystemSolver<LF> > qprop(this->qprop(state,invParam));
    quarkProp4_a<LF>(q_sol, xml_out, q_src, qprop, quarkSpinType, ncg_had);
  }
*/


  //------------------------------------------------------------------------------------

  // Return a linear operator solver for this action to solve M*psi=chi 
  /*! \ingroup qprop */

  template<>
  LinOpSystemSolver<LP>*
  WilsonTypeFermAct2QB<LP,LCM,LCM>::invLinOp(Handle< FermState<LP,LCM,LCM> > state,
					  const GroupXML_t& invParam) const
  {
    std::istringstream  xml(invParam.xml);
    XMLReader  paramtop(xml);
	
    return TheLinOpFerm2QBSystemSolverFactory::Instance().createObject(invParam.id,
								    paramtop,
								    invParam.path,
								    state,
								    this->linOp(state));
  }

  //! Return a linear operator solver for this action to solve MdagM*psi=chi 
  /*! \ingroup qprop */
  template<>
  MdagMSystemSolver<LP>*
  WilsonTypeFermAct2QB<LP,LCM,LCM>::invMdagM(Handle< FermState<LP,LCM,LCM> > state,
					  const GroupXML_t& invParam) const
  {
    std::istringstream  xml(invParam.xml);
    XMLReader  paramtop(xml);

    return TheMdagMFerm2QBSystemSolverFactory::Instance().createObject(invParam.id,
								    paramtop,
								    invParam.path,
								    state,
								    this->linOp(state));
  }


  //! Return a linear operator solver for this action to solve (M+shift_i)*psi_i = chi 
  /*! \ingroup qprop */
  template<>
  LinOpMultiSystemSolver<LP>*
  WilsonTypeFermAct2QB<LP,LCM,LCM>::mInvLinOp(Handle< FermState<LP,LCM,LCM> > state,
					   const GroupXML_t& invParam) const
  {
    std::istringstream  xml(invParam.xml);
    XMLReader  paramtop(xml);

    return TheLinOpFerm2QBMultiSystemSolverFactory::Instance().createObject(invParam.id,
									 paramtop,
									 invParam.path,
									 this->linOp(state));
  }


  //! Return a linear operator solver for this action to solve (MdagM+shift_i)*psi_i = chi 
  /*! \ingroup qprop */
  template<>
  MdagMMultiSystemSolver<LP>*
  WilsonTypeFermAct2QB<LP,LCM,LCM>::mInvMdagM(Handle< FermState<LP,LCM,LCM> > state,
					   const GroupXML_t& invParam) const
  {
    std::istringstream  xml(invParam.xml);
    XMLReader  paramtop(xml);

    return TheMdagMFerm2QBMultiSystemSolverFactory::Instance().createObject(invParam.id,
									 paramtop,
									 invParam.path,
									 state,
									 this->linOp(state));
  }

  //! Return a linear operator solver for this action to solve (MdagM+shift_i)*psi_i = chi 
  /*! \ingroup qprop */
  template<>
  MdagMMultiSystemSolverAccumulate<LP>*
  WilsonTypeFermAct2QB<LP,LCM,LCM>::mInvMdagMAcc(Handle< FermState<LP,LCM,LCM> > state,
					   const GroupXML_t& invParam) const
  {
    std::istringstream  xml(invParam.xml);
    XMLReader  paramtop(xml);

    return TheMdagMFerm2QBMultiSystemSolverAccumulateFactory::Instance().createObject(invParam.id,
									 paramtop,
									 invParam.path,
									 this->linOp(state));
  }



  //------------------------------------------------------------------------------------

  // Return a linear operator solver for this action to solve M*psi=chi 
  /*! \ingroup qprop */
  template<>
  LinOpSystemSolverArray<LP>*
  WilsonTypeFermAct5D2QB<LP,LCM,LCM>::invLinOp(Handle< FermState<LP,LCM,LCM> > state,
					    const GroupXML_t& invParam) const
  {
    std::istringstream  xml(invParam.xml);
    XMLReader  paramtop(xml);
	
    return TheLinOpFerm2QBSystemSolverArrayFactory::Instance().createObject(invParam.id,
									 paramtop,
									 invParam.path,
									 state,
									 this->linOp(state));
  }


  //! Return a linear operator solver for this action to solve MdagM*psi=chi 
  /*! \ingroup qprop */
  template<>
  MdagMSystemSolverArray<LP>*
  WilsonTypeFermAct5D2QB<LP,LCM,LCM>::invMdagM(Handle< FermState<LP,LCM,LCM> > state,
					    const GroupXML_t& invParam) const
  {
    std::istringstream  xml(invParam.xml);
    XMLReader  paramtop(xml);

    return TheMdagMFerm2QBSystemSolverArrayFactory::Instance().createObject(invParam.id,
									 paramtop,
									 invParam.path,
									 state,
									 this->linOp(state));
  }



  // Return a linear operator solver for this action to solve M*psi=chi 
  /*! \ingroup qprop */
  template<>
  LinOpSystemSolverArray<LP>*
  WilsonTypeFermAct5D2QB<LP,LCM,LCM>::invLinOpPV(Handle< FermState<LP,LCM,LCM> > state,
					      const GroupXML_t& invParam) const
  {
    std::istringstream  xml(invParam.xml);
    XMLReader  paramtop(xml);
	
    return TheLinOpFerm2QBSystemSolverArrayFactory::Instance().createObject(invParam.id,
									 paramtop,
									 invParam.path,
									 state,
									 this->linOpPV(state));
  }


  //! Return a linear operator solver for this action to solve PV^dag*PV*psi=chi 
  /*! \ingroup qprop */
  template<>
  MdagMSystemSolverArray<LP>*
  WilsonTypeFermAct5D2QB<LP,LCM,LCM>::invMdagMPV(Handle< FermState<LP,LCM,LCM> > state,
					      const GroupXML_t& invParam) const
  {
    std::istringstream  xml(invParam.xml);
    XMLReader  paramtop(xml);

    return TheMdagMFerm2QBSystemSolverArrayFactory::Instance().createObject(invParam.id,
									 paramtop,
									 invParam.path,
									 state,
									 this->linOpPV(state));
  }


  //! Return a linear operator solver for this action to solve (MdagM+shift_i)*psi_i = chi 
  /*! \ingroup qprop */
  template<>
  MdagMMultiSystemSolverArray<LP>*
  WilsonTypeFermAct5D2QB<LP,LCM,LCM>::mInvMdagM(Handle< FermState<LP,LCM,LCM> > state,
					     const GroupXML_t& invParam) const
  {
    std::istringstream  xml(invParam.xml);
    XMLReader  paramtop(xml);

    return TheMdagMFerm2QBMultiSystemSolverArrayFactory::Instance().createObject(invParam.id,
									      paramtop,
									      invParam.path,
									      state,
									      lMdagM(state));
  }

  //! Return a linear operator solver for this action to solve (MdagM+shift_i)*psi_i = chi 
  /*! \ingroup qprop */
  template<>
  MdagMMultiSystemSolverAccumulateArray<LP>*
  WilsonTypeFermAct5D2QB<LP,LCM,LCM>::mInvMdagMAcc(Handle< FermState<LP,LCM,LCM> > state,
					     const GroupXML_t& invParam) const
  {
    std::istringstream  xml(invParam.xml);
    XMLReader  paramtop(xml);

    return TheMdagMFerm2QBMultiSystemSolverAccumulateArrayFactory::Instance().createObject(invParam.id,
									      paramtop,
									      invParam.path,
									      lMdagM(state));
  }


  //! Return a linear operator solver for this action to solve (MdagM+shift_i)*psi_i = chi 
  /*! \ingroup qprop */
  template<>
  MdagMMultiSystemSolverArray<LP>*
  WilsonTypeFermAct5D2QB<LP,LCM,LCM>::mInvMdagMPV(Handle< FermState<LP,LCM,LCM> > state,
					       const GroupXML_t& invParam) const
  {
    std::istringstream  xml(invParam.xml);
    XMLReader  paramtop(xml);

    Handle< LinearOperatorArray<LP> > PV(this->linOpPV(state));
    Handle< LinearOperatorArray<LP> > MdagM(new MdagMLinOpArray<LP>(PV));

    return TheMdagMFerm2QBMultiSystemSolverArrayFactory::Instance().createObject(
      invParam.id,
      paramtop,
      invParam.path,
      state,
      MdagM);
  }

  //! Return a linear operator solver for this action to solve (MdagM+shift_i)*psi_i = chi 
  /*! \ingroup qprop */
  template<>
  MdagMMultiSystemSolverAccumulateArray<LP>*
  WilsonTypeFermAct5D2QB<LP,LCM,LCM>::mInvMdagMPVAcc(Handle< FermState<LP,LCM,LCM> > state,
					       const GroupXML_t& invParam) const
  {
    std::istringstream  xml(invParam.xml);
    XMLReader  paramtop(xml);

    Handle< LinearOperatorArray<LP> > PV(this->linOpPV(state));
    Handle< LinearOperatorArray<LP> > MdagM(new MdagMLinOpArray<LP>(PV));

    return TheMdagMFerm2QBMultiSystemSolverAccumulateArrayFactory::Instance().createObject(
      invParam.id,
      paramtop,
      invParam.path,
      MdagM);
  }


} // namespace Chroma
