/*! \file
 *  \brief Solve a MdagM*psi=chi linear system by CG2
 */

#include "actions/ferm/invert/multi_syssolver_mdagm_factory.h"
#include "actions/ferm/invert/multi_syssolver_mdagm_2qblock_factory.h"
#include "actions/ferm/invert/multi_syssolver_mdagm_aggregate.h"

#include "actions/ferm/invert/multi_syssolver_mdagm_cg.h"

namespace Chroma
{

  //! CG2 system solver namespace
  namespace MdagMMultiSysSolverCGEnv
  {
    //! Callback function
    MdagMMultiSystemSolver<LatticeFermion>* createFerm(XMLReader& xml_in,
						       const std::string& path,
						       Handle< FermState< LatticeFermion, multi1d<LatticeColorMatrix>, multi1d<LatticeColorMatrix> > >,
						       Handle< LinearOperator<LatticeFermion> > A)
    {
      return new MdagMMultiSysSolverCG<LatticeFermion>(A, MultiSysSolverCGParams(xml_in, path));
    }

//#if ENABLE_2QUARK_SOLVE //
    //! Callback function
    MdagMMultiSystemSolver<LatticePropagator>* createFerm(XMLReader& xml_in,
                               const std::string& path,
                               Handle< FermState< LatticePropagator, multi1d<LatticeColorMatrix>, multi1d<LatticeColorMatrix> > >,
                               Handle< LinearOperator<LatticePropagator> > A)
    {
      return new MdagMMultiSysSolverCG<LatticePropagator>(A, MultiSysSolverCGParams(xml_in, path));
    }

//#endif 


    //! Name to be used
    const std::string name("CG_INVERTER");

    //! Local registration flag
    static bool registered = false;

    //! Register all the factories
    bool registerAll() 
    {
      bool success = true; 
      if (! registered)
      {
	success &= Chroma::TheMdagMFermMultiSystemSolverFactory::Instance().registerObject(name, createFerm);

//#if ENABLE_2QUARK_SOLVE //

    //success &= Chroma::TheMdagMFerm2QBMultiSystemSolverFactory::Instance().registerObject(name, createFerm);

//#endif

	registered = true;
      }
      return success;
    }
  }
}
