/*! \file
 *  \brief Solve a MdagM*psi=chi linear system by multi-shift CG
 */

#include "actions/ferm/invert/multi_syssolver_mdagm_accumulate_factory.h"
#include "actions/ferm/invert/multi_syssolver_mdagm_accumulate_2qblock_factory.h"

#include "actions/ferm/invert/multi_syssolver_mdagm_accumulate_aggregate.h"

#include "actions/ferm/invert/multi_syssolver_mdagm_cg_accumulate_array.h"

using namespace QDP;
namespace Chroma
{

  //! CG system solver namespace
  namespace MdagMMultiSysSolverCGAccumulateArrayEnv
  {
    //! Callback function
    MdagMMultiSystemSolverAccumulateArray<LatticeFermion>* createFerm(XMLReader& xml_in,
							    const std::string& path,
							    Handle< LinearOperatorArray<LatticeFermion> > A)
    {
      return new MdagMMultiSysSolverCGAccumulateArray<LatticeFermion>(A, MultiSysSolverCGParams(xml_in, path));
    }

//#if ENABLE_2QUARK_SOLVE //

    //! Callback function
    MdagMMultiSystemSolverAccumulateArray<LatticePropagator>* createFerm(XMLReader& xml_in,
                                const std::string& path,
                                Handle< LinearOperatorArray<LatticePropagator> > A)
    {
      return new MdagMMultiSysSolverCGAccumulateArray<LatticePropagator>(A, MultiSysSolverCGParams(xml_in, path));
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
	success &= Chroma::TheMdagMFermMultiSystemSolverAccumulateArrayFactory::Instance().registerObject(name, createFerm);

//#if ENABLE_2QUARK_SOLVE //

    //success &= Chroma::TheMdagMFerm2QBMultiSystemSolverAccumulateArrayFactory::Instance().registerObject(name, createFerm);

//#endif


	registered = true;
      }
      return success;
    }
  }
}
