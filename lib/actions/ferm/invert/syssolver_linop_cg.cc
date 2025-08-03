/*! \file
 *  \brief Solve a M*psi=chi linear system by CG2
 */
#include "state.h"
#include "actions/ferm/invert/syssolver_linop_factory.h"
#include "actions/ferm/invert/syssolver_linop_2qblock_factory.h"
#include "actions/ferm/invert/syssolver_linop_aggregate.h"

#include "actions/ferm/invert/syssolver_linop_cg.h"

namespace Chroma
{

  //! CG1 system solver namespace
  namespace LinOpSysSolverCGEnv
  {
    //! Anonymous namespace
    namespace
    {
      //! Name to be used
      const std::string name("CG_INVERTER");

      //! Local registration flag
      bool registered = false;
    }


    //! Callback function
    LinOpSystemSolver<LatticeFermion>* createFerm(XMLReader& xml_in,
						  const std::string& path,
						  Handle< FermState<
						                     LatticeFermion, 
						                     multi1d<LatticeColorMatrix>,
						                     multi1d<LatticeColorMatrix> 
					 	  > 
							  > state, 

						  Handle< LinearOperator<LatticeFermion> > A)
    {
      return new LinOpSysSolverCG<LatticeFermion>(A, SysSolverCGParams(xml_in, path));
    }

    //! Callback function
    LinOpSystemSolver<LatticeFermionF>* createFermF(XMLReader& xml_in,
						  const std::string& path,
						  Handle< FermState<
						                     LatticeFermionF, 
						                     multi1d<LatticeColorMatrixF>,
						                     multi1d<LatticeColorMatrixF> 
						  > 
							  > state, 

						  Handle< LinearOperator<LatticeFermionF> > A)
    {
      return new LinOpSysSolverCG<LatticeFermionF>(A, SysSolverCGParams(xml_in, path));
    }

//#if ENABLE_2QUARK_SOLVE //

    //! Callback function
    LinOpSystemSolver<LatticePropagator>* createFerm2QB(XMLReader& xml_in,
                          const std::string& path,
                          Handle< FermState<
                                             LatticePropagator,
                                             multi1d<LatticeColorMatrix>,
                                             multi1d<LatticeColorMatrix>
                          >
                              > state,

                          Handle< LinearOperator<LatticePropagator> > A)
    {
      return new LinOpSysSolverCG<LatticePropagator>(A, SysSolverCGParams(xml_in, path));
    }

    //! Callback function
    LinOpSystemSolver<LatticePropagatorF>* createFermF2QB(XMLReader& xml_in,
                          const std::string& path,
                          Handle< FermState<
                                             LatticePropagatorF,
                                             multi1d<LatticeColorMatrixF>,
                                             multi1d<LatticeColorMatrixF>
                          >
                              > state,

                          Handle< LinearOperator<LatticePropagatorF> > A)
    {
      return new LinOpSysSolverCG<LatticePropagatorF>(A, SysSolverCGParams(xml_in, path));
    }

//#endif




    //! Callback function
    LinOpSystemSolver<LatticeStaggeredFermion>* createStagFerm(XMLReader& xml_in,
							       const std::string& path,
							       Handle< LinearOperator<LatticeStaggeredFermion> > A)
    {
      return new LinOpSysSolverCG<LatticeStaggeredFermion>(A, SysSolverCGParams(xml_in, path));
    }

    //! Register all the factories
    bool registerAll() 
    {
      bool success = true; 
      if (! registered)
      {
	success &= Chroma::TheLinOpFermSystemSolverFactory::Instance().registerObject(name, createFerm);
	success &= Chroma::TheLinOpFFermSystemSolverFactory::Instance().registerObject(name, createFermF);
	success &= Chroma::TheLinOpStagFermSystemSolverFactory::Instance().registerObject(name, createStagFerm);

//#if ENABLE_2QUARK_SOLVE //
    success &= Chroma::TheLinOpFerm2QBSystemSolverFactory::Instance().registerObject(name, createFerm2QB);
    //success &= Chroma::TheLinOpFFerm2QBSystemSolverFactory::Instance().registerObject(name, createFermF2QB);
//#endif

	registered = true;
      }
      return success;
    }
  }
}
