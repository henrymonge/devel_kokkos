// -*- C++ -*-
/*! \file
 *  \brief Factory for producing system solvers for M*psi = chi
 */

#ifndef __multi_syssolver_linop_factory_h__
#define __multi_syssolver_linop_factory_h__

#include "singleton.h"
#include "objfactory.h"
#include "linearop.h"
#include "actions/ferm/invert/multi_syssolver_linop.h"

namespace Chroma
{

  //! LinOp system solver factory 4Q (foundry)
  /*! @ingroup invert */
  typedef Chroma::SingletonHolder<
    ObjectFactory<LinOpMultiSystemSolver<LatticePropagator>,
          std::string,
          TYPELIST_3(XMLReader&, const std::string&, Handle< LinearOperator<LatticePropagator> >),
          LinOpMultiSystemSolver<LatticePropagator>* (*)(XMLReader&,
                                  const std::string&,
                                  Handle< LinearOperator<LatticePropagator> >),
          StringFactoryError> >
  TheLinOpFerm2QBMultiSystemSolverFactory;


#if 0
  //! LinOp system solver factory (foundry)
  /*! @ingroup invert */
:cn
    ObjectFactory<LinOpMultiSystemSolverArray<LatticePropagator>, 
		  std::string,
		  TYPELIST_3(XMLReader&, const std::string&, Handle< LinearOperatorArray<LatticePropagator> >),
		  LinOpMultiSystemSolverArray<LatticePropagator>* (*)(XMLReader&,
								   const std::string&,
								   Handle< LinearOperatorArray<LatticePropagator> >), 
		  StringFactoryError> >
  TheLinOpFerm2QBMultiSystemSolverArrayFactory;
#endif


  //! LinOp system solver factory (foundry)
  /*! @ingroup invert */
  typedef Chroma::SingletonHolder< 
    ObjectFactory<LinOpMultiSystemSolver<LatticeStaggeredPropagator>, 
		  std::string,
		  TYPELIST_3(XMLReader&, const std::string&, Handle< LinearOperator<LatticeStaggeredPropagator> >),
		  LinOpMultiSystemSolver<LatticeStaggeredPropagator>* (*)(XMLReader&,
								       const std::string&,
								       Handle< LinearOperator<LatticeStaggeredPropagator> >), 
		  StringFactoryError> >
  TheLinOpStagFerm2QBMultiSystemSolverFactory;

}


#endif
