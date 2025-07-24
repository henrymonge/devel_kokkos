// -*- C++ -*-
/*! \file
 *  \brief Factory for producing system solvers for MdagM*psi = chi
 */

#ifndef __multi_syssolver_mdagm_accumulate_2qblock_factory_h__
#define __multi_syssolver_mdagm_accumulate_2qblock_factory_h__

#include "singleton.h"
#include "objfactory.h"
#include "linearop.h"
#include "actions/ferm/invert/multi_syssolver_mdagm_accumulate.h"

namespace Chroma
{

  //! MdagM system solver factory (foundry)
  /*! @ingroup invert */
  typedef Chroma::SingletonHolder<
    ObjectFactory<MdagMMultiSystemSolverAccumulate<LatticePropagator>,
          std::string,
          TYPELIST_3(XMLReader&, const std::string&, Handle< LinearOperator<LatticePropagator> >),
          MdagMMultiSystemSolverAccumulate<LatticePropagator>* (*)(XMLReader&,
                                  const std::string&,
                                  Handle< LinearOperator<LatticePropagator> >),
          StringFactoryError> >
  TheMdagMFerm2QBMultiSystemSolverAccumulateFactory;

  //! MdagM system solver factory (foundry)
  /*! @ingroup invert */
  typedef Chroma::SingletonHolder<
    ObjectFactory<MdagMMultiSystemSolverAccumulateArray<LatticePropagator>,
          std::string,
          TYPELIST_3(XMLReader&, const std::string&, Handle< LinearOperatorArray<LatticePropagator> >),
          MdagMMultiSystemSolverAccumulateArray<LatticePropagator>* (*)(XMLReader&,
                                   const std::string&,
                                   Handle< LinearOperatorArray<LatticePropagator> >),
          StringFactoryError> >
  TheMdagMFerm2QBMultiSystemSolverAccumulateArrayFactory;


  //! MdagM system solver factory (foundry)
  /*! @ingroup invert */
  typedef Chroma::SingletonHolder< 
    ObjectFactory<MdagMMultiSystemSolverAccumulate<LatticeStaggeredPropagator>, 
		  std::string,
		  TYPELIST_3(XMLReader&, const std::string&, Handle< LinearOperator<LatticeStaggeredPropagator> >),
		  MdagMMultiSystemSolverAccumulate<LatticeStaggeredPropagator>* (*)(XMLReader&,
								       const std::string&,
								       Handle< LinearOperator<LatticeStaggeredPropagator> >), 
		  StringFactoryError> >
  TheMdagMStagFerm2QBMultiSystemSolverAccumulateFactory;

}


#endif
