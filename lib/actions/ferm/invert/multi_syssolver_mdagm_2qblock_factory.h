// -*- C++ -*-
/*! \file
 *  \brief Factory for producing system solvers for:
 * (MdagM  0)  (psi_1) = (chi_1)
 * (0  MdagM) (psi_2)   (chi_2)
 *For regular solver, the system solvers are for MdagM*psi = chi
 */

#ifndef __multi_syssolver_mdagm_2qblock_factory_h__
#define __multi_syssolver_mdagm_2qblock_factory_h__

#include "singleton.h"
#include "objfactory.h"
#include "linearop.h"
#include "state.h"

#include "actions/ferm/invert/multi_syssolver_mdagm.h"

namespace Chroma
{
  namespace { 
    typedef Handle< FermState< LatticePropagator, multi1d<LatticeColorMatrix>, multi1d<LatticeColorMatrix> > > FSHandle2QB;
  }

  //! MdagM system solver factory (foundry)
  /*! @ingroup invert */
  typedef Chroma::SingletonHolder<
    ObjectFactory<MdagMMultiSystemSolver<LatticePropagator>,
          std::string,
          TYPELIST_4(XMLReader&, const std::string&, FSHandle2QB, Handle< LinearOperator<LatticePropagator> >),
          MdagMMultiSystemSolver<LatticePropagator>* (*)(XMLReader&,
                                  const std::string&,
                                  FSHandle2QB,
                                  Handle< LinearOperator<LatticePropagator> >),
          StringFactoryError> >
  TheMdagMFerm2QBMultiSystemSolverFactory;

  //! MdagM system solver factory (foundry)
  /*! @ingroup invert */
  typedef Chroma::SingletonHolder<
    ObjectFactory<MdagMMultiSystemSolverArray<LatticePropagator>,
          std::string,
          TYPELIST_4(XMLReader&, const std::string&, FSHandle2QB, Handle< LinearOperatorArray<LatticePropagator> >),
          MdagMMultiSystemSolverArray<LatticePropagator>* (*)(XMLReader&,
                                   const std::string&,
                                   FSHandle2QB,
                                   Handle< LinearOperatorArray<LatticePropagator> >),
          StringFactoryError> >
  TheMdagMFerm2QBMultiSystemSolverArrayFactory;

}


#endif
