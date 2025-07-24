// -*- C++ -*-
/*! \file
 *  \brief Factory for producing system solvers for:
 * (MdagM  0)  (psi_1) = (chi_1)
 * (0  MdagM) (psi_2)   (chi_2)
 *For regular solver, the system solvers are for MdagM*psi = chi
 */


#ifndef __syssolver_mdagm_2qblock_factory_h__
#define __syssolver_mdagm_2qblock_factory_h__

#include "chromabase.h"
#include "state.h"
#include "singleton.h"
#include "objfactory.h"
#include "linearop.h"
#include "typelist.h"
#include "actions/ferm/invert/syssolver_mdagm.h"

using namespace QDP;

namespace Chroma
{

  namespace FactoryEnv { 
    typedef Handle< FermState< LatticePropagator, multi1d<LatticeColorMatrix>, multi1d<LatticeColorMatrix> > > FSHandle2QB;
    typedef Handle< FermState< LatticePropagatorF, multi1d<LatticeColorMatrix>, multi1d<LatticeColorMatrix> > > FSHandleF2QB;
    typedef Handle< FermState< LatticePropagatorD, multi1d<LatticeColorMatrix>, multi1d<LatticeColorMatrix> > > FSHandleD2QB;

  }

  //! MdagM system solver factory 2QB (foundry)
  /*! @ingroup invert */
  typedef Chroma::SingletonHolder<
    ObjectFactory<MdagMSystemSolver<LatticePropagator>,
          std::string,
          TYPELIST_4(XMLReader&, const std::string&, FactoryEnv::FSHandle2QB, Handle< LinearOperator<LatticePropagator> >),
          MdagMSystemSolver<LatticePropagator>* (*)(XMLReader&,
                             const std::string&,
                             FactoryEnv::FSHandle2QB,
                             Handle< LinearOperator<LatticePropagator> >),
          StringFactoryError> >
  TheMdagMFerm2QBSystemSolverFactory;

  typedef Chroma::SingletonHolder< 
    ObjectFactory<MdagMSystemSolver<LatticePropagatorF>, 
		  std::string,
		  TYPELIST_4(XMLReader&, const std::string&, FactoryEnv::FSHandleF2QB, Handle< LinearOperator<LatticePropagatorF > >),
		  MdagMSystemSolver<LatticePropagatorF>* (*)(XMLReader&,
							  const std::string&,
							  FactoryEnv::FSHandleF2QB, 
							  Handle< LinearOperator<LatticePropagatorF> >), 
		  StringFactoryError> >
  TheMdagMFermF2QBSystemSolverFactory;

  typedef Chroma::SingletonHolder< 
    ObjectFactory<MdagMSystemSolver<LatticePropagatorD>, 
		  std::string,
		  TYPELIST_4(XMLReader&, const std::string&, FactoryEnv::FSHandleD2QB, Handle< LinearOperator<LatticePropagatorD> >),
		  MdagMSystemSolver<LatticePropagatorD>* (*)(XMLReader&,
							  const std::string&,
							  FactoryEnv::FSHandleD2QB,
							  Handle< LinearOperator<LatticePropagatorD> >), 
		  StringFactoryError> >
  TheMdagMFermD2QBSystemSolverFactory;


  //! MdagM system solver factory (foundry)
  /*! @ingroup invert */
  typedef Chroma::SingletonHolder< 
    ObjectFactory<MdagMSystemSolverArray<LatticePropagator>, 
          std::string,       
          TYPELIST_4(XMLReader&, const std::string&, FactoryEnv::FSHandle2QB, Handle< LinearOperatorArray<LatticePropagator> >),
          MdagMSystemSolverArray<LatticePropagator>* (*)(XMLReader&,
                                  const std::string&,
                                  FactoryEnv::FSHandle2QB,
                                  Handle< LinearOperatorArray<LatticePropagator> >),
          StringFactoryError> >
  TheMdagMFerm2QBSystemSolverArrayFactory;

  //! MdagM system solver factory (foundry)
  /*! @ingroup invert */
  typedef Chroma::SingletonHolder< 
    ObjectFactory<MdagMSystemSolver<LatticeStaggeredPropagator>, 
		  std::string,
		  TYPELIST_3(XMLReader&, const std::string&, Handle< LinearOperator<LatticeStaggeredPropagator> >),
		  MdagMSystemSolver<LatticeStaggeredPropagator>* (*)(XMLReader&,
								  const std::string&,
								  Handle< LinearOperator<LatticeStaggeredPropagator> >), 
		  StringFactoryError> >
  TheMdagMStagFerm2QBSystemSolverFactory;

}


#endif
