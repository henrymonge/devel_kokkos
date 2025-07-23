// -*- C++ -*-
/*! \file
 *  \brief Factory for producing system solvers for MdagM*psi = chi
 */

#ifndef __syssolver_mdagm_factory_h__
#define __syssolver_mdagm_factory_h__

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
    typedef Handle< FermState< LatticeFermion, multi1d<LatticeColorMatrix>, multi1d<LatticeColorMatrix> > > FSHandle;
    typedef Handle< FermState< LatticeFermionF, multi1d<LatticeColorMatrixF>, multi1d<LatticeColorMatrixF> > > FSHandleF;
    typedef Handle< FermState< LatticeFermionD, multi1d<LatticeColorMatrixD>, multi1d<LatticeColorMatrixD> > > FSHandleD;
    typedef Handle< FermState< LatticePropagator, multi1d<LatticeColorMatrix>, multi1d<LatticeColorMatrix> > > FSHandle4Q;

  }

  //! MdagM system solver factory (foundry)
  /*! @ingroup invert */
  typedef Chroma::SingletonHolder< 
    ObjectFactory<MdagMSystemSolver<LatticeFermion>, 
		  std::string,
		  TYPELIST_4(XMLReader&, const std::string&, FactoryEnv::FSHandle, Handle< LinearOperator<LatticeFermion> >),
		  MdagMSystemSolver<LatticeFermion>* (*)(XMLReader&,
							 const std::string&,
							 FactoryEnv::FSHandle,
							 Handle< LinearOperator<LatticeFermion> >), 
		  StringFactoryError> >
  TheMdagMFermSystemSolverFactory;


  //! MdagM system solver factory 4Q (foundry)
  /*! @ingroup invert */
  typedef Chroma::SingletonHolder<
    ObjectFactory<MdagMSystemSolver<LatticePropagator>,
          std::string,
          TYPELIST_4(XMLReader&, const std::string&, FactoryEnv::FSHandle4Q, Handle< LinearOperator<LatticePropagator> >),
          MdagMSystemSolver<LatticePropagator>* (*)(XMLReader&,
                             const std::string&,
                             FactoryEnv::FSHandle4Q,
                             Handle< LinearOperator<LatticePropagator> >),
          StringFactoryError> >
  TheMdagMFerm4QSystemSolverFactory;

  typedef Chroma::SingletonHolder< 
    ObjectFactory<MdagMSystemSolver<LatticeFermionF>, 
		  std::string,
		  TYPELIST_4(XMLReader&, const std::string&, FactoryEnv::FSHandleF, Handle< LinearOperator<LatticeFermionF > >),
		  MdagMSystemSolver<LatticeFermionF>* (*)(XMLReader&,
							  const std::string&,
							  FactoryEnv::FSHandleF, 
							  Handle< LinearOperator<LatticeFermionF> >), 
		  StringFactoryError> >
  TheMdagMFermFSystemSolverFactory;

  typedef Chroma::SingletonHolder< 
    ObjectFactory<MdagMSystemSolver<LatticeFermionD>, 
		  std::string,
		  TYPELIST_4(XMLReader&, const std::string&, FactoryEnv::FSHandleD, Handle< LinearOperator<LatticeFermionD> >),
		  MdagMSystemSolver<LatticeFermionD>* (*)(XMLReader&,
							  const std::string&,
							  FactoryEnv::FSHandleD,
							  Handle< LinearOperator<LatticeFermionD> >), 
		  StringFactoryError> >
  TheMdagMFermDSystemSolverFactory;


  //! MdagM system solver factory (foundry)
  /*! @ingroup invert */
  typedef Chroma::SingletonHolder< 
    ObjectFactory<MdagMSystemSolverArray<LatticeFermion>, 
		  std::string,
		  TYPELIST_4(XMLReader&, const std::string&, FactoryEnv::FSHandle, Handle< LinearOperatorArray<LatticeFermion> >),
		  MdagMSystemSolverArray<LatticeFermion>* (*)(XMLReader&,
							      const std::string&,
							      FactoryEnv::FSHandle,
							      Handle< LinearOperatorArray<LatticeFermion> >), 
		  StringFactoryError> >
  TheMdagMFermSystemSolverArrayFactory;

  //! MdagM system solver factory (foundry)
  /*! @ingroup invert */
  typedef Chroma::SingletonHolder<
    ObjectFactory<MdagMSystemSolverArray<LatticePropagator>,
          std::string,
          TYPELIST_4(XMLReader&, const std::string&, FactoryEnv::FSHandle4Q, Handle< LinearOperatorArray<LatticePropagator> >),
          MdagMSystemSolverArray<LatticePropagator>* (*)(XMLReader&,
                                  const std::string&,
                                  FactoryEnv::FSHandle4Q,
                                  Handle< LinearOperatorArray<LatticePropagator> >),
          StringFactoryError> >
  TheMdagMFerm4QSystemSolverArrayFactory;

  //! MdagM system solver factory (foundry)
  /*! @ingroup invert */
  typedef Chroma::SingletonHolder< 
    ObjectFactory<MdagMSystemSolverArray<LatticePropagator>, 
          std::string,       
          TYPELIST_4(XMLReader&, const std::string&, FactoryEnv::FSHandle4Q, Handle< LinearOperatorArray<LatticePropagator> >),
          MdagMSystemSolverArray<LatticePropagator>* (*)(XMLReader&,
                                  const std::string&,
                                  FactoryEnv::FSHandle4Q,
                                  Handle< LinearOperatorArray<LatticePropagator> >),
          StringFactoryError> >
  TheMdagMFerm4QSystemSolverArrayFactory;

  //! MdagM system solver factory (foundry)
  /*! @ingroup invert */
  typedef Chroma::SingletonHolder< 
    ObjectFactory<MdagMSystemSolver<LatticeStaggeredFermion>, 
		  std::string,
		  TYPELIST_3(XMLReader&, const std::string&, Handle< LinearOperator<LatticeStaggeredFermion> >),
		  MdagMSystemSolver<LatticeStaggeredFermion>* (*)(XMLReader&,
								  const std::string&,
								  Handle< LinearOperator<LatticeStaggeredFermion> >), 
		  StringFactoryError> >
  TheMdagMStagFermSystemSolverFactory;

}


#endif
