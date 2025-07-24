// -*- C++ -*-
/*! \file
 *  \brief Factory for producing system solvers for:
 *  (M  0)  (psi_1) = (chi_1)
 *  (0  M) (psi_2)   (chi_2)
 *  where M is not hermitian or pos. def.
 *  For regular solver, the system solvers are for M*psi = chi
 */

#ifndef __syssolver_linop_2qblock_factory_h__
#define __syssolver_linop_2qblock_factory_h__

#include "chromabase.h"
#include "handle.h"
#include "state.h"
#include "singleton.h"
#include "typelist.h"
#include "objfactory.h"
#include "actions/ferm/invert/syssolver_linop.h"

namespace Chroma
{
  namespace { 

    typedef Handle< FermState< LatticePropagator, multi1d<LatticeColorMatrix>, multi1d<LatticeColorMatrix> > > FSHandle2QB;

    typedef Handle< FermState< LatticePropagatorF, multi1d<LatticeColorMatrix>, multi1d<LatticeColorMatrix> > > FSHandleF2QB;

    typedef Handle< FermState< LatticePropagatorD, multi1d<LatticeColorMatrix>, multi1d<LatticeColorMatrix> > > FSHandleD2QB;



  }
  //! LinOp system solver factory (foundry)
  /*! @ingroup invert */
  typedef Chroma::SingletonHolder<
    ObjectFactory<LinOpSystemSolver<LatticePropagator>,
          std::string,
          TYPELIST_4(XMLReader&, const std::string&, FSHandle2QB,  Handle< LinearOperator<LatticePropagator> >),
          LinOpSystemSolver<LatticePropagator>* (*)(XMLReader&,
                             const std::string&,

                             FSHandle2QB,
                             Handle< LinearOperator<LatticePropagator> >),
          StringFactoryError> >
  TheLinOpFerm2QBSystemSolverFactory;


  typedef Chroma::SingletonHolder< 
    ObjectFactory<LinOpSystemSolver<LatticePropagatorF>, 
		  std::string,
		  TYPELIST_4(XMLReader&, const std::string&, FSHandleF2QB,  Handle< LinearOperator<LatticePropagatorF> >),
		  LinOpSystemSolver<LatticePropagatorF>* (*)(XMLReader&,
							 const std::string&,

							 FSHandleF2QB,
							 Handle< LinearOperator<LatticePropagatorF> >), 
		  StringFactoryError> >
  TheLinOpFFerm2QBSystemSolverFactory;

  typedef Chroma::SingletonHolder< 
    ObjectFactory<LinOpSystemSolver<LatticePropagatorD>, 
		  std::string,
		  TYPELIST_4(XMLReader&, const std::string&, FSHandleD2QB,  Handle< LinearOperator<LatticePropagatorD> >),
		  LinOpSystemSolver<LatticePropagatorD>* (*)(XMLReader&,
							 const std::string&,

							 FSHandleD2QB,
							 Handle< LinearOperator<LatticePropagatorD> >), 
		  StringFactoryError> >
  TheLinOpDFerm2QBSystemSolverFactory;


  //! LinOp system solver factory (foundry)
  /*! @ingroup invert */
  typedef Chroma::SingletonHolder< 
    ObjectFactory<LinOpSystemSolverArray<LatticePropagator>,
          std::string,
          TYPELIST_4(XMLReader&, const std::string&, FSHandle2QB, Handle< LinearOperatorArray<LatticePropagator> >),
          LinOpSystemSolverArray<LatticePropagator>* (*)(XMLReader&,
                                  const std::string&,
                                  FSHandle2QB,
                                  Handle< LinearOperatorArray<LatticePropagator> >),
          StringFactoryError> >
  TheLinOpFerm2QBSystemSolverArrayFactory;


  //! LinOp system solver factory (foundry)
  /*! @ingroup invert */
  typedef Chroma::SingletonHolder< 
    ObjectFactory<LinOpSystemSolver<LatticeStaggeredFermion>, 
		  std::string,
		  TYPELIST_3(XMLReader&, const std::string&, Handle< LinearOperator<LatticeStaggeredFermion> >),
		  LinOpSystemSolver<LatticeStaggeredFermion>* (*)(XMLReader&,
								  const std::string&,
								  Handle< LinearOperator<LatticeStaggeredFermion> >), 
		  StringFactoryError> >
  TheLinOpStagFerm2QBSystemSolverFactory;

  //! Projector factory (foundry)
  /*! @ingroup projector */
  typedef SingletonHolder<
    ObjectFactory<Projector<LatticePropagator>,
          std::string,
          TYPELIST_4(XMLReader&, const std::string&, FSHandle2QB,  Handle< LinearOperator<LatticePropagator> >),
          Projector<LatticePropagator>* (*)(XMLReader&,
                             const std::string&,
                             FSHandle2QB,
                             Handle< LinearOperator<LatticePropagator> >),
          StringFactoryError> >
  TheLinOpFerm2QBProjectorFactory;

}


#endif
