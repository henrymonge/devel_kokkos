// -*- C++ -*-
/*! \file
 *  \brief 2quarks block Fermion action factories
 */

#ifndef __fermact_factory_2qblock_w_h__
#define __fermact_factory_2qblock_w_h__

#include "singleton.h"
#include "objfactory.h"
#include "wilstype_fermact_2qblock_w.h"

namespace Chroma
{


  //! Wilson-like fermion factory for 2quark blocks(foundry)
  typedef Chroma::SingletonHolder<
  ObjectFactory<FermionAction<LatticePropagator,
                  multi1d<LatticeColorMatrix>,
                  multi1d<LatticeColorMatrix> >,
    std::string,
    TYPELIST_2(XMLReader&, const std::string&),
    FermionAction<LatticePropagator,
          multi1d<LatticeColorMatrix>,
          multi1d<LatticeColorMatrix> >* (*)(XMLReader&,
                             const std::string&), StringFactoryError> >
  TheFermionAction2QBFactory;

  //! Wilson-like fermion 4D factory 2quark blocks (foundry)
  typedef Chroma::SingletonHolder<
  ObjectFactory<WilsonTypeFermAct2QB<LatticePropagator,
                  multi1d<LatticeColorMatrix>,
                  multi1d<LatticeColorMatrix> >,
    std::string,
    TYPELIST_2(XMLReader&, const std::string&),
    WilsonTypeFermAct2QB<LatticePropagator,
              multi1d<LatticeColorMatrix>,
              multi1d<LatticeColorMatrix> >* (*)(XMLReader&,
                             const std::string&),
        StringFactoryError> >
  TheWilsonTypeFermAct2QBFactory;


  //! Wilson-like fermion array factory for 2quark blocks (foundry)
  typedef Chroma::SingletonHolder< 
  ObjectFactory<WilsonTypeFermAct5D2QB<LatticePropagator, 
				    multi1d<LatticeColorMatrix>, 
				    multi1d<LatticeColorMatrix> >, 
    std::string,
    TYPELIST_2(XMLReader&, const std::string&),
    WilsonTypeFermAct5D2QB<LatticePropagator, 
			multi1d<LatticeColorMatrix>,
			multi1d<LatticeColorMatrix> >* (*)(XMLReader&,
							   const std::string&), 
		StringFactoryError> >
  TheWilsonTypeFermAct5D2QBFactory;

}


#endif
