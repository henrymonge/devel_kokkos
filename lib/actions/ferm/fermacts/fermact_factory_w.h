// -*- C++ -*-
/*! \file
 *  \brief Fermion action factories
 */

#ifndef __fermact_factory_w_h__
#define __fermact_factory_w_h__

#include "singleton.h"
#include "objfactory.h"
#include "wilstype_fermact_w.h"
#include "wilstype_fermact_4q_w.h"

namespace Chroma
{
  //! Wilson-like fermion factory (foundry)
  typedef Chroma::SingletonHolder< 
  ObjectFactory<FermionAction<LatticeFermion,
			      multi1d<LatticeColorMatrix>,
			      multi1d<LatticeColorMatrix> >,
    std::string,
    TYPELIST_2(XMLReader&, const std::string&),
    FermionAction<LatticeFermion,
		  multi1d<LatticeColorMatrix>,
		  multi1d<LatticeColorMatrix> >* (*)(XMLReader&,
						     const std::string&), StringFactoryError> >
  TheFermionActionFactory;



  //! Wilson-like fermion factory 4Q(foundry)
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
  TheFermionAction4QFactory;

  //! Wilson-like fermion 4D factory (foundry)
  typedef Chroma::SingletonHolder< 
  ObjectFactory<WilsonTypeFermAct<LatticeFermion, 
				  multi1d<LatticeColorMatrix>,
				  multi1d<LatticeColorMatrix> >, 
    std::string,
    TYPELIST_2(XMLReader&, const std::string&),
    WilsonTypeFermAct<LatticeFermion, 
		      multi1d<LatticeColorMatrix>,
		      multi1d<LatticeColorMatrix> >* (*)(XMLReader&,
							 const std::string&), 
		StringFactoryError> >
  TheWilsonTypeFermActFactory;

  //! Wilson-like fermion 4D factory (foundry)
  typedef Chroma::SingletonHolder<
  ObjectFactory<WilsonTypeFermAct4Q<LatticePropagator,
                  multi1d<LatticeColorMatrix>,
                  multi1d<LatticeColorMatrix> >,
    std::string,
    TYPELIST_2(XMLReader&, const std::string&),
    WilsonTypeFermAct4Q<LatticePropagator,
              multi1d<LatticeColorMatrix>,
              multi1d<LatticeColorMatrix> >* (*)(XMLReader&,
                             const std::string&),
        StringFactoryError> >
  TheWilsonTypeFermAct4QFactory;


  //! Wilson-like fermion array factory (foundry)
  typedef Chroma::SingletonHolder< 
  ObjectFactory<WilsonTypeFermAct5D<LatticeFermion, 
				    multi1d<LatticeColorMatrix>, 
				    multi1d<LatticeColorMatrix> >, 
    std::string,
    TYPELIST_2(XMLReader&, const std::string&),
    WilsonTypeFermAct5D<LatticeFermion, 
			multi1d<LatticeColorMatrix>,
			multi1d<LatticeColorMatrix> >* (*)(XMLReader&,
							   const std::string&), 
		StringFactoryError> >
  TheWilsonTypeFermAct5DFactory;

}


#endif
