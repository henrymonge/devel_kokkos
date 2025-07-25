/*! \file
 *  \brief Unpreconditioned Clover fermion action for two quark product solves
 */

#include "chromabase.h"
#include "actions/ferm/fermacts/fermact_factory_2qblock_w.h"

#include "actions/ferm/linop/unprec_clover_2qblock_linop_w.h"
//#include "actions/ferm/linop/unprec_clover_linop_w.h"

#include "actions/ferm/fermacts/unprec_clover_fermact_2qblock_w.h"
#include "actions/ferm/invert/syssolver_linop_2qblock_factory.h"

//#include "actions/ferm/fermacts/fermact_factory_w.h"
//#include "actions/ferm/fermstates/ferm_createstate_reader_w.h"
#include "actions/ferm/fermstates/ferm_createstate_reader_2qblock_w.h"


namespace Chroma
{

  //! Hooks to register the class with the fermact factory
  namespace UnprecCloverFermAct2QBEnv
  {
    //! Callback function
    WilsonTypeFermAct2QB<LatticePropagator,
		      multi1d<LatticeColorMatrix>,
		      multi1d<LatticeColorMatrix> >* createFermAct4D2QB(XMLReader& xml_in,
								     const std::string& path)
    {
      return new UnprecCloverFermAct2QB(CreateFermStateEnv::reader(xml_in, path), 
				     CloverFermActParams(xml_in, path));
    }

    //! Callback function
    /*! Differs in return type */
    FermionAction<LatticePropagator,
		  multi1d<LatticeColorMatrix>,
		  multi1d<LatticeColorMatrix> >* createFermAct(XMLReader& xml_in,
							       const std::string& path)
    {
      return createFermAct4D2QB(xml_in, path);
    }

    //! Name to be used
    const std::string name = "UNPRECONDITIONED_CLOVER_2QB";

    //! Local registration flag
    static bool registered = false;

    //! Register all the factories
    bool registerAll() 
    {
      bool success = true; 
      if (! registered)
      {
	success &= Chroma::TheFermionAction2QBFactory::Instance().registerObject(name, createFermAct);
	success &= Chroma::TheWilsonTypeFermAct2QBFactory::Instance().registerObject(name, createFermAct4D2QB);
	registered = true;
      }
      return success;
    }
  }


  //! Produce a linear operator for this action
  /*!
   * The operator acts on the entire lattice
   *
   * \param state	    gauge field     	       (Read)
   */
  UnprecLinearOperator<LatticePropagator,
		       multi1d<LatticeColorMatrix>,
		       multi1d<LatticeColorMatrix> >* 
  UnprecCloverFermAct2QB::linOp(Handle< FermState<T,P,Q> > state) const
  {
    return new UnprecClover2QBLinOp(state,param);

  }


  //! Return a linear operator solver for this action to solve M*psi=chi 
  Projector<LatticePropagator>* 
  UnprecCloverFermAct2QB::projector(Handle< FermState<T,P,Q> > state,
				     const GroupXML_t& projParam) const
  {
    std::istringstream  is(projParam.xml);
    XMLReader  paramtop(is);
	
    return TheLinOpFerm2QBProjectorFactory::Instance().createObject(projParam.id,
								    paramtop,
								    projParam.path,
								    state,
								    linOp(state));
  }
}

