// -*- C++ -*-
/*! \file
 *  \brief Unpreconditioned Clover fermion action for 2 quark product solves
 */

#ifndef __unprec_clover_fermact_2qblock_w_h__
#define __unprec_clover_fermact_2qblock_w_h__

#include "unprec_wilstype_fermact_2qblock_w.h"
#include "actions/ferm/linop/lgherm_w.h"
#include "actions/ferm/fermacts/clover_fermact_params_w.h"

namespace Chroma
{
  //! Name and registration
  /*! \ingroup fermacts */
  namespace UnprecCloverFermAct2QBEnv
  {
    extern const std::string name;
    bool registerAll();
  }

  //! Unpreconditioned Clover fermion action
  /*! \ingroup fermacts
   *
   * Unpreconditioned clover fermion action
   */
  class UnprecCloverFermAct2QB : public UnprecWilsonTypeFermAct2QB<LatticePropagator, 
			      multi1d<LatticeColorMatrix>, multi1d<LatticeColorMatrix> >
  {
  public:
    // Typedefs to save typing
    typedef LatticePropagator               T;
    typedef multi1d<LatticeColorMatrix>  P;
    typedef multi1d<LatticeColorMatrix>  Q;

    //! General FermBC
    /*! Isotropic action */
    UnprecCloverFermAct2QB(Handle< CreateFermState<T,P,Q> > cfs_,
			const CloverFermActParams& param_) : 
      cfs(cfs_), param(param_) {}

    //! Copy constructor
    UnprecCloverFermAct2QB(const UnprecCloverFermAct2QB& a) : 
      cfs(a.cfs), param(a.param) {}

    //! Produce a linear operator for this action
    UnprecLinearOperator<T,P,Q>* linOp(Handle< FermState<T,P,Q> > state) const;

    LinearOperator<T>* hermitianLinOp(Handle< FermState<T,P,Q> > state) const 
      { 
	return new lgherm<T>(linOp(state));
      }

    //! Return a projector after this action
    Projector<T>* projector(Handle< FermState<T,P,Q> > state,
                            const GroupXML_t& projParam) const override;

    //! Destructor is automatic
    ~UnprecCloverFermAct2QB() {}

  protected:
    //! Return the fermion BC object for this action
    const CreateFermState<T,P,Q>& getCreateState() const {return *cfs;}

    // Hide partial constructor
    UnprecCloverFermAct2QB() {}

    //! Assignment
    void operator=(const UnprecCloverFermAct2QB& a) {}

  private:
    Handle< CreateFermState<T,P,Q> > cfs;
    CloverFermActParams param;
  };

}

#endif
