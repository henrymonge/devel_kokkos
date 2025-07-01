// -*- C++ -*-
/*! \file
 *  \brief Unpreconditioned Clover fermion action
 */

#ifndef __unprec_clover_4q_fermact_w_h__
#define __unprec_clover_4q_fermact_w_h__

#include "unprec_wilstype_fermact_w.h"
#include "actions/ferm/linop/lgherm_w.h"
#include "actions/ferm/fermacts/clover_fermact_params_w.h"

namespace Chroma
{
  //! Name and registration
  /*! \ingroup fermacts */
  namespace UnprecClover4QFermActEnv
  {
    extern const std::string name;
    bool registerAll();
  }

  //! Unpreconditioned Clover fermion action
  /*! \ingroup fermacts
   *
   * Unpreconditioned clover fermion action
   */
  class UnprecClover4QFermAct : public UnprecWilsonTypeFermAct<LatticeFermion, 
			      multi1d<LatticeColorMatrix>, multi1d<LatticeColorMatrix> >
  {
  public:
    // Typedefs to save typing
    typedef LatticeFermion               T;
    typedef multi1d<LatticeColorMatrix>  P;
    typedef multi1d<LatticeColorMatrix>  Q;

    //! General FermBC
    /*! Isotropic action */
    UnprecClover4QFermAct(Handle< CreateFermState<T,P,Q> > cfs_,
			const CloverFermActParams& param_) : 
      cfs(cfs_), param(param_) {}

    //! Copy constructor
    UnprecClover4QFermAct(const UnprecClover4QFermAct& a) : 
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
    ~UnprecClover4QFermAct() {}

  protected:
    //! Return the fermion BC object for this action
    const CreateFermState<T,P,Q>& getCreateState() const {return *cfs;}

    // Hide partial constructor
    UnprecClover4QFermAct() {}

    //! Assignment
    void operator=(const UnprecClover4QFermAct& a) {}

  private:
    Handle< CreateFermState<T,P,Q> > cfs;
    CloverFermActParams param;
  };

}

#endif
