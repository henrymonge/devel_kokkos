// -*- C++ -*-
/*! \file
 *  \brief Unpreconditioned Clover fermion linear operator
 */

#ifndef __unprec_clover_linop_2qblock_w_h__
#define __unprec_clover_linop_2qblock_w_h__

#include "linearop.h"
#include "actions/ferm/linop/dslash_2qblock_w.h"
#include "actions/ferm/linop/clover_term_2qblock_w.h"


namespace Chroma 
{ 
  //! Unpreconditioned Clover-Dirac operator
  /*!
   * \ingroup linop
   *
   * This routine is specific to Wilson fermions!
   */
  
  class UnprecClover2QBLinOp : public UnprecLinearOperator<LatticePropagator, 
			    multi1d<LatticeColorMatrix>, multi1d<LatticeColorMatrix> >
  {
  public:
    // Typedefs to save typing
    typedef LatticePropagator               T;
    typedef multi1d<LatticeColorMatrix>  P;
    typedef multi1d<LatticeColorMatrix>  Q;

    //! Partial constructor
    UnprecClover2QBLinOp() {}

    //! Full constructor
    UnprecClover2QBLinOp(Handle< FermState<T,P,Q> > fs,
		      const CloverFermActParams& param_)
      {create(fs,param_);}
    
    //! Destructor is automatic
    ~UnprecClover2QBLinOp() {}

    //! Return the fermion BC object for this linear operator
    const FermBC<T,P,Q>& getFermBC() const {return D.getFermBC();}

    //! Creation routine
    void create(Handle< FermState<T,P,Q> > fs,
		const CloverFermActParams& param_);

    //! Apply the operator onto a source std::vector
    void operator() (LatticePropagator& chi, const LatticePropagator& psi, enum PlusMinus isign) const;

    //! Derivative of unpreconditioned Clover dM/dU
    void deriv(multi1d<LatticeColorMatrix>& ds_u, 
	       const LatticePropagator& chi, const LatticePropagator& psi, 
	       enum PlusMinus isign) const;

    //! Return flops performed by the operator()
    unsigned long nFlops() const;

  private:
    CloverFermActParams param;
    WilsonDslash        D;
    CloverTerm          A;
  };

} // End Namespace Chroma


#endif
