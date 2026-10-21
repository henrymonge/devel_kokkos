// -*- C++ -*-
/*! \file
 *  \brief Clover term linear operator
 */

#ifndef __exp_clover_term_base_w_h__
#define __exp_clover_term_base_w_h__

#include "chroma_config.h"
#include "linearop.h"
#include "actions/ferm/linop/clover_term_base_w.h"

namespace Chroma 
{ 
  //! Clover term
  /*!
   * \ingroup linop
   *
   */

  template<typename T, typename U>
	   class ExpCloverTermBase : public CloverTermBase< T, U>
  {
  public:
    //! No real need for cleanup here
    virtual ~ExpCloverTermBase() {}

    using CloverTermBase<T,U>::derivMultipole;

    void derivMultipole(multi1d<U>& ds_u,
			const multi1d<T>& chi, const multi1d<T>& psi,
			enum PlusMinus isign, int cb) const;

    //! d/dU Tr ln A_cb == 0 for exp-clover
    void derivTrLn(multi1d<U>& ds_u, enum PlusMinus isign, int cb) const
    {
      ds_u.resize(Nd);
      ds_u = zero;
    }


    //! Return flops performed by the operator()
    unsigned long nFlops() const;

  };

  //! Return flops performed by the operator()
  //! Flops per site of applyExpClov: one block-diagonal 12x12 Hermitian mat-vec
  template<typename T, typename U>
  unsigned long 
  ExpCloverTermBase<T,U>::nFlops() const {return 552;}

  template<typename T, typename U>
  void ExpCloverTermBase<T,U>::derivMultipole(multi1d<U>& ds_u, 
					   const multi1d<T>& chi, const multi1d<T>& psi, 
					   enum PlusMinus isign, int cb) const
  {
    // Multipole deriv
    START_CODE();
    
    // Do I still need to do this?
    if( ds_u.size() != Nd ) { 
      ds_u.resize(Nd);
    }

    ds_u = zero;

    // Get the links
    //const multi1d<U>& u = getU();


    // Now compute the insertions
    for(int mu=0; mu < Nd; mu++) {
      for(int nu = mu+1; nu < Nd; nu++) {
	
	// These will be appropriately overwritten - no need to zero them.
	// Contributions to mu links from mu-nu clover piece
	U ds_tmp_mu; 

	// -ve contribs  to the nu_links from the mu-nu clover piece 
	// -ve because of the exchange of gamma_mu gamma_nu <-> gamma_nu gamma_mu
	U ds_tmp_nu;

	// The weight for the terms
	Real factor = (Real(-1)/Real(8))*this->getCloverCoeff(mu,nu);

	// Get gamma_mu gamma_nu psi -- no saving here, from storing shifts because
	// I now only do every mu, nu pair only once.

	int mu_nu_index = (1 << mu) + (1 << nu); // 2^{mu} 2^{nu}

	// Accumulate all the trace spin outer products 
	U s_xy_dag = zero;

	for(int i=0; i < chi.size(); i++) {
      T ferm_tmp; 
	  ferm_tmp[rb[cb]]  = Gamma(mu_nu_index)*psi[i];
	  s_xy_dag[rb[cb]]  += traceSpin( outerProduct(ferm_tmp,chi[i]));
	}

    //For exp-clover, the psi and chi can be applied directly to the field, and thus the trace spin is zero?

	s_xy_dag[rb[cb]] *= Real(factor);

	// Compute contributions
	this->deriv_loops(mu, nu, cb, ds_tmp_mu, ds_tmp_nu, s_xy_dag);

	// Accumulate them
	ds_u[mu] += ds_tmp_mu;
	ds_u[nu] -= ds_tmp_nu;


      }
    }


    // Clear out the deriv on any fixed links
    (*this).getFermBC().zero(ds_u);
    END_CODE();
  }

} // End Namespace Chroma


#endif
