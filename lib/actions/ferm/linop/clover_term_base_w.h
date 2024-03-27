// -*- C++ -*-
/*! \file
 *  \brief Clover term linear operator
 */

#ifndef __clover_term_base_w_h__
#define __clover_term_base_w_h__

#include "chroma_config.h"
#include "linearop.h"
#include "actions/ferm/linop/clover_term_force_base_w.h"

namespace Chroma 
{ 
  //! Clover term
  /*!
   * \ingroup linop
   *
   */

  template<typename T, typename U>
	   class CloverTermBase : public CloverTermForceBase<T, U> //,public DslashLinearOperator<T,
							      //multi1d<U>,
							      //multi1d<U> >
  {
  public:
    //! No real need for cleanup here
    virtual ~CloverTermBase() {}

    //! Subset is all here
    const Subset& subset() const {return all;}


    virtual void applySite(T& chi, const T& psi, enum PlusMinus isign, int site) const = 0;

    //! Invert
    /*!
     * Computes the inverse of the term on cb using Cholesky
     */
    virtual void choles(int cb) = 0;

    //! Invert
    /*!
     * Computes the determinant of the term
     *
     * \return logarithm of the determinant  
     */
    virtual Double cholesDet(int cb) const = 0;

    //! Take derivative of TrLn D
    void derivTrLn(multi1d<U>& ds_u, 
		   enum PlusMinus isign, int cb) const;

    //! Return flops performed by the operator()
    //unsigned long nFlops() const;

    //! Calculates Tr_D ( Gamma_mat L )
    //virtual void triacntr(U& B, int mat, int cb) const = 0;

  protected:

    //! Get the u field
    virtual const multi1d<U>& getU() const = 0;

    //! get the clover coefficient 
    virtual Real getCloverCoeff(int mu, int nu) const = 0;

  };

  //! Take deriv of D using Trace Log
  /*!
   * \param chi     left std::vector on cb                           (Read)
   * \param psi     right std::vector on 1-cb                        (Read)
   * \param isign   D'^dag or D'  ( MINUS | PLUS ) resp.        (Read)
   * \param cb      Checkerboard of chi std::vector                  (Read)
   *
   * \return Computes   \f$\chi^\dag * \dot(D} * \psi\f$  
   */
  template<typename T, typename U>
  void CloverTermBase<T,U>::derivTrLn(multi1d<U>& ds_u, 
				 enum PlusMinus isign, int cb) const
  {
    START_CODE();
    
    // Do I still need to do this?
    if( ds_u.size() != Nd ) { 
      ds_u.resize(Nd);
    }
    
    ds_u = zero;

    for(int mu=0; mu < Nd; mu++) {
      for(int nu = mu+1; nu < Nd; nu++) { 

	  // Index 
	  int mu_nu_index = (1 << mu) + (1 << nu); // 2^{mu} 2^{nu}

	  // The actual coefficient factor
	  Real factor = Real(-1)*this->getCloverCoeff(mu,nu)/Real(8);
	  
	  U sigma_XY_dag=zero;

	  // Get  weight*Tr_spin gamma_mu gamma_nu A^{-1} piece
	  this->triacntr(sigma_XY_dag, mu_nu_index, cb);
	  sigma_XY_dag[rb[cb]] *= factor;

	  // These will be overwritten so no need to initialize to zero
	  U ds_tmp_mu;
	  U ds_tmp_nu;

	  // Get contributions from the loops and insersions
	  this->deriv_loops(mu, nu, cb, ds_tmp_mu, ds_tmp_nu, sigma_XY_dag);

	  // Accumulate
	  ds_u[mu] += ds_tmp_mu;
	  // -ve weight for nu from gamma_mu gamma_nu -> gamma_nu gamma_mu
	  // commutation.
	  ds_u[nu] -= ds_tmp_nu;

      } // End loop over nu

    } // end of loop over mu
    

    // Not sure this is needed here, but will be sure
    (*this).getFermBC().zero(ds_u);
    
    END_CODE();
  }
      


} // End Namespace Chroma


#endif
