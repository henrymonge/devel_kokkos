// -*- C++ -*-
/*! \file
 *  \brief Exponential Clover term linear operator base
 *         Only the derivatives are modified for exponential clover term
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

    //! Take deriv of D
    /*!
     * \param chi     left std::vector                                 (Read)
     * \param psi     right std::vector                                (Read)
     * \param isign   D'^dag or D'  ( MINUS | PLUS ) resp.        (Read)
     *
     * \return Computes   \f$chi^\dag * \dot(D} * psi\f$
     */
    void deriv(multi1d<U>& ds_u, 
	       const T& chi, const T& psi, 
	       enum PlusMinus isign) const;

    //! Take deriv of D
    /*!
     * \param chi     left std::vector on cb                           (Read)
     * \param psi     right std::vector on 1-cb                        (Read)
     * \param isign   D'^dag or D'  ( MINUS | PLUS ) resp.        (Read)
     * \param cb      Checkerboard of chi std::vector                  (Read)
     *
     * \return Computes   \f$chi^\dag * \dot(D} * psi\f$
     */
    void deriv(multi1d<U>& ds_u, 
	       const T& chi, const T& psi, 
	       enum PlusMinus isign, int cb) const;

    //! Take deriv of D
    /*!
     * \param chi     left vectors                           (Read)
     * \param psi     right vectors                         (Read)
     * \param isign   D'^dag or D'  ( MINUS | PLUS ) resp.        (Read)
     * \param cb      Checkerboard of chi std::vector                  (Read)
     *
     * \return Computes   \f$chi^\dag * \dot(D} * psi\f$
     */
    void derivMultipole(multi1d<U>& ds_u,
			const multi1d<T>& chi, const multi1d<T>& psi,
			enum PlusMinus isign) const;

    //! Take deriv of D
    /*!
     * \param chi     left vectors on cb                           (Read)
     * \param psi     right vectors on cb                        (Read)
     * \param isign   D'^dag or D'  ( MINUS | PLUS ) resp.        (Read)
     * \param cb      Checkerboard of chi std::vector                  (Read)
     *
     * \return Computes   \f$chi^\dag * \dot(D} * psi\f$
     */

    void derivMultipole(multi1d<U>& ds_u,
			const multi1d<T>& chi, const multi1d<T>& psi,
			enum PlusMinus isign, int cb) const;


    /*
    //! Take derivative of TrLn D
    void derivTrLn(multi1d<U>& ds_u, 
		   enum PlusMinus isign, int cb) const;
    */
/*
    void deriv_loops(const int u, const int mu, const int cb,
		     U& ds_u_mu,
		     U& ds_u_nu,
		     const U& Lambda) const;
*/
    //! Return flops performed by the operator()
    unsigned long nFlops() const;

    //! Calculates Tr_D ( Gamma_mat L )
    virtual void triacntr(U& B, int mat, int cb) const = 0;

  protected:

    //! Get the u field
    virtual const multi1d<U>& getU() const = 0;

    //! get the clover coefficient 
    virtual Real getCloverCoeff(int mu, int nu) const = 0;

  };

  //! Return flops performed by the operator()
  template<typename T, typename U>
  unsigned long 
  ExpCloverTermBase<T,U>::nFlops() const {return 552;}


  //! Take deriv of D
  /*!
   * \param chi     left std::vector                                 (Read)
   * \param psi     right std::vector                                (Read)
   * \param isign   D'^dag or D'  ( MINUS | PLUS ) resp.        (Read)
   *
   * \return Computes   \f$\chi^\dag * \dot(D} * \psi\f$
   */
  template<typename T, typename U>
  void ExpCloverTermBase<T,U>::deriv(multi1d<U>& ds_u, 
			     const T& chi, const T& psi, 
			     enum PlusMinus isign) const
  {
    START_CODE();

    // base deriv resizes.
    // Even even checkerboard
    deriv(ds_u, chi, psi, isign,0);
    
    // Odd Odd checkerboard
    multi1d<U> ds_tmp;
    deriv(ds_tmp, chi, psi, isign,1);
    
    ds_u += ds_tmp;
    
    END_CODE();
  }

  template<typename T, typename U>
  void ExpCloverTermBase<T,U>::derivMultipole(multi1d<U>& ds_u, 
			     const multi1d<T>& chi, const multi1d<T>& psi, 
			     enum PlusMinus isign) const
  {
    START_CODE();


    QDPIO::cout << "I am running derivMultiple now debug" << std::endl;

    // base deriv resizes.
    // Even even checkerboard
    derivMultipole(ds_u, chi, psi, isign,0);
    
    // Odd Odd checkerboard
    multi1d<U> ds_tmp;
    derivMultipole(ds_tmp, chi, psi, isign,1);
    
    ds_u += ds_tmp;
    
    END_CODE();
  }

  //! Take deriv of D
  /*!
   * \param chi     left std::vector on cb                           (Read)
   * \param psi     right std::vector on 1-cb                        (Read)
   * \param isign   D'^dag or D'  ( MINUS | PLUS ) resp.        (Read)
   * \param cb      Checkerboard of chi std::vector                  (Read)
   *
   * \return Computes   \f$\chi^\dag * \dot(D} * \psi\f$
   */
  template<typename T, typename U>
  void ExpCloverTermBase<T,U>::deriv(multi1d<U>& ds_u, 
			     const T& chi, const T& psi, 
			     enum PlusMinus isign, int cb) const
  {

    START_CODE();
    //StopWatch swatch;
    //swatch.reset(); swatch.start();


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
	Real factor = (Real(-1)/Real(8))*getCloverCoeff(mu,nu);

	// Get gamma_mu gamma_nu psi -- no saving here, from storing shifts because
	// I now only do every mu, nu pair only once.

	int mu_nu_index = (1 << mu) + (1 << nu); // 2^{mu} 2^{nu}
	T ferm_tmp = Gamma(mu_nu_index)*psi;
	U s_xy_dag = traceSpin( outerProduct(ferm_tmp,chi));
	s_xy_dag *= Real(factor);

	// Compute contributions
	CloverTermBase< T, U>::deriv_loops(mu, nu, cb, ds_tmp_mu, ds_tmp_nu, s_xy_dag);

	// Accumulate them
	ds_u[mu] += ds_tmp_mu;
	ds_u[nu] -= ds_tmp_nu;


      }
    }

    //swatch.stop();
    //QDPIO::cout << "\nInner Deriv function time: "<< swatch.getTimeInSeconds() <<" s\n";

    // Clear out the deriv on any fixed links
    (*this).getFermBC().zero(ds_u);
    END_CODE();
  }

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
	Real factor = (Real(-1)/Real(8))*getCloverCoeff(mu,nu);

	// Get gamma_mu gamma_nu psi -- no saving here, from storing shifts because
	// I now only do every mu, nu pair only once.

	int mu_nu_index = (1 << mu) + (1 << nu); // 2^{mu} 2^{nu}

	// Accumulate all the trace spin outer products 
	U s_xy_dag = zero;
	for(int i=0; i < chi.size(); i++) { 
	  T ferm_tmp = Gamma(mu_nu_index)*psi[i];
	  s_xy_dag += traceSpin( outerProduct(ferm_tmp,chi[i]));
	}

    //For exp-clover, the psi and chi can be applied directly to the field, and thus the trace spin is zero?

	s_xy_dag *= Real(factor);

	// Compute contributions
	CloverTermBase< T, U>::deriv_loops(mu, nu, cb, ds_tmp_mu, ds_tmp_nu, s_xy_dag);

	// Accumulate them
	ds_u[mu] += ds_tmp_mu;
	ds_u[nu] -= ds_tmp_nu;


      }
    }


    // Clear out the deriv on any fixed links
    (*this).getFermBC().zero(ds_u);
    END_CODE();
  }

  
  //! Take deriv of D using Trace Log
  /*!
   * \param chi     left std::vector on cb                           (Read)
   * \param psi     right std::vector on 1-cb                        (Read)
   * \param isign   D'^dag or D'  ( MINUS | PLUS ) resp.        (Read)
   * \param cb      Checkerboard of chi std::vector                  (Read)
   *
   * \return Computes   \f$\chi^\dag * \dot(D} * \psi\f$  
   */
  /*
  template<typename T, typename U>
  void ExpCloverTermBase<T,U>::derivTrLn(multi1d<U>& ds_u, 
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
	  Real factor = Real(-1)*getCloverCoeff(mu,nu)/Real(8);
	  
	  U sigma_XY_dag=zero;

	  // Get  weight*Tr_spin gamma_mu gamma_nu A^{-1} piece
	  triacntr(sigma_XY_dag, mu_nu_index, cb);
	  //sigma_XY_dag[rb[cb]] *= factor;

      sigma_XY_dag[rb[cb]] *= factor*0;

	  // These will be overwritten so no need to initialize to zero
	  U ds_tmp_mu;
	  U ds_tmp_nu;

	  // Get contributions from the loops and insersions
	  CloverTermBase< T, U>::deriv_loops(mu, nu, cb, ds_tmp_mu, ds_tmp_nu, sigma_XY_dag);

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
  */  


} // End Namespace Chroma


#endif
