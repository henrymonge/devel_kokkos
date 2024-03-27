// -*- C++ -*-
/*! \file
 *  \brief Clover term linear operator
 */

#ifndef __exp_clover_term_base_w_h__
#define __exp_clover_term_base_w_h__

#include "chroma_config.h"
#include "linearop.h"
#include "actions/ferm/linop/clover_term_force_base_w.h"
#include "actions/ferm/fermacts/clover_fermact_params_w.h"
#include "actions/ferm/linop/clov_triang_qdp_w.h"


namespace Chroma 
{ 
  //! Clover term
  /*!
   * \ingroup linop
   *
   */
  namespace
  {

    template <typename T>
    struct ExpClovTriang {

      // Unexponentiated part
      PrimitiveClovTriang<T> A;

      /*! Linear combination coefficients to generate the exponential */
      RScalar<T> q[2][6];

      // Exponentiated part
      RScalar<T> qinv[2][6];

      /*! Coefficients for the force*/
      RScalar<T> C[2][6][6];
    };


    /*! This accessor class allows me a convenient way to acces the
        diagonal + lower diagonal storage for the hermitian matrix */
    template <typename T, int block>
    struct ClovAccessor {
      ClovAccessor(PrimitiveClovTriang<T>& t) : tri(t)
      {
      }

      inline RComplex<T> operator()(int row, int col) const
      {
	RComplex<T> ret_val;
	if (row == col)
	{
	  // Diagonal Piece:
	  ret_val = tri.diag[block][row];
	}
	else if (row > col)
	{
	  // Lower triangular portion
	  ret_val = tri.offd[block][(row * (row - 1)) / 2 + col];
	}
	else if (row < col)
	{
	  // Upper triangular portion: transpose ( row <-> col) and conjugate
	  ret_val = conj(tri.offd[block][(col * (col - 1)) / 2 + row]);
	}

	return ret_val;
      }

      inline void insert(int row, int col, const RComplex<T>& value)
      {
	if (row == col)
	{
	  // Diagonal piece -- must be real.
	  tri.diag[block][row] = RScalar<T>(real(value));
	}
	else if (row > col)
	{
	  // Lower triangular portion
	  tri.offd[block][(row * (row - 1)) / 2 + col] = value;
	}
	else if (row < col)
	{
	  // Upper triangular portion: transpose ( row <-> col) and conjugate
	  tri.offd[block][(col * (col - 1)) / 2 + row] = conj(value);
	}
      }

    private:
      // A reference to the triangular storage
      PrimitiveClovTriang<T>& tri;
    };

    template <typename T, int block>
    struct Traces {
      Traces(ExpClovTriang<T>& E_) : E(E_)
      {
      }

      // Simple mat mult routine
      inline void multiply(ClovAccessor<T, block>& out, const ClovAccessor<T, block>& M1,
			   const ClovAccessor<T, block>& M2)
      {
	RComplex<T> zip(RScalar<T>((T)0), RScalar<T>((T)0));

	// NB: We only need to compute the diagonal and lower diagonal
	// elements because the matrices are hermitiean.
	for (int row = 0; row < 6; ++row)
	{
	  for (int col = 0; col <= row; ++col)
	  {
	    // Pour row down column
	    RComplex<T> dotprod = zip;
	    for (int k = 0; k < 6; ++k)
	    {
	      dotprod += M1(row, k) * M2(k, col);
	    }
	    out.insert(row, col, dotprod);
	  }
	}
      }

      // Simple mat mult routine
      inline void copy(ClovAccessor<T, block>& out, const ClovAccessor<T, block>& in)
      {
	// NB: We only need to compute the diagonal and lower diagonal
	// elements because the matrices are hermitiean.
	for (int row = 0; row < 6; ++row)
	{
	  for (int col = 0; col <= row; ++col)
	  {
	    out.insert(row, col, in(row, col));
	  }
	}
      }

      inline void traces(multi1d<RScalar<T>>& tr)
      {
	RComplex<T> zip(RScalar<T>((T)0), RScalar<T>((T)0));

	// The first 5 will map onto the hither powers.
	ClovAccessor<T, block> A(E.A);

	// Trace A
	tr.resize(6);
	tr[0] = RScalar<T>((T)0);
	for (int i = 0; i < 6; i++)
	{
	  tr[0] += real(A(i, i));
	}

	PrimitiveClovTriang<T> prev;
	PrimitiveClovTriang<T> curr;
	ClovAccessor<T, block> Prev(prev);
	ClovAccessor<T, block> Curr(curr);

	copy(Prev, A);

	for (int pow = 1; pow <= 5; pow++)
	{
	  multiply(Curr, Prev, A);
	  tr[pow] = RScalar<T>((T)0);
	  for (int i = 0; i < 6; i++)
	  {
	    tr[pow] += real(Curr(i, i));
	  }
	  copy(Prev, Curr);
	}
      }

    private:
      ExpClovTriang<T>& E;
    };


  } //end namespace


  template<typename T, typename U>
	   class ExpCloverTermBase : public CloverTermForceBase<T, U> //,public DslashLinearOperator<T,
							      //multi1d<U>,
							      //multi1d<U> >
  {
  public:
    //! No real need for cleanup here
    virtual ~ExpCloverTermBase() {}

    //! Subset is all here
    const Subset& subset() const {return all;}

    typedef typename WordType<T>::Type_t REALT;

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

    void deriv(multi1d<U>& ds_u,
           const T& chi, const T& psi,
           enum PlusMinus isign) const override;

    void deriv(multi1d<U>& ds_u,
           const T& chi, const T& psi,
           enum PlusMinus isign, int cb) const;

    void derivMultipole(multi1d<U>& ds_u,
            const multi1d<T>& chi, const multi1d<T>& psi,
            enum PlusMinus isign) const override;

    void derivMultipole(multi1d<U>& ds_u,
            const multi1d<T>& chi, const multi1d<T>& psi,
            enum PlusMinus isign, int cb) const;

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

    //Handle<FermBC<T, multi1d<U>, multi1d<U>>> fbc;
    //multi1d<U> u;
    //CloverFermActParams param;
    //LatticeDouble tr_M; // Fill this out during create

    ExpClovTriang<REALT>* tri;

  };


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
    
    ClovAccessor<T, 0> A(tri->A);
    //PrimitiveClovTriang<T> A = tri->A;
    
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


  template<typename T, typename U>
  void ExpCloverTermBase<T,U>::derivMultipole(multi1d<U>& ds_u,
                 const multi1d<T>& chi, const multi1d<T>& psi,
                 enum PlusMinus isign) const
  {
    START_CODE();

    // base deriv resizes.
    // Even even checkerboard
    derivMultipole(ds_u, chi, psi, isign,0);

    // Odd Odd checkerboard
    multi1d<U> ds_tmp;
    derivMultipole(ds_tmp, chi, psi, isign,1);

    ds_u += ds_tmp;

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

    s_xy_dag *= Real(factor);

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
