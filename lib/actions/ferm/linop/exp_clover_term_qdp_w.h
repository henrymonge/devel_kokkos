// -*- C++ -*-
/*! \file
 *  \brief Clover term linear operator
 */

#ifndef __exp_clover_term_qdp_w_h__
#define __exp_clover_term_qdp_w_h__

#include "actions/ferm/fermacts/clover_fermact_params_w.h"
#include "actions/ferm/linop/clov_triang_qdp_w.h"
#include "actions/ferm/linop/exp_clover_term_base_w.h"
#include "meas/glue/mesfield.h"
#include "qdp_allocator.h"
#include "state.h"
#include <cmath>
#include <complex>
namespace Chroma
{

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

    constexpr int N_exp_default = 17;

    template <typename REALT, int block>
    inline void siteApplicationBlock(RComplex<REALT>* __restrict__ cchi,
				     const PrimitiveClovTriang<REALT>& tri_in,
				     const RComplex<REALT>* const __restrict__ ppsi)
    {

      if constexpr (block == 0)
      {
	cchi[0] = tri_in.diag[0][0] * ppsi[0] + conj(tri_in.offd[0][0]) * ppsi[1] +
		  conj(tri_in.offd[0][1]) * ppsi[2] + conj(tri_in.offd[0][3]) * ppsi[3] +
		  conj(tri_in.offd[0][6]) * ppsi[4] + conj(tri_in.offd[0][10]) * ppsi[5];

	cchi[1] = tri_in.diag[0][1] * ppsi[1] + tri_in.offd[0][0] * ppsi[0] +
		  conj(tri_in.offd[0][2]) * ppsi[2] + conj(tri_in.offd[0][4]) * ppsi[3] +
		  conj(tri_in.offd[0][7]) * ppsi[4] + conj(tri_in.offd[0][11]) * ppsi[5];

	cchi[2] = tri_in.diag[0][2] * ppsi[2] + tri_in.offd[0][1] * ppsi[0] +
		  tri_in.offd[0][2] * ppsi[1] + conj(tri_in.offd[0][5]) * ppsi[3] +
		  conj(tri_in.offd[0][8]) * ppsi[4] + conj(tri_in.offd[0][12]) * ppsi[5];

	cchi[3] = tri_in.diag[0][3] * ppsi[3] + tri_in.offd[0][3] * ppsi[0] +
		  tri_in.offd[0][4] * ppsi[1] + tri_in.offd[0][5] * ppsi[2] +
		  conj(tri_in.offd[0][9]) * ppsi[4] + conj(tri_in.offd[0][13]) * ppsi[5];

	cchi[4] = tri_in.diag[0][4] * ppsi[4] + tri_in.offd[0][6] * ppsi[0] +
		  tri_in.offd[0][7] * ppsi[1] + tri_in.offd[0][8] * ppsi[2] +
		  tri_in.offd[0][9] * ppsi[3] + conj(tri_in.offd[0][14]) * ppsi[5];

	cchi[5] = tri_in.diag[0][5] * ppsi[5] + tri_in.offd[0][10] * ppsi[0] +
		  tri_in.offd[0][11] * ppsi[1] + tri_in.offd[0][12] * ppsi[2] +
		  tri_in.offd[0][13] * ppsi[3] + tri_in.offd[0][14] * ppsi[4];
      }
      else
      {
	cchi[6] = tri_in.diag[1][0] * ppsi[6] + conj(tri_in.offd[1][0]) * ppsi[7] +
		  conj(tri_in.offd[1][1]) * ppsi[8] + conj(tri_in.offd[1][3]) * ppsi[9] +
		  conj(tri_in.offd[1][6]) * ppsi[10] + conj(tri_in.offd[1][10]) * ppsi[11];

	cchi[7] = tri_in.diag[1][1] * ppsi[7] + tri_in.offd[1][0] * ppsi[6] +
		  conj(tri_in.offd[1][2]) * ppsi[8] + conj(tri_in.offd[1][4]) * ppsi[9] +
		  conj(tri_in.offd[1][7]) * ppsi[10] + conj(tri_in.offd[1][11]) * ppsi[11];

	cchi[8] = tri_in.diag[1][2] * ppsi[8] + tri_in.offd[1][1] * ppsi[6] +
		  tri_in.offd[1][2] * ppsi[7] + conj(tri_in.offd[1][5]) * ppsi[9] +
		  conj(tri_in.offd[1][8]) * ppsi[10] + conj(tri_in.offd[1][12]) * ppsi[11];

	cchi[9] = tri_in.diag[1][3] * ppsi[9] + tri_in.offd[1][3] * ppsi[6] +
		  tri_in.offd[1][4] * ppsi[7] + tri_in.offd[1][5] * ppsi[8] +
		  conj(tri_in.offd[1][9]) * ppsi[10] + conj(tri_in.offd[1][13]) * ppsi[11];

	cchi[10] = tri_in.diag[1][4] * ppsi[10] + tri_in.offd[1][6] * ppsi[6] +
		   tri_in.offd[1][7] * ppsi[7] + tri_in.offd[1][8] * ppsi[8] +
		   tri_in.offd[1][9] * ppsi[9] + conj(tri_in.offd[1][14]) * ppsi[11];

	cchi[11] = tri_in.diag[1][5] * ppsi[11] + tri_in.offd[1][10] * ppsi[6] +
		   tri_in.offd[1][11] * ppsi[7] + tri_in.offd[1][12] * ppsi[8] +
		   tri_in.offd[1][13] * ppsi[9] + tri_in.offd[1][14] * ppsi[10];
      }
    }


    template < int N_exp, typename REALT, int block = 0>
    inline void siteApplicationExpPack(ExpClovTriang<REALT>& tri_out,ExpClovTriang<REALT>& tri_in,double mclov)
    {

      //Set the highest power of A^n for the exp sum. This allows for N_exp < 5 to compare with clover 
 
      RScalar<REALT> qi[6];
      // Row 0

      for (int i = 0; i < 6; i++)
          qi[i] = ((RScalar<REALT>) mclov)*tri_in.q[block][i];

 
      int pow_max=5;
      if (N_exp <5)
         pow_max=N_exp;

      //Set the output clov triang to 1+A (power 1)
  
      for (int c=0; c < 2*Nc ;c++){
          tri_out.A.diag[block][c]=qi[0]+qi[1]*tri_in.A.diag[block][c];
      }
    
     //Add the off-diagonal entries
      for (int c=0; c < 2*Nc*Nc-Nc; c++){
        tri_out.A.offd[block][c]=qi[1]*tri_in.A.offd[block][c];
      }
     
      Traces<REALT, block> tr(tri_out);
      PrimitiveClovTriang<REALT> prev;
      PrimitiveClovTriang<REALT> curr;
      ClovAccessor<REALT, block> tri_in_A(tri_in.A);
      ClovAccessor<REALT, block> tri_out_A(tri_out.A);
      ClovAccessor<REALT, block> Prev(prev); 
      ClovAccessor<REALT,block> Curr(curr);  
    
      tr.copy(Prev, tri_in_A);
      for (int pow = 2; pow <= pow_max; pow++)
      {
            
            tr.multiply(Curr, Prev, tri_in_A);

            //Add the diagonal entries
            for (int c=0; c < 2*Nc ;c++){          
                tri_out.A.diag[block][c]+=curr.diag[block][c]*qi[pow];
            }

            //Add the off-diagonal entries
            for (int c=0; c < 2*Nc*Nc-Nc; c++){
                tri_out.A.offd[block][c]+=curr.offd[block][c]*qi[pow];

            }
            tr.copy(Prev, Curr);
      } 
}

    template <int N_exp, typename REALT, int i = 0>
    inline void siteApplicationExp(const RScalar<REALT> ddiag_mass,
                   RComplex<REALT>* __restrict__ cchi,
				   const ExpClovTriang<REALT>& tri_in,
				   const RComplex<REALT>* const __restrict__ ppsi)
    {
      //Set the highest power of A^n for the exp sum. This allows for N_exp < 5 to compare with clover 
      int pow_max=5;
      if (N_exp <5)
         pow_max=N_exp;

      // Accumulate exponential from only A
      RComplex<REALT> tmp[12];
      // Top block
      // chi = psi
      for (int cspin = 0; cspin < 6; ++cspin)
      {
	cchi[cspin] = ppsi[cspin];
      }

      // Main loop:  chi = psi + q[i]/q[i-1] A chi
      for (int pow = pow_max; pow > 0; --pow)
      {
	siteApplicationBlock<REALT, 0>(tmp, tri_in.A, cchi);

	for (int cspin = 0; cspin < 6; cspin++)
	{

	  if constexpr (i == 0)
	  {
	    // Operator
	    cchi[cspin] = ppsi[cspin] + (tri_in.q[0][pow] / tri_in.q[0][pow - 1]) * tmp[cspin];
	  }
	  else
	  {
	    // Inverse
	    cchi[cspin] =
	      ppsi[cspin] + (tri_in.qinv[0][pow] / tri_in.qinv[0][pow - 1]) * tmp[cspin];
	  }
	}
      }

      for (int cspin = 0; cspin < 6; ++cspin)
      {
	if constexpr (i == 0)
	{
	  // Operator
	  cchi[cspin] *= ddiag_mass*tri_in.q[0][0];
	}
	else
	{
	  // inverse
	  cchi[cspin] *= ddiag_mass*tri_in.qinv[0][0];
	}
      }

      // Second Block
      // chi = psi
      for (int cspin = 6; cspin < 12; ++cspin)
      {
	cchi[cspin] = ppsi[cspin];
      }

      // Main loop:  chi = psi + q[i]/q[i-1] A chi
      for (int pow = pow_max; pow > 0; --pow)
      {
	siteApplicationBlock<REALT, 1>(tmp, tri_in.A, cchi);

	for (int cspin = 6; cspin < 12; cspin++)
	{
	  if constexpr (i == 0)
	  {
	    // Operator
	    cchi[cspin] = ppsi[cspin] + (tri_in.q[1][pow] / tri_in.q[1][pow - 1]) * tmp[cspin];
	  }
	  else
	  {
	    // Inverse
	    cchi[cspin] =
	      ppsi[cspin] + (tri_in.qinv[1][pow] / tri_in.qinv[1][pow - 1]) * tmp[cspin];
	  }
	}
      }

      for (int cspin = 6; cspin < 12; ++cspin)
      {

	if constexpr (i == 0)
	{
	  // Operator
	  cchi[cspin] *= ddiag_mass*tri_in.q[1][0];
	}
	else
	{
	  // Inverse
	  cchi[cspin] *= ddiag_mass*tri_in.qinv[1][0];
	}
      }

    }

   //Apply coefficient to site
    template <typename REALT>
    inline void siteApplicationCoeff(RComplex<REALT>* __restrict__ cchi, const ExpClovTriang<REALT>& tri_in,
                               int pow_i,int pow_j,
				               const RComplex<REALT>* const __restrict__ ppsi)
    {
    
      // Top block
      for (int cspin = 0; cspin < 6; cspin++)
          cchi[cspin] =  tri_in.C[0][pow_i][pow_j] * ppsi[cspin];  
       
      // Second Block
      for (int cspin = 6; cspin < 12; cspin++)
          cchi[cspin] = tri_in.C[1][pow_i][pow_j]* ppsi[cspin];
    }


    template <typename REALT>
    inline void siteApplicationPower(RComplex<REALT>* __restrict__ cchi,
				     const ExpClovTriang<REALT>& tri_in,
				     const RComplex<REALT>* const __restrict__ ppsi, int power)
    {
      if (power == 0)
      {
	// Straight Copy
	for (int i = 0; i < 12; ++i)
	  cchi[i] = ppsi[i];
      }
      else
      {
	// Higher powers, just construct as before.
	RComplex<REALT> tmp[12];

	siteApplicationBlock<REALT, 0>(cchi, tri_in.A, ppsi);
	for (int p = power; p > 1; --p)
	{
	  for (int i = 0; i < 6; ++i)
	    tmp[i] = cchi[i];
	  siteApplicationBlock<REALT, 0>(cchi, tri_in.A, tmp);
	}

	siteApplicationBlock<REALT, 1>(cchi, tri_in.A, ppsi);
	for (int p = power; p > 1; --p)
	{
	  for (int i = 6; i < 12; ++i)
	    tmp[i] = cchi[i];
	  siteApplicationBlock<REALT, 1>(cchi, tri_in.A, tmp);
	}
      }
    }

  } // namespace
  // Reader/writers
  /*! \ingroup linop */

  //! Clover term
  /*!
   * \ingroup linop
   *
   */
  template <typename T, typename U, int N_exp = N_exp_default>
  class QDPExpCloverTermT : public ExpCloverTermBase<T, U>
  {
  public:
    // Typedefs to save typing
    typedef typename WordType<T>::Type_t REALT;

    typedef OLattice<PScalar<PScalar<RScalar<typename WordType<T>::Type_t>>>> LatticeREAL;
    typedef OScalar<PScalar<PScalar<RScalar<REALT>>>> RealT;

    //! Empty constructor. Must use create later
    QDPExpCloverTermT();

    //! No real need for cleanup here
    ~QDPExpCloverTermT()
    {
      if (tri != nullptr)
      {
	QDP::Allocator::theQDPAllocator::Instance().free(tri);
      }
    }

    //! Creation routine
    void create(Handle<FermState<T, multi1d<U>, multi1d<U>>> fs, const CloverFermActParams& param_);

    virtual void create(Handle<FermState<T, multi1d<U>, multi1d<U>>> fs,
            const CloverFermActParams& param_,
            const QDPExpCloverTermT<T, U, N_exp>& from_, int inv_op);

    virtual void create(Handle<FermState<T, multi1d<U>, multi1d<U>>> fs,
			const CloverFermActParams& param_,
			const QDPExpCloverTermT<T, U, N_exp>& from_);

    virtual void createInv(Handle<FermState<T, multi1d<U>, multi1d<U>>> fs,
            const CloverFermActParams& param_,
            const QDPExpCloverTermT<T, U, N_exp>& from_);

    //! Computes the inverse of the term on cb using Cholesky
    /*!
     * \param cb   checkerboard of work (Read)
     */
    void choles(int cb) override;

    //! Computes the inverse of the term on cb using Cholesky
    /*!
     * \param cb   checkerboard of work (Read)
     * \return logarithm of the determinant
     * derivTrLn = 0 for exp-clover, so cholesDet is never used 
     */
    Double cholesDet(int cb) const { return Double(0); }
    /**
     * Apply a dslash
     *
     * Performs the operation
     *
     *  chi <-   (L + D + L^dag) . psi
     *
     * where
     *   L       is a lower triangular matrix
     *   D       is the real diagonal. (stored together in type TRIANG)
     *
     * Arguments:
     * \param chi     result                                      (Write)
     * \param psi     source                                      (Read)
     * \param isign   D'^dag or D'  ( MINUS | PLUS ) resp.        (Read)
     * \param cb      Checkerboard of OUTPUT std::vector               (Read)
     */

   //! Take deriv of D^power
    /*!
     * \param chi     left std::vector on cb                           (Read)
     * \param psi     right std::vector on 1-cb                        (Read)
     * \param isign   D'^dag or D'  ( MINUS | PLUS ) resp.        (Read)
     * \param cb      Checkerboard of chi std::vector                  (Read)
     *
     * \return Computes   \f$chi^\dag * \dot(D} * psi\f$
     */
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


    void fillRefDiag(Real diag);

    // Reference exponential using old fashioned taylor expansion
    void applyRef(T& chi, const T& psi, enum PlusMinus isign, int N = N_exp) const;

    // Apply a power of a matrix from A^0 to A^5
    void applyPower(T& chi, const T& psi, enum PlusMinus isign, int cb, int power = 1) const;

    // Apply coefficients to powers of a matrix A
    void applyCoeff(T& chi, const T& psi, enum PlusMinus isign,int cb, int pow_i, int pow_j) const;

    // Apply exponential operator
    void apply(T& chi, const T& psi, enum PlusMinus isign, int cb) const override;

    void applyExpClov(T& chi, const T& psi, enum PlusMinus isign, int cb) const ;

    void makeExpClov(int cb);

    // Apply exponential operator
    void applyInv(T& chi, const T& psi, enum PlusMinus isign, int cb) const;

    // Apply exponential operator at a site
    void applySite(T& chi, const T& psi, enum PlusMinus isign, int site) const override;

    inline void applyUnexp(T& chi, const T& psi, enum PlusMinus isign, int cb) const
    {
      applyPower(chi, psi, isign, cb, 1); // Explicily apply just the clover term.
    }

    //! Calculates Tr_D ( Gamma_mat L )
    void triacntr(U& B, int mat, int cb) const override;

    //! Return the fermion BC object for this linear operator
    const FermBC<T, multi1d<U>, multi1d<U>>& getFermBC() const override
    {
      return *fbc;
    }

    //! PACK UP the Clover term for QUDA library:
    void packForQUDA(multi1d<QUDAPackedClovSite<REALT>>& quda_pack, int cb) const;
    void tracePowers();
    ExpClovTriang<REALT>* tri;

  protected:
    //! Create the clover term on cb
    /*!
     *  \param f         field strength tensor F(mu,nu)        (Read)
     *  \param cb        checkerboard                          (Read)
     */
    void makeClov(const multi1d<U>& f, const RealT& diag_mass);

    // void tracePowers();

    //! Get the u field
    const multi1d<U>& getU() const override
    {
      return u;
    }

    //! Calculates Tr_D ( Gamma_mat L )
    Real getCloverCoeff(int mu, int nu) const override;

  private:
    Handle<FermBC<T, multi1d<U>, multi1d<U>>> fbc;
    multi1d<U> u;
    CloverFermActParams param;
    RealT diag_mass;
    ExpClovTriang<REALT>* exp_tri;
  };

  // Empty constructor. Must use create later
  template <typename T, typename U, int N_exp>
  QDPExpCloverTermT<T, U, N_exp>::QDPExpCloverTermT()
  {
    // Always allocate on construction
    int nodeSites = Layout::sitesOnNode();
    tri = (ExpClovTriang<REALT>*)QDP::Allocator::theQDPAllocator::Instance().allocate(
      nodeSites * sizeof(ExpClovTriang<REALT>), QDP::Allocator::DEFAULT);
    exp_tri = (ExpClovTriang<REALT>*)QDP::Allocator::theQDPAllocator::Instance().allocate(
      nodeSites * sizeof(ExpClovTriang<REALT>), QDP::Allocator::DEFAULT);
  }

  // Now copy
  template <typename T, typename U, int N_exp>
  void QDPExpCloverTermT<T, U, N_exp>::create(Handle<FermState<T, multi1d<U>, multi1d<U>>> fs,
					      const CloverFermActParams& param_,
					      const QDPExpCloverTermT<T, U, N_exp>& from, int inv_op)
  {
#ifndef QDP_IS_QDPJIT
    START_CODE();
    u.resize(Nd);

    u = fs->getLinks();
    fbc = fs->getFermBC();
    param = param_;

    // Sanity check
    if (fbc.operator->() == 0)
    {
      QDPIO::cerr << "QDPCloverTerm: error: fbc is null" << std::endl;
      QDP_abort(1);
    }

    //
    // Yuk. Some bits of knowledge of the dslash term are buried in the
    // effective mass term. They show up here. If I wanted some more
    // complicated dslash then this will have to be fixed/adjusted.
    //
    
    RealT ff = param.anisoParam.anisoP ? param.anisoParam.nu / param.anisoParam.xi_0 : Real(1);
    diag_mass = 1 + (Nd - 1) * ff + param.Mass;
    
    if (inv_op==1){
       diag_mass = 1/( 1 + (Nd - 1) * ff + param.Mass);

    }
    else{
        diag_mass =  1 + (Nd - 1) * ff + param.Mass;

    }
    
    {
      RealT ff = param.anisoParam.anisoP ? Real(1) / param.anisoParam.xi_0 : Real(1);
      param.clovCoeffR *= Real(0.5) * ff / diag_mass;
      param.clovCoeffT *= Real(0.5) / diag_mass;
    }

    
    //diag_mass = 1/from.diag_mass;

    //if (inv_op==1)
    //     diag_mass = 1.0/diag_mass;
    /* Calculate F(mu,nu) */
    // multi1d<LatticeColorMatrix> f;
    // mesField(f, u);
    // makeClov(f, diag_mass);

    int nodeSites = Layout::sitesOnNode();
    // Deep copy.
#  pragma omp parallel for
    for (int site = 0; site < nodeSites; ++site)
    {
      for (int block = 0; block < 2; ++block)
      {

	for (int d = 0; d < 6; ++d)
	{
	  tri[site].A.diag[block][d] = from.tri[site].A.diag[block][d];

	}
    
	for (int od = 0; od < 15; ++od)
	{
	  tri[site].A.offd[block][od] = from.tri[site].A.offd[block][od];
	}

	// The exponentiation coefficients
	for (int i = 0; i < 6; ++i)
	{
      if (inv_op==1)
      {
        tri[site].qinv[block][i] = from.tri[site].q[block][i];
        tri[site].q[block][i] = from.tri[site].qinv[block][i]; 
      }
      else
      {
	    tri[site].q[block][i] = from.tri[site].q[block][i];
	    tri[site].qinv[block][i] = from.tri[site].qinv[block][i];
      }
 
	}

	// The force coefficients
	for (int i = 0; i < 6; ++i)
	{
	  for (int j = 0; j < 6; ++j)
	  {
	    tri[site].C[block][i][j] = from.tri[site].C[block][i][j];
	  }
	}
      } // End site loop

    }
     QDPIO::cout << "Diag mass " << diag_mass.elem().elem().elem().elem()<< "\n";

    END_CODE();
#endif
  }

#if 1
  // Now copy
  template <typename T, typename U, int N_exp>
  void QDPExpCloverTermT<T, U, N_exp>::create(Handle<FermState<T, multi1d<U>, multi1d<U>>> fs,
					      const CloverFermActParams& param_,
					      const QDPExpCloverTermT<T, U, N_exp>& from)
  {  
     QDPIO::cout << "Creating from copy\n";
     create(fs,param_,from,0);
  }
#endif
  // Now copy and create inv
  template <typename T, typename U, int N_exp>
  void QDPExpCloverTermT<T, U, N_exp>::createInv(Handle<FermState<T, multi1d<U>, multi1d<U>>> fs,
                          const CloverFermActParams& param_,
                          const QDPExpCloverTermT<T, U, N_exp>& from)
  {
     QDPIO::cout << "Creating inverse\n";
     create(fs,param_,from,1);
  }


  //! Creation routine
  template <typename T, typename U, int N_exp>
  void QDPExpCloverTermT<T, U, N_exp>::create(Handle<FermState<T, multi1d<U>, multi1d<U>>> fs,
					      const CloverFermActParams& param_)
  {
#ifndef QDP_IS_QDPJIT
    START_CODE()

    u.resize(Nd);

    u = fs->getLinks();
    fbc = fs->getFermBC();
    param = param_;

    // Sanity check
    if (fbc.operator->() == 0)
    {
      QDPIO::cerr << "QDPCloverTerm: error: fbc is null" << std::endl;
      QDP_abort(1);
    }

    //
    // Yuk. Some bits of knowledge of the dslash term are buried in the
    // effective mass term. They show up here. If I wanted some more
    // complicated dslash then this will have to be fixed/adjusted.
    //
    
    RealT ff = param.anisoParam.anisoP ? param.anisoParam.nu / param.anisoParam.xi_0 : Real(1);
    diag_mass = 1 + (Nd - 1) * ff + param.Mass;
    
    {
      RealT ff = param.anisoParam.anisoP ? Real(1) / param.anisoParam.xi_0 : Real(1);
      param.clovCoeffR *= RealT(0.5) * ff / diag_mass;
      param.clovCoeffT *= RealT(0.5) / diag_mass;
    }

    /* Calculate F(mu,nu) */
    multi1d<U> f;
    mesField(f, u);
    makeClov(f, diag_mass);
    tracePowers();


    END_CODE();
#endif
  }

  namespace QDPExpCloverEnv
  {

    template <typename U>
    struct FillRefArg {
      typedef typename WordType<U>::Type_t REALT;
      typedef OScalar<PScalar<PScalar<RScalar<REALT>>>> RealT;
      const REALT& ref_val;
      ExpClovTriang<REALT>* tri;
    };

    /* This is the extracted site loop for makeClover */
    template <typename U>
    inline void fillRefLoop(int lo, int hi, int myId, FillRefArg<U>* a)
    {
#ifndef QDP_IS_QDPJIT
      typedef typename FillRefArg<U>::RealT RealT;
      typedef typename FillRefArg<U>::REALT REALT;

      const REALT ref_val = a->ref_val;
      ExpClovTriang<REALT>* tri = a->tri;

      // SITE LOOP STARTS HERE
      for (int site = lo; site < hi; ++site)
      {
	for (int chiral = 0; chiral < 2; chiral++)
	{
	  for (int diag_index = 0; diag_index < 2 * Nc; diag_index++)
	  {

	    tri[site].A.diag[chiral][diag_index] = RScalar<REALT>(ref_val);
	  }

	  for (int offdiag_index = 0; offdiag_index < 15; offdiag_index++)
	  {
	    tri[site].A.offd[chiral][offdiag_index].real() = 0;
	    tri[site].A.offd[chiral][offdiag_index].imag() = 0;
	  }
	}
      }
#endif
    }

  } // namespace QDPExpCloverEnv

  template <typename T, typename U, int N_exp>
  void QDPExpCloverTermT<T, U, N_exp>::fillRefDiag(Real ref_val)
  {
#ifndef QDP_IS_QDPJIT
    START_CODE();

    if (Ns != 4)
    {
      QDPIO::cerr << __func__ << ": CloverTerm::apply requires Ns==4" << std::endl;
      QDP_abort(1);
    }

    QDPExpCloverEnv::FillRefArg<U> arg = {toDouble(ref_val), tri};
    int num_sites = all.siteTable().size();

    // The dispatch function is at the end of the file
    // ought to work for non-threaded targets too...
    dispatch_to_threads(num_sites, arg, QDPExpCloverEnv::fillRefLoop<U>);
    END_CODE();
#endif
  }

  /*Threaded this. It needs a QMT arg struct and I've extracted the site loop */
  namespace QDPExpCloverEnv
  {

    template <typename U>
    struct QDPCloverMakeClovArg {
      typedef typename WordType<U>::Type_t REALT;
      typedef OScalar<PScalar<PScalar<RScalar<REALT>>>> RealT;
      const RealT& diag_mass;
      const U& f0;
      const U& f1;
      const U& f2;
      const U& f3;
      const U& f4;
      const U& f5;
      ExpClovTriang<REALT>* tri;
    };

    /* This is the extracted site loop for makeClover */
    template <typename U>
    inline void makeClovSiteLoop(int lo, int hi, int myId, QDPCloverMakeClovArg<U>* a)
    {
#ifndef QDP_IS_QDPJIT
      typedef typename QDPCloverMakeClovArg<U>::RealT RealT;
      typedef typename QDPCloverMakeClovArg<U>::REALT REALT;

      const RealT& diag_mass = a->diag_mass;
      const U& f0 = a->f0;
      const U& f1 = a->f1;
      const U& f2 = a->f2;
      const U& f3 = a->f3;
      const U& f4 = a->f4;
      const U& f5 = a->f5;
      ExpClovTriang<REALT>* tri = a->tri;
      const int power = 1;

      // SITE LOOP STARTS HERE
      for (int site = lo; site < hi; ++site)
      {
	for (int jj = 0; jj < 2; jj++)
	{
	  for (int ii = 0; ii < 2 * Nc; ii++)
	  {
	    tri[site].A.diag[jj][ii] = 0;
	  }
	}

	RComplex<REALT> E_minus;
	RComplex<REALT> B_minus;
	RComplex<REALT> ctmp_0;
	RComplex<REALT> ctmp_1;
	RScalar<REALT> rtmp_0;
	RScalar<REALT> rtmp_1;

	for (int i = 0; i < Nc; ++i)
	{

	  /*# diag_L(i,0) = 1 - i*diag(E_z - B_z) */
	  /*#             = 1 - i*diag(F(3,2) - F(1,0)) */
	  ctmp_0 = f5.elem(site).elem().elem(i, i);
	  ctmp_0 -= f0.elem(site).elem().elem(i, i);
	  rtmp_0 = imag(ctmp_0);
	  tri[site].A.diag[0][i] += rtmp_0;

	  /*# diag_L(i+Nc,0) = 1 + i*diag(E_z - B_z) */
	  /*#                = 1 + i*diag(F(3,2) - F(1,0)) */
	  tri[site].A.diag[0][i + Nc] -= rtmp_0;

	  /*# diag_L(i,1) = 1 + i*diag(E_z + B_z) */
	  /*#             = 1 + i*diag(F(3,2) + F(1,0)) */
	  ctmp_1 = f5.elem(site).elem().elem(i, i);
	  ctmp_1 += f0.elem(site).elem().elem(i, i);
	  rtmp_1 = imag(ctmp_1);
	  tri[site].A.diag[1][i] -= rtmp_1;

	  /*# diag_L(i+Nc,1) = 1 - i*diag(E_z + B_z) */
	  /*#                = 1 - i*diag(F(3,2) + F(1,0)) */
	  tri[site].A.diag[1][i + Nc] += rtmp_1;
	}

	/*# Construct lower triangular portion */
	/*# Block diagonal terms */
	for (int i = 1; i < Nc; ++i)
	{
	  for (int j = 0; j < i; ++j)
	  {

	    int elem_ij = i * (i - 1) / 2 + j;
	    int elem_tmp = (i + Nc) * (i + Nc - 1) / 2 + j + Nc;

	    /*# L(i,j,0) = -i*(E_z - B_z)[i,j] */
	    /*#          = -i*(F(3,2) - F(1,0)) */
	    ctmp_0 = f0.elem(site).elem().elem(i, j);
	    ctmp_0 -= f5.elem(site).elem().elem(i, j);
	    tri[site].A.offd[0][elem_ij] = timesI(ctmp_0);

	    /*# L(i+Nc,j+Nc,0) = +i*(E_z - B_z)[i,j] */
	    /*#                = +i*(F(3,2) - F(1,0)) */
	    tri[site].A.offd[0][elem_tmp] = -tri[site].A.offd[0][elem_ij];

	    /*# L(i,j,1) = i*(E_z + B_z)[i,j] */
	    /*#          = i*(F(3,2) + F(1,0)) */
	    ctmp_1 = f5.elem(site).elem().elem(i, j);
	    ctmp_1 += f0.elem(site).elem().elem(i, j);
	    tri[site].A.offd[1][elem_ij] = timesI(ctmp_1);

	    /*# L(i+Nc,j+Nc,1) = -i*(E_z + B_z)[i,j] */
	    /*#                = -i*(F(3,2) + F(1,0)) */
	    tri[site].A.offd[1][elem_tmp] = -tri[site].A.offd[1][elem_ij];
	  }
	}

	/*# Off-diagonal */
	for (int i = 0; i < Nc; ++i)
	{
	  for (int j = 0; j < Nc; ++j)
	  {

	    // Flipped index
	    // by swapping i <-> j. In the past i would run slow
	    // and now j runs slow
	    int elem_ij = (i + Nc) * (i + Nc - 1) / 2 + j;

	    /*# i*E_- = (i*E_x + E_y) */
	    /*#       = (i*F(3,0) + F(3,1)) */
	    E_minus = timesI(f2.elem(site).elem().elem(i, j));
	    E_minus += f4.elem(site).elem().elem(i, j);

	    /*# i*B_- = (i*B_x + B_y) */
	    /*#       = (i*F(2,1) - F(2,0)) */
	    B_minus = timesI(f3.elem(site).elem().elem(i, j));
	    B_minus -= f1.elem(site).elem().elem(i, j);

	    /*# L(i+Nc,j,0) = -i*(E_- - B_-)  */
	    tri[site].A.offd[0][elem_ij] = B_minus - E_minus;

	    /*# L(i+Nc,j,1) = +i*(E_- + B_-)  */
	    tri[site].A.offd[1][elem_ij] = E_minus + B_minus;
	  }
	}
      } /* End Site loop */
#endif
    } /* Function */
  }   // namespace QDPExpCloverEnv

  /* This now just sets up and dispatches... */
  template <typename T, typename U, int N_exp>
  void QDPExpCloverTermT<T, U, N_exp>::makeClov(const multi1d<U>& f, const RealT& diag_mass)
  {
    START_CODE();

    if (Nd != 4)
    {
      QDPIO::cerr << __func__ << ": expecting Nd==4" << std::endl;
      QDP_abort(1);
    }

    if (Ns != 4)
    {
      QDPIO::cerr << __func__ << ": expecting Ns==4" << std::endl;
      QDP_abort(1);
    }

    U f0 = f[0] * getCloverCoeff(0, 1);
    U f1 = f[1] * getCloverCoeff(0, 2);
    U f2 = f[2] * getCloverCoeff(0, 3);
    U f3 = f[3] * getCloverCoeff(1, 2);
    U f4 = f[4] * getCloverCoeff(1, 3);
    U f5 = f[5] * getCloverCoeff(2, 3);

    const int nodeSites = QDP::Layout::sitesOnNode();
    QDPExpCloverEnv::QDPCloverMakeClovArg<U> arg = {diag_mass, f0, f1, f2, f3, f4, f5, tri};
    dispatch_to_threads(nodeSites, arg, QDPExpCloverEnv::makeClovSiteLoop<U>);

    END_CODE();
  }

  //! Invert
  /*!
   * Computes the inverse of the term on cb using Cholesky
   */
  template <typename T, typename U, int N_exp>
  void QDPExpCloverTermT<T, U, N_exp>::choles(int cb)
  {
    //This is not needed for exp-clover
    //QDPIO::cerr << "JITExpCloverTerm: unimplemented function for exponentiated clover" << std::endl;
  }

  //! Take deriv of D
  /*! 
   * \param chi     left std::vector                                 (Read)
   * \param psi     right std::vector                                (Read)
   * \param isign   D'^dag or D'  ( MINUS | PLUS ) resp.        (Read)
   *  
   * \return Computes   \f$\chi^\dag * \dot(D} * \psi\f$
   */ 
  template<typename T, typename U, int N_exp>
  void QDPExpCloverTermT<T, U, N_exp>::deriv(multi1d<U>& ds_u,
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
   * \param isign   D'^dag or D'  ( MINUS | PLUS ) resp.             (Read)
   * \param cb      Checkerboard of chi std::vector                  (Read)
   *
   * \return Computes   \f$\chi^\dag * \dot(D} * \psi\f$
   */
//Improved derivative
  template <typename T, typename U, int N_exp>
  void QDPExpCloverTermT<T, U, N_exp>::deriv(multi1d<U>& ds_u,
                 const T& chi, const T& psi,
                 enum PlusMinus isign, int cb) const
  {
    START_CODE();

    //StopWatch swatch;
    //swatch.reset(); swatch.start();
    //QDPIO::cout << "\nUsing improved deriv function \n";


    // Do I still need to do this?
    if( ds_u.size() != Nd ) {
      ds_u.resize(Nd);
    }

    ds_u = zero;
    // Get the links
    //const multi1d<U>& u = getU();
    T ppsi;
    T tmp_psi;

    multi1d<T> cchi_vec;
    multi1d<T> sum_psi_vec;
    cchi_vec.resize(6);
    sum_psi_vec.resize(6);

    // The exp derivative is computed as
    // A'+AA'/2+A'A/2+A'AA/6+AA'A/6+AAA'/6 = Sum A^i A' A^j
    // applyCoeff multiplies the chi by the exponential term factor 
    // and the factors from using the Caley Hamilton for A^n, for n>5

    for(int i=0;i<=5;i++){
        sum_psi_vec[i]= zero;
        cchi_vec[i]= zero;

        for(int j=0;j<=5;j++){
            (*this).applyCoeff(tmp_psi, psi, isign,cb,i,j);
            (*this).applyPower(ppsi, tmp_psi,isign, cb, j);
            sum_psi_vec[i][rb[cb]]+=ppsi;
        }

        (*this).applyPower(cchi_vec[i], chi, isign, cb,i);

    }

    ExpCloverTermBase<T,U>::derivMultipole(ds_u,cchi_vec,sum_psi_vec,isign,cb);
   
    // Clear out the deriv on any fixed links
    (*this).getFermBC().zero(ds_u);
    END_CODE();
  }


  template <typename T, typename U, int N_exp>
  void QDPExpCloverTermT<T, U, N_exp>::derivMultipole(multi1d<U>& ds_u,
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

//Improved derivative
  template <typename T, typename U, int N_exp>
  void QDPExpCloverTermT<T, U, N_exp>::derivMultipole(multi1d<U>& ds_u,
                 const multi1d<T>& chi, const multi1d<T>& psi,
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

    T tmp_psi; //psi;
    T ppsi;

    multi1d<T> cchi_vec;
    multi1d<T> sum_psi_vec;
   
    //for every fermion in chi, we have 6 terms, so total number is 6*chi.size()
    int num_terms=6*chi.size();
    cchi_vec.resize(num_terms);
    sum_psi_vec.resize(num_terms);

    // The exp derivative is computed as
    // A'+AA'/2+A'A/2+A'AA/6+AA'A/6+AAA'/6 = Sum A^i A' A^j
    // applyCoeff multiplies the chi by the exponential term factor 
    // and the factors from using the Caley Hamilton for A^n, for n>5
    int nterm=0;
    for(int k=0;k<chi.size();k++){       
        for(int i=0;i<=5;i++){
            sum_psi_vec[nterm]= zero;
            cchi_vec[nterm]= zero;

            for(int j=0;j<=5;j++){
                (*this).applyCoeff(tmp_psi, psi[k], isign,cb,i,j);
                (*this).applyPower(ppsi, tmp_psi, isign, cb, j);
                sum_psi_vec[nterm][rb[cb]]+=ppsi;
            }

            (*this).applyPower(cchi_vec[nterm], chi[k], isign, cb,i);
            nterm+=1;
        }
    }
    ExpCloverTermBase<T,U>::derivMultipole(ds_u,cchi_vec,sum_psi_vec,isign,cb);

    
    //swatch.stop();
    
    // Clear out the deriv on any fixed links
    (*this).getFermBC().zero(ds_u);
    END_CODE();
  } 

  namespace QDPExpCloverEnv
  {

    template <typename U>
    struct TracePowersArgs {
      typedef typename WordType<U>::Type_t REALT;
      typedef OScalar<PScalar<PScalar<RScalar<REALT>>>> RealT;
      typedef OLattice<PScalar<PScalar<RScalar<REALT>>>> LatticeRealT;

      ExpClovTriang<REALT>* tri;
      int cb;
      REALT diag_mass; 
    };

    template <typename U, int N_exp>
    inline void TracePowerSiteLoop(int lo, int hi, int myId, TracePowersArgs<U>* a)
    {
      typedef typename TracePowersArgs<U>::REALT REALT;
      typedef typename TracePowersArgs<U>::RealT RealT;
      typedef typename TracePowersArgs<U>::LatticeRealT LatticeRealT;

      ExpClovTriang<REALT>* tri = a->tri;
      int cb = a->cb;

      RScalar<REALT> zip = 0;

      // Loop through the sites.
      for (int ssite = lo; ssite < hi; ++ssite)
      {

	int site = rb[cb].siteTable()[ssite];

	// Compute the q-s for the block
	REALT tab[2][N_exp + 1][6] = {0};

	for (int block = 0; block < 2; ++block)
	{
	  constexpr int upper = N_exp + 1 < 6 ? N_exp + 1 : 6;
	  REALT ifact = 1;
	  for (int i = 0; i < upper; ++i)
	  {
	    tab[block][i][i] = (REALT)1;
	    ifact = -ifact;
	  }
	}

	Traces<REALT, 0> tr0(tri[site]);
	Traces<REALT, 1> tr1(tri[site]);

	multi1d<RScalar<REALT>> trace0(6);
	multi1d<RScalar<REALT>> trace1(6);
	tr0.traces(trace0);
	tr1.traces(trace1);

	if (N_exp + 1 > 6)
	{
	  REALT trace[2][6];
	  for (int i = 0; i < 6; ++i)
	    trace[0][i] = toDouble(trace0[i]);
	  for (int i = 0; i < 6; ++i)
	    trace[1][i] = toDouble(trace1[i]);

	  for (int block = 0; block < 2; ++block)
	  {
	    REALT p[5];

	    p[4] = ((REALT)1 / (REALT)2) * trace[block][1]; // (1/2) Tr A^2

	    p[3] = ((REALT)1 / (REALT)3) * trace[block][2]; // (1/3) Tr A^3
	    p[2] = ((REALT)1 / (REALT)4) * trace[block][3] -
		   ((REALT)1 / (REALT)8) * trace[block][1] *
		     trace[block][1]; // (1/4) Tr A^4 - (1/8) (Tr A^2)^2

	    p[1] = ((REALT)1 / (REALT)5) * trace[block][4] -
		   ((REALT)1 / (REALT)6) * trace[block][2] *
		     trace[block][1]; // (1/5) Tr A^5 - (1/6) Tr A^3 Tr A^2

	    p[0] = ((REALT)1 / (REALT)6) * trace[block][5] // (1/6) Tr A^6 - (1/8) Tr A^4 Tr A^2
		   - ((REALT)1 / (REALT)8) * trace[block][3] *
		       trace[block][1] //     - (1/18) [ Tr A^3 ]^2
		   - ((REALT)1 / (REALT)18) * trace[block][2] *
		       trace[block][2] //     + (1/48) [ Tr A^2 ]^3
		   + ((REALT)1 / (REALT)48) * trace[block][1] * trace[block][1] * trace[block][1];

	    // Row 6
	    for (int i = 0; i < 5; ++i)
	    {
	      tab[block][6][i] = p[i];
	    }

	    // Row 7+

	    for (int row = 7; row <= N_exp; ++row)
	    {
	      for (int i = 0; i < 5; i++)
	      {
		for (int j = 0; j < 6; j++)
		{
		  tab[block][row][j] += p[i] * tab[block][row - 6 + i][j];
		}
	      }
	    }
	  } // Blocks
	}   // N_exp + 1 >

	// Sum into the q
	for (int block = 0; block < 2; ++block)
	{

	  // Row 0
	  for (int i = 0; i < 6; i++)
	  {
	    tri[site].q[block][i] = RScalar<REALT>(tab[block][0][i]);
	    tri[site].qinv[block][i] = RScalar<REALT>(tab[block][0][i]);
	  }

	  unsigned long fact = 1;
	  for (unsigned int row = 1; row <= N_exp; ++row)
	  {
	    fact *= (unsigned long)row;
	    REALT sign = (row % 2 == 0) ? (REALT)1 : (REALT)(-1);
	    for (int i = 0; i < 6; i++)
	    {
	      tri[site].q[block][i] += RScalar<REALT>(tab[block][row][i] / (REALT)(fact));
	      tri[site].qinv[block][i] += RScalar<REALT>(sign * tab[block][row][i] / (REALT)(fact));
	    }
	  }

      //HMC: adding the calculation of the C_ij
      for (int i = 0; i < 6; i++)
      {
        for (int j = 0; j < 6; j++)
        {
            tri[site].C[block][i][j] = RScalar<REALT>(tab[block][0][i])*RScalar<REALT>(tab[block][0][j]);
        }
      }

      fact = 1;
      unsigned long fact_row = 1;

      for (int row = 0; row <= N_exp - 1; ++row)
      { 
        if (row!=0)
           fact_row *= (unsigned long)(row);
        fact=fact_row*(unsigned long)(row+1);
        for(int col = 0; col <= N_exp-1-row; ++col)
        {
            if(row!=0 || col!=0) //row=0, col=0 computed above
            {
                
                //This is the factor on the exp = c_n x^n, for the derivative of the n-term x^row x'x^col 
                //the factor is row+col+1,where row+col=n-1 
                if( col !=0)
                    fact *= (unsigned long)(row+col+1);  
                for (int i = 0; i < 6; i++)
                {
                  for (int j = 0; j < 6; j++)
                  {            
                      tri[site].C[block][i][j] += RScalar<REALT>(tab[block][col][j])*RScalar<REALT>(tab[block][row][i] / (REALT)(fact));    
                  }
                }
            }
        }
      }

      const RScalar<REALT> dm(a->diag_mass);
      for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++)
          tri[site].C[block][i][j] *= dm;

	}//for block ends

	// Assemble te exponential from the q-s and powers of A.
	// siteExponentiate(tri[site]);
      } /* End Site Loop */
    }	/* End Function */

  } // namespace QDPExpCloverEnv

  /* This now just sets up and dispatches... */
  template <typename T, typename U, int N_exp>
  void QDPExpCloverTermT<T, U, N_exp>::tracePowers()
  {
    START_CODE();

    if (Nd != 4)
    {
      QDPIO::cerr << __func__ << ": expecting Nd==4" << std::endl;
      QDP_abort(1);
    }

    if (Ns != 4)
    {
      QDPIO::cerr << __func__ << ": expecting Ns==4" << std::endl;
      QDP_abort(1);
    }

    const REALT dm = diag_mass.elem().elem().elem().elem();
    for (int cb = 0; cb < 2; ++cb)
    {
      const int num_sites = rb[cb].siteTable().size();
      QDPExpCloverEnv::TracePowersArgs<U> arg = {tri, cb,dm};
      dispatch_to_threads(num_sites, arg, QDPExpCloverEnv::TracePowerSiteLoop<U, N_exp>);
    }
    END_CODE();
  }

   /**
   * Apply a dslash
   *
   * Performs the operation
   *
   *  chi <-   (L + D + L^dag) . psi
   *
   * where
   *   L       is a lower triangular matrix
   *   D       is the real diagonal. (stored together in type TRIANG)
   *
   * Arguments:
   * \param chi     result                                      (Write)
   * \param psi     source                                      (Read)
   * \param isign   D'^dag or D'  ( MINUS | PLUS ) resp.        (Read)
   * \param cb      Checkerboard of OUTPUT std::vector               (Read)
   */
  template <typename T, typename U, int N_exp>
  void QDPExpCloverTermT<T, U, N_exp>::applySite(T& chi, const T& psi, enum PlusMinus isign,
						 int site) const
  {
#ifndef QDP_IS_QDPJIT
    START_CODE();

    if (Ns != 4)
    {
      QDPIO::cerr << __func__ << ": CloverTerm::applySite requires Ns==4" << std::endl;
      QDP_abort(1);
    }

    RComplex<REALT>* cchi = (RComplex<REALT>*)&(chi.elem(site).elem(0).elem(0));
    const RComplex<REALT>* const ppsi =
      (const RComplex<REALT>* const) & (psi.elem(site).elem(0).elem(0));
    //siteApplicationExp(cchi, tri[site], ppsi);
    const RScalar<REALT> ddiag_mass = diag_mass.elem(site).elem().elem();
    siteApplicationExp<N_exp>(ddiag_mass, cchi, tri[site], ppsi);

    END_CODE();
#endif
  }

  //! Returns the appropriate clover coefficient for indices mu and nu
  template <typename T, typename U, int N_exp>
  Real QDPExpCloverTermT<T, U, N_exp>::getCloverCoeff(int mu, int nu) const
  {
    START_CODE();

    if (param.anisoParam.anisoP)
    {
      if (mu == param.anisoParam.t_dir || nu == param.anisoParam.t_dir)
      {
	return param.clovCoeffT;
      }
      else
      {
	// Otherwise return the spatial coeff
	return param.clovCoeffR;
      }
    }
    else
    {
      // If there is no anisotropy just return the spatial one, it will
      // be the same as the temporal one
      return param.clovCoeffR;
    }

    END_CODE();
  }

  namespace QDPExpCloverEnv
  {
    template <typename T>
    struct ApplyPowerArgs {
      typedef typename WordType<T>::Type_t REALT;
      T& chi;
      const T& psi;
      const ExpClovTriang<REALT>* tri;
      int cb;
      int power = 1;
      ApplyPowerArgs(T& _chi, const T& _psi,const ExpClovTriang<REALT>* _tri,int _cb,int _power) : chi(_chi), psi(_psi), tri(_tri), cb(_cb), power(_power) {}
    };

    template <typename T>
    struct ApplyDerivCoeffArgs {
      typedef typename WordType<T>::Type_t REALT;
      T& chi;
      const T& psi;       
      const ExpClovTriang<REALT>* tri;
      int cb;
      int pow_i = 1;
      int pow_j = 1;
      ApplyDerivCoeffArgs(T& _chi, const T& _psi,const ExpClovTriang<REALT>* _tri,int _cb,int _pow_i, int _pow_j) : chi(_chi), psi(_psi), tri(_tri), cb(_cb), pow_i(_pow_i), pow_j(_pow_j) {}

    };


    template <typename T>
    void applySitePowerLoop(int lo, int hi, int MyId, ApplyPowerArgs<T>* arg)
    {
#ifndef QDP_IS_QDPJIT
      // This is essentially the body of the previous "Apply"
      // but now the args are handed in through user arg struct...

      START_CODE();

      typedef typename WordType<T>::Type_t REALT;
      // Unwrap the args...
      T& chi = arg->chi;
      const T& psi = arg->psi;
      const ExpClovTriang<REALT>* tri = arg->tri;
      int cb = arg->cb;
      int power = arg->power;
      const int n = 2 * Nc;

      for (int ssite = lo; ssite < hi; ++ssite)
      {

	int site = rb[cb].siteTable()[ssite];

	RComplex<REALT>* cchi = (RComplex<REALT>*)&(chi.elem(site).elem(0).elem(0));

	const RComplex<REALT>* const ppsi =
	  (const RComplex<REALT>* const) & (psi.elem(site).elem(0).elem(0));

	siteApplicationPower<REALT>(cchi, tri[site], ppsi, power);
      }
      END_CODE();
#endif
    } // Function

    template <typename T>
    struct ApplyArgs {
      typedef typename WordType<T>::Type_t REALT;
      typedef OScalar<PScalar<PScalar<RScalar<REALT>>>> RealT;

      RealT diag_mass;
      T& chi;
      const T& psi;
      const ExpClovTriang<REALT>* tri;
      int cb;
    };

    template <typename T>
    void applySiteCoeffLoop(int lo, int hi, int MyId, ApplyDerivCoeffArgs<T>* arg)
    {
#ifndef QDP_IS_QDPJIT
      // This is essentially the body of the previous "Apply"
      // but now the args are handed in through user arg struct...

      START_CODE();

      typedef typename WordType<T>::Type_t REALT;
      // Unwrap the args...
      T& chi = arg->chi;
      const T& psi = arg->psi;
      const ExpClovTriang<REALT>* tri = arg->tri;
      int cb = arg->cb;
      int pow_i = arg->pow_i;
      int pow_j = arg->pow_j;
      const int n = 2 * Nc;

      for (int ssite = lo; ssite < hi; ++ssite)
      {

    int site = rb[cb].siteTable()[ssite];

    RComplex<REALT>* cchi = (RComplex<REALT>*)&(chi.elem(site).elem(0).elem(0));

    const RComplex<REALT>* const ppsi =
      (const RComplex<REALT>* const) & (psi.elem(site).elem(0).elem(0));

    siteApplicationCoeff<REALT>(cchi, tri[site], pow_i,pow_j, ppsi);
      }
      END_CODE();
#endif
    } // Function

    template <typename T>
    struct makeExpClovArgs {
      typedef typename WordType<T>::Type_t REALT;
      //typedef OLattice<PScalar<PScalar<RScalar<REALT>>>> LatticeRealT;
      ExpClovTriang<REALT>* tri_out;
      ExpClovTriang<REALT>* tri;
      int cb;
      Real diag_mass;
    };


    template <int N_exp, typename T>
    void makeExpClovSiteLoop(int lo, int hi, int MyId, makeExpClovArgs<T>* arg)
    {
#ifndef QDP_IS_QDPJIT
      // This is essentially the body of the previous "Apply"
      // but now the args are handed in through user arg struct...

      START_CODE();

      typedef typename WordType<T>::Type_t REALT;
      typedef OLattice<PScalar<PScalar<RScalar<REALT>>>> LatticeRealT;
      // Unwrap the args...
      ExpClovTriang<REALT>* tri = arg->tri;
      ExpClovTriang<REALT>* tri_out = arg->tri_out;
      double mclov=  (arg->diag_mass).elem().elem().elem().elem();

      int cb = arg->cb;
      const int n = 2 * Nc;
	
      for (int ssite = lo; ssite < hi; ++ssite)
      {
         int site = rb[cb].siteTable()[ssite];
         siteApplicationExpPack<N_exp,REALT,0>(tri_out[site],tri[site], mclov);
         siteApplicationExpPack<N_exp,REALT,1>(tri_out[site],tri[site], mclov);

      }
      END_CODE();
#endif
    } // Function



    template <int N_exp, typename T, int inv = 0>
    void applySiteLoop(int lo, int hi, int MyId, ApplyArgs<T>* arg)
    {
#ifndef QDP_IS_QDPJIT
      // This is essentially the body of the previous "Apply"
      // but now the args are handed in through user arg struct...

      START_CODE();

      typedef typename WordType<T>::Type_t REALT;
      typedef OScalar<PScalar<PScalar<RScalar<REALT>>>> RealT;
      // Unwrap the args...
      RealT diag_mass=arg->diag_mass;
      T& chi = arg->chi;
      const T& psi = arg->psi;
      const ExpClovTriang<REALT>* tri = arg->tri;
      int cb = arg->cb;
      const int n = 2 * Nc;

      for (int ssite = lo; ssite < hi; ++ssite)
      {

	int site = rb[cb].siteTable()[ssite];
	RComplex<REALT>* cchi = (RComplex<REALT>*)&(chi.elem(site).elem(0).elem(0));
	const RComplex<REALT>* const ppsi =
	  (const RComplex<REALT>* const) & (psi.elem(site).elem(0).elem(0));
        const RScalar<REALT> ddiag_mass = diag_mass.elem(site).elem().elem();
    siteApplicationExp<N_exp,REALT, inv>(ddiag_mass,cchi, tri[site], ppsi);

	//siteApplicationExp<REALT, inv>(cchi, tri[site], ppsi);
      }
      END_CODE();
#endif
    } // Function

    template <typename T, int inv = 0>
    void applyExpClovSiteLoop(int lo, int hi, int MyId, ApplyArgs<T>* arg)

    {

      // This is essentially the body of the previous "Apply"
      // but now the args are handed in through user arg struct...
#ifndef QDP_IS_QDPJIT
      START_CODE();

      typedef typename WordType<T>::Type_t REALT;
      typedef OScalar<PScalar<PScalar<RScalar<REALT>>>> RealT;
      // Unwrap the args...
      RealT diag_mass=arg->diag_mass;
      T& chi = arg->chi;
      const T& psi = arg->psi;
      const ExpClovTriang<REALT>* tri = arg->tri;
      int cb = arg->cb;

      const int n = 2 * Nc;

      const int* tab = rb[cb].siteTable().slice();

      // Now just loop from low to high sites...
      for (int ssite = lo; ssite < hi; ++ssite)
      {


	int site = tab[ssite];
#define NEW
#ifndef NEW
	RComplex<REALT>* cchi = (RComplex<REALT>*)&(chi.elem(site).elem(0).elem(0));
	const RComplex<REALT>* ppsi = (const RComplex<REALT>*)&(psi.elem(site).elem(0).elem(0));
#else
	std::complex<REALT>* cchi = (std::complex<REALT>*)&(chi.elem(site).elem(0).elem(0));
	std::complex<REALT>* ppsi = (std::complex<REALT>*)&(psi.elem(site).elem(0).elem(0));
	const REALT* const diag0 = (const REALT* const)(&(tri[site].A.diag[0][0].elem()));
	const REALT* const diag1 = (const REALT* const)(&(tri[site].A.diag[1][0].elem()));
	const std::complex<REALT>* const offdiag0 =
	  (const std::complex<REALT>* const)(&(tri[site].A.offd[0][0].real()));
	const std::complex<REALT>* const offdiag1 =
	  (const std::complex<REALT>* const)(&(tri[site].A.offd[1][0].real()));

#endif

#if 1
#  warning "Using unrolled clover term"
	// Rolled version
	for (int i = 0; i < n; ++i)
	{
#  ifndef NEW
	  cchi[0 * n + i] = tri[site].A.diag[0][i] * ppsi[0 * n + i];
	  cchi[1 * n + i] = tri[site].A.diag[1][i] * ppsi[1 * n + i];
#  else
	  cchi[0 * n + i] = diag0[i] * ppsi[0 * n + i];
	  cchi[1 * n + i] = diag1[i] * ppsi[1 * n + i];

#  endif
	}

	int kij = 0;
	for (int i = 0; i < n; ++i)
	{

	  for (int j = 0; j < i; j++)
	  {
#  ifndef NEW
	    cchi[0 * n + i] += tri[site].A.offd[0][kij] * ppsi[0 * n + j];
	    cchi[0 * n + j] += conj(tri[site].A.offd[0][kij]) * ppsi[0 * n + i];
	    cchi[1 * n + i] += tri[site].A.offd[1][kij] * ppsi[1 * n + j];
	    cchi[1 * n + j] += conj(tri[site].A.offd[1][kij]) * ppsi[1 * n + i];
#  else
	    cchi[0 * n + i] += offdiag0[kij] * ppsi[0 * n + j];
	    cchi[0 * n + j] += conj(offdiag0[kij]) * ppsi[0 * n + i];
	    cchi[1 * n + i] += offdiag1[kij] * ppsi[1 * n + j];
	    cchi[1 * n + j] += conj(offdiag1[kij]) * ppsi[1 * n + i];
#  endif
	    kij++;
	  }
	}
#endif
      }

      END_CODE();
#endif
    } // Function


  }   // namespace QDPExpCloverEnv

  template <typename T, typename U, int N_exp>
  void QDPExpCloverTermT<T, U, N_exp>::applyRef(T& chi, const T& psi, enum PlusMinus isign,
						int N) const
  {
    chi = psi;
    T tmp;
    for (int j = N; j > 1; --j)
    {
      for (int cb = 0; cb < 2; ++cb)
	(*this).applyPower(tmp, chi, isign, cb, 1); // M^1 chi
      tmp /= Real(j);				    // (1/N) M chi
      chi = psi + tmp;				    // ( psi + (1/N)M chi)
    }
    for (int cb = 0; cb < 2; ++cb)
      (*this).applyPower(tmp, chi, isign, cb, 1); // M chi

    chi = psi + tmp; // psi + M chi
    (*this).getFermBC().modifyF(chi);
  }
  /**
   * Apply a dslash
   *
   * Performs the operation
   *
   *  chi <-   (L + D + L^dag) . psi
   *
   * where
   *   L       is a lower triangular matrix
   *   D       is the real diagonal. (stored together in type TRIANG)
   *
   * Arguments:
   * \param chi     result                                      (Write)
   * \param psi     source                                      (Read)
   * \param isign   D'^dag or D'  ( MINUS | PLUS ) resp.        (Read)
   * \param cb      Checkerboard of OUTPUT std::vector               (Read)
   */

  template <typename T, typename U, int N_exp>
  void QDPExpCloverTermT<T, U, N_exp>::applyPower(T& chi, const T& psi, enum PlusMinus isign,
						  int cb, int power) const
  {
#ifndef QDP_IS_QDPJIT
    START_CODE();

    if (Ns != 4)
    {
      QDPIO::cerr << __func__ << ": CloverTerm::apply requires Ns==4" << std::endl;
      QDP_abort(1);
    }

    QDPExpCloverEnv::ApplyPowerArgs<T> arg = {chi, psi, tri, cb, power};
    int num_sites = rb[cb].siteTable().size();

    // The dispatch function is at the end of the file
    // ought to work for non-threaded targets too...
    dispatch_to_threads(num_sites, arg, QDPExpCloverEnv::applySitePowerLoop<T>);
    (*this).getFermBC().modifyF(chi, QDP::rb[cb]);

    END_CODE();
#endif
  }

  template <typename T, typename U, int N_exp>
  void QDPExpCloverTermT<T, U,N_exp>::applyCoeff(T& chi, const T& psi, enum PlusMinus isign,int cb, int pow_i, int pow_j) const
  {
#ifndef QDP_IS_QDPJIT
    START_CODE();

    if (Ns != 4)
    {
      QDPIO::cerr << __func__ << ": CloverTerm::apply requires Ns==4" << std::endl;
      QDP_abort(1);
    }

    QDPExpCloverEnv::ApplyDerivCoeffArgs<T> arg = {chi, psi, tri, cb, pow_i, pow_j};
    int num_sites = rb[cb].siteTable().size();

    // The dispatch function is at the end of the file
    // ought to work for non-threaded targets too...
    dispatch_to_threads(num_sites, arg, QDPExpCloverEnv::applySiteCoeffLoop<T>);
    (*this).getFermBC().modifyF(chi, QDP::rb[cb]);

    END_CODE();
#endif
  }

  template <typename T, typename U, int N_exp>
  void QDPExpCloverTermT<T, U, N_exp>::makeExpClov(int cb)
  {
#ifndef QDP_IS_QDPJIT
    START_CODE();

    if (Ns != 4)
    {
      QDPIO::cerr << __func__ << ": CloverTerm::apply requires Ns==4" << std::endl;
      QDP_abort(1);
    }

    Real mclov= diag_mass; //RealT(Nd) + param.Mass;
    //if(inverse==1)
    //    mclov= 1.0/mclov ; 

    QDPExpCloverEnv::makeExpClovArgs<T> arg = {exp_tri, tri, cb,mclov};
    int num_sites = rb[cb].siteTable().size();


    // The dispatch function is at the end of the file
    // ought to work for non-threaded targets too...
    dispatch_to_threads(num_sites, arg, QDPExpCloverEnv::makeExpClovSiteLoop<N_exp,T>);
    
    //(*this).getFermBC().modifyF(chi, QDP::rb[cb]);

    END_CODE();
#endif
  }


  template <typename T, typename U, int N_exp>
  void QDPExpCloverTermT<T, U, N_exp>::applyExpClov(T& chi, const T& psi, enum PlusMinus isign,
					     int cb) const
  {
#ifndef QDP_IS_QDPJIT
    START_CODE();

    if (Ns != 4)
    {
      QDPIO::cerr << __func__ << ": CloverTerm::apply requires Ns==4" << std::endl;
      QDP_abort(1);
    }




    QDPExpCloverEnv::ApplyArgs<T> arg = {diag_mass,chi, psi, exp_tri, cb};
    //QDPExpCloverEnv::ApplyArgs<T> arg = {chi, psi, tri, cb};

    int num_sites = rb[cb].siteTable().size();

    //printExpClov();

    // The dispatch function is at the end of the file
    // ought to work for non-threaded targets too...
    dispatch_to_threads(num_sites, arg, QDPExpCloverEnv::applyExpClovSiteLoop<T, 0>);
    (*this).getFermBC().modifyF(chi, QDP::rb[cb]);

    END_CODE();
#endif
  }


  template <typename T, typename U, int N_exp>
  void QDPExpCloverTermT<T, U, N_exp>::apply(T& chi, const T& psi, enum PlusMinus isign,
					     int cb) const
  {
#ifndef QDP_IS_QDPJIT
    START_CODE();

    if (Ns != 4)
    {
      QDPIO::cerr << __func__ << ": CloverTerm::apply requires Ns==4" << std::endl;
      QDP_abort(1);
    }

    QDPExpCloverEnv::ApplyArgs<T> arg = {diag_mass,chi, psi, tri, cb};
    int num_sites = rb[cb].siteTable().size();

    // The dispatch function is at the end of the file
    // ought to work for non-threaded targets too...
    dispatch_to_threads(num_sites, arg, QDPExpCloverEnv::applySiteLoop<N_exp,T, 0>);
    (*this).getFermBC().modifyF(chi, QDP::rb[cb]);

    END_CODE();
#endif
  }

  template <typename T, typename U, int N_exp>
  void QDPExpCloverTermT<T, U, N_exp>::applyInv(T& chi, const T& psi, enum PlusMinus isign,
						int cb) const
  {
#ifndef QDP_IS_QDPJIT
    START_CODE();

    if (Ns != 4)
    {
      QDPIO::cerr << __func__ << ": CloverTerm::apply requires Ns==4" << std::endl;
      QDP_abort(1);
    }

    RealT ddiag_mass = 1.0/diag_mass;
    QDPExpCloverEnv::ApplyArgs<T> arg = {ddiag_mass,chi, psi, tri, cb};
    int num_sites = rb[cb].siteTable().size();

    // The dispatch function is at the end of the file
    // ought to work for non-threaded targets too...
    dispatch_to_threads(num_sites, arg, QDPExpCloverEnv::applySiteLoop<N_exp,T, 1>);
    (*this).getFermBC().modifyF(chi, QDP::rb[cb]);

    END_CODE();
#endif
  }

  //! TRIACNTR
  /*!
   * \ingroup linop
   *
   *  Calculates
   *     Tr_D ( Gamma_mat L )
   *
   * This routine is specific to Wilson fermions!
   *
   *  the trace over the Dirac indices for one of the 16 Gamma matrices
   *  and a hermitian color x spin matrix A, stored as a block diagonal
   *  complex lower triangular matrix L and a real diagonal diag_L.

   *  Here 0 <= mat <= 15 and
   *  if mat = mat_1 + mat_2 * 2 + mat_3 * 4 + mat_4 * 8
   *
   *  Gamma(mat) = gamma(1)^(mat_1) * gamma(2)^(mat_2) * gamma(3)^(mat_3)
   *             * gamma(4)^(mat_4)
   *
   *  Further, in basis for the Gamma matrices used, A is of the form
   *
   *      | A_0 |  0  |
   *  A = | --------- |
   *      |  0  | A_1 |
   *
   *
   * Arguments:
   *
   *  \param B         the resulting SU(N) color matrix	  (Write)
   *  \param clov      clover term                        (Read)
   *  \param mat       label of the Gamma matrix          (Read)
   */

  template <typename T, typename U, int N_exp>
  void QDPExpCloverTermT<T, U, N_exp>::triacntr(U& B, int mat, int cb) const
  {
#ifndef QDP_IS_QDPJIT
    QDPIO::cerr << "JITExpCloverTerm::triacntr: not used for exp-clover" << std::endl;
    QDP_abort(1);
#endif
  }

  namespace QDPExpCloverEnv
  {
    template <typename R>
    struct QUDAPackArgs {
      //typedef OScalar<PScalar<PScalar<RScalar<REALT>>>> RealT;
      int cb;
      multi1d<QUDAPackedClovSite<R>>& quda_array;
      const ExpClovTriang<R>* tri;
    };

    template <typename R>
    void qudaPackSiteLoop(int lo, int hi, int myId, QUDAPackArgs<R>* a)
    {
#if 1
      int cb = a->cb;
      int Ns2 = Ns / 2;

      multi1d<QUDAPackedClovSite<R>>& quda_array = a->quda_array;
      const ExpClovTriang<R>* tri = a->tri;
      for (int ssite = lo; ssite < hi; ++ssite) {
      
	      int site = rb[cb].siteTable()[ssite];
	      // First Chiral Block
	      for (int i = 0; i < 6; i++) {
      	  quda_array[site].diag1[i] = tri[site].A.diag[0][i].elem(); 
	      }

      	int target_index = 0;

	      for (int col = 0; col < Nc * Ns2 - 1; col++) {
	        for (int row = col + 1; row < Nc * Ns2; row++) {

    	      int source_index = row * (row - 1) / 2 + col;

	          quda_array[site].offDiag1[target_index][0] = tri[site].A.offd[0][source_index].real();
	          quda_array[site].offDiag1[target_index][1] = tri[site].A.offd[0][source_index].imag();
	          target_index++;
	        }
	      }


	      // Second Chiral Block
	      for (int i = 0; i < 6; i++) {
	        quda_array[site].diag2[i] = tri[site].A.diag[1][i].elem();
	      }

      	target_index = 0;
	      for (int col = 0; col < Nc * Ns2 - 1; col++) {
	        for (int row = col + 1; row < Nc * Ns2; row++) {

      	    int source_index = row * (row - 1) / 2 + col;
	          quda_array[site].offDiag2[target_index][0] = tri[site].A.offd[1][source_index].real();
	          quda_array[site].offDiag2[target_index][1] = tri[site].A.offd[1][source_index].imag();
	          target_index++;
	        }
	      }
      }
#endif
    }
  } // namespace QDPExpCloverEnv

  template <typename T, typename U, int N_exp>
  void QDPExpCloverTermT<T, U, N_exp>::packForQUDA(
    multi1d<QUDAPackedClovSite<typename WordType<T>::Type_t>>& quda_array, int cb) const
  {
    typedef typename WordType<T>::Type_t REALT;
    int num_sites = rb[cb].siteTable().size();

    QDPExpCloverEnv::QUDAPackArgs<REALT> args = {cb, quda_array, exp_tri};

    dispatch_to_threads(num_sites, args, QDPExpCloverEnv::qudaPackSiteLoop<REALT>);
  }

  template <int N = N_exp_default>
  using QDPExpCloverTerm = QDPExpCloverTermT<LatticeFermion, LatticeColorMatrix, N>;

  template <int N = N_exp_default>
  using QDPExpCloverTermF = QDPExpCloverTermT<LatticeFermionF, LatticeColorMatrixF, N>;

  template <int N = N_exp_default>
  using QDPExpCloverTermD = QDPExpCloverTermT<LatticeFermionD, LatticeColorMatrixD, N>;
} // End Namespace Chroma

#endif

