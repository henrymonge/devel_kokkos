// -*- C++ -*-
/*! \file
 *  \brief Clover term linear operator
 */

#ifndef __exp_clover_term_jit_w_h__
#define __exp_clover_term_jit_w_h__

//#warning "Using QDP-JIT clover term"

#include "state.h"
#include "actions/ferm/fermacts/clover_fermact_params_w.h"
#include "actions/ferm/linop/exp_clover_term_base_w.h"
#include "actions/ferm/linop/clover_term_jit_w.h"
#include "meas/glue/mesfield.h"

#if ! defined (QDP_IS_QDPJIT2)

namespace QDP
{

  constexpr int N_exp_default = 17;
  template<typename T>
  struct Pq
  {
    typedef T Sub_t;
    enum { ThisSize = 6 };
    T q[6];
  };

  template<class T> struct PqREG;

  template<typename T>
  struct PqJIT: public BaseJIT<T,6>
  {
    template<class T1>
    PqJIT& operator=( const PqREG<T1>& rhs) {
      //std::cout << __PRETTY_FUNCTION__ << "\n";
      for ( int i = 0 ; i < 6 ; i++ )
	elem(i) = rhs.elem(i);
      return *this;
    }

    inline       T elem(int i)       { return this->arrayF(i); }
  };

  template<class T>
  struct PqREG 
  {
    T F[6];
    void setup( PqJIT< typename JITType<T>::Type_t > rhs ) {
      for (int i=0;i<6;++i)
	F[i].setup( rhs.elem(i) );
    }
    inline       T& elem(int i)       { return F[i]; }
    inline const T& elem(int i) const { return F[i]; }
  };


  template<class T> 
  struct ScalarType<Pq<T> >
  {
    typedef Pq<typename ScalarType<T>::Type_t>  Type_t;
  };
  
  template<class T> 
  struct ScalarType<PqJIT<T> >
  {
    typedef PqJIT<typename ScalarType<T>::Type_t>  Type_t;
  };

  
  template<class T> 
  struct JITType<Pq<T> >
  {
    typedef PqJIT<typename JITType<T>::Type_t>  Type_t;
  };

  template<class T> 
  struct JITType<PqREG<T> >
  {
    typedef PqJIT<typename JITType<T>::Type_t>  Type_t;
  };

  template<class T> 
  struct REGType<PqJIT<T> >
  {
    typedef PqREG<typename REGType<T>::Type_t>  Type_t;
  };

  template<class T>
  struct WordType<Pq<T> > 
  {
    typedef typename WordType<T>::Type_t  Type_t;
  };

  template<class T>
  struct WordType<PqJIT<T> > 
  {
    typedef typename WordType<T>::Type_t  Type_t;
  };


  template<class T>
  struct LeafFunctor<Pq<T>, PrintTag>
  {
    typedef int Type_t;
    static int apply(const PrintTag &f)
    {
      f.os_m << "Pq<";
      LeafFunctor<T,PrintTag>::apply(f);
      f.os_m << ">";
      return 0;
    }
  };

/**************************************************/


    /*! This accessor class allows me a convenient way to acces the
        diagonal + lower diagonal storage for the hermitian matrix */
    template <typename T, typename X, typename Y, int block>
    struct ClovAccessor {
       typedef typename LeafFunctor<X, ParamLeafScalar>::Type_t  XJIT;
       typedef typename LeafFunctor<Y, ParamLeafScalar>::Type_t  YJIT;
       typedef typename WordType<T>::Type_t REALT;
      ClovAccessor(typename REGType< typename XJIT::Subtype_t >::Type_t tri_dia_in,
                   typename REGType< typename YJIT::Subtype_t >::Type_t tri_off_in) : tri_dia_r(tri_dia_in),tri_off_r(tri_off_in)
      {
      }

      inline RComplexREG<WordREG<REALT> > operator()(int row, int col) const
      {

    RComplexREG<WordREG<REALT> > ret_val;    

	if (row == col)
	{
	  // Diagonal Piece:
	  ret_val = tri_dia_r.elem(block).elem(row);
	}
	else if (row > col)
	{
	  // Lower triangular portion
	  ret_val = tri_off_r.elem(block).elem((row * (row - 1)) / 2 + col);
	}
	else if (row < col)
	{
	  // Upper triangular portion: transpose ( row <-> col) and conjugate
	  ret_val = conj(tri_off_r.elem(block).elem((col * (col - 1)) / 2 + row));
	}

	return ret_val;
      }

      //inline void insert(int row, int col, const RComplex<T>& value)
      inline void insert(int row, int col, const RComplexREG<WordREG<REALT> >& value)
      {
	if (row == col)
	{
	  // Diagonal piece -- must be real.
      tri_dia_r.elem(block).elem(row) = RScalarREG<WordREG<REALT> >(real(value));
	}
	else if (row > col)
	{
	  // Lower triangular portion
	  tri_off_r.elem(block).elem((row * (row - 1)) / 2 + col) = value;
	}
	else if (row < col)
	{
	  // Upper triangular portion: transpose ( row <-> col) and conjugate
	  tri_off_r.elem(block).elem((col * (col - 1)) / 2 + row) = conj(value);
	}
      }

    private:
       typename REGType< typename XJIT::Subtype_t >::Type_t  tri_dia_r;
       typename REGType< typename YJIT::Subtype_t >::Type_t  tri_off_r;

    };

/**************************************************/

    template <typename T, typename X, typename Y, int block>
    struct Traces {
       typedef typename LeafFunctor<X, ParamLeafScalar>::Type_t  XJIT;
       typedef typename LeafFunctor<Y, ParamLeafScalar>::Type_t  YJIT;
       typedef typename WordType<T>::Type_t REALT;

      Traces(typename REGType< typename XJIT::Subtype_t >::Type_t tri_dia_in,
                   typename REGType< typename YJIT::Subtype_t >::Type_t tri_off_in) : tri_dia_r(tri_dia_in),tri_off_r(tri_off_in)

      {
      }

      // Simple mat mult routine
      inline void multiply(ClovAccessor<T, X, Y, block>& out, const ClovAccessor<T, X, Y, block>& M1,
			   const ClovAccessor<T, X, Y, block>& M2)
      {   

  	// NB: We only need to compute the diagonal and lower diagonal
	// elements because the matrices are hermitiean.
	for (int row = 0; row < 6; ++row)
	{
	  for (int col = 0; col <= row; ++col)
	  {
	    // Pour row down column
	    RComplexREG<WordREG<REALT> > dotprod; // = zip;
        dotprod.real() = 0;
        dotprod.imag() =0;
	    for (int k = 0; k < 6; ++k)
	    {
            dotprod += M1(row, k) * M2(k, col);
	    }
	    out.insert(row, col, dotprod);
	  }
	}
    
      }

      // Simple mat mult routine
      inline void copy(ClovAccessor<T, X, Y, block>& out, const ClovAccessor<T, X, Y, block>& in)
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

      inline void traces(Pq<RComplexREG<WordREG<REALT>>>& tr)
      {

	// The first 5 will map onto the hither powers.
    ClovAccessor<REALT,X,Y,block> A(tri_dia_r,tri_off_r);

    zero_rep(tr.q[0]);
    
	for (int i = 0; i < 6; i++)
	{
	  tr.q[0] += A(i, i);
	}
    
    typename REGType< typename XJIT::Subtype_t >::Type_t tri_dia_curr;
    typename REGType< typename YJIT::Subtype_t >::Type_t tri_off_curr;
    typename REGType< typename XJIT::Subtype_t >::Type_t tri_dia_prev;
    typename REGType< typename YJIT::Subtype_t >::Type_t tri_off_prev;

    ClovAccessor<REALT,X,Y,block> Prev(tri_dia_prev,tri_off_prev);
    ClovAccessor<REALT,X,Y,block> Curr(tri_dia_curr,tri_off_curr);

	copy(Prev, A);

	for (int pow = 1; pow <= 5; pow++)
	{
	  multiply(Curr, Prev, A);
	 
      zero_rep(tr.q[pow]);
	  for (int i = 0; i < 6; i++)
	  {
	    tr.q[pow] += real(Curr(i, i));
	  }
	  copy(Prev, Curr);
	} 
      
      }

    private:
       typename REGType< typename XJIT::Subtype_t >::Type_t  tri_dia_r;
       typename REGType< typename YJIT::Subtype_t >::Type_t  tri_off_r;
    };

} // QDP




#if defined (QDP_BACKEND_AVX)
#define WORD WordVec
#else
#define WORD Word
#endif


namespace Chroma 
{ 

#if 0
  template<typename R>
  struct QUDAPackedClovSite {
    R diag1[6];
    R offDiag1[15][2];
    R diag2[6];
    R offDiag2[15][2];
  };
#endif


  template<typename T, typename U,int N_exp = N_exp_default>
  class JITExpCloverTermT : public ExpCloverTermBase<T, U>
  {
  public:
    // Typedefs to save typing
    typedef typename WordType<T>::Type_t REALT;

    typedef OLattice< PScalar< PScalar< RScalar< WORD< REALT> > > > > LatticeREAL;
    typedef OScalar<  PScalar< PScalar< RScalar< Word< REALT> > > > > RealT;

    //! Empty constructor. Must use create later
    JITExpCloverTermT();

    //! No real need for cleanup here
    ~JITExpCloverTermT() {}

    //! Creation routine
    void create(Handle< FermState<T, multi1d<U>, multi1d<U> > > fs,
		const CloverFermActParams& param_);


    virtual void create(Handle< FermState<T, multi1d<U>, multi1d<U> > > fs,
            const CloverFermActParams& param_,
            const JITExpCloverTermT<T,U>& from_, int inv_op);

    virtual void create(Handle< FermState<T, multi1d<U>, multi1d<U> > > fs,
			const CloverFermActParams& param_,
			const JITExpCloverTermT<T,U>& from_);

    virtual void createInv(Handle< FermState<T, multi1d<U>, multi1d<U> > > fs,
            const CloverFermActParams& param_,
            const JITExpCloverTermT<T,U>& from_);

    //! Computes the inverse of the term on cb using Cholesky
    /*!
     * \param cb   checkerboard of work (Read)
     */
    void choles(int cb){
      QDPIO::cerr << "JITExpCloverTerm: unimplemented function for exponentiated clover" << std::endl;
    };

    //! Computes the inverse of the term on cb using Cholesky
    /*!
     * \param cb   checkerboard of work (Read)
     * \return logarithm of the determinant  
     */
    Double cholesDet(int cb) const ;

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
    void apply(T& chi, const T& psi, enum PlusMinus isign, int cb) const;

                
    //apply Coefficients to fermion 
    void applyCoeff(T& chi, const T& psi, enum PlusMinus isign,int cb, int pow_i, int pow_j) const;



    // Apply a power of a matrix from A^0 to A^5
    void applyPower(T& chi, const T& psi, enum PlusMinus isign, int cb, int power = 1) const;

    // Apply exponential operator
    void applyInv(T& chi, const T& psi, enum PlusMinus isign, int cb) const;

    inline void applyUnexp(T& chi, const T& psi, enum PlusMinus isign, int cb) const
    {
      applyPower(chi, psi, isign, cb, 1); // Explicily apply just the clover term.
    } 

    void applySite(T& chi, const T& psi, enum PlusMinus isign, int site) const;

    void applyExpClov(T& chi, const T& psi, enum PlusMinus isign, int cb) const ;

    void makeExpClov(enum PlusMinus isign, int cb, int inverse);
 
    void deriv(multi1d<U>& ds_u,
           const T& chi, const T& psi,
           enum PlusMinus isign) const;//{ExpCloverTermBase<T, U>::deriv(ds_u,chi,psi,isign);}

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

    //! Calculates Tr_D ( Gamma_mat L )
    void triacntr(U& B, int mat, int cb) const;

    //! Return the fermion BC object for this linear operator
    const FermBC<T, multi1d<U>, multi1d<U> >& getFermBC() const {return *fbc;}

    //! PACK UP the Clover term for QUDA library:
    void packForQUDA(multi1d<QUDAPackedClovSite<REALT> >& quda_pack, int cb) const; 

    int getDiaId() const { return tri_dia.getId(); }
    int getOffId() const { return tri_off.getId(); }

  protected:
    //! Create the clover term on cb
    /*!
     *  \param f         field strength tensor F(mu,nu)        (Read)
     *  \param cb        checkerboard                          (Read)
     */
    void makeClov(const multi1d<U>& f, const RealT& diag_mass);

    //! Get the u field
    const multi1d<U>& getU() const {return u;}

    //! Calculates Tr_D ( Gamma_mat L )
    Real getCloverCoeff(int mu, int nu) const;


  private:
    Handle< FermBC<T,multi1d<U>,multi1d<U> > >      fbc;
    multi1d<U>  u;
    CloverFermActParams          param;
    LatticeREAL                  tr_log_diag_; // Fill this out during create
    LatticeDouble tr_M; // Fill this out during create

    // but save the global sum until needed.

    OLattice<PComp<PTriDia<RScalar <WORD<REALT> > > > >  tri_dia;
    OLattice<PComp<PTriOff<RComplex<WORD<REALT> > > > >  tri_off;    
    OLattice<PComp<PTriDia<RScalar <WORD<REALT> > > > >  exp_tri_dia;
    OLattice<PComp<PTriOff<RComplex<WORD<REALT> > > > >  exp_tri_off;
    OLattice<PComp<Pq<RScalar <WORD<REALT> > > > >  qc;
    OLattice<PComp<Pq<RScalar <WORD<REALT> > > > >  qc_inv; 
    OLattice<PComp<Pq<Pq<RScalar<WORD<REALT>>>> > > C;
    multi3d<LatticeDouble> C_arr; // Fill this out during create;
    OScalar<  PScalar< PScalar< RScalar< Word< REALT> > > > > diag_mass;

  };

#undef WORD

//Starts tracePowers()

#if defined (QDP_BACKEND_AVX)
#define WORD WordVec
#else
#define WORD Word
#endif

  template<typename T, typename X,typename Y, typename Q, typename Z, typename W>
  void function_tracePowers_exec(JitFunction& function,
                RScalar<T> &dummy,
                const X& tri_dia,
                const Y& tri_off,
                Q& qc,
                Z& qc_inv,
                W& Cv)
  {

    AddressLeaf addr_leaf(all);

    forEach(tri_dia, addr_leaf, NullCombine());
    forEach(tri_off, addr_leaf, NullCombine());


    forEach(qc, addr_leaf, NullCombine());
    forEach(qc_inv, addr_leaf, NullCombine());
    forEach(Cv, addr_leaf, NullCombine());

    int th_count = Layout::sitesOnNode();
    WorkgroupGuardExec workgroupGuardExec(th_count);

    std::vector<QDPCache::ArgKey> ids;
    workgroupGuardExec.check(ids);
    ids.push_back( all.getIdSiteTable() );
    for(unsigned i=0; i < addr_leaf.ids.size(); ++i)
      ids.push_back( addr_leaf.ids[i] );
    jit_launch(function,th_count,ids);
   
  }


  template<typename T, typename X,typename Y, typename Q, typename Z, typename W>
  void function_tracePowers_build(JitFunction& function,
                  RScalar<T> &dummy,
                  const X& tri_dia,
                  const Y& tri_off,
                  Q& qc,
                  Z& qc_inv,
                  W& Cv)
  {
    llvm_start_new_function("tracePowers",__PRETTY_FUNCTION__);


     
    WorkgroupGuard workgroupGuard;
    ParamRef p_site_table = llvm_add_param<int*>();

    ParamLeafScalar param_leaf;
    typedef typename WordType<T>::Type_t REALT;
    typedef typename LeafFunctor<T, ParamLeafScalar>::Type_t  TJIT;
    typedef typename LeafFunctor<X, ParamLeafScalar>::Type_t  XJIT;
    typedef typename LeafFunctor<Y, ParamLeafScalar>::Type_t  YJIT;
    typedef typename LeafFunctor<Q, ParamLeafScalar>::Type_t  QJIT;
    typedef typename LeafFunctor<Z, ParamLeafScalar>::Type_t  ZJIT;
    typedef typename LeafFunctor<W, ParamLeafScalar>::Type_t  WJIT;

    XJIT tri_dia_jit(forEach(tri_dia, param_leaf, TreeCombine()));
    typename REGType< typename XJIT::Subtype_t >::Type_t tri_dia_r;

    YJIT tri_off_jit(forEach(tri_off, param_leaf, TreeCombine()));
    typename REGType< typename YJIT::Subtype_t >::Type_t tri_off_r;

    QJIT qc_jit(forEach(qc, param_leaf, TreeCombine()));
    ZJIT qc_inv_jit(forEach(qc_inv, param_leaf, TreeCombine()));
    WJIT Cv_jit(forEach(Cv, param_leaf, TreeCombine()));


    llvm::Value* r_idx_thread = llvm_thread_idx();

    workgroupGuard.check(r_idx_thread);

    llvm::Value* r_idx = llvm_array_type_indirection<int>( p_site_table , r_idx_thread );

    tri_dia_r.setup( tri_dia_jit.elem(JitDeviceLayout::Coalesced,r_idx) );
    tri_off_r.setup( tri_off_jit.elem(JitDeviceLayout::Coalesced,r_idx) );

    auto Cv_j = Cv_jit.elem(JitDeviceLayout::Coalesced,r_idx);
    auto qc_j = qc_jit.elem(JitDeviceLayout::Coalesced,r_idx);
    auto qc_inv_j = qc_inv_jit.elem(JitDeviceLayout::Coalesced,r_idx);

    int N_exp =17;
    // Compute the q-s for the block
    RScalarREG<WordREG<REALT> > tab[2][N_exp + 1][6];
    RScalarREG<WordREG<REALT> > itab[2][N_exp + 1][6];
 
    for (int a = 0; a < 2; ++a)
      for (int i = 0; i < N_exp + 1; ++i)
        for (int j = 0; j < 6; ++j){
          tab[a][i][j] = (RScalarREG<WordREG<REALT> >)(0);
          itab[a][i][j] = (RScalarREG<WordREG<REALT> >)(0);
        }


    for (int block = 0; block < 2; ++block)
    {
      int upper = N_exp + 1 < 6 ? N_exp + 1 : 6;
      //enum { upper = (N_exp + 1 < 6 ? N_exp + 1 : 6) };
      RScalarREG<WordREG<REALT> > ifact(1);
      for (int i = 0; i < upper; ++i)
      {
        tab[block][i][i] = (RScalarREG<WordREG<REALT> >)1;
        itab[block][i][i] = ifact;
        ifact = -ifact;
      }
    }

    Traces<REALT,X, Y ,0> tr0(tri_dia_r,tri_off_r);
    Traces<REALT,X, Y ,1> tr1(tri_dia_r,tri_off_r);

    Pq<RComplexREG<WordREG<REALT>>> trace0;
    Pq<RComplexREG<WordREG<REALT>>> trace1;

    tr0.traces(trace0);
    tr1.traces(trace1);


	if (N_exp + 1 > 6)
	{
      RScalarREG<WordREG<REALT> > trace[2][6];

	  for (int i = 0; i < 6; ++i)
	    trace[0][i] = real(trace0.q[i]);
	  for (int i = 0; i < 6; ++i)
	    trace[1][i] = real(trace1.q[i]);

	  for (int block = 0; block < 2; ++block)
	  {
	    RScalarREG<WordREG<REALT> > p[5];

	    p[4] = RScalarREG<WordREG<REALT> >(1.0/2.0) * trace[block][1]; // (1/2) Tr A^2

	    p[3] = (RScalarREG<WordREG<REALT> >)(1.0/3.0) * trace[block][2]; // (1/3) Tr A^3
	    p[2] = (RScalarREG<WordREG<REALT> >)(1.0/4.0) * trace[block][3] -
		       (RScalarREG<WordREG<REALT> >)(1.0/8.0) * trace[block][1] *
		        trace[block][1]; // (1/4) Tr A^4 - (1/8) (Tr A^2)^2

	    p[1] = (RScalarREG<WordREG<REALT> >)(1.0 / 5.0) * trace[block][4] -
		       (RScalarREG<WordREG<REALT> >) (1.0 / 6.0) * trace[block][2] *
		        trace[block][1]; // (1/5) Tr A^5 - (1/6) Tr A^3 Tr A^2

	    p[0] = (RScalarREG<WordREG<REALT> >)( 1.0 / 6.0) * trace[block][5] // (1/6) Tr A^6 - (1/8) Tr A^4 Tr A^2
		   - (RScalarREG<WordREG<REALT> >)(1.0 / 8.0) * trace[block][3] *
		       trace[block][1] //     - (1/18) [ Tr A^3 ]^2
		   - (RScalarREG<WordREG<REALT> >)(1.0 /18.0) * trace[block][2] *
		       trace[block][2] //     + (1/48) [ Tr A^2 ]^3
		   + (RScalarREG<WordREG<REALT> >)(1.0 / 48.0) * trace[block][1] * trace[block][1] * trace[block][1];


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
        qc_j.elem(block).elem(i) =tab[block][0][i];
        qc_inv_j.elem(block).elem(i) = tab[block][0][i];
	  }

	  unsigned long fact = 1;
	  for (unsigned int row = 1; row <= N_exp; ++row)
	  {
	    fact *= (unsigned long)row;
	    RScalarREG<WordREG<REALT> > sign((row % 2 == 0) ? (RScalarREG<WordREG<REALT> >)1 : (RScalarREG<WordREG<REALT> >)(-1));
	    for (int i = 0; i < 6; i++)
	    {
	      qc_j.elem(block).elem(i) += (tab[block][row][i] / (RScalarREG<WordREG<REALT> >)(fact));
	      qc_inv_j.elem(block).elem(i) += (sign * tab[block][row][i] / (RScalarREG<WordREG<REALT> >)(fact));
	    }
	  }

      //HMC: adding the calculation of the C_ij
      for (int i = 0; i < 6; i++)
      {
        for (int j = 0; j < 6; j++)
        {
            Cv_j.elem(block).elem(i).elem(j) = tab[block][0][i]*tab[block][0][j];
        }
      }

      fact = 1;
      unsigned long fact_row = 1;

      for (unsigned int row = 0; row <= N_exp; ++row)
      { 
        if (row!=0)
           fact_row *= (unsigned long)(row);
        fact=fact_row*(unsigned long)(row+1);
        for(unsigned int col = 0; col <= N_exp-row; ++col)
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
                      Cv_j.elem(block).elem(i).elem(j) += tab[block][col][j]*tab[block][row][i] / (RScalarREG<WordREG<REALT> >)(fact);    
                  }
                }
            }
        }
      }

	}//for block ends

 
    jit_get_function(function);

  }

#undef WORD


  // Empty constructor. Must use create later
  template<typename T, typename U,int N_exp>
  JITExpCloverTermT<T,U,N_exp>::JITExpCloverTermT() {}

  // Now copy
  template<typename T, typename U,int N_exp>
  void JITExpCloverTermT<T,U,N_exp>::create(Handle< FermState<T,multi1d<U>,multi1d<U> > > fs,
				   const CloverFermActParams& param_,
				   const JITExpCloverTermT<T,U>& from, int inv_op)
  {
    START_CODE();

    u.resize(Nd);

    u = fs->getLinks();
    fbc = fs->getFermBC();
    param = param_;
    
    // Sanity check
    if (fbc.operator->() == 0) {
      QDPIO::cerr << "JITExpCloverTerm: error: fbc is null" << std::endl;
      QDP_abort(1);
    }
   
    //
    // Yuk. Some bits of knowledge of the dslash term are buried in the 
    // effective mass term. They show up here. If I wanted some more 
    // complicated dslash then this will have to be fixed/adjusted.
    //
    
    RealT ff = param.anisoParam.anisoP ? param.anisoParam.nu / param.anisoParam.xi_0 : Real(1);
    diag_mass = 1 + (Nd-1)*ff + param.Mass;
    
     
    {
      RealT ff = param.anisoParam.anisoP ? Real(1) / param.anisoParam.xi_0 : Real(1);
      param.clovCoeffR *= Real(0.5) * ff / diag_mass;
      param.clovCoeffT *= Real(0.5) / diag_mass;
    }
   
    if (inv_op==1)
        diag_mass = 1.0/diag_mass;
 
    /* Calculate F(mu,nu) */
    //multi1d<LatticeColorMatrix> f;
    //mesField(f, u);
    //makeClov(f, diag_mass);
        
    tr_log_diag_ = from.tr_log_diag_;
   
    tr_M = from.tr_M;

    tri_dia = from.tri_dia;
    tri_off = from.tri_off;
    C=from.C;
    if(inv_op == 1){
        qc = from.qc_inv;
        qc_inv = from.qc;
    }
    else{
        qc = from.qc;
        qc_inv = from.qc_inv;
    }

    END_CODE();  
  }

  template<typename T, typename U,int N_exp>
  void JITExpCloverTermT<T,U,N_exp>::create(Handle< FermState<T,multi1d<U>,multi1d<U> > > fs,
                   const CloverFermActParams& param_,
                   const JITExpCloverTermT<T,U>& from)
  {
     create(fs,param_,from,0);
  }


  template<typename T, typename U,int N_exp>
  void JITExpCloverTermT<T,U,N_exp>::createInv(Handle< FermState<T,multi1d<U>,multi1d<U> > > fs,
                   const CloverFermActParams& param_,
                   const JITExpCloverTermT<T,U>& from)
  {
     create(fs,param_,from,1);
  }


  //! Creation routine
  template<typename T, typename U,int N_exp>
  void JITExpCloverTermT<T,U,N_exp>::create(Handle< FermState<T,multi1d<U>,multi1d<U> > > fs,
				   const CloverFermActParams& param_)
  {
    START_CODE();

    QDPIO::cout << "Creating JITExpCloverTerm" << std::endl; 
    u.resize(Nd);
    
    u = fs->getLinks();
    fbc = fs->getFermBC();
    param = param_;
    
    // Sanity check
    if (fbc.operator->() == 0) {
      QDPIO::cerr << "JITExpCloverTerm: error: fbc is null" << std::endl;
      QDP_abort(1);
    }
   
    //
    // Yuk. Some bits of knowledge of the dslash term are buried in the 
    // effective mass term. They show up here. If I wanted some more 
    // complicated dslash then this will have to be fixed/adjusted.
    //
    RealT ff = param.anisoParam.anisoP ? param.anisoParam.nu / param.anisoParam.xi_0 : Real(1);
    diag_mass = 1 + (Nd-1)*ff + param.Mass;

    {
      RealT ff = param.anisoParam.anisoP ? Real(1) / param.anisoParam.xi_0 : Real(1);
      param.clovCoeffR *= RealT(0.5) * ff / diag_mass;
      param.clovCoeffT *= RealT(0.5) / diag_mass;
    }


    /* Calculate F(mu,nu) */
    multi1d<U> f;
    mesField(f, u);
    makeClov(f, diag_mass);

    static JitFunction function;
    RScalar<T> dummy; 
    if (function.empty()){
      function_tracePowers_build( function, dummy, tri_dia, tri_off, qc,qc_inv, C);
    }

    // Execute the function
    function_tracePowers_exec(function, dummy, tri_dia, tri_off, qc,qc_inv, C);

    C_arr.resize(2,6,6); 
#  pragma omp parallel for
    for (int site = 0; site < Layout::sitesOnNode(); ++site)
    {
      tr_M.elem(site).elem().elem().elem() = 0;

      for (int block = 0; block < 2; ++block)
      {
    for (int d = 0; d < 6; ++d)

    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 6; ++j)
        {
            C_arr[block][i][j].elem(site).elem().elem() = C.elem(site).comp[block].q[i].q[j].elem().elem(); 
        }
    }

      }
    }


    END_CODE();
  }


  /*
   * MAKCLOV 
   *
   *  In this routine, MAKCLOV calculates

   *    1 - (1/4)*sigma(mu,nu) F(mu,nu)

   *  using F from mesfield

   *    F(mu,nu) =  (1/4) sum_p (1/2) [ U_p(x) - U^dag_p(x) ]

   *  using basis of SPPROD and stores in a lower triangular matrix
   *  (no diagonal) plus real diagonal

   *  where
   *    U_1 = u(x,mu)*u(x+mu,nu)*u_dag(x+nu,mu)*u_dag(x,nu)
   *    U_2 = u(x,nu)*u_dag(x-mu+nu,mu)*u_dag(x-mu,nu)*u(x-mu,mu)
   *    U_3 = u_dag(x-mu,mu)*u_dag(x-mu-nu,nu)*u(x-mu-nu,mu)*u(x-nu,nu)
   *    U_4 = u_dag(x-nu,nu)*u(x-nu,mu)*u(x-nu+mu,nu)*u_dag(x,mu)

   *  and

   *         | sigF(1)   sigF(3)     0         0     |
   *  sigF = | sigF(5)  -sigF(1)     0         0     |
   *         |   0         0      -sigF(0)  -sigF(2) |
   *         |   0         0      -sigF(4)   sigF(0) |
   *  where
   *    sigF(i) is a color matrix

   *  sigF(0) = i*(ClovT*E_z + ClovR*B_z)
   *          = i*(ClovT*F(3,2) + ClovR*F(1,0))
   *  sigF(1) = i*(ClovT*E_z - ClovR*B_z)
   *          = i*(ClovT*F(3,2) - ClovR*F(1,0))
   *  sigF(2) = i*(E_+ + B_+)
   *  sigF(3) = i*(E_+ - B_+)
   *  sigF(4) = i*(E_- + B_-)
   *  sigF(5) = i*(E_- - B_-)
   *  i*E_+ = (i*ClovT*E_x - ClovT*E_y)
   *        = (i*ClovT*F(3,0) - ClovT*F(3,1))
   *  i*E_- = (i*ClovT*E_x + ClovT*E_y)
   *        = (i*ClovT*F(3,0) + ClovT*F(3,1))
   *  i*B_+ = (i*ClovR*B_x - ClovR*B_y)
   *        = (i*ClovR*F(2,1) + ClovR*F(2,0))
   *  i*B_- = (i*ClovR*B_x + ClovR*B_y)
   *        = (i*ClovR*F(2,1) - ClovR*F(2,0))

   *  NOTE: I am using  i*F  of the usual F defined by UKQCD, Heatlie et.al.

   *  NOTE: the above definitions assume that the time direction, t_dir,
   *        is 3. In general F(k,j) is multiplied with ClovT if either
   *        k=t_dir or j=t_dir, and with ClovR otherwise.

   *+++
   *  Here are some notes on the origin of this routine. NOTE, ClovCoeff or u0
   *  are not actually used in MAKCLOV.
   *
   *  The clover mass term is suppose to act on a std::vector like
   *
   *  chi = (1 - (ClovCoeff/u0^3) * kappa/4 * sum_mu sum_nu F(mu,nu)*sigma(mu,nu)) * psi

   *  Definitions used here (NOTE: no "i")
   *   sigma(mu,nu) = gamma(mu)*gamma(nu) - gamma(nu)*gamma(mu)
   *                = 2*gamma(mu)*gamma(nu)   for mu != nu
   *       
   *   chi = sum_mu sum_nu F(mu,nu)*gamma(mu)*gamma(nu)*psi   for mu < nu
   *       = (1/2) * sum_mu sum_nu F(mu,nu)*gamma(mu)*gamma(nu)*psi   for mu != nu
   *       = (1/4) * sum_mu sum_nu F(mu,nu)*sigma(mu,nu)*psi
   *
   *
   * chi = (1 - (ClovCoeff/u0^3) * kappa/4 * sum_mu sum_nu F(mu,nu)*sigma(mu,nu)) * psi
   *     = psi - (ClovCoeff/u0^3) * kappa * chi
   *     == psi - kappa * chi
   *
   *  We have absorbed ClovCoeff/u0^3 into kappa. A u0 was previously absorbed into kappa
   *  for compatibility to ancient conventions. 
   *---

   * Arguments:
   *  \param f         field strength tensor F(cb,mu,nu)        (Read)
   *  \param diag_mass effective mass term                      (Read)
   */

  template<typename RealT,typename U,typename X,typename Y>
  void function_make_exp_clov_exec(JitFunction& function, 
			       const RealT& diag_mass, 
			       const U& f0,
			       const U& f1,
			       const U& f2,
			       const U& f3,
			       const U& f4,
			       const U& f5,
			       X& tri_dia,
			       Y& tri_off)
  {
#ifdef QDP_DEEP_LOG
    function.type_W = typeid(REAL).name();
    //function.set_dest_id( tri_dia.getId() );
    function.set_dest_id( tri_off.getId() );
    function.set_is_lat(true);
#endif
    
    AddressLeaf addr_leaf(all);

    forEach(diag_mass, addr_leaf, NullCombine());
    forEach(f0, addr_leaf, NullCombine());
    forEach(f1, addr_leaf, NullCombine());
    forEach(f2, addr_leaf, NullCombine());
    forEach(f3, addr_leaf, NullCombine());
    forEach(f4, addr_leaf, NullCombine());
    forEach(f5, addr_leaf, NullCombine());
    forEach(tri_dia, addr_leaf, NullCombine());
    forEach(tri_off, addr_leaf, NullCombine());

    int th_count = Layout::sitesOnNode();

    WorkgroupGuardExec workgroupGuardExec(th_count);

    std::vector<QDPCache::ArgKey> ids;
    workgroupGuardExec.check(ids);
    ids.push_back( all.getIdSiteTable() );
    for(unsigned i=0; i < addr_leaf.ids.size(); ++i) 
      ids.push_back( addr_leaf.ids[i] );
    jit_launch(function,th_count,ids);
  }



  template<typename RealT,typename U,typename X,typename Y>
  void function_make_exp_clov_build(JitFunction& function,
				const RealT& diag_mass, 
				const U& f0,
				const U& f1,
				const U& f2,
				const U& f3,
				const U& f4,
				const U& f5,
				const X& tri_dia,
				const Y& tri_off)
  {
    //std::cout << __PRETTY_FUNCTION__ << ": entering\n";

    typedef typename WordType<RealT>::Type_t REALT;

    llvm_start_new_function("make_clov",__PRETTY_FUNCTION__ );

    WorkgroupGuard workgroupGuard;
    ParamRef p_site_table = llvm_add_param<int*>();

    ParamLeafScalar param_leaf;
    
    typedef typename LeafFunctor<RealT, ParamLeafScalar>::Type_t  RealTJIT;
    RealTJIT diag_mass_jit(forEach(diag_mass, param_leaf, TreeCombine()));

    typedef typename LeafFunctor<U, ParamLeafScalar>::Type_t  UJIT;
    UJIT f0_jit(forEach(f0, param_leaf, TreeCombine()));
    UJIT f1_jit(forEach(f1, param_leaf, TreeCombine()));
    UJIT f2_jit(forEach(f2, param_leaf, TreeCombine()));
    UJIT f3_jit(forEach(f3, param_leaf, TreeCombine()));
    UJIT f4_jit(forEach(f4, param_leaf, TreeCombine()));
    UJIT f5_jit(forEach(f5, param_leaf, TreeCombine()));

    typedef typename LeafFunctor<X, ParamLeafScalar>::Type_t  XJIT;
    XJIT tri_dia_jit(forEach(tri_dia, param_leaf, TreeCombine()));

    typedef typename LeafFunctor<Y, ParamLeafScalar>::Type_t  YJIT;
    YJIT tri_off_jit(forEach(tri_off, param_leaf, TreeCombine()));

    llvm::Value* r_idx_thread = llvm_thread_idx();

    workgroupGuard.check(r_idx_thread);
    llvm::Value* r_idx = llvm_array_type_indirection<int>( p_site_table , r_idx_thread );

    auto f0_j = f0_jit.elem(JitDeviceLayout::Coalesced , r_idx );
    auto f1_j = f1_jit.elem(JitDeviceLayout::Coalesced , r_idx );
    auto f2_j = f2_jit.elem(JitDeviceLayout::Coalesced , r_idx );
    auto f3_j = f3_jit.elem(JitDeviceLayout::Coalesced , r_idx );
    auto f4_j = f4_jit.elem(JitDeviceLayout::Coalesced , r_idx );
    auto f5_j = f5_jit.elem(JitDeviceLayout::Coalesced , r_idx );

    auto tri_dia_j = tri_dia_jit.elem(JitDeviceLayout::Coalesced , r_idx );
    auto tri_off_j = tri_off_jit.elem(JitDeviceLayout::Coalesced , r_idx );

    typename REGType< typename RealTJIT::Subtype_t >::Type_t diag_mass_reg;

    diag_mass_reg.setup_value( diag_mass_jit.elem() );


    for(int jj = 0; jj < 2; jj++) {
      for(int ii = 0; ii < 2*Nc; ii++) {
          zero_rep(tri_dia_j.elem(jj).elem(ii));
      }
    }


    RComplexREG<WordREG<REALT> > E_minus;
    RComplexREG<WordREG<REALT> > B_minus;
    RComplexREG<WordREG<REALT> > ctmp_0;
    RComplexREG<WordREG<REALT> > ctmp_1;
    RScalarREG<WordREG<REALT> > rtmp_0;
    RScalarREG<WordREG<REALT> > rtmp_1;


    for(int i = 0; i < Nc; ++i) {
      ctmp_0 = f5_j.elem().elem(i,i);
      ctmp_0 -= f0_j.elem().elem(i,i);
      rtmp_0 = imag(ctmp_0);
      tri_dia_j.elem(0).elem(i) += rtmp_0;
	  
      tri_dia_j.elem(0).elem(i+Nc) -= rtmp_0;
	  
      ctmp_1 = f5_j.elem().elem(i,i);
      ctmp_1 += f0_j.elem().elem(i,i);
      rtmp_1 = imag(ctmp_1);
      tri_dia_j.elem(1).elem(i) -= rtmp_1;
	  
      tri_dia_j.elem(1).elem(i+Nc) += rtmp_1;
    }

    for(int i = 1; i < Nc; ++i) {
      for(int j = 0; j < i; ++j) {
	    
	int elem_ij  = i*(i-1)/2 + j;
	int elem_tmp = (i+Nc)*(i+Nc-1)/2 + j+Nc;
	    
	ctmp_0 = f0_j.elem().elem(i,j);
	ctmp_0 -= f5_j.elem().elem(i,j);
	tri_off_j.elem(0).elem(elem_ij) = timesI(ctmp_0);
	    
	zero_rep( tri_off_j.elem(0).elem(elem_tmp) );
	tri_off_j.elem(0).elem(elem_tmp) -= tri_off_j.elem(0).elem(elem_ij);// * -1.0;
	    
	ctmp_1 = f5_j.elem().elem(i,j);
	ctmp_1 += f0_j.elem().elem(i,j);
	tri_off_j.elem(1).elem(elem_ij) = timesI(ctmp_1);
	    
	zero_rep( tri_off_j.elem(1).elem(elem_tmp) );
	tri_off_j.elem(1).elem(elem_tmp) -= tri_off_j.elem(1).elem(elem_ij);
      }
    }

    for(int i = 0; i < Nc; ++i) {
      for(int j = 0; j < Nc; ++j) {
	    
	int elem_ij  = (i+Nc)*(i+Nc-1)/2 + j;
	    
	//E_minus = timesI(f2_j.elem().elem(i,j));
	E_minus = f2_j.elem().elem(i,j);
	E_minus = timesI( E_minus );

	E_minus += f4_j.elem().elem(i,j);
	    
	//B_minus = timesI(f3_j.elem().elem(i,j));
	B_minus = f3_j.elem().elem(i,j);
	B_minus = timesI( B_minus );

	B_minus -= f1_j.elem().elem(i,j);
	    
	tri_off_j.elem(0).elem(elem_ij) = B_minus - E_minus;
	    
	tri_off_j.elem(1).elem(elem_ij) = E_minus + B_minus;
      }
    }

    //    std::cout << __PRETTY_FUNCTION__ << ": leaving\n";

    jit_get_function(function);
  }

 
  /* This now just sets up and dispatches... */
  template<typename T, typename U, int N_exp>
  void JITExpCloverTermT<T,U,N_exp>::makeClov(const multi1d<U>& f, const RealT& diag_mass)
  {
    START_CODE();
    
    if ( Nd != 4 ){
      QDPIO::cerr << __func__ << ": expecting Nd==4" << std::endl;
      QDP_abort(1);
    }
    
    if ( Ns != 4 ){
      QDPIO::cerr << __func__ << ": expecting Ns==4" << std::endl;
      QDP_abort(1);
    }
    U f0 = f[0] * getCloverCoeff(0,1);
    U f1 = f[1] * getCloverCoeff(0,2);
    U f2 = f[2] * getCloverCoeff(0,3);
    U f3 = f[3] * getCloverCoeff(1,2);
    U f4 = f[4] * getCloverCoeff(1,3);
    U f5 = f[5] * getCloverCoeff(2,3);    

    static JitFunction function;
    T dummy;
    if (function.empty())
      function_make_exp_clov_build(function, diag_mass, f0,f1,f2,f3,f4,f5, tri_dia , tri_off );

    // Execute the function
    function_make_exp_clov_exec(function, diag_mass, f0,f1,f2,f3,f4,f5,tri_dia, tri_off);

    END_CODE();
  }
  
  //! Invert
  /*!
   * Computes the inverse of the term on cb using Cholesky
   *
   * \return logarithm of the determinant  
   */
  template<typename T, typename U,int N_exp>
  Double JITExpCloverTermT<T,U,N_exp>::cholesDet(int cb) const
  {
    START_CODE();

    LatticeREAL ff=tr_log_diag_;


    END_CODE();

    // Need to thread generic sums in QDP++?
    // Need to thread generic norm2() in QDP++?
    return sum(tr_M, rb[cb]);

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
  
 
  template<typename T, typename U,int N_exp>
  void JITExpCloverTermT<T,U,N_exp>::triacntr(U& B, int mat, int cb) const
  {
    START_CODE();

    B = zero;

    if ( mat < 0  ||  mat > 15 )
      {
	QDPIO::cerr << __func__ << ": Gamma out of range: mat = " << mat << std::endl;
	QDP_abort(1);
      }

    static JitFunction function;

    if (function.empty())
      function_triacntr_build<U>( function, B, tri_dia, tri_off, mat, rb[cb] );

    // Execute the function
    function_triacntr_exec(function, B, tri_dia, tri_off, mat, rb[cb] );

    END_CODE();
  }


  //! Returns the appropriate clover coefficient for indices mu and nu
  template<typename T, typename U,int N_exp>
  Real
  JITExpCloverTermT<T,U,N_exp>::getCloverCoeff(int mu, int nu) const 
  { 
    START_CODE();

    if( param.anisoParam.anisoP )  {
      if (mu==param.anisoParam.t_dir || nu == param.anisoParam.t_dir) { 
	return param.clovCoeffT;
      }
      else { 
	// Otherwise return the spatial coeff
	return param.clovCoeffR;
      }
    }
    else { 
      // If there is no anisotropy just return the spatial one, it will
      // be the same as the temporal one
      return param.clovCoeffR; 
    } 
    
    END_CODE();
  }

#if 1

  template<typename RealT, typename X,typename Y, typename Q>
  void function_make_exp_tri_clov_exec(JitFunction& function,
                const RealT& diag_mass,
                const X& exp_tri_dia,
                const Y& exp_tri_off,
				const X& tri_dia,
				const Y& tri_off,
                const Q& qc,
				const Subset& s)
  {
#ifdef QDP_DEEP_LOG
    function.type_W = typeid(REAL).name();
    function.set_dest_id( chi.getId() );
    function.set_is_lat(true);
#endif
   
    AddressLeaf addr_leaf(s);

    forEach(diag_mass, addr_leaf, NullCombine());
    forEach(exp_tri_dia, addr_leaf, NullCombine());
    forEach(exp_tri_off, addr_leaf, NullCombine());
    forEach(tri_dia, addr_leaf, NullCombine());
    forEach(tri_off, addr_leaf, NullCombine());
    forEach(qc, addr_leaf, NullCombine());

    int th_count = s.numSiteTable();
    WorkgroupGuardExec workgroupGuardExec(th_count);

    std::vector<QDPCache::ArgKey> ids;
    workgroupGuardExec.check(ids);
    ids.push_back( s.getIdSiteTable() );
    for(unsigned i=0; i < addr_leaf.ids.size(); ++i) 
      ids.push_back( addr_leaf.ids[i] );
    jit_launch(function,th_count,ids);

  }



   template<typename REALT, typename X,typename Y, int block>
   inline RComplexREG<WordREG<REALT> > A_ij(int row, int col, X& tri_dia_r,
                                            Y& tri_off_r)
    {

    RComplexREG<WordREG<REALT> > ret_val;

    if (row == col)
    {
      ret_val = tri_dia_r.elem(block).elem(row);
    }
    else if (row > col)
    {
      // Lower triangular portion
      ret_val = tri_off_r.elem(block).elem((row * (row - 1)) / 2 + col);
    }
    else if (row < col)
    {
      // Upper triangular portion: transpose ( row <-> col) and conjugate
      ret_val = conj(tri_off_r.elem(block).elem((col * (col - 1)) / 2 + row));
    }

    return ret_val;
      }

   template<typename REALT, typename X,typename Y, int block>
   inline void A_ij_ins(int row, int col,RComplexREG<QDP::WordREG<REALT> > val, X& tri_dia_r,
                                            Y& tri_off_r)
    {

    if (row == col)
    {
       tri_dia_r.elem(block).elem(row)=RScalarREG<WordREG<REALT> >(real(val));
    }
    else if (row > col)
    {
      // Lower triangular portion
      tri_off_r.elem(block).elem((row * (row - 1)) / 2 + col)=val;
    }
    else if (row < col)
    {
      // Upper triangular portion: transpose ( row <-> col) and conjugate
      tri_off_r.elem(block).elem((col * (col - 1)) / 2 + row)=conj(val);
    }

    }

      // Simple mat mult routine
      template<typename REALT, typename X,typename Y, int block>
      inline void multiply(X& tri_dia_r_out,Y& tri_off_r_out, X& tri_dia_r_1,Y& tri_off_r_1,
                           X& tri_dia_r_2,Y& tri_off_r_2)
      {

    // NB: We only need to compute the diagonal and lower diagonal
    // elements because the matrices are hermitiean.
    for (int row = 0; row < 6; ++row)
    {
      for (int col = 0; col <= row; ++col)
      {
        // Pour row down column
        RComplexREG<WordREG<REALT> > dotprod; // = zip;
        dotprod.real() = 0;
        dotprod.imag() =0;
        for (int k = 0; k < 6; ++k)
        {
            dotprod += A_ij<REALT,X,Y,block>(row,k,tri_dia_r_1,tri_off_r_1) * A_ij<REALT,X,Y,block>(k,col,tri_dia_r_2,tri_off_r_2);
        }
        //out.insert(row, col, dotprod);
        A_ij_ins<REALT,X,Y,block>(row, col,dotprod, tri_dia_r_out,tri_off_r_out);
      }
    }

      }


  template<typename RealT, typename X,typename Y, typename Q>
  void function_make_exp_tri_clov_build( JitFunction& function,
                  const RealT& diag_mass,
                  const X& exp_tri_dia,
                  const Y& exp_tri_off,
				  const X& tri_dia,
				  const Y& tri_off,
                  const Q& qc,
				  const Subset& s)
  {
    llvm_start_new_function("apply_exp_clov",__PRETTY_FUNCTION__);

    WorkgroupGuard workgroupGuard;
    ParamRef p_site_table = llvm_add_param<int*>();

    ParamLeafScalar param_leaf;

    typedef typename WordType<RealT>::Type_t REALT;

    typedef typename LeafFunctor<RealT, ParamLeafScalar>::Type_t  RealTJIT;
    RealTJIT diag_mass_jit(forEach(diag_mass, param_leaf, TreeCombine()));

    typedef typename LeafFunctor<X, ParamLeafScalar>::Type_t  XJIT;
    XJIT exp_tri_dia_jit(forEach(exp_tri_dia, param_leaf, TreeCombine()));
    typename REGType< typename XJIT::Subtype_t >::Type_t exp_tri_dia_r;

    typedef typename LeafFunctor<Y, ParamLeafScalar>::Type_t  YJIT;
    YJIT exp_tri_off_jit(forEach(exp_tri_off, param_leaf, TreeCombine()));
    typename REGType< typename YJIT::Subtype_t >::Type_t exp_tri_off_r;

    XJIT tri_dia_jit(forEach(tri_dia, param_leaf, TreeCombine()));
    typename REGType< typename XJIT::Subtype_t >::Type_t tri_dia_r;


    YJIT tri_off_jit(forEach(tri_off, param_leaf, TreeCombine()));
    typename REGType< typename YJIT::Subtype_t >::Type_t tri_off_r;;


    typename REGType< typename XJIT::Subtype_t >::Type_t tmp_tri_dia_r;
    typename REGType< typename YJIT::Subtype_t >::Type_t tmp_tri_off_r;

    typename REGType< typename XJIT::Subtype_t >::Type_t curr_tri_dia_r;
    typename REGType< typename YJIT::Subtype_t >::Type_t curr_tri_off_r;

    typedef typename REGType< typename XJIT::Subtype_t >::Type_t TRIDIA_R;
    typedef typename REGType< typename YJIT::Subtype_t >::Type_t TRIOFF_R;

    typedef typename LeafFunctor<Q, ParamLeafScalar>::Type_t  QJIT;
    QJIT qc_jit(forEach(qc, param_leaf, TreeCombine()));
    typename REGType< typename QJIT::Subtype_t >::Type_t qc_r;

    llvm::Value* r_idx_thread = llvm_thread_idx();

    workgroupGuard.check(r_idx_thread);

    llvm::Value* r_idx = llvm_array_type_indirection<int>( p_site_table , r_idx_thread );


    typename REGType< typename RealTJIT::Subtype_t >::Type_t diag_mass_reg;
    diag_mass_reg.setup_value( diag_mass_jit.elem() );

    auto exp_tri_dia_j = exp_tri_dia_jit.elem(JitDeviceLayout::Coalesced,r_idx);
    auto exp_tri_off_j = exp_tri_off_jit.elem(JitDeviceLayout::Coalesced,r_idx);

    tri_dia_r.setup( tri_dia_jit.elem(JitDeviceLayout::Coalesced,r_idx) );
    tri_off_r.setup( tri_off_jit.elem(JitDeviceLayout::Coalesced,r_idx) );

    qc_r.setup( qc_jit.elem(JitDeviceLayout::Coalesced,r_idx) );

    //Set the highest power of A^n for the exp sum. This allows for N_exp_default < 5 to compare with clover 
    int pow_max=5;
    if (N_exp_default <5)
       pow_max=N_exp_default;
   
    
    RScalarREG<QDP::WordREG<REALT>> qi[2][6];

    for(int block=0; block < 2; block++) 
    {
      for (int i = 0; i < 6; i++)
          qi[block][i] = diag_mass_reg.elem().elem()*qc_r.elem(block).elem(i);

      //Set the output clov triang to 1+A (power 1)
      for (int c=0; c < 2*Nc ;c++){
          exp_tri_dia_r.elem(block).elem(c)=qi[block][0];
          tmp_tri_dia_r.elem(block).elem(c)= tri_dia_r.elem(block).elem(c);

      }
     //Add the off-diagonal entries
      for (int c=0; c < 2*Nc*Nc-Nc; c++){
        tmp_tri_off_r.elem(block).elem(c)=tri_off_r.elem(block).elem(c);
  
      }
    }
   
      for (int pow = 1;pow <= pow_max; pow++)
      {

          for (int block = 0; block < 2; block++)
          {
                //Add the diagonal entries
            for (int c=0; c < 2*Nc ;c++){
                exp_tri_dia_r.elem(block).elem(c)+=qi[block][pow]*tmp_tri_dia_r.elem(block).elem(c);
                curr_tri_dia_r.elem(block).elem(c)= tmp_tri_dia_r.elem(block).elem(c);
            }

            for (int c=0;c <2*Nc*Nc-Nc;c++){
                if(pow==1){
                    exp_tri_off_r.elem(block).elem(c)=qi[block][pow]*tmp_tri_off_r.elem(block).elem(c);
                }else{
                    exp_tri_off_r.elem(block).elem(c)+=qi[block][pow]*tmp_tri_off_r.elem(block).elem(c);
                }
                curr_tri_off_r.elem(block).elem(c)=tmp_tri_off_r.elem(block).elem(c);
            } 

          }
          multiply<REALT,TRIDIA_R,TRIOFF_R,0>(tmp_tri_dia_r,tmp_tri_off_r, curr_tri_dia_r, curr_tri_off_r, tri_dia_r, tri_off_r);
          multiply<REALT,TRIDIA_R,TRIOFF_R,1>(tmp_tri_dia_r,tmp_tri_off_r, curr_tri_dia_r, curr_tri_off_r, tri_dia_r, tri_off_r);
      }

    exp_tri_dia_j= exp_tri_dia_r;
    exp_tri_off_j= exp_tri_off_r;


    jit_get_function(function);
  }

#endif


  template<typename RealT,typename T, typename X,typename Y, typename Q>
  void function_apply_exp_clov_exec(JitFunction& function,
                const RealT& diag_mass,
				T& chi,
				const T& psi,
				const X& tri_dia,
				const Y& tri_off,
                const Q& qc,
				const Subset& s)
  {
#ifdef QDP_DEEP_LOG
    function.type_W = typeid(REAL).name();
    function.set_dest_id( chi.getId() );
    function.set_is_lat(true);
#endif
   
    AddressLeaf addr_leaf(s);

    forEach(diag_mass, addr_leaf, NullCombine());
    forEach(chi, addr_leaf, NullCombine());
    forEach(psi, addr_leaf, NullCombine());
    forEach(tri_dia, addr_leaf, NullCombine());
    forEach(tri_off, addr_leaf, NullCombine());

    forEach(qc, addr_leaf, NullCombine());

    int th_count = s.numSiteTable();
    WorkgroupGuardExec workgroupGuardExec(th_count);

    std::vector<QDPCache::ArgKey> ids;
    workgroupGuardExec.check(ids);
    ids.push_back( s.getIdSiteTable() );
    for(unsigned i=0; i < addr_leaf.ids.size(); ++i) 
      ids.push_back( addr_leaf.ids[i] );
    jit_launch(function,th_count,ids);

  }


  template<typename T, typename X,typename Y, int blck>
  inline void applySiteBlock(T& tmp_r,
                  const T& chi_r,
                  const X& tri_dia_r,
                  const Y& tri_off_r)
{

    int n = 2*Nc;

    for(int i = 0; i < n; ++i)
      {
	tmp_r.elem((blck*n+i)/3).elem((blck*n+i)%3) = tri_dia_r.elem(blck).elem(i) * chi_r.elem((blck*n+i)/3).elem((blck*n+i)%3);
	// cchi[blck*n+i] = tri[site].diag[blck][i] * ppsi[blck*n+i];
      }

    int kij = 0;  
    for(int i = 0; i < n; ++i)
      {
	for(int j = 0; j < i; j++)
	  {
	    tmp_r.elem((blck*n+i)/3).elem((blck*n+i)%3) +=  tri_off_r.elem(blck).elem(kij) * chi_r.elem((blck*n+j)/3).elem((blck*n+j)%3);
	    // cchi[blck*n+i] += tri[site].offd[blck][kij] * ppsi[blck*n+j];

	    tmp_r.elem((blck*n+j)/3).elem((blck*n+j)%3) +=  conj(tri_off_r.elem(blck).elem(kij)) * chi_r.elem((blck*n+i)/3).elem((blck*n+i)%3);
	    // cchi[blck*n+j] += conj(tri[site].offd[blck][kij]) * ppsi[blck*n+i];
	    kij++;
	  }
      }

}


  template<typename RealT,typename T, typename X,typename Y, typename Q>
  void function_apply_exp_clov_build( JitFunction& function,
                  const RealT& diag_mass,
				  const T& chi,
				  const T& psi,
				  const X& tri_dia,
				  const Y& tri_off,
                  const Q& qc,
				  const Subset& s)
  {
    llvm_start_new_function("apply_exp_clov",__PRETTY_FUNCTION__);

    WorkgroupGuard workgroupGuard;
    ParamRef p_site_table = llvm_add_param<int*>();

    ParamLeafScalar param_leaf;

    typedef typename WordType<T>::Type_t REALT;

    typedef typename LeafFunctor<RealT, ParamLeafScalar>::Type_t  RealTJIT;
    RealTJIT diag_mass_jit(forEach(diag_mass, param_leaf, TreeCombine()));

    typedef typename LeafFunctor<T, ParamLeafScalar>::Type_t  TJIT;
    TJIT chi_jit(forEach(chi, param_leaf, TreeCombine()));
    TJIT psi_jit(forEach(psi, param_leaf, TreeCombine()));
    typename REGType< typename ScalarType<typename TJIT::Subtype_t>::Type_t >::Type_t psi_r;
    typename REGType< typename ScalarType<typename TJIT::Subtype_t>::Type_t >::Type_t chi_r;
    typename REGType< typename ScalarType<typename TJIT::Subtype_t>::Type_t >::Type_t tmp_r;
    typedef typename REGType< typename ScalarType<typename TJIT::Subtype_t>::Type_t >::Type_t CHI_R;


    typedef typename LeafFunctor<X, ParamLeafScalar>::Type_t  XJIT;
    XJIT tri_dia_jit(forEach(tri_dia, param_leaf, TreeCombine()));
    typename REGType< typename XJIT::Subtype_t >::Type_t tri_dia_r;
    typedef typename REGType< typename XJIT::Subtype_t >::Type_t TRIDIA_R;


    typedef typename LeafFunctor<Y, ParamLeafScalar>::Type_t  YJIT;
    YJIT tri_off_jit(forEach(tri_off, param_leaf, TreeCombine()));
    typename REGType< typename YJIT::Subtype_t >::Type_t tri_off_r;
    typedef typename REGType< typename YJIT::Subtype_t >::Type_t TRIOFF_R;

    typedef typename LeafFunctor<Q, ParamLeafScalar>::Type_t  QJIT;
    QJIT qc_jit(forEach(qc, param_leaf, TreeCombine()));
    typename REGType< typename QJIT::Subtype_t >::Type_t qc_r;

    llvm::Value* r_idx_thread = llvm_thread_idx();

    workgroupGuard.check(r_idx_thread);

    llvm::Value* r_idx = llvm_array_type_indirection<int>( p_site_table , r_idx_thread );


    typename REGType< typename RealTJIT::Subtype_t >::Type_t diag_mass_reg;
    diag_mass_reg.setup_value( diag_mass_jit.elem() );

    auto chi_j = chi_jit.elem(JitDeviceLayout::Coalesced,r_idx);
    psi_r.setup( psi_jit.elem(JitDeviceLayout::Coalesced,r_idx) );
    tri_dia_r.setup( tri_dia_jit.elem(JitDeviceLayout::Coalesced,r_idx) );
    tri_off_r.setup( tri_off_jit.elem(JitDeviceLayout::Coalesced,r_idx) );
 
    qc_r.setup( qc_jit.elem(JitDeviceLayout::Coalesced,r_idx) );

    //Set the highest power of A^n for the exp sum. This allows for N_exp_default < 5 to compare with clover 
    int pow_max=5;
    if (N_exp_default <5)
       pow_max=N_exp_default;  

    int n = 2*Nc;
    for(int cspin = 0; cspin < n; ++cspin)
    {
        chi_r.elem((0*n+cspin)/3).elem((0*n+cspin)%3) = psi_r.elem((0*n+cspin)/3).elem((0*n+cspin)%3);
        // cchi[0*n+i] = ppsi[0*n+i];
        chi_r.elem((1*n+cspin)/3).elem((1*n+cspin)%3) = psi_r.elem((1*n+cspin)/3).elem((1*n+cspin)%3);
        // cchi[1*n+i] = tri[site].diag[1][i] * ppsi[1*n+i];
    }

      // Main loop:  chi = psi + q[i]/q[i-1] A chi
      for (int pow = pow_max; pow > 0; --pow)
      {

        applySiteBlock<CHI_R,TRIDIA_R,TRIOFF_R,0>(tmp_r,chi_r,tri_dia_r,tri_off_r);
        applySiteBlock<CHI_R,TRIDIA_R,TRIOFF_R,1>(tmp_r,chi_r,tri_dia_r,tri_off_r);
        for(int cspin = 0; cspin < n; ++cspin)
        {
            // Operator
            //cchi[cspin] = ppsi[cspin] + (tri_in.q[0][pow] / tri_in.q[0][pow - 1]) * tmp[cspin];
            chi_r.elem((0*n+cspin)/3).elem((0*n+cspin)%3) = psi_r.elem((0*n+cspin)/3).elem((0*n+cspin)%3)
                    +(qc_r.elem(0).elem(pow) / qc_r.elem(0).elem(pow - 1)) * tmp_r.elem((0*n+cspin)/3).elem((0*n+cspin)%3) ;
            chi_r.elem((1*n+cspin)/3).elem((1*n+cspin)%3) = psi_r.elem((1*n+cspin)/3).elem((1*n+cspin)%3)
                    +(qc_r.elem(1).elem(pow) / qc_r.elem(1).elem(pow - 1)) * tmp_r.elem((1*n+cspin)/3).elem((1*n+cspin)%3) ;

        }

      }

    for(int cspin = 0; cspin < n; ++cspin)
    {   

        chi_r.elem((0*n+cspin)/3).elem((0*n+cspin)%3) *= qc_r.elem(0).elem(0)*diag_mass_reg.elem().elem();
        chi_r.elem((1*n+cspin)/3).elem((1*n+cspin)%3) *= qc_r.elem(1).elem(0)*diag_mass_reg.elem().elem();
    }


    chi_j = chi_r;

    jit_get_function(function);
  }




  template<typename T>
  void function_copy_exec(JitFunction& function,
				T& chi,
				const T& psi,
				const Subset& s)
  {
#ifdef QDP_DEEP_LOG
    function.type_W = typeid(REAL).name();
    function.set_dest_id( chi.getId() );
    function.set_is_lat(true);
#endif
   
    AddressLeaf addr_leaf(s);

    forEach(chi, addr_leaf, NullCombine());
    forEach(psi, addr_leaf, NullCombine());

    int th_count = s.numSiteTable();
    WorkgroupGuardExec workgroupGuardExec(th_count);

    std::vector<QDPCache::ArgKey> ids;
    workgroupGuardExec.check(ids);
    ids.push_back( s.getIdSiteTable() );

    for(unsigned i=0; i < addr_leaf.ids.size(); ++i) 
      ids.push_back( addr_leaf.ids[i] );
    jit_launch(function,th_count,ids);

  }

  template<typename T>
  void function_copy_build( JitFunction& function,
                  const T& chi,
                  const T& psi,
				  const Subset& s)
  {
    llvm_start_new_function("copy",__PRETTY_FUNCTION__);

    WorkgroupGuard workgroupGuard;
    ParamRef p_site_table = llvm_add_param<int*>();

    ParamLeafScalar param_leaf;

    typedef typename WordType<T>::Type_t REALT;

    typedef typename LeafFunctor<T, ParamLeafScalar>::Type_t  TJIT;
    TJIT chi_jit(forEach(chi, param_leaf, TreeCombine()));
    TJIT psi_jit(forEach(psi, param_leaf, TreeCombine()));
    typename REGType< typename ScalarType<typename TJIT::Subtype_t>::Type_t >::Type_t psi_r;
    typename REGType< typename ScalarType<typename TJIT::Subtype_t>::Type_t >::Type_t chi_r;

    llvm::Value* r_idx_thread = llvm_thread_idx();

    workgroupGuard.check(r_idx_thread);

    llvm::Value* r_idx = llvm_array_type_indirection<int>( p_site_table , r_idx_thread );

    auto chi_j = chi_jit.elem(JitDeviceLayout::Coalesced,r_idx);
    psi_r.setup( psi_jit.elem(JitDeviceLayout::Coalesced,r_idx) );

    typedef typename WordType<T>::Type_t REALT;

    int n = 2*Nc;
    for(int cspin = 0; cspin < n; ++cspin)
    {
        chi_r.elem((0*n+cspin)/3).elem((0*n+cspin)%3) = psi_r.elem((0*n+cspin)/3).elem((0*n+cspin)%3);
        chi_r.elem((1*n+cspin)/3).elem((1*n+cspin)%3) = psi_r.elem((1*n+cspin)/3).elem((1*n+cspin)%3);
        // cchi[1*n+i] =  ppsi[1*n+i];
    }

    chi_j = chi_r;

    jit_get_function(function);
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

  template <typename T, typename U,int N_exp>
  void JITExpCloverTermT<T,U,N_exp>::apply(T& chi, const T& psi, enum PlusMinus isign,
                        int cb) const
  {

    START_CODE();

    if (Ns != 4)
    {
      QDPIO::cerr << __func__ << ": ExpCloverTerm::apply requires Ns==4" << std::endl;
      QDP_abort(1);
    }

    static JitFunction function;

    if (function.empty()){
      function_apply_exp_clov_build( function, diag_mass,  chi, psi, tri_dia, tri_off, qc, rb[cb]);

    }

    // Execute the function
    function_apply_exp_clov_exec( function, diag_mass, chi, psi, tri_dia, tri_off, qc, rb[cb]);


    (*this).getFermBC().modifyF(chi, QDP::rb[cb]);

    END_CODE();

  }


  template <typename T, typename U,int N_exp>
  void JITExpCloverTermT<T,U,N_exp>::applyExpClov(T& chi, const T& psi, enum PlusMinus isign,
                        int cb) const
  { 
  
    START_CODE();
           
    if (Ns != 4)
    {
      QDPIO::cerr << __func__ << ": ExpCloverTerm::apply requires Ns==4" << std::endl;
      QDP_abort(1); 
    }      
    static JitFunction function;
#if 0
    if (function.empty())
      function_apply_exp_clov_build( function, diag_mass,  chi, psi, exp_tri_dia, exp_tri_off, qc, rb[cb]);
    // Execute the function
    function_apply_exp_clov_exec(function, diag_mass, chi, psi, exp_tri_dia, exp_tri_off,qc, rb[cb] );

#else
    
    if (function.empty()){
      //function_apply_exp_clov_build( function, diag_mass,  chi, psi, exp_tri_dia, exp_tri_off, qc, rb[cb]);
      function_apply_clov_build( function, chi, psi, exp_tri_dia, exp_tri_off, rb[cb] );
    }

    // Execute the function
    function_apply_clov_exec( function, chi, psi, exp_tri_dia, exp_tri_off, rb[cb] );
    //function_apply_exp_clov_exec( function, diag_mass, chi, psi, exp_tri_dia, exp_tri_off, qc, rb[cb]);

#endif

    (*this).getFermBC().modifyF(chi, QDP::rb[cb]);

    END_CODE();

  }



  template<typename T, typename W>
  void function_apply_coeff_exec(JitFunction& function,
                T& chi,
                const T& psi,
                const int pow_i,
                const int pow_j,
                const W& Cv,
                const Subset& s)
  {
#ifdef QDP_DEEP_LOG
    function.type_W = typeid(REAL).name();
    function.set_dest_id( chi.getId() );
    function.set_is_lat(true);
#endif

    AddressLeaf addr_leaf(s);

    forEach(chi, addr_leaf, NullCombine());
    forEach(psi, addr_leaf, NullCombine());

    forEach(Cv, addr_leaf, NullCombine());

    int th_count = s.numSiteTable();
    WorkgroupGuardExec workgroupGuardExec(th_count);

    JitParam jit_pow_i( QDP_get_global_cache().addJitParamInt( pow_i ) );
    JitParam jit_pow_j( QDP_get_global_cache().addJitParamInt( pow_j ) );

    std::vector<QDPCache::ArgKey> ids;
    workgroupGuardExec.check(ids);
    ids.push_back( s.getIdSiteTable() );
    ids.push_back(jit_pow_i.get_id());
    ids.push_back(jit_pow_j.get_id());

    for(unsigned i=0; i < addr_leaf.ids.size(); ++i)
      ids.push_back( addr_leaf.ids[i] );
    jit_launch(function,th_count,ids);

  }


  template<typename T,typename W>
  void function_apply_coeff_build( JitFunction& function,
                  const T& chi,
                  const T& psi,
                  const int pow_i,
                  const int pow_j,
                  const W& Cv,
                  const Subset& s)
  {
    llvm_start_new_function("apply_coeff",__PRETTY_FUNCTION__);

    WorkgroupGuard workgroupGuard;
    ParamRef p_site_table = llvm_add_param<int*>();
    ParamRef p_pow_i = llvm_add_param<int>();
    ParamRef p_pow_j = llvm_add_param<int>();

    ParamLeafScalar param_leaf;

    typedef typename WordType<T>::Type_t REALT;

    typedef typename LeafFunctor<T, ParamLeafScalar>::Type_t  TJIT;
    TJIT chi_jit(forEach(chi, param_leaf, TreeCombine()));
    TJIT psi_jit(forEach(psi, param_leaf, TreeCombine()));
    typename REGType< typename ScalarType<typename TJIT::Subtype_t>::Type_t >::Type_t psi_r;
    typename REGType< typename ScalarType<typename TJIT::Subtype_t>::Type_t >::Type_t chi_r;

    typedef typename LeafFunctor<W, ParamLeafScalar>::Type_t  WJIT;
    WJIT Cv_jit(forEach(Cv, param_leaf, TreeCombine()));
    typename REGType< typename WJIT::Subtype_t >::Type_t Cv_r;


    llvm::Value* r_idx_thread = llvm_thread_idx();
    llvm::Value* v_pow_i = llvm_derefParam(p_pow_i);
    llvm::Value* v_pow_j = llvm_derefParam(p_pow_j);

    workgroupGuard.check(r_idx_thread);

    llvm::Value* r_idx = llvm_array_type_indirection<int>( p_site_table , r_idx_thread );


    auto Cv_j     = Cv_jit.elem(JitDeviceLayout::Coalesced,r_idx);
    Cv_r.setup( Cv_j );

    auto chi_j = chi_jit.elem(JitDeviceLayout::Coalesced,r_idx);
    psi_r.setup( psi_jit.elem(JitDeviceLayout::Coalesced,r_idx) );

    int n = 2*Nc;

    for(int cspin = 0; cspin < n; ++cspin)
    {
       chi_r.elem((0*n+cspin)/3).elem((0*n+cspin)%3) = Cv_r.elem(0).elem(pow_i).elem(pow_j)*psi_r.elem((0*n+cspin)/3).elem((0*n+cspin)%3);
       chi_r.elem((1*n+cspin)/3).elem((1*n+cspin)%3) = Cv_r.elem(1).elem(pow_i).elem(pow_j)*psi_r.elem((1*n+cspin)/3).elem((1*n+cspin)%3);
       // cchi[1*n+i] =  ppsi[1*n+i];
    }

    chi_j = chi_r;

    jit_get_function(function);
  }


  template<typename T, typename U,int N_exp>
  void JITExpCloverTermT<T,U,N_exp>::applyCoeff(T& chi, const T& psi,
                  enum PlusMinus isign, int cb, int pow_i, int pow_j) const
  {
    START_CODE();

    static JitFunction function;

    if (function.empty())
      function_apply_coeff_build( function, chi, psi,pow_i, pow_j, C, rb[cb]);
    
    // Execute the function
    function_apply_coeff_exec( function, chi, psi, pow_i,pow_j, C, rb[cb]);

    (*this).getFermBC().modifyF(chi, QDP::rb[cb]);

    END_CODE();
  }


  template<typename T, typename U,int N_exp>
  void JITExpCloverTermT<T,U,N_exp>::applyPower(T& chi, const T& psi,
                  enum PlusMinus isign, int cb, int power) const
  {
    START_CODE();

    if ( Ns != 4 ) {
      QDPIO::cerr << __func__ << ": ExpCloverTerm::apply requires Ns==4" << std::endl;
      QDP_abort(1);
    }


    T tmp;
    static JitFunction functionCopy;
    static JitFunction function;   

    if(power==0){
        if (functionCopy.empty())
          function_copy_build( functionCopy, chi, psi, rb[cb]);

        // Execute the function
        function_copy_exec( functionCopy, chi, psi, rb[cb]);
    }else{
        if (functionCopy.empty())
          function_copy_build( functionCopy, tmp, psi, rb[cb]);
        // Execute the function
        function_copy_exec( functionCopy, tmp, psi, rb[cb]);

        for(int p=power; p > 0; --p){
            if (function.empty())
              function_apply_clov_build( function, chi, tmp, tri_dia, tri_off, rb[cb] );

            // Execute the function
            function_apply_clov_exec(function, chi, tmp, tri_dia, tri_off, rb[cb] );
            function_copy_exec( functionCopy, tmp, chi, rb[cb]);
        }
    }
    (*this).getFermBC().modifyF(chi, QDP::rb[cb]);

    END_CODE();
  }


  template <typename T, typename U,int N_exp>
  //void JITExpCloverTermT<T, U, N_exp>::applyInv(T& chi, const T& psi, enum PlusMinus isign,
  //                      int cb) const
  void JITExpCloverTermT<T,U,N_exp>::applyInv(T& chi, const T& psi, enum PlusMinus isign,
                        int cb) const
  {

    START_CODE();

    if (Ns != 4)
    {
      QDPIO::cerr << __func__ << ": ExpCloverTerm::apply requires Ns==4" << std::endl;
      QDP_abort(1);
    }
      
    static JitFunction function;
    RealT inv_diag_mass = 1.0 /diag_mass;

    if (function.empty()){
      function_apply_exp_clov_build( function, inv_diag_mass,  chi, psi, tri_dia, tri_off, qc_inv, rb[cb]);
    }
    // Execute the function
    function_apply_exp_clov_exec( function, inv_diag_mass, chi, psi, tri_dia, tri_off, qc_inv, rb[cb]);


    (*this).getFermBC().modifyF(chi, QDP::rb[cb]);

    END_CODE();

  }


#ifndef  BUILD_QUDA_DEVIFACE_CLOVER

#if 0
  namespace QDPCloverEnv {

    template<typename R,typename TD,typename TO> 
    struct QUDAPackArgs { 
      int cb;
      multi1d<QUDAPackedClovSite<R> >& quda_array;
      const TD&  tri_dia;
      const TO&  tri_off;
    };
    
    template<typename R,typename TD,typename TO>
    void qudaPackSiteLoop(int lo, int hi, int myId, QUDAPackArgs<R,TD,TO>* a) {
      int cb = a->cb;
      int Ns2 = Ns/2;

      multi1d<QUDAPackedClovSite<R> >& quda_array = a->quda_array;

      const TD& tri_dia = a->tri_dia;
      const TO& tri_off = a->tri_off;

      const int idtab[15]={0,1,3,6,10,2,4,7,11,5,8,12,9,13,14};

      for(int ssite=lo; ssite < hi; ++ssite) {
	int site = rb[cb].siteTable()[ssite];
	// First Chiral Block
	for(int i=0; i < 6; i++) { 
	  quda_array[site].diag1[i] = tri_dia.elem(site).comp[0].diag[i].elem().elem();
	}

	int target_index=0;
	
	for(int col=0; col < Nc*Ns2-1; col++) { 
	  for(int row=col+1; row < Nc*Ns2; row++) {

	    int source_index = row*(row-1)/2 + col;

	    quda_array[site].offDiag1[target_index][0] = tri_off.elem(site).comp[0].offd[source_index].real().elem();
	    quda_array[site].offDiag1[target_index][1] = tri_off.elem(site).comp[0].offd[source_index].imag().elem();
	    target_index++;
	  }
	}
	// Second Chiral Block
	for(int i=0; i < 6; i++) { 
	  quda_array[site].diag2[i] = tri_dia.elem(site).comp[1].diag[i].elem().elem();
	}

	target_index=0;
	for(int col=0; col < Nc*Ns2-1; col++) { 
	  for(int row=col+1; row < Nc*Ns2; row++) {

	    int source_index = row*(row-1)/2 + col;

	    quda_array[site].offDiag2[target_index][0] = tri_off.elem(site).comp[1].offd[source_index].real().elem();
	    quda_array[site].offDiag2[target_index][1] = tri_off.elem(site).comp[1].offd[source_index].imag().elem();
	    target_index++;
	  }
	}
      }
      QDPIO::cout << "\n";
    }
  }

#endif

  template <typename T, typename U,int N_exp>
  void JITExpCloverTermT<T,U,N_exp>::packForQUDA(multi1d<QUDAPackedClovSite<typename WordType<T>::Type_t> >& quda_array, int cb) const
    {
      typedef typename WordType<T>::Type_t REALT;
      int num_sites = rb[cb].siteTable().size();

      typedef OLattice<PComp<PTriDia<RScalar <Word<REALT> > > > > TD;
      typedef OLattice<PComp<PTriOff<RComplex<Word<REALT> > > > > TO;

      StopWatch watch;
      watch.start();

      QDPCloverEnv::QUDAPackArgs<REALT,TD,TO> args = { cb, quda_array , exp_tri_dia , exp_tri_off };
      dispatch_to_threads(num_sites, args, QDPCloverEnv::qudaPackSiteLoop<REALT,TD,TO>);

      watch.stop();
      PackForQUDATimer::Instance().get() += watch.getTimeInMicroseconds();
    }
#endif

  template<typename T, typename U,int N_exp>
  void JITExpCloverTermT<T,U,N_exp>::applySite(T& chi, const T& psi, 
				      enum PlusMinus isign, int site) const
  {
    QDP_error_exit("JITExpCloverTermT<T,U>::applySite(T& chi, const T& psi,..) not implemented ");
  }


  template <typename T, typename U,int N_exp>
  void JITExpCloverTermT<T,U,N_exp>::makeExpClov(enum PlusMinus isign,
                         int cb, int inverse)
  {
    START_CODE();

    if (Ns != 4)
    {
      QDPIO::cerr << __func__ << ": ExpCloverTerm::apply requires Ns==4" << std::endl;
      QDP_abort(1);
    }

    Real mclov;


    static JitFunction function;
    T dummy;


    if (inverse==0){
        if (function.empty())
          function_make_exp_tri_clov_build(function, diag_mass, exp_tri_dia, exp_tri_off, tri_dia , tri_off, qc, rb[cb]);

        // Execute the function
        function_make_exp_tri_clov_exec(function, diag_mass, exp_tri_dia, exp_tri_off, tri_dia , tri_off, qc, rb[cb]);
    }else{
        RealT inv_diag_mass = 1.0 /diag_mass;
        //diag_mass=1.0/diag_mass;       
        function_make_exp_tri_clov_build(function, inv_diag_mass, exp_tri_dia, exp_tri_off, tri_dia , tri_off, qc_inv, rb[cb]);

        // Execute the function
        function_make_exp_tri_clov_exec(function, inv_diag_mass, exp_tri_dia, exp_tri_off, tri_dia , tri_off, qc_inv, rb[cb]);
    }



    END_CODE();

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
  void JITExpCloverTermT<T, U, N_exp>::deriv(multi1d<U>& ds_u,
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


//Improved derivative
  template <typename T, typename U, int N_exp>
  void JITExpCloverTermT<T, U, N_exp>::deriv(multi1d<U>& ds_u,
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
    multi1d<U> ds_u_tmp;
    ds_u_tmp.resize(Nd);

    // Get the links
    //const multi1d<U>& u = getU();
    T ppsi= zero;
    T cchi= zero;
    T tmp_psi= psi;
    T sum_psi= zero;

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
            //(*this).applyCoeff(tmp_psi, psi, isign,cb,i,j);
            tmp_psi=psi*C_arr[cb][i][j]; //.elem(0).comp[cb].q[i].q[j].elem().elem();
            (*this).applyPower(ppsi, tmp_psi,isign, cb, j);
            sum_psi_vec[i]+=ppsi;
        }

        (*this).applyPower(cchi_vec[i], chi, isign, cb,i);

    }

    ExpCloverTermBase<T,U>::derivMultipole(ds_u,cchi_vec,sum_psi_vec,isign,cb);
 

    // Clear out the deriv on any fixed links
    (*this).getFermBC().zero(ds_u);
    

    END_CODE();
  }

#if 1
  template <typename T, typename U, int N_exp>
  void JITExpCloverTermT<T, U, N_exp>::derivMultipole(multi1d<U>& ds_u,
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


  template <typename T, typename U, int N_exp>
  void JITExpCloverTermT<T, U, N_exp>::derivMultipole(multi1d<U>& ds_u,
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
    multi1d<U> ds_u_tmp;
    ds_u_tmp.resize(Nd);

    // Get the links
    //const multi1d<U>& u = getU();

    T ppsi= zero;
    T cchi= zero;
    T tmp_psi= zero; //psi;
    T sum_psi= zero;


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
        tmp_psi= psi[k];
        for(int i=0;i<=5;i++){
            sum_psi_vec[nterm]= zero;
            cchi_vec[nterm]= zero;

            for(int j=0;j<=5;j++){
                //(*this).applyCoeff(tmp_psi, psi[k], isign,cb,i,j);
                tmp_psi=psi[k]*C_arr[cb][i][j]; //.elem(0).comp[cb].q[i].q[j].elem().elem();

                (*this).applyPower(ppsi, tmp_psi, isign, cb, j);
                sum_psi_vec[nterm]+=ppsi;
            }

            (*this).applyPower(cchi_vec[nterm], chi[k], isign, cb,i);
            nterm+=1;
        }
    }
    
     ExpCloverTermBase<T,U>::derivMultipole(ds_u,cchi_vec,sum_psi_vec,isign,cb);

    // Clear out the deriv on any fixed links
    (*this).getFermBC().zero(ds_u);
    END_CODE();
  }
#endif

  typedef JITExpCloverTermT<LatticeFermion, LatticeColorMatrix, N_exp_default> JITExpCloverTerm; 
  typedef JITExpCloverTermT<LatticeFermionF, LatticeColorMatrixF, N_exp_default> JITExpCloverTermF;
  typedef JITExpCloverTermT<LatticeFermionD, LatticeColorMatrixD, N_exp_default> JITExpCloverTermD;

} // End Namespace Chroma

#endif
#endif
