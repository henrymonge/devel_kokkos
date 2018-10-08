// -*- C++ -*-
/*! @file
 * @brief Symmetric even-odd const determinant Wilson-like fermact
 */

#ifndef __seoprec_constdet_wilstype_fermact_w_h__
#define __seoprec_constdet_wilstype_fermact_w_h__

#include "seoprec_wilstype_fermact_w.h"
#include "seoprec_constdet_linop.h"

namespace Chroma
{
  //-------------------------------------------------------------------------------------------
  //! Symmetric even-odd preconditioned Wilson-like fermion actions specialised to Wilson Like (gauge independent diagonal term) actions.
  /*! @ingroup actions
   *
   * Symmetric even-odd preconditioned like Wilson-like fermion actions
   */
  template<typename T, typename P, typename Q>
  class SymEvenOddPrecConstDetWilsonTypeFermAct : public SymEvenOddPrecWilsonTypeFermAct<T,P,Q>
  {
  public:
    //! Virtual destructor to help with cleanup;
    virtual ~SymEvenOddPrecConstDetWilsonTypeFermAct() {}

    //! Override to produce a symmetric even-odd prec. linear operator for this action
    /*! Covariant return rule - override base class function */
    virtual SymEvenOddPrecConstDetLinearOperator<T,P,Q>* linOp(Handle< FermState<T,P,Q> > state) const = 0;

  };


  //! Symmetric even-odd preconditioned Wilson-like fermion actions including derivatives
  /*! @ingroup actions
   *
   * Even-odd preconditioned like Wilson-like fermion actions
   * Here, use arrays of matter fields.
   */
  template<typename T, typename P, typename Q>
  class SymEvenOddPrecConstDetWilsonTypeFermAct5D : public SymEvenOddPrecWilsonTypeFermAct5D<T,P,Q>
  {
  public:
    //! Virtual destructor to help with cleanup;
    virtual ~SymEvenOddPrecConstDetWilsonTypeFermAct5D() {}

    //! Override to produce a symmetric even-odd prec. linear operator for this action
    /*! Covariant return rule - override base class function */
    virtual SymEvenOddPrecConstDetLinearOperatorArray<T,P,Q>* linOp(Handle< FermState<T,P,Q> > state) const = 0;

    //! Override to produce a symmetric even-odd prec. Pauli-Villars linear operator for this action
    /*! Covariant return rule - override base class function */
    virtual SymEvenOddPrecConstDetLinearOperatorArray<T,P,Q>* linOpPV(Handle< FermState<T,P,Q> > state) const = 0;

  };

}


#endif
