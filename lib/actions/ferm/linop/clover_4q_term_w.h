// -*- C++ -*-
/*! \file
 *  \brief Include possibly optimized Clover terms
 */

#ifndef __clover_term_w_h__
#define __clover_term_w_h__

#include "chroma_config.h"
#include "qdp_config.h"

// The QDP naive clover term
//


// The following is an ifdef lis that switches in optimised
// terms. Currently only optimised dslash is the SSE One;
// Bottom line, if no optimised Dslash-s exist then the naive QDP Dslash
// becomes the WilsonDslash

#include "clover_term_qdp_4q_w.h"
namespace Chroma {
  using CloverTerm  = QDP4QCloverTerm;
  using CloverTermF = QDP4QCloverTermF;
  using CloverTermD = QDP4QCloverTermD;

  template<typename T,typename U>
  using CloverTermT = QDP4QCloverTermT<T,U>;
}  // end namespace Chroma



#endif
