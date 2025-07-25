// -*- C++ -*-
/*! \file
 *  \brief Include possibly optimized Clover terms
 */

#ifndef __clover_term_2qblock_w_h__
#define __clover_term_2qblock_w_h__

#include "chroma_config.h"
#include "qdp_config.h"

// The QDP naive clover term
//


// The following is an ifdef lis that switches in optimised
// terms. Currently only optimised dslash is the SSE One;
// Bottom line, if no optimised Dslash-s exist then the naive QDP Dslash
// becomes the WilsonDslash

#include "clover_term_qdp_2qblock_w.h"
namespace Chroma {
  using CloverTerm  = QDPCloverTerm2QB;
  using CloverTermF = QDPCloverTerm2QBF;
  using CloverTermD = QDPCloverTerm2QBD;

  template<typename T,typename U>
  using CloverTermT = QDPCloverTerm2QBT<T,U>;
}  // end namespace Chroma



#endif
