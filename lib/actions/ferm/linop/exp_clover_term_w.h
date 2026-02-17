// -*- C++ -*-
/*! \file
 *  \brief Include possibly optimized Clover terms
 */

#ifndef __exp_clover_term_w_h__
#define __exp_clover_term_w_h__

#include "chroma_config.h"
#include "qdp_config.h"

// The QDP naive clover term
//
#if defined(BUILD_JIT_CLOVER_TERM)
#include "actions/ferm/linop/exp_clover_term_jit_w.h"

namespace Chroma
{


  using ExpCloverTerm = JITExpCloverTerm;
  using ExpCloverTermF = JITExpCloverTermF;
  using ExpCloverTermD = JITExpCloverTermD;

  template<typename T,typename U>
  using ExpCloverTermT = JITExpCloverTermT<T,U>;

}
#else
#include "actions/ferm/linop/exp_clover_term_qdp_w.h"
namespace Chroma
{

  using ExpCloverTerm = QDPExpCloverTerm<>;
  using ExpCloverTermF = QDPExpCloverTermF<>;
  using ExpCloverTermD = QDPExpCloverTermD<>;

  template<typename T,typename U>
  using ExpCloverTermT = QDPExpCloverTermT<T,U>;

}
#endif

#endif
