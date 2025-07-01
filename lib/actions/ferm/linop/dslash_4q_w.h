// -*- C++ -*-
/*! \file
 *  \brief Include possibly optimized Wilson dslash
 */

#ifndef DSLASH_4Q_W_H
#define DSLASH_4Q_W_H

#include "qdp_config.h"
#include "chroma_config.h"

// QDP Completely naive Dslash class
//#include "lwldslash_w.h"
#include "lwldslash_4q_w.h"



// Bottom line, if no optimised Dslash-s exist then the naive QDP Dslash
// becomes the WilsonDslash
namespace Chroma {

  typedef QDPWilson4QDslash WilsonDslash;
  typedef QDPWilson4QDslashF WilsonDslashF;
  typedef QDPWilson4QDslashD WilsonDslashD;

}  // end namespace Chroma



// 3D Dslashes
// These guards make sure 3D is only ever considered in the right situations
#include "qdp_config.h"
#if QDP_NS==4
#if QDP_NC==3
#if QDP_ND==4

#include "lwldslash_3d_qdp_w.h"
#ifdef BUILD_SSE_WILSON_DSLASH
#include "lwldslash_3d_sse_w.h"

// For now this is the naive Wilson Dslash but 
// I put in this clause because 
namespace Chroma {
typedef SSEWilsonDslash3D WilsonDslash3D;
}

#else

// Bottom line, if no optimised 3d Dslash-s exist then the naive QDP Dslash3D
// becomes the WilsonDslash
namespace Chroma {
typedef QDPWilsonDslash3D WilsonDslash3D;
}  // end namespace Chroma
#endif


#endif
#endif
#endif


#endif
