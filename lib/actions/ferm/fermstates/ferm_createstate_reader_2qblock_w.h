// -*- C++ -*-
/*! \file
 *  \brief All ferm create-state method
 */

#ifndef __ferm_createstate_reader_2qblock_w_h__
#define __ferm_createstate_reader_2qblock_w_h__

#include "create_state.h"
#include "handle.h"

namespace Chroma
{
  //! State reader
  /*! @ingroup fermstates */
  namespace CreateFermStateEnv
  {
    //! Helper function for the CreateFermState readers
    Handle< CreateFermState< LatticePropagator,
			     multi1d<LatticeColorMatrix>, 
			     multi1d<LatticeColorMatrix> > > reader(XMLReader& xml_in, 
								    const std::string& path);
  }

}

#endif
