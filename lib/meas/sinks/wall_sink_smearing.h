// -*- C++ -*-
// $Id: wall_sink_smearing.h,v 1.1 2007-08-27 20:05:42 uid3790 Exp $
/*! \file
 *  \brief Wall sink smearing
 */

#ifndef __wall_sink_smearing_h__
#define __wall_sink_smearing_h__

#include "meas/smear/quark_source_sink.h"
#include "io/xml_group_reader.h"

namespace Chroma
{

  //! Name and registration
  /*! @ingroup sinks */
  namespace WallQuarkSinkSmearingEnv
  {
    extern const std::string name;
    bool registerAll();


    //! Wall sink parameters
    /*! @ingroup sinks */
    struct Params
    {
      Params();
      Params(XMLReader& in, const std::string& path);
      void writeXML(XMLWriter& in, const std::string& path) const;
    };


    //! Wall sink smearing
    /*! @ingroup sinks
     *
     * Make a wall propagator sink
     */
    template<typename T>
    class SinkSmear : public QuarkSourceSink<T>
    {
    public:
      //! Full constructor
      SinkSmear(const Params& p, const multi1d<LatticeColorMatrix>& u) : params(p)
	{
	  this->create(u_smr, params.link_smearing);
	}

      //! Smear the sink
      void operator()(T& obj) const;

    private:
      //! Hide partial constructor
      SinkSmear() {}

    private:
      Params  params;                         /*!< sink params */
    };

  } // end namespace


  //! Reader
  /*! @ingroup sinks */
  void read(XMLReader& xml, const string& path, WallQuarkSinkSmearingEnv::Params& param);

  //! Writer
  /*! @ingroup sinks */
  void write(XMLWriter& xml, const string& path, const WallQuarkSinkSmearingEnv::Params& param);

}  // end namespace Chroma


#endif