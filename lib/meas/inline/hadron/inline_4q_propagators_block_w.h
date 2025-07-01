// -*- C++ -*-
/*! \file
 * \brief Inline construction of propagator
 *
 * Propagator calculations
 */

#ifndef __inline_4q_propagators_block_h__
#define __inline_4q_propagators_block_h__

#include "chromabase.h"
#include "meas/inline/abs_inline_measurement.h"
#include "io/qprop_io.h"

namespace Chroma 
{ 
  /*! \ingroup inlinehadron */
  namespace Inline4QPropagatorsBlockEnv 
  {
    extern const std::string name;
    bool registerAll();
  }

  //! Parameter structure
  /*! \ingroup inlinehadron */ 
  struct Inline4QPropagatorsBlockParams 
  {
    Inline4QPropagatorsBlockParams();
    Inline4QPropagatorsBlockParams(XMLReader& xml_in, const std::string& path);
    void writeXML(XMLWriter& xml_out, const std::string& path);

    unsigned long     frequency;

    ChromaProp_t      param;

    struct NamedObject_t
    {
      std::string     gauge_id;
      std::string     source_id_1;
      std::string     source_id_2;
      std::string     prop_id;
    } named_obj;

    std::string xml_file;  // Alternate XML file pattern
  };

  //! Inline propagator calculation
  /*! \ingroup inlinehadron */
  class Inline4QPropagatorsBlock : public AbsInlineMeasurement 
  {
  public:
    ~Inline4QPropagatorsBlock() {}
    Inline4QPropagatorsBlock(const Inline4QPropagatorsBlockParams& p) : params(p) {}
    Inline4QPropagatorsBlock(const Inline4QPropagatorsBlock& p) : params(p.params) {}

    unsigned long getFrequency(void) const {return params.frequency;}

    //! Do the measurement
    void operator()(const unsigned long update_no,
		    XMLWriter& xml_out); 

  protected:
    //! Do the measurement
    void func(const unsigned long update_no,
	      XMLWriter& xml_out); 

  private:
    Inline4QPropagatorsBlockParams params;
  };

}

#endif
