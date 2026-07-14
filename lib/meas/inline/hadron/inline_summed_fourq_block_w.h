// -*- C++ -*-
/*! \file
 * \brief Inline measurement of SUMMED_FOURQ_BLOCK
 *
 * Reads a propagator and multiplies it by a constant c.
 */

#ifndef __inline_summed_fourq_block_w_h__
#define __inline_summed_fourq_block_w_h__

#include "chromabase.h"
#include "meas/inline/abs_inline_measurement.h"

namespace Chroma
{
  /*! \ingroup inlinehadron */
  namespace InlineSummedFourQBlockEnv
  {
    extern const std::string name;
    bool registerAll();

    //! Parameter structure
    /*! \ingroup inlinehadron */
    struct Params
    {
      Params();
      Params(XMLReader& xml_in, const std::string& path);
      void writeXML(XMLWriter& xml_out, const std::string& path);

      unsigned long frequency;

      bool         write_files; /*!< Write each prop_out to disk and erase from map */

      struct NamedObject_t
      {
        multi1d<int>         insertion_position; /*!< Current insertion spacetime position */
        multi1d<std::string> operators;          /*!< Operator names for the current insertion */
        std::string  prop_xy_id;      /*!< Input propagator Pxy (x=sink, y=insertion) */
        std::string  prop_yz_id;      /*!< Input propagator Pyz (y=insertion, z=source) */
        std::string  result_id;       /*!< Output propagator stem */
      } named_obj;
    };


    //! Inline measurement: reads a propagator and multiplies it by constant c
    /*! \ingroup inlinehadron */
    class InlineMeas : public AbsInlineMeasurement
    {
    public:
      ~InlineMeas() {}
      InlineMeas(const Params& p) : params(p) {}
      InlineMeas(const InlineMeas& p) : params(p.params) {}

      unsigned long getFrequency(void) const {return params.frequency;}

      //! Do the measurement
      void operator()(const unsigned long update_no,
                      XMLWriter& xml_out);

    private:
      Params params;
    };

  }

}

#endif
