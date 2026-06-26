/*! \file
 * \brief Inline measurement of SUMMED_FOURQ_BLOCK
 *
 * Reads a propagator and multiplies it by a constant c.
 */

#include "meas/inline/hadron/inline_summed_fourq_block_w.h"
#include "meas/inline/abs_inline_measurement_factory.h"
#include "meas/inline/io/named_objmap.h"
#include <sstream>

namespace Chroma
{
  //! NamedObject_t reader
  void read(XMLReader& xml, const std::string& path,
            InlineSummedFourQBlockEnv::Params::NamedObject_t& input)
  {
    XMLReader inputtop(xml, path);

    read(inputtop, "t_srce",         input.t_srce);
    read(inputtop, "curr_insertion", input.curr_insertion);
    read(inputtop, "prop_xy_id",     input.prop_xy_id);
    read(inputtop, "prop_yz_id",     input.prop_yz_id);
    read(inputtop, "result_id",      input.result_id);
  }

  //! NamedObject_t writer
  void write(XMLWriter& xml, const std::string& path,
             const InlineSummedFourQBlockEnv::Params::NamedObject_t& input)
  {
    push(xml, path);

    write(xml, "t_srce",         input.t_srce);
    write(xml, "curr_insertion", input.curr_insertion);
    write(xml, "prop_xy_id",     input.prop_xy_id);
    write(xml, "prop_yz_id",     input.prop_yz_id);
    write(xml, "result_id",      input.result_id);

    pop(xml);
  }


  namespace InlineSummedFourQBlockEnv
  {
    namespace
    {
      AbsInlineMeasurement* createMeasurement(XMLReader& xml_in,
                                              const std::string& path)
      {
        return new InlineMeas(Params(xml_in, path));
      }

      //! Local registration flag
      bool registered = false;
    }

    const std::string name = "SUMMED_FOURQ_BLOCK";

    //! Register all the factories
    bool registerAll()
    {
      bool success = true;
      if (! registered)
      {
        success &= TheInlineMeasurementFactory::Instance().registerObject(name, createMeasurement);
        registered = true;
      }
      return success;
    }


    // Param stuff
    Params::Params() { frequency = 0; write_files = false; }

    Params::Params(XMLReader& xml_in, const std::string& path)
    {
      try
      {
        XMLReader paramtop(xml_in, path);

        if (paramtop.count("Frequency") == 1)
          read(paramtop, "Frequency", frequency);
        else
          frequency = 1;

        read(paramtop, "currents",    currents);
        read(paramtop, "write_files", write_files);
        read(paramtop, "NamedObject", named_obj);
      }
      catch(const std::string& e)
      {
        QDPIO::cerr << __func__ << ": Caught Exception reading XML: " << e << std::endl;
        QDP_abort(1);
      }
    }


    void
    Params::writeXML(XMLWriter& xml_out, const std::string& path)
    {
      push(xml_out, path);

      write(xml_out, "currents",    currents);
      write(xml_out, "write_files", write_files);
      write(xml_out, "NamedObject", named_obj);

      pop(xml_out);
    }


    // Function call
    void
    InlineMeas::operator()(unsigned long update_no,
                           XMLWriter& xml_out)
    {
      START_CODE();

      push(xml_out, "summed_fourq_block");
      write(xml_out, "update_no", update_no);

      QDPIO::cout << name << ": reading propagators Pxy and Pyz" << std::endl;

      // Write out the input parameters
      params.writeXML(xml_out, "Input");

      //
      // Read Pxy from the named object map
      //
      XMLReader prop_xy_file_xml, prop_xy_record_xml;
      LatticePropagator Pxy;

      QDPIO::cout << name << ": reading Pxy from " << params.named_obj.prop_xy_id << std::endl;
      try
      {
        Pxy = TheNamedObjMap::Instance().getData<LatticePropagator>(params.named_obj.prop_xy_id);

        TheNamedObjMap::Instance().get(params.named_obj.prop_xy_id).getFileXML(prop_xy_file_xml);
        TheNamedObjMap::Instance().get(params.named_obj.prop_xy_id).getRecordXML(prop_xy_record_xml);
      }
      catch (std::bad_cast)
      {
        QDPIO::cerr << name << ": caught dynamic cast error reading Pxy" << std::endl;
        QDP_abort(1);
      }
      catch (const std::string& e)
      {
        QDPIO::cerr << name << ": error reading Pxy: " << e << std::endl;
        QDP_abort(1);
      }

      //
      // Read Pyz from the named object map
      //
      XMLReader prop_yz_file_xml, prop_yz_record_xml;
      LatticePropagator Pyz;

      QDPIO::cout << name << ": reading Pyz from " << params.named_obj.prop_yz_id << std::endl;
      try
      {
        Pyz = TheNamedObjMap::Instance().getData<LatticePropagator>(params.named_obj.prop_yz_id);

        TheNamedObjMap::Instance().get(params.named_obj.prop_yz_id).getFileXML(prop_yz_file_xml);
        TheNamedObjMap::Instance().get(params.named_obj.prop_yz_id).getRecordXML(prop_yz_record_xml);
      }
      catch (std::bad_cast)
      {
        QDPIO::cerr << name << ": caught dynamic cast error reading Pyz" << std::endl;
        QDP_abort(1);
      }
      catch (const std::string& e)
      {
        QDPIO::cerr << name << ": error reading Pyz: " << e << std::endl;
        QDP_abort(1);
      }

      //t_srce should be named as the current insertion position
      const multi1d<int>& coord = params.named_obj.t_srce;

      Propagator srcProp = peekSite(Pyz, coord);

      ColorMatrix Ic;
      Ic = 1;

          
      int curr = params.currents[0];

      /*The pion matrix element <pi+(x)|Gamma_1*Gamma_2(y)|pi-(z)>:ME
       *Indices: Gamma_1:G1[delta,sigma], Gamma_2:G2[gamma,rho]
       *ME = G1[delta,sigma]*G2[gamma,rho] *{[B1[rho,delta,c,b]*B2[sigma,gamma,b,c] - B1[sigma,delta,c,c]*B2[rho,gamma,d,d] +\
       *                                     [rho<->sigma,delta<->gamma]}
       *the flipped indices contributions are not compute, it is easier to flip indices on G1 and G2
       *T1=B1[rho,delta,c,b]*B2[sigma,gamma,b,c]
       *T3=B1[sigma,delta,c,c]*B2[rho,gamma,d,d] 
       *B1=P^d(yz)*P^u_dagger(yz)*g5
       *B2=g5*P^d_dagger(xy)*P^u(xy)
       *We will store results into 16 propagators with indices. The spin indices for the propagator are sigma,delta and the 16
       *propagators will be tag for rho, gamma
       */

      Propagator        B1 = srcProp * adj(srcProp) * Gamma(15); //Block 1 from source to current insertion
      LatticePropagator B2   = Gamma(15) * adj(Pxy) * Pxy;         //Block 2 from current insertion to sink

      SpinMatrix        T3_yz = traceColor(B1);
      LatticeSpinMatrix T3_xy = traceColor(B2);
      LatticeSpinMatrix T1,T3;
      LatticePropagator prop_out;
      LatticeComplex tmpCplx;
 
      for (int s1 = 0; s1 < Ns; ++s1) //rho loop
        {
          for (int s2 = 0; s2 < Ns; ++s2){ //gamma 

            for (int s3 = 0; s3 < Ns; ++s3){ //sigma 
               for (int s4 = 0; s4 < Ns; ++s4)//delta
              {
                  tmpCplx=traceColor(peekSpin( B1,s1,s4)*peekSpin(B2, s3, s2));
       		  pokeSpin(T1,tmpCplx,s3,s4);	           
	      }
            }


            T3 = T3_yz* peekSpin(T3_xy, s1, s2)+T3_xy* peekSpin(T3_yz, s1, s2); //The second has the flipped indices
            prop_out = (T1 - T3) * Ic;

            std::ostringstream key;
            key << params.named_obj.result_id
                << "_curr_" << curr
                << "_s1_"   << s1
                << "_s2_"   << s2;
            std::string prop_key = key.str();

            XMLBufferWriter record_xml;
            push(record_xml, "FourQBlock");
            write(record_xml, "current", curr);
            write(record_xml, "s1", s1);

            write(record_xml, "s2", s2);
            pop(record_xml);

            TheNamedObjMap::Instance().create<LatticePropagator>(prop_key);
            TheNamedObjMap::Instance().getData<LatticePropagator>(prop_key) = prop_out;
            TheNamedObjMap::Instance().get(prop_key).setFileXML(prop_xy_file_xml);
            TheNamedObjMap::Instance().get(prop_key).setRecordXML(record_xml);

            if (params.write_files)
            {
              XMLBufferWriter file_xml;
              push(file_xml, "FileXML"); pop(file_xml);

              std::string fname = prop_key + ".lime";
              QDPFileWriter to(file_xml, fname, QDPIO_SINGLEFILE, QDPIO_SERIAL, QDPIO_OPEN);
              write(to, record_xml, prop_out);
              close(to);

              TheNamedObjMap::Instance().erase(prop_key);
              QDPIO::cout << name << ": wrote and erased " << fname << std::endl;
            }
            else
            {
              QDPIO::cout << name << ": stored " << prop_key << std::endl;
            }
          }
        }

      QDPIO::cout << name << ": ran successfully" << std::endl;

      END_CODE();
    }

  }

}
