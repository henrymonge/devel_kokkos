/*! \file
 * \brief Inline measurement of SUMMED_FOURQ_BLOCK
 *
 * Reads a propagator and multiplies it by a constant c.
 */

#include "meas/inline/hadron/inline_summed_fourq_block_w.h"
#include "meas/inline/abs_inline_measurement_factory.h"
#include "util/ft/sftmom.h"
#include "meas/inline/io/named_objmap.h"
#include <sstream>

namespace Chroma
{
  //! NamedObject_t reader
  void read(XMLReader& xml, const std::string& path,
            InlineSummedFourQBlockEnv::Params::NamedObject_t& input)
  {
    XMLReader inputtop(xml, path);

    read(inputtop, "insertion_position", input.insertion_position);
    read(inputtop, "operators",          input.operators);
    read(inputtop, "prop_xy_id",         input.prop_xy_id);
    read(inputtop, "prop_yz_id",         input.prop_yz_id);
    read(inputtop, "result_id",          input.result_id);
  }

  //! NamedObject_t writer
  void write(XMLWriter& xml, const std::string& path,
             const InlineSummedFourQBlockEnv::Params::NamedObject_t& input)
  {
    push(xml, path);

    write(xml, "insertion_position", input.insertion_position);
    write(xml, "operators",          input.operators);
    write(xml, "prop_xy_id",         input.prop_xy_id);
    write(xml, "prop_yz_id",         input.prop_yz_id);
    write(xml, "result_id",          input.result_id);

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

    namespace
    {
      //! Recognized four-quark current-insertion operators
      enum FourQOperatorType
      {
        OP_VV,
        OP_AA,
        OP_SS,
        OP_PP,
        OP_UNKNOWN
      };

      //! Map an operator name string onto its enum tag for use in a switch statement
      FourQOperatorType parseFourQOperator(const std::string& op)
      {
        if (op == "VV") return OP_VV;
        if (op == "AA") return OP_AA;
        if (op == "SS") return OP_SS;
        if (op == "PP") return OP_PP;
        return OP_UNKNOWN;
      }

      //! Apply each requested operator to the full set of s1,s2 block propagators.
      //! The actual contraction for each operator is a placeholder to be filled in.
      void applyOperators(const multi1d<std::string>& operators,
                          const multi2d<LatticePropagator>& prop_outs)
      {
        for (int i = 0; i < operators.size(); ++i)
        {
          const std::string& op = operators[i];

          switch (parseFourQOperator(op))
          {
          case OP_VV:
            // TODO: Vector-Vector operator contraction
            break;

          case OP_AA:
            // TODO: Axial-Axial operator contraction
            break;

          case OP_SS:
            // TODO: Scalar-Scalar operator contraction
            break;

          case OP_PP:
            // TODO: Pseudoscalar-Pseudoscalar operator contraction
            break;

          default:
            QDPIO::cerr << name << ": unrecognized operator \"" << op << "\"" << std::endl;
            QDP_abort(1);
          }
        }
      }


      //! Apply each requested operator to the full set of s1,s2 block propagators.
      //! The actual contraction for each operator is a placeholder to be filled in.
      //contract_sigma_delta(rho_gamma_blocks,params.named_obj.operators,T1-T3,s1,s2);
      void contract_sigma_delta(multi1d<LatticeSpinMatrix>& rho_gamma_blocks,
                                          const multi1d<std::string>& operators,
                                          LatticeSpinMatrix diquark_block,
                                          int s1, int s2)
      {

        LatticeComplex result=zero;
        for (int i = 0; i < operators.size(); ++i)
        {
          const std::string& op = operators[i];

          switch (parseFourQOperator(op))
          {
          case OP_VV:
            // TODO: Vector-Vector operator contraction
            for(int mu=0;mu < Nd; ++mu){
               int gamma_index = 1 << mu;
               result += traceSpin(Gamma(gamma_index)*diquark_block);
            }
            break;

          case OP_AA:
            // TODO: Axial-Axial operator contraction
            for(int mu=0;mu < Nd; ++mu){
               int gamma_index = 1 << mu;
               result += traceSpin(Gamma(15)*(Gamma(gamma_index)*diquark_block));
            }            
            break;

          case OP_SS:
            //Scalar-Scalar operator contraction
            //Multiplies by identity, so just do the trace
            result = traceSpin(diquark_block);
            break;

          case OP_PP:
            // TODO: Pseudoscalar-Pseudoscalar operator contraction
            result = traceSpin(Gamma(15)*diquark_block);   
            break;

          default:
            QDPIO::cerr << name << ": unrecognized operator \"" << op << "\"" << std::endl;
            QDP_abort(1);
          }
          pokeSpin(rho_gamma_blocks[i],result,s1,s2);
        }
      } 

      //! Apply each requested operator to the full set of s1,s2 block propagators.
      //! The actual contraction for each operator is a placeholder to be filled in.
      //contract_sigma_delta(rho_gamma_blocks,params.named_obj.operators,T1-T3,s1,s2);
      LatticeComplex contract_rho_gamma(const std::string& op,LatticeSpinMatrix rho_gamma_block)
      {

        LatticeComplex result=zero;
 
          switch (parseFourQOperator(op))
          {
          case OP_VV:
            // TODO: Vector-Vector operator contraction
            for(int mu=0;mu < Nd; ++mu){
               int gamma_index = 1 << mu;
               result += traceSpin(Gamma(gamma_index)*rho_gamma_block);
            }
            break;

          case OP_AA:
            // TODO: Axial-Axial operator contraction
            for(int mu=0;mu < Nd; ++mu){
               int gamma_index = 1 << mu;
               result += traceSpin(Gamma(15)*(Gamma(gamma_index)*rho_gamma_block));
            }
            break;

          case OP_SS:
            //Scalar-Scalar operator contraction
            //Multiplies by identity, so just do the trace
            result = traceSpin(rho_gamma_block);
            break;

          case OP_PP:
            // TODO: Pseudoscalar-Pseudoscalar operator contraction
            result = traceSpin(Gamma(15)*rho_gamma_block);
            break;

          default:
            QDPIO::cerr << name << ": unrecognized operator \"" << op << "\"" << std::endl;
            QDP_abort(1);
          }
          return result;

      } 


      //! Extract the insertion-position list already recorded on a propagator's record XML.
      //! Returns an empty list if none is present yet (first contribution, or a record
      //! written before this bookkeeping existed).
      multi1d< multi1d<int> > readInsertionPositions(XMLReader& record_xml)
      {
        multi1d< multi1d<int> > positions;
        try
        {
          read(record_xml, "/FourQBlock/insertion_positions", positions);
        }
        catch (const std::string&)
        {
          positions.resize(0);
        }
        return positions;
      }

      //! Print an insertion-position list to stdout in a compact "[x y z t] [x y z t] ..." form
      void printPositions(const std::string& prop_key, const multi1d< multi1d<int> >& positions)
      {
        QDPIO::cout << name << ": " << prop_key << " has " << positions.size()
                    << " insertion position(s) summed so far:";
        for (int i = 0; i < positions.size(); ++i)
        {
          QDPIO::cout << " [";
          for (int mu = 0; mu < positions[i].size(); ++mu)
            QDPIO::cout << positions[i][mu] << (mu+1 < positions[i].size() ? " " : "");
          QDPIO::cout << "]";
        }
        QDPIO::cout << std::endl;
      }
    }

      bool positionAlreadySummed(const multi1d< multi1d<int> >& positions,
                                 const multi1d<int>& pos)
      {
        for (int i = 0; i < positions.size(); ++i)
        {
          if (positions[i].size() != pos.size())
            continue;

          bool same = true;
          for (int mu = 0; mu < pos.size(); ++mu)
          {
            if (positions[i][mu] != pos[mu])
            {
              same = false;
              break;
            }
          }
          if (same)
            return true;
        }
        return false;
      }

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
      const multi1d<int>& coord = params.named_obj.insertion_position;


      QDPIO::cout <<"Insertion position " << coord[0] <<" " << coord[1] << " " <<coord[2]<<" ";
      QDPIO::cout<< coord[3]<< std::endl;


      Propagator srcProp = peekSite(Pyz, coord);

      ColorMatrix Ic;
      Ic = 1;
      int nOps=params.named_obj.operators.size();

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
      multi1d<LatticeSpinMatrix> rho_gamma_blocks;

      rho_gamma_blocks.resize(nOps);

      for (int s1 = 0; s1 < Ns; ++s1) //rho loop
        {
          for (int s2 = 0; s2 < Ns; ++s2){ //gamma 

            std::ostringstream key;
            key << params.named_obj.result_id << "_s1_" << s1 << "_s2_" << s2;
            std::string prop_key = key.str();

            // Pick up the insertion positions already recorded for this key (if any).
            bool exists = TheNamedObjMap::Instance().check(prop_key);
            multi1d< multi1d<int> > old_positions;
            if (exists)
            {
              XMLReader existing_record_xml;
              TheNamedObjMap::Instance().get(prop_key).getRecordXML(existing_record_xml);
              old_positions = readInsertionPositions(existing_record_xml);
              printPositions(prop_key, old_positions);
            }

            // If this insertion position has already been summed into this key, skip it.
            if (positionAlreadySummed(old_positions, params.named_obj.insertion_position))
            {
              QDPIO::cout << name << ": " << prop_key
                          << " already contains insertion position, skipping" << std::endl;
              continue;
            }


            for (int s3 = 0; s3 < Ns; ++s3){ //sigma 
               for (int s4 = 0; s4 < Ns; ++s4)//delta
              {
                  tmpCplx=traceColor(peekSpin( B1,s1,s4)*peekSpin(B2, s3, s2));
       		  pokeSpin(T1,tmpCplx,s3,s4);	           
	          }
            }

            T3 = T3_yz* peekSpin(T3_xy, s1, s2);
            prop_out = (T1 - T3) * Ic;

            //contract_sigma_delta=
            contract_sigma_delta(rho_gamma_blocks,params.named_obj.operators,T1-T3,s1,s2);
            // Pick up the insertion positions already recorded for this key (if any),
            // and append the position used for this contribution.
            // NB: multi1d::resize() discards existing contents (it is not a realloc),
            // so the grown array is built fresh rather than resized in place.

            multi1d< multi1d<int> > positions(old_positions.size() + 1);
            for (int i = 0; i < old_positions.size(); ++i)
              positions[i] = old_positions[i];
            positions[old_positions.size()] = params.named_obj.insertion_position;

            XMLBufferWriter record_xml;
            push(record_xml, "FourQBlock");
            write(record_xml, "s1", s1);
            write(record_xml, "s2", s2);
            write(record_xml, "insertion_positions", positions);
            pop(record_xml);

            if (exists)
            {
              // Propagator already in memory - accumulate onto it
              TheNamedObjMap::Instance().getData<LatticePropagator>(prop_key) += prop_out;
              TheNamedObjMap::Instance().get(prop_key).setRecordXML(record_xml);
              QDPIO::cout << name << ": accumulated into existing " << prop_key << std::endl;
            }
            else
            {
              TheNamedObjMap::Instance().create<LatticePropagator>(prop_key);
              TheNamedObjMap::Instance().getData<LatticePropagator>(prop_key) = prop_out;
              TheNamedObjMap::Instance().get(prop_key).setFileXML(prop_xy_file_xml);
              TheNamedObjMap::Instance().get(prop_key).setRecordXML(record_xml);
            }

            // Record the running insertion-position list for this key in the output log
            push(xml_out, "block");
            write(xml_out, "prop_key", prop_key);
            write(xml_out, "insertion_positions", positions);
            pop(xml_out);

            if (params.write_files)
            {
              XMLBufferWriter file_xml;
              push(file_xml, "FileXML"); pop(file_xml);

              std::string fname = prop_key + ".lime";
              QDPFileWriter to(file_xml, fname, QDPIO_SINGLEFILE, QDPIO_SERIAL, QDPIO_OPEN);
              write(to, record_xml, TheNamedObjMap::Instance().getData<LatticePropagator>(prop_key));
              close(to);

              TheNamedObjMap::Instance().erase(prop_key);
              QDPIO::cout << name << ": wrote and erased " << fname << std::endl;
            }
            else
            {
              QDPIO::cout << name << ": stored " << prop_key << std::endl;
            }
          }//ends s2 loop
        }//ends s1 loop

      //SftMom phases_nomom(0, true, Nd -1);
      //Do the final contractions for the operator
      //  rho_gamma_block
      // corr_t[t] = sum over all spatial sites at timeslice t, no momentum phase

      //multi1d<DComplex> corr_t = sumMulti(traceSpin(rho_gamma_block), phases_nomom.getSet());


      SftMom phases_nomom(0, true, Nd - 1);

      // Do the final contractions for the operators: trace over spin and
      // sum over all spatial sites at each timeslice, no momentum phase.
      LatticeComplex tmpME;
      push(xml_out, "FourQCorrelators");
      for (int i = 0; i < params.named_obj.operators.size(); ++i)
      {
        tmpME=contract_rho_gamma(params.named_obj.operators[i],rho_gamma_blocks[i]);
        multi1d<DComplex> corr_t = sumMulti(tmpME, phases_nomom.getSet());

        push(xml_out, "elem");
        write(xml_out, "operator", params.named_obj.operators[i]);
        write(xml_out, "corr_t", corr_t);
        pop(xml_out);
      }
      pop(xml_out);


      QDPIO::cout << name << ": ran successfully" << std::endl;

      pop(xml_out);  // summed_fourq_block

      END_CODE();
    }

  }

}
