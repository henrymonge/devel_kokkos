/*! \file
 *  \brief Unit tests for SUMMED_FOURQ_BLOCK inline measurement
 *
 *  Verifies that the measurement passes the propagator through unchanged and
 *  correctly reads t_srce / curr_insertion spacetime positions from XML.
 */

#include "chroma.h"
#include "meas/inline/hadron/inline_summed_fourq_block_w.h"
#include "meas/inline/abs_inline_measurement_factory.h"
#include "meas/inline/io/named_objmap.h"

using namespace Chroma;

namespace
{
  // Helper: store a LatticePropagator in the named object map with empty XML metadata
  void storeInMap(const std::string& id, const LatticePropagator& prop)
  {
    XMLBufferWriter file_xml, record_xml;
    push(file_xml,   "test_file");   pop(file_xml);
    push(record_xml, "test_record"); pop(record_xml);

    TheNamedObjMap::Instance().create<LatticePropagator>(id);
    TheNamedObjMap::Instance().getData<LatticePropagator>(id) = prop;
    TheNamedObjMap::Instance().get(id).setFileXML(file_xml);
    TheNamedObjMap::Instance().get(id).setRecordXML(record_xml);
  }

  // Helper: run measurement and return norm2 of (result - ref)
  double runAndCheck(const std::string& prop_xy_id,
                     const std::string& prop_yz_id,
                     const std::string& result_id,
                     const multi1d<int>& t_srce,
                     const multi1d<int>& curr_insertion,
                     const LatticePropagator& ref,
                     XMLWriter& xml_out)
  {
    InlineSummedFourQBlockEnv::Params p;
    p.frequency                   = 1;
    p.named_obj.t_srce            = t_srce;
    p.named_obj.curr_insertion    = curr_insertion;
    p.named_obj.prop_xy_id        = prop_xy_id;
    p.named_obj.prop_yz_id        = prop_yz_id;
    p.named_obj.result_id         = result_id;

    InlineSummedFourQBlockEnv::InlineMeas meas(p);
    meas(0, xml_out);

    const LatticePropagator& result =
      TheNamedObjMap::Instance().getData<LatticePropagator>(result_id);

    double err = toDouble(norm2(result - ref));

    TheNamedObjMap::Instance().erase(result_id);
    return err;
  }

  multi1d<int> makeCoord(int x, int y, int z, int t)
  {
    multi1d<int> c(Nd);
    c[0] = x; c[1] = y; c[2] = z; c[3] = t;
    return c;
  }
}

int main(int argc, char* argv[])
{
  Chroma::initialize(&argc, &argv);

  // Small 4^4 lattice
  const int dims[] = {4, 4, 4, 4};
  multi1d<int> nrow(Nd);
  nrow = dims;
  Layout::setLattSize(nrow);
  Layout::create();

  InlineSummedFourQBlockEnv::registerAll();

  XMLFileWriter xml_out("t_summed_fourq_block_w.xml");
  push(xml_out, "t_summed_fourq_block_w");

  push(xml_out, "lattice");
  write(xml_out, "Nd", Nd);
  write(xml_out, "Nc", Nc);
  write(xml_out, "Ns", Ns);
  write(xml_out, "nrow", nrow);
  pop(xml_out);

  int n_failures = 0;

  QDP::RNG::setrn(42);

  LatticePropagator prop_in;
  gaussian(prop_in);
  storeInMap("test_prop", prop_in);

  // -----------------------------------------------------------------------
  // Test 1: result must equal input (no scaling); t_srce at origin
  // -----------------------------------------------------------------------
  {
    double err = runAndCheck("test_prop", "test_prop", "test_result",
                             makeCoord(0,0,0,0), makeCoord(1,0,0,0),
                             prop_in, xml_out);
    QDPIO::cout << "TEST passthrough (origin): ||result - prop||^2 = " << err << std::endl;
    write(xml_out, "test_passthrough_origin_err", err);

    if (err > 1e-22)
    {
      QDPIO::cerr << "FAIL: result should equal input (err=" << err << ")" << std::endl;
      ++n_failures;
    }
    else
    {
      QDPIO::cout << "PASS: passthrough (origin)" << std::endl;
    }
  }

  // -----------------------------------------------------------------------
  // Test 2: result must equal input; t_srce and curr_insertion at interior sites
  // -----------------------------------------------------------------------
  {
    double err = runAndCheck("test_prop", "test_prop", "test_result",
                             makeCoord(2,1,0,3), makeCoord(3,2,1,0),
                             prop_in, xml_out);
    QDPIO::cout << "TEST passthrough (interior): ||result - prop||^2 = " << err << std::endl;
    write(xml_out, "test_passthrough_interior_err", err);

    if (err > 1e-22)
    {
      QDPIO::cerr << "FAIL: result should equal input (err=" << err << ")" << std::endl;
      ++n_failures;
    }
    else
    {
      QDPIO::cout << "PASS: passthrough (interior)" << std::endl;
    }
  }

  // -----------------------------------------------------------------------
  // Test 3: factory path  (XML round-trip with t_srce / curr_insertion)
  // -----------------------------------------------------------------------
  {
    const std::string xml_str =
      "<elem>"
      "  <Name>SUMMED_FOURQ_BLOCK</Name>"
      "  <Frequency>1</Frequency>"
      "  <NamedObject>"
      "    <t_srce>0 0 0 0</t_srce>"
      "    <curr_insertion>1 0 0 0</curr_insertion>"
      "    <prop_xy_id>test_prop</prop_xy_id>"
      "    <prop_yz_id>test_prop</prop_yz_id>"
      "    <result_id>test_result</result_id>"
      "  </NamedObject>"
      "</elem>";

    std::istringstream is(xml_str);
    XMLReader xml_elem(is);

    Handle<AbsInlineMeasurement> the_meas(
      TheInlineMeasurementFactory::Instance().createObject(
        "SUMMED_FOURQ_BLOCK", xml_elem, "/elem"));

    (*the_meas)(0, xml_out);

    const LatticePropagator& result =
      TheNamedObjMap::Instance().getData<LatticePropagator>("test_result");

    double err = toDouble(norm2(result - prop_in));
    QDPIO::cout << "TEST factory passthrough: ||result - prop||^2 = " << err << std::endl;
    write(xml_out, "test_factory_err", err);

    TheNamedObjMap::Instance().erase("test_result");

    if (err > 1e-22)
    {
      QDPIO::cerr << "FAIL: factory result mismatch (err=" << err << ")" << std::endl;
      ++n_failures;
    }
    else
    {
      QDPIO::cout << "PASS: factory passthrough" << std::endl;
    }
  }

  // -----------------------------------------------------------------------
  // Test 4: Params XML write/read round-trip
  //   Ensures t_srce and curr_insertion survive serialisation
  // -----------------------------------------------------------------------
  {
    InlineSummedFourQBlockEnv::Params p_orig;
    p_orig.frequency                = 3;
    p_orig.named_obj.t_srce         = makeCoord(1, 2, 3, 0);
    p_orig.named_obj.curr_insertion = makeCoord(0, 1, 2, 3);
    p_orig.named_obj.prop_xy_id     = "test_prop";
    p_orig.named_obj.prop_yz_id     = "test_prop";
    p_orig.named_obj.result_id      = "test_result2";

    XMLBufferWriter buf;
    push(buf, "TestParams");
    p_orig.writeXML(buf, "Params");
    pop(buf);

    std::istringstream is(buf.printCurrentContext());
    XMLReader xml_reader(is);
    InlineSummedFourQBlockEnv::Params p_read(xml_reader, "/TestParams/Params");

    bool coords_match = true;
    for (int mu = 0; mu < Nd; ++mu)
    {
      if (p_read.named_obj.t_srce[mu]         != p_orig.named_obj.t_srce[mu])         coords_match = false;
      if (p_read.named_obj.curr_insertion[mu] != p_orig.named_obj.curr_insertion[mu]) coords_match = false;
    }

    bool ok = coords_match
           && (p_read.named_obj.prop_xy_id == p_orig.named_obj.prop_xy_id)
           && (p_read.named_obj.prop_yz_id == p_orig.named_obj.prop_yz_id)
           && (p_read.named_obj.result_id  == p_orig.named_obj.result_id);

    QDPIO::cout << "TEST Params round-trip: " << (ok ? "PASS" : "FAIL") << std::endl;
    write(xml_out, "test_params_roundtrip", ok ? std::string("PASS") : std::string("FAIL"));

    if (!ok)
    {
      QDPIO::cerr << "FAIL: Params XML round-trip mismatch" << std::endl;
      ++n_failures;
    }
  }

  // -----------------------------------------------------------------------
  // Summary
  // -----------------------------------------------------------------------
  write(xml_out, "n_failures", n_failures);
  if (n_failures == 0)
  {
    QDPIO::cout << "\nALL TESTS PASSED\n" << std::endl;
    write(xml_out, "status", std::string("PASSED"));
  }
  else
  {
    QDPIO::cout << "\n" << n_failures << " TEST(S) FAILED\n" << std::endl;
    write(xml_out, "status", std::string("FAILED"));
  }

  pop(xml_out);

  Chroma::finalize();
  return n_failures;
}
