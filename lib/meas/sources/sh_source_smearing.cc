// $Id: sh_source_smearing.cc,v 2.8 2006-02-22 04:34:05 edwards Exp $
/*! \file
 *  \brief Shell source construction
 */

#include "chromabase.h"
#include "handle.h"

#include "meas/sources/source_const_factory.h"
#include "meas/sources/sh_source_smearing.h"
#include "meas/sources/source_smearing_factory.h"

#include "meas/smear/quark_smearing_aggregate.h"
#include "meas/smear/quark_smearing_factory.h"

#include "meas/smear/link_smearing_aggregate.h"
#include "meas/smear/link_smearing_factory.h"

#include "meas/smear/quark_displacement_aggregate.h"
#include "meas/smear/quark_displacement_factory.h"

namespace Chroma
{
  // Read parameters
  void read(XMLReader& xml, const string& path, ShellQuarkSourceSmearingEnv::Params& param)
  {
    ShellQuarkSourceSmearingEnv::Params tmp(xml, path);
    param = tmp;
  }

  // Writer
  void write(XMLWriter& xml, const string& path, const ShellQuarkSourceSmearingEnv::Params& param)
  {
    param.writeXML(xml, path);
  }



  //! Hooks to register the class
  namespace ShellQuarkSourceSmearingEnv
  {
    //! Callback function
    QuarkSourceSink<LatticeFermion>* createFerm(XMLReader& xml_in,
						const std::string& path,
						const multi1d<LatticeColorMatrix>& u)
    {
      return new SourceSmearing<LatticeFermion>(Params(xml_in, path), u);
    }

    //! Name to be used
    const std::string name("SHELL_SOURCE");

    //! Register all the factories
    bool registerAll()
    {
      bool foo = true;
      foo &= LinkSmearingEnv::registered;
      foo &= QuarkSmearingEnv::registered;
      foo &= QuarkDisplacementEnv::registered;
      foo &= Chroma::TheFermSourceSmearingFactory::Instance().registerObject(name, createFerm);
      return foo;
    }

    //! Register the source construction
    const bool registered = registerAll();


    //! Read parameters
    Params::Params()
    {
    }

    //! Read parameters
    Params::Params(XMLReader& xml, const string& path)
    {
      XMLReader paramtop(xml, path);

      int version;
      read(paramtop, "version", version);

      switch (version) 
      {
      case 1:
	break;

      default:
	QDPIO::cerr << __func__ << ": parameter version " << version 
		    << " unsupported." << endl;
	QDP_abort(1);
      }

      read(paramtop, "SourceType",  source_type);

      {
	XMLReader xml_tmp(paramtop, "SmearingParam");
	std::ostringstream os;
	xml_tmp.print(os);
	read(xml_tmp, "wvf_kind", quark_smearing_type);
	quark_smearing = os.str();
      }

      if (paramtop.count("Displacement") != 0)
      {
	XMLReader xml_tmp(paramtop, "Displacement");
	std::ostringstream os;
	xml_tmp.print(os);
	read(xml_tmp, "DisplacementType", quark_displacement_type);
	quark_displacement = os.str();
      }

      if (paramtop.count("LinkSmearing") != 0)
      {
	XMLReader xml_tmp(paramtop, "LinkSmearing");
	std::ostringstream os;
	xml_tmp.print(os);
	read(xml_tmp, "LinkSmearingType", link_smearing_type);
	link_smearing = os.str();
      }
    }


    // Writer
    void Params::writeXML(XMLWriter& xml, const string& path) const
    {
      push(xml, path);

      int version = 1;
      write(xml, "version", version);

      write(xml, "SourceType", source_type);
      xml << quark_smearing;
      xml << quark_displacement;
      xml << link_smearing;

      pop(xml);
    }


    //! Smear the source
    template<>
    void
    SourceSmearing<LatticeFermion>::operator()(LatticeFermion& quark_source) const
    {
//      QDPIO::cout << "Shell source smearing" << endl;
 
      try
      {
	//
	// Create the quark smearing object
	//
	std::istringstream  xml_s(params.quark_smearing);
	XMLReader  smeartop(xml_s);
	const string smear_path = "/SmearingParam";
	
	Handle< QuarkSmearing<LatticeFermion> >
	  quarkSmearing(TheFermSmearingFactory::Instance().createObject(params.quark_smearing_type,
									smeartop,
									smear_path));

	//
	// Create the quark displacement object
	//
	std::istringstream  xml_d(params.quark_displacement);
	XMLReader  displacetop(xml_d);
	const string displace_path = "/Displacement";
	
	Handle< QuarkDisplacement<LatticeFermion> >
	  quarkDisplacement(TheFermDisplacementFactory::Instance().createObject(params.quark_displacement_type,
										displacetop,
										displace_path));

	//
	// Displace and then smear quark source
	//
	(*quarkDisplacement)(quark_source, u_smr, MINUS);

	(*quarkSmearing)(quark_source, u_smr);

      }
      catch(const std::string& e) 
      {
	QDPIO::cerr << name << ": Caught Exception smearing: " << e << endl;
	QDP_abort(1);
      }
    }

  }
}
