/*! \file
 *  \brief Solve a MdagM*psi=chi linear system by CG2
 */

#include "actions/ferm/invert/syssolver_mdagm_factory.h"
#include "actions/ferm/invert/syssolver_mdagm_aggregate.h"
#include "actions/ferm/invert/quda_solvers/syssolver_quda_clover_params.h"
#include "actions/ferm/invert/quda_solvers/syssolver_mdagm_exp_clover_quda_w.h"
#include "io/aniso_io.h"


#include "handle.h"
#include "actions/ferm/fermstates/periodic_fermstate.h"
#include "actions/ferm/linop/lwldslash_w.h"
#include "meas/glue/mesplq.h"
// QUDA Headers
#include <quda.h>
// #include <util_quda.h>


//#undef BUILD_QUDA_DEVIFACE_GAUGE
//#undef BUILD_QUDA_DEVIFACE_SPINOR
//#undef BUILD_QUDA_DEVIFACE_CLOVER


namespace Chroma
{
  namespace MdagMSysSolverQUDAExpCloverEnv
  {

    //! Anonymous namespace
    namespace
    {
      //! Name to be used
      const std::string name("QUDA_EXP_CLOVER_INVERTER");

      //! Local registration flag
      bool registered = false;
    }



    MdagMSystemSolver<LatticeFermion>* createFerm(XMLReader& xml_in,	
						  const std::string& path,
						  Handle< FermState< LatticeFermion, multi1d<LatticeColorMatrix>, multi1d<LatticeColorMatrix> > > state, 
						  
						  Handle< LinearOperator<LatticeFermion> > A)
    {
      return new MdagMSysSolverQUDAExpClover(A, state,SysSolverQUDACloverParams(xml_in, path));
    }

    //! Register all the factories
    bool registerAll() 
    {
      bool success = true; 
      if (! registered)
      {
	success &= Chroma::TheMdagMFermSystemSolverFactory::Instance().registerObject(name, createFerm);
	registered = true;
      }
      return success;
    }
  }

  SystemSolverResults_t 
  MdagMSysSolverQUDAExpClover::qudaInvert(const ExpCloverTermT<T, U>& clover,
				       const ExpCloverTermT<T, U>& invclov,
				       const T& chi_s,
				       T& psi_s) const{

    SystemSolverResults_t ret;

    void *spinorIn;
    void *spinorOut;

#ifdef BUILD_QUDA_DEVIFACE_SPINOR
    std::vector<QDPCache::ArgKey> ids;
#endif
  
      // Solve A_oo - D A^{-1}_ee D -- chroma conventions.
      // No need to transform source
#ifndef BUILD_QUDA_DEVIFACE_SPINOR
      spinorIn =(void *)&(chi_s.elem(rb[1].start()).elem(0).elem(0).real());
#else
      //spinorIn = GetMemoryPtr( chi_s.getId() );
      ids.push_back(chi_s.getId());
#endif
  
#ifndef BUILD_QUDA_DEVIFACE_SPINOR
    spinorOut =(void *)&(psi_s.elem(rb[1].start()).elem(0).elem(0).real());
#else
    ids.push_back(psi_s.getId());
    auto dev_ptr = GetMemoryPtr(ids);
    spinorIn  = dev_ptr[0];
    spinorOut = dev_ptr[1];
    //void* spinorOut = GetMemoryPtr( psi_s.getId() );
#endif

    // Do the solve here 
    StopWatch swatch1; 
    swatch1.reset();
    swatch1.start();
    invertQuda(spinorOut, spinorIn, (QudaInvertParam*)&quda_inv_param);
    swatch1.stop();



    QDPIO::cout << "QUDA_"<<solver_string<<"_EXP_CLOVER_SOLVER: time="<< quda_inv_param.secs <<" s" ;
    QDPIO::cout << "\tPerformance="<<  quda_inv_param.gflops/quda_inv_param.secs<<" GFLOPS" ; 
    QDPIO::cout << "\tTotal Time (incl. load gauge)=" << swatch1.getTimeInSeconds() <<" s"<<std::endl;

    ret.n_count =quda_inv_param.iter;

    return ret;

  }
  

}

