#include "chromabase.h"
#include "chroma.h"
#include "handle.h"
#include <cmath>
#include <fstream>
#include <iostream>

#include "util/gauge/reunit.h"
#include "gtest/gtest.h"

#include "actions/ferm/fermacts/clover_fermact_params_w.h"
#include "actions/ferm/fermstates/simple_fermstate.h"
//#include "actions/ferm/linop/clover_term_qdp_w.h"
//#include "actions/ferm/linop/exp_clover_term_qdp_w.h"
#include "actions/ferm/linop/unprec_clover_linop_w.h"
#include "actions/ferm/linop/unprec_exp_clover_linop_w.h"
#include "actions/ferm/linop/eoprec_exp_clover_linop_w.h"

// QUDA Headers
//#include <quda.h>
// #include <util_quda.h>

using namespace Chroma;
using namespace QDP;


template <typename TestType>
class ExpCloverFixtureT : public TestType
{
public:
  using T = LatticeFermion;
  using Q = multi1d<LatticeColorMatrix>;
  using P = multi1d<LatticeColorMatrix>;


  void SetUp()
  {
    u.resize(Nd);

    
    for (int mu = 0; mu < Nd; ++mu)
    {
      gaussian(u[mu]);
      reunit(u[mu]);
      // u[mu] = 1;
    }


    multi1d<int> bcs(4);
    bcs[0] = bcs[1] = bcs[2] = 1;
    bcs[3] = -1;
    simpleFermState = new SimpleFermState<T, P, Q>(bcs, u);

    CloverFermActParams p;
    p.Mass = Real(Mass);
    p.clovCoeffR = 1;
    p.clovCoeffT = 1;
    p.u0 = 1;
    p.anisoParam.anisoP = false;
    p.anisoParam.t_dir = 3;
    p.anisoParam.xi_0 = Real(1);
    p.twisted_m = 0;
    p.twisted_m_usedP = false;

    clov.create(simpleFermState, p);
    eclov.create(simpleFermState, p);
    //inv_eclov.create(simpleFermState, p);
    //invclov.create(simpleFermState, p);


#ifndef QDP_IS_QDPJIT
    inv_eclov.create(simpleFermState, p);
    inv_eclov.choles(0);
#else
    inv_eclov.createInv(simpleFermState,p,eclov);  // make a copy
    inv_eclov.choles(0);
#endif
  }

  static constexpr double Mass = 0.1;

  void TearDown()
  {
  }

  Q u;
  LatticePropagator quark_propagator_jit;
  LatticePropagator quark_propagator_qdpxx;

  Handle<FermState<T, P, Q>> simpleFermState;
  CloverTerm clov;
  ExpCloverTerm eclov;
  ExpCloverTerm inv_eclov; 
  ExpCloverTerm invclovCopy;
  CloverTerm invclov;
};

class ExpClovFixture : public ExpCloverFixtureT<::testing::Test>
{
};


TEST_F(ExpClovFixture, CheckOp)
{
  LatticeFermion src, res, res_exp, dummy, diff;
  gaussian(src);
  res = zero;
  res_exp = zero;

  // We will be going for an exponential in the end:
  //
  // so:
  //  exp(x) = (diag mass)[ 1 + E + 1/2 E^2 + .... ]
  //
  //  First test: (diag mass)[ 1 + E ] = regular clover term.
  for (int cb = 0; cb < 2; ++cb)
  {
    clov.apply(res, src, PLUS, cb);
    eclov.applyPower(dummy, src, PLUS, cb, 1);
    res_exp[rb[cb]] = src + dummy;
    res_exp[rb[cb]] *= Real(Nd + Mass);

    diff[rb[cb]] = res_exp - res;
    Double normdiff = sqrt(norm2(diff, rb[cb]) / norm2(src, rb[cb]));
    QDPIO::cout << "Diff (" << cb << ") = " << normdiff << "\n";

    ASSERT_LT(toDouble(normdiff), 1.0e-14);
  }
}

#if 0 //must be fixed for jit version
TEST_F(ExpClovFixture, CheckRefOp)
{
  LatticeFermion src, res, diff;
  src = zero;
  src.elem(0).elem(0).elem(0).real() = 1;
  eclov.fillRefDiag(0.5);

  res = zero;
  eclov.applyRef(res, src, PLUS, 2);

  double r2 = res.elem(0).elem(0).elem(0).real();

  res = zero;
  eclov.applyRef(res, src, PLUS, 10);

  double r10 = res.elem(0).elem(0).elem(0).real();

  res = zero;
  eclov.applyRef(res, src, PLUS, 13);

  double r13 = res.elem(0).elem(0).elem(0).real();

  double ref = exp(0.5);
  QDPIO::cout << "N=2   res=" << r2 << " exp(0.5)=" << ref << " abs. diff=" << abs(ref - r2)
	      << "\n";
  QDPIO::cout << "N=10   res=" << r10 << " exp(0.5)=" << ref << " abs. diff=" << abs(ref - r10)
	      << "\n";
  QDPIO::cout << "N=13   res=" << r13 << " exp(0.5)=" << ref << " abs. diff=" << abs(ref - r13)
	      << "\n";

  ASSERT_LT(abs(ref - r13), 1.0e-15);
}
#endif

TEST_F(ExpClovFixture, CheckApplyClover)
{
  LatticeFermion src, res, res2, res_exp, diff;
  gaussian(src);
  res = zero;
  res2 = zero;
 
  // A^0 = I
  for (int cb = 0; cb < 2; ++cb)
  {

    clov.apply(res, src, PLUS, cb);
    eclov.applyPower(res2, src, PLUS, cb, 1);
    res_exp[rb[cb]] = src + res2;
    res_exp[rb[cb]] *= Real(Nd + Mass);

    diff[rb[cb]] = res_exp - res;


    Double normdiff = sqrt(norm2(diff, rb[cb]) / norm2(src, rb[cb]));
    QDPIO::cout << "Diff (" << cb << ") = " << normdiff << "\n";

    ASSERT_LT(toDouble(normdiff), 1.0e-14);
  }
}

TEST_F(ExpClovFixture, CheckApplyPower0)
{
  LatticeFermion src, res, res2, res_exp, diff;
  gaussian(src);
  res = zero;
  res2 = zero;
 
  // A^0 = I
  for (int cb = 0; cb < 2; ++cb)
  {

    eclov.applyPower(res, src, PLUS, cb, 0);

    diff[rb[cb]] = res - src;

    Double normdiff = sqrt(norm2(diff, rb[cb]) / norm2(src, rb[cb]));
    QDPIO::cout << "Diff (" << cb << ") = " << normdiff << "\n";

    ASSERT_LT(toDouble(normdiff), 1.0e-14);
  }
}


TEST_F(ExpClovFixture, CheckApplyPower1)
{
  LatticeFermion src, res, res2, diff;
  gaussian(src);
  res = zero;
  res2 = zero;

  for (int cb = 0; cb < 2; ++cb)
  {
    //applyUnexp just wrapss applyPower(chi, psi, isign, cb, 1);
    eclov.applyUnexp(res, src, PLUS, cb);  
    eclov.applyPower(res2, src, PLUS, cb, 1);

    diff[rb[cb]] = res2 - res;
    Double normdiff = sqrt(norm2(diff, rb[cb]) / norm2(src, rb[cb]));
    QDPIO::cout << "Diff (" << cb << ") = " << normdiff << "\n";

    ASSERT_LT(toDouble(normdiff), 1.0e-14);
  }
}
 

TEST_F(ExpClovFixture, CheckApplyPower2)
{
  LatticeFermion src, res, res2, diff;
  gaussian(src);
  res = zero;
  res2 = zero;

  for (int cb = 0; cb < 2; ++cb)
  {
    eclov.applyPower(res, src, PLUS, cb, 1);
    eclov.applyPower(res2, res, PLUS, cb, 1);
    eclov.applyPower(res, src, PLUS, cb, 2);

    diff[rb[cb]] = res2 - res;
    Double normdiff = sqrt(norm2(diff, rb[cb]) / norm2(src, rb[cb]));
    QDPIO::cout << "Diff (" << cb << ") = " << normdiff << "\n";

    ASSERT_LT(toDouble(normdiff), 1.0e-14);
  }
}

TEST_F(ExpClovFixture, CheckApplyPower3)
{
  LatticeFermion src, res, res2, diff;
  gaussian(src);
  res = zero;
  res2 = zero;

  for (int cb = 0; cb < 2; ++cb)
  {
    eclov.applyPower(res, src, PLUS, cb, 1);
    eclov.applyPower(res2, res, PLUS, cb, 1);
    eclov.applyPower(res, res2, PLUS, cb, 1);

    eclov.applyPower(res2, src, PLUS, cb, 3);

    diff[rb[cb]] = res2 - res;
    Double normdiff = sqrt(norm2(diff, rb[cb]) / norm2(src, rb[cb]));
    QDPIO::cout << "Diff (" << cb << ") = " << normdiff << "\n";

    ASSERT_LT(toDouble(normdiff), 1.0e-14);
  }
}


TEST_F(ExpClovFixture, CheckApplyPower4)
{
  LatticeFermion src, res, res2, diff;
  gaussian(src);
  res = zero;
  res2 = zero;

  for (int cb = 0; cb < 2; ++cb)
  {
    eclov.applyPower(res, src, PLUS, cb, 1);
    eclov.applyPower(res2, res, PLUS, cb, 1);
    eclov.applyPower(res, res2, PLUS, cb, 1);
    eclov.applyPower(res2, res, PLUS, cb, 1);

    eclov.applyPower(res, src, PLUS, cb, 4);

    diff[rb[cb]] = res2 - res;
    Double normdiff = sqrt(norm2(diff, rb[cb]) / norm2(src, rb[cb]));
    QDPIO::cout << "Diff (" << cb << ") = " << normdiff << "\n";

    ASSERT_LT(toDouble(normdiff), 1.0e-14);
  }
}

TEST_F(ExpClovFixture, CheckApplyPower5)
{
  LatticeFermion src, res, res2, diff;
  gaussian(src);
  res = zero;
  res2 = zero;

  for (int cb = 0; cb < 2; ++cb)
  {
    eclov.applyPower(res, src, PLUS, cb, 1);
    eclov.applyPower(res2, res, PLUS, cb, 1);
    eclov.applyPower(res, res2, PLUS, cb, 1);
    eclov.applyPower(res2, res, PLUS, cb, 1);
    eclov.applyPower(res, res2, PLUS, cb, 1);

    eclov.applyPower(res2, src, PLUS, cb, 5);

    diff[rb[cb]] = res2 - res;
    Double normdiff = sqrt(norm2(diff, rb[cb]) / norm2(src, rb[cb]));
    QDPIO::cout << "Diff (" << cb << ") = " << normdiff << "\n";

    ASSERT_LT(toDouble(normdiff), 1.0e-14);
  }
}

TEST_F(ExpClovFixture, CheckApplyPower6)
{
  LatticeFermion src, res, res2, diff;
  gaussian(src);
  res = zero;
  res2 = zero;

  for (int cb = 0; cb < 2; ++cb)
  {

    eclov.applyPower(res, src, PLUS, cb, 1);
    eclov.applyPower(res2, res, PLUS, cb, 1);
    eclov.applyPower(res, res2, PLUS, cb, 1);
    eclov.applyPower(res2, res, PLUS, cb, 1);
    eclov.applyPower(res, res2, PLUS, cb, 1);
    eclov.applyPower(res2, res, PLUS, cb, 1);

    eclov.applyPower(res, src, PLUS, cb, 6);

    diff[rb[cb]] = res2 - res;
    Double normdiff = sqrt(norm2(diff, rb[cb]) / norm2(src, rb[cb]));
    QDPIO::cout << "Diff (" << cb << ") = " << normdiff << "\n";

    ASSERT_LT(toDouble(normdiff), 1.0e-14);
  }
}



TEST_F(ExpClovFixture, CheckApplyPower7)
{
  LatticeFermion src, res, res2, diff;
  gaussian(src);
  res = zero;
  res2 = zero;

  for (int cb = 0; cb < 2; ++cb)
  {

    eclov.applyPower(res, src, PLUS, cb, 1);
    eclov.applyPower(res2, res, PLUS, cb, 1);
    eclov.applyPower(res, res2, PLUS, cb, 1);
    eclov.applyPower(res2, res, PLUS, cb, 1);
    eclov.applyPower(res, res2, PLUS, cb, 1);
    eclov.applyPower(res2, res, PLUS, cb, 1);
    eclov.applyPower(res, res2, PLUS, cb, 1);

    eclov.applyPower(res2, src, PLUS, cb, 7);

    diff[rb[cb]] = res2 - res;
    Double normdiff = sqrt(norm2(diff, rb[cb]) / norm2(src, rb[cb]));
    QDPIO::cout << "Diff (" << cb << ") = " << normdiff << "\n";

    ASSERT_LT(toDouble(normdiff), 1.0e-14);
  }
}
 



TEST_F(ExpClovFixture, CheckApplyInv)
{
  LatticeFermion src, res, res2, dummy,diff;
  gaussian(src);
  res = zero;
  res2 = zero;
  dummy=zero;

  // We will be going for an exponential in the end:
  //
  // so:
  //  exp(x) = (diag mass)[ 1 + E + 1/2 E^2 + .... ]
  //
  //  First test: (diag mass)[ 1 + E ] = regular clover term.


  for (int cb = 0; cb < 2; ++cb)
  {
      eclov.apply(res, src, PLUS, cb);
      eclov.applyInv(res2, res, PLUS, cb);
  }
  diff = res2-src;
  Double normdiff = sqrt(norm2(diff) / norm2(src));
  QDPIO::cout << "Diff  = " << normdiff << "\n";

  ASSERT_LT(toDouble(normdiff), 1.0e-14);

}

#if 0 //must be fixed for jit exp-clover
TEST_F(ExpClovFixture, CheckApplyExpClov)
{
  LatticeFermion src, res, res2, dummy, diff;
  gaussian(src);
  res = zero;
  res2 = zero;

  // We will be going for an exponential in the end:
  //
  // so:
  //  exp(x) = (diag mass)[ 1 + E + 1/2 E^2 + .... ]
  //
  //  First test: (diag mass)[ 1 + E ] = regular clover term.
  eclov.makeExpClov(PLUS,0,0);
  eclov.makeExpClov(PLUS,1,0);

  //eclov.printExpClov();

  for (int cb = 0; cb < 2; ++cb)
  {
    eclov.apply(res2, src, PLUS, cb);

    eclov.applyExpClov(dummy, src, PLUS, cb);
    res[rb[cb]] = dummy/Real(Nd + Mass);
  }

  diff = res-res2;
  Double normdiff = sqrt(norm2(diff) / norm2(src));
  QDPIO::cout << "Diff  = " << normdiff << "\n";

  ASSERT_LT(toDouble(normdiff), 1.0e-14);
  }
#endif 

#if 0 //must be fixed fot jit version
TEST_F(ExpClovFixture, CheckOpExpClov)
{
  LatticeFermion src, res, res_exp, dummy, diff;
  gaussian(src);
  res = zero;
  res_exp = zero;

  // We will be going for an exponential in the end:
  //
  // so:
  //  exp(x) = (diag mass)[ 1 + E + 1/2 E^2 + .... ]
  //
  //  First test: (diag mass)[ 1 + E ] = regular clover term.

  eclov.makeExpClov(PLUS,0,0);
  eclov.makeExpClov(PLUS,1,0);

  eclov.makeExpClov(MINUS,0,0);
  eclov.makeExpClov(MINUS,1,0);

  for (int cb = 0; cb < 2; ++cb)
  {
    clov.apply(res, src, PLUS, cb);
    eclov.applyExpClov(dummy, src, PLUS, cb);

    //eclov.apply(dummy, src, PLUS, cb);

    res_exp[rb[cb]] = dummy;
    //res_exp[rb[cb]] *= Real(Nd + Mass);

    QDPIO::cout << " Real(Nd + Mass) = " << Real(Nd + Mass) << " \n";

    diff[rb[cb]] = res_exp-res;
    Double normdiff = sqrt(norm2(diff, rb[cb]) / norm2(src, rb[cb]));
    //normdiff = sqrt(norm2(diff, rb[cb]));

    QDPIO::cout << "Diff (" << cb << ") = " << normdiff << "\n";

    ASSERT_LT(toDouble(normdiff), 1.0e-14);
  }
}
#endif 

#if 0 //must be fixed for jit exp-clover
TEST_F(ExpClovFixture, CheckApplyInvExpClov)
{
  LatticeFermion src, res, res2, dummy,diff;
  gaussian(src);
  res = zero;
  res2 = zero;
  dummy=zero;

  // We will be going for an exponential in the end:
  //
  // so:
  //  exp(x) = (diag mass)[ 1 + E + 1/2 E^2 + .... ]
  //
  //  First test: (diag mass)[ 1 + E ] = regular clover term.
  eclov.makeExpClov(PLUS,0,0);
  eclov.makeExpClov(PLUS,1,0);

  inv_eclov.makeExpClov(PLUS,0,0);
  inv_eclov.makeExpClov(PLUS,1,0);


  inv_eclov.cholesTest(0);
  inv_eclov.cholesTest(1);

  //invclov.choles(0);
  //invclov.choles(1);


  for (int cb = 0; cb < 2; ++cb)
  {
    inv_eclov.applyExpClov(res, src, PLUS, cb);
    eclov.applyExpClov(res2, res, PLUS, cb);
    //invclov.apply(res2,res, PLUS, cb);
  }

  diff = src-res2;
  Double normdiff = sqrt(norm2(diff) / norm2(src));
  QDPIO::cout << "Diff  = " << normdiff << "\n";

  ASSERT_LT(toDouble(normdiff), 1.0e-14);
  }
#endif

