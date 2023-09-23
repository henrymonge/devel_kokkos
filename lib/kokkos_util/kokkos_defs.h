// -*- C++ -*-

/*! \file
 * \brief Kokkos views definitions
 */
#include <Kokkos_Core.hpp>
#include "qdp.h"

#ifndef KOKKOS_DEFS_H
#define KOKKOS_DEFS_H

using namespace QDP;

#if Kokkos_ENABLE_HIP == 1
using MemorySpace=Kokkos::Experimental::HIP;
using Kokkos_Layout = Kokkos::LayoutLeft;
#endif

#if Kokkos_ENABLE_OPENMP == 1
using MemorySpace = Kokkos::OpenMP;
using Kokkos_Layout = Kokkos::LayoutRight;
#endif

using ExecSpace = MemorySpace::execution_space;
//using range_policy = Kokkos::RangePolicy<ExecSpace>;

using range_policy = Kokkos::RangePolicy<ExecSpace,Kokkos::LaunchBounds<128,1>>;


//template<typename T>
//typedef Kokkos::View<WordType<LatticePropagator> * [4][4][3][3][2],Kokkos_Layout,MemorySpace,Kokkos::MemoryTraits<Kokkos::Unmanaged>> QDPViewType;
//Kokkos::View<T* [4][4][3][3][2],Kokkos_Layout,MemorySpace,Kokkos::MemoryTraits<Kokkos::Unmanaged>> QDPViewType;
//using QDPViewType = Kokkos::View<T*[Ns][Ns][Nc][Nc][2],Kokkos_Layout,MemorySpace,Kokkos::MemoryTraits<Kokkos::Unmanaged>
using FType=typename WordType<LatticePropagator>::Type_t;
using WordLatticeComplexType=typename WordType<LatticeComplex>::Type_t;
//using WordMulti1dIntType =WordType<multi1d<int>>::Type_t;


typedef Kokkos::View<FType * [4][4][3][3][2],Kokkos_Layout,MemorySpace,Kokkos::MemoryTraits<Kokkos::Unmanaged>> QDPViewType;
typedef Kokkos::View<WordLatticeComplexType * [2],Kokkos_Layout,MemorySpace,Kokkos::MemoryTraits<Kokkos::Unmanaged>> QDPLattComplexViewType;
typedef Kokkos::View<int *,Kokkos::LayoutRight,Kokkos::HostSpace,Kokkos::MemoryTraits<Kokkos::Unmanaged>> View_int_1d_Unmag;


//typedef Kokkos::View<WordMulti1dIntType * ,Kokkos_Layout,MemorySpace,Kokkos::MemoryTraits<Kokkos::Unmanaged>> QDPMulti1dIntViewType;

typedef Kokkos::View<Kokkos::complex<double> * [4][4][3][3],Kokkos_Layout,MemorySpace>  View_prop_type;
typedef Kokkos::View<Kokkos::complex<double> [4][4][3][3],Kokkos_Layout,MemorySpace>  View_site_prop_type;
typedef Kokkos::View<Kokkos::complex<double> [4][4][3][3],Kokkos_Layout,MemorySpace>  View_site_prop_typeB;
typedef Kokkos::View<QDP::RComplex<double> * [4][4][3][3],Kokkos_Layout,MemorySpace>  View_QDPD_prop_type;


typedef Kokkos::View<int *,Kokkos_Layout,MemorySpace>  View_LatticeInteger;
typedef Kokkos::View<int *[4],Kokkos_Layout,MemorySpace>  View_LatticeIntegerPos;


typedef Kokkos::View<Kokkos::complex<double> **,Kokkos_Layout,MemorySpace>  View_LatticeComplex1d;
typedef Kokkos::View<Kokkos::complex<double> *,Kokkos_Layout,MemorySpace>  View_LatticeComplex;


typedef Kokkos::View<Kokkos::complex<double> * [16][4][4][3][3],Kokkos_Layout,MemorySpace>  View_prop_gamma_type;
typedef Kokkos::View<Kokkos::complex<double> ** [4][4][3][3],Kokkos_Layout,MemorySpace>  View_prop_shifted_type;
typedef Kokkos::View<Kokkos::complex<double> * [16][4][4],Kokkos_Layout,MemorySpace>  View_spin_gamma_type;
typedef Kokkos::View<Kokkos::complex<double> *[16],Kokkos_Layout,MemorySpace> View_corrs_type;
typedef Kokkos::View<Kokkos::complex<double> *,Kokkos_Layout,MemorySpace> View_corr_type;
typedef Kokkos::View<Kokkos::complex<double> **,Kokkos_Layout,MemorySpace> View_corr_array_type;
typedef Kokkos::View<Kokkos::complex<double> [3][3],Kokkos_Layout,MemorySpace> View_corr_color_type;
typedef Kokkos::View<Kokkos::complex<double> [3][3],Kokkos_Layout,MemorySpace> View_color_matrix_type;

typedef Kokkos::View<Kokkos::complex<double> ** [3][3],Kokkos_Layout,MemorySpace> View_color_matrix_arr_type;
typedef Kokkos::View<Kokkos::complex<double> * [3][3],Kokkos_Layout,MemorySpace> View_Latt_color_matrix_type;





typedef Kokkos::View<Kokkos::complex<double> [4][4],Kokkos_Layout,MemorySpace> View_spin_matrix_type;
typedef Kokkos::View<Kokkos::complex<double>* [4][4],Kokkos_Layout,MemorySpace> View_Latt_spin_matrix_type;
typedef Kokkos::View<int *[4],Kokkos_Layout,MemorySpace> View_Lattice_coords;
typedef Kokkos::View<int ****,Kokkos_Layout,MemorySpace> View_Lattice_int;
typedef Kokkos::View<int *,Kokkos_Layout,MemorySpace> View_int_1d;
typedef Kokkos::View<int **,Kokkos_Layout,MemorySpace> View_int_2d;
typedef Kokkos::View<Kokkos::complex<double> ***,Kokkos_Layout,MemorySpace> View_LatticeComplex_2d;
typedef Kokkos::View<int ***,Kokkos_Layout,MemorySpace> View_int_3d;



typedef Kokkos::View<int **,Kokkos_Layout,MemorySpace> View_int_arr2d;

KOKKOS_INLINE_FUNCTION Kokkos::complex<double> QDPtoKokkosComplex(const QDP::RComplex<double> c1)
{
  Kokkos::complex<double> d;
  d.real()=c1.real();
  d.imag() = c1.imag();
  return d;
}

KOKKOS_INLINE_FUNCTION Kokkos::complex<double> QDPtoKokkosComplex(const QDP::RComplex<QDP::Word<double>> c)
{
  Kokkos::complex<double> d;
  d.real()=c.real().elem();
  d.imag() = c.imag().elem();
  return d;
}


KOKKOS_INLINE_FUNCTION void QDPSpMatrixToKokkosSpMatrix(View_spin_matrix_type sout, SpinMatrix sin)
{

  for(int i=0; i <4;i++){
    for(int j=0; j<4;j++){
       //sout(i,j)=QDPtoKokkosComplex(sin.elem(0).elem(i,j).elem());
       sout(i,j)=QDPtoKokkosComplex(sin.elem().elem(i,j).elem());

    }
  }
}

KOKKOS_INLINE_FUNCTION void QDPSpMatrixToKokkosSpMatrix(auto sout, SpinMatrix sin)
{

  for(int i=0; i <4;i++){
    for(int j=0; j<4;j++){
       //sout(i,j)=QDPtoKokkosComplex(sin.elem(0).elem(i,j).elem());
       sout(i,j)=QDPtoKokkosComplex(sin.elem().elem(i,j).elem());

    }
  }
}

struct ComplexHistogram {
  // In this case, the reduction result is an array of float.
  using value_type = Kokkos::complex<double>[];

  //using value_type = View_LatticeComplex;
  using size_type = View_LatticeComplex::size_type;

  // Tell Kokkos the result array's number of entries.
  // This must be a public value in the functor.
  size_type value_count;

  View_LatticeComplex X_;
  Kokkos::View<int *> T_;

  // As with the above examples, you may supply an
  // execution_space typedef. If not supplied, Kokkos
  // will use the default execution space for this functor.

  // Be sure to set value_count in the constructor.
  ComplexHistogram(const View_LatticeComplex& X, int const size,const Kokkos::View<int *>& T )
      : value_count(size),  // # columns in X
        X_(X),T_(T) {}

  // value_type here is already a "reference" type,
  // so we don't pass it in by reference here.
  KOKKOS_INLINE_FUNCTION void operator()(const size_type i,
                                         value_type sum) const {
    // You may find it helpful to put pragmas above this loop
    // to convince the compiler to vectorize it. This is
    // probably only helpful if the View type has LayoutRight.
      sum[T_(i)] = sum[T_(i)]+X_(i);
  }

  // value_type here is already a "reference" type,
  // so we don't pass it in by reference here.
  KOKKOS_INLINE_FUNCTION void join(value_type dst, const value_type src) const {
    for (size_type j = 0; j < value_count; ++j) {
      dst[j] += src[j];
    }
  }

  KOKKOS_INLINE_FUNCTION void init(value_type sum) const {
    for (size_type j = 0; j < value_count; ++j) {
      sum[j] = 0.0;
    }
  }
};

struct Kokkos_sft {
  // In this case, the reduction result is an array of float.
  using value_type = Kokkos::complex<double>[];

  //using value_type = View_LatticeComplex;
  using size_type = View_LatticeComplex::size_type;

  // Tell Kokkos the result array's number of entries.
  // This must be a public value in the functor.
  size_type value_count;

  View_LatticeComplex X_;
  View_LatticeComplex1d P_;
  Kokkos::View<int *> T_;
  int mom_num_;
  // As with the above examples, you may supply an
  // execution_space typedef. If not supplied, Kokkos
  // will use the default execution space for this functor.

  // Be sure to set value_count in the constructor.
  Kokkos_sft(const View_LatticeComplex& X, int const size,const Kokkos::View<int *>& T,const View_LatticeComplex1d& P,int const mom_num )
      : value_count(size),  // # columns in X
        X_(X),T_(T),P_(P),mom_num_(mom_num) {}

  // value_type here is already a "reference" type,
  // so we don't pass it in by reference here.
  KOKKOS_INLINE_FUNCTION void operator()(const size_type i,
                                         value_type sum) const {
    // You may find it helpful to put pragmas above this loop
    // to convince the compiler to vectorize it. This is
    // probably only helpful if the View type has LayoutRight.
      sum[T_(i)] = sum[T_(i)]+X_(i)*P_(mom_num_,i);
  }

  // value_type here is already a "reference" type,
  // so we don't pass it in by reference here.
  KOKKOS_INLINE_FUNCTION void join(value_type dst, const value_type src) const {
    for (size_type j = 0; j < value_count; ++j) {
      dst[j] += src[j];
    }
  }

  KOKKOS_INLINE_FUNCTION void init(value_type sum) const {
    for (size_type j = 0; j < value_count; ++j) {
      sum[j] = 0.0;
    }
  }
};

/*
struct Kokkos_sft_multi {
  // In this case, the reduction result is an array of float.
  using value_type = Kokkos::complex<double>[][];

  //using value_type = View_LatticeComplex;
  using size_type = View_LatticeComplex1d::size_type;

  // Tell Kokkos the result array's number of entries.
  // This must be a public value in the functor.
  size_type value_count;

  View_LatticeComplex1d X_;
  View_LatticeComplex P_;
  Kokkos::View<int *> T_;

  // As with the above examples, you may supply an
  // execution_space typedef. If not supplied, Kokkos
  // will use the default execution space for this functor.

  // Be sure to set value_count in the constructor.
  Kokkos_sft(const View_LatticeComplex1d& X, int const size,const Kokkos::View<int *>& T,const View_LatticeComplex& P )
      : value_count(size),  // # columns in X
        X_(X),T_(T),P_(P) {}

  // value_type here is already a "reference" type,
  // so we don't pass it in by reference here.
  KOKKOS_INLINE_FUNCTION void operator()(const size_type i,
                                         value_type sum) const {
    // You may find it helpful to put pragmas above this loop
    // to convince the compiler to vectorize it. This is
    // probably only helpful if the View type has LayoutRight.
       for (size_type j = 0; j < value_count; ++j) {
         sum[T_(i,j)] = sum[T_(i,j)]+X_(i,j)*P_(i);
       }
  }

  // value_type here is already a "reference" type,
  // so we don't pass it in by reference here.
  KOKKOS_INLINE_FUNCTION void join(value_type dst, const value_type src) const {
    for (size_type j = 0; j < value_count; ++j) {
      for (size_type i = 0; i < value_count; ++i) {
          dst[j][i] += src[j][i];
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void init(value_type sum) const {
    for (size_type j = 0; j < value_count; ++j) {
      for (size_type i = 0; i < value_count; ++i) {
         sum[j][i] = 0.0;
      }
    }
  }
};
*/

#endif
