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
using range_policy = Kokkos::RangePolicy<ExecSpace>;

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

#endif
