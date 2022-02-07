// -*- C++ -*-
/*! \file                                                                    
 * \brief Copy, permuting, contracting tensors with superbblas
 *                                                                             
 * Hadron spectrum calculations utilities
 */

#ifndef __INCLUDE_SUPERB_CONTRACTIONS__
#define __INCLUDE_SUPERB_CONTRACTIONS__

#include "chromabase.h"

#ifdef BUILD_SB

// Activate the MPI support in Superbblas
#  define SUPERBBLAS_USE_MPI
//#  define SUPERBBLAS_USE_MPIIO

#  include "actions/ferm/fermacts/fermact_factory_w.h"
#  include "actions/ferm/fermacts/fermacts_aggregate_w.h"
#  include "meas/smear/link_smearing_factory.h"
#  include "qdp.h"
#  include "qdp_map_obj_disk_multiple.h"
#  include "superbblas.h"
#  include "util/ferm/key_timeslice_colorvec.h"
#  include "util/ft/sftmom.h"
#  include <algorithm>
#  include <array>
#  include <chrono>
#  include <cmath>
#  include <cstring>
#  include <iomanip>
#  include <map>
#  include <memory>
#  include <set>
#  include <sstream>
#  include <stdexcept>
#  include <string>
#  include <type_traits>

#  ifndef M_PI
#    define M_PI                                                                                   \
      3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068L
#  endif

#  ifdef BUILD_PRIMME
#    include <primme.h>
#  endif

#  if defined(QDP_IS_QDPJIT) && defined(BUILD_MAGMA)
#    include "magma_v2.h"
#  endif

namespace Chroma
{

  namespace SB
  {

    using Index = superbblas::IndexType;
    using Complex = std::complex<REAL>;
    using ComplexD = std::complex<REAL64>;
    using ComplexF = std::complex<REAL32>;
    template <std::size_t N>
    using Coor = superbblas::Coor<N>;
    using checksum_type = superbblas::checksum_type;

    /// Where to store the tensor (see class Tensor)
    enum DeviceHost {
      OnHost,	      ///< on cpu memory
      OnDefaultDevice ///< on GPU memory if possible
    };

    /// How to distribute the tensor (see class Tensor)
    enum Distribution {
      OnMaster,		    ///< Fully supported on node with index zero
      OnEveryone,	    ///< Distributed the lattice dimensions (x, y, z, t) as chroma does
      OnEveryoneReplicated, ///< All nodes have a copy of the tensor
      Local		    ///< Non-collective
    };

    /// Whether complex conjugate the elements before contraction (see Tensor::contract)
    enum Conjugation { NotConjugate, Conjugate };

    /// Whether the tensor is dense or sparse (see StorageTensor)
    enum Sparsity { Dense, Sparse };

    /// Whether to copy or add the values into the destination tensor (see Tensor::doAction)
    enum Action { CopyTo, AddTo };

     /// Auxiliary class for initialize Maybe<T> with no value
    struct None {
    };

    /// Class for optional values
    template <typename T>
    struct Maybe {
      /// opt_val.first is whether a value was set, and opt_val.second has the value if that's the case
      std::pair<bool, T> opt_val;

      /// Constructor without a value
      Maybe() : opt_val{false, {}}
      {
      }

      /// Constructor without a value
      Maybe(None) : Maybe()
      {
      }

      /// Constructor with a value
      template <typename Q>
      Maybe(Q t) : opt_val{true, T(t)}
      {
      }

      /// Return whether it has been initialized with a value
      bool hasSome() const
      {
	return opt_val.first;
      }

      /// Return the value if it has been initialized with some
      T getSome() const
      {
	if (opt_val.first)
	  return opt_val.second;
	throw std::runtime_error("W!");
      }

      /// Return the value if it has been initialized with some; otherwise return `def`
      T getSome(T def) const
      {
	if (opt_val.first)
	  return opt_val.second;
	else
	  return def;
      }
    };

    /// Initialize Maybe<T> without value
    constexpr None none = None{};

    /// Return the number of seconds from some start
    inline double w_time()
    {
      return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch())
	.count();
    }

    namespace detail
    {
      /// Throw an error if it is not a valid order, that is, if some label is repeated
      template <std::size_t N>
      void check_order(const std::string& order)
      {
	if (order.size() != N)
	{
	  std::stringstream ss;
	  ss << "The length of the dimension labels `" << order
	     << "` should match the template argument N `" << N << "`";
	  throw std::runtime_error(ss.str());
	}

	char s[256];
	for (unsigned int i = 0; i < 256; ++i)
	  s[i] = 0;
	for (unsigned int i = 0; i < N; ++i)
	{
	  if (s[order[i]] != 0)
	  {
	    std::stringstream ss;
	    ss << "Invalid order: some label names are repeated `" << order << "`";
	    throw std::runtime_error(ss.str());
	  }
	  s[order[i]] = 1;
	}
      }
    }

    enum Throw_kvcoors { NoThrow, ThrowOnUnmatchLabel, ThrowOnMissing };

    template <std::size_t N>
    Coor<N> kvcoors(const std::string& order, const std::map<char, int>& m, Index missing = 0,
		    Throw_kvcoors t = ThrowOnUnmatchLabel)
    {
      detail::check_order<N>(order);

      Coor<N> r;
      unsigned int found = 0;
      for (std::size_t i = 0; i < N; ++i)
      {
	auto it = m.find(order[i]);
	if (it != m.end())
	{
	  r[i] = it->second;
	  ++found;
	}
	else if (t == ThrowOnMissing)
	{
	  std::stringstream ss;
	  ss << "kvcoors: Missing value for label `" << order[i] << "` on dimension labels `"
	     << order << "`";
	  throw std::runtime_error(ss.str());
	}
	else
	{
	  r[i] = missing;
	}
      }

      if (found != m.size() && t == ThrowOnUnmatchLabel)
      {
	std::stringstream ss;
	ss << "kvcoors: Some dimension label on the given map m does not correspond to a dimension "
	      "label `"
	   << order << "`.";
	throw std::runtime_error(ss.str());
      }

      return r;
    }

    template <std::size_t N>
    Coor<N> latticeSize(const std::string& order, const std::map<char, int>& m = {})
    {
#  if QDP_USE_LEXICO_LAYOUT
      // No red-black ordering
      std::map<char, int> m0 = {{'x', Layout::lattSize()[0]},
				{'y', Layout::lattSize()[1]},
				{'z', Layout::lattSize()[2]},
				{'t', Layout::lattSize()[3]},
				{'X', 1},
				{'s', Ns},
				{'c', Nc},
				{'.', 2}};
#  elif QDP_USE_CB2_LAYOUT
      // Red-black ordering
      assert(Layout::lattSize()[0] % 2 == 0);
      std::map<char, int> m0 = {{'x', Layout::lattSize()[0] / 2},
				{'y', Layout::lattSize()[1]},
				{'z', Layout::lattSize()[2]},
				{'t', Layout::lattSize()[3]},
				{'X', 2},
				{'s', Ns},
				{'c', Nc},
				{'.', 2}};
#  else
      throw std::runtime_error("Unsupported layout");
#  endif
      for (const auto& it : m)
	m0[it.first] = it.second;
      return kvcoors<N>(order, m0, 0, NoThrow);
    }

    // Replace a label by another label
    using remap = std::map<char, char>;

    // Return the equivalent value of the coordinate `v` in the interval [0, dim[ for a periodic
    // dimension with length `dim`.

    inline int normalize_coor(int v, int dim)
    {
      return (v + dim * (v < 0 ? -v / dim + 1 : 0)) % dim;
    }

    // Return the equivalent value of the coordinate `v` in the interval [0, dim[ for a periodic
    // dimension with length `dim`.

    template <std::size_t N>
    Coor<N> normalize_coor(Coor<N> v, Coor<N> dim)
    {
      Coor<N> r;
      for (std::size_t i = 0; i < N; ++i)
	r[i] = normalize_coor(v[i], dim[i]);
      return r;
    }

    namespace detail
    {
      using namespace superbblas::detail;

      // Throw an error if `order` does not contain a label in `should_contain`
      inline void check_order_contains(const std::string& order, const std::string& should_contain)
      {
	for (char c : should_contain)
	{
	  if (order.find(c) == std::string::npos)
	  {
	    std::stringstream ss;
	    ss << "The input order `" << order << "` is missing the label `" << c << "`";
	    throw std::runtime_error(ss.str());
	  }
	}
      }

      // Throw an error if `order` does not contain a label in `should_contain`
      inline std::string remove_dimensions(const std::string& order, const std::string& remove_dims)
      {
	std::string out;
	out.reserve(order.size());
	for (char c : order)
	  if (remove_dims.find(c) == std::string::npos)
	    out.push_back(c);
	return out;
      }

      template <std::size_t N>
      std::string update_order(std::string order, const remap& m)
      {
	for (std::size_t i = 0; i < N; ++i)
	{
	  auto it = m.find(order[i]);
	  if (it != m.end())
	    order[i] = it->second;
	}
	check_order<N>(order);
	return order;
      }

      template <std::size_t N>
      Coor<N - 1> remove_coor(Coor<N> v, std::size_t pos)
      {
	assert(pos < N);
	Coor<N - 1> r;
	for (std::size_t i = 0, j = 0; i < N; ++i)
	  if (i != pos)
	    r[j++] = v[i];
	return r;
      }

      inline std::string remove_coor(const std::string& v, std::size_t pos)
      {
	std::string r = v;
	r.erase(pos, 1);
	return r;
      }

      template <std::size_t N>
      Coor<N + 1> insert_coor(Coor<N> v, std::size_t pos, Index value)
      {
	assert(pos <= N);
	Coor<N + 1> r;
	for (std::size_t i = 0, j = 0; j < N + 1; ++j)
	{
	  if (j != pos)
	    r[j] = v[i++];
	  else
	    r[j] = value;
	}
	return r;
      }

      template <std::size_t N>
      Coor<N> replace_coor(Coor<N> v, std::size_t pos, Index value)
      {
	assert(pos <= N);
	v[pos] = value;
	return v;
      }

      inline std::string insert_coor(std::string v, std::size_t pos, char value)
      {
	assert(pos <= v.size());
	v.insert(pos, 1, value);
	return v;
      }

      inline std::string replace_coor(std::string v, std::size_t pos, char value)
      {
	assert(pos <= v.size());
	v[pos] = value;
	return v;
      }

#  if defined(QDP_IS_QDPJIT) && defined(BUILD_MAGMA)
      // Return a MAGMA context
      inline std::shared_ptr<magma_queue_t> getMagmaContext(Maybe<int> device = none)
      {
	static std::shared_ptr<magma_queue_t> queue;
	if (!queue)
	{
	  // Start MAGMA and create a queue
	  int dev = device.getSome(-1);
	  if (dev < 0)
	  {
#    ifdef SUPERBBLAS_USE_CUDA
	    superbblas::detail::cudaCheck(cudaGetDevice(&dev));
#    elif defined(SUPERBBLAS_USE_HIP)
	    superbblas::detail::hipCheck(hipGetDevice(&dev));
#    else
#      error superbblas was not build with support for GPUs
#    endif
	  }
	  magma_init();
	  magma_queue_t q;
	  magma_queue_create(dev, &q);
	  queue = std::make_shared<magma_queue_t>(q);
	}
	return queue;
      }
#  endif

      // Return a context on either the host or the device
      inline std::shared_ptr<superbblas::Context> getContext(DeviceHost dev)
      {
	// Creating GPU context can be expensive; so do it once
	static std::shared_ptr<superbblas::Context> cudactx;
	static std::shared_ptr<superbblas::Context> cpuctx;
	if (!cpuctx)
	  cpuctx = std::make_shared<superbblas::Context>(superbblas::createCpuContext());

	switch (dev)
	{
	case OnHost: return cpuctx;
	case OnDefaultDevice:
#  ifdef QDP_IS_QDPJIT
	  if (!cudactx)
	  {

	    int dev = -1;
#    ifdef SUPERBBLAS_USE_CUDA
	    superbblas::detail::cudaCheck(cudaGetDevice(&dev));
#    elif defined(SUPERBBLAS_USE_HIP)
	    superbblas::detail::hipCheck(hipGetDevice(&dev));
#    else
#      error superbblas was not build with support for GPUs
#    endif

#    if defined(BUILD_MAGMA)
	    // Force the creation of the queue before creating a superbblas context (otherwise cublas complains)
	    getMagmaContext();
#    endif

	    // Workaround on a potential issue in qdp-jit: avoid passing through the pool allocator
	    if (jit_config_get_max_allocation() == 0)
	    {
	      cudactx = std::make_shared<superbblas::Context>(superbblas::createGpuContext(dev));
	    }
	    else
	    {
	      cudactx = std::make_shared<superbblas::Context>(superbblas::createGpuContext(
		dev,

		// Make superbblas use the same memory allocator for gpu as any other qdp-jit lattice object
		[](std::size_t size, superbblas::platform plat) -> void* {
		  if (size == 0)
		    return nullptr;
		  if (plat == superbblas::CPU)
		    return malloc(size);
		  void* ptr = nullptr;
		  QDP_get_global_cache().addDeviceStatic(&ptr, size, true);
		  assert(superbblas::detail::getPtrDevice(ptr) >= 0);
		  return ptr;
		},

		// The corresponding deallocator
		[](void* ptr, superbblas::platform plat) {
		  if (ptr == nullptr)
		    return;
		  if (plat == superbblas::CPU)
		    free(ptr);
		  else
		    QDP_get_global_cache().signoffViaPtr(ptr);
		}));
	    }
	  }
	  return cudactx;
#  else
	  return cpuctx;
#  endif
	}
	throw std::runtime_error("Unsupported `DeviceHost`");
      }

      /// Return if two devices are the same

      inline bool is_same(DeviceHost a, DeviceHost b)
      {
#  ifdef QDP_IS_QDPJIT
	return a == b;
#  else
	// Without gpus, OnHost and OnDefaultDevice means on cpu.
	return true;
#  endif
      }

      /// Return an ordering with labels 0, 1, ...
      inline std::string getTrivialOrder(std::size_t N)
      {
	std::string r(N, 0);
	for (std::size_t i = 0; i < N; ++i)
	  r[i] = i % 128;
	return r;
      }

      /// Stores the subtensor supported on each node (used by class Tensor)
      template <std::size_t N>
      struct TensorPartition {
      public:
	using PartitionStored = std::vector<superbblas::PartitionItem<N>>;
	Coor<N> dim;	   ///< Dimensions of the tensor
	PartitionStored p; ///< p[i] = {first coordinate, size} of tensor on i-th node
	bool isLocal;	   ///< Whether the partition is non-collective

	/// Constructor
	/// \param order: dimension labels (use x, y, z, t for lattice dimensions)
	/// \param dim: dimension size for the tensor
	/// \param dist: how to distribute the tensor among the nodes

	TensorPartition(const std::string& order, Coor<N> dim, Distribution dist) : dim(dim)
	{
	  detail::check_order<N>(order);
	  isLocal = false;
	  switch (dist)
	  {
	  case OnMaster: p = all_tensor_on_master(dim); break;
	  case OnEveryone: p = partitioning_chroma_compatible(order, dim); break;
	  case OnEveryoneReplicated: p = all_tensor_replicated(dim); break;
	  case Local:
	    p = local(dim);
	    isLocal = true;
	    break;
	  }
	}

	/// Constructor for `insert_dimension`
	/// \param dim: dimension size for the tensor
	/// \param p: partition
	/// \praam isLocal: whether the tensor is local

	TensorPartition(Coor<N> dim, const PartitionStored& p, bool isLocal = false)
	  : dim(dim), p(p), isLocal(isLocal)
	{
	}

	/// Return the volume of the tensor supported on this node
	std::size_t localVolume() const
	{
	  return superbblas::detail::volume(p[MpiProcRank()][1]);
	}

	/// Return the first coordinate store locally
	Coor<N> localFrom() const
	{
	  return p[MpiProcRank()][0];
	}

	/// Return the number of elements store locally in each dimension
	Coor<N> localSize() const
	{
	  return p[MpiProcRank()][1];
	}

	/// Return how many processes have support on this tensor
	/// Note that it may differ from MPI's numProcs if the tensor does not have support on all processes.

	unsigned int numProcs() const
	{
	  unsigned int numprocs = 0;
	  for (const auto& i : p)
	    if (superbblas::detail::volume(i[1]) > 0)
	      numprocs++;
	  return numprocs;
	}

	/// Return the process rank on this tensor or -1 if this process does not have support on the tensor
	/// Note that it may differ from MPI's rank if the tensor does not have support on all processes.

	int procRank() const
	{
	  // Return -1 if this process does not have support on the tensor
	  int mpi_rank = MpiProcRank();
	  if (superbblas::detail::volume(p[mpi_rank][1]) == 0)
	    return -1;

	  // Return as rank how many processes with MPI rank smaller than this process have support
	  int this_rank = 0;
	  for (int i = 0; i < mpi_rank; ++i)
	    if (superbblas::detail::volume(p[i][1]) > 0)
	      this_rank++;
	  return this_rank;
	}

	/// Return the MPI process rank
	int MpiProcRank() const
	{
	  return (isLocal ? 0 : Layout::nodeNumber());
	}

	/// Insert a new non-distributed dimension

	TensorPartition<N + 1> insert_dimension(std::size_t pos, std::size_t dim_size) const
	{
	  typename TensorPartition<N + 1>::PartitionStored r;
	  r.reserve(p.size());
	  for (const auto& i : p)
	    r.push_back({insert_coor(i[0], pos, 0), insert_coor(i[1], pos, dim_size)});
	  return TensorPartition<N + 1>{insert_coor(dim, pos, dim_size), r};
	}

	/// Remove a non-distributed dimension

	TensorPartition<N - 1> remove_dimension(std::size_t pos) const
	{
	  typename TensorPartition<N - 1>::PartitionStored r;
	  r.reserve(p.size());
	  for (const auto& i : p)
	    r.push_back({remove_coor(i[0], pos), remove_coor(i[1], pos)});
	  return TensorPartition<N - 1>{remove_coor(dim, pos), r};
	}

	/// Split a dimension into a non-distributed dimension and another dimension

	TensorPartition<N + 1> split_dimension(std::size_t pos, Index step) const
	{
	  typename TensorPartition<N + 1>::PartitionStored r;
	  r.reserve(p.size());
	  for (const auto& i : p)
	  {
	    if (i[1][pos] % step != 0 && i[1][pos] > step)
	      throw std::runtime_error("Unsupported splitting a dimension with an uneven lattice "
				       "portions in all processes");
	    r.push_back(
	      {insert_coor(replace_coor(i[0], pos, i[0][pos] % step), pos + 1, i[0][pos] / step),
	       insert_coor(replace_coor(i[1], pos, std::min(i[1][pos], step)), pos + 1,
			   (i[1][pos] + step - 1) / step)});
	  }
	  return TensorPartition<N + 1>{
	    insert_coor(replace_coor(dim, pos, std::min(dim[pos], step)), pos + 1, dim[pos] / step),
	    r};
	}

	/// Return a partition with the local portion of the tensor

	TensorPartition<N> get_local_partition() const
	{
	  return TensorPartition<N>{
	    localSize(), PartitionStored(1, superbblas::PartitionItem<N>{{{}, localSize()}}), true};
	}

	/// Return a copy of this tensor with a compatible distribution to be contracted with the given tensor
	/// \param order: labels for this distribution
	/// \param t: given tensor distribution
	/// \param ordert: labels for the given distribution

	template <std::size_t Nt>
	TensorPartition<N> make_suitable_for_contraction(const std::string& order,
							 const TensorPartition<Nt>& t,
							 const std::string& ordert) const
	{
	  PartitionStored r(p.size());
	  std::map<char, Index> mf, ms;
	  for (std::size_t i = 0; i < N; ++i)
	    mf[order[i]] = 0;
	  for (std::size_t i = 0; i < N; ++i)
	    ms[order[i]] = dim[i];
	  for (std::size_t pi = 0; pi < p.size(); ++pi)
	  {
	    std::map<char, Index> mfrom = mf;
	    for (std::size_t i = 0; i < Nt; ++i)
	      mfrom[ordert[i]] = t.p[pi][0][i];
	    for (std::size_t i = 0; i < N; ++i)
	      r[pi][0][i] = mfrom[order[i]];

	    std::map<char, Index> msize = ms;
	    for (std::size_t i = 0; i < Nt; ++i)
	      msize[ordert[i]] = t.p[pi][1][i];
	    for (std::size_t i = 0; i < N; ++i)
	      r[pi][1][i] = msize[order[i]];
	  }
	  return TensorPartition<N>{dim, r, isLocal};
	}

      private:
	/// Return a partitioning for a non-collective tensor
	/// \param dim: dimension size for the tensor

	static PartitionStored local(Coor<N> dim)
	{
	  return PartitionStored(1, superbblas::PartitionItem<N>{{{}, dim}});
	}

	/// Return a partitioning where the root node has support for the whole tensor
	/// \param dim: dimension size for the tensor

	static PartitionStored all_tensor_on_master(Coor<N> dim)
	{
	  int nprocs = Layout::numNodes();
	  // Set the first coordinate and size of tensor supported on each proc to zero excepting
	  // on proc 0, where the size is set to dim
	  PartitionStored fs(nprocs);
	  if (1 <= nprocs)
	    fs[0][1] = dim;
	  return fs;
	}

	/// Return a partitioning where all nodes have support for the whole tensor
	/// \param dim: dimension size for the tensor

	static PartitionStored all_tensor_replicated(Coor<N> dim)
	{
	  int nprocs = Layout::numNodes();
	  // Set the first coordinate of the tensor supported on each prop to zero and the size
	  // to dim
	  PartitionStored fs(nprocs);
	  for (auto& it : fs)
	    it[1] = dim;
	  return fs;
	}

	/// Return a partitioning for a tensor of `dim` dimension onto a grid of processes
	/// \param order: dimension labels (use x, y, z, t for lattice dimensions)
	/// \param dim: dimension size for the tensor

	static PartitionStored partitioning_chroma_compatible(const std::string& order, Coor<N> dim)
	{
	  // Find a dimension label in `order` that is going to be distributed
	  const char dist_labels[] = "xyzt"; // distributed dimensions
	  int first_dist_label = -1;
	  for (unsigned int i = 0; i < std::strlen(dist_labels); ++i)
	  {
	    const auto& it = std::find(order.begin(), order.end(), dist_labels[i]);
	    if (it != order.end())
	    {
	      first_dist_label = it - order.begin();
	      break;
	    }
	  }

	  // If no dimension is going to be distributed, the whole tensor will have support only on node zero
	  if (first_dist_label < 0)
	    return all_tensor_on_master(dim);

	  // Get the number of procs use in each dimension; for know we put as many as chroma
	  // put onto the lattice dimensions
	  multi1d<int> procs_ = Layout::logicalSize();
	  Coor<N> procs = kvcoors<N>(
	    order, {{'x', procs_[0]}, {'y', procs_[1]}, {'z', procs_[2]}, {'t', procs_[3]}}, 1,
	    NoThrow);

	  // For each proc, get its coordinate in procs (logical coordinate) and compute the
	  // fair range of the tensor supported on the proc
	  int num_procs = Layout::numNodes();
	  PartitionStored fs(num_procs);
	  for (int rank = 0; rank < num_procs; ++rank)
	  {
	    multi1d<int> cproc_ = Layout::getLogicalCoordFrom(rank);
	    Coor<N> cproc = kvcoors<N>(
	      order, {{'x', cproc_[0]}, {'y', cproc_[1]}, {'z', cproc_[2]}, {'t', cproc_[3]}}, 0,
	      NoThrow);
	    for (unsigned int i = 0; i < N; ++i)
	    {
	      // First coordinate in process with rank 'rank' on dimension 'i'
	      fs[rank][0][i] = dim[i] / procs[i] * cproc[i] + std::min(cproc[i], dim[i] % procs[i]);
	      // Number of elements in process with rank 'cproc[i]' on dimension 'i'
	      fs[rank][1][i] =
		dim[i] / procs[i] + (dim[i] % procs[i] > cproc[i] ? 1 : 0) % procs[i];
	    }

	    // Avoid replicating parts of tensor if some of the lattice dimensions does not participate on this tensor
	    for (unsigned int i = 0; i < std::strlen(dist_labels); ++i)
	    {
	      if (std::find(order.begin(), order.end(), dist_labels[i]) != order.end())
		continue;
	      if (cproc_[i] > 0)
	      {
		fs[rank][1][first_dist_label] = 0;
		break;
	      }
	    }
	  }
	  return fs;
	}
      };

      template <typename T>
      struct WordType {
	using type = T;
      };

      template <typename T>
      struct WordType<std::complex<T>> {
	using type = T;
      };

      /// Return a Nan for float, double, and complex variants
      template <typename T>
      struct NaN;

      /// Specialization for float
      template <>
      struct NaN<float> {
	static float get()
	{
	  return std::nanf("");
	}
      };

      /// Specialization for double
      template <>
      struct NaN<double> {
	static double get()
	{
	  return std::nan("");
	}
      };

      /// Specialization for std::complex
      template <typename T>
      struct NaN<std::complex<T>> {
	static std::complex<T> get()
	{
	  return std::complex<T>{NaN<T>::get(), NaN<T>::get()};
	}
      };

      /// Return if a float, double, and std::complex is finite
      template <typename T>
      struct IsFinite {
	static bool get(T v)
	{
	  return std::isfinite(v);
	}
      };

      /// Specialization for std::complex
      template <typename T>
      struct IsFinite<std::complex<T>> {
	static bool get(std::complex<T> v)
	{
	  return std::isfinite(v.real()) && std::isfinite(v.imag());
	}
      };

      namespace repr
      {
	template <typename Ostream, std::size_t N>
	Ostream& operator<<(Ostream& s, Coor<N> o)
	{
	  s << "[";
	  if (N > 0)
	    s << o[0];
	  for (unsigned int i = 1; i < N; ++i)
	    s << "," << o[i];
	  s << "]";
	  return s;
	}

	template <typename Ostream>
	Ostream& operator<<(Ostream& s, Distribution dist)
	{
	  switch (dist)
	  {
	  case OnMaster: s << "OnMaster"; break;
	  case OnEveryone: s << "OnEveryone"; break;
	  case OnEveryoneReplicated: s << "OnEveryoneReplicated"; break;
	  }
	  return s;
	}

	template <typename Ostream, typename T>
	Ostream& operator<<(Ostream& s, std::complex<T> o)
	{
	  s << std::real(o) << "+" << std::imag(o) << "i";
	  return s;
	}

	template <typename Ostream, typename T,
		  typename std::enable_if<std::is_floating_point<T>::value, bool>::type = true>
	Ostream& operator<<(Ostream& s, T o)
	{
	  s.operator<<(o);
	  return s;
	}

	template <typename Ostream, typename T>
	Ostream& operator<<(Ostream& s, const std::vector<T>& o)
	{
	  s << "{";
	  for (const auto& i : o)
	    s << i;
	  s << "}";
	  return s;
	}

	template <typename Ostream, typename T, std::size_t N>
	Ostream& operator<<(Ostream& s, const std::array<T, N>& o)
	{
	  s << "{";
	  for (const auto& i : o)
	    s << i;
	  s << "}";
	  return s;
	}
      }

      inline void log(int level, const std::string& s)
      {
	static int log_level = []() {
	  const char* l = std::getenv("SB_LOG");
	  if (l)
	    return std::atoi(l);
	  return 0;
	}();
	if (log_level < level)
	  return;
	QDPIO::cout << s << std::endl;
	QDPIO::cout.flush();
      }

      inline void log_mem()
      {
	if (!superbblas::getTrackingMemory())
	  return;
	std::stringstream ss;
	ss << "mem usage, CPU: " << std::fixed << std::setprecision(0)
	   << superbblas::getCpuMemUsed(0) / 1024 / 1024
	   << " MiB   GPU: " << superbblas::getGpuMemUsed(0) / 1024 / 1024 << " MiB";
	log(1, ss.str());
      }

      /// is_complex<T>::value is true if `T` is complex

      template <typename T>
      struct is_complex : std::false_type {
      };

      template <typename T>
      struct is_complex<std::complex<T>> : std::true_type {
      };

      template <typename T, typename A, typename B,
		typename std::enable_if<!is_complex<T>::value, bool>::type = true>
      T safe_div(A a, B b)
      {
	if (std::fabs(std::imag(a)) != 0 || std::fabs(std::imag(b)) != 0)
	  throw std::runtime_error("Invalid division");
	return std::real(a) / std::real(b);
      }

      template <typename T, typename A, typename B,
		typename std::enable_if<is_complex<T>::value, bool>::type = true>
      T safe_div(A a, B b)
      {
	return (T)a / (T)b;
      }

      inline bool is_default_device_gpu()
      {
	static bool v = []() {
	  const char* l = std::getenv("SB_DEFAULT_DEVICE_GPU");
	  if (l)
	    return std::atoi(l) != 0;
	  return true;
	}();
	return v;
      }
    }

    template <std::size_t N, typename T>
    struct Tensor {
      static_assert(superbblas::supported_type<T>::value, "Not supported type");

    public:
      std::string order;			///< Labels of the tensor dimensions
      Coor<N> dim;				///< Length of the tensor dimensions
      std::shared_ptr<superbblas::Context> ctx; ///< Tensor storage information (device/host)
      std::shared_ptr<T> data;			///< Pointer to the tensor storage
      std::shared_ptr<detail::TensorPartition<N>>
	p;		 ///< Distribution of the tensor among the processes
      Distribution dist; ///< Whether the tensor is stored on the cpu or a device
      Coor<N> from;	 ///< First active coordinate in the tensor
      Coor<N> size;	 ///< Number of active coordinates on each dimension
      Coor<N> strides;	 ///< Displacement for the next element along every direction
      T scalar;		 ///< Scalar factor of the tensor

      // Return a string describing the tensor
      std::string repr(T* ptr = nullptr) const
      {
	using namespace detail::repr;
	std::stringstream ss;
	ss << "Tensor{";
	if (ptr)
	  ss << "data:" << ptr << ", ";
	std::size_t sizemb = p->localVolume() * sizeof(T) / 1024 / 1024;
	ss << "order:" << order << ", dim:" << dim << ", dist:" << dist
	   << ", local_storage:" << sizemb << " MiB}";
	return ss.str();
      }

      // Construct used by non-Chroma tensors
      Tensor(const std::string& order, Coor<N> dim, DeviceHost dev = OnDefaultDevice,
	     Distribution dist = OnEveryone)
	: Tensor(order, dim, dev, dist,
		 std::make_shared<detail::TensorPartition<N>>(
		   detail::TensorPartition<N>(order, dim, dist)))
      {
      }

      // Empty constructor
      Tensor()
	: order(detail::getTrivialOrder(N)),
	  dim{},
	  ctx(detail::getContext(OnHost)),
	  p(std::make_shared<detail::TensorPartition<N>>(
	    detail::TensorPartition<N>(detail::getTrivialOrder(N), {}, OnEveryoneReplicated))),
	  dist(OnEveryoneReplicated),
	  from{},
	  size{},
	  strides{},
	  scalar{0}
      {
      }

      // Construct used by Chroma tensors (see `asTensorView`)
      Tensor(const std::string& order, Coor<N> dim, DeviceHost dev, Distribution dist,
	     std::shared_ptr<T> data)
	: order(order),
	  dim(dim),
	  ctx(detail::getContext(dev)),
	  data(data),
	  dist(dist),
	  from{},
	  size(dim),
	  strides(detail::get_strides<N>(dim, superbblas::FastToSlow)),
	  scalar{1}
      {
	checkOrder();

	// For now, TensorPartition creates the same distribution as chroma for tensor with
	// dimensions divisible by chroma logical dimensions
	p = std::make_shared<detail::TensorPartition<N>>(
	  detail::TensorPartition<N>(order, dim, dist));
      }

      // Construct for toFakeReal
      Tensor(const std::string& order, Coor<N> dim, std::shared_ptr<superbblas::Context> ctx,
	     std::shared_ptr<T> data, std::shared_ptr<detail::TensorPartition<N>> p,
	     Distribution dist, Coor<N> from, Coor<N> size, T scalar)
	: order(order),
	  dim(dim),
	  ctx(ctx),
	  data(data),
	  p(p),
	  dist(dist),
	  from(normalize_coor(from, dim)),
	  size(size),
	  strides(detail::get_strides<N>(dim, superbblas::FastToSlow)),
	  scalar(scalar)
      {
	checkOrder();
      }

    protected:
      // Construct used by non-Chroma tensors and make_suitable_for_contraction
      Tensor(const std::string& order, Coor<N> dim, DeviceHost dev, Distribution dist,
	     std::shared_ptr<detail::TensorPartition<N>> p)
	: order(order),
	  dim(dim),
	  ctx(detail::getContext(dev)),
	  p(p),
	  dist(dist),
	  from{},
	  size(dim),
	  strides(detail::get_strides<N>(dim, superbblas::FastToSlow)),
	  scalar{1}
      {
	checkOrder();
	superbblas::Context ctx0 = *ctx;
	std::string s = repr();
	detail::log(1, "allocating " + s);
	T* ptr = superbblas::allocate<T>(p->localVolume(), *ctx);
	detail::log_mem();
	data = std::shared_ptr<T>(ptr, [=](const T* ptr) {
	  superbblas::deallocate(ptr, ctx0);
	  detail::log(1, "deallocated " + s);
	  detail::log_mem();
	});
      }

      // Construct a slice of a tensor
      Tensor(const Tensor& t, const std::string& order, Coor<N> from, Coor<N> size)
	: order(order),
	  dim(t.dim),
	  ctx(t.ctx),
	  data(t.data),
	  p(t.p),
	  dist(t.dist),
	  from(normalize_coor(from, t.dim)),
	  size(size),
	  strides(t.strides),
	  scalar{t.scalar}
      {
	checkOrder();
      }

      // Construct a scaled tensor
      Tensor(const Tensor& t, T scalar)
	: order(t.order),
	  dim(t.dim),
	  ctx(t.ctx),
	  data(t.data),
	  p(t.p),
	  dist(t.dist),
	  from(t.from),
	  size(t.size),
	  strides(t.strides),
	  scalar{scalar}
      {
	checkOrder();
      }

    public:
      /// Return whether the tensor is not empty
      explicit operator bool() const noexcept
      {
	return superbblas::detail::volume(size) > 0;
      }

      // Return whether from != 0 or size != dim
      bool isSubtensor() const
      {
	return (from != Coor<N>{} || size != dim);
      }

      // Return the first coordinate supported by the tensor
      std::map<char, int> kvfrom() const
      {
	std::map<char, int> d;
	for (unsigned int i = 0; i < N; ++i)
	  d[order[i]] = from[i];
	return d;
      }

      // Return the dimensions of the tensor
      std::map<char, int> kvdim() const
      {
	std::map<char, int> d;
	for (unsigned int i = 0; i < N; ++i)
	  d[order[i]] = size[i];
	return d;
      }

      // Return the volume of the tensor
      std::size_t volume() const
      {
	return superbblas::detail::volume(size);
      }

      // Get an element of the tensor
      T get(Coor<N> coor) const
      {
	if (ctx->plat != superbblas::CPU)
	  throw std::runtime_error(
	    "Unsupported to `get` elements from tensors not stored on the host");
	if (dist == OnEveryone)
	  throw std::runtime_error(
	    "Unsupported to `get` elements on a distributed tensor; change the distribution to "
	    "be supported on master, replicated among all nodes, or local");

	// coor[i] = coor[i] + from[i]
	for (unsigned int i = 0; i < N; ++i)
	  coor[i] = normalize_coor(normalize_coor(coor[i], size[i]) + from[i], dim[i]);

	return data.get()[detail::coor2index<N>(coor, dim, strides)] * scalar;
      }

      // Set an element of the tensor
      void set(Coor<N> coor, T v)
      {
	if (ctx->plat != superbblas::CPU)
	  throw std::runtime_error(
	    "Unsupported to `get` elements from tensors not stored on the host");
	if (dist == OnEveryone)
	  throw std::runtime_error(
	    "Unsupported to `set` elements on a distributed tensor; change the distribution to "
	    "be supported on master, replicated among all nodes, or local");

	// coor[i] = coor[i] + from[i]
	for (unsigned int i = 0; i < N; ++i)
	  coor[i] = normalize_coor(normalize_coor(coor[i], size[i]) + from[i], dim[i]);

	data.get()[detail::coor2index<N>(coor, dim, strides)] = v / scalar;
      }

      /// Rename dimensions
      Tensor<N, T> rename_dims(const SB::remap& m) const
      {
	return Tensor<N, T>(*this, detail::update_order<N>(order, m), this->from, this->size);
      }

      // Return a slice of the tensor starting at coordinate `kvfrom` and taking `kvsize` elements in each direction.
      // The missing dimension in `kvfrom` are set to zero and the missing direction in `kvsize` are set to the active size of the tensor.
      Tensor<N, T> kvslice_from_size(const std::map<char, int>& kvfrom = {},
				     const std::map<char, int>& kvsize = {}) const
      {
	std::map<char, int> updated_kvsize = this->kvdim();
	for (const auto& it : kvsize)
	  updated_kvsize[it.first] = it.second;
	return slice_from_size(kvcoors<N>(order, kvfrom), kvcoors<N>(order, updated_kvsize));
      }

      // Return a slice of the tensor starting at coordinate `from` and taking `size` elements in each direction.
      Tensor<N, T> slice_from_size(Coor<N> from, Coor<N> size) const
      {
	for (unsigned int i = 0; i < N; ++i)
	{
	  if (size[i] > this->size[i])
	    throw std::runtime_error(
	      "The size of the slice cannot be larger than the original tensor");
	  if (normalize_coor(from[i], this->size[i]) + size[i] > this->size[i] &&
	      this->size[i] != this->dim[i])
	    throw std::runtime_error(
	      "Unsupported to make a view on a non-contiguous range on the tensor");
	}

	using superbblas::detail::operator+;
	return Tensor<N, T>(*this, order, this->from + from, size);
      }

      /// Return a tensor on the same device and following the same distribution
      /// \param new_order: dimension labels of the new tensor
      /// \param kvsize: override the length of the given dimensions
      /// \param new_dev: device
      /// \param new_dist: distribution

      template <std::size_t Nn = N, typename Tn = T>
      Tensor<Nn, Tn>
      like_this(const Maybe<std::string>& new_order = none, const std::map<char, int>& kvsize = {},
		Maybe<DeviceHost> new_dev = none, Maybe<Distribution> new_dist = none) const
      {
	std::map<char, int> new_kvdim = kvdim();
	for (const auto& it : kvsize)
	  new_kvdim[it.first] = it.second;
	std::string new_order_ = new_order.getSome(order);
	return Tensor<Nn, Tn>(new_order_, kvcoors<Nn>(new_order_, new_kvdim, 0, ThrowOnMissing),
			      new_dev.getSome(getDev()), new_dist.getSome(dist));
      }

      /// Return a tensor on the same device and following the same distribution
      /// \param new_order: dimension labels of the new tensor
      /// \param remaining_char: placeholder for the remaining dimensions
      /// \param kvsize: override the length of the given dimensions
      /// \param new_dev: device
      /// \param new_dist: distribution

      template <std::size_t Nn = N, typename Tn = T>
      Tensor<Nn, Tn>
      like_this(const std::string& new_order, char remaining_char,
		const std::string& remove_dims = "", const std::map<char, int>& kvsize = {},
		Maybe<DeviceHost> new_dev = none, Maybe<Distribution> new_dist = none) const
      {
	std::map<char, int> new_kvdim = kvdim();
	for (const auto& it : kvsize)
	  new_kvdim[it.first] = it.second;
	std::string new_order_ =
	  detail::remove_dimensions(get_order_for_reorder(new_order, remaining_char), remove_dims);
	return Tensor<Nn, Tn>(new_order_, kvcoors<Nn>(new_order_, new_kvdim, 0, ThrowOnMissing),
			      new_dev.getSome(getDev()), new_dist.getSome(dist));
      }

      /// Return a copy of this tensor, possible with a new precision `nT`

      template <typename Tn = T>
      Tensor<N, Tn> clone() const
      {
	return cloneOn<Tn>(getDev());
      }

      /// Return a copy of this tensor on device `new_dev`, possible with a new precision `nT`
      /// \param new_dev: device that will hold the new tensor

      template <typename Tn = T>
      Tensor<N, Tn> cloneOn(DeviceHost new_dev) const
      {
	Tensor<N, Tn> r = like_this<N, Tn>(none, {}, new_dev);
	copyTo(r);
	return r;
      }

    private:
      /// Return the new ordering based on a partial reordering
      /// \param new_order: new dimension labels order
      /// \param remaining_char: if it isn't the null char, placeholder for the dimensions not given
      ///
      /// If the dimension labels order does not match the current order, return a copy of this
      /// tensor with that ordering. If the given order does not contain all dimensions, only the
      /// involved dimensions are permuted.

      std::string get_order_for_reorder(const std::string& new_order, char remaining_char = 0) const
      {
	std::string new_order1;
	if (remaining_char != 0)
	{
	  std::string::size_type rem_pos = new_order.find(remaining_char);
	  if (rem_pos == std::string::npos)
	  {
	    new_order1 = new_order;
	  }
	  else
	  {
	    new_order1 = new_order.substr(0, rem_pos) +
			 detail::remove_dimensions(order, new_order) +
			 new_order.substr(rem_pos + 1, new_order.size() - rem_pos - 1);
	  }
	}
	else
	{
	  new_order1 = order;
	  unsigned int j = 0;
	  for (unsigned int i = 0; i < N; ++i)
	    if (new_order.find(order[i]) != std::string::npos)
	      new_order1[i] = new_order[j++];
	  if (j < new_order.size())
	    throw std::runtime_error("Unknown labels in the given order");
	}

	return new_order1;
      }

    public:
      /// Return a copy of this tensor with the given ordering
      /// \param new_order: new dimension labels order
      /// \param remaining_char: if it isn't the null char, placeholder for the dimensions not given
      ///
      /// If the dimension labels order does not match the current order, return a copy of this
      /// tensor with that ordering. If the given order does not contain all dimensions, only the
      /// involved dimensions are permuted.

      Tensor<N, T> reorder(const std::string& new_order, char remaining_char = 0) const
      {
	std::string new_order1 = get_order_for_reorder(new_order, remaining_char);
	if (order == new_order1)
	  return *this;
	Tensor<N, T> r = like_this(new_order1);
	copyTo(r);
	return r;
      }

      /// Return whether the tensor has complex components although being stored with a non-complex type `T`

      bool isFakeReal() const
      {
	return order.find('.') != std::string::npos;
      }

      /// Check that the dimension labels are valid

      void checkOrder() const
      {
	// Check that all labels are different there are N
	detail::check_order<N>(order);

	/// Throw exception if this a fake real tensor but with a complex type `T`
	if (isFakeReal() && detail::is_complex<T>::value)
	  throw std::runtime_error("Invalid tensor: it is fake real and complex!");

	for (auto s : size)
	  if (s < 0)
	    std::runtime_error("Invalid tensor size: it should be positive");
      }

      /// Return a fake real view of this tensor

      template <typename U = T,
		typename std::enable_if<detail::is_complex<U>::value, bool>::type = true>
      Tensor<N + 1, typename U::value_type> toFakeReal() const
      {
	assert(!isFakeReal());

	std::string new_order = "." + order;
	Coor<N + 1> new_from = {0};
	std::copy_n(from.begin(), N, new_from.begin() + 1);
	Coor<N + 1> new_size = {2};
	std::copy_n(size.begin(), N, new_size.begin() + 1);
	Coor<N + 1> new_dim = {2};
	std::copy_n(dim.begin(), N, new_dim.begin() + 1);
	if (std::fabs(std::imag(scalar)) != 0)
	  throw std::runtime_error(
	    "Unsupported conversion to fake real tensors with an implicit complex scale");
	using new_T = typename T::value_type;
	new_T new_scalar = std::real(scalar);
	auto this_data = data;
	auto new_data =
	  std::shared_ptr<new_T>((new_T*)data.get(), [=](const new_T* ptr) { (void)this_data; });
	auto new_p = std::make_shared<detail::TensorPartition<N + 1>>(p->insert_dimension(0, 2));

	return Tensor<N + 1, new_T>(new_order, new_dim, ctx, new_data, new_p, dist, new_from,
				    new_size, new_scalar);
      }

      template <typename U = T,
		typename std::enable_if<!detail::is_complex<U>::value, bool>::type = true>
      Tensor<N - 1, std::complex<U>> toComplex(bool allow_cloning = true) const
      {
	assert(isFakeReal() && kvdim()['.'] == 2);

	std::size_t dot_pos = order.find('.');
	std::string new_order = detail::remove_coor(order, dot_pos);

	if (dot_pos != 0)
	{
	  if (allow_cloning)
	    return reorder("." + new_order).toComplex(false);
	  else
	    throw std::runtime_error("Not allow to create a new tensor in `toComplex`");
	}

	Coor<N - 1> new_from = detail::remove_coor(from, dot_pos);
	Coor<N - 1> new_size = detail::remove_coor(size, dot_pos);
	Coor<N - 1> new_dim = detail::remove_coor(dim, dot_pos);
	using new_T = std::complex<T>;
	new_T new_scalar = new_T{scalar};
	auto this_data = data;
	auto new_data =
	  std::shared_ptr<new_T>((new_T*)data.get(), [=](const new_T* ptr) { (void)this_data; });
	auto new_p = std::make_shared<detail::TensorPartition<N - 1>>(p->remove_dimension(dot_pos));

	return Tensor<N - 1, new_T>(new_order, new_dim, ctx, new_data, new_p, dist, new_from,
				    new_size, new_scalar);
      }

      template <typename U = T,
		typename std::enable_if<!detail::is_complex<U>::value, bool>::type = true>
      Tensor<N, U> toFakeReal() const
      {
	assert(isFakeReal());
	return *this;
      }

      template <typename U = T,
		typename std::enable_if<detail::is_complex<U>::value, bool>::type = true>
      Tensor<N, U> toComplex(bool allow_cloning = true) const
      {
	(void)allow_cloning;
	assert(!isFakeReal());
	return *this;
      }

      /// Return a fake real view of this tensor

      Tensor<N + 1, T> split_dimension(char dim_label, std::string new_labels, Index step) const
      {
	using namespace detail;

	// Find the position of dim_label in order
	std::string::size_type pos = order.find(dim_label);
	if (pos == std::string::npos)
	{
	  std::stringstream ss;
	  ss << "Not found label `" << dim_label << "` in this tensor with dimension labels `"
	     << order;
	  throw std::runtime_error(ss.str());
	}

	// Check the other arguments
	if (new_labels.size() != 2)
	  throw std::runtime_error("`new_labels` should have two labels!");

	if (step < 1)
	  throw std::runtime_error("`step` cannot be zero or negative");

	if (size[pos] % step != 0 && size[pos] > step)
	  throw std::runtime_error("Not supporting `split_dimension` for this lattice dimensions");

	// Set the new characteristics of the tensor
	std::string new_order =
	  insert_coor(replace_coor(order, pos, new_labels[0]), pos + 1, new_labels[1]);
	Coor<N + 1> new_from =
	  insert_coor(replace_coor(from, pos, from[pos] % step), pos + 1, from[pos] / step);
	Coor<N + 1> new_size = insert_coor(replace_coor(size, pos, std::min(size[pos], step)),
					   pos + 1, (size[pos] + step - 1) / step);
	Coor<N + 1> new_dim = insert_coor(replace_coor(dim, pos, std::min(dim[pos], step)), pos + 1,
					  (dim[pos] + step - 1) / step);

	auto new_p =
	  std::make_shared<detail::TensorPartition<N + 1>>(p->split_dimension(pos, step));

	return Tensor<N + 1, T>(new_order, new_dim, ctx, data, new_p, dist, new_from, new_size,
				scalar);
      }

      /// Copy/add this tensor into the given one
      /// NOTE: if this tensor or the given tensor is fake real, force both to be fake real

      template <std::size_t Nw, typename Tw,
		typename std::enable_if<
		  detail::is_complex<T>::value != detail::is_complex<Tw>::value, bool>::type = true>
      void doAction(Action action, Tensor<Nw, Tw> w) const
      {
	toFakeReal().doAction(action, w.toFakeReal());
      }

      /// Return the local support of this tensor
      Tensor<N, T> getLocal() const
      {
	// Compute the size of the intersection of the current view and the local support
	Coor<N> lfrom, lsize;
	superbblas::detail::intersection(p->localFrom(), p->localSize(), from, size, dim, lfrom,
					 lsize);

	// If the current process has no support, return the empty tensor
	if (superbblas::detail::volume(lsize) == 0)
	  return Tensor<N, T>{};

	using superbblas::detail::operator-;
	return Tensor<N, T>(order, p->localSize(), ctx, data,
			    std::make_shared<detail::TensorPartition<N>>(p->get_local_partition()),
			    Local, normalize_coor(from - p->localFrom(), dim), lsize, scalar);
      }

      /// Set zero
      void set_zero()
      {
	T* ptr = this->data.get();
	MPI_Comm comm = (dist == OnMaster || dist == Local ? MPI_COMM_SELF : MPI_COMM_WORLD);
	if (dist != OnMaster || Layout::nodeNumber() == 0)
	  superbblas::copy<N, N>(T{0}, p->p.data(), 1, order.c_str(), from, size, (const T**)&ptr,
				 &*ctx, p->p.data(), 1, order.c_str(), from, &ptr, &*ctx, comm,
				 superbblas::FastToSlow, superbblas::Copy);
      }

      /// Copy/Add this tensor into the given one
      template <std::size_t Nw, typename Tw,
		typename std::enable_if<
		  detail::is_complex<T>::value == detail::is_complex<Tw>::value, bool>::type = true>
      void doAction(Action action, Tensor<Nw, Tw> w) const
      {
	Coor<N> wsize = kvcoors<N>(order, w.kvdim(), 1, NoThrow);
	for (unsigned int i = 0; i < N; ++i)
	  if (size[i] > wsize[i])
	    throw std::runtime_error("The destination tensor is smaller than the source tensor");

	if (action == AddTo && w.scalar != Tw{1})
	  throw std::runtime_error("Not allowed to add to a tensor whose implicit scalar factor is not one");

	if ((dist == Local && w.dist != Local) || (dist != Local && w.dist == Local))
	{
	  getLocal().doAction(action, w.getLocal());
	  return;
	}

	T* ptr = this->data.get();
	Tw* w_ptr = w.data.get();
	MPI_Comm comm =
	  ((dist == OnMaster && w.dist == OnMaster) || dist == Local ? MPI_COMM_SELF
								     : MPI_COMM_WORLD);
	if (dist != OnMaster || w.dist != OnMaster || Layout::nodeNumber() == 0)
	{
	  superbblas::copy<N, Nw>(
	    detail::safe_div<T>(scalar, w.scalar), p->p.data(), 1, order.c_str(), from, size,
	    (const T**)&ptr, &*ctx, w.p->p.data(), 1, w.order.c_str(), w.from, &w_ptr, &*w.ctx,
	    comm, superbblas::FastToSlow, action == CopyTo ? superbblas::Copy : superbblas::Add);
	}
      }

      /// Copy this tensor into the given one
      template <std::size_t Nw, typename Tw>
      void copyTo(Tensor<Nw, Tw> w) const
      {
	doAction(CopyTo, w);
      }

      // Add `this` tensor into the given one
      template <std::size_t Nw, typename Tw>
      void addTo(Tensor<Nw, Tw> w) const
      {
	doAction(AddTo, w);
      }

      /// Return a copy of this tensor with a compatible distribution to be contracted with the given tensor
      /// \param v: given tensor

      template <std::size_t Nv, typename Tv,
		typename std::enable_if<std::is_same<T, Tv>::value, bool>::type = true>
      Tensor<N, T> make_suitable_for_contraction(Tensor<Nv, Tv> v) const
      {
	if (dist != OnEveryoneReplicated)
	  throw std::runtime_error("Invalid tensor distribution for this function");

	Coor<N> vsize = kvcoors<N>(order, v.kvdim(), 0, NoThrow);
	for (unsigned int i = 0; i < N; ++i)
	  if (vsize[i] != 0 && vsize[i] != size[i])
	    throw std::runtime_error("Invalid tensor contractions: one of the dimensions does not match");

	auto new_p = std::make_shared<detail::TensorPartition<N>>(
	  p->make_suitable_for_contraction(order, *v.p, v.order));

	Tensor<N, T> r(order, dim, getDev(), OnEveryone, new_p);
	copyTo(r);
	return r;
      }

      // Contract the dimensions with the same label in `v` and `w` than do not appear on `this` tensor.
      template <std::size_t Nv, std::size_t Nw>
      void contract(Tensor<Nv, T> v, const remap& mv, Conjugation conjv, Tensor<Nw, T> w,
		    const remap& mw, Conjugation conjw, const remap& mr = {}, T beta = T{0})
      {
	// If either v or w is on OnDevice, force both to be on device
	if (v.ctx->plat != w.ctx->plat)
	{
	  if (v.getDev() != OnDefaultDevice)
	    v = v.cloneOn(OnDefaultDevice);
	  if (w.getDev() != OnDefaultDevice)
	    w = w.cloneOn(OnDefaultDevice);
	}

	// Superbblas tensor contraction is shit and those not deal with subtensors or contracting a host and
	// device tensor (for now)
	if (v.isSubtensor())
	  v = v.clone();
	if (w.isSubtensor())
	  w = w.clone();
	if (isSubtensor() || getDev() != v.getDev())
	{
	  Tensor<N, T> aux =
	    std::norm(beta) == 0 ? like_this(none, {}, v.getDev()) : cloneOn(v.getDev());
	  aux.contract(v, mv, conjv, w, mw, conjw, mr, beta);
	  aux.copyTo(*this);
	  return;
	}

	if ((v.dist == Local) != (w.dist == Local) || (w.dist == Local) != (dist == Local))
	  throw std::runtime_error(
	    "One of the contracted tensors or the output tensor is local and others are not!");

	if ((v.dist == OnMaster && w.dist == OnEveryone) ||
	    (v.dist == OnEveryone && w.dist == OnMaster))
	  throw std::runtime_error("Incompatible layout for contractions: one of the tensors is on "
				   "the master node and the other is distributed");

	if ((v.dist == OnMaster && w.dist == OnEveryoneReplicated) ||
	    (v.dist == OnEveryoneReplicated && w.dist == OnMaster))
	{
	  contract(v.make_sure(none, none, OnMaster), mv, conjv, w.make_sure(none, none, OnMaster),
		   mw, conjw, mr, beta);
	  return;
	}

	if (v.dist == OnEveryone && w.dist == OnEveryoneReplicated)
	  w = w.make_suitable_for_contraction(v);

	if (v.dist == OnEveryoneReplicated && w.dist == OnEveryone)
	  v = v.make_suitable_for_contraction(w);

	T* v_ptr = v.data.get();
	T* w_ptr = w.data.get();
	T* ptr = this->data.get();
	std::string orderv_ = detail::update_order<Nv>(v.order, mv);
	std::string orderw_ = detail::update_order<Nw>(w.order, mw);
	std::string order_ = detail::update_order<N>(order, mr);
	T vscalar = conjv == NotConjugate ? v.scalar : std::conj(v.scalar);
	T wscalar = conjw == NotConjugate ? w.scalar : std::conj(w.scalar);
	superbblas::contraction<Nv, Nw, N>(
	  vscalar * wscalar / scalar, v.p->p.data(), 1, orderv_.c_str(), conjv == Conjugate,
	  (const T**)&v_ptr, &*v.ctx, w.p->p.data(), 1, orderw_.c_str(), conjw == Conjugate,
	  (const T**)&w_ptr, &*w.ctx, beta, p->p.data(), 1, order_.c_str(), &ptr, &*ctx,
	  MPI_COMM_WORLD, superbblas::FastToSlow);
      }

      Tensor<N, T> scale(T s) const
      {
	return Tensor<N, T>(*this, scalar * s);
      }

      void release()
      {
	dim = {};
	data.reset();
	p.reset();
	ctx.reset();
	from = {};
	size = {};
	strides = {};
	scalar = T{0};
      }

      // Return whether the current view is contiguous in memory
      bool isContiguous() const
      {
	// Meaningless for tensors not been fully supported on a single node
	if (dist != OnMaster && dist != Local)
	  return false;

	if (superbblas::detail::volume(size) > 0 && N > 1)
	{
	  bool non_full_dim = false; // some dimension is not full
	  for (unsigned int i = 0; i < N - 1; ++i)
	  {
	    if (from[i] != 0 || size[i] != dim[i])
	    {
	      if (non_full_dim && size[i] != 1)
		return false;
	      non_full_dim = true;
	    }
	  }
	}
	return true;
      }

      /// Return a copy of this tensor if it does not have the same type, the same order, or is not on the given device or distribution
      /// \param new_order: dimension labels of the new tensor
      /// \param new_dev: device
      /// \param new_dist: distribution
      /// \tparam Tn: new precision

      template <typename Tn = T,
		typename std::enable_if<std::is_same<T, Tn>::value, bool>::type = true>
      Tensor<N, Tn> make_sure(const Maybe<std::string>& new_order = none,
			      Maybe<DeviceHost> new_dev = none,
			      Maybe<Distribution> new_dist = none) const
      {
	if (new_order.getSome(order) != order ||
	    !detail::is_same(new_dev.getSome(getDev()), getDev()) || new_dist.getSome(dist) != dist)
	{
	  Tensor<N, Tn> r = like_this(new_order, {}, new_dev, new_dist);
	  copyTo(r);
	  return r;
	}
	else
	{
	  return *this;
	}
      }

      template <typename Tn = T,
		typename std::enable_if<!std::is_same<T, Tn>::value, bool>::type = true>
      Tensor<N, Tn> make_sure(const Maybe<std::string>& new_order = none,
			      Maybe<DeviceHost> new_dev = none,
			      Maybe<Distribution> new_dist = none) const
      {
	Tensor<N, Tn> r = like_this<N, Tn>(new_order, {}, new_dev, new_dist);
	copyTo(r);
	return r;
      }

      /// Get where the tensor is stored

      DeviceHost getDev() const
      {
#  ifdef QDP_IS_QDPJIT
	return (ctx->plat != superbblas::CPU ? OnDefaultDevice : OnHost);
#  else
	return OnDefaultDevice;
#  endif
      }

      void binaryRead(BinaryReader& bin)
      {
	if (ctx->plat != superbblas::CPU)
	  throw std::runtime_error("Only supported to read on `OnHost` tensors");
	if (dist != OnMaster)
	  throw std::runtime_error("Only supported to read on `OnMaster` tensors");
	if (!isContiguous())
	  throw std::runtime_error("Only supported contiguous views in memory");
	if (scalar != T{1})
	  throw std::runtime_error("Not allowed for tensor with a scale not being one");

	// Only on primary node read the data
	std::size_t vol = superbblas::detail::volume(size);
	std::size_t disp = detail::coor2index<N>(from, dim, strides);
	std::size_t word_size = sizeof(typename detail::WordType<T>::type);
	bin.readArrayPrimaryNode((char*)&data.get()[disp], word_size, sizeof(T) / word_size * vol);
      }

      void binaryWrite(BinaryWriter& bin) const
      {
	// If the writing is collective, the root process needs to hold the whole tensor
	if (!bin.isLocal() && dist != OnMaster)
	  throw std::runtime_error("For collective writing, the tensor should be `OnMaster`");

	// If the writing is non-collective, the tensor should be local
	if (bin.isLocal() && dist != Local)
	  throw std::runtime_error("For non-collective writing, the tensor should be `Local`");

	// If the tensor has an implicit scale, view, or is not on host, make a copy
	if (scalar != T{1} || isSubtensor() || ctx->plat != superbblas::CPU)
	{
	  cloneOn(OnHost).binaryWrite(bin);
	  return;
	}

	// Write the local data
	std::size_t vol = p->localVolume();
	std::size_t word_size = sizeof(typename detail::WordType<T>::type);
	bin.writeArrayPrimaryNode((char*)data.get(), word_size, sizeof(T) / word_size * vol);
      }

      void print(const std::string& name) const
      {
	std::stringstream ss;
	auto t = toComplex();
	auto t_host = t.like_this(none, {}, OnHost, OnMaster);
	t.copyTo(t_host);
	if (Layout::nodeNumber() == 0)
	{
	  using namespace detail::repr;
	  ss << "% " << repr(data.get()) << std::endl;
	  ss << "% dist=" << p->p << std::endl;
	  ss << name << "=reshape([";
	  std::size_t vol = superbblas::detail::volume(size);
	  for (std::size_t i = 0; i < vol; ++i)
	  {
	    //using detail::repr::operator<<;
	    ss << " ";
	    detail::repr::operator<<(ss, t_host.data.get()[i]);
	  }
	  ss << "], [" << size << "]);" << std::endl;
	}
	detail::log(1, ss.str());
      }
#  if 0
      /// Get where the tensor is stored
      void setNan() 
      {
#    ifndef QDP_IS_QDPJIT
	T nan = detail::NaN<T>::get();
	std::size_t vol = superbblas::detail::volume<N>(dim);
	T* p = data.get();
	for (std::size_t i = 0; i < vol; i++)
	  p[i] = nan;
#    endif
    }

    void
    checkNan() const
    {
#    ifndef QDP_IS_QDPJIT
	std::size_t vol = superbblas::detail::volume<N>(dim);
	T* p = data.get();
	for (std::size_t i = 0; i < vol; i++)
	  assert(detail::IsFinite<T>::get(p[i]));
#    endif
      }
#  endif
    };

    //     inline void* getQDPPtrFromId(int id)
    //     {
    // #  ifdef QDP_IS_QDPJIT
    //       std::vector<QDPCache::ArgKey> v(id, 1);
    //       return QDP_get_global_cache().get_kernel_args(v, false)[0];
    // #  else
    //       return nullptr;
    // #  endif
    //     }

    template <typename T>
    void* getQDPPtr(const T& t)
    {
#  ifdef QDP_IS_QDPJIT
      std::vector<QDPCache::ArgKey> v(1, t.getId());
      void* r = QDP_get_global_cache().get_dev_ptrs(v)[0];
      assert(superbblas::detail::getPtrDevice(r) >= 0);
      return r;
#  else
      return t.getF();
#  endif
    }

    template <typename T>
    using LatticeColorVectorT = OLattice<PScalar<PColorVector<RComplex<T>, Nc>>>;

    template <typename T>
    Tensor<Nd + 2, std::complex<T>> asTensorView(const LatticeColorVectorT<T>& v)
    {
      using Complex = std::complex<T>;
      Complex* v_ptr = reinterpret_cast<Complex*>(v.getF());
      return Tensor<Nd + 2, Complex>("cxyztX", latticeSize<Nd + 2>("cxyztX"), OnHost, OnEveryone,
				     std::shared_ptr<Complex>(v_ptr, [](Complex*) {}));
    }

#  ifndef QDP_IS_QDPJIT
    inline Tensor<Nd + 3, Complex> asTensorView(const LatticeFermion& v)
    {
      Complex* v_ptr = reinterpret_cast<Complex*>(v.getF());
      return Tensor<Nd + 3, Complex>("csxyztX", latticeSize<Nd + 3>("csxyztX"), OnHost, OnEveryone,
				     std::shared_ptr<Complex>(v_ptr, [](Complex*) {}));
    }
#  else
    inline Tensor<Nd + 4, REAL> asTensorView(const LatticeFermion& v)
    {
      REAL* v_ptr = reinterpret_cast<REAL*>(getQDPPtr(v));
      return Tensor<Nd + 4, REAL>("xyztXsc.", latticeSize<Nd + 4>("xyztXsc."), OnDefaultDevice,
				  OnEveryone, std::shared_ptr<REAL>(v_ptr, [](REAL*) {}));
    }
#  endif

#  ifndef QDP_IS_QDPJIT
    inline Tensor<Nd + 1, Complex> asTensorView(const LatticeComplex& v)
    {
      Complex* v_ptr = reinterpret_cast<Complex*>(v.getF());
      return Tensor<Nd + 1, Complex>("xyztX", latticeSize<Nd + 1>("xyztX"), OnHost, OnEveryone,
				     std::shared_ptr<Complex>(v_ptr, [](Complex*) {}));
    }
#  else
    inline Tensor<Nd + 2, REAL> asTensorView(const LatticeComplex& v)
    {
      REAL* v_ptr = reinterpret_cast<REAL*>(getQDPPtr(v));
      return Tensor<Nd + 2, REAL>("xyztX.", latticeSize<Nd + 2>("xyztX."), OnDefaultDevice,
				  OnEveryone, std::shared_ptr<REAL>(v_ptr, [](REAL*) {}));
    }
#  endif

#  ifndef QDP_IS_QDPJIT
    inline Tensor<Nd + 3, Complex> asTensorView(const LatticeColorMatrix& v)
    {
      Complex* v_ptr = reinterpret_cast<Complex*>(v.getF());
      return Tensor<Nd + 3, Complex>("jixyztX",
				     latticeSize<Nd + 3>("jixyztX", {{'i', Nc}, {'j', Nc}}), OnHost,
				     OnEveryone, std::shared_ptr<Complex>(v_ptr, [](Complex*) {}));
    }
#  else
    inline Tensor<Nd + 4, REAL> asTensorView(const LatticeColorMatrix& v)
    {
      REAL* v_ptr = reinterpret_cast<REAL*>(getQDPPtr(v));
      return Tensor<Nd + 4, REAL>(
	"xyztXji.", latticeSize<Nd + 4>("xyztXji.", {{'i', Nc}, {'j', Nc}}), OnDefaultDevice,
	OnEveryone, std::shared_ptr<REAL>(v_ptr, [](REAL*) {}));
    }
#  endif

    inline Tensor<Nd + 4, Complex> asTensorView(const LatticeColorVectorSpinMatrix& v)
    {
      Complex* v_ptr = reinterpret_cast<Complex*>(v.getF());
      return Tensor<Nd + 4, Complex>(
	"cjixyztX", latticeSize<Nd + 4>("cjixyztX", {{'i', Ns}, {'j', Ns}}), OnHost, OnEveryone,
	std::shared_ptr<Complex>(v_ptr, [](Complex*) {}));
    }

    template <typename COMPLEX>
    Tensor<1, COMPLEX> asTensorView(std::vector<COMPLEX>& v,
				    Distribution dist = OnEveryoneReplicated)
    {
      return Tensor<1, COMPLEX>("i", Coor<1>{Index(v.size())}, OnHost, dist,
				std::shared_ptr<COMPLEX>(v.data(), [](COMPLEX*) {}));
    }

    inline Tensor<2, Complex> asTensorView(SpinMatrix& smat)
    {
      Complex* v_ptr = reinterpret_cast<Complex*>(smat.getF());
      return Tensor<2, Complex>("ji", Coor<2>{Ns, Ns}, OnHost, OnEveryoneReplicated,
				std::shared_ptr<Complex>(v_ptr, [](Complex*) {}));
    }

    inline SpinMatrix SpinMatrixIdentity()
    {
      SpinMatrix one;
      // valgrind complains if all elements of SpinMatrix are not initialized!
      for (int i = 0; i < Ns; ++i)
	for (int j = 0; j < Ns; ++j)
	  pokeSpin(one, cmplx(Real(0), Real(0)), i, j);
      for (int i = 0; i < Ns; ++i)
	pokeSpin(one, cmplx(Real(1), Real(0)), i, i);
      return one;
    }

    template <typename COMPLEX = Complex>
    Tensor<2, COMPLEX> Gamma(int gamma, DeviceHost dev = OnDefaultDevice)
    {
      SpinMatrix g = QDP::Gamma(gamma) * SpinMatrixIdentity();
      Tensor<2, COMPLEX> r("ij", {Ns, Ns}, dev, OnEveryoneReplicated);
      asTensorView(g).copyTo(r);
      return r;
    }

    /// Broadcast a string from process zero
    inline std::string broadcast(const std::string& s)
    {
      // Broadcast the size of the string
      std::vector<float> size_orig(1, s.size()), size_dest(1, 0);
      asTensorView(size_orig, OnMaster).copyTo(asTensorView(size_dest));

      // Broadcast the content of the string
      std::vector<float> orig(s.begin(), s.end());
      orig.resize(size_dest[0]);
      std::vector<float> dest(size_dest[0]);
      asTensorView(orig, OnMaster).copyTo(asTensorView(dest));
      return std::string(dest.begin(), dest.end());
    }

    template <std::size_t N, typename T>
    struct StorageTensor {
      static_assert(superbblas::supported_type<T>::value, "Not supported type");

    public:
      std::string filename; ///< Storage file
      std::string metadata; ///< metadata
      std::string order;    ///< Labels of the tensor dimensions
      Coor<N> dim;	    ///< Length of the tensor dimensions
      Sparsity sparsity;    ///< Sparsity of the storage
      std::shared_ptr<superbblas::detail::Storage_context_abstract>
	ctx;	    ///< Superbblas storage handler
      Coor<N> from; ///< First active coordinate in the tensor
      Coor<N> size; ///< Number of active coordinates on each dimension
      T scalar;	    ///< Scalar factor of the tensor

      // Empty constructor
      StorageTensor()
	: filename{},
	  metadata{},
	  order(detail::getTrivialOrder(N)),
	  dim{},
	  sparsity(Dense),
	  ctx{},
	  from{},
	  size{},
	  scalar{0}
      {
      }

      // Create storage construct
      StorageTensor(const std::string& filename, const std::string& metadata,
		    const std::string& order, Coor<N> dim, Sparsity sparsity = Dense,
		    checksum_type checksum = checksum_type::NoChecksum)
	: filename(filename),
	  metadata(metadata),
	  order(order),
	  dim(dim),
	  sparsity(sparsity),
	  from{},
	  size{dim},
	  scalar{1}
      {
	checkOrder();
	superbblas::Storage_handle stoh;
	superbblas::create_storage<N, T>(dim, superbblas::FastToSlow, filename.c_str(),
					 metadata.c_str(), metadata.size(), checksum,
					 MPI_COMM_WORLD, &stoh);
	ctx = std::shared_ptr<superbblas::detail::Storage_context_abstract>(
	  stoh, [=](superbblas::detail::Storage_context_abstract* ptr) {
	    superbblas::close_storage<N, T>(ptr, MPI_COMM_WORLD);
	  });

	// If the tensor to store is dense, create the block here; otherwise, create the block on copy
	if (sparsity == Dense)
	{
	  superbblas::PartitionItem<N> p{Coor<N>{}, dim};
	  superbblas::append_blocks<N, T>(&p, 1, stoh, MPI_COMM_WORLD, superbblas::FastToSlow);
	}
      }

      // Open storage construct
      StorageTensor(const std::string& filename, bool read_order = true,
		    const Maybe<std::string>& order_tag = none)
	: filename(filename), sparsity(Sparse), from{}, scalar{1}
      {
	// Read information from the storage
	superbblas::values_datatype values_dtype;
	std::vector<char> metadatav;
	std::vector<superbblas::IndexType> dimv;
	superbblas::read_storage_header(filename.c_str(), superbblas::FastToSlow, values_dtype,
					metadatav, dimv, MPI_COMM_WORLD);

	// Check that storage tensor dimension and value type match template arguments
	if (dimv.size() != N)
	  throw std::runtime_error(
	    "The storage tensor dimension does not match the template parameter N");
	if (superbblas::detail::get_values_datatype<T>() != values_dtype)
	  throw std::runtime_error("Storage type does not match template argument T");

	// Fill out the information of this class with storage header information
	std::copy(dimv.begin(), dimv.end(), dim.begin());
	size = dim;
	metadata = std::string(metadatav.begin(), metadatav.end());

	// Read the order
	if (read_order)
	{
	  std::istringstream is(metadata);
	  XMLReader xml_buf(is);
	  read(xml_buf, order_tag.getSome("order"), order);
	  checkOrder();
	}

	superbblas::Storage_handle stoh;
	superbblas::open_storage<N, T>(filename.c_str(), MPI_COMM_WORLD, &stoh);
	ctx = std::shared_ptr<superbblas::detail::Storage_context_abstract>(
	  stoh, [=](superbblas::detail::Storage_context_abstract* ptr) {
	    superbblas::close_storage<N, T>(ptr, MPI_COMM_WORLD);
	  });
      }

    protected:
      // Construct a slice/scale storage
      StorageTensor(const StorageTensor& t, const std::string& order, Coor<N> from, Coor<N> size,
		    T scalar)
	: filename(t.filename),
	  metadata(t.metadata),
	  order(order),
	  dim(t.dim),
	  ctx(t.ctx),
	  sparsity(t.sparsity),
	  from(normalize_coor(from, t.dim)),
	  size(size),
	  scalar{t.scalar}
      {
	checkOrder();
      }

    public:
      /// Return whether the tensor is not empty
      explicit operator bool() const noexcept
      {
	return superbblas::detail::volume(size) > 0;
      }

      // Return the dimensions of the tensor
      std::map<char, int> kvdim() const
      {
	std::map<char, int> d;
	for (unsigned int i = 0; i < N; ++i)
	  d[order[i]] = size[i];
	return d;
      }

      /// Rename dimensions
      StorageTensor<N, T> rename_dims(const SB::remap& m) const
      {
	return StorageTensor<N, T>(*this, detail::update_order<N>(order, m), this->from,
				   this->size);
      }

      // Return a slice of the tensor starting at coordinate `kvfrom` and taking `kvsize` elements in each direction.
      // The missing dimension in `kvfrom` are set to zero and the missing direction in `kvsize` are set to the active size of the tensor.
      StorageTensor<N, T> kvslice_from_size(const std::map<char, int>& kvfrom = {},
					    const std::map<char, int>& kvsize = {}) const
      {
	std::map<char, int> updated_kvsize = this->kvdim();
	for (const auto& it : kvsize)
	  updated_kvsize[it.first] = it.second;
	return slice_from_size(kvcoors<N>(order, kvfrom), kvcoors<N>(order, updated_kvsize));
      }

      // Return a slice of the tensor starting at coordinate `from` and taking `size` elements in each direction.
      StorageTensor<N, T> slice_from_size(Coor<N> from, Coor<N> size) const
      {
	for (unsigned int i = 0; i < N; ++i)
	{
	  if (size[i] > this->size[i])
	    throw std::runtime_error(
	      "The size of the slice cannot be larger than the original tensor");
	  if (normalize_coor(from[i], this->size[i]) + size[i] > this->size[i] &&
	      this->size[i] != this->dim[i])
	    throw std::runtime_error(
	      "Unsupported to make a view on a non-contiguous range on the tensor");
	}

	using superbblas::detail::operator+;
	return StorageTensor<N, T>(*this, order, this->from + from, size, scalar);
      }

      StorageTensor<N, T> scale(T s) const
      {
	return StorageTensor<N, T>(*this, order, from, scalar * s);
      }

      void release()
      {
	dim = {};
	ctx.reset();
	from = {};
	size = {};
	scalar = T{0};
	filename = "";
	metadata = "";
      }

      /// Check that the dimension labels are valid

      void checkOrder() const
      {
	// Check that all labels are different there are N
	detail::check_order<N>(order);

	for (auto s : size)
	  if (s < 0)
	    std::runtime_error("Invalid tensor size: it should be positive");
      }

      /// Preallocate space for the storage file
      /// \param size: expected final file size in bytes

      void preallocate(std::size_t size)
      {
	superbblas::preallocate_storage(ctx.get(), size);
      }

      /// Save content from the storage into the given tensor
      template <std::size_t Nw, typename Tw,
		typename std::enable_if<
		  detail::is_complex<T>::value == detail::is_complex<Tw>::value, bool>::type = true>
      void copyFrom(Tensor<Nw, Tw> w) const
      {
	Coor<N> wsize = kvcoors<N>(order, w.kvdim(), 1, NoThrow);
	for (unsigned int i = 0; i < N; ++i)
	  if (wsize[i] > size[i])
	    throw std::runtime_error("The destination tensor is smaller than the source tensor");

	MPI_Comm comm = (w.dist == Local ? MPI_COMM_SELF : MPI_COMM_WORLD);

	// If the storage is sparse, add blocks for the new content
	if (sparsity == Sparse)
	{
	  superbblas::append_blocks<Nw, N, T>(w.p->p.data(), w.p->p.size(), w.order.c_str(), w.from,
					      w.size, order.c_str(), from, ctx.get(), comm,
					      superbblas::FastToSlow);
	}

	Tw* w_ptr = w.data.get();
	superbblas::save<Nw, N, Tw, T>(detail::safe_div<Tw>(w.scalar, scalar), w.p->p.data(), 1,
				       w.order.c_str(), w.from, w.size, (const Tw**)&w_ptr, &*w.ctx,
				       order.c_str(), from, ctx.get(), comm,
				       superbblas::FastToSlow);
      }

      /// Load content from the storage into the given tensor
      template <std::size_t Nw, typename Tw,
		typename std::enable_if<
		  detail::is_complex<T>::value == detail::is_complex<Tw>::value, bool>::type = true>
      void copyTo(Tensor<Nw, Tw> w) const
      {
	Coor<N> wsize = kvcoors<N>(order, w.kvdim(), 1, NoThrow);
	for (unsigned int i = 0; i < N; ++i)
	  if (size[i] > wsize[i])
	    throw std::runtime_error("The destination tensor is smaller than the source tensor");

	Tw* w_ptr = w.data.get();
	MPI_Comm comm = (w.dist == Local ? MPI_COMM_SELF : MPI_COMM_WORLD);
	superbblas::load<N, Nw, T, Tw>(detail::safe_div<T>(scalar, w.scalar), ctx.get(),
				       order.c_str(), from, size, w.p->p.data(), 1, w.order.c_str(),
				       w.from, &w_ptr, &*w.ctx, comm, superbblas::FastToSlow,
				       superbblas::Copy);
      }
    };

    /// Return a tensor filled with the value of the function applied to each element
    /// \param order: dimension labels, they should start with "xyztX"
    /// \param size: length of each dimension
    /// \param dev: either OnHost or OnDefaultDevice
    /// \param func: function (Coor<N-1>) -> COMPLEX

    template <std::size_t N, typename COMPLEX, typename Func>
    Tensor<N, COMPLEX> fillLatticeField(const std::string& order, const std::map<char, int>& size,
					DeviceHost dev, Func func)
    {
      static_assert(N >= 5, "The minimum number of dimensions should be 5");
      if (order.size() < 5 || order.compare(0, 5, "xyztX") != 0)
	throw std::runtime_error("Wrong `order`, it should start with xyztX");

      // Get final object dimension
      Coor<N> dim = latticeSize<N>(order, size);

      // Populate the tensor on CPU
      Tensor<N, COMPLEX> r(order, dim, OnHost);
      Coor<N> local_latt_size = r.p->localSize(); // local dimensions for xyztX
      Coor<N> stride =
	superbblas::detail::get_strides<Nd + 1>(local_latt_size, superbblas::FastToSlow);
      Coor<N> local_latt_from =
	r.p->localFrom(); // coordinates of first elements stored locally for xyztX
      std::size_t vol = superbblas::detail::volume(local_latt_size);
      Index nX = r.kvdim()['X'];
      COMPLEX* ptr = r.data.get();

#  ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#  endif
      for (std::size_t i = 0; i < vol; ++i)
      {
	// Get the global coordinates
	using superbblas::detail::operator+;
	Coor<N> c = normalize_coor(
	  superbblas::detail::index2coor(i, local_latt_size, stride) + local_latt_from, dim);

	// Translate even-odd coordinates to natural coordinates
	Coor<N - 1> coor;
	coor[0] = c[0] * 2 + (c[1] + c[2] + c[3] + c[4]) % nX; // x
	coor[1] = c[1];					       // y
	coor[2] = c[2];					       // z
	coor[3] = c[3];					       // t
	std::copy_n(c.begin() + 5, N - 5, coor.begin() + 4);

	// Call the function
	ptr[i] = func(coor);
      }

      return r.make_sure(none, dev);
    }

    /// Compute a shift of v onto the direction dir
    /// \param v: tensor to apply the displacement
    /// \param first_tslice: global index in the t direction of the first element
    /// \param len: step of the displacement
    /// \param dir: 0 is x; 1 is y...

    template <typename COMPLEX, std::size_t N>
    Tensor<N, COMPLEX> shift(const Tensor<N, COMPLEX> v, Index first_tslice, int len, int dir,
			     Maybe<Action> action = none, Maybe<Tensor<N, COMPLEX>> w=none)
    {
      if (dir < 0 || dir >= Nd - 1)
	throw std::runtime_error("Invalid direction");

      if (action.hasSome() != w.hasSome())
	throw std::runtime_error("Invalid default value");

      // Address zero length case
      if (len == 0)
      {
	if (!w.hasSome())
	  return v;
	v.doAction(action.getSome(), w.getSome());
	return w.getSome();
      }

      // NOTE: chroma uses the reverse convention for direction: shifting FORWARD moves the sites on the negative direction
      len = -len;

      const char dir_label[] = "xyz";
#  if QDP_USE_LEXICO_LAYOUT
      // If we are not using red-black ordering, return a view where the tensor is shifted on the given direction
      v = v.kvslice_from_size({{dir_label[dir], -len}});

      if (!w.hasSome())
	return v;

      v.doAction(action, w.getSome());
      return w.getSome();

#  elif QDP_USE_CB2_LAYOUT
      // Assuming that v has support on the origin and destination lattice elements
      int dimX = v.kvdim()['X'];
      if (dimX != 2 && len % 2 != 0)
	throw std::runtime_error("Unsupported shift");

      if (dir != 0)
      {
	if (!w.hasSome())
	  return v.kvslice_from_size({{'X', -len}, {dir_label[dir], -len}});
	v.doAction(action.getSome(),
		   w.getSome().kvslice_from_size({{'X', len}, {dir_label[dir], len}}));
	return w.getSome();
      }
      else
      {
	int t = v.kvdim()['t'];
	if (t > 1 && t % 2 == 1)
	  throw std::runtime_error(
	    "The t dimension should be zero, one, or even when doing shifting on the X dimension");
	int maxT = std::min(2, t);
	auto v_eo = v.split_dimension('y', "Yy", 2)
		      .split_dimension('z', "Zz", 2)
		      .split_dimension('t', "Tt", maxT);
	Tensor<N, COMPLEX> r = w.hasSome() ? w.getSome() : v.like_this();
	auto r_eo = r.split_dimension('y', "Yy", 2)
		      .split_dimension('z', "Zz", 2)
		      .split_dimension('t', "Tt", maxT);
	while (len < 0)
	  len += v.kvdim()['x'] * 2;
	for (int T = 0; T < maxT; ++T)
	{
	  for (int Z = 0; Z < 2; ++Z)
	  {
	    for (int Y = 0; Y < 2; ++Y)
	    {
	      for (int X = 0; X < 2; ++X)
	      {
		auto v_eo_slice = v_eo.kvslice_from_size({{'X', X}, {'Y', Y}, {'Z', Z}, {'T', T}},
							 {{'X', 1}, {'Y', 1}, {'Z', 1}, {'T', 1}});
		auto r_eo_slice =
		  r_eo.kvslice_from_size({{'X', X + len},
					  {'x', (len + ((X + Y + Z + T + first_tslice) % 2)) / 2},
					  {'Y', Y},
					  {'Z', Z},
					  {'T', T}},
					 {{'Y', 1}, {'Z', 1}, {'T', 1}});
		v_eo_slice.doAction(action.getSome(CopyTo), r_eo_slice);
	      }
	    }
	  }
	}
	return r;
      }
#  else
      throw std::runtime_error("Unsupported layout");
#  endif
    }

    /// Compute a displacement of v onto the direction dir
    /// \param u: Gauge field
    /// \param v: tensor to apply the displacement
    /// \param first_tslice: global index in the t direction of the first element
    /// \param dir: 0: nothing; 1: forward x; -1: backward x; 2: forward y...

    template <typename COMPLEX, std::size_t N>
    Tensor<N, COMPLEX> displace(const std::vector<Tensor<Nd + 3, COMPLEX>>& u, Tensor<N, COMPLEX> v,
				Index first_tslice, int dir, Maybe<Action> action = none,
				Maybe<Tensor<N, COMPLEX>> w = none)
    {
      if (std::abs(dir) > Nd)
	throw std::runtime_error("Invalid direction");

      if (action.hasSome() != w.hasSome())
	throw std::runtime_error("Invalid default value");

      // Address the zero direction case
      if (dir == 0)
      {
	if (!w.hasSome())
	  return v;
	v.doAction(action.getSome(), w.getSome());
	return w.getSome();
      }

      int d = std::abs(dir) - 1;    // space lattice direction, 0: x, 1: y, 2: z
      int len = (dir > 0 ? 1 : -1); // displacement unit direction
      assert(d < u.size());

      if (len > 0)
      {
	// Do u[d] * shift(x,d)
	Tensor<N, COMPLEX> r = w.hasSome() ? w.getSome() : v.like_this();
	v = shift(std::move(v), first_tslice, len, d);
	r.contract(std::move(v), {}, NotConjugate, u[d], {{'j', 'c'}}, NotConjugate, {{'c', 'i'}},
		   action.getSome(CopyTo) == CopyTo ? 0.0 : 1.0);
	return r;
      }
      else
      {
	// Do shift(adj(u[d]) * x,d)
	Tensor<N, COMPLEX> r = v.like_this();
	r.contract(std::move(v), {}, NotConjugate, u[d], {{'i', 'c'}}, Conjugate, {{'c', 'j'}});
	return shift(std::move(r), first_tslice, len, d, action, w);
      }
    }

    /// Apply right nabla onto v on the direction dir
    /// \param u: Gauge field
    /// \param v: tensor to apply the derivative
    /// \param first_tslice: global index in the t direction of the first element
    /// \param dir: 0: nothing; 1: forward x; -1: backward x; 2: forward y...
    ///
    /// NOTE: the code returns U_\mu(x)f(x+\mu) - U_{-\mu}(x)f(x-\mu)

    template <typename COMPLEX, std::size_t N>
    Tensor<N, COMPLEX> rightNabla(const std::vector<Tensor<Nd + 3, COMPLEX>>& u,
				  Tensor<N, COMPLEX> v, Index first_tslice, int dir)
    {
      auto r = displace(u, v, first_tslice, dir);
      displace(u, v, first_tslice, -dir).scale(-1).addTo(r);
      return r;
    }

    /// Compute a displacement of v onto the direction dir
    /// \param u: Gauge field
    /// \param v: tensor to apply the derivative
    /// \param first_tslice: global index in the t direction of the first element
    /// \param dir: 0: nothing; 1: forward x; -1: backward x; 2: forward y...
    /// \param moms: list of input momenta
    /// \param conjUnderAdd: if true, return a version, R(dir), so that
    ////       adj(R(dir)) * D(dir') == D(dir+dir'), where D(dir') is what this function returns
    ////       when conjUnderAdd is false.

    template <typename COMPLEX, std::size_t N>
    Tensor<N, COMPLEX> leftRightNabla(const std::vector<Tensor<Nd + 3, COMPLEX>>& u,
				      Tensor<N, COMPLEX> v, Index first_tslice, int dir,
				      const std::vector<Coor<3>>& moms = {},
				      bool conjUnderAdd = false)
    {
      if (std::abs(dir) > Nd)
	throw std::runtime_error("Invalid direction");

      int d = std::abs(dir) - 1; // space lattice direction, 0: x, 1: y, 2: z

      // conj(phase)*displace(u, v, -dir) - phase*displace(u, v, dir)
      std::vector<COMPLEX> phases(moms.size());
      for (unsigned int i = 0; i < moms.size(); ++i)
      {

	typename COMPLEX::value_type angle = 2 * M_PI * moms[i][d] / Layout::lattSize()[d];
	phases[i] = COMPLEX{1} + COMPLEX{cos(angle), sin(angle)};
	if (conjUnderAdd)
	  phases[i] = std::sqrt(phases[i]);
      }

      // r = conj(phases) * displace(u, v, dir)
      Tensor<N, COMPLEX> r = v.like_this("c%xyzXtm", '%');
      r.contract(displace(u, v, first_tslice, -dir), {}, NotConjugate,
		 asTensorView(phases), {{'i', 'm'}}, Conjugate);

      // r = r - phases * displace(u, v, dir) if !ConjUnderAdd else r + phases * displace(u, v, dir)
      r.contract(displace(u, v, first_tslice, dir).scale(conjUnderAdd ? 1 : -1),
		 {}, NotConjugate, asTensorView(phases), {{'i', 'm'}}, NotConjugate, {}, 1.0);

      return r;
    }

    // template <std::size_t N, typename T>
    // class Transform : public Tensor<N,T> {
    // public:
    //   Transform(Tensor<N, T> t, remap input, remat output, std::string no_contract = std::string(),
    //     	bool conj = false)
    //     : Tensor<N, T>(t), input(input), output(output), no_contract(no_contract), conj(conj)
    //   {
    //   }

    //   Transform<N, T> rename_dims(remap m) const
    //   {
    //     remap new_input = input, new_output = output;
    //     std::string new_no_contract = new_constract;
    //     for (auto& it : new_input)
    //     {
    //       auto j = m.find(it->first);
    //       if (j != m.end())
    //         it->second = j->second;
    //     }
    //   	for (auto& it : new_output)
    //     {
    //       auto j = m.find(it->first);
    //       if (j != m.end())
    //         it->second = j->second;
    //     }
    //     for (auto& it : m) {
    //         auto n = new_no_contract.find(it->first);
    //         if (n != std::string::npos)
    //           new_no_contract[n] = it->second;
    //     }
    //     return Transform<N, T>(*this, new_input, new_output, new_no_contract, conj);
    //   }

    //   const remap input;  ///< rename dimensions before contraction
    //   const remap output; ///< rename dimensions after contraction
    //   const std::string
    //     no_contract; ///< dimension labels that should not be on the other contracted tensor
    //   const bool conj;

    //   virtual Transform<N, T> adj() const
    //   {
    //     return Transform<N, T>{*this, input, output, no_contract, !conj};
    //   }
    // };

    // template <std::size_t Nt>
    // Tensor<Nt, T> contractTo(Tensor<Nt, T> t) const
    // {
    // }

    // template <typename T>
    // class MatrixTransform : Transform<2, T>
    // {
    // public:
    //   MatrixTransform(Transform<2, T> t) : Transform<2, T>(t)
    //   {
    //   }

    //   MatrixTransform<N, T> adj() const override
    //   {
    //     remap m{{order[0], order[1]}, {order[1], order[0]}};
    //     return MatrixTransform<N, T>{
    //       Transform<N, T>{this->Tensor<N, T>::rename_dims(m), input, output, no_contract, !conj}};
    //   }
    // };

    //
    // Get colorvecs
    //

    typedef QDP::MapObjectDiskMultiple<KeyTimeSliceColorVec_t, Tensor<Nd, ComplexF>> MODS_t;
    typedef QDP::MapObjectDisk<KeyTimeSliceColorVec_t, Tensor<Nd, ComplexF>> MOD_t;

    // Represent either FILEDB or S3T handle for file containing distillation vectors

    struct ColorvecsStorage {
      std::shared_ptr<MODS_t> mod;	   // old storage
      StorageTensor<Nd + 2, ComplexD> s3t; // cxyztn
    };

    namespace ns_getColorvecs
    {
      /// Return the permutation from a natural layout to a red-black that is used by `applyPerm`
      /// \param t: index in the t-direction of the input elements
      /// NOTE: assuming the input layout is xyz and the output layout is xyzX for the given input
      ///       time-slice `t`

      inline std::vector<Index> getPermFromNatToRB(Index t)
      {
	if (Layout::nodeNumber() != 0)
	  return {};

	const Index x1 = Layout::lattSize()[0];
	const Index y1 = Layout::lattSize()[1];
	const Index z1 = Layout::lattSize()[2];
	std::vector<Index> perm(x1 * y1 * z1);

#  if QDP_USE_LEXICO_LAYOUT
	unsigned int n = x1 * y1 * z1;
#    ifdef _OPENMP
#      pragma omp parallel for schedule(static)
#    endif
	for (unsigned int i = 0; i < n; ++n)
	  perm[i] = i;

#  elif QDP_USE_CB2_LAYOUT
#    ifdef _OPENMP
#      pragma omp parallel for collapse(3) schedule(static)
#    endif
	for (unsigned int z = 0; z < z1; ++z)
	{
	  for (unsigned int y = 0; y < y1; ++y)
	  {
	    for (unsigned int x = 0; x < x1; ++x)
	    {
	      // index on natural ordering
	      Index i0 = x + y * x1 + z * x1 * y1;
	      // index in red-black
	      Index i1 = x / 2 + y * (x1 / 2) + z * (x1 * y1 / 2) +
			 ((x + y + z + t) % 2) * (x1 * y1 * z1 / 2);
	      perm[i1] = i0;
	    }
	  }
	}

#  else
	throw std::runtime_error("Unsupported layout");
#  endif

	return perm;
      }

      /// Apply a permutation generated by `getPermFromNatToRB`
      /// \param perm: permutation generated with getPermFromNatToRB
      /// \param tnat: input tensor with ordering cxyz
      /// \param trb: output tensor with ordering cxyzX

      template <typename T>
      void toRB(const std::vector<Index>& perm, Tensor<Nd, T> tnat, Tensor<Nd + 1, T> trb)
      {
	assert(tnat.order == "cxyz");
	assert(trb.order == "cxyzX");
	assert(tnat.p->localVolume() == perm.size() * Nc);

	unsigned int i1 = perm.size();
	const T* x = tnat.data.get();
	T* y = trb.data.get();

#  ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#  endif
	for (unsigned int i = 0; i < i1; ++i)
	  for (unsigned int c = 0; c < Nc; ++c)
	    y[i * Nc + c] = x[perm[i] * Nc + c];
      }

      /// Apply a permutation generated by `getPermFromNatToRB`
      /// \param perm: permutation generated with getPermFromNatToRB
      /// \param tnat: input tensor with ordering cxyz
      /// \param trb: output tensor with ordering cxyzX

      template <typename T>
      void toNat(const std::vector<Index>& perm, Tensor<Nd + 1, T> trb, Tensor<Nd, T> tnat)
      {
	assert(tnat.order == "cxyz");
	assert(trb.order == "cxyzX");
	assert(tnat.p->localVolume() == perm.size() * Nc);

	unsigned int i1 = perm.size();
	T* x = tnat.data.get();
	const T* y = trb.data.get();

#  ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#  endif
	for (unsigned int i = 0; i < i1; ++i)
	  for (unsigned int c = 0; c < Nc; ++c)
	    x[perm[i] * Nc + c] = y[i * Nc + c];
      }

      /// Return a lattice field with value exp(2*pi*(x./dim)'*phase) for each lattice site x
      /// \param phase: integer phase
      /// \param dev: device of the returned tensor

      template <typename T>
      Tensor<Nd + 1, T> getPhase(Coor<Nd - 1> phase, DeviceHost dev = OnDefaultDevice)
      {
	// Get spatial dimensions of the current lattice
	Coor<Nd> dim = latticeSize<Nd>("xyzX", {});
	dim[0] *= dim[3];
	return fillLatticeField<5, T>("xyztX", {}, dev, [=](Coor<Nd> c) {
	  typename T::value_type phase_dot_coor = 0;
	  for (int i = 0; i < Nd - 1; ++i)
	    phase_dot_coor += c[i] * 2 * M_PI * phase[i] / dim[i];

	  return T{cos(phase_dot_coor), sin(phase_dot_coor)};
	});
      }

      // NOTE: for now, the GPU version requires MAGMA
#  if defined(BUILD_PRIMME) && (!defined(QDP_IS_QDPJIT) || defined(BUILD_MAGMA))

      // Apply the laplacian operator on the spatial dimensions
      /// \param u: Gauge fields restricted to the same t-slice as chi and psi
      /// \param first_tslice: global t index of the zero t index
      /// \param chi: output vector
      /// \param psi: input vector

      inline void LaplacianOperator(const std::vector<Tensor<Nd + 3, ComplexD>>& u,
				    Index first_tslice, Tensor<Nd + 3, ComplexD> chi,
				    const Tensor<Nd + 3, ComplexD> psi)
      {
	int N = Nd - 1; // Only the spatial dimensions

	// chi = -2*N*psi
	psi.scale(-2 * N).copyTo(chi);

	// I have no idea how to do this....
	using MaybeTensor = Maybe<Tensor<Nd + 3, ComplexD>>;

	for (int mu = 0; mu < N; ++mu)
	{
	  displace(u, psi, first_tslice, mu + 1, Action::AddTo, MaybeTensor(chi));
	  displace(u, psi, first_tslice, -(mu + 1), Action::AddTo, MaybeTensor(chi));
	}
      }

      // Auxiliary structure passed to PRIMME's matvec

      struct OperatorAux {
	const std::vector<Tensor<Nd + 3, ComplexD>> u; // Gauge fields
	const Index first_tslice;		       // global t index
	const std::string order;		       // Laplacian input/output tensor's order
      };

      // Wrapper for PRIMME of `LaplacianOperator`
      /// \param x: pointer to input vector
      /// \param ldx: leading dimension for `x`
      /// \param y: pointer to output vector
      /// \param ldy: leading dimension for `y`
      /// \param blockSize: number of input/output vectors
      /// \param ierr: output error state (zero means ok)

      extern "C" inline void primmeMatvec(void* x, PRIMME_INT* ldx, void* y, PRIMME_INT* ldy,
					  int* blockSize, primme_params* primme, int* ierr)
      {
	*ierr = -1;
	try
	{
	  // The implementation assumes that ldx and ldy is nLocal
	  if (*blockSize > 1 && (*ldx != primme->nLocal || *ldy != primme->nLocal))
	    throw std::runtime_error("We cannot play with the leading dimensions");

	  OperatorAux& opaux = *(OperatorAux*)primme->matrix;
	  Coor<Nd + 3> size = latticeSize<Nd + 3>(opaux.order, {{'n', *blockSize}, {'t', 1}});
	  Tensor<Nd + 3, ComplexD> tx(opaux.order, size, OnDefaultDevice, OnEveryone,
				      std::shared_ptr<ComplexD>((ComplexD*)x, [](ComplexD*) {}));
	  Tensor<Nd + 3, ComplexD> ty(opaux.order, size, OnDefaultDevice, OnEveryone,
				      std::shared_ptr<ComplexD>((ComplexD*)y, [](ComplexD*) {}));
	  LaplacianOperator(opaux.u, opaux.first_tslice, ty, tx);
	  *ierr = 0;
	} catch (...)
	{
	}
      }

      /// Wrapper for PRIMME of a global sum for double
      /// \param sendBuf: pointer to input vector
      /// \param recvBuf: pointer to output vector
      /// \param count: number of elements in the input/output vector
      /// \param primme: pointer to the current primme_params
      /// \param ierr: output error state (zero means ok)

      extern "C" inline void primmeGlobalSum(void* sendBuf, void* recvBuf, int* count,
					     primme_params* primme, int* ierr)
      {
	if (sendBuf == recvBuf)
	{
	  *ierr = MPI_Allreduce(MPI_IN_PLACE, recvBuf, *count, MPI_DOUBLE, MPI_SUM,
				MPI_COMM_WORLD) != MPI_SUCCESS;
	}
	else
	{
	  *ierr = MPI_Allreduce(sendBuf, recvBuf, *count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) !=
		  MPI_SUCCESS;
	}
      }

      /// Compute the eigenpairs of the laplacian operator on the spatial dimensions using PRIMME
      /// \param u: Gauge field
      /// \param from_tslice: index of the first t-slice to compute the eigenvectors from
      /// \param n_tslices: number of tslices to compute
      /// \param n_colorvecs: number of eigenpairs to compute
      /// \param order_: order of the output tensor for the eigenvectors
      /// \return: a pair of the eigenvectors and the eigenvalues

      inline std::pair<Tensor<Nd + 3, ComplexD>, std::vector<std::vector<double>>>
      computeColorvecs(const multi1d<LatticeColorMatrix>& u, int from_tslice, int n_tslices,
		       int n_colorvecs, const Maybe<const std::string>& order_ = none)
      {
	const std::string order = order_.getSome("cxyztXn");
	detail::check_order_contains(order, "cxyztXn");
	Tensor<Nd + 3, ComplexD> all_evecs(
	  order, latticeSize<Nd + 3>(order, {{'n', n_colorvecs}, {'t', n_tslices}}),
	  OnDefaultDevice, OnEveryone);
	std::vector<std::vector<double>> all_evals;

	for (Index t = 0; t < n_tslices; ++t)
	{
	  // Make a copy of the time-slicing of u[d] also supporting left and right
	  std::vector<Tensor<Nd + 3, ComplexD>> ut(Nd);
	  for (unsigned int d = 0; d < Nd - 1; d++)
	  {
	    ut[d] = asTensorView(u[d])
		      .kvslice_from_size({{'t', from_tslice + t}}, {{'t', 1}})
		      .toComplex()
		      .template make_sure<ComplexD>("ijxyztX");
	  }

	  // Create an auxiliary struct for the PRIMME's matvec
	  // NOTE: Please keep 'n' as the slowest index; the rows of vectors taken by PRIMME's matvec has dimensions 'cxyztX',
          // and 'n' is the dimension for the columns.
	  OperatorAux opaux{ut, from_tslice + t, "cxyztXn"};

	  // Make a bigger structure holding
	  primme_params primme;
	  primme_initialize(&primme);

	  // Get the global and local size of evec
	  std::size_t n, nLocal;
	  {
	    Tensor<Nd + 3, Complex> aux_tensor(
	      opaux.order, latticeSize<Nd + 3>(opaux.order, {{'n', 1}, {'t', 1}}), OnDefaultDevice,
	      OnEveryone);
	    n = aux_tensor.volume();
	    nLocal = aux_tensor.getLocal().volume();
	  }

	  if (n_colorvecs > n)
	  {
	    std::cerr << "ERROR: the rank of the distillation basis shouldn't be larger than the "
			 "spatial dimensions"
		      << std::endl;
	    exit(1);
	  }

	  // Primme solver setup
	  primme.numEvals = n_colorvecs;
	  primme.printLevel = 0;
	  primme.n = n;
	  primme.eps = 1e-9;
	  primme.target = primme_largest;

	  // Set parallel settings
	  primme.nLocal = nLocal;
	  primme.numProcs = QDP::Layout::numNodes();
	  primme.procID = QDP::Layout::nodeNumber();
	  primme.globalSumReal = primmeGlobalSum;

	  // No preconditioner for my matrix
	  primme.matrixMatvec = primmeMatvec;
	  primme.matrix = &opaux;

	  // Set block size
          primme.maxBasisSize = 64;
	  primme.maxBlockSize = 4;
          primme.ldOPs = primme.nLocal;

	  // Should set lots of defaults
	  if (primme_set_method(PRIMME_DEFAULT_MIN_TIME, &primme) < 0)
	  {
	    QDPIO::cerr << __func__ << ": invalid preset method\n";
	    QDP_abort(1);
	  }

	  // Allocate space for converged Ritz values and residual norms
	  std::vector<double> evals(primme.numEvals);
	  std::vector<double> rnorms(primme.numEvals);
	  Tensor<Nd + 3, ComplexD> evecs(
	    opaux.order, latticeSize<Nd + 3>(opaux.order, {{'n', primme.numEvals}, {'t', 1}}),
	    OnDefaultDevice, OnEveryone);
#    if defined(QDP_IS_QDPJIT) && defined(BUILD_MAGMA)
	  primme.queue = &*detail::getMagmaContext();
#    endif

	  // Call primme
#    if defined(QDP_IS_QDPJIT) && defined(BUILD_MAGMA)
	  int ret = magma_zprimme(evals.data(), evecs.data.get(), rnorms.data(), &primme);
#    else
	  int ret = zprimme(evals.data(), evecs.data.get(), rnorms.data(), &primme);
#    endif

	  if (primme.procID == 0)
	  {
	    fprintf(stdout, " %d eigenpairs converged for tslice %d\n", primme.initSize,
		    from_tslice + t);
	    fprintf(stdout, "Tolerance : %-22.15E\n", primme.aNorm * primme.eps);
	    fprintf(stdout, "Iterations: %-d\n", (int)primme.stats.numOuterIterations);
	    fprintf(stdout, "Restarts  : %-d\n", (int)primme.stats.numRestarts);
	    fprintf(stdout, "Matvecs   : %-d\n", (int)primme.stats.numMatvecs);
	    fprintf(stdout, "Preconds  : %-d\n", (int)primme.stats.numPreconds);
	    fprintf(stdout, "T. ortho  : %g\n", primme.stats.timeOrtho);
	    fprintf(stdout, "T. matvec : %g\n", primme.stats.timeMatvec);
	    fprintf(stdout, "Total time: %g\n", primme.stats.elapsedTime);
	  }

	  if (ret != 0)
	  {
	    QDPIO::cerr << "Error: primme returned with nonzero exit status\n";
	    QDP_abort(1);
	  }

	  // Cleanup
	  primme_free(&primme);

	  // Check the residuals, |laplacian*v-lambda*v|_2<=|laplacian|*tol
	  if (evals.size() > 0)
	  {
	    auto r = evecs.like_this();
	    LaplacianOperator(opaux.u, opaux.first_tslice, r, evecs);
	    std::vector<std::complex<double>> evals_cmpl(evals.begin(), evals.end());
	    r.contract(evecs, {}, NotConjugate,
		       asTensorView(evals_cmpl).rename_dims({{'i', 'n'}}).scale(-1), {},
		       NotConjugate, {}, 1);
	    std::vector<std::complex<double>> norm2_r(evals.size());
	    asTensorView(norm2_r)
	      .rename_dims({{'i', 'n'}})
	      .contract(r, {}, Conjugate, r, {}, NotConjugate);
	    for (const auto& i : norm2_r)
	    {
	      if (std::sqrt(std::real(i)) > primme.stats.estimateLargestSVal * primme.eps * 10)
	      {
		QDPIO::cerr << "Error: primme returned eigenpairs with too much error\n";
		QDP_abort(1);
	      }
	    }
	  }

	  // Copy evecs into all_evecs
	  evecs.copyTo(all_evecs.kvslice_from_size({{'t', t}}, {{'t', 1}}));
	  all_evals.push_back(evals);
	}

	return {all_evecs, all_evals};
      }
#  else	 // BUILD_PRIMME
      inline std::pair<Tensor<Nd + 3, ComplexD>, std::vector<std::vector<double>>>
      computeColorvecs(const multi1d<LatticeColorMatrix>& u, int from_tslice, int n_tslices,
		       int n_colorvecs, const Maybe<const std::string>& order_ = none)
      {
	(void)u;
	(void)from_tslice;
	(void)n_tslices;
	(void)n_colorvecs;
	(void)order_;
	throw std::runtime_error("Functionality isn't available without compiling with PRIMME");
      }
#  endif // BUILD_PRIMME

      /// Read colorvecs from a FILEDB
      /// \param eigen_source: database handle
      /// \param decay_dir: something we assume is always three
      /// \param from_tslice: first tslice to read
      /// \param n_tslices: number of tslices to read
      /// \param n_colorvecs: number of eigenpairs to read
      /// \param order_: order of the output tensor for the eigenvectors
      /// \return: a tensor containing the eigenvectors

      template <typename COMPLEX = ComplexF>
      Tensor<Nd + 3, COMPLEX> getColorvecs(MODS_t& eigen_source, int decay_dir, int from_tslice,
					   int n_tslices, int n_colorvecs,
					   const Maybe<const std::string>& order_ = none)
      {
	const std::string order = order_.getSome("cxyztXn");
	detail::check_order_contains(order, "cxyztXn");

	from_tslice = normalize_coor(from_tslice, Layout::lattSize()[decay_dir]);

	// Allocate tensor to return
	Tensor<Nd + 3, COMPLEX> r(
	  order, latticeSize<Nd + 3>(order, {{'t', n_tslices}, {'n', n_colorvecs}}));

	// Allocate a single time slice colorvec in natural ordering, as colorvec are stored
	Tensor<Nd, ComplexF> tnat("cxyz", latticeSize<Nd>("cxyz", {{'x', Layout::lattSize()[0]}}),
				  OnHost, OnMaster);

	// Allocate a single time slice colorvec in case of using RB ordering
	Tensor<Nd + 1, ComplexF> trb("cxyzX", latticeSize<Nd + 1>("cxyzX"), OnHost, OnMaster);

	// Allocate all colorvecs for the same time-slice
	Tensor<Nd + 2, ComplexF> t("cxyzXn", latticeSize<Nd + 2>("cxyzXn", {{'n', n_colorvecs}}),
				   OnHost, OnMaster);

	const int Nt = Layout::lattSize()[decay_dir];
	for (int t_slice = from_tslice, i_slice = 0; i_slice < n_tslices;
	     ++i_slice, t_slice = (t_slice + 1) % Nt)
	{
	  // Compute the permutation from natural ordering to red-black
	  std::vector<Index> perm = ns_getColorvecs::getPermFromNatToRB(t_slice);

	  for (int colorvec = 0; colorvec < n_colorvecs; ++colorvec)
	  {
	    // Read a single time-slice and colorvec
	    KeyTimeSliceColorVec_t key(t_slice, colorvec);
	    if (!eigen_source.exist(key))
	      throw std::runtime_error(
		"no colorvec exists with key t_slice= " + std::to_string(t_slice) +
		" colorvec= " + std::to_string(colorvec));
	    eigen_source.get(key, tnat);

	    // Correct ordering
	    ns_getColorvecs::toRB(perm, tnat, trb);

	    // t[n=colorvec] = trb
	    trb.copyTo(t.kvslice_from_size({{'n', colorvec}}, {{'n', 1}}));
	  }

	  // r[t=i_slice] = t, distribute the tensor from master to the rest of the nodes
	  t.copyTo(r.kvslice_from_size({{'t', i_slice}}));
	}

	return r;
      }

      /// Read colorvecs from a S3T file
      /// \param s3t: database handle
      /// \param u: gauge field
      /// \param decay_dir: something we assume is always three
      /// \param from_tslice: first tslice to read
      /// \param n_tslices: number of tslices to read
      /// \param n_colorvecs: number of eigenpairs to read
      /// \param order_: order of the output tensor for the eigenvectors
      /// \return: a tensor containing the eigenvectors

      template <typename COMPLEX = ComplexF>
      Tensor<Nd + 3, COMPLEX> getColorvecs(StorageTensor<Nd + 2, ComplexD> s3t,
					   const multi1d<LatticeColorMatrix>& u, int decay_dir,
					   int from_tslice, int n_tslices, int n_colorvecs,
					   const Maybe<const std::string>& order_ = none)
      {
	const std::string order = order_.getSome("cxyztXn");
	detail::check_order_contains(order, "cxyztXn");

	from_tslice = normalize_coor(from_tslice, Layout::lattSize()[decay_dir]);

	// Read the metadata and check that the file stores colorvecs and from a lattice of the same size
	std::istringstream is(s3t.metadata);
	XMLReader xml_buf(is);
	bool write_fingerprint = false;
	read(xml_buf, "/MODMetaData/fingerprint", write_fingerprint);
	GroupXML_t link_smear =
	  readXMLGroup(xml_buf, "/MODMetaData/LinkSmearing", "LinkSmearingType");

	// Smear the gauge field if needed
	multi1d<LatticeColorMatrix> u_smr = u;
	try
	{
	  std::istringstream xml_l(link_smear.xml);
	  XMLReader linktop(xml_l);
	  Handle<LinkSmearing> linkSmearing(TheLinkSmearingFactory::Instance().createObject(
	    link_smear.id, linktop, "/LinkSmearing"));
	  (*linkSmearing)(u_smr);
	} catch (const std::string& e)
	{
	  QDPIO::cerr << ": Caught Exception link smearing: " << e << std::endl;
	  QDP_abort(1);
	} catch (...)
	{
	  QDPIO::cerr << ": Caught unexpected exception" << std::endl;
	  QDP_abort(1);
	}

	// Allocate tensor with the content of s3t
	Tensor<Nd + 3, ComplexD> colorvecs_s3t(
	  order, latticeSize<Nd + 3>(order, {{'t', n_tslices}, {'n', n_colorvecs}}));

	// Allocate a single time slice colorvec in natural ordering, as colorvec are stored
	Tensor<Nd, ComplexD> tnat("cxyz", latticeSize<Nd>("cxyz", {{'x', Layout::lattSize()[0]}}),
				  OnHost, OnMaster);

	// Allocate a single time slice colorvec in case of using RB ordering
	Tensor<Nd + 1, ComplexD> trb("cxyzX", latticeSize<Nd + 1>("cxyzX"), OnHost, OnMaster);

	const int Nt = Layout::lattSize()[decay_dir];
	for (int t_slice = from_tslice, i_slice = 0; i_slice < n_tslices;
	     ++i_slice, t_slice = (t_slice + 1) % Nt)
	{
	  // Compute the permutation from natural ordering to red-black
	  std::vector<Index> perm = ns_getColorvecs::getPermFromNatToRB(t_slice);

	  for (int colorvec = 0; colorvec < n_colorvecs; ++colorvec)
	  {
	    // Read a single time-slice and colorvec
	    tnat.set_zero();
	    s3t.kvslice_from_size({{'t', t_slice}, {'n', colorvec}}, {{'t', 1}, {'n', 1}})
	      .copyTo(tnat);

	    // Correct ordering
	    ns_getColorvecs::toRB(perm, tnat, trb);

	    // colorvecs_s3t[t=i_slice,n=colorvec] = trb
	    trb.copyTo(colorvecs_s3t.kvslice_from_size({{'t', i_slice}, {'n', colorvec}}));
	  }
	}

	// Compute the 2-norm of colorvecs_s3t and check that no vector is null

	Tensor<2, ComplexD> colorvecs_s3t_norms2("nt", Coor<2>{n_colorvecs, n_tslices}, OnHost,
						 OnEveryoneReplicated);
	colorvecs_s3t_norms2.contract(colorvecs_s3t, {}, Conjugate, colorvecs_s3t, {},
				      NotConjugate);

	for (int t = 0; t < n_tslices; ++t)
	  for (int n = 0; n < n_colorvecs; ++n)
	    if (std::norm(colorvecs_s3t_norms2.get({n, t})) == 0)
	      throw std::runtime_error(
		"no colorvec exists with key t_slice= " + std::to_string(t + from_tslice) +
		" colorvec= " + std::to_string(n));

	if (write_fingerprint)
	{
	  // Compute the colorvecs
	  auto colorvecs =
	    ns_getColorvecs::computeColorvecs(u_smr, from_tslice, n_tslices, n_colorvecs, order_)
	      .first;

	  // We need to phase the individual eigenvectors so that the have the same phase as the
	  // s3t's colorvecs. That is, we need to apply a phase phi[i] to each eigenvector so that
	  //
	  //    colorvecs_s3t[i] = colorvecs[i] * phi[i].
	  //
	  // We have a subset of the s3t's colorvecs, so we restrict the above equation to that:
	  //
	  //    colorvecs_s3t[i]^\dagger * colorvecs_s3t[i] = colorvecs_s3t[i]^\dagger * colorvecs[i] * phi[i].
	  //
	  // Therefore, phi[i] = (colorvecs_s3t[i]^\dagger * colorvecs_s3t[i]) / (colorvecs_s3t[i]^\dagger * colorvecs[i])

	  auto ip = colorvecs_s3t_norms2.like_this();
	  ip.contract(colorvecs_s3t, {}, Conjugate, colorvecs, {}, NotConjugate);

	  auto phi = ip.like_this();
	  for (int t = 0; t < n_tslices; ++t)
	  {
	    for (int n = 0; n < n_colorvecs; ++n)
	    {
	      auto phi_i = colorvecs_s3t_norms2.get({n, t}) / ip.get({n, t});
	      if (std::fabs(std::fabs(phi_i) - 1) > 1e-4)
		throw std::runtime_error(
		  "The colorvec fingerprint does not correspond to current gates field");
	      phi.set({n, t}, phi_i);
	    }
	  }

	  // Apply the phase of the colorvecs in s3t to the computed colorvecs
	  colorvecs_s3t.contract(colorvecs, {}, NotConjugate, phi, {}, NotConjugate);
	}

	return colorvecs_s3t.make_sure<COMPLEX>();
      }
    }

    /// Read colorvecs from either a FILEDB or S3T file
    /// \param colorvec_files: filenames
    /// \return: a handle

    inline ColorvecsStorage openColorvecStorage(const std::vector<std::string>& colorvec_files)
    {
      ColorvecsStorage sto{}; // returned object

      std::string metadata; // the metadata content of the file

      // Try to open the file as a s3t database
      try
      {
	if (colorvec_files.size() == 1)
	  sto.s3t = StorageTensor<Nd + 2, ComplexD>(colorvec_files[0], true, "/MODMetaData/order");
	metadata = sto.s3t.metadata;
      } catch (...)
      {
      }

      // Try to open the files as a MOD database
      if (!sto.s3t)
      {
	sto.mod = std::make_shared<MODS_t>();
	sto.mod->setDebug(0);

	try
	{
	  // Open
	  sto.mod->open(colorvec_files);
	  sto.mod->getUserdata(metadata);
	} catch (std::bad_cast)
	{
	  QDPIO::cerr << ": caught dynamic cast error" << std::endl;
	  QDP_abort(1);
	} catch (const std::string& e)
	{
	  QDPIO::cerr << ": error extracting source_header: " << e << std::endl;
	  QDP_abort(1);
	} catch (const char* e)
	{
	  QDPIO::cerr << ": Caught some char* exception:" << std::endl;
	  QDPIO::cerr << e << std::endl;
	  QDP_abort(1);
	}
      }

      // Check that the file stores colorvecs and is from a lattice of the same size

      std::istringstream is(metadata);
      XMLReader xml_buf(is);

      std::string id;
      read(xml_buf, "/MODMetaData/id", id);
      if (id != "eigenVecsTimeSlice")
      {
	std::stringstream ss;
	ss << "The file `" << colorvec_files[0] << "' does not contain colorvecs";
	throw std::runtime_error(ss.str());
      }

      multi1d<int> spatialLayout(3);
      read(xml_buf, "/MODMetaData/lattSize", spatialLayout);
      if (spatialLayout[0] != Layout::lattSize()[0] || spatialLayout[1] != Layout::lattSize()[1] ||
	  spatialLayout[2] != Layout::lattSize()[2])
      {
	std::stringstream ss;
	ss << "The spatial dimensions of the colorvecs in `" << colorvec_files[0]
	   << "' do not much the current lattice";
	throw std::runtime_error(ss.str());
      }

      return sto;
    }

    /// Close a colorvec storage
    /// \param sto: colorvec storage handle

    inline void closeColorvecStorage(ColorvecsStorage& sto)
    {
      if (!sto.s3t)
      {
	sto.s3t.release();
      }
      else if (sto.mod)
      {
	sto.mod->close();
	sto.mod.reset();
      }
    }

    /// Phase colorvecs
    /// \param colorvecs: tensor with the colorvecs
    /// \param from_tslice: first tslice of the tensor
    /// \param phase: apply a phase to the eigenvectors
    /// \return: a tensor containing the eigenvectors phased

    template <typename COMPLEX>
    Tensor<Nd + 3, COMPLEX> phaseColorvecs(Tensor<Nd + 3, COMPLEX> colorvecs, int from_tslice,
					   Coor<Nd - 1> phase = {})
    {
      // Phase colorvecs if phase != (0,0,0)
      if (phase == Coor<Nd - 1>{})
	return colorvecs;

      Tensor<Nd + 1, COMPLEX> tphase = ns_getColorvecs::getPhase<COMPLEX>(phase).kvslice_from_size(
	{{'t', from_tslice}}, {{'t', colorvecs.kvdim()['t']}});
      Tensor<Nd + 3, COMPLEX> r = colorvecs.like_this();
      r.contract(colorvecs, {}, NotConjugate, tphase, {}, NotConjugate);
      return r;
    }

    /// Read colorvecs from a handle returned by `openColorvecStorage`
    /// \param sto: database handle
    /// \param u: gauge field
    /// \param decay_dir: something we assume is always three
    /// \param from_tslice: first tslice to read
    /// \param n_tslices: number of tslices to read
    /// \param n_colorvecs: number of eigenpairs to read
    /// \param order: order of the output tensor for the eigenvectors
    /// \param phase: apply a phase to the eigenvectors
    /// \return: a tensor containing the eigenvectors

    template <typename COMPLEX = ComplexF>
    Tensor<Nd + 3, COMPLEX>
    getColorvecs(const ColorvecsStorage& sto, const multi1d<LatticeColorMatrix>& u, int decay_dir,
		 int from_tslice, int n_tslices, int n_colorvecs,
		 const Maybe<const std::string>& order = none, Coor<Nd - 1> phase = {})
    {
      StopWatch sw;
      sw.reset();
      sw.start();

      if (decay_dir != 3)
	throw std::runtime_error("Only support for decay_dir being the temporal dimension");

      // Read the colorvecs with the proper function
      Tensor<Nd + 3, COMPLEX> r;
      if (sto.s3t)
	r = ns_getColorvecs::getColorvecs<COMPLEX>(sto.s3t, u, decay_dir, from_tslice, n_tslices,
						   n_colorvecs, order);
      else if (sto.mod)
	r = ns_getColorvecs::getColorvecs<COMPLEX>(*sto.mod, decay_dir, from_tslice, n_tslices,
						   n_colorvecs, order);

      // Phase colorvecs
      r = phaseColorvecs(r, from_tslice, phase);

      sw.stop();
      QDPIO::cout << "Time to read " << n_colorvecs << " colorvecs from " << n_tslices
		  << " time slices: " << sw.getTimeInSeconds() << " secs" << std::endl;

      return r;
    }

    /// Compute and store colorvecs
    /// \param colorvec_file: file to store the colorvecs
    /// \param link_smear: smearing gauge field options before building the laplacian
    /// \param u: gauge field
    /// \param from_tslice: first tslice to read
    /// \param n_tslices: number of tslices to read
    /// \param n_colorvecs: number of eigenpairs to read
    /// \param use_s3t_storage: if true S3T is used, otherwise FILEDB
    /// \param fingerprint: whether to store only a few sites of each colorvecs
    /// \param phase: apply a phase to the eigenvectors
    /// \param colorvec_file_src: if given, read the colorvecs from that file and if they
    ///        match the computed ones, they are the ones stored; this guarantee that the
    ///        that given smearing options were used to generate the colorvecs in `colorvec_file_src`

    inline void
    createColorvecStorage(const std::string& colorvec_file, GroupXML_t link_smear,
			  const multi1d<LatticeColorMatrix>& u, int from_tslice, int n_tslices,
			  int n_colorvecs, bool use_s3t_storage = false, bool fingerprint = false,
			  Coor<Nd - 1> phase = {},
			  const Maybe<std::vector<std::string>>& colorvec_file_src = none)
    {
      // Check input
      const int Nt = Layout::lattSize()[3];
      if (from_tslice < 0)
	throw std::runtime_error("The first t-slice to compute colorvecs is negative!");
      if (n_tslices < 0 || n_tslices > Nt)
	throw std::runtime_error(" The number of t-slices to compute colorvecs is negative or "
				 "greater than the t dimension of the lattice");

      // Smear the gauge field if needed
      multi1d<LatticeColorMatrix> u_smr = u;
      try
      {
	std::istringstream xml_l(link_smear.xml);
	XMLReader linktop(xml_l);
	Handle<LinkSmearing> linkSmearing(
	  TheLinkSmearingFactory::Instance().createObject(link_smear.id, linktop, link_smear.path));
	(*linkSmearing)(u_smr);
      } catch (const std::string& e)
      {
	QDPIO::cerr << ": Caught Exception link smearing: " << e << std::endl;
	QDP_abort(1);
      } catch (...)
      {
	QDPIO::cerr << ": Caught unexpected exception" << std::endl;
	QDP_abort(1);
      }

      // Some tasks read the eigenvalues from metadata but they not used; so we are going to give fake values
      multi1d<multi1d<double>> evals(n_colorvecs);
      for (int i = 0; i < n_colorvecs; ++i)
      {
	evals[i].resize(n_tslices);
	for (int t = 0; t < n_tslices; ++t)
	  evals[i][t] = 0;
      }

      // Open the DB and write metada
      MOD_t mod;
      StorageTensor<Nd + 2, ComplexD> sto;
      Coor<3> fingerprint_dim{};

      if (!use_s3t_storage)
      {
	XMLBufferWriter file_xml;

	push(file_xml, "MODMetaData");
	write(file_xml, "id", "eigenVecsTimeSlice");
	multi1d<int> spatialLayout(3);
	spatialLayout[0] = Layout::lattSize()[0];
	spatialLayout[1] = Layout::lattSize()[1];
	spatialLayout[2] = Layout::lattSize()[2];
	write(file_xml, "lattSize", spatialLayout);
	write(file_xml, "decay_dir", 3);
	write(file_xml, "num_vecs", n_colorvecs);
	write(file_xml, "Weights", evals);
	file_xml << link_smear.xml;
	pop(file_xml);

	mod.setDebug(0);

	mod.insertUserdata(file_xml.str());
	mod.open(colorvec_file, std::ios_base::in | std::ios_base::out | std::ios_base::trunc);
      }
      else
      {
	std::string sto_order = "cxyztn"; // order for storing the colorvecs

	// If fingerprint, we store only the support of the colorvecs on a subset of the lattice;
	// compute the size of that subset
	for (int i = 0; i < 3; ++i)
	  fingerprint_dim[i] = std::min(4, Layout::lattSize()[i]);

	// Prepare metadata
	XMLBufferWriter file_xml;

	push(file_xml, "MODMetaData");
	write(file_xml, "id", "eigenVecsTimeSlice");
	multi1d<int> spatialLayout(3);
	spatialLayout[0] = Layout::lattSize()[0];
	spatialLayout[1] = Layout::lattSize()[1];
	spatialLayout[2] = Layout::lattSize()[2];
	write(file_xml, "lattSize", spatialLayout);
	write(file_xml, "decay_dir", 3);
	write(file_xml, "num_vecs", n_colorvecs);
	write(file_xml, "Weights", evals);
	write(file_xml, "order", sto_order);
	write(file_xml, "fingerprint", fingerprint);
	if (fingerprint)
	{
	  spatialLayout[0] = fingerprint_dim[0];
	  spatialLayout[1] = fingerprint_dim[1];
	  spatialLayout[2] = fingerprint_dim[2];
	  write(file_xml, "fingerprint_lattice", spatialLayout);
	}
	file_xml << link_smear.xml;
	pop(file_xml);

	// NOTE: file_xml has nonzero value only at the master node; so do a broadcast

	sto = StorageTensor<Nd + 2, ComplexD>(
	  colorvec_file, broadcast(file_xml.str()), sto_order,
	  latticeSize<Nd + 2>(sto_order, {{'n', n_colorvecs}, {'x', Layout::lattSize()[0]}}),
	  Sparse, superbblas::BlockChecksum);
      }

      // Open colorvec_file_src
      ColorvecsStorage colorvecsSto;
      if (colorvec_file_src.getSome({}).size() > 0)
	colorvecsSto = openColorvecStorage(colorvec_file_src.getSome());

      for (int i_tslice = 0; i_tslice < n_tslices; ++i_tslice, from_tslice = (from_tslice + 1) % Nt)
      {
	// Compute colorvecs
	std::string order = "cxyzXtn";
	auto colorvecs_and_evals =
	  ns_getColorvecs::computeColorvecs(u_smr, from_tslice, 1, n_colorvecs, order);
	auto colorvecs = colorvecs_and_evals.first;

	// Read the eigenvectors from another source if indicated
	if (colorvec_file_src.getSome({}).size() > 0)
	{
	  auto colorvecs_src =
	    getColorvecs<ComplexD>(colorvecsSto, u, 3, from_tslice, 1, n_colorvecs);

	  Tensor<2, ComplexD> ip("nt", Coor<2>{n_colorvecs, 1}, OnHost, OnEveryoneReplicated);
	  ip.contract(colorvecs, {}, Conjugate, colorvecs_src, {}, NotConjugate);
	  for (int n = 0; n < n_colorvecs; ++n)
	    if (std::fabs(std::fabs(ip.get({n, 0})) - 1) > 1e-4)
	      throw std::runtime_error(
		"The given colorvec does not correspond to current gates field and smearing");
	  colorvecs = colorvecs_src;
	}

	// Phase colorvecs
	colorvecs = phaseColorvecs(colorvecs, from_tslice, phase);

	// Compute the permutation from natural ordering to red-black
	std::vector<Index> perm = ns_getColorvecs::getPermFromNatToRB(from_tslice);

	// Store the colorvecs in natural order (not in red-black ordering)
	if (!use_s3t_storage)
	{
	  // Allocate a single time slice colorvec in natural ordering, as colorvec are stored
	  Tensor<Nd, ComplexF> tnat("cxyz", latticeSize<Nd>("cxyz", {{'x', Layout::lattSize()[0]}}),
				    OnHost, OnMaster);

	  // Allocate a single time slice colorvec in case of using RB ordering
	  Tensor<Nd + 1, ComplexF> trb("cxyzX", latticeSize<Nd + 1>("cxyzX"), OnHost, OnMaster);

	  for (int n = 0; n < n_colorvecs; ++n)
	  {
	    KeyTimeSliceColorVec_t time_key;
	    time_key.t_slice = from_tslice;
	    time_key.colorvec = n;
	    colorvecs.kvslice_from_size({{'t', 0}, {'n', n}}, {{'t', 1}, {'n', 1}}).copyTo(trb);
	    ns_getColorvecs::toNat(perm, trb, tnat);
	    mod.insert(time_key, tnat);
	  }
	}
	else
	{
	  // Allocate a single time slice colorvec in natural ordering, as colorvec are stored
	  Tensor<Nd, ComplexD> tnat("cxyz", latticeSize<Nd>("cxyz", {{'x', Layout::lattSize()[0]}}),
				    OnHost, OnMaster);

	  // Allocate a single time slice colorvec in case of using RB ordering
	  Tensor<Nd + 1, ComplexD> trb("cxyzX", latticeSize<Nd + 1>("cxyzX"), OnHost, OnMaster);

	  std::map<char, int> colorvec_size{};
	  if (fingerprint)
	    colorvec_size = std::map<char, int>{
	      {'x', fingerprint_dim[0]}, {'y', fingerprint_dim[1]}, {'z', fingerprint_dim[2]}};

	  for (int n = 0; n < n_colorvecs; ++n)
	  {
	    colorvecs.kvslice_from_size({{'t', 0}, {'n', n}}, {{'t', 1}, {'n', 1}}).copyTo(trb);
	    ns_getColorvecs::toNat(perm, trb, tnat);
	    sto.kvslice_from_size({{'t', from_tslice}, {'n', n}}, {{'t', 1}, {'n', 1}})
	      .copyFrom(tnat.kvslice_from_size({}, colorvec_size));
	  }
	}
      }

      if (!use_s3t_storage)
	mod.close();
    }

    //
    // High-level chroma operations
    //

    /// Apply the inverse to LatticeColorVec tensors for a list of spins
    /// \param PP: invertor
    /// \param chi: lattice color tensor on a t_slice, cxyzXn
    /// \param t_source: time-slice in chi
    /// \param Nt_forward: return the next Nt_forward time-slices after t_source
    /// \param Nt_backward: return the previous Nt_backward time-slices before t_source
    /// \param spin_sources: list of spins
    /// \param max_rhs: maximum number of vectors solved at once
    /// \param order_out: coordinate order of the output tensor, a permutation of cSxyztXns where
    ///        s is the spin source and S is the spin sink
    /// \return: tensor cSxyztXns where the first t_slice is the t_source-Nt_backward time-slice of
    ///        the vectors after the inversion, and goes increasingly until time-source t_source+Nt_forward

    template <typename COMPLEX_CHI, typename COMPLEX_OUT>
    Tensor<Nd + 5, COMPLEX_OUT> doInversion(const SystemSolver<LatticeFermion>& PP,
					    const Tensor<Nd + 3, COMPLEX_CHI> chi, int t_source,
					    int first_tslice_out, int n_tslice_out,
					    const std::vector<int>& spin_sources, int max_rhs,
					    const std::string& order_out = "cSxyztXns")
    {
      detail::check_order_contains(order_out, "cSxyztXns");
      if (chi.kvdim()['t'] != 1)
	throw std::runtime_error("Expected one time-slice");
      const int num_vecs = chi.kvdim()['n'];

      if (n_tslice_out > Layout::lattSize()[3])
	throw std::runtime_error("Too many tslices");

      Tensor<Nd + 5, COMPLEX_OUT> psi(
	order_out,
	latticeSize<Nd + 5>(
	  order_out,
	  {{'t', n_tslice_out}, {'S', Ns}, {'s', spin_sources.size()}, {'n', num_vecs}}));

      int max_step = std::max(num_vecs, max_rhs);
      std::vector<std::shared_ptr<LatticeFermion>> chis(max_step), quark_solns(max_step);
      for (int col = 0; col < max_step; col++)
	chis[col].reset(new LatticeFermion);
      for (int col = 0; col < max_step; col++)
	quark_solns[col].reset(new LatticeFermion);

      StopWatch snarss1;
      snarss1.reset();
      snarss1.start();

      for (int spin_source : spin_sources)
      {
	for (int n0 = 0, n_step = std::min(max_rhs, num_vecs); n0 < num_vecs;
	     n0 += n_step, n_step = std::min(n_step, num_vecs - n0))
	{
	  for (int n = n0, col = 0; col < n_step; ++n, ++col)
	  {
	    // Put the colorvec sources for the t_source on chis for spin `spin_source`
	    // chis[col][s=spin_source] = chi[n=n0]
	    *chis[col] = zero;
	    chi.kvslice_from_size({{'n', n}}, {{'n', 1}})
	      .copyTo(SB::asTensorView(*chis[col])
			.kvslice_from_size({{'t', t_source}, {'s', spin_source}}));

	    *quark_solns[col] = zero;
	  }

	  // Solve
	  std::vector<SystemSolverResults_t> res =
	    PP(std::vector<std::shared_ptr<LatticeFermion>>(quark_solns.begin(),
							    quark_solns.begin() + n_step),
	       std::vector<std::shared_ptr<const LatticeFermion>>(chis.begin(),
								  chis.begin() + n_step));

	  for (int n = n0, col = 0; col < n_step; ++n, ++col)
	  {
	    // psi[n=n] = quark_solns[col][t=first_tslice+(0:n_tslice_out-1)]
	    asTensorView(*quark_solns[col])
	      .kvslice_from_size({{'t', first_tslice_out}}, {{'t', n_tslice_out}})
	      .rename_dims({{'s', 'S'}})
	      .copyTo(psi.kvslice_from_size({{'n', n}, {'s', spin_source}}));
	  }
	}
      }

      snarss1.stop();
      QDPIO::cout << "Time to compute inversions for " << spin_sources.size()
		  << " spin sources and " << num_vecs
		  << " colorvecs : " << snarss1.getTimeInSeconds() << " secs" << std::endl;

      return psi;
    }

    namespace detail
    {
      /// Path Node
      struct PathNode {
	std::map<int, PathNode> p; ///< following nodes
	int disp_index;		   ///< if >= 0, the index in the displacement list
      };

      /// Return the directions that are going to be use and the maximum number of displacements keep in memory
      inline void get_tree_mem_stats(const PathNode& disps, std::array<bool, Nd>& dirs,
				     unsigned int& max_rhs)
      {
	unsigned int max_rhs_sub = 0;
	for (const auto it : disps.p)
	{
	  unsigned int max_rhs_sub_it = 0;
	  get_tree_mem_stats(it.second, dirs, max_rhs_sub_it);
	  max_rhs_sub = std::max(max_rhs_sub, max_rhs_sub_it);

	  if (std::abs(it.first) <= Nd)
	    dirs[std::abs(it.first) - 1] = true;
	}

	if (disps.p.size() == 0)
	{
	  max_rhs = 0;
	}
	else if (disps.p.size() == 1)
	{
	  max_rhs = std::max(1u, max_rhs_sub);
	}
	else
	{
	  max_rhs = 1 + max_rhs_sub;
	}
      }

      const int path_separator = Nd + 1;

      /// Return the tree representing all paths
      /// \param paths: list of displacements
      /// \param allow_separator: allow a special direction Nd+1

      inline PathNode get_tree(const std::vector<std::vector<int>>& paths,
			       bool allow_separator = false)
      {
	PathNode r{{}, -1};
	int path_index = 0;
	for (const std::vector<int>& path : paths)
	{
	  PathNode* n = &r;
	  for (int d : path)
	  {
	    if (d == 0 || (std::abs(d) > Nd && (!allow_separator || d != path_separator)))
	      throw std::runtime_error("Invalid direction: " + std::to_string(d));

	    auto it = n->p.find(d);
	    if (it != n->p.end())
	      n = &it->second;
	    else
	    {
	      n->p[d] = PathNode{{}, -1};
	      n = &n->p[d];
	    }
	  }
	  if (n->disp_index < 0)
	    n->disp_index = path_index++;
	}
	return r;
      }

    }

    namespace ns_doMomGammaDisp_contractions
    {
      using namespace detail;

      /// Contract two LatticeFermion with different momenta, gammas, and displacements.
      /// \param leftconj: left lattice fermion tensor, cSxyzXN
      /// \param right: right lattice fermion tensor, csxyzXn
      /// \param disps: tree of displacements/derivatives
      /// \param deriv: if true, do left-right nabla derivatives
      /// \param gammas: tensor with spins, QSg
      /// \param moms: list of momenta
      /// \param max_rhs: maximum number of vectors hold in memory
      /// \param r tensor holding the contractions, sqnNmgd where
      ///        q and N (s and n) are the spin and vector from left (right) vectors, m is the momentum
      ///        index, g is the gamma index, and d is the displacement index
      /// \param: disp_indices: dictionary that map each `d` index in r displacement index.

      template <typename COMPLEX, std::size_t Nleft, std::size_t Nright, std::size_t Nout>
      void doMomGammaDisp_contractions(const std::vector<Tensor<Nd + 3, Complex>>& u,
				       const Tensor<Nleft, COMPLEX> leftconj,
				       Tensor<Nright, COMPLEX> right, Index first_tslice,
				       const PathNode& disps, bool deriv, Tensor<3, COMPLEX> gammas,
				       const std::vector<Coor<Nd - 1>>& moms, int max_rhs,
				       Tensor<Nout, COMPLEX> r, std::vector<int>& disp_indices)
      {
	max_rhs = std::max(1, max_rhs);

	if (disps.disp_index >= 0)
	{
	  detail::log(1, "contracting for disp_index=" + std::to_string(disps.disp_index));
	  // Contract the spatial components and the color of the leftconj and right tensors
	  Tensor<Nout, COMPLEX> aux =
	    r.template like_this<Nout, COMPLEX>("mNQqnSst%", '%', "gd", {{'S', Ns}, {'Q', Ns}});
	  aux.contract(leftconj, {}, Conjugate, right, {}, NotConjugate,
		       {});

	  // Contract the spin components S and Q with the gammas, and put the result on r[d=disp_indices.size()]
	  Tensor<Nout - 1, COMPLEX> aux0 =
	    r.template like_this<Nout - 1, COMPLEX>("gmNqnst%", '%', "d");
	  aux0.contract(gammas, {}, NotConjugate, aux, {}, NotConjugate);
	  aux0.copyTo(r.kvslice_from_size({{'d', disp_indices.size()}}, {{'d', 1}}));

	  // Annotate on disp_indices the displacement being computed for the current `d`
	  disp_indices.push_back(disps.disp_index);
	}

	// Apply displacements on the right and call recursively
	const int num_vecs = right.kvdim()['n'];
	unsigned int node_disp = 0;
	for (const auto it : disps.p)
	{
	  detail::log(1, "push on direction " + std::to_string(it.first));
	  // Apply displacement on the right vectors
	  // NOTE: avoid that the memory requirements grow linearly with the number of displacements
	  //       by killing the reference to `right` as soon as possible
	  Tensor<Nright, COMPLEX> right_disp =
	    !deriv ? displace(u, right, first_tslice, it.first)
		   : leftRightNabla(u, right, first_tslice, it.first, moms);
	  if (node_disp == disps.p.size() - 1)
	    right.release();
	  doMomGammaDisp_contractions(u, leftconj, std::move(right_disp), first_tslice, it.second,
				      deriv, gammas, moms, max_rhs - num_vecs, r, disp_indices);
	  node_disp++;
	  detail::log(1, "pop direction");
	}
      }
    }

    template <typename COMPLEX = Complex>
    using Moms = std::pair<Tensor<Nd + 2, COMPLEX>, std::vector<Coor<3>>>;

    /// Copy several momenta into a single tensor
    /// \param decay_dir: something that should be three
    /// \param moms: momenta to apply
    /// \param first_mom: first momentum to extract
    /// \param num_moms: number of momenta to extract
    /// \param first_tslice: first time-slice to extract
    /// \param num_tslice: number of time-slices to extract
    /// \param order_out: coordinate order of the output tensor, a permutation of mxyzXt
    /// \return: the tensor with the momenta

    template <typename COMPLEX = Complex>
    Moms<COMPLEX> getMoms(int decay_dir, const SftMom& moms, Maybe<int> first_mom = none,
			  Maybe<int> num_moms = none, Maybe<Index> first_tslice = none,
			  Maybe<int> num_tslices = none, const std::string& order_out = "mxyzXt")
    {
      // Copy moms into a single tensor
      const int Nt = Layout::lattSize()[decay_dir];
      int tfrom = first_tslice.getSome(0);	   // first tslice to extract
      int tsize = num_tslices.getSome(Nt);	   // number of tslices to extract
      int mfrom = first_mom.getSome(0);		   // first momentum to extract
      int msize = num_moms.getSome(moms.numMom()); // number of momenta to extract

      Tensor<Nd + 2, COMPLEX> momst(order_out,
				    latticeSize<Nd + 2>(order_out, {{'t', tsize}, {'m', msize}}));
      for (unsigned int mom = 0; mom < msize; ++mom)
      {
	asTensorView(moms[mfrom + mom])
	  .kvslice_from_size({{'t', tfrom}}, {{'t', tsize}})
	  .copyTo(momst.kvslice_from_size({{'m', mom}}, {{'m', 1}}));
      }

      // Create mom_list
      std::vector<Coor<Nd - 1>> mom_list(msize);
      for (unsigned int mom = 0; mom < msize; ++mom)
      {
	for (unsigned int i = 0; i < 3; ++i)
	  mom_list[mom][i] = moms.numToMom(mfrom + mom)[i];
      }

      return {momst, mom_list};
    }

    /// Contract two LatticeFermion with different momenta, gammas, and displacements.
    /// \param leftconj: left lattice fermion tensor, cxyzXNQqt
    /// \param right: right lattice fermion tensor, cxyzXnSst
    /// \param first_tslice: first time-slice in leftconj and right
    /// \param moms: momenta to apply
    /// \param moms_first: index of the first momenta to apply
    /// \param num_moms: number of momenta to apply (if none, apply all of them)
    /// \param gammas: list of gamma matrices to apply
    /// \param disps: list of displacements/derivatives
    /// \param deriv: if true, do left-right nabla derivatives
    /// \param max_rhs: maximum number of vectors hold in memory
    /// \param order_out: coordinate order of the output tensor, a permutation of nNSQmgd where
    ///        q and N (s and n) are the spin and vector from left (right) vectors, m is the momentum
    ///        index, g is the gamma index, and d is the displacement index
    /// \return: a pair made of a tensor sqnNmgd and a vector that is a dictionary that map each `d`
    ///        index in the tensor with an input displacement index.

    template <std::size_t Nout, std::size_t Nleft, std::size_t Nright, typename COMPLEX>
    std::pair<Tensor<Nout, COMPLEX>, std::vector<int>> doMomGammaDisp_contractions(
      const multi1d<LatticeColorMatrix>& u, Tensor<Nleft, COMPLEX> leftconj,
      Tensor<Nright, COMPLEX> right, Index first_tslice, const SftMom& moms, int first_mom,
      Maybe<int> num_moms, const std::vector<Tensor<2, COMPLEX>>& gammas,
      const std::vector<std::vector<int>>& disps, bool deriv,
      const std::string& order_out = "gmNndsqt", Maybe<int> max_active_tslices = none)
    {
      detail::check_order_contains(order_out, "gmNndsqt");
      detail::check_order_contains(leftconj.order, "cxyzXNQqt");
      detail::check_order_contains(right.order, "cxyzXnSst");

      if (right.kvdim()['t'] != leftconj.kvdim()['t'])
	throw std::runtime_error("The t component of `right' and `left' does not match");
      int Nt = right.kvdim()['t'];

      int max_t = max_active_tslices.getSome(Nt);
      if (max_t <= 0)
	max_t = Nt;

      // Form a tree with the displacement paths
      detail::PathNode tree_disps = ns_doMomGammaDisp_contractions::get_tree(disps);

      // Get what directions are going to be used and the maximum number of displacements in memory
      std::array<bool, Nd> active_dirs{};
      unsigned int max_active_disps = 0;
      detail::get_tree_mem_stats(tree_disps, active_dirs, max_active_disps);

      // Number of moments to apply
      int numMom = num_moms.getSome(moms.numMom());
      if (first_mom + numMom > moms.numMom())
	throw std::runtime_error("Invalid range of momenta");

      // Allocate output tensor
      std::map<char, int> r_size = {{'t', Nt},
				    {'n', right.kvdim()['n']},
				    {'s', right.kvdim()['s']},
				    {'N', leftconj.kvdim()['N']},
				    {'q', leftconj.kvdim()['q']},
				    {'m', numMom},
				    {'g', gammas.size()},
				    {'d', disps.size()}};
      for (char c : detail::remove_dimensions(order_out, "gmNndsqt"))
	r_size[c] = leftconj.kvdim()[c];
      Tensor<Nout, COMPLEX> r(order_out, kvcoors<Nout>(order_out, r_size));

      // Create mom_list
      std::vector<Coor<Nd - 1>> mom_list(numMom);
      for (unsigned int mom = 0; mom < numMom; ++mom)
      {
	for (unsigned int i = 0; i < Nd - 1; ++i)
	  mom_list[mom][i] = moms.numToMom(first_mom + mom)[i];
      }

      // Copy all gammas into a single tensor
      Tensor<3, COMPLEX> gammast("gQS", {(Index)gammas.size(), Ns, Ns}, OnDefaultDevice,
				 OnEveryoneReplicated);
      for (unsigned int g = 0; g < gammas.size(); g++)
      {
	gammas[g]
	  .rename_dims({{'i', 'Q'}, {'j', 'S'}})
	  .copyTo(gammast.kvslice_from_size({{'g', g}}, {{'g', 1}}));
      }

      // Iterate over time-slices
      std::vector<int> disp_indices;

      for (int tfrom = 0, tsize = std::min(max_t, Nt); tfrom < Nt;
	   tfrom += tsize, tsize = std::min(max_t, Nt - tfrom))
      {
	// Make tsize one or even
	if (tsize > 1 && tsize % 2 != 0)
	  --tsize;

	detail::log(1, "contracting " + std::to_string(tsize) +
			 " tslices from tslice= " + std::to_string(tfrom));

	disp_indices.resize(0);

	// Copy moms into a single tensor
	std::string momst_order = "mxyzXt";
	Tensor<Nd + 2, COMPLEX> momst(
	  momst_order, latticeSize<Nd + 2>(momst_order, {{'t', tsize}, {'m', numMom}}));
	for (unsigned int mom = 0; mom < numMom; ++mom)
	{
	  asTensorView(moms[first_mom + mom])
	    .kvslice_from_size({{'t', tfrom + first_tslice}}, {{'t', tsize}})
	    .copyTo(momst.kvslice_from_size({{'m', mom}}, {{'m', 1}}));
	}

	// Apply momenta conjugated to the left tensor and rename the spin components s and Q to q and Q,
	// and the colorvector component n to N
	Tensor<Nleft + 1, COMPLEX> moms_left = leftconj.template like_this<Nleft + 1>(
	  "mQNqc%xyzXt", '%', "", {{'m', numMom}, {'t', tsize}});
	moms_left.contract(std::move(momst), {}, Conjugate,
			   leftconj.kvslice_from_size({{'t', tfrom}}, {{'t', tsize}}), {},
			   NotConjugate);
	if (tfrom + tsize >= Nt)
	  leftconj.release();

	// Make a copy of the time-slicing of u[d] also supporting left and right
	std::vector<Tensor<Nd + 3, Complex>> ut(Nd);
	for (unsigned int d = 0; d < Nd - 1; d++)
	{
	  if (!active_dirs[d])
	    continue;

	  // NOTE: This is going to create a tensor with the same distribution of the t-dimension as leftconj and right
	  ut[d] = asTensorView(u[d])
		    .kvslice_from_size({{'t', first_tslice + tfrom}}, {{'t', tsize}})
		    .toComplex();
	}

	// Do the thing
	auto this_right = right.kvslice_from_size({{'t', tfrom}}, {{'t', tsize}});
	if (tfrom + tsize >= Nt)
	  right.release();
	auto this_r = r.kvslice_from_size({{'t', tfrom}}, {{'t', tsize}});
	if (!deriv)
	{
	  ns_doMomGammaDisp_contractions::doMomGammaDisp_contractions(
	    ut, std::move(moms_left), std::move(this_right), first_tslice + tfrom, tree_disps,
	    deriv, gammast, mom_list, 0, this_r, disp_indices);
	}
	else
	{
	  throw std::runtime_error("Derivatives are not implemented! Sorry!");
	  // std::vector<COMPLEX> ones(moms.numMom(), COMPLEX(1));
	  // std::string right_moms_order = std::string(right.order.begin(), right.order.size()) + "m";
	  // Tensor<Nright + 1, COMPLEX> right_moms =
	  //   right.like_this<Nright + 1>(right_moms_order.c_str());
	  // right_moms.contract(asTensorView(ones), {{'i', 'm'}}, NotConjugate, std::move(right), {},
	  // 		    NotConjugate);
	  // doMomGammaDisp_contractions(u, gammast_moms_left, right_moms, tree_disps, deriv, mom_list,
	  // 			    max_rhs, r, disp_indices);
	}
      }

      return {r, disp_indices};
    }

    /// Callback function for each displacement/derivate, and chunk of time-slices and momenta
    /// Arguments of the callback:
    /// \param tensor: output tensor with order ijkmt
    /// \param disp: index of the displacement/derivative
    /// \param first_timeslice: index of the first time-slice in the tensor
    /// \param first_mom: index of the first momentum in the tensor

    template <typename COMPLEX = Complex>
    using ColorContractionFn = std::function<void(Tensor<5, COMPLEX>, int, int, int)>;

    namespace ns_doMomDisp_colorContractions
    {
      using namespace detail;

      /// Return the tree representing all paths
      inline PathNode get_tree(const std::vector<std::array<std::vector<int>, 3>>& paths)
      {
	// Concatenate the three paths in each displacement
	std::vector<std::vector<int>> paths_out;
	for (const std::array<std::vector<int>, 3>& tripletpath : paths)
	{
	  std::vector<int> p;
	  for (int i = 0; i < 3; ++i)
	  {
	    p.insert(p.end(), tripletpath[i].begin(), tripletpath[i].end());
	    if (i < 2)
	      p.push_back(path_separator);
	  }
	  paths_out.push_back(p);
	}
	return detail::get_tree(paths_out, true);
      }

      /// Contract three LatticeColorvec with different momenta and displacements.
      /// Auxiliary function traversing the tree for disps2.
      /// \param colorvecs: lattice color tensor on several t_slices, ctxyzXn
      /// \param disps: tree of displacements/derivatives for colorvecs
      /// \param deriv: if true, do right nabla derivatives
      /// \param moms: momenta tensor on several t_slices, mtxyzX
      /// \param first_mom: index of the first momentum being computed
      /// \param max_cols: maximum number from colorvecs[0] to be contracted at once
      /// \param order_out: coordinate order of the output tensor, a permutation of ijkmt
      /// \param call: function to call for each combination of disps0, disps1,
      ///        and disps2.

      template <typename COMPLEX, std::size_t Nin>
      void doMomDisp_colorContractions(const std::vector<Tensor<Nd + 3, COMPLEX>>& u,
				       std::array<Tensor<Nin, COMPLEX>, 3> colorvecs,
				       Index first_tslice, const PathNode& disps, bool deriv,
				       int current_colorvec, const Moms<COMPLEX> moms,
				       int first_mom, int max_cols, const std::string& order_out,
				       DeviceHost dev, Distribution dist,
				       const ColorContractionFn<COMPLEX>& call)
      {
	if (disps.disp_index >= 0)
	{
	  detail::log(1, "contracting for disp_index=" + std::to_string(disps.disp_index));

	  // Create the output tensor
	  Tensor<5, COMPLEX> colorvec012m =
	    colorvecs[0].template like_this<5, COMPLEX>(order_out,
							{{'i', colorvecs[0].kvdim()['n']},
							 {'j', colorvecs[1].kvdim()['n']},
							 {'k', colorvecs[2].kvdim()['n']},
							 {'m', moms.first.kvdim()['m']}},
							dev, dist);

	  // Contract colorvec2 and moms
	  Tensor<Nd + 4, COMPLEX> colorvec2m =
	    colorvecs[2]
	      .template like_this<Nd + 4, COMPLEX>("ncm%xyzXt", '%', "",
						   {{'m', moms.first.kvdim()['m']}})
	      .rename_dims({{'n', 'k'}});
	  colorvec2m.contract(colorvecs[2].rename_dims({{'n', 'k'}}), {}, NotConjugate, moms.first,
			      {}, NotConjugate);

	  int imax = max_cols;
	  if (imax <= 0)
	    imax = colorvecs[0].kvdim()['n'];

	  for (int i0 = 0, i1 = colorvecs[0].kvdim()['n'], isize = std::min(imax, i1); i0 < i1;
	       i0 += isize, isize = std::min(imax, i1 - i0))
	  {
	    // Color-contract colorvec0 and colorvec1
	    Tensor<Nin + 1, COMPLEX> colorvec01 =
	      colorvecs[0]
		.template like_this<Nin + 1, COMPLEX>(
		  "njcxyzXt%", '%', "", {{'n', isize}, {'j', colorvecs[1].kvdim()['n']}})
		.rename_dims({{'n', 'i'}});
	    auto colorvec0 =
	      colorvecs[0].rename_dims({{'n', 'i'}}).kvslice_from_size({{'i', i0}}, {{'i', isize}});
	    auto colorvec1 = colorvecs[1].rename_dims({{'n', 'j'}});
	    colorvec01.contract(colorvec0.kvslice_from_size({{'c', 2}}), {}, NotConjugate,
				colorvec1.kvslice_from_size({{'c', 1}}), {}, NotConjugate);
	    colorvec01.contract(colorvec0.kvslice_from_size({{'c', 1}}), {}, NotConjugate,
				colorvec1.kvslice_from_size({{'c', 2}}), {}, NotConjugate, {}, -1);
	    colorvec0.release();
	    colorvec1.release();

	    // Contract colorvec01 and colorvec2m
	    colorvec012m.kvslice_from_size({{'i', i0}}, {{'i', isize}})
	      .contract(std::move(colorvec01), {}, NotConjugate, colorvec2m, {}, NotConjugate);
	  }

	  // Do whatever
	  call(std::move(colorvec012m), disps.disp_index, first_tslice, first_mom);
	}

	// Apply displacements on colorvec2 and call recursively
	unsigned int node_disp = 0;
	for (const auto it : disps.p)
	{
	  detail::log(1, "for disps, push on direction " + std::to_string(it.first));
	  // Apply displacement on the current colorvec
	  // NOTE: avoid that the memory requirements grow linearly with the number of displacements
	  //       by killing the reference to `colorvec2` as soon as possible
	  std::array<Tensor<Nin, COMPLEX>, 3> colorvecs_disp = colorvecs;
	  int this_current_colorvec = current_colorvec;
	  if (abs(it.first) <= Nd)
	    colorvecs_disp[current_colorvec] =
	      !deriv ? displace(u, colorvecs[current_colorvec], first_tslice, it.first)
		     : rightNabla(u, colorvecs[current_colorvec], first_tslice, it.first);
	  else if (it.first == path_separator)
	    ++this_current_colorvec;
	  else
	    throw std::runtime_error("Invalid direction");
	  if (node_disp == disps.p.size() - 1)
	    for (auto& i : colorvecs)
	      i.release();
	  doMomDisp_colorContractions(u, std::move(colorvecs_disp), first_tslice, it.second, deriv,
				      this_current_colorvec, moms, first_mom, max_cols, order_out,
				      dev, dist, call);
	  node_disp++;
	  detail::log(1, "for disps, pop direction");
	}
      }
    }

    /// Contract three LatticeColorvec with different momenta and displacements.
    /// \param colorvecs: lattice color tensor on several t_slices, ctxyzXn
    /// \param moms: momenta tensor on several t_slices, mtxyzX
    /// \param disps: list of displacements/derivatives
    /// \param deriv: if true, do right nabla derivatives
    /// \param call: function to call for each combination of disps0, disps1, and disps2
    /// \param order_out: coordinate order of the output tensor, a permutation of ijkmt where
    ///        i, j, and k are the n index in colorvecs; and m is the momentum index

    template <std::size_t Nin, typename COMPLEX>
    void doMomDisp_colorContractions(
      const multi1d<LatticeColorMatrix>& u, Tensor<Nin, COMPLEX> colorvec, Moms<COMPLEX> moms,
      Index first_tslice, const std::vector<std::array<std::vector<int>, 3>>& disps, bool deriv,
      const ColorContractionFn<COMPLEX>& call, Maybe<int> max_active_tslices = none,
      Maybe<int> max_active_momenta = none, Maybe<int> max_cols = none,
      const Maybe<std::string>& order_out = none, Maybe<DeviceHost> dev = none,
      Maybe<Distribution> dist = none)
    {
      const std::string order_out_str = order_out.getSome("ijkmt");
      detail::check_order_contains(order_out_str, "ijkmt");
      detail::check_order_contains(colorvec.order, "cxyzXtn");
      detail::check_order_contains(moms.first.order, "xyzXtm");

      // Form a tree with the displacement paths
      detail::PathNode tree_disps = ns_doMomDisp_colorContractions::get_tree(disps);

      // Get what directions are going to be used and the maximum number of displacements in memory
      std::array<bool, Nd> active_dirs{};
      unsigned int max_active_disps = 0;
      detail::get_tree_mem_stats(tree_disps, active_dirs, max_active_disps);

      // Check that all tensors have the same number of time
      int Nt = colorvec.kvdim()['t'];
      if (Nt != moms.first.kvdim()['t'])
	throw std::runtime_error("The t component of `colorvec' and `moms' does not match");

      int max_t = max_active_tslices.getSome(Nt);
      if (max_t <= 0)
	max_t = Nt;

      int Nmom = moms.first.kvdim()['m'];
      int max_active_moms = max_active_momenta.getSome(Nmom);
      if (max_active_moms <= 0)
	max_active_moms = Nmom;

      // Iterate over time-slices
      for (int tfrom = 0, tsize = std::min(max_t, Nt); tfrom < Nt;
	   tfrom += tsize, tsize = std::min(max_t, Nt - tfrom))
      {
	// Make tsize one or even
	if (tsize > 1 && tsize % 2 != 0)
	  --tsize;

	detail::log(
	  1, "color contracting " + std::to_string(tsize) + " tslices from tslice= " +
	       std::to_string(tfrom));

	// Make a copy of the time-slicing of u[d] also supporting left and right
	std::vector<Tensor<Nd + 3, COMPLEX>> ut(Nd);
	for (unsigned int d = 0; d < Nd - 1; d++)
	{
	  if (!active_dirs[d])
	    continue;

	  // NOTE: This is going to create a tensor with the same distribution of the t-dimension as colorvec and moms
	  ut[d] = asTensorView(u[d])
		    .kvslice_from_size({{'t', first_tslice + tfrom}}, {{'t', tsize}})
		    .toComplex();
	}

	// Get the time-slice for colorvec
	auto this_colorvec = colorvec.kvslice_from_size({{'t', tfrom}}, {{'t', tsize}});

	// Loop over the momenta
	for (int mfrom = 0, msize = std::min(max_active_moms, Nmom); mfrom < Nmom;
	     mfrom += msize, msize = std::min(max_active_moms, Nmom - mfrom))
	{
	  auto this_moms = moms.first.kvslice_from_size({{'t', tfrom}, {'m', mfrom}},
							{{'t', tsize}, {'m', msize}});
	  if (tfrom + tsize >= Nt && mfrom + msize >= Nmom)
	  {
	    colorvec.release();
	    moms.first.release();
	  }
	  std::vector<Coor<3>> moms_list(moms.second.begin() + mfrom,
					 moms.second.begin() + mfrom + msize);
	  if (!deriv)
	  {
	    ns_doMomDisp_colorContractions::doMomDisp_colorContractions<COMPLEX, Nin>(
	      ut, {this_colorvec, this_colorvec, this_colorvec}, first_tslice + tfrom, tree_disps,
	      deriv, 0, {this_moms, moms_list}, mfrom, max_cols.getSome(0), order_out_str,
	      dev.getSome(OnDefaultDevice), dist.getSome(OnEveryoneReplicated), call);
	  }
	  else
	  {
	    // When using derivatives, each momenta has a different effect
	    std::vector<COMPLEX> ones(msize, COMPLEX(1));
	    Tensor<Nin + 1, COMPLEX> this_colorvec_m =
	      this_colorvec.template like_this<Nin + 1>("%m", '%', "", {{'m', msize}});
	    this_colorvec_m.contract(this_colorvec, {}, NotConjugate, asTensorView(ones),
				     {{'i', 'm'}}, NotConjugate);
	    ns_doMomDisp_colorContractions::doMomDisp_colorContractions<COMPLEX, Nin + 1>(
	      ut, {this_colorvec_m, this_colorvec_m, this_colorvec_m}, first_tslice + tfrom,
	      tree_disps, deriv, 0, {this_moms, moms_list}, mfrom, max_cols.getSome(0),
	      order_out_str, dev.getSome(OnDefaultDevice), dist.getSome(OnEveryoneReplicated),
	      call);
	  }
	}
      }
    }

    /// Callback function for each displacement/derivate, and chunk of time-slices and momenta
    /// Arguments of the callback:
    /// \param tensor: output tensor with order ijmt, where i and j are the right and left colorvec indices,
    ///        m is the momentum index, and t is the t-slice
    /// \param disp: index of the displacement/derivative
    /// \param first_timeslice: index of the first time-slice in the tensor
    /// \param first_mom: index of the first momentum in the tensor

    template <typename COMPLEX = Complex>
    using ContractionFn = std::function<void(Tensor<4, COMPLEX>, int, int, int)>;

    namespace ns_doMomDisp_contractions
    {
      using namespace detail;

      /// Contract two LatticeColorvec with different momenta and displacements.
      /// Auxiliary function traversing the tree for disps2.
      /// \param colorvecs: lattice color tensor on several t_slices, ctxyzXn
      /// \param disps: tree of displacements/derivatives for colorvecs
      /// \param moms: momenta tensor on several t_slices, mtxyzX
      /// \param first_mom: index of the first momentum being computed
      /// \param order_out: coordinate order of the output tensor, a permutation of ijkmt
      /// \param call: function to call for each combination of displacement, t-slice, and momentum

      template <typename COMPLEX, std::size_t Nleft, std::size_t Nright>
      void doMomDisp_contractions(const std::vector<Tensor<Nd + 3, COMPLEX>>& u,
				  Tensor<Nleft, COMPLEX> left, Tensor<Nright, COMPLEX> right,
				  Index first_tslice, const PathNode& disps, bool deriv,
				  const std::vector<Coor<Nd - 1>>& moms, int first_mom,
				  const std::string& order_out, DeviceHost dev, Distribution dist,
				  const ContractionFn<COMPLEX>& call)
      {
	if (disps.disp_index >= 0)
	{
	  detail::log(1, "contracting for disp_index=" + std::to_string(disps.disp_index));

	  // Contract left and right
	  auto this_right = right.rename_dims({{'n', 'i'}});
	  auto this_left = left.rename_dims({{'n', 'j'}});
	  Tensor<4, COMPLEX> r = this_left.template like_this<4, COMPLEX>(
	    "jimt", {{'i', this_right.kvdim()['i']}}, dev, dist);
	  r.contract(std::move(this_left), {}, Conjugate, std::move(this_right), {}, NotConjugate);

	  // Do whatever
	  call(std::move(r), disps.disp_index, first_tslice, first_mom);
	}

	// Apply displacements on right and call recursively
	unsigned int node_disp = 0;
	for (const auto it : disps.p)
	{
	  detail::log(1, "for disps, push on direction " + std::to_string(it.first));
	  // Apply displacement on the right colorvec
	  // NOTE: avoid that the memory requirements grow linearly with the number of displacements
	  //       by killing the reference to `right` as soon as possible
	  Tensor<Nright, COMPLEX> right_disp =
	    !deriv ? displace(u, right, first_tslice, it.first)
		   : leftRightNabla(u, right, first_tslice, it.first, moms);
	  if (node_disp == disps.p.size() - 1)
	    right.release();
	  doMomDisp_contractions(u, left, std::move(right_disp), first_tslice, it.second, deriv,
				 moms, first_mom, order_out, dev, dist, call);
	  node_disp++;
	  detail::log(1, "for disps, pop direction");
	}
      }
    }

    /// Contract three LatticeColorvec with different momenta and displacements.
    /// It computes
    ///    \eta_j^\dagger exp(- i left_phase \cdot x) \Gamma \eta_k exp(i right_phase \cdot x)
    /// where \eta_i is the ith colorvec, x is a lattice site, and \Gamma is a combination of
    /// derivatives and momenta.
    /// \param colorvecs: lattice color tensor on several t_slices, ctxyzXn
    /// \param left_phase: phase to the left colorvecs
    /// \param right_phase: phase to the right colorvecs
    /// \param moms: momenta tensor on several t_slices, mtxyzX
    /// \param disps: list of displacements/derivatives
    /// \param deriv: if true, do left-right nabla derivatives
    /// \param call: function to call for each combination of disps0, disps1, and disps2
    /// \param order_out: coordinate order of the output tensor, a permutation of ijmt where
    ///        i and j are the n index in the right and left colorvec respectively; and
    ///        m is the momentum index

    template <std::size_t Nin, typename COMPLEX>
    void doMomDisp_contractions(const multi1d<LatticeColorMatrix>& u, Tensor<Nin, COMPLEX> colorvec,
				Coor<3> left_phase, Coor<3> right_phase, Moms<COMPLEX> moms,
				Index first_tslice, const std::vector<std::vector<int>>& disps,
				bool deriv, const ContractionFn<COMPLEX>& call,
				const Maybe<std::string>& order_out = none,
				Maybe<DeviceHost> dev = none, Maybe<Distribution> dist = none)
    {
      const std::string order_out_str = order_out.getSome("ijmt");
      detail::check_order_contains(order_out_str, "ijmt");
      detail::check_order_contains(colorvec.order, "cxyzXtn");
      detail::check_order_contains(moms.first.order, "xyzXtm");

      // Form a tree with the displacement paths
      detail::PathNode tree_disps = detail::get_tree(disps);

      // Get what directions are going to be used and the maximum number of displacements in memory
      std::array<bool, Nd> active_dirs{};
      unsigned int max_active_disps = 0;
      detail::get_tree_mem_stats(tree_disps, active_dirs, max_active_disps);

      // Check that all tensors have the same number of time
      int Nt = colorvec.kvdim()['t'];
      if (Nt != moms.first.kvdim()['t'])
	throw std::runtime_error("The t component of `colorvec' and `moms' does not match");

      // Iterate over time-slices
      for (int tfrom = 0, tsize = Nt; tfrom < Nt; tfrom += tsize, tsize = Nt - tfrom)
      {
	// Make tsize one or even
	if (tsize > 1 && tsize % 2 != 0)
	  --tsize;

	detail::log(1, "contracting " + std::to_string(tsize) +
			 " tslices from tslice= " + std::to_string(tfrom));

	// Make a copy of the time-slicing of u[d] also supporting left and right
	std::vector<Tensor<Nd + 3, COMPLEX>> ut(Nd);
	for (unsigned int d = 0; d < Nd - 1; d++)
	{
	  if (!active_dirs[d])
	    continue;

	  // NOTE: This is going to create a tensor with the same distribution of the t-dimension as colorvec and moms
	  ut[d] = asTensorView(u[d])
		    .kvslice_from_size({{'t', first_tslice + tfrom}}, {{'t', tsize}})
		    .toComplex();
	}

	// Get the time-slice for colorvec
	auto this_colorvec = colorvec.kvslice_from_size({{'t', tfrom}}, {{'t', tsize}});
	auto this_moms = moms.first.kvslice_from_size({{'t', tfrom}}, {{'t', tsize}});

	// Apply left phase and momenta conjugated to the left tensor
	// NOTE: look for the minus sign on left_phase in the doc of this function
	int Nmom = moms.first.kvdim()['m'];
	Tensor<Nin + 1, COMPLEX> moms_left =
	  colorvec.template like_this<Nin + 1>("mc%xyzXt", '%', "", {{'m', Nmom}});
	moms_left.contract(std::move(this_moms), {}, Conjugate,
			   phaseColorvecs(this_colorvec, first_tslice + tfrom, left_phase),
			   {}, NotConjugate);

	// Apply the right phase
	this_colorvec = phaseColorvecs(this_colorvec, first_tslice + tfrom, right_phase);

	if (tfrom + tsize >= Nt)
	{
	  colorvec.release();
	  moms.first.release();
	}

	if (!deriv)
	{
	  ns_doMomDisp_contractions::doMomDisp_contractions<COMPLEX>(
	    ut, std::move(moms_left), this_colorvec, first_tslice + tfrom, tree_disps, deriv,
	    moms.second, 0, order_out_str, dev.getSome(OnDefaultDevice),
	    dist.getSome(OnEveryoneReplicated), call);
	}
	else
	{
	  // When using derivatives, each momenta has a different effect
	  std::vector<COMPLEX> ones(Nmom, COMPLEX(1));
	  Tensor<Nin + 1, COMPLEX> this_colorvec_m =
	    this_colorvec.template like_this<Nin + 1>("%m", '%', "", {{'m', Nmom}});
	  this_colorvec_m.contract(std::move(this_colorvec), {}, NotConjugate, asTensorView(ones),
				   {{'i', 'm'}}, NotConjugate);
	  ns_doMomDisp_contractions::doMomDisp_contractions<COMPLEX>(
	    ut, std::move(moms_left), this_colorvec_m, first_tslice + tfrom, tree_disps, deriv,
	    moms.second, 0, order_out_str, dev.getSome(OnDefaultDevice),
	    dist.getSome(OnEveryoneReplicated), call);
	}
      }
    }

    /// Return the smallest interval containing the union of two intervals
    /// \param from0: first element of the first interval
    /// \param size0: length of the first interval
    /// \param from1: first element of the second interval
    /// \param size1: length of the second interval
    /// \param dim: dimension length
    /// \param fromr: (output) first element of the union of the two intervals
    /// \param sizer: (output) length of the union of the two intervals

    inline void union_interval(Index from0, Index size0, Index from1, Index size1, Index dim,
			       Index& fromr, Index& sizer)
    {
      // Check inputs
      if (size0 > dim || size1 > dim)
	throw std::runtime_error(
	  "Invalid interval to union! Some of input intervals exceeds the lattice dimension");

      // Normalize from and take as from0 the leftmost interval of the two input intervals
      from0 = normalize_coor(from0, dim);
      from1 = normalize_coor(from1, dim);
      if (from0 > from1)
      {
	std::swap(from0, from1);
	std::swap(size0, size1);
      }

      // If some interval is empty, return the other
      if (size0 == 0)
      {
	fromr = from1;
	sizer = size1;
      }
      else if (size1 == 0)
      {
	fromr = from0;
	sizer = size0;
      }
      else
      {
	// Return the shortest interval resulting from the leftmost point of the
	// first interval and the rightmost point of both intervals, and the
	// leftmost point of the second interval and the rightmost point of both
	// intervals

	Index fromra = from0;
	Index sizera = std::max(from0 + size0, from1 + size1) - from0;
	Index fromrb = from1;
	Index sizerb = std::max(from0 + dim + size0, from1 + size1) - from1;
	fromr = (sizera <= sizerb ? fromra : fromrb);
	sizer = (sizera <= sizerb ? sizera : sizerb);
      }

      // Normalize the output if the resulting interval is the whole dimension
      if (sizer >= dim)
      {
	fromr = 0;
	sizer = dim;
      }
    }
  }
}

namespace QDP
{
  //! Binary input
  template <std::size_t N, typename T>
  void read(BinaryReader& bin, Chroma::SB::Tensor<N, T> t)
  {
    t.binaryRead(bin);
  }

  //! Binary output
  template <std::size_t N, typename T>
  inline void write(BinaryWriter& bin, const Chroma::SB::Tensor<N, T> t)
  {
    t.binaryWrite(bin);
  }
}

#endif // BUILD_SB
#endif // __INCLUDE_SUPERB_CONTRACTIONS__
