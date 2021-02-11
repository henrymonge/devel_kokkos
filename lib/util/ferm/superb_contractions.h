// -*- C++ -*-
/*! \file                                                                    
 * \brief Copy, permuting, contracting tensors with superbblas
 *                                                                             
 * Hadron spectrum calculations utilities
 */

#ifndef __INCLUDE_SUPERB_CONTRACTIONS__
#define __INCLUDE_SUPERB_CONTRACTIONS__

// Activate the MPI support in Superbblas
#include <sstream>
#include <sys/cdefs.h>
#define SUPERBBLAS_USE_MPI

#include "chromabase.h"

#include "actions/ferm/fermacts/fermact_factory_w.h"
#include "actions/ferm/fermacts/fermacts_aggregate_w.h"
#include "qdp.h"
#include "qdp_map_obj_disk_multiple.h"
#include "superbblas.h"
#include "util/ferm/key_timeslice_colorvec.h"
#include "util/ft/sftmom.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>

#ifndef M_PI
#  define M_PI                                                                                     \
    3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068L
#endif


namespace Chroma
{

  namespace SB
  {

    using Index = superbblas::IndexType;
    using Complex = std::complex<REAL>;
    using ComplexF = std::complex<REAL32>;
    template <std::size_t N>
    using Coor = superbblas::Coor<N>;
    template <std::size_t N>
    using Order = superbblas::Order<N>;

    /// Where to store the tensor (see class Tensor)
    enum DeviceHost {
      OnHost,	      ///< on cpu memory
      OnDefaultDevice ///< on GPU memory if possible
    };

    /// How to distribute the tensor (see class Tensor)
    enum Distribution {
      OnMaster,		   ///< Fully supported on node with index zero
      OnEveryone,	   ///< Distributed the lattice dimensions (x, y, z, t) as chroma does
      OnEveryoneReplicated ///< All nodes have a copy of the tensor
    };

    /// Whether complex conjugate the elements before contraction (see Tensor::contract)
    enum Conjugation { NotConjugate, Conjugate };

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

    namespace detail
    {
      namespace repr
      {
	template <typename Ostream, std::size_t N>
	Ostream& operator<<(Ostream& s, Order<N> o)
	{
	  s << "\"";
	  for (char c : o)
	    s << c;
	  s << "\"";
	  return s;
	}
      }

      /// Throw an error if it is not a valid order, that is, if some label is repeated
      template <std::size_t N>
      void check_order(const char* order)
      {
	if (std::strlen(order) != N)
	{
	  std::stringstream ss;
	  ss << "The length of the dimension labels `" << order
	     << "` should match the template argument N `" << N << "`";
	  throw std::runtime_error(ss.str());
	}

	std::set<char> s;
	for (unsigned int i = 0; i < N; ++i)
	{
	  if (!s.insert(order[i]).second)
	  {
	    std::stringstream ss;
	    ss << "Invalid order: some label names are repeated `" << order << "`";
	    throw std::runtime_error(ss.str());
	  }
	}
      }

      /// Throw an error if it is not a valid order, that is, if some label is repeated
      template <std::size_t N>
      void check_order(Order<N> order)
      {
	std::set<char> s;
	for (unsigned int i = 0; i < N; ++i)
	{
	  if (!s.insert(order[i]).second)
	  {
	    using namespace repr;
	    std::stringstream ss;
	    ss << "Invalid order: some label names are repeated `" << order << "`";
	    throw std::runtime_error(ss.str());
	  }
	}
      }
    }

    enum Throw_kvcoors { NoThrow, ThrowOnUnmatchLabel, ThrowOnMissing };

    template <std::size_t N>
    Coor<N> kvcoors(const char* order, std::map<char, int> m, Index missing = 0,
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
    Coor<N> latticeSize(const char* order, std::map<char, int> m = {})
    {
#if QDP_USE_LEXICO_LAYOUT
      // No red-black ordering
      std::map<char, int> m0 = {{'x', Layout::lattSize()[0]},
				{'y', Layout::lattSize()[1]},
				{'z', Layout::lattSize()[2]},
				{'t', Layout::lattSize()[3]},
				{'X', 1},
				{'s', Ns},
				{'c', Nc}};
#elif QDP_USE_CB2_LAYOUT
      // Red-black ordering
      assert(Layout::lattSize()[0] % 2 == 0);
      std::map<char, int> m0 = {{'x', Layout::lattSize()[0] / 2},
				{'y', Layout::lattSize()[1]},
				{'z', Layout::lattSize()[2]},
				{'t', Layout::lattSize()[3]},
				{'X', 2},
				{'s', Ns},
				{'c', Nc}};
#else
      throw std::runtime_error("Unsupported layout");
#endif
      for (const auto& it : m)
	m0[it.first] = it.second;
      return kvcoors<N>(order, m0, 0, NoThrow);
    }

    // Replace a label by another label
    using remap = std::map<char, char>;

    namespace detail
    {
      using namespace superbblas::detail;

      template <std::size_t N>
      Order<N> toOrder(const char* order)
      {
	check_order<N>(order);
	Order<N> o;
	std::copy_n(order, N, o.begin());
	return o;
      }

      template <std::size_t N>
      Order<N + 1> toOrderStr(const Order<N>& order)
      {
	Order<N + 1> o;
	std::copy_n(order.begin(), N, o.begin());
	o[N] = 0;
	return o;
      }

      template <std::size_t N>
      Coor<N> kvcoors(Order<N> order, std::map<char, int> m, Index missing = 0,
		      Throw_kvcoors t = ThrowOnUnmatchLabel)
      {
	Order<N + 1> order_str = toOrderStr(order);
	return SB::kvcoors<N>(&order_str[0], m, missing, t);
      }

      // Throw an error if `order` does not contain a label in `should_contain`
      inline void check_order_contains(const char* order, const char* should_contain)
      {
	bool ok = false;
	if (std::strlen(order) == std::strlen(should_contain))
	{
	  int n = std::strlen(order);
	  unsigned int i;
	  for (i = 0; i < n; ++i)
	  {
	    unsigned int j;
	    for (j = 0; j < n && order[i] != should_contain[j]; ++j)
	      ;
	    if (j >= n)
	      break;
	  }
	  if (i >= n)
	    ok = true;
	}
	if (!ok)
	{
	  std::stringstream ss;
	  ss << "The input order `" << order
	     << "` is missing one of this labels: " << should_contain;
	  throw std::runtime_error(ss.str());
	}
      }

      // Throw an error if `order` does not contain a label in `should_contain`
      template <std::size_t N>
      inline void check_order_contains(Order<N> order, const char* should_contain)
      {
	Order<N + 1> order_str = toOrderStr(order);
	return check_order_contains(&order_str[0], should_contain);
      }

      // Return the equivalent value of the coordinate `v` in the interval [0, dim[ for a periodic
      // dimension with length `dim`.

      int normalize_coor(int v, int dim)
      {
	return (v + dim * (v < 0 ? -v / dim + 1 : 0)) % dim;
      }

      template <std::size_t N>
      Coor<N> normalize_coor(Coor<N> v, Coor<N> dim)
      {
	Coor<N> r;
	for (std::size_t i = 0; i < N; ++i)
	  r[i] = normalize_coor(v[i], dim[i]);
	return r;
      }

      template <std::size_t N>
      Order<N> update_order(Order<N> order, remap m)
      {
	for (std::size_t i = 0; i < N; ++i)
	{
	  auto it = m.find(order[i]);
	  if (it != m.end())
	    order[i] = it->second;
	}
	check_order(order);
	return order;
      }

      /// Stores the subtensor supported on each node (used by class Tensor)
      template <std::size_t N>
      struct TensorPartition
      {
      public:
	using PartitionStored = std::vector<superbblas::PartitionItem<N>>;
	Coor<N> dim;	   ///< Dimensions of the tensor
	PartitionStored p; ///< p[i] = {first coordinate, size} of tensor on i-th node

	/// Constructor
	/// \param order: dimension labels (use x, y, z, t for lattice dimensions)
	/// \param dim: dimension size for the tensor
	/// \param dist: how to distribute the tensor among the nodes

	TensorPartition(Order<N> order, Coor<N> dim, Distribution dist) : dim(dim)
	{
	  switch (dist)
	  {
	  case OnMaster: p = all_tensor_on_master(dim); break;
	  case OnEveryone: p = partitioning_chroma_compatible(order, dim); break;
	  case OnEveryoneReplicated: p = all_tensor_replicated(dim); break;
	  }
	}

	/// Return the volume of the tensor supported on this node
	std::size_t localVolume() const
	{
	  return superbblas::detail::volume(p[Layout::nodeNumber()][1]);
	}

      private:
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

	static PartitionStored partitioning_chroma_compatible(Order<N> order, Coor<N> dim)
	{
	  // Find a dimension label in `order` that is going to be distributed
	  const char dist_labels[] = "xyzt"; // distributed dimensions
	  int first_dist_label = -1;
	  for (unsigned int i = 0; i < std::strlen(dist_labels); ++i)
	  {
	    const auto &it = std::find(order.begin(), order.end(), dist_labels[i]);
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
	  Coor<N> procs = detail::kvcoors(
	    order, {{'x', procs_[0]}, {'y', procs_[1]}, {'z', procs_[2]}, {'t', procs_[3]}}, 1,
	    NoThrow);

	  // For each proc, get its coordinate in procs (logical coordinate) and compute the
	  // fair range of the tensor supported on the proc
	  int num_procs = Layout::numNodes();
	  PartitionStored fs(num_procs);
	  for (int rank = 0; rank < num_procs; ++rank)
	  {
	    multi1d<int> cproc_ = Layout::getLogicalCoordFrom(rank);
	    Coor<N> cproc = detail::kvcoors(
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

      void log(int level, std::string s)
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
      }

    }

    template <std::size_t N, typename T>
    struct Tensor
    {
      static_assert(superbblas::supported_type<T>::value, "Not supported type");

    public:
      Order<N> order;			///< Labels of the tensor dimensions
      Coor<N> dim;		        ///< Length of the tensor dimensions
      std::shared_ptr<superbblas::Context> ctx; ///< Tensor storage information (device/host)
      std::shared_ptr<T> data;		///< Pointer to the tensor storage
      std::shared_ptr<detail::TensorPartition<N>>
	p;				///< Distribution of the tensor among the processes
      Distribution dist;		///< Whether the tensor is stored on the cpu or a device
      Coor<N> from;			///< First active coordinate in the tensor
      Coor<N> size;			///< Number of active coordinates on each dimension
      Coor<N> strides;			///< Displacement for the next element along every direction
      T scalar;				///< Scalar factor of the tensor

      // Return a string describing the tensor
      std::string repr(T* ptr) const
      {
	using namespace detail::repr;
	std::stringstream ss;
	ss << "Tensor{data:" << ptr << ", order:" << order << ", dim:" << dim
	   << ", dist:" << dist << "}";
	return ss.str();
      }

      // Construct used by non-Chroma tensors
      Tensor(const char* order_, Coor<N> dim, DeviceHost dev = OnDefaultDevice,
	     Distribution dist = OnEveryone)
	: order(detail::toOrder<N>(order_)),
	  dim(dim),
	  ctx(getContext(dev)),
	  dist(dist),
	  from{},
	  size(dim),
	  strides(detail::get_strides<N>(dim, superbblas::FastToSlow)),
	  scalar{1}
      {
	superbblas::Context ctx0 = *ctx;
	p = std::make_shared<detail::TensorPartition<N>>(
	  detail::TensorPartition<N>(order, dim, dist));
	T* ptr = superbblas::allocate<T>(p->localVolume(), *ctx);
	std::string s = repr(ptr);
	detail::log(1, std::string("allocated ") + s);
	data = std::shared_ptr<T>(ptr, [=](const T* ptr) {
	  superbblas::deallocate(ptr, ctx0);
	  detail::log(1, std::string("deallocated ") + s);
	});
      }

      // Construct used by Chroma tensors (see `asTensorView`)
      Tensor(const char* order_, Coor<N> dim, DeviceHost dev, Distribution dist,
	     std::shared_ptr<T> data)
	: order(detail::toOrder<N>(order_)),
	  dim(dim),
	  ctx(getContext(dev)),
	  data(data),
	  dist(dist),
	  from{},
	  size(dim),
	  strides(detail::get_strides<N>(dim, superbblas::FastToSlow)),
	  scalar{1}
      {

	// For now, TensorPartition creates the same distribution as chroma for tensor with
	// dimensions divisible by chroma logical dimensions
	p = std::make_shared<detail::TensorPartition<N>>(
	  detail::TensorPartition<N>(order, dim, dist));
      }

    protected:
      Tensor() {}

      // Construct a slice of a tensor
      Tensor(const Tensor& t, Order<N> order, Coor<N> from, Coor<N> size)
	: order(order),
	  dim(t.dim),
	  ctx(t.ctx),
	  data(t.data),
	  p(t.p),
	  dist(t.dist),
	  from(detail::normalize_coor(from, t.dim)),
	  size(size),
	  strides(t.strides),
	  scalar{t.scalar}
      {
      }

      // Construct for insert_dimension
      template <std::size_t Nn>
      Tensor(const Tensor<Nn, T>& t, Order<N> order, Coor<N> dim, std::shared_ptr<T> data,
	     Coor<N> from, Coor<N> size)
	: order(order),
	  dim(dim),
	  ctx(t.ctx),
	  data(data),
	  dist(t.dist),
	  from(detail::normalize_coor(from, dim)),
	  size(size),
	  strides(detail::get_strides<N>(dim, superbblas::FastToSlow)),
	  scalar(t.scalar)
      {
	p = std::make_shared<detail::TensorPartition<N>>(
	  detail::TensorPartition<N>(order, dim, dist));
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
      }

    public:
      // Return whether from != 0 or size != dim
      bool isSubtensor() const
      {
	return from != Coor<N>{} || size != dim;
      }

      // Return the dimensions of the tensor
      std::map<char, int> kvdim() const
      {
	std::map<char, int> d;
	for (unsigned int i = 0; i < N; ++i)
	  d[order[i]] = size[i];
	return d;
      }

      // Get an element of the tensor
      T get(Coor<N> coor) const
      {
	if (ctx->plat != superbblas::CPU)
	  throw std::runtime_error(
	    "Unsupported to `get` elements from tensors not stored on the host");
	if (dist != OnMaster)
	  throw std::runtime_error(
	    "Unsupported to `get` elements on tensor not being fully on the master node");

	// coor[i] = coor[i] + from[i]
	for (unsigned int i = 0; i < N; ++i)
	  coor[i] =
	    detail::normalize_coor(detail::normalize_coor(coor[i], size[i]) + from[i], dim[i]);

	return data.get()[detail::coor2index<N>(coor, dim, strides)] * scalar;
      }

      /// Rename dimensions
      Tensor<N, T> rename_dims(SB::remap m) const
      {
	return Tensor<N, T>(*this, detail::update_order(order, m), this->from, this->size);
      }

      // Return a slice of the tensor starting at coordinate `kvfrom` and taking `kvsize` elements in each direction.
      // The missing dimension in `kvfrom` are set to zero and the missing direction in `kvsize` are set to the active size of the tensor.
      Tensor<N, T> kvslice_from_size(std::map<char, int> kvfrom = {},
				     std::map<char, int> kvsize = {}) const
      {
	std::map<char, int> updated_kvsize = this->kvdim();
	for (const auto& it : kvsize)
	  updated_kvsize[it.first] = it.second;
	return slice_from_size(detail::kvcoors(order, kvfrom), detail::kvcoors(order, updated_kvsize));
      }

      // Return a slice of the tensor starting at coordinate `from` and taking `size` elements in each direction.
      Tensor<N, T> slice_from_size(Coor<N> from, Coor<N> size) const
      {
	for (unsigned int i = 0; i < N; ++i)
	{
	  if (size[i] > this->size[i])
	    throw std::runtime_error(
	      "The size of the slice cannot be larger than the original tensor");
	  if (from[i] + size[i] >= this->size[i] && this->size[i] != this->dim[i])
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
      Tensor<Nn, Tn> like_this(Maybe<const char*> new_order = none, std::map<char, int> kvsize = {},
			       Maybe<DeviceHost> new_dev = none,
			       Maybe<Distribution> new_dist = none) const
      {
	std::map<char, int> new_kvdim = kvdim();
	for (const auto& it : kvsize)
	  new_kvdim[it.first] = it.second;
	Order<N + 1> order_ = detail::toOrderStr(order);
	const char * new_order_ = new_order.getSome(&order_[0]);
	return Tensor<Nn, Tn>(new_order_, kvcoors<Nn>(new_order_, new_kvdim, 0, ThrowOnMissing),
			      new_dev.getSome(getDev()), new_dist.getSome(dist));
      }

      template <typename Tn = T>
      Tensor<N, Tn> clone() const
      {
	return cloneOn<Tn>(getDev());
      }

      template <typename Tn = T>
      Tensor<N, Tn> cloneOn(DeviceHost new_dev) const
      {
	Tensor<N, Tn> r = like_this<N, Tn>(none, {}, new_dev);
	copyTo(r);
	return r;
      }

      Tensor<N, T> reorder(const char *new_order) const
      {
	if (order == detail::toOrder<N>(new_order))
	  return *this;
	Tensor<N, T> r = like_this(new_order);
	copyTo(r);
	return r;
      }

      // Copy `this` tensor into the given one
      template <std::size_t Nw, typename Tw>
      void copyTo(Tensor<Nw, Tw> w) const
      {
	Coor<N> wsize = detail::kvcoors(order, w.kvdim(), 1, NoThrow);
	for (unsigned int i = 0; i < N; ++i)
	  if (size[i] > wsize[i])
	    throw std::runtime_error("The destination tensor is smaller than the source tensor");

	T* ptr = this->data.get();
	Tw* w_ptr = w.data.get();
	Order<N + 1> order_ = detail::toOrderStr(order);
	Order<Nw + 1> orderw_ = detail::toOrderStr(w.order);
	superbblas::copy<N, Nw>(scalar / (T)w.scalar, p->p.data(), 1, &order_[0], from, size,
				(const T**)&ptr, &*ctx, w.p->p.data(), 1, &orderw_[0], w.from,
				&w_ptr, &*w.ctx, MPI_COMM_WORLD, superbblas::FastToSlow,
				superbblas::Copy);
      }

      // Add `this` tensor into the given one
      template <std::size_t Nw, typename Tw>
      void addTo(Tensor<Nw, Tw> w) const
      {
	Coor<N> wsize = detail::kvcoors(order, w.kvdim(), 1, NoThrow);
	for (unsigned int i = 0; i < N; ++i)
	  if (size[i] > wsize[i])
	    throw std::runtime_error("The destination tensor is smaller than the source tensor");

	if (w.scalar != T{1})
	  throw std::runtime_error("Not allowed to addTo to tensor with a scalar not being one");

	T* ptr = this->data.get();
	Tw* w_ptr = w.data.get();
	Order<N + 1> order_ = detail::toOrderStr(order);
	Order<Nw + 1> orderw_ = detail::toOrderStr(w.order);
	superbblas::copy<N, Nw>(scalar, p->p.data(), 1, &order_[0], from, size, (const T**)&ptr,
				&*ctx, w.p->p.data(), 1, &orderw_[0], w.from, &w_ptr, &*w.ctx,
				MPI_COMM_WORLD, superbblas::FastToSlow, superbblas::Add);
      }

      // Contract the dimensions with the same label in `v` and `w` than do not appear on `this` tensor.
      template <std::size_t Nv, std::size_t Nw>
      void contract(Tensor<Nv, T> v, remap mv, Conjugation conjv, Tensor<Nw, T> w, remap mw,
		    Conjugation conjw, remap mr = {}, T beta = T{0})
      {
	// If either v or w is on OnDevice, force both to be on device
	if (v.ctx->plat != w.ctx->plat)
	{
	  if (v.getDev() != OnDefaultDevice)
	    v = v.cloneOn(OnDefaultDevice);
	  if (w.getDev() != OnDefaultDevice)
	    w = w.cloneOn(OnDefaultDevice);
	}

	// Superbblas tensor contraction is shit and those not deal with subtensors (for now)
	if (v.isSubtensor()) v = v.clone();
	if (w.isSubtensor()) w = w.clone();
	if (isSubtensor()) {
		Tensor<N, T> aux = like_this();
		aux.contract(v, mv, conjv, w, mw, conjw, mr);
		aux.copyTo(*this);
		return;
	}

	T* v_ptr = v.data.get();
	T* w_ptr = w.data.get();
	T* ptr = this->data.get();
	Order<Nv + 1> orderv_ = detail::toOrderStr(detail::update_order(v.order, mv));
	Order<Nw + 1> orderw_ = detail::toOrderStr(detail::update_order(w.order, mw));
	Order<N + 1> order_ = detail::toOrderStr(detail::update_order(order, mr));
	superbblas::contraction<Nv, Nw, N>(
	  v.scalar * w.scalar / scalar, v.p->p.data(), 1, &orderv_[0], conjv == Conjugate,
	  (const T**)&v_ptr, &*v.ctx, w.p->p.data(), 1, &orderw_[0], conjw == Conjugate,
	  (const T**)&w_ptr, &*w.ctx, beta, p->p.data(), 1, &order_[0], &ptr, &*ctx, MPI_COMM_WORLD,
	  superbblas::FastToSlow);
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
	from = {};
	size = {};
	strides = {};
	scalar = T{0};
      }

      // Return whether the current view is contiguous in memory
      bool isContiguous() const
      {
	// Meaningless for tensors not been fully supported on a single node
	if (dist != OnMaster)
	  return false;

	if (superbblas::detail::volume<N>(size) > 0 && N > 1)
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

      // Return a context on either the host or the device
      static std::shared_ptr<superbblas::Context> getContext(DeviceHost dev)
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
#ifdef QDP_IS_QDPJIT
	  if (!cudactx)
	    cudactx = std::make_shared<superbblas::Context>(superbblas::createCudaContext());
	  return cudactx;
#else
	  return cpuctx;
#endif
	}
	throw std::runtime_error("Unsupported `DeviceHost`");
      }

      /// Get where the tensor is stored

      DeviceHost getDev() const
      {
#ifdef QDP_IS_QDPJIT
	return (ctx->plat != superbblas::CPU ? OnDefaultDevice : OnHost);
#else
	return OnDefaultDevice;
#endif
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
	std::size_t vol = superbblas::detail::volume<N>(size);
	std::size_t disp = detail::coor2index<N>(from, dim, strides);
	std::size_t word_size = sizeof(typename detail::WordType<T>::type);
	bin.readArrayPrimaryNode((char*)&data.get()[disp], word_size, sizeof(T) / word_size * vol);
      }

      void binaryWrite(BinaryWriter& bin) const
      {
	if (dist != OnMaster)
	  throw std::runtime_error("Only supported to write from `OnMaster` tensors");

	if (scalar != T{1} || !isContiguous())
	{
	  clone().binaryWrite(bin);
	  return;
	}

	// Only on primary node write the data
	std::size_t vol = superbblas::detail::volume<N>(size);
	std::size_t disp = detail::coor2index<N>(from, dim, strides);
	bin.writeArrayPrimaryNode((char*)&data.get()[disp], sizeof(T), vol);
      }

      void print(std::string name) const
      {
	std::stringstream ss;
	Tensor<N, T> t_host = like_this(none, {}, OnHost, OnMaster);
	copyTo(t_host);
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
#if 0
      /// Get where the tensor is stored
      void setNan() 
      {
#  ifndef QDP_IS_QDPJIT
	T nan = detail::NaN<T>::get();
	std::size_t vol = superbblas::detail::volume<N>(dim);
	T* p = data.get();
	for (std::size_t i = 0; i < vol; i++)
	  p[i] = nan;
#  endif
    }

    void
    checkNan() const
    {
#  ifndef QDP_IS_QDPJIT
	std::size_t vol = superbblas::detail::volume<N>(dim);
	T* p = data.get();
	for (std::size_t i = 0; i < vol; i++)
	  assert(detail::IsFinite<T>::get(p[i]));
#  endif
      }
#endif
    };

    template <typename T>
    using LatticeColorVectorT = OLattice<PScalar<PColorVector<RComplex<T>, Nc>>>;

    template <typename T>
    Tensor<Nd + 2, std::complex<T>> asTensorView(const LatticeColorVectorT<T>& v)
    {
      using Complex = std::complex<T>;
      Complex* v_ptr = reinterpret_cast<Complex*>(v.getF());
      return Tensor<Nd + 2, Complex>("cxyztX", latticeSize<Nd + 2>("cxyztX"), OnDefaultDevice,
				     OnEveryone, std::shared_ptr<Complex>(v_ptr, [](Complex*) {}));
    }

    Tensor<Nd + 3, Complex> asTensorView(const LatticeFermion& v)
    {
      Complex* v_ptr = reinterpret_cast<Complex*>(v.getF());
      return Tensor<Nd + 3, Complex>("csxyztX", latticeSize<Nd + 3>("csxyztX"), OnDefaultDevice,
				     OnEveryone, std::shared_ptr<Complex>(v_ptr, [](Complex*) {}));
    }

    Tensor<Nd + 1, Complex> asTensorView(const LatticeComplex& v)
    {
      Complex* v_ptr = reinterpret_cast<Complex*>(v.getF());
      return Tensor<Nd + 1, Complex>("xyztX", latticeSize<Nd + 1>("xyztX"), OnDefaultDevice,
				     OnEveryone, std::shared_ptr<Complex>(v_ptr, [](Complex*) {}));
    }

    Tensor<Nd + 3, Complex> asTensorView(const LatticeColorMatrix& v)
    {
      Complex* v_ptr = reinterpret_cast<Complex*>(v.getF());
      return Tensor<Nd + 3, Complex>(
	"jixyztX", latticeSize<Nd + 3>("jixyztX", {{'i', Nc}, {'j', Nc}}), OnDefaultDevice,
	OnEveryone, std::shared_ptr<Complex>(v_ptr, [](Complex*) {}));
    }

    Tensor<Nd + 4, Complex> asTensorView(const LatticeColorVectorSpinMatrix& v)
    {
      Complex* v_ptr = reinterpret_cast<Complex*>(v.getF());
      return Tensor<Nd + 4, Complex>(
	"cjixyztX", latticeSize<Nd + 4>("cjixyztX", {{'i', Ns}, {'j', Ns}}), OnDefaultDevice,
	OnEveryone, std::shared_ptr<Complex>(v_ptr, [](Complex*) {}));
    }

    template <typename COMPLEX>
    Tensor<1, COMPLEX> asTensorView(std::vector<COMPLEX>& v)
    {
      return Tensor<1, COMPLEX>("i", Coor<1>{Index(v.size())}, OnHost, OnEveryoneReplicated,
				std::shared_ptr<COMPLEX>(v.data(), [](COMPLEX*) {}));
    }

    Tensor<2, Complex> asTensorView(SpinMatrix& smat)
    {
      Complex* v_ptr = reinterpret_cast<Complex*>(smat.getF());
      return Tensor<2, Complex>("ji", Coor<2>{Ns, Ns}, OnHost, OnEveryoneReplicated,
				std::shared_ptr<Complex>(v_ptr, [](Complex*) {}));
    }

    SpinMatrix SpinMatrixIdentity()
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

    namespace ns_getColorvecs
    {
      /// Return the permutation from a natural layout to a red-black that is used by `applyPerm`
      /// \param t: index in the t-direction of the input elements
      /// NOTE: assuming the input layout is xyz and the output layout is xyzX for the given input
      ///       time-slice `t`

      std::vector<Index> getPermFromNatToRB(Index t)
      {
	if (Layout::nodeNumber() != 0)
	  return {};

	const Index x1 = Layout::lattSize()[0];
	const Index y1 = Layout::lattSize()[1];
	const Index z1 = Layout::lattSize()[2];
	std::vector<Index> perm(x1 * y1 * z1);

#if QDP_USE_LEXICO_LAYOUT
	unsigned int n = x1 * y1 * z1;
#  ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#  endif
	for (unsigned int i = 0; i < n; ++n)
	  perm[i] = i;

#elif QDP_USE_CB2_LAYOUT
#  ifdef _OPENMP
#    pragma omp parallel for collapse(3) schedule(static)
#  endif
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

#else
	throw std::runtime_error("Unsupported layout");
#endif

	return perm;
      }

      /// Apply a permutation generated by `getPermFromNatToRB`
      /// \param perm: permutation generated with getPermFromNatToRB
      /// \param tnat: input tensor with ordering cxyz
      /// \param trb: output tensor with ordering cxyzX

      template <typename T>
      void applyPerm(const std::vector<Index>& perm, Tensor<Nd, T> tnat, Tensor<Nd + 1, T> trb)
      {
	assert((tnat.order == Order<Nd>{'c', 'x', 'y', 'z'}));
	assert((trb.order == Order<Nd + 1>{'c', 'x', 'y', 'z', 'X'}));
	assert(tnat.p->localVolume() == perm.size() * Nc);

	unsigned int i1 = perm.size();
	const T *x = tnat.data.get();
	T* y = trb.data.get();

#ifdef _OPENMP
#  pragma omp parallel for schedule(static)
#endif
	for (unsigned int i = 0; i < i1; ++i)
	  for (unsigned int c = 0; c < Nc; ++c)
	    y[i * Nc + c] = x[perm[i] * Nc + c];
      }
    }

    typedef QDP::MapObjectDiskMultiple<KeyTimeSliceColorVec_t, Tensor<Nd, ComplexF>> MODS_t;

    /// Get colorvecs(t,n) for t=from_slice..(from_slice+n_tslices-1) and n=0..(n_colorvecs-1)

    Tensor<Nd + 3, ComplexF> getColorvecs(MODS_t& eigen_source, int decay_dir, int from_tslice,
					  int n_tslices, int n_colorvecs,
					  const char order[] = "cxyztXn")
    {
      using namespace ns_getColorvecs;

      StopWatch sw;
      sw.reset();
      sw.start();

      if (decay_dir != 3)
	throw std::runtime_error("Only support for decay_dir being the temporal dimension");
      detail::check_order_contains(order, "cxyztXn");

      from_tslice = detail::normalize_coor(from_tslice, Layout::lattSize()[decay_dir]);

      // Allocate tensor to return
      Tensor<Nd + 3, ComplexF> r(
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
	std::vector<Index> perm = getPermFromNatToRB(t_slice);

	for (int colorvec = 0; colorvec < n_colorvecs; ++colorvec)
	{
	  // Read a single time-slice and colorvec
	  KeyTimeSliceColorVec_t key(t_slice, colorvec);
	  eigen_source.get(key, tnat);

	  // Correct ordering
	  applyPerm(perm, tnat, trb);

	  // t[n=colorvec] = trb
	  trb.copyTo(t.kvslice_from_size({{'n', colorvec}}, {{'n', 1}}));
	}

	// r[t=i_slice] = t, distribute the tensor from master to the rest of the nodes
	t.copyTo(r.kvslice_from_size({{'t', i_slice}}));
      }

      sw.stop();
      QDPIO::cout << "Time to read " << n_colorvecs << " colorvecs from " << n_tslices
		  << " time slices: " << sw.getTimeInSeconds() << " secs" << std::endl;

      return r;
    }

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
    Tensor<Nd + 5, COMPLEX_OUT>
    doInversion(const SystemSolver<LatticeFermion>& PP, const Tensor<Nd + 3, COMPLEX_CHI> chi,
		int t_source, int first_tslice_out, int n_tslice_out, std::vector<int> spin_sources,
		int max_rhs, const char* order_out = "cSxyztXns")
    {
      detail::check_order_contains(order_out, "cSxyztXns");
      if (chi.kvdim()['t'] != 1)
	throw std::runtime_error("Expected one time-slice");
      const int num_vecs = chi.kvdim()['n'];

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

      for (int spin_source : spin_sources)
      {
	for (int n0 = 0, n_step = std::min(max_rhs, num_vecs); n0 < num_vecs;
	     n0 += n_step, n_step = std::min(n_step, num_vecs - n0))
	{
	  StopWatch snarss1;
	  snarss1.reset();
	  snarss1.start();

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

	  snarss1.stop();
	  QDPIO::cout << "Time to compute inversions for spin_source= " << spin_source
		      << "  colorvec_src= " << n0 << " to " << n0 + n_step - 1
		      << "  time = " << snarss1.getTimeInSeconds() << " secs" << std::endl;

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

      return psi;
    }

    template <typename COMPLEX, std::size_t N>
    Tensor<N, COMPLEX> shift(const Tensor<N, COMPLEX> v, int len, int dir)
    {
      if (dir < 0 || dir >= Nd - 1)
	throw std::runtime_error("Invalid direction");

      if (len == 0)
	return v;

      const char dir_label[] = "xyz";
#if QDP_USE_LEXICO_LAYOUT
      // If we are not using red-black ordering, return a view where the tensor is shifted on the given direction
      return v.kvslice_from_size({{dir_label[dir], -len}});

#elif QDP_USE_CB2_LAYOUT
      // Assuming that v has support on the origin and destination lattice elements
      if (v.kvdim()['X'] != 2 && len % 2 != 0)
	throw std::runtime_error("Unsupported shift");

      Tensor<N, COMPLEX> r = v.like_this();
      if (dir != 0)
      {
	v.copyTo(r.kvslice_from_size({{'X', len}, {dir_label[dir], -len}}));
      }
      else
      {
	throw std::runtime_error("Unsupported shift on X direction for this layout");
	// for (int X = 0; X < 2; ++X)
	//   v.kvslice_from_size({{'X', X}})
	//     .copyTo(r.kvslice_from_size(
	//       {{'X', (X + len) % 2}, {dir_label[dir], (X + len + (len >= 0 ? 0 : -1)) / 2}}));
      }
      return r;
#else
      throw std::runtime_error("Unsupported layout");
#endif
    }

    /// Compute displace/derivate of v in the direction dir
    /// \param u: Gauge field
    /// \param v: tensor to apply the derivative
    /// \param dir: 0: nothing; 1: forward x; -1: backward x; 2: forward y...
    /// \patam deriv: whether use derivative
    /// \param moms: list of input momenta
    /// \param conjUnderAdd: if true, return a version, R(dir), so that
    ////       adj(R(dir)) * D(dir') == D(dir+dir'), where D(dir') is what this function returns
    ////       when conjUnderAdd is false.

    template <typename COMPLEX, std::size_t N>
    Tensor<N, COMPLEX> displace(const std::vector<Tensor<Nd + 3, Complex>>& u,
				Tensor<N, COMPLEX> v, int dir, bool deriv = false,
				std::vector<Coor<Nd - 1>> moms = {}, bool conjUnderAdd = false)
    {
      if (std::abs(dir) > Nd)
	throw std::runtime_error("Invalid direction");

      if (dir == 0)
	return v;

      int d = std::abs(dir) - 1; // space lattice direction, 0: x, 1: y, 2: z
      int len = (dir > 0 ? 1 : -1); // displacement unit direction

      Tensor<N, COMPLEX> r = v.like_this("cnSsxyzXt");
      if (!deriv)
      {
	assert(d < u.size());

	if (conjUnderAdd)
	  len *= -1;

	v = v.reorder("cnSsxyzXt");
	if (len > 0)
	{
	  // Do u[d] * shift(x,d)
	  v = shift(std::move(v), len, d);
	  r.contract(u[d], {{'j', 'c'}}, NotConjugate, std::move(v), {}, NotConjugate,
		     {{'c', 'i'}});
	}
	else
	{
	  // Do shift(adj(u[d]) * x,d)
	  r.contract(u[d], {{'i', 'c'}}, Conjugate, std::move(v), {}, NotConjugate, {{'c', 'j'}});
	  r = shift(std::move(r), len, d);
	}
      }
      else
      {
	// conj(phase)*displace(u, v, -length, d) - phase*displace(u, v, length, d)
	std::vector<COMPLEX> phases(moms.size());
	for (unsigned int i = 0; i < moms.size(); ++i)
	{
	  
	  double angle = 2 * M_PI * moms[i][d] / Layout::lattSize()[d];
	  phases[i] = COMPLEX{1} + COMPLEX{cos(angle), sin(angle)};
	  if (conjUnderAdd)
	    phases[i] = std::sqrt(phases[i]);
	}

	// r = conj(phases) * displace(u, v, dir)
	r.contract(displace(u, v, -dir), {}, NotConjugate, asTensorView(phases), {{'i', 'm'}},
		   Conjugate);
	// r = r - phases * displace(u, v, dir) if !ConjUnderAdd else r + phases * displace(u, v, dir)
	r.contract(displace(u, v, dir, false).scale(conjUnderAdd ? 1 : -1), {}, NotConjugate,
		   asTensorView(phases), {{'i', 'm'}}, NotConjugate, {}, 1.0);
      }
      return r;
    }

    namespace ns_doMomGammaDisp_contractions
    {
      /// Path Node
      struct PathNode {
	std::map<char, PathNode> p; ///< following nodes
	int disp_index;		    ///< if >= 0, the index in the displacement list 
      };

      /// Return the tree representing all paths
      PathNode get_tree(const std::vector<std::vector<int>>& paths)
      {
	PathNode r{{}, -1};
	int path_index = 0;
	for (const std::vector<int>& path : paths)
	{
	  PathNode* n = &r;
	  for (char d : path)
	  {
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

      /// Contract two LatticeFermion with different momenta, gammas, and displacements.
      /// \param leftconj: left lattice fermion tensor, cSxyzXN
      /// \param right: right lattice fermion tensor, csxyzXn
      /// \param disps: tree of displacements/derivatives
      /// \param deriv: whether use derivatives instead of displacements
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
				       Tensor<Nright, COMPLEX> right, const PathNode& disps,
				       bool deriv, Tensor<3, COMPLEX> gammas,
				       const std::vector<Coor<Nd - 1>>& moms, int max_rhs,
				       Tensor<Nout, COMPLEX> r, std::vector<int>& disp_indices)
      {
	max_rhs = std::max(1, max_rhs);

	if (disps.disp_index >= 0)
	{
	  detail::log(1, std::string("contracting for disp_index=") +
			   std::to_string(disps.disp_index));
	  // Contract the spatial components and the color of the leftconj and right tensors
	  Tensor<Nout, COMPLEX> aux =
	    r.template like_this<Nout, COMPLEX>("mNQqnSst", {{'S', Ns}, {'Q', Ns}});
	  aux.contract(leftconj, {}, Conjugate, right.reorder("cxyzXnSst"), {}, NotConjugate, {});

	  // Contract the spin components S and Q with the gammas, and put the result on r[d=disp_indices.size()]
	  aux = aux.reorder("QSmNqnst");
	  Tensor<Nout - 1, COMPLEX> aux0 = r.template like_this<Nout - 1, COMPLEX>("gmNqnst");
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
	  detail::log(1, std::string("push on direction ") + std::to_string(it.first));
	  // Apply displacement on the right vectors
	  // NOTE: avoid that the memory requirements grow linearly with the number of displacements
	  //       by killing the reference to `right` as soon as possible
	  Tensor<Nright, COMPLEX> right_disp =
	    (node_disp < disps.p.size() - 1) ? displace(u, right, it.first, deriv, moms)
					     : displace(u, std::move(right), it.first, deriv, moms);
	  doMomGammaDisp_contractions(u, leftconj, std::move(right_disp), it.second, deriv, gammas,
				      moms, max_rhs - num_vecs, r, disp_indices);
	  node_disp++;
	  detail::log(1, std::string("pop direction"));
	}
      }
    }

    /// Contract two LatticeFermion with different momenta, gammas, and displacements.
    /// \param leftconj: left lattice fermion tensor, cxyzXNQqt
    /// \param right: right lattice fermion tensor, cxyzXnSst
    /// \param first_tslice: first time-slice in leftconj and right
    /// \param moms: momenta to apply
    /// \param gammas: list of gamma matrices to apply
    /// \param disps: list of displacements/derivatives
    /// \param deriv: whether use derivatives instead of displacements 
    /// \param max_rhs: maximum number of vectors hold in memory
    /// \param order_out: coordinate order of the output tensor, a permutation of nNSQmgd where
    ///        q and N (s and n) are the spin and vector from left (right) vectors, m is the momentum
    ///        index, g is the gamma index, and d is the displacement index
    /// \return: a pair made of a tensor sqnNmgd and a vector that is a dictionary that map each `d`
    ///        index in the tensor with an input displacement index.

    template <std::size_t Nout, std::size_t Nleft, std::size_t Nright, typename COMPLEX>
    std::pair<Tensor<Nout, COMPLEX>, std::vector<int>> doMomGammaDisp_contractions(
      const multi1d<LatticeColorMatrix>& u, Tensor<Nleft, COMPLEX> leftconj,
      const Tensor<Nright, COMPLEX> right, int first_tslice, const SftMom& moms,
      const std::vector<Tensor<2, COMPLEX>>& gammas, std::vector<std::vector<int>> disps,
      bool deriv, int max_rhs, const char* order_out = "gmNndsqt")
    {
      using namespace ns_doMomGammaDisp_contractions;

      detail::check_order_contains(order_out, "gmNndsqt");
      detail::check_order_contains(leftconj.order, "cxyzXNQqt");
      detail::check_order_contains(right.order, "cxyzXnSst");

      assert(right.kvdim()['t'] == leftconj.kvdim()['t']);
      unsigned int Nt = right.kvdim()['t'];

      // Form a tree with the displacement paths
      PathNode tree_disps = get_tree(disps);

      // Allocate output tensor
      Tensor<Nout, COMPLEX> r(order_out, kvcoors<Nout>(order_out, {{'t', Nt},
								   {'n', right.kvdim()['n']},
								   {'s', right.kvdim()['s']},
								   {'N', leftconj.kvdim()['N']},
								   {'q', leftconj.kvdim()['q']},
								   {'m', moms.numMom()},
								   {'g', gammas.size()},
								   {'d', disps.size()}}));

      // Copy moms into a single tensor
      const char momst_order[] = "mxyzXt";
      Tensor<Nd + 2, COMPLEX> momst(
	momst_order, latticeSize<Nd + 2>(momst_order, {{'t', Nt}, {'m', moms.numMom()}}));
      for (unsigned int mom = 0; mom < moms.numMom(); ++mom)
      {
	asTensorView(moms[mom])
	  .kvslice_from_size({}, {{'t', Nt}})
	  .copyTo(momst.kvslice_from_size({{'m', mom}}, {{'m', 1}}));
      }

      // Apply momenta conjugated to the left tensor and rename the spin components s and Q to q and Q,
      // and the colorvector component n to N
      Tensor<Nleft + 1, COMPLEX> moms_left =
	leftconj.template like_this<Nleft + 1>("mQNqcxyzXt", {{'m', moms.numMom()}});
      leftconj = leftconj.reorder("QNqcxyzXt");
      moms_left.contract(std::move(momst), {}, Conjugate, std::move(leftconj), {}, NotConjugate);
      moms_left = moms_left.reorder("cxyzXmNQqt");

      // Create mom_list
      std::vector<Coor<Nd - 1>> mom_list(moms.numMom());
      for (unsigned int mom = 0; mom < moms.numMom(); ++mom)
      {
	for (unsigned int i = 0; i < Nd - 1; ++i)
	  mom_list[mom][i] = moms.numToMom(mom)[i];
      }

      // Copy all gammas into a single tensor
      const char gammast_order[] = "gQS";
      Tensor<3, COMPLEX> gammast(gammast_order, {(Index)gammas.size(), Ns, Ns}, OnDefaultDevice,
				 OnEveryoneReplicated);
      for (unsigned int g = 0; g < gammas.size(); g++) {
	gammas[g]
	  .rename_dims({{'i', 'Q'}, {'j', 'S'}})
	  .copyTo(gammast.kvslice_from_size({{'g', g}}, {{'g', 1}}));
      }

      // Make a copy of the time-slicing of u[d] also supporting left and right
      std::vector<Tensor<Nd + 3, Complex>> ut;
      for (unsigned int d = 0; d < Nd - 1; d++)
      {
	// NOTE: This is going to create a tensor with the same distribution of the t-dimension as leftconj and right
	ut.push_back(asTensorView(u[d])
		       .kvslice_from_size({{'t', first_tslice}}, {{'t', Nt}})
		       .reorder("ijxyzXt"));
      }

      // Do the thing
      std::vector<int> disp_indices;
      if (!deriv)
      {
	doMomGammaDisp_contractions(ut, std::move(moms_left), std::move(right), tree_disps, deriv,
				    gammast, mom_list, max_rhs, r, disp_indices);
      }
      else
      {
	// std::vector<COMPLEX> ones(moms.numMom(), COMPLEX(1));
	// std::string right_moms_order = std::string(right.order.begin(), right.order.size()) + "m";
	// Tensor<Nright + 1, COMPLEX> right_moms =
	//   right.like_this<Nright + 1>(right_moms_order.c_str());
	// right_moms.contract(asTensorView(ones), {{'i', 'm'}}, NotConjugate, std::move(right), {},
	// 		    NotConjugate);
	// doMomGammaDisp_contractions(u, gammast_moms_left, right_moms, tree_disps, deriv, mom_list,
	// 			    max_rhs, r, disp_indices);
      }

      return {r, disp_indices};
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

#endif // __INCLUDE_SUPERB_CONTRACTIONS__
