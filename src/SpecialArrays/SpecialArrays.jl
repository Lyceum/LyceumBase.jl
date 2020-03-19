module SpecialArrays

using Base: @propagate_inbounds, @pure, @_inline_meta, require_one_based_indexing
using Base.MultiplicativeInverses: SignedMultiplicativeInverse

using UnsafeArrays
using Adapt
using StaticNumbers
using StaticArrays: SOneTo
using ..MacroTools: @forward

using ..LyceumBase.LyceumCore
using Shapes

const IDims{N} = NTuple{N,Integer}
const IVararg{N} = Vararg{Integer,N}

include("viewtype.jl")

export flatten, flatten!, nest, nest!
export nestedview, flatview
export innereltype, innerndims, innersize, innerlength, inneraxes
include("functions.jl")

export Slices
include("slices.jl")

export Align
include("align.jl")

include("nestedarrays/util.jl")
include("nestedarrays/nestedview.jl")

export ElasticArray, shrinklastdim!, growlastdim!, resizelastdim!
include("elasticarray.jl")

export BatchedVector
include("batchedvector.jl")

end # module