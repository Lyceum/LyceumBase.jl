module SpecialArrays

using UnsafeArrays, Adapt, Shapes
using Base: @propagate_inbounds, @pure, @_inline_meta, ViewIndex, require_one_based_indexing, tail
using Base.MultiplicativeInverses: SignedMultiplicativeInverse

const AbsArr{T,N} = AbstractArray{T, N}
const AbsMat{T} = AbstractMatrix{T}
const AbsVec{T} = AbstractVector{T}

const IDims{N} = NTuple{N,Integer}
const IVararg{N} = Vararg{Integer,N}

include("util.jl")
include("viewtype.jl")

export Slices
include("slices.jl")

export
    NestedView,
    innerview,
    outerview,
    flatten,
    flattento!,
    flatview,
    inner_eltype,
    inner_ndims,
    inner_size,
    inner_length,
    inner_axes

include("nestedarrays/util.jl")
include("nestedarrays/functions.jl")
include("nestedarrays/nestedview.jl")

export ElasticArray, shrinklastdim!, growlastdim!, resizelastdim!
include("elasticarray.jl")

export BatchedVector
include("batchedvector.jl")

end # module