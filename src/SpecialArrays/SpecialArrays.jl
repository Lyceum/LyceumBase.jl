module SpecialArrays

using UnsafeArrays, Adapt, Shapes
using Base: @propagate_inbounds, @pure, @_inline_meta, ViewIndex, require_one_based_indexing
using Base.MultiplicativeInverses: SignedMultiplicativeInverse

const AbsVec{T} = AbstractVector{T}
const AbsMat{T} = AbstractMatrix{T}
const AbsArr{T,N} = AbstractArray{T, N}

include("util.jl")
include("viewtype.jl")

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
include("elasticbuffer.jl")

export BatchedVector
include("batchedvector.jl")

end # module