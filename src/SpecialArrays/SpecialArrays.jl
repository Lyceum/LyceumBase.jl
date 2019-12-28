module SpecialArrays

using UnsafeArrays, Adapt
using Base: @propagate_inbounds, @pure, front, last, @_inline_meta, ViewIndex
using Base.MultiplicativeInverses: SignedMultiplicativeInverse

const AbsArr{T,N} = AbstractArray{T, N}

include("util.jl")

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
include("nestedarrays/nestedarray.jl")

export ElasticBuffer, shrinkend!, growend!, resizeend!
include("elasticbuffer.jl")

end # module