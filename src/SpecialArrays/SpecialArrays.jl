module SpecialArrays

using Base: @propagate_inbounds, @pure, @_inline_meta, require_one_based_indexing
using Base.MultiplicativeInverses: SignedMultiplicativeInverse

using UnsafeArrays
using Adapt
using StaticNumbers
using ..MacroTools: @forward

using ..LyceumBase.LyceumCore
using Shapes

const IDims{N} = NTuple{N,Integer}
const IVararg{N} = Vararg{Integer,N}

include("viewtype.jl")

export innereltype, innerndims, innersize, innerlength, inneraxes
export flatten, flatten!, flatview
export nest, nest!
include("functions.jl")

export Slices, slice
include("slices.jl")

export FlattenedArray
include("flattenedarray.jl")

export ElasticArray, shrinklastdim!, growlastdim!, resizelastdim!
include("elasticarray.jl")

export BatchedVector
include("batchedvector.jl")

end # module