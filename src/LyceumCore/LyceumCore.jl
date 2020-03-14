module LyceumCore

using Base: @pure

using ..StaticNumbers

const Maybe{T} = Union{T,Nothing}

const TupleN{T,N} = NTuple{N,T}
const VarargN{N,T} = Vararg{T,N}

const AbsArr{T,N} = AbstractArray{T,N}
const AbsMat{T} = AbstractMatrix{T}
const AbsVec{T} = AbstractVector{T}

const RealArr{N} = AbstractArray{<:Real,N}
const RealMat = AbstractMatrix{<:Real}
const RealVec = AbstractVector{<:Real}

const AbsNestedArr{N} = AbstractArray{<:AbstractArray,N}
const AbsSimilarNestedArr{V,M,N} = AbstractArray{<:AbstractArray{V,M},N}


export Maybe
export TupleN, VargargN
export AbsArr, AbsMat, AbsVec
export RealArr, RealMat, RealVec
export AbsNestedArr, AbsSimilarNestedArr

export wrapval, unwrapval
export StaticTrue, StaticFalse, StaticOr
export static_not, static_in, static_sum, static_filter
include("static.jl")

export ncolons, front, tail, split, tuple_length
include("util.jl")

end