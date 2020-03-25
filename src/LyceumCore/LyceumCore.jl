module LyceumCore

using Base: @pure

using ..StaticNumbers
using ..BenchmarkTools

const Maybe{T} = Union{T,Nothing}

const TupleN{T,N} = NTuple{N,T}
const LTuple{L} = NTuple{L,Any}
const NVararg{N,T} = Vararg{T,N}

const AbsArr{T,N} = AbstractArray{T,N}
const AbsMat{T} = AbstractMatrix{T}
const AbsVec{T} = AbstractVector{T}

const RealArr{N} = AbstractArray{<:Real,N}
const RealMat = AbstractMatrix{<:Real}
const RealVec = AbstractVector{<:Real}

const AbsNestedArr{N} = AbstractArray{<:AbstractArray,N}
const AbsSimilarNestedArr{V,M,N} = AbstractArray{<:AbstractArray{V,M},N}


export Maybe
export TupleN, LTuple, NVararg
export AbsArr, AbsMat, AbsVec
export RealArr, RealMat, RealVec
export AbsNestedArr, AbsSimilarNestedArr

export StaticOrVal, SBool, STrue, SFalse
export wrapval, wrapstatic, unwrap
export static_not, static_and, static_or
export static_filter, static_merge, static_in, static_sum
include("static.jl")

export ncolons, front, tail, tuplesplit
include("util.jl")

end