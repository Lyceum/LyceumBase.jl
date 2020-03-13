module LyceumCore

const Maybe{T} = Union{T,Nothing}

const TupleN{T,N} = NTuple{N,T}
const VarargN{N,T} = Vararg{T,N}

const AbsArr{T,N} = AbstractArray{T,N}
const AbsMat{T} = AbstractMatrix{T}
const AbsVec{T} = AbstractVector{T}

const RealArr{N} = AbstractArray{<:Real,N}
const RealMat = AbstractMatrix{<:Real}
const RealVec = AbstractVector{<:Real}


export Maybe
export TupleN, VargargN
export AbsArr, AbsMat, AbsVec
export RealArr, RealMat, RealVec

export wrapval, unwrapval
export TypedBool, True, False, Flags, not, untyped
include("traits.jl")

end