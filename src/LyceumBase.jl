module LyceumBase

using Base: @propagate_inbounds
using Test
using Random
using Shapes
using BenchmarkTools: @benchmark
using DocStringExtensions
#using Reexport



const Maybe{T} = Union{T,Nothing}
const TupleN{T,N} = NTuple{N,T}

const AbsArr{T,N} = AbstractArray{T,N}
const AbsMat{T} = AbstractMatrix{T}
const AbsVec{T} = AbstractVector{T}

const RealArr{N} = AbstractArray{<:Real,N}
const RealMat = AbstractMatrix{<:Real}
const RealVec = AbstractVector{<:Real}



include("util.jl")
#include("setfield.jl")
#@reexport using .SetfieldImpl



####
#### Interfaces
####

"""
    tconstruct(T::Type, n::Integer, args...; kwargs...) --> NTuple{n, <:T}

Return a Tuple of `n` instances of `T`. By default, this returns
`ntuple(_ -> T(args...; kwargs...), n)`, but this function can be
extended for types that can share data across instances for greater
cache efficiency/performance.
"""
function tconstruct(T::Type, n::Integer, args...; kwargs...)
    n > 0 || throw(ArgumentError("n must be > 0"))
    ntuple(_ -> T(args...; kwargs...), n)
end
export tconstruct

export
    AbstractEnvironment,

    statespace,
    getstate!,
    setstate!,
    getstate,

    obsspace,
    getobs!,
    getobs,

    actionspace,
    getaction!,
    setaction!,
    getaction,

    rewardspace,
    getreward,

    evalspace,
    geteval,

    reset!,
    randreset!,
    step!,
    isdone,
    timestep,
    spaces
include("abstractenvironment.jl")

export Tools
include("Tools/Tools.jl")

end # module
