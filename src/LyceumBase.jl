module LyceumBase

using Base: @propagate_inbounds
using BenchmarkTools: BenchmarkTools
using DocStringExtensions
using MacroTools
using Random
using Reexport
using Shapes
using StaticNumbers


include("util.jl")

include("setfield.jl")
@reexport using .SetfieldImpl


####
#### Interfaces
####

# TODO MOVE
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


export AbstractEnvironment
export statespace, getstate!, setstate!, getstate
export obsspace, getobs!, getobs
export actionspace, getaction!, setaction!, getaction
export rewardspace, getreward
export evalspace, geteval
export reset!, randreset!
export step!, isdone, timestep
export spaces
include("abstractenvironment.jl")


####
#### Submodules
####

export Tools
include("Tools/Tools.jl")
using .Tools

end # module
