module LyceumBase

using Adapt
using AutoHashEquals

using Base: @propagate_inbounds
using Base.Threads: Atomic, atomic_add!, atomic_sub!

using Dates
using Distributions: Distributions, Sampleable, sample, sample!
using DocStringExtensions
using ElasticArrays
using Future: randjump
using IterTools
using LinearAlgebra
using MacroTools
using Pkg
using Printf
using Random
using Shapes

using SpecialArrays
using SpecialArrays: True, False

using StaticArrays
using StructArrays
using UnicodePlots: UnicodePlots
using UnsafeArrays
using Unpack


include("util.jl")


####
#### Interfaces
####

"""
    tconstruct(T::Type, n::Integer, args...; kwargs...)

Return a `AbstractVector` of `n` instances of `T`. By default, this returns
`[T(args...; kwargs...) for _=1:n], but this function can be extended for types that can share
data across instances for greater cache efficiency/performance.
"""
function tconstruct(T::Type, n::Integer, args...; kwargs...)
    n > 0 || throw(ArgumentError("n must be > 0"))
    ntuple(_ -> T(args...; kwargs...), n)
end
export tconstruct


export AbstractEnvironment
export statespace, getstate!, getstate, setstate!
export observationspace, getobservation!, getobservation
export actionspace, getaction!, getaction, setaction!
export rewardspace, getreward
export reset!, randreset!
export step!, isdone, timestep
include("abstractenvironment.jl")


####
#### Tools
####

include("setfield.jl")
using .SetfieldImpl: @set!

include("math.jl")

export tseed!, getrange, splitrange, nblasthreads, @with_blasthreads
include("threading.jl")

export SPoint3D, MPoint3D
include("geometry.jl")

export Trajectory, TrajectoryBuffer, ntrajectories, nsamples, rollout!, rollout
include("trajectory.jl")

export EnvironmentSampler
include("environmentsampler.jl")

include("plotting.jl")

end # module
