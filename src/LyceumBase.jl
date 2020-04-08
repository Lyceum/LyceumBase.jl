module LyceumBase

using AutoHashEquals
using Adapt

using Base: @propagate_inbounds
using Base.Threads: Atomic, atomic_add!, atomic_sub!

using Dates
using DocStringExtensions
using Distributions: Distributions, Sampleable, sample, sample!
using FastClosures: @closure
using Future: randjump
using InteractiveUtils
using JLSO
using LinearAlgebra
using LibGit2
using Logging
using LyceumCore
using MacroTools
using Parameters
using Pkg
using Printf: @sprintf
using Random
using Reexport

using SpecialArrays
using SpecialArrays: True, False

using Shapes
using StaticArrays
using UnicodePlots: UnicodePlots
using UniversalLogger
using UnsafeArrays
import UniversalLogger: finish!


####
#### Interfaces
####

"""
    $(SIGNATURES)

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
@reexport using .SetfieldImpl

include("math.jl")

export tseed!, getrange, splitrange, nblasthreads, @with_blasthreads
include("threading.jl")

export SPoint3D, MPoint3D
include("geometry.jl")

export Trajectory, TrajectoryBuffer
include("trajectory.jl")

export EnvironmentSampler, sample, sample!
include("environmentsampler.jl")

include("projectmeta.jl")

export Line, termplot
include("plotting.jl")

export Experiment, finish!
include("experiment.jl")

end # module
