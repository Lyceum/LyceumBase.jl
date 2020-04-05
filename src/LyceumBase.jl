module LyceumBase

using AutoHashEquals

using Base: @propagate_inbounds
using Base.Threads: Atomic, atomic_add!, atomic_sub!

using Dates
using DocStringExtensions
using Distributions: Sampleable
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


include("util.jl")

include("setfield.jl")
@reexport using .SetfieldImpl

export Converged, scaleandcenter!, symmul!, wraptopi
export perturb!, perturbn!, perturb, perturbn
include("math.jl")

export seed_threadrngs!, threadrngs, getrange, splitrange, nblasthreads, @with_blasthreads
include("threading.jl")

export SPoint3D, MPoint3D
include("geometry.jl")

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
export statespace, getstate!, setstate!, getstate
export obsspace, getobs!, getobs
export actionspace, getaction!, setaction!, getaction
export rewardspace, getreward
export reset!, randreset!
export step!, isdone, timestep
export spaces
include("abstractenvironment.jl")


####
#### Tools
####

export Trajectory, TrajectoryBuffer
include("trajectory.jl")

export EnvironmentSampler, sample, sample!
include("environmentsampler.jl")

include("projectmeta.jl")

export Line, expplot
include("plotting.jl")

export Experiment, finish!
include("experiment.jl")

end # module
