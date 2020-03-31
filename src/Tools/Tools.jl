module Tools

using ..LyceumBase

using Adapt
using Dates
using Distributions: Sampleable
using ElasticArrays
using EllipsisNotation
using Future: randjump
using InteractiveUtils
using JLSO
using LibGit2
using LinearAlgebra
using Logging
using LyceumCore
using MacroTools
using Parameters
using Pkg
using Random
using Shapes
using StaticArrays
using Statistics
using Test: @test
using UnicodePlots: UnicodePlots
using UniversalLogger
import UniversalLogger: finish!
using UnsafeArrays

include("misc.jl")

export
    delta,
    scaleandcenter!,
    zerofn,
    noop,
    symmul!,
    @forwardfield,
    wraptopi,
    tuplecat,
    Converged,
    @noalloc,
    mkgoodpath,
    filter_nt,
    perturb!,
    perturbn!,
    perturb,
    perturbn

include("projectmeta.jl")

include("plotting.jl")
export Line, expplot

include("experiment.jl")
export Experiment, finish!

include("threading.jl")
export seed_threadrngs!, threadrngs, getrange, splitrange, nblasthreads, @with_blasthreads

include("meta.jl")

include("elasticbuffer.jl")
export ElasticBuffer, fieldarrays, grow!

include("batchedarray.jl")
export BatchedArray, flatview, batchlike

include("envsampler.jl")
export EnvSampler, sample!, TrajectoryBuffer, grow!

include("geom.jl")
export SPoint3D, MPoint3D

end # module
