module Tools

# stdlib
using LibGit2, Pkg, Statistics, Random, LinearAlgebra, InteractiveUtils, Logging, Dates
using Future: randjump
using Test: @test

# 3rd party
import UnicodePlots
using UnsafeArrays, StaticArrays, EllipsisNotation, ElasticArrays, MacroTools, JLSO, Parameters, Adapt
using Distributions: Sampleable
using BenchmarkTools: @benchmark

# Lyceum
import UniversalLogger: finish!
using ..LyceumBase, Shapes, UniversalLogger
using LyceumCore
#using ..LyceumBase: TupleN, Maybe, AbsMat

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
