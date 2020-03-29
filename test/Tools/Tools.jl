module ToolsTest

using LyceumBase.Tools
using Test
using Shapes
using ElasticArrays
using UnsafeArrays
using Random
using Distributions: Uniform
using LinearAlgebra

@testset "LyceumTools.jl" begin
    @testset "stats" begin
        include("stats.jl")
    end
    @testset "threading" begin
        include("threading.jl")
    end
    @testset "misc" begin
        include("misc.jl")
    end
    @testset "geom" begin
        include("geom.jl")
    end
end

end
