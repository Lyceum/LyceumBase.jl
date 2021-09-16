using LyceumBase.Tools, Test
using Shapes, ElasticArrays, Random, KahanSummation, LinearAlgebra

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
