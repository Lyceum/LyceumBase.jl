using LyceumBase
using Test, BenchmarkTools, Random


@testset "LyceumBase.jl" begin
    #@testset "Tools" begin include("Tools/Tools.jl") end

    @testset "SpecialArrays" begin
        include("SpecialArrays/SpecialArrays.jl")
    end
end