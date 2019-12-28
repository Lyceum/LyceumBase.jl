using LyceumBase
using Test
using BenchmarkTools
using Random


@testset "LyceumBase.jl" begin
    @testset "Tools" begin include("Tools/Tools.jl") end
    @testset "setfield" begin include("setfield.jl") end
    @testset "SpecialArrays" begin
        include("SpecialArrays/SpecialArrays.jl")
    end
end
