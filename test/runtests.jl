using LyceumBase
using Test
using BenchmarkTools

@testset "LyceumBase.jl" begin
    #@testset "Tools" begin include("Tools/Tools.jl") end
    #@testset "setfield" begin include("setfield.jl") end
    @testset "SpecialArrays" begin
        include("SpecialArrays/SpecialArrays.jl")
    end
end
