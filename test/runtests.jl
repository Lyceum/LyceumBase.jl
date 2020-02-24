using Test
using LyceumBase
using Random: MersenneTwister
using Distributions: Uniform

@testset "LyceumBase.jl" begin
    @testset "Tools" begin include("Tools/Tools.jl") end
    @testset "setfield" begin include("setfield.jl") end
end
