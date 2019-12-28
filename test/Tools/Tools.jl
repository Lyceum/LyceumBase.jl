module ToolsTest

using LyceumBase.Tools, Test
using Shapes, ElasticArrays, UnsafeArrays, Random, KahanSummation, LinearAlgebra

@testset "stats" begin
    include("stats.jl")
end
@testset "threading" begin
    include("threading.jl")
end
@testset "misc" begin
    include("misc.jl")
end

end