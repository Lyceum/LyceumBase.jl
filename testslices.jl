include("src/LyceumBase.jl")
#using LyceumBase
#using LyceumBase.SpecIlArrays
#using LyceumBase.LyceumCore

using Random
using UnsafeArrays
using StaticNumbers
using AxisArrays: AxisArray



module Mod

using ..LyceumBase.LyceumCore
using Random
using UnsafeArrays
using StaticNumbers
using ..LyceumBase.SpecialArrays

using ..InteractiveUtils
using BenchmarkTools

include("testutil.jl")

function test()
    A = rand(2,3,4,1)
    #B = reshape(mapreduce(vcat, hcat, A), (2,3,4))
    #B = [view(A, :, :, i) for i=1:4]
    B = [A[:, :, i] for i=1:4, _=1:1]
    test_array(B) do
        Slices(deepcopy(A), 1, 2)
    end
end
using Test


end