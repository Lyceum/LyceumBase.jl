module SpecialArraysTest

t1 = time()

using Base: index_shape, index_dimsum, index_ndims, to_indices
using Test
using Random

using AxisArrays: AxisArrays
using UnsafeArrays
using BenchmarkTools
using StaticNumbers

using ..LyceumBase.LyceumCore
using ..LyceumBase.SpecialArrays
using ..LyceumBase.SpecialArrays: _maybe_unsqueeze
using ..LyceumBase.TestUtil
t2 = time()

include("testutil.jl")

nones(N::Integer) = ntuple(_ -> 1, Val(unstatic(N)))

testdims(L::Integer) = ntuple(i -> 3 + i, Val(unstatic(L)))

#@testset "functions" begin
#    include("functions.jl")
#end

#@testset "elasticarray" begin
#    include("elasticarray.jl")
#end
t3 = time()
@testset "slices" begin
    include("slices.jl")
end
t4 = time()

@info "" t1 t2 t3 t4 t4 - t3 t2 - t1 t3 - t2

end