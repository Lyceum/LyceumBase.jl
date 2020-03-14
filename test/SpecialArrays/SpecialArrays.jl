module SpecialArraysTest
using Test, Random

import AxisArrays
using UnsafeArrays
using BenchmarkTools
using StaticNumbers

using LyceumBase.LyceumCore
import ..LyceumBase: SpecialArrays
using .SpecialArrays
using .SpecialArrays: ncolons, check_nestedarray_parameters, _maybe_unsqueeze, NestedView

# TODO move
macro test_inferred(ex)
    ex = quote
        $Test.@test (($Test.@inferred $ex); true)
    end
    esc(ex)
end

macro test_noalloc(ex)
    ex = quote
        local tmp = $BenchmarkTools.@benchmark $ex samples = 1 evals = 1
        $Test.@test iszero(tmp.allocs)
    end
    esc(ex)
end

testdims(M::Integer, N::Integer = static(0)) = ntuple(i -> 2i, @stat(M + N))

randA(T::Type, M::Integer, N::Integer = static(0)) = rand(T, testdims(M, A))
randA(M::Integer, N::Integer = static(0)) = randA(Float64, M, N)

function randNA(T::Type, M::Integer, N::Integer = static(0))
    dims = testdims(M, N)
    M_dims, N_dims = SpecialArrays.split(dims, static(M))
    nested = Array{Array{T, unstatic(M)}, unstatic(N)}(undef, N_dims...)
    for i in eachindex(nested)
        x = rand!(zeros(T, M_dims...))
        nested[i] = x
    end
    flat = reshape(mapreduce(vec, vcat, nested), dims)
    return nested, flat
end
randNA(M::Integer, N::Integer) = randNA(Float64, M, N)

randN(T::Type, M::Integer, N::Integer) = first(randNA(T, M, N))
randN(M::Integer, N::Integer) = randN(Float64, M, N)

nones(::Val{N}) where {N} = ntuple(_ -> 1, Val(N))

#@testset "functions" begin
#    include("functions.jl")
#end

#@testset "nestedarrays" begin
#    include("nestedarrays/nestedview.jl")
#end

#@testset "elasticarray" begin
#    include("elasticarray.jl")
#end

@testset "slices" begin
    include("slices.jl")
end

end