module SpecialArraysTest

using Test
using Random

using AxisArrays: AxisArrays
using UnsafeArrays
using BenchmarkTools
using StaticNumbers

using ..LyceumBase.LyceumCore
using ..LyceumBase.SpecialArrays
using ..LyceumBase.SpecialArrays: _maybe_unsqueeze
using ..LyceumBase.TestUtil: @test_inferred, @test_noalloc

include("testutil.jl")

const DEFAULT_ELTYPE = Float64


nones(N::Integer) = ntuple(_ -> 1, Val(unstatic(N)))

testdims(L::Integer) = ntuple(i -> 2i, Val(unstatic(L)))

randA(T::Type, L::Integer) = rand(T, testdims(L)...)
randA(L::Integer) = randA(DEFAULT_ELTYPE, L)

function randN(T::Type, innersz::Dims{M}, outersz::Dims{N}) where {M,N}
    dims = testdims(M + N)
    nested = Array{Array{T,M},N}(undef, outersz...)
    for i in eachindex(nested)
        # rand!(zeros(...)) because when M == 0 rand(()) fails
        nested[i] = rand!(zeros(T, innersz...))
    end
    return nested
end
randAN(M::Integer, N::Integer) = randAN(DEFAULT_ELTYPE, M, N)

randN(T::Type, M::Integer, N::Integer) = last(randAN(T, M, N))
randN(M::Integer, N::Integer) = randN(DEFAULT_ELTYPE, M, N)


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