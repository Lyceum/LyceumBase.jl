#include("src/LyceumBase.jl")
using LyceumBase
using LyceumBase.SpecialArrays
using LyceumBase.LyceumCore

using Random
using UnsafeArrays
using StaticNumbers
using AxisArrays: AxisArray
using Test

function test()
    A = rand(2,3,4,1)
    #B = reshape(mapreduce(vcat, hcat, A), (2,3,4))
    #B = [view(A, :, :, i) for i=1:4]
    B = [A[:, :, i] for i=1:4, _=1:1]
    test_array(B) do
        Slices(deepcopy(A), 1, 2)
    end
end

function make_slices(V::Type, alongs::TupleN{SBool})
    L = length(alongs)
    pdims = ntuple(i -> 1 + i, L)
    sdims = Tuple(i for i=1:L if unstatic(alongs[i]))
    innersz = Tuple(pdims[i] for i in 1:L if unstatic(alongs[i]))
    outersz = Tuple(pdims[i] for i in 1:L if !unstatic(alongs[i]))
    M, N = length(innersz), length(outersz)
    flat = rand!(Array{V,L}(undef, pdims...))
    nested = Vector{Array{V,M}}(undef, prod(outersz))
    i = 0
    mapslices(el -> nested[i+=1] = copy(el), flat, dims=sdims)
    Slices(flat, sdims), nested, flat
end

L = 4
#al = ntuple(i -> static(isodd(i)), L)
al = ntuple(i -> static(i==1), L)
dims = ntuple(i -> 1 + i, L)
I = Tuple(i for i=1:L if (al[i] === STrue()))
flat = rand(dims...)
outax = Tuple(axes(flat, i) for i=1:L if (al[i] === SFalse()))

#@code_warntype static_filter(STrue(), al, dims)
#@btime static_filter(STrue(), $al, $dims)
#@btime static_merge($al, $I, $outax)
#static_merge(al, I, outax)
#@code_warntype static_merge(al, I, outax)
#@btime static_merge($al, $I, $outax)

function foo!(A, B)
    @inbounds for I in eachindex(A, B)
        A[I] = B[I]
    end
    A
end

using EllipsisNotation, UnsafeArrays
function foo2!(A, B)
    @inbounds for i in axes(A, ndims(A))
        A[:, :, :, i] = B[.., i]
    end
    A
end

function make(L)
    al = ntuple(i -> static(isodd(i)), L)
    al = ntuple(i -> static(i==1), L)
    dims = ntuple(i -> 1+i, L)
    A = rand(dims...)
    S = Slices(A, dims)
    B = [Array(el) for el in S]
    return A,S,B
end

function test()
    #@btime foo!($B, $S)
    #@btime foo2!($B, $S)
    @btime foo!($S, $B)
    @btime foo2!($S, $B)
    nothing
end

module Mod
using Test
using LyceumBase: TestUtil
using LyceumBase.LyceumCore

randlike(x::Number) = rand(typeof(x))
randlike(A::AbstractArray{<:Number}) = rand(eltype(A), size(A)...)
randlike(A::AbstractArray{<:Number,0}) = rand!(zeros(eltype(A)))
function randlike(A::AbsArr{<:AbsArr{V,M},N}) where {V,M,N}
    B = Array{Array{V,M},N}(undef, size(A)...)
    for I in eachindex(B)
        B[I] = randlike(A[I])
    end
    return B
end
function randlike!(A::AbsArr{<:AbsArr})
    for I in eachindex(A)
        copyto!(A[I], randlike(A[I]))
    end
    return A
end
function randlike!(A::AbsArr{<:Number})
    for I in eachindex(A)
        A[I] = randlike(A[I])
    end
    return A
end

macro test_index(A, I)
    ex = quote
        if $I isa $Base.Tuple
            $TestUtil.@test_inferred $A[$I...]
            if $Base.index_dimsum($I...) == ()
                $Test.@test $Base.typeof($A[$I...]) == $Base.eltype($A)
            else
                $Test.@test $Base.size($A[$I...]) == $Base.map($Base.length, $Base.index_shape($Base.to_indices($A, $I)))
                $Test.@test $Base.eltype($A[$I...]) == $Base.eltype($A)
            end
            local x = $randlike($A[$I...])
            $TestUtil.@test_inferred $Base.setindex!($A, x, $I...)
            $Test.@test $Base.setindex!($A, x, $I...) === $A
            $Test.@test $A[$I...] == x
        else
            $TestUtil.@test_inferred $A[$I]
            if $Base.index_dimsum($I) == ()
                $Test.@test $Base.typeof($A[$I]) == $Base.eltype($A)
            else
                $Test.@test $Base.size($A[$I]) == $Base.map($Base.length, $Base.index_shape($Base.to_indices($A, ($I, ))))
                $Test.@test $Base.eltype($A[$I]) == $Base.eltype($A)
            end
            local x = $randlike($A[$I])
            $TestUtil.@test_inferred $Base.setindex!($A, x, $I)
            $Test.@test $Base.setindex!($A, x, $I) === $A
            $Test.@test $A[$I] == x

        end
    end
    esc(ex)
end

macro test_index(A, B, I)
    ex = quote
        @test_index $A $I
        if $I isa $Base.Tuple
            $Test.@test $A[$I...] == $B[$I...]
        else
            $Test.@test $A[$I] == $B[$I]
        end
    end
    esc(ex)
end



end


#module Bam
#using ..Mod
#using ..SpecialArrays
using MacroTools

dims=(2,3,4,1)
#dims = dims .* 10
A = rand(dims...)
S = Slices(A, (1, ))
B = [Array(el) for el in S]
#B2 = [Array(el) for el in S]
B1 = [view(rand!(copy(A)), :, i, j) for i=1:dims[2], j=1:dims[3]]
B2 = [view(rand!(copy(A)), :, i, j) for i=1:dims[2], j=1:dims[3]]