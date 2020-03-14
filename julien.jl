include("src/LyceumBase.jl")
using Random
using UnsafeArrays
using StaticNumbers

#using .LyceumBase.SpecialArrays
#using .LyceumBase.SpecialArrays: NestedView

module Mod

using ..LyceumBase.LyceumCore
using Random
using UnsafeArrays
using StaticNumbers
using ..LyceumBase.SpecialArrays

function dam(parent::AbsArr{T,N}, alongs::Vararg{StaticInteger,M}) where {T,N,M}
    #alongs = map(dim -> dim in alongs ? True() : False(), ntuple(identity, Val(N)))
    Slices(parent, alongs)
end

function dam(parent::AbsArr{T,N}, alongs::Vararg{StaticBool,M}) where {T,N,M}
    Slices(parent, alongs)
end
using ..InteractiveUtils
using BenchmarkTools
function test()
    A = rand(2,3,4,5)
    #al = (1, 3, 5)
    #A = rand(2,3,4)
    al = (1,2,3)
    #al = static.(al)
    #Slices(A, al)
    #@code_llvm Slices(A, al)

    #@code_warntype Slices(A, (static(1), static(2)))
    #@code_llvm Slices(A, al)
    #@code_lowered Slices(A, (static(1), static(2)))
    #@code_warntype Slices(A, (1, 2))

    #@code_warntype g(A); g(A)
    #@code_warntype f(A); f(A)
    #S = f(A)
    S = Slices(A, al)
    @info "NDIMS" ndims(S) ndims(A) length(al)
    I = ntuple(_ -> 1, ndims(S))
    @info I
    x = Array(first(S))
    @btime setindex!($S, $x, 1)
    #@btime setindex!($(parent(S)), $x, $I...)
    #@btime setindex!($A, $x, :, 1, 1, 1)
    #@btime setindex!($A, $x, :, :, 1)
    #@info "I" parentindices(S, I...)
    #@code_warntype parentindices(S, I)
    #@code_llvm parentindices(S, I...)
    @info "YO" innersize(S) inneraxes(S)

    return S
end

f(A) = Slices(A, (static(1), static(2), static(5)))
g(A) = Slices(A, (1, 2, 5))
using Base: @pure

@inline function foo(A::AbsArr{T,N}, al::Dims{M}) where {T,N,M}
    #ntuple(dim -> foo(dim, al), Val(N))
    ntuple(dim -> foo(dim, al), Val(N))
end

@pure foo(d::Int, al::Dims{M}) where {M} = static(d in al)

end