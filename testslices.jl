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

using Test
using LyceumBase: TestUtil
using LyceumBase.TestUtil
using LyceumBase.LyceumCore

const TEST_ALONGS = [
    (static(true), ),
    (static(false), ),

    (static(true), static(true)),
    (static(true), static(false)),
    (static(false), static(true)),
    (static(false), static(false)),
]

#al = TEST_ALONGS[4]
#al = (SFalse(),)
al = (SFalse(), STrue())
al = reverse(al)
#al = (STrue(), STrue())
#al = (STrue(),)
#al = (STrue(),)

testdims(L::Integer) = ntuple(i -> 2i, Val(unstatic(L)))
slicedims(al::TupleN{SBool}) = Tuple(i for i=1:length(al) if unstatic(al[i]))
V=Float64

function test_SNF()
    L = length(al)
    pdims = testdims(L)
    sdims = slicedims(al)
    innersz = Tuple(pdims[i] for i in 1:L if unstatic(al[i]))
    outersz = Tuple(pdims[i] for i in 1:L if !unstatic(al[i]))
    M, N = length(innersz), length(outersz)
    flat = rand!(Array{V,L}(undef, pdims...))
    nested = Array{Array{V,M},N}(undef, outersz...)
    i = 0
    Base.mapslices(flat, dims=sdims) do el
        i += 1
        nested[i] = zeros(V, innersz...)
        nested[i] .= el
        el
    end
    Slices(flat, al), nested, flat
end
S,nested,flat = test_SNF()

_maybe_squeeze(x::Number) = (y = zeros(typeof(x)); y .= x)
_maybe_squeeze(x) = x
for f in (
        identity,
        el -> sum(el),
        el -> el isa AbsArr ? reshape(el, reverse(size(el))) : el,
        el -> el isa AbsArr ? reshape(el, Val(1)) : el,
    )
    continue
    @info "F ---- F"

    B1 = Base.mapslices(f, flat, dims=slicedims(al))
    B2 = SpecialArrays.mapslices(f, flat, dims=slicedims(al), dropdims=false)
    @assert B1 == parent(B2)

    @info "DORPDIM"
    B3 = map(el -> f(el), slice(flat, al))
    B4 = SpecialArrays.mapslices(f, flat, dims=slicedims(al), dropdims=true)
    @assert B3 == B4
end
f = el -> sum(el)
f = el -> el isa AbsArr ? reshape(el, reverse(size(el))) : el
f = el -> el isa AbsArr ? reshape(el, Val(1)) : el
S = slice(flat, al)

flat = rand(100,100)

g1(el) = reshape(el, reverse(size(el)))
@code_warntype SpecialArrays.mapslices(g1, flat, dims=al, dropdims=static(true))

@btime SpecialArrays.mapslices(sum, $flat, dims=$al, dropdims=static(false))
@btime Base.mapslices(sum, $flat, dims=$(slicedims(al)))

#@btime SpecialArrays.mapslices(sum, $flat, dims=$al, dropdims=static(true))
#@btime map(sum, slice($flat, $al))