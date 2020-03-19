
module Ju
using JuliennedArrays
using Base: @pure
using JuliennedArrays: not, setindex_unrolled, getindex_unrolled
using InteractiveUtils
using StaticNumbers
const STrue = typeof(static(true))
const SFalse = typeof(static(false))
const SBool = Union{STrue, SFalse}
const StaticOrVal{X} = Union{StaticInteger{X}, StaticReal{X}, StaticNumber{X}, Val{X}}
const TupleN{T,N} = NTuple{N,T}
using Base: tail
using BenchmarkTools
function test()
    N=40
    #xy = ntuple(i -> mod1(i, 4), N)
    xy = ntuple(identity, N)
    x = Tuple(x for x in xy if isodd(x))
    y = Tuple(y for y in xy if !isodd(y))
    @assert length(x) == length(y) == div(N,2)
    al = ntuple(i -> isodd(i) ? True() : False(), N)
    al2 = Tuple(a isa True ? static(true) : static(false) for a in al)
    #A = rand(xy...)
    #S = Slices(A, al...)
    #display(axes(S))
    #display(merge(al2, x, y))
    #display(split(al2, xy))
    #@code_warntype axes(al)

    #display(static_merge(al2, x, (y[1:end-1]..., )))
    #display(static_merge(al2, (x[1:end-1]...,), y))
    display(static_merge(al2, (x..., 2), y))
    #@code_warntype(static_merge(al2, x, y))
    #@btime(static_merge($al2, $x, $y))
    #display(setindex_unrolled(xy, x, al))
    #@code_warntype(setindex_unrolled(xy, x, al))
    #@btime(setindex_unrolled($xy, $x, $al))

    #@code_warntype(static_filter(al2, xy))
    #display(static_filter(al2, xy))
    #@btime(static_filter($al2, $xy))
    #@btime(static_filter2($al2, $xy))
    #@code_warntype getindex_unrolled(xy, al)
    #display(getindex_unrolled(xy, al))
    #@btime(getindex_unrolled($xy, $al))
end

@pure function static_filter2(by::NTuple{N,SBool}, xs::NTuple{N,Any}) where {N}
    Tuple(xs[i] for i=1:N if by[i] === STrue())
    #ntuple(i -> by[i] === STrue() ? xs[i] :)
    #(_filter1(first(by), first(xs))..., static_filter(tail(by), tail(xs))...)
end

#@inline function split(not::SBool, by::Tuple{Vararg{SBool}}, xy::Tuple)
#    (split1(not, first(by), first(xy))..., split(not, tail(by), tail(xy))...)
#end
#function split(not::SBool, ::Tuple{}, xy::Tuple{})
#    ()
#end
#split1(::STrue, x) = (x, )
#split1(::SFalse, x) = ()
split1(::T, ::T, x) where {T<:SBool} = (x, )
split1(::SBool, ::SBool, x) = ()
#split1(::STrue, ::STrue, x) = ()
#split1(::STrue, ::SFalse, x) = (x, )

@inline function merge(by::Tuple{STrue, Vararg{SBool}}, x::Tuple, y::Tuple)
    (first(x), merge(tail(by), tail(x), y)...)
end
@inline function merge(by::Tuple{SFalse, Vararg{SBool}}, x::Tuple, y::Tuple)
    (first(y), merge(tail(by), x, tail(y))...)
end
merge(by::Tuple{}, x::Tuple{}, y::Tuple{}) = ()
merge(by::Tuple{STrue, Vararg{STrue,N}}, x::NTuple{N,Any}, y::Tuple{}) where {N} = x
merge(by::Tuple{SFalse, Vararg{STrue,N}}, x::Tuple{}, y::NTuple{N,Any}) where {N} = y


end
