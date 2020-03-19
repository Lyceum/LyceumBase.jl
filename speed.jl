module Mod
using StaticNumbers
using InteractiveUtils
const SBool = StaticBool
const STrue = typeof(static(true))
const SFalse = typeof(static(false))

#@inline function _matched_sum(::Type{T}, ::T, alongs::SBool...) where {T<:SBool}
#    (true, _matched_sum(T, alongs...)...)
#end
#@inline function _matched_sum(::Type{<:SBool}, ::SBool, alongs::SBool...)
#    (_matched_sum(T, alongs...)..., )
#end
#_matched_sum(::Type) = ()
#
#@inline _matched_sum2(::STrue, alongs...) = (static(true), _matched_sum2(alongs...)...)
#@inline _matched_sum2(::SFalse, alongs...) = (_matched_sum2(alongs...)..., )
#_matched_sum2() = ()
#const TupleN{T,N} = NTuple{N,T}
##Base.@pure matched_sum(alongs::TupleN{SBool}) = _matched_sum2(alongs...)
#
using Base: tail
#
#@inline _sel(by::STrue, x, y) = first(x), tail(x), y
#@inline _sel(by::SFalse, x, y) = first(y), x, tail(y)
#@inline function _select3(x::Tuple, y::Tuple, by::SBool, bys::SBool...)
#    s, x, y = _sel(by, x, y)
#    (s, _select(x, y, bys...))
#end
#@inline _select3(::Tuple{}, ::Tuple{}) = ()
#
#@inline function _select(by::Tuple{STrue, Vararg{SBool}}, x::Tuple, y::Tuple)
#    (first(x), _select(tail(by), tail(x), y)...)
#end
#@inline function _select(by::Tuple{SFalse, Vararg{SBool}}, x::Tuple, y::Tuple)
#    (first(y), _select(tail(by), x, tail(y))...)
#end
#_select(by::Tuple{}, x::Tuple{}, y::Tuple{}) = ()
#_select(by::Tuple{STrue, Vararg{STrue,N}}, x::NTuple{N,Any}, y::Tuple{}) where {N} = x
#_select(by::Tuple{SFalse, Vararg{STrue,N}}, x::Tuple{}, y::NTuple{N,Any}) where {N} = y
const TupleN{T,N} = NTuple{N,T}
@generated function static_merge(x::NTuple{M,Any}, y::NTuple{N,Any}, ::Bys) where {M,N,Bys<:TupleN{SBool}}
    xy = Expr(:tuple)
    i = j = 1
    for By in Bys.parameters
        #if By.parameters[k] === STrue
        if By === STrue
            #i > M ? return :(error("boo")) : push!(xy.args, :(x[$i]))
            i <= M ? push!(xy.args, :(x[$i])) : return :(error("sum"))
            i += 1
        else
            j > N && return :(error("doo"))
            push!(xy.args, :(y[$j]))
            j += 1
        end
    end
    quote
        Base.@_inline_meta
        $xy
    end
end

x=(1,2,3)
y=(3,5,7)
al=ntuple(i->static(isodd(i)), length(x) + length(y))
#al = static.((true,false,true,false,true,true))

using BenchmarkTools
function foo(al, x, y)
    @btime static_merge($x, $y, $al)
end

#@code_warntype _matched_sum(STrue, al...)
#@code_warntype _matched_sum2(al...)
#@code_warntype _select(al, x, y)
foo(al,x,y)
#@code_warntype _select2(al, x, y)
#@code_warntype matched_sum(al)
#display(matched_sum(al))
end