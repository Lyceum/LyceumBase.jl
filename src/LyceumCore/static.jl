wrapval(x) = Val(x)
@pure wrapval(v::Val) = v
unwrapval(x) = x
@pure unwrapval(::Val{x}) where {x} = x


const StaticTrue = typeof(static(true))
const StaticFalse = typeof(static(false))

const StaticOr = Union{StaticInteger, StaticNumber, StaticReal, Int, Float64, Bool}

static_not(::StaticTrue) = static(false)
static_not(::StaticFalse) = static(true)


@inline static_in(x::StaticOr, itr::Tuple{Vararg{StaticOr}}) = _static_in(x, itr)

@pure function _static_in(x::StaticOr, itr::Tuple{Vararg{StaticOr}})
    for y in itr
        y == x && return static(true)
    end
    return static(false)
end

@pure function _static_in(x::T, itr::Tuple{Vararg{T}}) where {T<:StaticOr}
    for y in itr
        y === x && return static(true)
    end
    return static(false)
end


@inline static_sum(xs::Tuple{Vararg{StaticInteger}}) = _static_sum(xs)
@inline static_sum(xs::StaticInteger...) = _static_sum(xs)
@pure function _static_sum(xs::Tuple{Vararg{StaticInteger}})
    s = 0
    for x in xs
        s += x
    end
    return static(s)
end

@inline function static_filter(flags::NTuple{N,StaticBool}, xs::Tuple{Vararg{Any,N}}) where {N}
    (static_filter1(first(flags), first(xs))..., static_filter(tail(flags), tail(xs))...)
end
static_filter(::Tuple{}, ::Tuple{}) = ()
@inline static_filter1(::StaticTrue, x) = (x, )
@inline static_filter1(::StaticFalse, ::Any) = ()