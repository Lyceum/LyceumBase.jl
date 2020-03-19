const SBool = StaticBool
const STrue = typeof(static(true))
const SFalse = typeof(static(false))

const StaticOrVal{X} = Union{StaticInteger{X}, StaticReal{X}, StaticNumber{X}, Val{X}}

wrapval(::StaticOrVal{X}) where {X} = Val(X)
wrapval(::Type{<:StaticOrVal{X}}) where {X} = Val(X)
@pure wrapval(x) = Val(x)

wrapstatic(::StaticOrVal{X}) where {X} = static(X)
wrapstatic(::Type{<:StaticOrVal{X}}) where {X} = static(X)
@pure wrapstatic(x) = static(x)

unwrap(::StaticOrVal{X}) where {X} = X
unwrap(::Type{<:StaticOrVal{X}}) where {X} = X
unwrap(x) = x


static_not(::SFalse) = static(true)
static_not(::STrue) = static(false)

static_and(::STrue, ::STrue) = static(true)
static_and(::SBool, ::SBool) = static(false)

static_or(::SBool, ::STrue) = static(true)
static_or(::STrue, ::SBool) = static(true)
static_or(::SBool, ::SBool) = static(false)


#static_sum(xs::StaticInteger...) = static_sum(xs)
#static_sum(xs::TupleN{StaticInteger}) = static_sum
#@generated function static_sum(xs::NTuple{N,StaticInteger}) where {N}
#    s = static(0)
#    for i = 1:N
#        s =
#    end
#    return s
#end
#function StaticNumbers.maybe_static(::typeof(static_sum), xs::TupleN{StaticInteger})
#    static(static_sum(xs))
#end

static_in(x::StaticOrInt, itr::TupleN{StaticOrInt}) = _static_in(x, itr)
@pure function _static_in(x::StaticOrInt, itr::TupleN{StaticOrInt})
    for y in itr
        unwrap(y) === unwrap(x) && return static(true)
    end
    return static(false)
end

@inline function static_filter(by::NTuple{N,SBool}, xs::NTuple{N,Any}) where {N}
    (_filter1(first(by), first(xs))..., static_filter(tail(by), tail(xs))...)
end
static_filter(::Tuple{}, ::Tuple{}) = ()
@inline _filter1(::STrue, x) = (x, )
@inline _filter1(::SFalse, x) = ()

function static_merge(by::TupleN{SBool}, x::Tuple, y::Tuple)
    val, x2, y2 = _merge1(first(by), x, y)
    (val, static_merge(tail(by), x2, y2)..., )
end
static_merge(::Tuple{}, ::Tuple{}, ::Tuple{}) = ()
@inline _merge1(::STrue, x::Tuple, y::Tuple) = (first(x), tail(x), y)
@inline _merge1(::SFalse, x::Tuple, y::Tuple) = (first(y), x, tail(y))