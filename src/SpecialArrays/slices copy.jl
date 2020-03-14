# TODO _inneraxes size etc
# TODO LyceumBase --
#export True
#export False
#
#
#abstract type TypedBool end
#
#struct True <: TypedBool end
#
#struct False <: TypedBool end
#
#
#@inline untyped(::True) = true
#@inline untyped(::False) = false
#
#@inline not(::False) = True()
#@inline not(::True) = False()

#const TypedBool = TypedFlag

#const TupleN{T,N} = NTuple{N,T}
#const AbsArr{T,N} = AbstractArray{T,N}

# --------

struct Slices{T,N,P,A} <: AbstractArray{T,N}
    parent::P
    alongs::A
    function Slices{T,N,P,A}(parent, alongs) where {T,N,P,A}
        # TODO check type parameters
        new(parent, alongs)
    end
end

@inline function Slices(parent::P, alongs::A) where {P,A}
    T = viewtype(parent, map(_slice_or_firstindex, alongs, axes(parent)))
    N = bool_sum(map(not, alongs)...)
    Slices{T,N,P,A}(parent, alongs)
end

@inline Slices(parent, alongs::TypedBool...) = Slices(parent, alongs)

@inline function Slices(parent::AbsArr{<:Any,N}, alongs::Dims) where {N}
    alongs = map(dim -> dim in alongs ? True() : False(), ntuple(identity, Val(N)))
    Slices(parent, alongs)
end

@inline Slices(parent, alongs::Integer...) = Slices(parent, convert(Dims, alongs))

@inline function Slices(parent::AbsArr{<:Any,N}) where {N}
    Slices(parent, ntuple(_ -> False(), Val(N)))
end


####
#### Core Array Interface
####

@inline Base.size(S::Slices) = bool_filter(map(not, S.alongs), size(parent(S)))

@propagate_inbounds function Base.getindex(S::Slices{<:Any,N}, I::Vararg{Any,N}) where {N}
    view(parent(S), parentindices(S, I)...)
end

@propagate_inbounds function Base.setindex!(S::Slices{<:Any,N}, v, I::Vararg{Any,N}) where {N}
    setindex!(parent(S), _maybe_unsqueeze(S, v), parentindices(S, I)...)
    return S
end

Base.IndexStyle(::Type{<:Slices}) = IndexCartesian()

Base.similar(A::Slices, T::Type, dims::Dims) = similar(parent(A), T, dims)

@inline Base.axes(S::Slices) = bool_filter(map(not, S.alongs), axes(parent(S)))


####
#### Misc
####

@inline Base.:(==)(A::Slices, B::Slices) = A.alongs == B.alongs && parent(A) == parent(B)


@inline Base.parent(S::Slices) = S.parent

@inline Base.dataids(S::Slices) = Base.dataids(parent(S))


function Base.copyto!(dest::Slices, src::Slices)
    if size(dest) != size(src) || innersize(dest) != innersize(src)
        throw(ArgumentError("Both `size` and `innersize` of dest & src must match"))
    end
    copyto!(parent(dest), parent(src))
    return dest
end
@inline Base.copyto!(dest::AbsNestedArr, src::Slices) = nest!(dest, parent(src))
@inline function Base.copyto!(dest::Slices, src::AbsNestedArr)
    flatten!(parent(dest), src)
    return dest
end

Base.copy(S::Slices) = typeof(S)(copy(parent(S)), S.alongs)


@inline Base.parentindices(S::Slices, I...) = parentindices(S, I)
# TODO safe to use Base._ind2sub?
@inline Base.parentindices(S::Slices, i::Int) = parentindices(S, Base._ind2sub(axes(S), i))

@inline Base.parentindices(S::Slices, I::CartesianIndex) = parentindices(S, Tuple(I))

@inline function Base.parentindices(S::Slices{<:Any,N}, I::NTuple{N,Any}) where {N}
    # TODO can't checkbounds here if parent(S) has non-standard indexing e.g. AxisArrays
    # results in "ArgumentError: unable to check bounds for indices of type Symbol"
    # Validity of returned indices is therefore not guaranteed
    #@boundscheck checkbounds(S, I...)
    _parentindices(axes(parent(S)), map(not, S.alongs), I)
end

@inline function _parentindices(parentaxes::Tuple, alongs::Tuple, I::Tuple)
    if untyped(first(alongs))
        (first(I), _parentindices(tail(parentaxes), tail(alongs), tail(I))...)
    else
        (Base.Slice(first(parentaxes)), _parentindices(tail(parentaxes), tail(alongs), I)...)
    end
end

_parentindices(parentaxes::Tuple{}, alongs::Tuple{}, I::Tuple{}) = ()


####
#### 3rd Part
####

@inline UnsafeArrays.unsafe_uview(S::Slices) = Slices(uview(parent(S)), S.alongs)


####
#### SpecialArrays functions
####

@inline flatten(S::Slices) = parent(S)

@inline flatview(S::Slices) = flatten(S)

@inline innersize(S::Slices) = bool_filter(S.alongs, size(parent(S)))

@inline inneraxes(S::Slices) = bool_filter(S.alongs, axes(parent(S)))


####
#### Util
####

@inline function _slice_or_firstindex(switch::TypedBool, ax)
    untyped(switch) ? Base.Slice(ax) : first(ax)
end

@inline _maybe_unsqueeze(S::Slices{<:AbsArr{<:Any,0}}, v::AbsArr{<:Any,0}) = v[]
@inline _maybe_unsqueeze(S::Slices, v) = v

@inline bool_sum(xs::TypedBool...) = length(_bool_sum(xs...))
@inline _bool_sum(::True, xs::TypedBool...) = (True(), _bool_sum(xs...)...)
@inline _bool_sum(::False, xs::TypedBool...) = (_bool_sum(xs...)...,)
_bool_sum() = ()

@inline function bool_filter(switches::NTuple{N,Any}, xs::NTuple{N,Any}) where {N}
    (bool_filter1(first(switches), first(xs))..., bool_filter(tail(switches), tail(xs))...)
end
bool_filter(::Tuple{}, ::Tuple{}) = ()
@inline bool_filter1(::True, x) = (x, )
@inline bool_filter1(::False, ::Any) = ()
# --
