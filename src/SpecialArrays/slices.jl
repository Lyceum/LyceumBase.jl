struct Slices{T,N,P,A} <: AbstractArray{T,N}
    parent::P
    alongs::A
    function Slices{T,N,P,A}(parent, alongs) where {T,N,P,A}
        # TODO check type parameters
        N::Int
        new(parent, alongs)
    end
end

@inline function Slices(parent::AbsArr{<:Any,L}, alongs::NTuple{L,StaticBool}) where {L}
    T = viewtype(parent, map(_slice_or_firstindex, alongs, axes(parent)))
    N = unstatic(static_sum(ntuple(dim -> static_not(alongs[dim]), Val(L))))
    Slices{T,N,typeof(parent),typeof(alongs)}(parent, alongs)
end

@inline function Slices(parent::AbsArr{<:Any,L}, alongs::NTuple{M,StaticOrInt}) where {L,M}
    Slices(parent, ntuple(dim -> static_in(dim, alongs), Val(L)))
end

@inline function Slices(parent::AbsArr{<:Any,L}, alongs::Vararg{StaticOrInt,M}) where {L,M}
    Slices(parent, alongs)
end

@inline function Slices(parent::AbsArr{<:Any,L}) where {L}
    Slices(parent, ntuple(_ -> static(false), Val(L)))
end


####
#### Core Array Interface
####

@inline Base.size(S::Slices) = static_filter(map(static_not, S.alongs), size(parent(S)))

@propagate_inbounds function Base.getindex(S::Slices{<:Any,N}, I::Vararg{Any,N}) where {N}
    view(parent(S), parentindices(S, I...)...)
end

#@propagate_inbounds function Base.setindex!(S::Slices{<:Any,N}, v, I::Vararg{Any,N}) where {N}
@propagate_inbounds function Base.setindex!(S::Slices{<:Any,N}, v, I::Vararg{Int,N}) where {N}
    setindex!(parent(S), v, _parentindices(S, I)...)
    #setindex!(parent(S), _maybe_unsqueeze(S, v), _parentindices(S, I)...)
    return S
end

Base.IndexStyle(::Type{<:Slices}) = IndexCartesian()

Base.similar(A::Slices, T::Type, dims::Dims) = similar(parent(A), T, dims)

@inline Base.axes(S::Slices) = static_filter(map(static_not, S.alongs), axes(parent(S)))


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

# TODO dont' extend Base?
@inline function Base.parentindices(S::Slices{<:Any,N}, I::Vararg{Any,N}) where {N}
    _parentindices(S, I)
end

# TODO safe to use Base._ind2sub?
@inline Base.parentindices(S::Slices, i::Int) = _parentindices(S, Base._ind2sub(axes(S), i))
@inline function Base.parentindices(S::Slices{<:Any,1}, i::Int) # method ambiguity
    _parentindices(S, Base._ind2sub(axes(S), i))
end
@inline Base.parentindices(S::Slices, I::CartesianIndex) = _parentindices(S, Tuple(I))

@inline function _parentindices(S::Slices{<:Any,N}, I::NTuple{N,Any}) where {N}
    # TODO can't checkbounds here if parent(S) has non-standard indexing e.g. AxisArrays
    # results in "ArgumentError: unable to check bounds for indices of type Symbol"
    # Validity of returned indices is therefore not guaranteed
    #@boundscheck checkbounds(S, I...)
    _parentindices(axes(parent(S)), map(static_not, S.alongs), I)
end

@inline function _parentindices(parentaxes::NTuple{L,Any}, alongs::Tuple{Vararg{StaticBool,L}}, I::NTuple{N,Any}) where {L,N}
    if unstatic(first(alongs))
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

@inline flatten(S::Slices) = copy(parent(S))

@inline flatview(S::Slices) = parent(S)

@inline innersize(S::Slices) = static_filter(S.alongs, size(parent(S)))

@inline inneraxes(S::Slices) = static_filter(S.alongs, axes(parent(S)))


####
#### Util
####

@inline function _slice_or_firstindex(flag::StaticBool, ax)
    unstatic(flag) ? Base.Slice(ax) : first(ax)
end

@inline _maybe_unsqueeze(S::Slices{<:AbsArr{<:Any,0}}, v::AbsArr{<:Any,0}) = v[]
@inline _maybe_unsqueeze(S::Slices, v) = v

@inline function bool_filter(switches::NTuple{N,Any}, xs::NTuple{N,Any}) where {N}
    (bool_filter1(first(switches), first(xs))..., bool_filter(tail(switches), tail(xs))...)
end
bool_filter(::Tuple{}, ::Tuple{}) = ()
@inline bool_filter1(::StaticTrue, x) = (x, )
@inline bool_filter1(::StaticFalse, ::Any) = ()
# --
