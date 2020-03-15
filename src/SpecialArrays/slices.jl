struct Slices{T,N,M,P,A} <: AbstractArray{T,N}
    parent::P
    alongs::A
    function Slices(parent::AbsArr{<:Any,L}, alongs::NTuple{L,StaticBool}) where {L}
        T = viewtype(parent, map(_slice_or_firstindex, alongs, axes(parent)))
        N = unstatic(static_sum(ntuple(dim -> static_not(alongs[dim]), Val(L))))
        M = L - N
        new{T,N,M,typeof(parent),typeof(alongs)}(parent, alongs)
    end
end

@inline function Slices(parent::AbsArr{<:Any,L}, alongs::Vararg{StaticBool,L}) where {L}
    Slices(parent, alongs)
end

@inline function Slices(parent::AbsArr{<:Any,L}, alongs::NTuple{M,StaticOrInt}) where {L,M}
    Slices(parent, ntuple(dim -> static_in(dim, alongs), Val(L)))
end

@inline function Slices(parent::AbsArr{<:Any,L}, alongs::Vararg{StaticOrInt,M}) where {L,M}
    Slices(parent, alongs)
end


####
#### Core Array Interface
####

@inline Base.size(S::Slices) = static_filter(map(static_not, S.alongs), size(parent(S)))



const SliceIndexing{N} = Union{CartesianIndex{N}, Colon}

@propagate_inbounds function Base.getindex(S::Slices{<:Any,N}, I::Vararg{Any,N}) where {N}
    _getindex(S, I...)
end

@propagate_inbounds function Base.getindex(S::Slices{<:Any,N}, I::SliceIndexing{N}) where {N}
    _getindex(S, I)
end
# for method ambiguity
@propagate_inbounds function Base.getindex(S::Slices{<:Any,1}, I::SliceIndexing{1}) where {N}
    _getindex(S, I)
end

@propagate_inbounds _getindex(S::Slices, I...) = view(parent(S), parentindices(S, I...)...)
@propagate_inbounds _getindex(S::Slices, ::Colon) = collect(S)


@propagate_inbounds function Base.setindex!(S::Slices{<:Any,N}, v, I::Vararg{Any,N}) where {N}
    _setindex!(S, v, I...)
end

@propagate_inbounds function Base.setindex!(S::Slices{<:Any,N}, v, I::SliceIndexing{N}) where {N}
    _setindex!(S, v, I)
end
# for method ambiguity
@propagate_inbounds function Base.setindex!(S::Slices{<:Any,1}, v, I::SliceIndexing{1}) where {N}
    _setindex!(S, v, I)
end

@propagate_inbounds function _setindex!(S::Slices, v, I...)
    setindex!(parent(S), _maybe_unsqueeze(S, v), parentindices(S, I...)...)
    return S
end
@propagate_inbounds _setindex!(S::Slices, v, ::Colon) = (copyto!(S, v); S)



# TODO support IndexLinear()?
Base.IndexStyle(::Type{<:Slices}) = IndexCartesian()

# TODO always shove inner dims to front?
function Base.similar(S::Slices{<:Any,<:Any,M1}, ::Type{<:AbsArr{V,M2}}, dims::Dims) where {M1,V,M2}
    M1 == M2 || throw(ArgumentError("ndims(T) must equal ndims(S) or unspecified"))
    _similar(S, V, dims)
end
Base.similar(S::Slices, ::Type{<:AbsArr{V}}, dims::Dims) where {V} = _similar(S, V, dims)
function Base.similar(::Slices, ::Type{<:AbsArr}, ::Dims)
    throw(ArgumentError("$T has no element type"))
end

Base.similar(S::Slices, T::Type, dims::Dims) = _similar(S, T, dims)

function _similar(S::Slices{<:Any,<:Any,M}, ::Type{V}, dims::Dims{N}) where {M,V,N}
    newparent = similar(parent(S), V, innersize(S)..., dims...)
    newalongs = (ntuple(_ -> static(true), Val(M))..., ntuple(_ -> static(false), Val(N))...)
    Slices(newparent, newalongs)
end

@inline Base.axes(S::Slices) = static_filter(map(static_not, S.alongs), axes(parent(S)))


####
#### Misc
####

# TODO special case when leading slices (e.g. NestedView) / support prepend!/append!/push!

# TODO
#@inline Base.:(==)(A::Slices, B::Slices) = A.alongs == B.alongs && parent(A) == parent(B)


@inline Base.parent(S::Slices) = S.parent

@inline Base.dataids(S::Slices) = Base.dataids(parent(S))


# TODO copyto with absarr
function Base.copyto!(dest::Slices{<:Any,N,M,<:Any,A}, src::Slices{<:Any,N,M,<:Any,A}) where {N,M,A}
    axes(parent(dest)) == axes(parent(src)) || throwdm(axes(parent(dest)), axes(parent(src)))
    copyto!(parent(dest), parent(src))
    return dest
end

#@inline Base.copyto!(dest::AbsNestedArr, src::Slices) = nest!(dest, parent(src))
#@inline function Base.copyto!(dest::Slices, src::AbsNestedArr)
#    flatten!(parent(dest), src)
#    return dest
#end
#

Base.copy(S::Slices) = Slices(copy(parent(S)), S.alongs)


@inline function parentindices(S::Slices{<:Any,N}, I::Vararg{Any,N}) where {N}
    _parentindices(S, I...)
end

# TODO safe to use Base._ind2sub?
@inline parentindices(S::Slices, i::Int) = _parentindices(S, Base._ind2sub(axes(S), i)...)
# for method ambiguity
@inline function parentindices(S::Slices{<:Any,1}, i::Int) # method ambiguity
    _parentindices(S, Base._ind2sub(axes(S), i)...)
end

@inline function parentindices(S::Slices{<:Any,N}, I::CartesianIndex{N}) where {N}
    _parentindices(S, Tuple(I)...)
end
# for method ambiguity
@inline function parentindices(S::Slices{<:Any,1}, I::CartesianIndex{1})
    _parentindices(S, Tuple(I)...)
end

@inline function _parentindices(S::Slices{<:Any,N}, I::Vararg{Any,N}) where {N}
    #__parentindices(axes(parent(S)), map(static_not, S.alongs), I)
    __parentindices(I, S.alongs...)
end
function _parentindices(::Slices{<:Any,N}, I) where {N}
    throw(ArgumentError("Expected $N1 indices but got: $I"))
end

#@inline function __parentindices(parentaxes::Tuple, alongs::TupleN{StaticBool}, I::Tuple)
#    if unstatic(first(alongs))
#        (first(I), __parentindices(tail(parentaxes), tail(alongs), tail(I))...)
#    else
#        (first(parentaxes), __parentindices(tail(parentaxes), tail(alongs), I)...)
#    end
#end
#__parentindices(parentaxes::Tuple{}, alongs::Tuple{}, I::Tuple{}) = ()

@inline function __parentindices(I::Tuple, alongs1::StaticFalse, alongs::Vararg{StaticBool})
    (first(I), __parentindices(tail(I), alongs...)...)
end
@inline function __parentindices(I::Tuple, alongs1::StaticTrue, alongs::Vararg{StaticBool})
    (Colon(), __parentindices(I, alongs...)...)
end
__parentindices(::Tuple{}) = ()


####
#### SpecialArrays functions
####

@inline flatten(S::Slices) = copy(parent(S))

@inline flatview(S::Slices) = parent(S)

@inline innersize(S::Slices) = static_filter(S.alongs, size(parent(S)))

@inline inneraxes(S::Slices) = static_filter(S.alongs, axes(parent(S)))

slicedims(S::Slices) = slicedims(S.alongs)
function slicedims(alongs::NTuple{N,StaticBool}) where {N}
    static_filter(alongs, ntuple(identity, static(N)))
end


####
#### 3rd Party
####

@inline UnsafeArrays.unsafe_uview(S::Slices) = Slices(uview(parent(S)), S.alongs)


####
#### Util
####

@noinline function throwdm(axdest, axsrc)
    throw(DimensionMismatch("destination axes $axdest are not compatible with source axes $axsrc"))
end

@inline function _slice_or_firstindex(flag::StaticBool, ax)
    unstatic(flag) ? Colon() : first(ax)
end

@inline _maybe_unsqueeze(S::Slices{<:Any,<:Any,0}, v::AbsArr{<:Any,0}) = v[]
@inline _maybe_unsqueeze(S::Slices, v) = v