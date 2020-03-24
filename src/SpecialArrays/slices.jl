struct Slices{T,N,M,P<:AbsArr,A} <: AbstractArray{T,N}
    parent::P
    alongs::A
    function Slices{T,N,M,P,A}(parent, alongs) where {T,N,M,P<:AbsArr,A}
        check_slices_parameters(T, Val(N), Val(M), P, A)
        new(parent, alongs)
    end
end

@inline function Slices(parent::AbsArr{<:Any,L}, alongs::NTuple{L,SBool}, inaxes::NTuple{M,Any}, outaxes::NTuple{N,Any}) where {L,M,N}
    I = ntuple(i -> first(outaxes[i]), Val(N))
    J = static_merge(alongs, ntuple(i -> Base.Slice(inaxes[i]), Val(M)), I)
    T = viewtype(parent, J...)
    Slices{T,N,M,typeof(parent),typeof(alongs)}(parent, alongs)
end

@inline function Slices(parent::AbsArr{<:Any,L}, alongs::NTuple{L,SBool}) where {L}
    paxes = axes(parent)
    inaxes = static_filter(STrue(), alongs, paxes)
    outaxes = static_filter(SFalse(), alongs, paxes)
    Slices(parent, alongs, inaxes, outaxes)
end


####
#### Core Array Interface
####

@inline Base.size(S::Slices) = static_filter(SFalse(), S.alongs, size(S.parent))

# Cartesian indexing
@propagate_inbounds function Base.getindex(S::Slices{<:Any,N}, I::Vararg{Int,N}) where {N}
    view(S.parent, parentindices(S, I)...)
end

@propagate_inbounds function Base.setindex!(S::Slices{<:Any,N}, v, I::Vararg{Int,N}) where {N}
    setindex!(S.parent, v, parentindices(S, I)...)
    return S
end
@propagate_inbounds function Base.setindex!(S::Slices{<:Any,N,0}, v::AbsArr{<:Any,0}, I::Vararg{Int,N}) where {N}
    setindex!(S.parent, v[], parentindices(S, I)...)
    return S
end

# If `I isa Vararg{SliceIdx,N} && length(Base.index_ndims(I...)) == N`:
# We can just forward the indices to the parent array and drop the corresponding
# entries in `S.alongs` (that is, we can drop `S.alongs[i]` iff
# `S.alongs[i] === static(false) && and Base.index_shape(I[i]) === ())`.
const SliceIdx = Union{Colon, Real, AbstractArray}

@propagate_inbounds function Base.getindex(S::Slices{<:Any,N}, I::Vararg{SliceIdx,N}) where {N}
    J = Base.to_indices(S, I)
    _getindex(S, J, Base.index_ndims(J...))
end

@propagate_inbounds function _getindex(S::Slices{<:Any,N}, J::NTuple{N,SliceIdx}, ::NTuple{N,Bool}) where {N}
    K = parentindices(S, J)
    _maybe_wrap(view(S.parent, K...), reslice(S.alongs, K))
end

@inline _maybe_wrap(A::AbstractArray, alongs::TupleN{SBool}) = Slices(A, alongs)
# A single element, so no need to wrap with a Slices
@inline _maybe_wrap(A::AbstractArray{<:Any,M}, ::NTuple{M,STrue}) where {M} = A

# add/drop non-sliced dimensions (i.e. alongs[dim] == SFalse()) to match J
@inline function reslice(alongs::NTuple{L,SBool}, K::NTuple{L,Any}) where {L}
    (_reslice1(first(alongs), first(K))..., reslice(tail(alongs), tail(K))...)
end
reslice(::Tuple{}, ::Tuple{}) = ()
@inline _reslice1(::STrue, k) = (static(true), ) # keep inner dimension
@inline _reslice1(::SFalse, k) = _reslicefalse(k)
@inline _reslicefalse(::Real) = () # drop this dimension
@inline _reslicefalse(::Colon) = (static(false), ) # keep this dimension
@inline function _reslicefalse(::AbstractArray{<:Any,N}) where {N}
    ntuple(_ -> static(false), Val(N))
end

# Fall back to Cartesian indexing.
@propagate_inbounds _getindex(S::Slices, J::Tuple, ::Tuple) = getindex(Indexer(A), J...)

Base.IndexStyle(::Type{<:Slices}) = IndexCartesian()

## TODO always shove inner dims to front?
#function Base.similar(S::Slices{<:Any,<:Any,M1}, ::Type{<:AbsArr{V,M2}}, dims::Dims) where {M1,V,M2}
#    M1 == M2 || throw(ArgumentError("ndims(T) must equal ndims(S) or unspecified"))
#    _similar(S, V, dims)
#end
#Base.similar(S::Slices, ::Type{<:AbsArr{V}}, dims::Dims) where {V} = _similar(S, V, dims)
#function Base.similar(::Slices, ::Type{<:AbsArr}, ::Dims)
#    throw(ArgumentError("$T has no element type"))
#end
#
#Base.similar(S::Slices, T::Type, dims::Dims) = _similar(S, T, dims)
#
#function _similar(S::Slices{<:Any,<:Any,M}, ::Type{V}, dims::Dims{N}) where {M,V,N}
#    newparent = similar(S.parent, V, innersize(S)..., dims...)
#    newalongs = (ntuple(_ -> static(true), Val(M))..., ntuple(_ -> static(false), Val(N))...)
#    Slices(newparent, newalongs)
#end

@inline Base.axes(S::Slices) = static_filter(SFalse(), S.alongs, axes(S.parent))

#####
##### Misc
#####

@inline Base.:(==)(A::Slices, B::Slices) = A.alongs == B.alongs && A.parent == B.parent

@inline Base.parent(S::Slices) = S.parent

@inline Base.dataids(S::Slices) = Base.dataids(S.parent)


function Base.copyto!(dest::Slices{<:Any,N,M,<:Any,A}, src::Slices{<:Any,N,M,<:Any,A}) where {N,M,A}
    checkbounds(dest, axes(src)...)
    if size(dest.parent) == size(src.parent)
        copyto!(dest.parent, src.parent)
    else
        src2 = unalias(dest, src)
        for I in eachindex(IndexStyle(src2, dest), src2)
            @inbounds dest[I] = src2[I]
        end
    end
    return dest
end

Base.copy(S::Slices) = Slices(copy(S.parent), S.alongs)


#####
##### Extra
#####

function slice(parent::AbsArr, alongs::TupleN{SBool})
    throw(ArgumentError("length(alongs) != ndims(parent)"))
end

@inline function slice(parent::AbsArr{<:Any,L}, ::Tuple{}) where {L}
    Slices(parent, ntuple(_ -> static(false), Val(L)))
end
@inline function slice(parent::AbsArr{<:Any,L}, alongs::NTuple{L,SBool}) where {L}
    Slices(parent, alongs)
end
@inline function slice(parent::AbsArr{<:Any,L}, alongs::Vararg{SBool,L}) where {L}
    Slices(parent, alongs)
end

@inline function slice(parent::AbsArr{<:Any,L}, alongs::TupleN{StaticOrInt}) where {L}
    Slices(parent, ntuple(dim -> static_in(dim, alongs), Val(L)))
end
@inline slice(parent::AbsArr, alongs::Vararg{StaticOrInt}) = slice(parent, alongs)


@inline flatten(S::Slices) = copy(S.parent)
@inline flatview(S::Slices) = S.parent

@inline innersize(S::Slices) = static_filter(STrue(), S.alongs, size(S.parent))
@inline inneraxes(S::Slices) = static_filter(STrue(), S.alongs, axes(S.parent))


function mapslices(f, A::AbstractArray; dims::TupleN{StaticOrInt}, dropdims::StaticOrBool = static(false))
    S = slice(A, dims)
    B = unstatic(dropdims) ? _alloc_dropdims(S, f(first(S))) : _alloc_keepdims(S, f(first(S)))
    @assert axes(S) == axes(B)
    for I in eachindex(S, B)
        _unsafe_copy_inner!(B, f(S[I]), I)
    end
    return B
end

@inline _unsafe_copy_inner!(B::AbsArr, b, I) = @inbounds B[I] = b
@inline function _unsafe_copy_inner!(B::AbsArr, b::AbsArr, I)
    error("Internal error. Please file a bug report.")
end

@inline function _unsafe_copy_inner!(B::Slices, b, I)
    @inbounds Bv = B[I]
    @inbounds Bv .= b
end
@inline function _unsafe_copy_inner!(B::Slices, b::AbsArr, I)
    Bv = @inbounds B[I]
    if length(Bv) != length(b)
        throw(DimensionMismatch("Expected of $(length(Bv)) for f(slice). Got: $(length(b))"))
    end
    copyto!(Bv, b)
    return B
end

function _alloc_dropdims(S::Slices{<:Any,N}, b1::AbstractArray{T,M}) where {N,T,M}
    alongs = (ntuple(_ -> static(true), Val(M))..., ntuple(_ -> static(false), Val(N))...)
    slice(Array{T,M+N}(undef, size(b1)..., size(S)...), alongs)
end
function _alloc_dropdims(S::Slices{<:Any,N}, b1::T) where {N,T}
    Array{T,N}(undef, size(S)...)
end

function _alloc_keepdims(S::Slices{<:Any,N,M}, b1) where {T,N,M}
    innerax = _reshape_axes(axes(b1), Val(M))
    innersz = ntuple(i -> (Base.@_inline_meta; length(innerax[i])), Val(M))
    parentsz = static_merge(S.alongs, innersz, size(S))
    slice(Array{eltype(b1),M+N}(undef, parentsz...), S.alongs)
end

_reshape_axes(axes::Tuple, ::Val{N}) where {N} = Base.rdims(Val(N), axes)
_reshape_axes(axes::NTuple{N,Any}, ::Val{N}) where {N} = axes


#####
##### 3rd Party
#####

@inline function UnsafeArrays.unsafe_uview(S::Slices)
    Slices(UnsafeArrays.unsafe_uview(S.parent), S.alongs)
end


#####
##### Util
#####

@inline _maybe_unsqueeze(S::Slices{<:Any,<:Any,0}, v::AbsArr{<:Any,0}) = v[]
@inline _maybe_unsqueeze(S::Slices, v) = v

@generated function check_slices_parameters(::Type{T}, ::Val{N}, ::Val{M}, ::Type{P}, ::Type{A}) where {T,N,M,P,A}
    if !(N isa Int && M isa Int)
        return :(throw(ArgumentError("Slices paramters N and M must be of type Int")))
    elseif !(A <: NTuple{M+N,SBool})
        return :(throw(ArgumentError("Slices parameter A should be of type NTuple{M+N,$SBool}")))
    elseif N < 0 || M < 0 || ndims(P) != N + M && (M > 0 && sum(unwrap, A.parameters) != M)
        return :(throw(ArgumentError("Dimension mismatch in Slices parameters"))) # got N=$N, M=$M, ndims(P)=$(ndims(P)), and sum(A)=$(sum(A))")))
    else
        return nothing
    end
    error("Internal error. Please file a bug report")
end

@pure function parentindices(S::Slices{<:Any,N,M}, I::NTuple{N,Any}) where {N,M}
    inaxes = inneraxes(S)
    slices = ntuple(i -> (Base.@_inline_meta; Base.Slice(inaxes[i])), Val(M))
    static_merge(S.alongs, slices, I)
end