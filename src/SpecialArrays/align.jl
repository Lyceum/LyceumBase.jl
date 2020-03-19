struct Indexer{T,N,A<:AbsArr{T,N}} <: AbstractArray{T,N}
    parent::A
end
export Indexer # TODO remove

@forward Indexer.parent Base.size, Base.axes, Base.length
@propagate_inbounds function Base.getindex(A::Indexer{<:Any,N}, I::Vararg{Int,N}) where {N}
    getindex(A.parent, I...)
end
@propagate_inbounds function Base.setindex!(A::Indexer{<:Any,N}, v, I::Vararg{Int,N}) where {N}
    setindex!(A.parent, v, I...)
end
Base.IndexStyle(::Type{<:Indexer}) = IndexCartesian()
function Base.similar(A::Indexer, T::Type, dims::Dims)
    similar(A.parent, T, dims)
end

struct Align{M,N,V,L,S<:AbsSimilarNestedArr{V,M,N},InAx<:NTuple{M,Any},InFirst} <: AbstractArray{V,L}
    slices::S
    inneraxes::InAx
    @inline function Align{M,N,V,L,S,InAx,InFirst}(slices, axes) where {M,N,V,L,S<:AbsSimilarNestedArr{V,M,N},InAx,InFirst}
        _check_dims_match(Val(L), Val(M), Val(N), typeof(InFirst))
        new{M,N,V,L,S,InAx,InFirst}(slices, inneraxes(slices))
    end
end

@inline function Align(slices::AbsSimilarNestedArr{V,M,N}, inneraxes::NTuple{M,Any}; innerfirst::StaticOrBool = static(false)) where {V,M,N}
    Align{M,N,V,M+N,typeof(slices),typeof(inneraxes),unstatic(innerfirst)}(slices, inneraxes)
end

@inline function Align(slices::AbsSimilarNestedArr; kwargs...)
    Align(slices, inneraxes(slices); kwargs...)
end

function _check_dims_match(::Val{L}, ::Val{M}, ::Val{N}, ::Type{InFirst}) where {L,M,N,InFirst}
    if !(L isa Int && M isa Int && N isa Int)
        throw(ArgumentError("Align type parameters L, M, and N must be of type Int"))
    end
    if L !== M + N
        throw(ArgumentError("Align type parameters L, M, and N must satisfy: L === M + N"))
    end
    if !(InFirst <: Bool)
        throw(ArgumentError("Align type parameter InFirst must be of type Bool"))
    end
    return nothing
end


####
#### Core Array Interface
####

const InnerAlign{M,N,V,L,S,InnerAx} = Align{M,N,V,L,S,InnerAx,true}
const OuterAlign{M,N,V,L,S,InnerAx} = Align{M,N,V,L,S,InnerAx,false}

@inline Base.axes(A::InnerAlign) = (A.inneraxes..., axes(A.slices)...)
@inline Base.axes(A::OuterAlign) = (axes(A.slices)..., A.inneraxes...)

@inline Base.size(A::Align) = map(length, axes(A)) # TODO index_length?


@propagate_inbounds function Base.getindex(A::Align{<:Any,<:Any,<:Any,L}, I::Vararg{Any,L}) where {L}
    outI = _outer_indices(A, I)
    _maybe_getindex(A, I, outI, Base.index_dimsum(outI...))
end

@propagate_inbounds function _maybe_getindex(A::Align, I::Tuple, ::Tuple, ::Tuple)
    getindex(Indexer(A), I...)
end
@propagate_inbounds function _maybe_getindex(A::Align, I::Tuple, outI::Tuple, ::Tuple{})
    inI = _inner_indices(A, I)
    _maybe_reshape(A, A.slices[outI...][inI...], I, Base.index_dimsum(I...))
end

@inline function _maybe_reshape(::Align, x::AbsArr{<:Any,N}, ::Tuple, ::NTuple{N,Bool}) where {N}
    x
end
@inline function _maybe_reshape(::Align, x, ::Tuple, ::Tuple{})
    x
end
@inline function _maybe_reshape(A::Align, x::AbsArr, I::Tuple, ::NTuple{N,Bool}) where {N}
    # TODO AxisArrays will fail with to_indices
    reshape(x, Base.index_shape(Base.to_indices(A, I)...))
end


@propagate_inbounds function Base.setindex!(A::Align{<:Any,<:Any,<:Any,L}, v, I::Vararg{Any,L}) where {L}
    outI = _outer_indices(A, I)
    _maybe_setindex!(A, I, v, outI, Base.index_dimsum(outI...))
end

@propagate_inbounds function _maybe_setindex!(A::Align, I::Tuple, v, ::Tuple, ::Tuple)
    setindex!(Indexer(A), v, I...)
end
@propagate_inbounds function _maybe_setindex!(A::Align, I::Tuple, v, outI::Tuple, ::Tuple{})
    inI = _inner_indices(A, I)
    A.slices[outI...][inI...] = v
end


@inline _inner_indices(::OuterAlign{M}, I::Tuple) where {M} = tail(I, static(M))
@inline _outer_indices(::OuterAlign{<:Any,N}, I::Tuple) where {N} = front(I, static(N))

@inline _inner_indices(::InnerAlign{M}, I::Tuple) where {M} = front(I, static(M))
@inline _outer_indices(::InnerAlign{<:Any,N}, I::Tuple) where {N} = tail(I, static(N))


####
#### SpecialArrays functions
####

@inline inneraxes(A::Align) = A.inneraxes