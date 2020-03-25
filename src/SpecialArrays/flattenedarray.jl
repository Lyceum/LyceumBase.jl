struct FlattenedArray{V,L,M,S,InAx} <: AbstractArray{V,L}
    slices::S
    inneraxes::InAx
    @inline function FlattenedArray{V,L,M,S,InAx}(slices, inneraxes) where {V,L,M,S,InAx}
        _check_flatview_parameters(V, Val(L), Val(M), S, InAx)
        new{V,L,M,S,InAx}(slices, inneraxes)
    end
end

@inline function FlattenedArray(slices::AbsSimilarNestedArr{V,M,N}, inneraxes::NTuple{M,Any}) where {V,M,N}
    FlattenedArray{V,M+N,M,typeof(slices),typeof(inneraxes)}(slices, inneraxes)
end

@inline function FlattenedArray(slices::AbsSimilarNestedArr)
    FlattenedArray(slices, inneraxes(slices))
end

function _check_flatview_parameters(::Type{V}, ::Val{L}, ::Val{M}, ::Type{S}, ::Type{InAx}) where {V,L,M,S,InAx}
    if !(L isa Int && M isa Int)
        throw(ArgumentError("FlattenedArray type parameters L and M must be of type Int"))
    end
    #if L !== M + N
    #    throw(ArgumentError("FlattenedArray type parameters L, M, and N must satisfy: L === M + N"))
    #end
    return nothing
end


####
#### Core Array Interface
####

@inline Base.axes(A::FlattenedArray) = (A.inneraxes..., axes(A.slices)...)

@inline Base.size(A::FlattenedArray) = map(Base.unsafe_length, axes(A))

# standard Cartesian indexing
@propagate_inbounds function Base.getindex(A::FlattenedArray{<:Any,L}, I::Vararg{Int,L}) where {L}
    A.slices[_outer_indices(A, I)...][_inner_indices(A, I)...]
end

@propagate_inbounds function Base.setindex!(A::FlattenedArray{<:Any,L}, v, I::Vararg{Int,L}) where {L}
    A.slices[_outer_indices(A, I)...][_inner_indices(A, I)...] = v
    return A
end

#@propagate_inbounds function Base.getindex(A::FlattenedArray{<:Any,<:Any,<:Any,L}, I::Vararg{Any,L}) where {L}
#    outI = _outer_indices(A, I)
#    _maybe_getindex(A, I, outI, Base.index_dimsum(outI...))
#end
#
#@propagate_inbounds function _maybe_getindex(A::FlattenedArray, I::Tuple, ::Tuple, ::Tuple)
#    getindex(CartesianIndexer(A), I...)
#end
#@propagate_inbounds function _maybe_getindex(A::FlattenedArray, I::Tuple, outI::Tuple, ::Tuple{})
#    inI = _inner_indices(A, I)
#    _maybe_reshape(A, A.slices[outI...][inI...], I, Base.index_dimsum(I...))
#end
#
#@inline function _maybe_reshape(::FlattenedArray, x::AbsArr{<:Any,N}, ::Tuple, ::NTuple{N,Bool}) where {N}
#    x
#end
#@inline function _maybe_reshape(::FlattenedArray, x, ::Tuple, ::Tuple{})
#    x
#end
#@inline function _maybe_reshape(A::FlattenedArray, x::AbsArr, I::Tuple, ::NTuple{N,Bool}) where {N}
#    # TODO AxisArrays will fail with to_indices
#    reshape(x, Base.index_shape(Base.to_indices(A, I)...))
#end

#@propagate_inbounds function Base.setindex!(A::FlattenedArray{<:Any,<:Any,<:Any,L}, v, I::Vararg{Any,L}) where {L}
#    outI = _outer_indices(A, I)
#    _maybe_setindex!(A, I, v, outI, Base.index_dimsum(outI...))
#end
#
#@propagate_inbounds function _maybe_setindex!(A::FlattenedArray, I::Tuple, v, ::Tuple, ::Tuple)
#    setindex!(CartesianIndexer(A), v, I...)
#end
#@propagate_inbounds function _maybe_setindex!(A::FlattenedArray, I::Tuple, v, outI::Tuple, ::Tuple{})
#    inI = _inner_indices(A, I)
#    A.slices[outI...][inI...] = v
#end

@inline function _inner_indices(::FlattenedArray{<:Any,L,M}, I::NTuple{L,Any}) where {L,M}
    front(I, Val(M))
end
@inline function _outer_indices(::FlattenedArray{<:Any,L,M}, I::NTuple{L,Any}) where {L,M}
    tail(I, Val(L - M))
end

####
#### SpecialArrays functions
####

@inline inneraxes(A::FlattenedArray) = A.inneraxes