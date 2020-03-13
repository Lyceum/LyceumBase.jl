struct NestedView{M,T,N,P,F,R} <: AbstractArray{T,N}
    parent::P
    reshaped::R
    function NestedView{M,T,N,P}(parent::P) where {M,T,N,P<:AbsArr}
        check_nestedarray_parameters(Val(M), T, Val(N), P)
        reshaped = _maybe_reshape(IndexStyle(parent), Val(M), parent)
        F = _has_fast_indexing(IndexStyle(parent, reshaped), Val(N))
        new{M,T,N,P,F,typeof(reshaped)}(parent, reshaped)
    end
end

@inline function NestedView{M,T,N,P}(parent::AbsArr) where {M,T,N,P}
    NestedView{M,T,N,P}(convert(P, parent))
end

@inline function NestedView{M,T,N}(parent::AbsArr) where {M,T,N}
    NestedView{M,T,N,typeof(parent)}(parent)
end

@inline function NestedView{M,T}(parent::AbsArr) where {M,T}
    N = ndims(parent) - M
    NestedView{M,T,N}(parent)
end

@inline function NestedView{M}(parent::AbsArr) where {M}
    T = _nested_viewtype(parent, Val(M), Val(ndims(parent) - M))
    NestedView{M,T}(parent)
end

Base.convert(::Type{T}, A::AbsArr) where {T<:NestedView} = A isa T ? A : T(A)


const SlowNestedView{M,T,N,P,R} = NestedView{M,T,N,P,false,R}
const FastNestedView{M,T,N,P,R} = NestedView{M,T,N,P,true,R}

@inline Base.IndexStyle(::Type{<:FastNestedView}) = IndexLinear()
@inline Base.IndexStyle(::Type{<:SlowNestedView}) = IndexCartesian()

# Need to unpack our zero-dimensional views first
@inline _maybe_unsqueeze(x::AbsArr{<:Any,0}) = x[]
@inline _maybe_unsqueeze(x) = x

@inline _maybe_wrap(A::NestedView{M}, B::AbsArr{<:Any,M}) where {M} = B
@inline _maybe_wrap(A::NestedView{M}, B::AbsArr) where {M} = NestedView{M}(B)

@inline _flat_indices(::NestedView{M}, i) where {M} = (ncolons(Val(M))..., i)
@inline function _flat_indices(::NestedView{M,<:Any,N}, I::Tuple{Vararg{Any, N}}) where {M,N}
    (ncolons(Val(M))..., I...)
end


@propagate_inbounds function Base.getindex(A::SlowNestedView{M,<:Any,N}, I::Vararg{Any,N}) where {M,N}
    @inbounds _maybe_wrap(A, view(A.parent, _flat_indices(A, I)...))
end

@propagate_inbounds function Base.getindex(A::FastNestedView{M}, i) where {M}
    @inbounds _maybe_wrap(A, view(A.reshaped, _flat_indices(A, i)...))
end

@inline Base.getindex(A::NestedView{<:Any,<:Any,0}) = A.parent

@inline function Base.getindex(A::SlowNestedView{M}, c::Colon) where {M}
    NestedView{M}(reshape(copy(A.parent), Val(M + 1)))
end

@inline function Base.getindex(A::FastNestedView{M}, c::Colon) where {M}
    NestedView{M}(reshape(copy(A.parent), Val(M + 1)))
end


@propagate_inbounds function Base.view(A::NestedView{M,<:Any,N}, I::Vararg{Any, N}) where {M,N}
    @boundscheck checkbounds(A, I...)
    @inbounds NestedView{M}(view(A.parent, _flat_indices(A, I)...))
end


@propagate_inbounds function Base.setindex!(A::SlowNestedView{M,<:Any,N}, v, I::Vararg{Any,N}) where {M,N}
    @boundscheck checkbounds(A, I...)
    @inbounds setindex!(A.parent, _maybe_unsqueeze(v), _flat_indices(A, I)...)
    v
end

@propagate_inbounds function Base.setindex!(A::FastNestedView{M}, v, i::Int) where {M}
    @boundscheck checkbounds(A, i)
    @inbounds setindex!(A.reshaped, _maybe_unsqueeze(v), _flat_indices(A, i)...)
    v
end


@inline Base.size(A::NestedView) = back_tuple(size(A.parent), Val(ndims(A)))

@inline Base.axes(A::NestedView) = back_tuple(axes(A.parent), Val(ndims(A)))

function Base.reshape(A::NestedView{M}, ::Val{O}) where {M,N,O}
    NestedView{M}(reshape(A.parent, Val(M + O)))
end

@inline function Base.:(==)(A::NestedView{M,<:Any,N}, B::NestedView{M,<:Any,N}) where {M,N}
    A.parent == B.parent
end

@inline Base.parent(A::NestedView) = A.parent

@inline function Base.resize!(A::NestedView{<:Any,<:Any,N}, dims::NTuple{N,Integer}) where {N}
    resize!(A.parent, inner_size(A)..., dims...)
    A
end
@inline Base.resize!(A::NestedView, dims...) = resize!(A, dims)


@inline function Base.similar(A::NestedView, T::Type{<:AbsArr}, dims::Dims)
    NestedView{ndims(T)}(similar(A.parent, eltype(T), inner_size(A)..., dims...))
end


function Base.deepcopy(A::NestedView{M,T,N,P}) where {M,T,N,P}
    NestedView{M,T,N,P}(deepcopy(A.parent))
end

@propagate_inbounds function Base.copyto!(A::NestedView{M,<:Any,N}, B::NestedView{M,<:Any,N}) where {M,N}
    @boundscheck if size(A) != size(B) || inner_size(A) != inner_size(B)
        throw(ArgumentError("Both `size` and `inner_size` of A & B must match"))
    end
    copyto!(A.parent, B.parent)
    A
end

@propagate_inbounds function Base.copyto!(A::NestedView{M,<:Any,N}, B::AbsArr{<:AbsArr{<:Any,M},N}) where {M,N}
    flattento!(A, B)
end


function Base.append!(A::NestedView{M,<:Any,N}, B::NestedView{M,<:Any,N}) where {M,N}
    inner_size(A) == inner_size(B) || throw(DimensionMismatch("inner_size(A) != inner_size(B)"))
    append!(A.parent, B.parent)
    A
end

function Base.append!(A::NestedView{M,<:Any,N}, B::AbsArr{<:AbsArr{<:Any,M},N}) where {M,N}
    inner_size(A) == inner_size(B) || throw(DimensionMismatch("inner_size(A) != inner_size(B)"))
    for b in B
        append!(A.parent, b)
    end
    A
end


const NestedVector{M,T,P,F,R} = NestedView{M,T,1,P,F,R}

@inline function NestedVector{M}(A::AbsArr) where {M}
    T = _nested_viewtype(A, Val(M), Val(1))
    NestedVector{M,T}(A)
end

@inline function NestedVector(A::AbsArr{<:Any, L}) where {L}
    M = L - 1
    NestedVector{M}(A)
end

function NestedVector(A::AbsArr{<:Any, 0})
    throw(ArgumentError("Cannot create a NestedVector from a 0-dimensonal array."))
end

@inline function Base.push!(A::NestedVector, x) where {M}
    inner_size(A) == size(x) || throw(DimensionMismatch("inner_size(A) != size(x)"))
    append!(A.parent, x)
    A
end

@inline function Base.push!(A::NestedVector{0}, x::AbsArr{<:Any,0}) where {M}
    inner_size(A) == size(x) || throw(DimensionMismatch("inner_size(A) != size(x)"))
    append!(A.parent, x[])
    A
end


"""
    innerview(A::AbsArr{M+N}, ::Val{M})
    innerview(A::AbsArr{M+N}, M::Integer)

View array `A` as an `N`-dimensional array of `M`-dimensional arrays by
wrapping it into an [`NestedView`](@ref). See also: [`outerview`](@ref).
"""
function innerview end

@inline innerview(A::AbsArr, ::Val{M}) where {M} = NestedView{M}(A)
@inline innerview(A::AbsArr, M::Integer) = innerview(A, Val(convert(Int, M)))

"""
    outerview(A::AbsArr{M+N}, ::Val{N})
    outerview(A::AbsArr{M+N}, N::Integer)

View array `A` as an `N`-dimensional array of `M`-dimensional arrays by
wrapping it into an [`NestedView`](@ref). See also: [`innerview`](@ref).
"""
function outerview end

@inline function outerview(A::AbsArr{<:Any, L}, ::Val{N}) where {L,N}
    M = L - N
    NestedView{M}(A)
end
@inline outerview(A::AbsArr, M::Integer) = outerview(A, Val(convert(Int, M)))

"""
    flatview(A::NestedView{M,T,N,P}) --> Array{eltype(T),M+N}

Returns the array of dimensionality `M + N` wrapped by `A`. The shape of
the result may be freely changed without breaking the inner consistency of `A`.
"""
@inline flatview(A::NestedView) = parent(A)

@inline inner_eltype(::Type{<:NestedView{<:Any,T}}) where {T} = eltype(T)

@inline inner_ndims(A::Type{<:NestedView{M}}) where {M} = M

@inline function inner_size(A::NestedView{M}) where {M}
    front_tuple(size(A.parent), Val(M))
end

@inline function inner_axes(A::NestedView{M}) where {M}
    front_tuple(axes(A.parent), Val(M))
end


@inline function UnsafeArrays.unsafe_uview(A::NestedView{M}) where {M}
    NestedView{M}(uview(A.parent))
end