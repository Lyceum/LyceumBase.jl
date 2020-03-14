@generated function check_nestedarray_parameters(::Val{M}, ::Type{P}) where {M,P}
    L = ndims(P)
    M isa StaticOrInt || return :(throw(ArgumentError("NestedView parameter M must be of type $(StaticOrInt)")))
    0 <= M <= L || return :(throw(ArgumentError("NestedView parameter M must be in range [0, ndims(parent)]")))
    nothing
end

@generated function check_dims_match(::Val{L}, ::Val{M}, ::Val{N}) where {L,M,N}
    !isa(M, Int) && return :(throw(ArgumentError("NestedView parameter M must be of type Int")))
    !isa(N, Int) && return :(throw(ArgumentError("NestedView parameter N must be of type Int")))
    if !isa(L, Int)
        return :(throw(ArgumentError("Type mismatch in NestedView parameter P. ndims(P) must return an Int")))
    end
    M < 0 && return :(throw(DomainError(M, "NestedViews parameter M cannot be negative")))
    N < 0 && return :(throw(DomainError(N, "NestedViews parameter N cannot be negative")))
    if L < 0
        return :(throw(DomainError(L, "NestedView parameter P cannot have a negative number of dimensions")))
    end
    if M + N != L
        return :(throw(ArgumentError(
            "Dimension mismatch in NestedViews paramaters. Got M = $M, N = $N, and ndims(P) = $L"
        )))
    end
    nothing
end

@inline function _nested_viewtype(A::AbstractArray{<:Any, L}, ::Val{M}, ::Val{N}) where {L,M,N}
    check_dims_match(Val(L), Val(M), Val(N))
    ax = axes(A)
    inner = ntuple(_ -> Colon(), Val(M))
    outer = ntuple(i -> first(ax[M + i]), Val(N))
    viewtype(A, inner..., outer...)
end

@propagate_inbounds function flatindices(::Val{M}, I::NTuple{N,Any}) where {M,N}
    isa(M, Int) && isa(N, Int) || throw(ArgumentError("M must be an Int"))
    (ntuple(_ -> Colon(), Val(M))..., I...)
end

@inline _has_fast_indexing(::IndexLinear, ::Val) = true
@inline _has_fast_indexing(::IndexLinear, ::Val{1}) = true
@inline _has_fast_indexing(::IndexStyle, ::Val{1}) = true
@inline _has_fast_indexing(::IndexStyle, ::Val) = false

@inline function _maybe_reshape(::IndexLinear, ::Val{M}, parent::AbsArr) where {M}
    reshape(parent, Val(M + 1))
end
@inline _maybe_reshape(::IndexStyle, ::Val, parent::AbsArr) = parent
