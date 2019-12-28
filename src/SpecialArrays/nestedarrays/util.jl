@generated function check_nestedarray_parameters(::Val{M}, ::Type{T}, ::Val{N}, ::Type{P}) where {M,T,N,P}
    L = ndims(P)
    !isa(M, Int) && return :(throw(ArgumentError("NestedView parameter M must be of type Int")))
    !isa(N, Int) && return :(throw(ArgumentError("NestedView parameter N must be of type Int")))
    !isa(L, Int) && return :(throw(ArgumentError("NestedView parameter L must be of type Int")))
    M < 0 && return :(throw(DomainError($M, "NestedView parameter M cannot be negative")))
    N < 0 && return :(throw(DomainError($N, "NestedView parameter N cannot be negative")))
    L < 0 && return :(throw(DomainError($L, "NestedView parameter L cannot be negative")))
    eltype(T) != eltype(P) && :(throw(ArgumentError("eltype mistmatch in NestedView parameters T and L.")))
    if M + N != L
        return :(throw(ArgumentError(
            "Dimension mismatch in NestedViews paramaters. Got M = $M, N = $N, and ndims(P) = $(ndims(P))"
        )))
    end
    U = eltype(P)
    if M == 0 && T != U
        :(throw(ArgumentError("Type mismatch in NestedView parameters. Got T = $T and U = $U")))
    elseif M > 0 && !(T <: AbstractArray{U,M} && P <: AbstractArray{U,L})
        return :(throw(ArgumentError("Type mismatch in NestedView parameters. Got T = $T and P = $P")))
    end
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

@propagate_inbounds function flatindices(::Val{M}, I::NTuple{N,Any}) where {M,N}
    isa(M, Int) && isa(N, Int) || throw(ArgumentError("M must be an Int"))
    (ntuple(_ -> Colon(), Val(M))..., I...)
end

function _try_infer_viewtype(::Type{A}, ::Val{M}, ::Val{N}) where {A,M,N}
    signature = Tuple{A, ntuple(_ -> Colon, Val(M))..., ntuple(_ -> Int, Val(N))...}
    T = Core.Compiler.return_type(view, signature)
    isconcretetype(T) ? T : nothing
end

# For a::A, M, and N where A <: Array{U,L} and L = M + N
# compute T = view(a, ncolons(Val(M))..., I::NTuple{Int, N}...)
# 1) First try T = view(a::A, Val(M), Val(N))
# 2) If that fails and all the axes of A have non-zero length,
#    try T = view(a, ncolons(Val(M))..., I::NTuple{Int, N}...))
# 3) Else, try to infer the return type, which may give the incorrect result.


## Base.Array
function viewtype(::Type{A}, ::Val{M}, ::Val{N}) where {A<:Array,M,N}
    L = ndims(A)
    check_dims_match(Val(L), Val(M), Val(N))
    I_M = ntuple(_ -> Base.Slice{Base.OneTo{Int}}, Val(M))
    I_N = ntuple(_ -> Int, Val(N))
    I = Tuple{I_M..., I_N...}
    SubArray{eltype(A),M,A,I,true}
end

# UnsafeArrays.UnsafeArray
@inline viewtype(::Type{A}, ::Val{M}, ::Val) where {A<:UnsafeArray,M} = UnsafeArray{eltype(A),M}

# Fall back
@inline function viewtype(::Type{A}, ::Val{M}, ::Val{N}) where {A,M,N}
    check_dims_match(Val(ndims(A)), Val(M), Val(N))
    _try_infer_viewtype(A, Val(M), Val(N))
end

function unsafe_viewtype(A::AbstractArray, ::Val{M}, ::Val{N}) where {M,N}
    check_dims_match(Val(ndims(typeof(A))), Val(M), Val(N))
    I = @inbounds (ncolons(Val(M))..., back_tuple(axes(A), Val(N))...)
    @inbounds typeof(view(A, I...))
end

_viewtype(::AbstractArray, ::Val, ::Val, T) = T
@propagate_inbounds function _viewtype(A::AbstractArray, ::Val{M}, ::Val{N}, ::Nothing) where {M,N}
    @boundscheck begin
        inbounds = checkbounds(Bool, A, flatindices(Val(M), axes(A))...)
        !inbounds && throw(ArgumentError(
            """
            Unable to infer the type of T, where T = viewtype(::Type{$(typeof(A))}, ::Val{$M}, ::Val{$N}).
            Only other option is to perform a view into A, but one of the axes has zero length.
            Try passing in an AbstractArray with non-zero length axes or implement:
                viewtype(::Type{$(typeof(A))}, ::Val{M}, ::Val{N}) -> T
            """
        ))
    end
    @inbounds unsafe_viewtype(A, Val(M), Val(N))
end

# Main entry point
@inline function viewtype(A::AbstractArray, ::Val{M}, ::Val{N}) where {M,N}
    check_dims_match(Val(ndims(A)), Val(M), Val(N))
    _viewtype(A, Val(M), Val(N), viewtype(typeof(A), Val(M), Val(N)))
end

@inline _has_fast_indexing(::IndexLinear, ::Val) = true
@inline _has_fast_indexing(::IndexLinear, ::Val{1}) = true
@inline _has_fast_indexing(::IndexStyle, ::Val{1}) = true
@inline _has_fast_indexing(::IndexStyle, ::Val) = false

@inline function _maybe_reshape(::IndexLinear, ::Val{M}, parent::AbsArr) where {M}
    reshape(parent, Val(add(Val(M), Val(1))))
end
@inline _maybe_reshape(::IndexStyle, ::Val, parent::AbsArr) = parent
