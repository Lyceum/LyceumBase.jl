"""
    ElasticArray{T,N,M} <: DenseArray{T,N}

An `ElasticArray` can grow/shrink in its last dimension. `N` is the total number of
dimensions, `M == N - 1` the number of non-resizable dimensions (all but the last dimension).

Constructors:

    ElasticArray(kernel_size::Dims{M}, data::Vector{T}, len::Int)
    ElasticArray{T}(::UndefInitializer, dims::NTuple{N,Integer})
    ElasticArray{T}(::UndefInitializer, dims::Integer...)
    convert(ElasticArray, A::AbstractArray)
"""
struct ElasticArray{T,N,M} <: DenseArray{T,N}
    kernel_size::Dims{M}
    kernel_length::SignedMultiplicativeInverse{Int}
    data::Vector{T}

    function ElasticArray{T,N,M}(kernel_size::Dims{M}, data::Vector{T}) where {T,N,M}
        check_elasticbuffer_parameters(T, Val(N), Val(M))
        kernel_length = SignedMultiplicativeInverse{Int}(prod(kernel_size))
        if rem(length(eachindex(data)), kernel_length) != 0
            throw(ArgumentError("length(data) must be integer multiple of prod(kernel_size)"))
        end
        new{T,N,M}(kernel_size, kernel_length, data)
    end
end

@inline function ElasticArray{T,N}(kernel_size::Dims{M}, data::Vector{T}) where {T,N,M}
    ElasticArray{T,N,M}(kernel_size, data)
end

@inline function ElasticArray{T}(kernel_size::Dims{M}, data::Vector{T}) where {T,M}
    ElasticArray{T,M+1,M}(kernel_size, data)
end

@inline function ElasticArray(kernel_size::Dims{M}, data::Vector{T}) where {T,M}
    ElasticArray{T,M+1,M}(kernel_size, data)
end


@inline function ElasticArray{T,N,M}(::UndefInitializer, dims::Vararg{Integer,N}) where {T,N,M}
    ElasticArray{T,N,M}(undef, dims)
end

@inline function ElasticArray{T,N}(::UndefInitializer, dims::Vararg{Integer,N}) where {T,N}
    ElasticArray{T,N}(undef, dims)
end

@inline function ElasticArray{T}(::UndefInitializer, dims::Integer...) where {T}
    ElasticArray{T}(undef, dims)
end


@inline function ElasticArray{T,N,M}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N,M}
    ElasticArray{T,N,M}(undef, convert(Dims{N}, dims))
end

@inline function ElasticArray{T,N}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N}
    ElasticArray{T,N}(undef, convert(Dims{N}, dims))
end

@inline function ElasticArray{T}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N}
    ElasticArray{T,N}(undef, convert(Dims{N}, dims))
end


@inline function ElasticArray{T,N,M}(::UndefInitializer, dims::Dims{N}) where {T,N,M}
    ElasticArray{T,N,M}(front(dims), Vector{T}(undef, prod(dims)))
end

@inline function ElasticArray{T,N}(::UndefInitializer, dims::Dims{N}) where {T,N}
    ElasticArray{T,N,N-1}(undef, dims)
end

@inline function ElasticArray{T}(::UndefInitializer, dims::Dims{N}) where {T,N}
    ElasticArray{T,N}(undef, dims)
end


@propagate_inbounds function ElasticArray{T,N,M}(A::AbstractArray{<:Any, N}) where {T,N,M}
    ElasticArray{T,N,M}(front(size(A)), copyto!(Vector{T}(undef, length(A)), A))
end

@propagate_inbounds function ElasticArray{T,N}(A::AbstractArray) where {T,N}
    ElasticArray{T,N,N-1}(A)
end

@propagate_inbounds function ElasticArray{T}(A::AbstractArray) where {T}
    ElasticArray{T,ndims(A)}(A)
end

@propagate_inbounds ElasticArray(A::AbstractArray) = ElasticArray{eltype(A)}(A)


Base.convert(::Type{T}, A::AbstractArray) where {T<:ElasticArray} = A isa T ? A : T(A)

@inline function Base.:(==)(A::ElasticArray{<:Any,N,M}, B::ElasticArray{<:Any,N,M}) where {N,M}
    A.kernel_size == B.kernel_size && A.data == B.data
end
@inline Base.:(==)(A::ElasticArray, B::ElasticArray) = false

@inline function Base.size(A::ElasticArray)
    (A.kernel_size..., div(length(eachindex(A.data)), A.kernel_length))
end
@inline function Base.size(A::ElasticArray, d)
    if d == ndims(A)
        div(length(eachindex(A.data)), A.kernel_length)
    else
        A.kernel_size[d]
    end
end

@inline Base.length(A::ElasticArray) = length(A.data)


Base.IndexStyle(::Type{<:ElasticArray}) = IndexLinear()

@propagate_inbounds function Base.getindex(A::ElasticArray, i::Integer)
    @boundscheck checkbounds(A, i)
    @inbounds getindex(A.data, i)
end

@propagate_inbounds function Base.setindex!(A::ElasticArray, x, i::Integer)
    @boundscheck checkbounds(A, i)
    @inbounds setindex!(A.data, x, i)
end


@inline function Base.resize!(A::ElasticArray{T,N}, dims::Vararg{Integer,N}) where {T,N}
    kernel_size, size_lastdim = _split_resize_dims(A, dims)
    resize!(A.data, A.kernel_length.divisor * size_lastdim)
    A
end

@inline function growlastdim!(B::ElasticArray, n::Integer)
    n < 0 && throw(DomainError(n, "n must be positive"))
    resizelastdim!(B, size(B, ndims(B)) + n)
end

@inline function shrinklastdim!(B::ElasticArray, n::Integer)
    n < 0 && throw(DomainError(n, "n must be positive"))
    resizelastdim!(B, size(B, ndims(B)) - n)
end

@inline function resizelastdim!(B::ElasticArray, n::Integer)
    kernel_size = front(size(B))
    dims = (kernel_size..., n)
    resize!(B, dims...)
end

@inline function Base.sizehint!(A::ElasticArray{T,N}, dims::Vararg{Integer,N}) where {T,N}
    kernel_size, size_lastdim = _split_resize_dims(A, dims)
    sizehint!(A.data, A.kernel_length.divisor * size_lastdim)
    A
end


function Base.append!(dest::ElasticArray, src::AbstractArray)
    rem(length(eachindex(src)), dest.kernel_length) != 0 && throw(DimensionMismatch("Can't append, length of source array is incompatible"))
    append!(dest.data, src)
    dest
end

function Base.prepend!(dest::ElasticArray, src::AbstractArray)
    rem(length(eachindex(src)), dest.kernel_length) != 0 && throw(DimensionMismatch("Can't prepend, length of source array is incompatible"))
    prepend!(dest.data, src)
    dest
end


@inline function Base.copyto!(dest::ElasticArray, doffs::Integer, src::AbstractArray, soffs::Integer, N::Integer)
    copyto!(dest.data, doffs, src, soffs, N)
    dest
end
@inline function Base.copyto!(dest::AbstractArray, doffs::Integer, src::ElasticArray, soffs::Integer, N::Integer)
    copyto!(dest, doffs, src.data, soffs, N)
end

@inline Base.copyto!(dest::ElasticArray, src::AbstractArray) = (copyto!(dest.data, src); dest)
@inline Base.copyto!(dest::AbstractArray, src::ElasticArray) = copyto!(dest, src.data)

@inline function Base.copyto!(dest::ElasticArray, doffs::Integer, src::ElasticArray, soffs::Integer, N::Integer)
    copyto!(dest.data, doffs, src.data, soffs, N)
    dest
end
@inline function Base.copyto!(dest::ElasticArray, src::ElasticArray)
    copyto!(dest.data, src.data)
    dest
end


@inline function Base.similar(A::ElasticArray, T::Type, dims::Dims{N}) where {N}
    ElasticArray{T,N}(front(dims), similar(A.data, T, prod(dims)))
end

Broadcast.BroadcastStyle(::Type{<:ElasticArray}) = Broadcast.ArrayStyle{ElasticArray}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{ElasticArray}}, ::Type{ElType}) where ElType
    similar(ElasticArray{ElType}, axes(bc))
end


@inline function Base.unsafe_convert(::Type{Ptr{T}}, A::ElasticArray{T}) where T
    Base.unsafe_convert(Ptr{T}, A.data)
end

@inline Base.pointer(A::ElasticArray, i::Integer) = pointer(A.data, i)

@inline Base.dataids(A::ElasticArray) = Base.dataids(A.data)


@inline function _split_resize_dims(A::ElasticArray, dims::NTuple{N,Integer}) where {N}
    kernel_size, size_lastdim = frontlast(dims)
    kernel_size != A.kernel_size && throw(ArgumentError("Can only resize last dimension of an ElasticArray"))
    kernel_size, size_lastdim
end

@generated function check_elasticbuffer_parameters(::Type{T}, ::Val{N}, ::Val{M}) where {T,N,M}
    !isa(N, Int) && return :(throw(ArgumentError("ElasticArray parameter N must be of type Int")))
    !isa(M, Int) && return :(throw(ArgumentError("ElasticArray parameter M must be of type Int")))
    if M != N - 1
        return :(throw(ArgumentError("ElasticArray{$T,$N,$M} does not satisfy requirement M == N - 1")))
    end
end