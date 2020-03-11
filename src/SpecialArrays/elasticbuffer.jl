"""
    ElasticBuffer{T,N,M} <: DenseArray{T,N}

An `ElasticBuffer` can grow/shrink in its last dimension. `N` is the total number of
dimensions, `M == N - 1` the number of non-resizable dimensions (all but the last dimension).

Constructors:

    ElasticBuffer(kernel_size::Dims{M}, data::Vector{T}, len::Int)
    ElasticBuffer{T}(::UndefInitializer, dims::NTuple{N,Integer})
    ElasticBuffer{T}(::UndefInitializer, dims::Integer...)
    convert(ElasticBuffer, A::AbstractArray)
"""
struct ElasticBuffer{T,N,M} <: DenseArray{T,N}
    kernel_size::Dims{M}
    kernel_length::SignedMultiplicativeInverse{Int}
    data::Vector{T}

    function ElasticBuffer{T,N,M}(kernel_size::Dims{M}, data::Vector{T}) where {T,N,M}
        check_elasticbuffer_parameters(T, Val(N), Val(M))
        kernel_length = SignedMultiplicativeInverse{Int}(prod(kernel_size))
        if rem(length(eachindex(data)), kernel_length) != 0
            throw(ArgumentError("length(data) must be integer multiple of prod(kernel_size)"))
        end
        new{T,N,M}(kernel_size, kernel_length, data)
    end
end

@inline function ElasticBuffer{T,N}(kernel_size::Dims{M}, data::Vector{T}) where {T,N,M}
    ElasticBuffer{T,N,M}(kernel_size, data)
end

@inline function ElasticBuffer{T}(kernel_size::Dims{M}, data::Vector{T}) where {T,M}
    ElasticBuffer{T,M+1,M}(kernel_size, data)
end

@inline function ElasticBuffer(kernel_size::Dims{M}, data::Vector{T}) where {T,M}
    ElasticBuffer{T,M+1,M}(kernel_size, data)
end


@inline function ElasticBuffer{T,N,M}(::UndefInitializer, dims::Vararg{Integer,N}) where {T,N,M}
    ElasticBuffer{T,N,M}(undef, dims)
end

@inline function ElasticBuffer{T,N}(::UndefInitializer, dims::Vararg{Integer,N}) where {T,N}
    ElasticBuffer{T,N}(undef, dims)
end

@inline function ElasticBuffer{T}(::UndefInitializer, dims::Integer...) where {T}
    ElasticBuffer{T}(undef, dims)
end


@inline function ElasticBuffer{T,N,M}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N,M}
    ElasticBuffer{T,N,M}(undef, convert(Dims{N}, dims))
end

@inline function ElasticBuffer{T,N}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N}
    ElasticBuffer{T,N}(undef, convert(Dims{N}, dims))
end

@inline function ElasticBuffer{T}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N}
    ElasticBuffer{T,N}(undef, convert(Dims{N}, dims))
end


@inline function ElasticBuffer{T,N,M}(::UndefInitializer, dims::Dims{N}) where {T,N,M}
    ElasticBuffer{T,N,M}(front(dims), Vector{T}(undef, prod(dims)))
end

@inline function ElasticBuffer{T,N}(::UndefInitializer, dims::Dims{N}) where {T,N}
    ElasticBuffer{T,N,N-1}(undef, dims)
end

@inline function ElasticBuffer{T}(::UndefInitializer, dims::Dims{N}) where {T,N}
    ElasticBuffer{T,N}(undef, dims)
end


@propagate_inbounds function ElasticBuffer{T,N,M}(A::AbstractArray{<:Any, N}) where {T,N,M}
    ElasticBuffer{T,N,M}(front(size(A)), copyto!(Vector{T}(undef, length(A)), A))
end

@propagate_inbounds function ElasticBuffer{T,N}(A::AbstractArray) where {T,N}
    ElasticBuffer{T,N,N-1}(A)
end

@propagate_inbounds function ElasticBuffer{T}(A::AbstractArray) where {T}
    ElasticBuffer{T,ndims(A)}(A)
end

@propagate_inbounds ElasticBuffer(A::AbstractArray) = ElasticBuffer{eltype(A)}(A)


Base.convert(::Type{T}, A::AbstractArray) where {T<:ElasticBuffer} = A isa T ? A : T(A)

@inline function Base.:(==)(A::ElasticBuffer{<:Any,N,M}, B::ElasticBuffer{<:Any,N,M}) where {N,M}
    A.kernel_size == B.kernel_size && A.data == B.data
end
@inline Base.:(==)(A::ElasticBuffer, B::ElasticBuffer) = false

@inline function Base.size(A::ElasticBuffer)
    (A.kernel_size..., div(length(eachindex(A.data)), A.kernel_length))
end
@inline function Base.size(A::ElasticBuffer, d)
    if d == ndims(A)
        div(length(eachindex(A.data)), A.kernel_length)
    else
        A.kernel_size[d]
    end
end

@inline Base.length(A::ElasticBuffer) = length(A.data)


Base.IndexStyle(::Type{<:ElasticBuffer}) = IndexLinear()

@propagate_inbounds function Base.getindex(A::ElasticBuffer, i::Integer)
    @boundscheck checkbounds(A, i)
    @inbounds getindex(A.data, i)
end

@propagate_inbounds function Base.setindex!(A::ElasticBuffer, x, i::Integer)
    @boundscheck checkbounds(A, i)
    @inbounds setindex!(A.data, x, i)
end


@inline function Base.resize!(A::ElasticBuffer{T,N}, dims::Vararg{Integer,N}) where {T,N}
    kernel_size, size_lastdim = _split_resize_dims(A, dims)
    resize!(A.data, A.kernel_length.divisor * size_lastdim)
    A
end

@inline function growlastdim!(B::ElasticBuffer, n::Integer)
    n < 0 && throw(DomainError(n, "n must be positive"))
    resizelastdim!(B, size(B, ndims(B)) + n)
end

@inline function shrinklastdim!(B::ElasticBuffer, n::Integer)
    n < 0 && throw(DomainError(n, "n must be positive"))
    resizelastdim!(B, size(B, ndims(B)) - n)
end

@inline function resizelastdim!(B::ElasticBuffer, n::Integer)
    kernel_size = front(size(B))
    dims = (kernel_size..., n)
    resize!(B, dims...)
end

@inline function Base.sizehint!(A::ElasticBuffer{T,N}, dims::Vararg{Integer,N}) where {T,N}
    kernel_size, size_lastdim = _split_resize_dims(A, dims)
    sizehint!(A.data, A.kernel_length.divisor * size_lastdim)
    A
end


function Base.append!(dest::ElasticBuffer, src::AbstractArray)
    rem(length(eachindex(src)), dest.kernel_length) != 0 && throw(DimensionMismatch("Can't append, length of source array is incompatible"))
    append!(dest.data, src)
    dest
end

function Base.prepend!(dest::ElasticBuffer, src::AbstractArray)
    rem(length(eachindex(src)), dest.kernel_length) != 0 && throw(DimensionMismatch("Can't prepend, length of source array is incompatible"))
    prepend!(dest.data, src)
    dest
end


@inline function Base.copyto!(dest::ElasticBuffer, doffs::Integer, src::AbstractArray, soffs::Integer, N::Integer)
    copyto!(dest.data, doffs, src, soffs, N)
    dest
end
@inline function Base.copyto!(dest::AbstractArray, doffs::Integer, src::ElasticBuffer, soffs::Integer, N::Integer)
    copyto!(dest, doffs, src.data, soffs, N)
end

@inline Base.copyto!(dest::ElasticBuffer, src::AbstractArray) = (copyto!(dest.data, src); dest)
@inline Base.copyto!(dest::AbstractArray, src::ElasticBuffer) = copyto!(dest, src.data)

@inline function Base.copyto!(dest::ElasticBuffer, doffs::Integer, src::ElasticBuffer, soffs::Integer, N::Integer)
    copyto!(dest.data, doffs, src.data, soffs, N)
    dest
end
@inline function Base.copyto!(dest::ElasticBuffer, src::ElasticBuffer)
    copyto!(dest.data, src.data)
    dest
end


@inline function Base.similar(A::ElasticBuffer, T::Type, dims::Dims{N}) where {N}
    ElasticBuffer{T,N}(front(dims), similar(A.data, T, prod(dims)))
end

Broadcast.BroadcastStyle(::Type{<:ElasticBuffer}) = Broadcast.ArrayStyle{ElasticBuffer}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{ElasticBuffer}}, ::Type{ElType}) where ElType
    similar(ElasticBuffer{ElType}, axes(bc))
end


@inline function Base.unsafe_convert(::Type{Ptr{T}}, A::ElasticBuffer{T}) where T
    Base.unsafe_convert(Ptr{T}, A.data)
end

@inline Base.pointer(A::ElasticBuffer, i::Integer) = pointer(A.data, i)

@inline Base.dataids(A::ElasticBuffer) = Base.dataids(A.data)


@inline function _split_resize_dims(A::ElasticBuffer, dims::NTuple{N,Integer}) where {N}
    kernel_size, size_lastdim = frontlast(dims)
    kernel_size != A.kernel_size && throw(ArgumentError("Can only resize last dimension of an ElasticBuffer"))
    kernel_size, size_lastdim
end

@generated function check_elasticbuffer_parameters(::Type{T}, ::Val{N}, ::Val{M}) where {T,N,M}
    !isa(N, Int) && return :(throw(ArgumentError("ElasticBuffer parameter N must be of type Int")))
    !isa(M, Int) && return :(throw(ArgumentError("ElasticBuffer parameter M must be of type Int")))
    if M != N - 1
        return :(throw(ArgumentError("ElasticBuffer{$T,$N,$M} does not satisfy requirement M == N - 1")))
    end
end