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

    function ElasticBuffer(kernel_size::Dims{M}, data::Vector{T}) where {T,M}
        kernel_length = SignedMultiplicativeInverse{Int}(prod(kernel_size))
        if rem(length(eachindex(data)), kernel_length) != 0
            throw(ArgumentError("length(data) must be integer multiple of prod(kernel_size)"))
        end
        new{T,M+1,M}(kernel_size, kernel_length, data)
    end
end


ElasticBuffer{T}(::UndefInitializer, dims::Integer...) where {T} = ElasticBuffer{T}(undef, dims)

function ElasticBuffer{T}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N}
    ElasticBuffer{T}(undef, convert(NTuple{N,Int}, dims))
end

function ElasticBuffer{T}(::UndefInitializer, dims::Dims{N}) where {T,N}
    kernel_size, size_lastdim = frontlast(dims)
    data = Vector{T}(undef, prod(kernel_size) * size_lastdim)
    ElasticBuffer(kernel_size, data)
end


@propagate_inbounds function ElasticBuffer{T,N,M}(A::AbstractArray) where {T,N,M}
    M == N - 1 || throw(ArgumentError("ElasticBuffer{$T,$N,$M} does not satisfy requirement M == N - 1"))
    ElasticBuffer{T,N}(A)
end

@propagate_inbounds function ElasticBuffer{T,N}(A::AbstractArray{U,N}) where {T,N,U}
    copyto!(ElasticBuffer{T}(undef, size(A)...), A)
end

@propagate_inbounds ElasticBuffer{T}(A::AbstractArray{U,N}) where {T,N,U} = ElasticBuffer{T,N}(A)

@propagate_inbounds ElasticBuffer(A::AbstractArray{T,N}) where {T,N} = ElasticBuffer{T,N}(A)


Base.convert(::Type{T}, A::AbstractArray) where {T<:ElasticBuffer} = A isa T ? A : T(A)


@inline function Base.:(==)(A::ElasticBuffer{<:Any,N,M}, B::ElasticBuffer{<:Any,N,M}) where {N,M}
    A.kernel_size == B.kernel_size && A.data == B.data
end

@inline function Base.size(A::ElasticBuffer)
    (A.kernel_size..., div(length(eachindex(A.data)), A.kernel_length))
end

@inline Base.length(A::ElasticBuffer) = length(A.data)

@inline Base.IndexStyle(::Type{<:ElasticBuffer}) = IndexLinear()

@propagate_inbounds function Base.getindex(A::ElasticBuffer, i::Integer)
    @boundscheck checkbounds(A, i)
    @inbounds getindex(A.data, i)
end

@propagate_inbounds function Base.setindex!(A::ElasticBuffer, x, i::Integer)
    @boundscheck checkbounds(A, i)
    @inbounds setindex!(A.data, x, i)
end

# TODO
#@inline Base.parent(A::ElasticBuffer) = reshape(A.data, size(A))
@inline Base.parent(A::ElasticBuffer) = A.data

@inline Base.dataids(A::ElasticBuffer) = Base.dataids(A.data)

@inline function Base.resize!(A::ElasticBuffer{T,N}, dims::Vararg{Integer,N}) where {T,N}
    kernel_size, size_lastdim = _split_resize_dims(A, dims)
    resize!(A.data, A.kernel_length.divisor * size_lastdim)
    A
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

#@inline function Base.similar(A::ElasticBuffer, T::Type, dims::Dims)
#    ElasticBuffer{T}(undef, dims)
#end
# TODO
@inline function Base.similar(::Type{<:ElasticBuffer}, T::Type, dims::Dims)
    ElasticBuffer{T}(undef, dims)
end


@inline Base.unsafe_convert(::Type{Ptr{T}}, A::ElasticBuffer{T}) where T = Base.unsafe_convert(Ptr{T}, A.data)
@inline Base.pointer(A::ElasticBuffer, i::Integer) = pointer(A.data, i)

@inline function growend!(B::ElasticBuffer, n::Integer = 1)
    n < 0 && throw(DomainError(n, "n must be positive"))
    resizeend!(B, n)
end

@inline function shrinkend!(B::ElasticBuffer, n::Integer = 1)
    n < 0 && throw(DomainError(n, "n must be positive"))
    resizeend!(B, -n)
end

@inline function resizeend!(B::ElasticBuffer, n::Integer)
    kernel_size, size_lastdim = frontlast(size(B))
    dims = (kernel_size..., size_lastdim + n)
    resize!(B, dims...)
end


@inline function _split_resize_dims(A::ElasticBuffer, dims::NTuple{N,Integer}) where {N}
    kernel_size, size_lastdim = frontlast(dims)
    kernel_size != A.kernel_size && throw(ArgumentError("Can only resize last dimension of an ElasticBuffer"))
    kernel_size, size_lastdim
end