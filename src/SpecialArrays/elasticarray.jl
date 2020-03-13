"""
    ElasticArray{T,N,M} <: DenseArray{T,N}

An `ElasticArray` can grow/shrink in its last dimension. `N` is the total number of
dimensions, `M == N - 1` the number of non-resizable dimensions (all but the last dimension).

Constructors:

    ElasticArray(kernel_size::Dims{M}, data::Vector{T}, len::Int)
    ElasticArray{T}(::UndefInitializer, dims::NTuple{N,Integer})
    ElasticArray{T}(::UndefInitializer, dims::Integer...)
    convert(ElasticArray, A::AbsArr)
"""
struct ElasticArray{T,N,M} <: DenseArray{T,N}
    kernel_size::Dims{M}
    kernel_length::SignedMultiplicativeInverse{Int}
    data::Vector{T}

    function ElasticArray{T,N,M}(kernel_size::Dims{M}, data::Vector{T}) where {T,N,M}
        check_elasticarray_parameters(T, Val(N), Val(M))
        kernel_length = SignedMultiplicativeInverse{Int}(prod(kernel_size))
        if rem(length(eachindex(data)), kernel_length) != 0
            throw(ArgumentError("length(data) must be integer multiple of prod(kernel_size)"))
        end
        new{T,N,M}(kernel_size, kernel_length, data)
    end
end

@inline function ElasticArray{T,N,M}(kernel_size::IDims{M}, data::AbsVec) where {T,N,M}
    check_elasticarray_parameters(T, Val(N), Val(M))
    ElasticArray{T,N,M}(convert(Dims{M}, kernel_size), convert(Vector{T}, data))
end
@inline function ElasticArray{T,N}(kernel_size::IDims{M}, data::AbsVec) where {T,N,M}
    ElasticArray{T,N,M}(kernel_size, data)
end
@inline function ElasticArray{T}(kernel_size::IDims{M}, data::AbsVec) where {T,M}
    ElasticArray{T,M+1}(kernel_size, data)
end
@inline function ElasticArray(kernel_size::IDims, data::AbsVec{T}) where {T}
    ElasticArray{T}(kernel_size, data)
end


@inline function ElasticArray{T,N,M}(::UndefInitializer, dims::IDims{N}) where {T,N,M}
    check_elasticarray_parameters(T, Val(N), Val(M))
    ElasticArray{T,N,M}(front(dims), Vector{T}(undef, prod(dims)))
end
@inline function ElasticArray{T,N}(::UndefInitializer, dims::IDims{N}) where {T,N}
    ElasticArray{T,N,N-1}(undef, dims)
end
@inline function ElasticArray{T}(::UndefInitializer, dims::IDims{N}) where {T,N}
    ElasticArray{T,N}(undef, dims)
end

@inline function ElasticArray{T,N,M}(::UndefInitializer, dims::IVararg{N}) where {T,N,M}
    ElasticArray{T,N,M}(undef, dims)
end
@inline function ElasticArray{T,N}(::UndefInitializer, dims::IVararg{N}) where {T,N}
    ElasticArray{T,N}(undef, dims)
end
@inline function ElasticArray{T}(::UndefInitializer, dims::IVararg{N}) where {T,N}
    ElasticArray{T,N}(undef, dims)
end


@propagate_inbounds function ElasticArray{T,N,M}(A::AbsArr{<:Any,N}) where {T,N,M}
    check_elasticarray_parameters(T, Val(N), Val(M))
    ElasticArray{T,N,M}(front(size(A)), copyto!(Vector{T}(undef, length(A)), A))
end
@propagate_inbounds function ElasticArray{T,N}(A::AbsArr{<:Any,N}) where {T,N}
    ElasticArray{T,N,N-1}(A)
end
@propagate_inbounds function ElasticArray{T}(A::AbsArr) where {T}
    ElasticArray{T,ndims(A)}(A)
end

@propagate_inbounds ElasticArray(A::AbsArr) = ElasticArray{eltype(A)}(A)


Base.convert(::Type{T}, A::AbsArr) where {T<:ElasticArray} = A isa T ? A : T(A)


####
#### Core Array Interface
####

@inline function Base.size(A::ElasticArray)
    (A.kernel_size..., div(length(eachindex(A.data)), A.kernel_length))
end
@inline function Base.size(A::ElasticArray, d)
    d == ndims(A) ? div(length(eachindex(A.data)), A.kernel_length) : A.kernel_size[d]
end

@propagate_inbounds Base.getindex(A::ElasticArray, i::Int) = getindex(A.data, i)
@propagate_inbounds Base.setindex!(A::ElasticArray, x, i::Int) = setindex!(A.data, x, i)

Base.IndexStyle(::Type{<:ElasticArray}) = IndexLinear()

@inline Base.length(A::ElasticArray) = length(A.data)

@inline function Base.similar(A::ElasticArray, T::Type, dims::Dims{N}) where {N}
    ElasticArray{T,N}(front(dims), similar(A.data, T, prod(dims)))
end


####
#### Misc
####

@inline function Base.:(==)(A::ElasticArray{<:Any,N,M}, B::ElasticArray{<:Any,N,M}) where {N,M}
    A.kernel_size == B.kernel_size && A.data == B.data
end
@inline Base.:(==)(A::ElasticArray, B::ElasticArray) = false


@inline function Base.unsafe_convert(::Type{Ptr{T}}, A::ElasticArray{T}) where T
    Base.unsafe_convert(Ptr{T}, A.data)
end

@inline Base.pointer(A::ElasticArray, i::Integer) = pointer(A.data, i)

@inline Base.dataids(A::ElasticArray) = Base.dataids(A.data)


@inline function Base.copyto!(dest::ElasticArray, doffs::Integer, src::AbsArr, soffs::Integer, N::Integer)
    copyto!(dest.data, doffs, src, soffs, N)
    return dest
end
@inline function Base.copyto!(dest::AbsArr, doffs::Integer, src::ElasticArray, soffs::Integer, N::Integer)
    copyto!(dest, doffs, src.data, soffs, N)
end

@inline Base.copyto!(dest::ElasticArray, src::AbsArr) = (copyto!(dest.data, src); dest)
@inline Base.copyto!(dest::AbsArr, src::ElasticArray) = copyto!(dest, src.data)

@inline function Base.copyto!(dest::ElasticArray, doffs::Integer, src::ElasticArray, soffs::Integer, N::Integer)
    copyto!(dest.data, doffs, src.data, soffs, N)
    return dest
end
@inline function Base.copyto!(dest::ElasticArray, src::ElasticArray)
    copyto!(dest.data, src.data)
    return dest
end


@inline function Base.resize!(A::ElasticArray{<:Any,N}, dims::IDims{N}) where {N}
    kernel_size, size_lastdim = _split_resize_dims(A, dims)
    resize!(A.data, A.kernel_length.divisor * size_lastdim)
    return A
end
@inline function Base.resize!(A::ElasticArray{<:Any,N}, dims::IVararg{N}) where {N}
    resize!(A, dims)
end

@inline function growlastdim!(A::ElasticArray, n::Integer)
    n < 0 && throw(DomainError(n, "n must be positive"))
    return resizelastdim!(A, size(A, ndims(A)) + n)
end

@inline function shrinklastdim!(A::ElasticArray, n::Integer)
    n < 0 && throw(DomainError(n, "n must be positive"))
    return resizelastdim!(A, size(A, ndims(A)) - n)
end

@inline resizelastdim!(A::ElasticArray, n::Integer) = resize!(A, (A.kernel_size..., n))


@inline function Base.sizehint!(A::ElasticArray{<:Any,N}, dims::IDims{N}) where {N}
    kernel_size, size_lastdim = _split_resize_dims(A, dims)
    sizehint!(A.data, A.kernel_length.divisor * size_lastdim)
    return A
end
@inline function Base.sizehint!(A::ElasticArray{<:Any,N}, dims::IVararg{N}) where {N}
    sizehint!(A, dims)
end


function Base.append!(dest::ElasticArray, src::AbsArr)
    if rem(length(eachindex(src)), dest.kernel_length) != 0
        throw(DimensionMismatch("Can't append, length of source array is incompatible"))
    end
    append!(dest.data, src)
    return dest
end

function Base.prepend!(dest::ElasticArray, src::AbsArr)
    if rem(length(eachindex(src)), dest.kernel_length) != 0
        throw(DimensionMismatch("Can't prepend, length of source array is incompatible"))
    end
    prepend!(dest.data, src)
    return dest
end


####
#### Broadcasting
####

Broadcast.BroadcastStyle(::Type{<:ElasticArray}) = Broadcast.ArrayStyle{ElasticArray}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{ElasticArray}}, ::Type{ElType}) where ElType
    similar(ElasticArray{ElType}, axes(bc))
end


####
#### Util
####

@inline function _split_resize_dims(A::ElasticArray, dims::IDims{N}) where {N}
    kernel_size, size_lastdim = frontlast(dims)
    if kernel_size != A.kernel_size
        throw(ArgumentError("Can only resize last dimension of an ElasticArray"))
    end
    return kernel_size, size_lastdim
end

@generated function check_elasticarray_parameters(::Type{T}, ::Val{N}, ::Val{M}) where {T,N,M}
    !isa(N, Int) && return :(throw(ArgumentError("ElasticArray parameter N must be of type Int")))
    !isa(M, Int) && return :(throw(ArgumentError("ElasticArray parameter M must be of type Int")))
    M < 0 && return :(throw(DomainError($M, "ElasticArray parameter M cannot be negative")))
    N < 0 && return :(throw(DomainError($N, "ElasticArray parameter N cannot be negative")))
    if M != N - 1
        return :(throw(ArgumentError("ElasticArray{$T,$N,$M} does not satisfy requirement M == N - 1")))
    end
end