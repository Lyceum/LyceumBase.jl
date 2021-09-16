Base.@pure function geteltype(::Type{A}) where {T,N,A<:AbstractArray{T,N}}
    I = Tuple{ntuple(i -> Base.Slice{Base.OneTo{Int}}, N - 1)...,UnitRange{Int}}
    SubArray{T,N,A,I,true}
end

#Base.@pure geteltype(::Type{A}) where {A<:UnsafeArray} = A

Base.@pure _ncolons(::Val{N}) where {N} = ntuple(_ -> Colon(), Val{N}())

function check_type_parameters(T, N::Int, M::Int, P, O)
    @assert ndims(P) === N === M + 1
    @assert T === geteltype(P)
    @assert eltype(T) === eltype(P)
end


struct BatchedArray{T,N,M,P<:AbstractArray,O<:AbstractVector{Int}} <: AbstractVector{T}
    parent::P
    offsets::O
    kernel_size::Dims{M}

    function BatchedArray{T,N,M,P,O}(
        parent,
        offsets,
        kernel_size,
        checks = full_consistency_check,
    ) where {T,N,M,O,P}
        check_type_parameters(T, N, M, P, O)
        A = new{T,N,M,P,O}(parent, offsets, kernel_size)
        checks(A)
        A
    end
end

BatchedArray(shape::AbstractShape) = BatchedArray(eltype(shape), size(shape))


function BatchedArray(parent, offsets::AbstractVector{<:Integer}, kernel_size::Dims)
    T = geteltype(typeof(parent))
    N = ndims(parent)
    M = N - 1
    BatchedArray{T,N,M,typeof(parent),typeof(offsets)}(parent, offsets, kernel_size)
end

function BatchedArray(parent, offsets::AbstractVector{<:Integer})
    BatchedArray(parent, offsets, Base.front(size(parent)))
end

function BatchedArray(parent, batch_ranges::AbstractVector{<:AbstractUnitRange{<:Integer}})
    offsets = Vector{Int}(undef, length(batch_ranges) + 1)
    offsets[1] = 1
    for i in eachindex(batch_ranges)
        offsets[i+1] = offsets[i] + length(batch_ranges[i])
    end
    BatchedArray(parent, offsets, Base.front(size(parent)))
end

function BatchedArray(T::Type, kernel_size::Dims)
    parent = ElasticArray{T}(undef, kernel_size..., 0)
    offsets = [1]
    BatchedArray(parent, offsets, kernel_size)
end


# TODO hash?
import Base.==
(==)(A::BatchedArray, B::BatchedArray) =
    A.parent == B.parent && A.offsets == B.offsets && A.kernel_size == B.kernel_size

Base.IndexStyle(A::BatchedArray) = IndexLinear()

Base.size(A::BatchedArray) = (length(A.offsets) - 1,)


Base.@propagate_inbounds function Base.getindex(
    A::BatchedArray{T,N,M},
    I::Integer,
) where {T,N,M}
    @boundscheck checkbounds(A, I)
    r, s = _batchrange(A, I)
    view(A.parent, _ncolons(Val(M))..., r)
end

Base.@propagate_inbounds function Base._getindex(
    ::IndexStyle,
    A::BatchedArray{T,N,M},
    I::AbstractUnitRange{<:Integer},
) where {T,N,M}
    @boundscheck checkbounds(A, I)

    from = first(I)
    to = last(I)
    r, s = _batchrange(A, I)

    newoffsets = A.offsets[from:(to+1)]
    if from != 1
        newoffsets .-= (first(newoffsets) - 1)
    end
    parentview = view(A.parent, _ncolons(Val(M))..., r)

    BatchedArray(parentview, newoffsets, A.kernel_size)
end

function Base.similar(A::BatchedArray)
    BatchedArray(similar(A.parent), copy(A.offsets), A.kernel_size)
end

function batchlike(A::BatchedArray{T,N}, other::AbstractArray{U,N}) where {T,U,N}
    Base.front(size(A.parent)) == Base.front(size(other)) || error("Size of other must match size(flatview(B))")
    BatchedArray(other, copy(A.offsets), A.kernel_size)
end

Base.@propagate_inbounds function Base.setindex!(
    A::BatchedArray{T,N,M},
    x::AbstractArray{U,N},
    I::Integer,
) where {T,N,M,U}
    r, s = _batchrange(A, I)
    @boundscheck s == size(
        x,
    ) || throw(DimensionMismatch("Can't assign array to element $I of BatchedArray, array size is incompatible"))
    A.parent[_ncolons(Val(M))..., r] = x
    A
end

Base.copy(V::T) where {T<:BatchedArray} = T(copy(V.parent), copy(V.offsets), V.kernel_size)

Base.sizehint!(A::BatchedArray, n::Integer) = (sizehint!(A.parent, A.kernel_size..., n); A)

function Base.empty!(A::BatchedArray)
    newsize = (A.kernel_size..., 0)
    resize!(A.parent, newsize...)
    resize!(A.offsets, 1)
    @assert A.offsets[1] == 1
    A
end


flatview(A::BatchedArray) = A.parent

function Base.push!(A::BatchedArray{T,N}, B::AbstractArray{U,N}) where {T,U,N}
    @boundscheck if A.kernel_size != Base.front(size(B))
        throw(DimensionMismatch("Can't push element onto BatchedArray, array size is incompatible"))
    end
    append!(A.parent, B)
    push!(A.offsets, A.offsets[end] + size(B, ndims(B)))
    A
end

Base.append!(A::BatchedArray, xs) = (foreach(x -> push!(A, x), xs); A)

#UnsafeArrays.uview(A::BatchedArray) =
#    BatchedArray(uview(A.parent), uview(A.offsets), A.kernel_size)

nsamples(A::BatchedArray) = size(A.parent, ndims(A.parent))

Base.@propagate_inbounds function _batchrange(A::BatchedArray, i::Int, j::Int = i + 1)
    from = A.offsets[i]
    until = A.offsets[j]
    to = until - 1
    len = until - from
    from:to, (A.kernel_size..., len)
end
_batchrange(A::BatchedArray, I::UnitRange{Int}) = _batchrange(A, first(I), last(I) + 1)

function full_consistency_check(A::BatchedArray)
    A.offsets[1] == firstindex(A.parent) || throw(ArgumentError("First offset not equal to firstindex(BatchAarray.parent)"))
    len = 0
    for i = 2:length(A.offsets)
        o1 = A.offsets[i-1]
        o2 = A.offsets[i]
        o2 > o1 || throw(ArgumentError("Overlapping indices in BatchArray.offsets"))
        len += o2 - o1
    end
    if len != size(A.parent, ndims(A.parent))
        throw(ArgumentError("Length computed from BatchArray.offsets inconsistent with length of last dimension of BatchArray.parent"))
    end
    nothing
end

no_consistency_checks(A::BatchedArray) = nothing
