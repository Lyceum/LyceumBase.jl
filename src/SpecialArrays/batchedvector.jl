@inline function _batched_viewtype(A::AbsVec)
    i = firstindex(A)
    viewtype(A, i:i)
end

# TODO
# 1. @boundscheck

struct BatchedVector{T,P<:AbsVec,O<:AbsVec{<:Integer}} <: AbsVec{T}
    parent::P
    offsets::O
    @propagate_inbounds function BatchedVector{T,P,O}(parent, offsets) where {T,P<:AbsVec, O<:AbsVec{<:Integer}}
        full_consistency_check(parent, offsets)
        new{T,P,O}(parent, offsets)
    end
end

@propagate_inbounds function BatchedVector{T,P}(parent::AbsVec, offsets::AbsVec{<:Integer}) where {T,P<:AbsVec}
    BatchedVector{T,P,typeof(offsets)}(parent, offsets)
end

@propagate_inbounds function BatchedVector{T}(parent::AbsVec, offsets::AbsVec{<:Integer}) where {T}
    BatchedVector{T,typeof(parent),typeof(offsets)}(parent, offsets)
end

@propagate_inbounds function BatchedVector(parent::AbsVec, offsets::AbsVec{<:Integer})
    T = _batched_viewtype(parent)
    BatchedVector{T,typeof(parent),typeof(offsets)}(parent, offsets)
end


@inline function _check_isordered(r::AbstractRange)
    first(r) <= last(r) || throw(ArgumentError("first(range) must be <= last(range). Got: $r"))
    nothing
end
function _check_isordered_disjoint(x::AbsVec{<:AbstractRange})
    i, rest = Iterators.peel(eachindex(x))
    r1 = x[i]
    _check_isordered(r1)
    for j in rest
        r2 = x[j]
        _check_isordered(r2)
        last(r1) < first(r2) || throw(ArgumentError("Ranges cannot intersect: $r1, $r2"))
        r1 = r2
    end
    nothing
end

@propagate_inbounds function batchedvector(parent::AbsVec, batch_ranges::AbsVec{<:AbstractUnitRange{<:Integer}})
    @boundscheck _check_isordered_disjoint(batch_ranges)
    offsets = Vector{Int}(undef, length(batch_ranges) + 1)
    offsets[1] = 0
    for (i, r) in zip(2:length(offsets), batch_ranges)
        offsets[i] = last(r)
    end
    BatchedVector(parent, offsets)
end


@propagate_inbounds function batchedvector(parent, batch_lengths::AbsVec{<:Integer})
    offsets = Vector{Int}(undef, length(batch_lengths) + 1)
    offsets[1] = cumsum = 0
    for (i, l) in zip(2:length(offsets), batch_lengths)
        cumsum += l
        offsets[i] = cumsum
    end
    BatchedVector(parent, offsets)
end


"""
    batchlike(A::AbstractVector, B::BatchedVector) --> BatchedVector

Create a BatchedVector view of `A` using the same batch indices of `B`.
"""
@propagate_inbounds function batchlike(A::AbsVec, B::BatchedVector)
    @boundscheck if length(A) != length(B.parent)
        throw(ArgumentError("length(A) != length(parent(B))"))
    end
    BatchedVector(A, copy(B.offsets))
end


# TODO hash?
@inline function Base.:(==)(A::BatchedVector, B::BatchedVector)
    A.offsets == B.offsets && A.parent == B.parent
end


@inline Base.size(A::BatchedVector) = (length(A), )
@inline Base.length(A::BatchedVector) = length(A.offsets) - 1

@inline Base.IndexStyle(A::BatchedVector) = IndexLinear()

# TODO test boundscheck + get rid of double boundscheck
@propagate_inbounds function Base.getindex(A::BatchedVector, i::Int)
    @boundscheck checkbounds(A, i)
    batch_range = batchrange(A, i)
    @boundscheck checkbounds(A.parent, batch_range)
    @inbounds getindex(A.parent, batch_range)
end


# TODO test boundscheck + get rid of double boundscheck
@propagate_inbounds function Base.setindex!(A::BatchedVector, x, i::Int)
    @boundscheck checkbounds(A, i)
    batch_range = batchrange(A, i)
    @boundscheck checkbounds(A.parent, batch_range)
    @inbounds setindex!(A.parent, x, batch_range)
end


function Base.push!(A::BatchedVector{<:AbsArr{<:Any, M}}, x::AbsArr{<:Any, M}) where {M}
    append!(A.parent, x)
    push!(A.offsets, A.offsets[end] + length(x))
    full_consistency_check(A)
    A
end

function Base.push!(A::BatchedVector, x) where {M}
    push!(A.parent, x)
    push!(A.offsets, A.offsets[end] + 1)
    full_consistency_check(A)
    A
end

Base.append!(A::BatchedVector, xs) = (for x in xs push!(A, x) end; A)

# TODO handle differnt eltype + size or error
@inline function Base.similar(A::BatchedVector)
    BatchedVector(similar(A.parent), copy(A.offsets))
end

@inline Base.copy(A::BatchedVector) = typeof(A)(copy(A.parent), copy(A.offsets))

@inline function UnsafeArrays.unsafe_uview(A::BatchedVector)
    BatchedVector(uview(A.parent), uview(A.offsets))
end

@propagate_inbounds function batchrange(A::BatchedVector, i::Int)
    @boundscheck (checkbounds(A.offsets, i); checkbounds(A.offsets, i + 1))
    j = firstindex(A.parent)
    from = A.offsets[i] + j
    to = A.offsets[i + 1] - 1 + j
    from:to
end

@propagate_inbounds function batchrange(A::BatchedVector, r::AbstractUnitRange)
    @boundscheck (checkbounds(A.offsets, i); checkbounds(A.offsets, i + 1))
    j = firstindex(A.parent)
    from = A.offsets[i] + j
    to = A.offsets[i + 1] - 1 + j
    from:to
end

full_consistency_check(A::BatchedVector) = full_consistency_check(A.parent, A.offsets)
function full_consistency_check(parent::AbsVec, offsets::AbsVec{<:Integer})
    Base.require_one_based_indexing(parent)
    Base.require_one_based_indexing(offsets)
    length(offsets) >= 1 || throw(ArgumentError("offsets cannot be empty"))
    first(offsets) == 0 || throw(ArgumentError("First offset is non-zero"))
    len = 0
    for i = 2:length(offsets)
        o1 = offsets[i-1]
        o2 = offsets[i]
        o2 > o1 || throw(ArgumentError("Overlapping indices found in offsets"))
        len += o2 - o1
    end
    if len != length(parent)
        throw(ArgumentError("Length computed from offsets is not equal to length(parent)"))
    end
    nothing
end