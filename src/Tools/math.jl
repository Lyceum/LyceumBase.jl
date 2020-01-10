function _cache_mmtv(A, x)
    T = Base.promote_op(*, eltype(A), eltype(x))
    Vector{T}(undef, size(A, 2))
end

"""
    mmtv!(y, A, x, alpha, beta[, cache = similar(x)])

Update the vector `y` as `alpha*A*(A'*x) + beta*y`.

A vector `cache` of length `size(A, 2)` can be supplied to avoid a temporary array allocation.
"""
function mmtv!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector, alpha = true, beta = false; cache::AbstractVector = _cache_mmtv(A, x))
    mul!(cache, transpose(A), x)
    mul!(y, A, cache, alpha, beta)
end

"""
    mmtv(A, x, alpha, beta[, cache = similar(x)])

Return `alpha*A*(A'*x)

A vector `cache` of length `size(A, 2)` can be supplied to avoid a temporary array allocation.
"""
@inline function mmtv(A::AbstractMatrix, x::AbstractVector, alpha = true; cache::AbstractVector = _cache_mmtv(A, x))
    T = Base.promote_op(*, eltype(A), eltype(x))
    y = Vector{T}(undef, size(A, 1))
    mmtv!(y, A, x, alpha, false, cache = cache)
end