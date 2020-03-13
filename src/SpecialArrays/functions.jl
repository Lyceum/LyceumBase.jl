"""
    flatten(A::AbstractArray{<:AbstractArray{U,M},N}

Flatten `A` into an Array{U,M+N}. Fails if the elements of `A` do not all
have the same size. If the `A` is not a  nested array, the return value is `A` itself.
"""
function flatten end

@inline flatten(A::AbsArr) = A

@inline function flatten(A::AbsArr{<:AbsArr{U,M},N}) where {U,M,N}
    L = M + N
    sz_inner = innersize(A)
    A_flat = Array{U,L}(undef, sz_inner..., size(A)...)
    flattento!(A_flat, A, sz_inner)
end


@inline function flattento!(dest::AbsArr{<:Any, L}, src::AbsArr{<:AbsArr{<:Any,M},N}, sz_inner::Dims{M} = innersize(src)) where {L,M,N}
    L == M + N || throw(ArgumentError("ndims(dest) != innerndims(src)"))
    if size(dest) != (sz_inner..., size(src)...)
        throw(DimensionMismatch("innersize(dest) != innersize(src)"))
    end

    len_inner = prod(sz_inner)
    from = firstindex(dest)
    for x in src
        copyto!(dest, from, x, firstindex(x), len_inner)
        from += len_inner
    end
    @assert from  == len_inner * length(src) + 1
    return dest
end

@inline function nestto!(dest::AbsArr{<:AbsArr{<:Any,M},N}, src::AbsArr{<:Any,L}, sz_inner::Dims{M} = innersize(dest)) where {L,M,N}
    L == M + N || throw(ArgumentError("ndims(dest) != innerndims(src)"))
    if size(src) != (sz_inner..., size(dest)...)
        throw(DimensionMismatch("innersize(dest) != innersize(src)"))
    end

    len_inner = prod(sz_inner)
    from = firstindex(src)
    for x in dest
        copyto!(x, firstindex(x), src, from, len_inner)
        from += len_inner
    end
    @assert from  == len_inner * length(dest) + 1
    return dest
end

"""
    flatview(A::AbstractArray)
    flatview(A::AbstractArray{<:AbstractArray{<:...}})

View array `A` in a suitable flattened form. The shape of the flattened form
will depend on the type of `A`. If the `A` is not a nested array, the return
value is `A` itself. When no type-specific method is available, `flatview`
will fall back to `flatten(A)`.
"""
function flatview end

@inline flatview(A::AbsArr) = A
# TODO: Lazy flatview?
@inline flatview(A::AbsArr{<:AbsArr}) = flatten(A)


"""
    innereltype(A::AbstractArray{<:AbstractArray})
    innereltype(::Type{<:AbstractArray{<:AbstractArray}})

Returns the element type of the element arrays of `A`.
Equivalent to `eltype(eltype(A))`.
"""
function innereltype end

@inline innereltype(::Type{A}) where {A <: AbsArr} = eltype(eltype(A))
@inline innereltype(A::AbsArr) = innereltype(typeof(A))


"""
    innerndims(A::AbstractArray{<:AbstractArray})
    innerndims(::Type{<:AbstractArray{<:AbstractArray}})

Returns the dimensionality of the element arrays of `A`.
Throws an error if the elements of `A` do not have equal dimensionality.
"""
function innerndims end

@inline innerndims(::Type{<:AbsArr{<:AbsArr{<:Any,N}}}) where {N} = N
@inline function innerndims(::Type{<:AbsArr{<:AbsArr}})
    throw(DimensionMismatch("The elements of A do not have equal dimensionality"))
end
@inline innerndims(A::AbsArr) = innerndims(typeof(A))


"""
    innersize(A::AbstractArray{<:AbstractArray}[, d])

Returns the size of the element arrays of `A`.
Throws an error if the elements of `A` do not have equal size.
"""
function innersize end

function innersize(A::AbsArr{<:AbsArr{<:Any,M}}) where {M}
    if isempty(A)
        sz = ntuple(_ -> zero(Int), Val(M))
    else
        sz = size(first(A))
        for a in A
            size(a) == sz || throw(DimensionMismatch("The elements of A do not have equal sizes"))
        end
    end
    sz
end
function innersize(A::AbsArr{<:AbsArr})
    throw(DimensionMismatch("The elements of A do not have equal dimensionality"))
end
@inline function innersize(A::AbsArr, d::Integer)
    sz = innersize(A)
    d <= length(sz) ? sz[d] : 1 # TODO offset arrays
end


"""
    innerlength(A::AbstractArray{<:AbstractArray})

Returns the common length of the element arrays of `A`.
Throws an error if the element arrays of `A` do not have equal size.
"""
function innerlength end

@inline innerlength(A::AbsArr{<:AbsArr}) = prod(innersize(A))


"""
    inneraxes(A::AbstractArray{<:AbstractArray}[, d])

Returns the common length of the element arrays of `A`.
Throws an error if the element arrays of `A` do not have equal axes.
"""
function inneraxes end

function inneraxes(A::AbsArr{<:AbsArr{<:Any, M}}) where {M}
    if isempty(A)
        # TODO this would be wrong for offset arrays?
        ax = ntuple(_ -> Base.OneTo(0), Val(M))
    else
        ax = axes(first(A))
        for a in A
            axes(a) == ax || throw(DimensionMismatch("The elements of A do not have equal axes"))
        end
    end
    ax
end
function inneraxes(A::AbsArr{<:AbsArr})
    throw(DimensionMismatch("The elements of A do not have equal dimensionality"))
end

@inline function inneraxes(A::AbsArr, d::Integer)
    ax = inneraxes(A)
    d <= length(ax) ? ax[d] : Base.OneTo(1) # TODO offset arrays
end
