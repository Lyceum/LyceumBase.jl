####
#### inner_* functions
####

"""
    innereltype(A::AbstractArray{<:AbstractArray})
    innereltype(A::Type{<:AbstractArray{<:AbstractArray}})

Returns the common element type of the element arrays of `A`.
Equivalent to eltype(eltype(A)).
"""
function innereltype end

innereltype(::Type{A}) where {A<:AbsNestedArr} = eltype(eltype(A))
innereltype(A::AbsNestedArr) = eltype(eltype(A))


"""
    innerndims(A::AbstractArray{<:AbstractArray})
    innerndims(A::Type{<:AbstractArray{<:AbstractArray}})

Returns the dimensionality of the element arrays of `A`.
Equivalent to ndims(eltype(A)).
"""
function innerndims end

innerndims(::Type{A}) where {A<:AbsNestedArr} = ndims(eltype(A))
innerndims(A::AbsNestedArr) = ndims(eltype(A))


"""
    inneraxes(A::AbstractArray{<:AbstractArray}[, d])

Returns the common length of the element arrays of `A`.
Throws an error if the element arrays of `A` do not have equal axes.
"""
function inneraxes end

function inneraxes(A::AbsNestedArr)
    M = innerndims(A)
    if isempty(A)
        # TODO this would be wrong for offset arrays?
        ax = ntuple(_ -> Base.OneTo(0), Val(M))
    else
        ax = axes(first(A))
        length(ax) == M || throw(DimensionMismatch("inner axes do not match inner ndims"))
        for a in A
            if axes(a) != ax
                throw(DimensionMismatch("The elements of A do not have equal axes"))
            end
        end
    end
    return ax
end

@inline function inneraxes(A::AbsNestedArr, d::Integer)
    d <= innerndims(A) ? inneraxes(A)[d] : Base.OneTo(1)
end


"""
    innersize(A::AbstractArray{<:AbstractArray}[, d])

Returns the size of the element arrays of `A`.
Throws an error if the elements of `A` do not have equal size.
"""
function innersize end

function innersize(A::AbsNestedArr)
    M = innerndims(A)
    if isempty(A)
        sz = ntuple(_ -> 0, Val(M))
    else
        sz = size(first(A))
        length(sz) == M || throw(DimensionMismatch("inner size does not match inner ndims"))
        for a in A
            if size(a) != sz
                throw(DimensionMismatch("The elements of A do not have equal size"))
            end
        end
    end
    return sz
end

@inline innersize(A::AbsNestedArr, d::Integer) = d <= innerndims(A) ? innersize(A)[d] : 1


"""
    innerlength(A::AbstractArray{<:AbstractArray})

Returns the common length of the element arrays of `A`.
Throws an error if the element arrays of `A` do not have equal size.
"""
function innerlength end

@inline innerlength(A::AbsNestedArr) = prod(innersize(A))


####
#### Conversions between nested and flat
####

"""
    flatten(A::AbstractArray{<:AbstractArray{U,M},N}

Flatten `A` into an Array{U,M+N}. Fails if the elements of `A` do not all
have the same size. If the `A` is not a  nested array, the return value is `A` itself.
"""
function flatten end

flatten(A::AbsArr) = A

function flatten(nested::AbsNestedArr)
    sz_inner = innersize(nested)
    sz_outer = size(nested)
    L = length(sz_inner) + length(sz_outer)
    V = innereltype(nested)
    flat = Array{V,L}(undef, sz_inner..., sz_outer...)
    return _flatten!(flat, nested, sz_inner)
end

flatten!(flat::AbsArr, nested::AbsNestedArr) = _flatten!(flat, nested, innersize(nested))

function _flatten!(flat::AbsArr, nested::AbsNestedArr, sz_inner)
    _check_compatible(flat, nested, sz_inner)
    if prod(tail(size(flat), ndims(nested))) < length(nested)
        throw(ArgumentError("prod(size(flat)[M+1:end]) must be >= length(nested)"))
    end
    len_inner = prod(sz_inner)
    from = firstindex(flat)
    @inbounds for a in nested
        copyto!(flat, from, a, firstindex(a), len_inner)
        from += len_inner
    end
    return flat
end


function nest!(nested::AbsSimilarNestedArr, flat::AbsArr)
    sz_inner = innersize(nested)
    _check_compatible(flat, nested, sz_inner)
    if length(nested) < prod(tail(size(flat), ndims(nested)))
        throw(ArgumentError("length(nested) must >= prod(size(flat)[M+1:end])"))
    end
    len_inner = prod(sz_inner)
    from = firstindex(flat)
    for a in nested
        copyto!(a, firstindex(a), flat, from, len_inner)
        from += len_inner
    end
    return nested
end


function _check_compatible(flat::AbsArr, nested::AbsSimilarNestedArr{<:Any,M}, sz_inner::Dims{M}) where {M}
    ndims(flat) == M + ndims(nested)|| throw(DimensionMismatch("ndims(flat) must equal innerndims(nested) + ndims(nested)"))
    front(size(flat), static(M)) == sz_inner || throw(DimensionMismatch("inner sizes of flat and nested must match"))
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

flatview(A::AbsArr) = A
# TODO: Lazy flatview?
flatview(A::AbsSimilarNestedArr) = flatten(A)



