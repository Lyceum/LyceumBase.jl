"""
    flatten(A::AbstractArray)
    flatten(A::AbstractArray{<:AbstractArray{U,M},N}

Flatten `A` into an Array{U,M+N}. Fails if the elements of `A`
do not all have the same size.
If the `A` is not a nested array, the return value is `A` itself.
"""
function flatten end

@inline flatten(A::AbsArr) = A

@inline function flatten(A::AbsArr{<:AbsArr{U,M},N}) where {U,M,N}
    L = add(Val(M), Val(N))
    sz_inner = inner_size(A)
    A_flat = Array{U,L}(undef, sz_inner..., size(A)...)
    unsafe_flattento!(A_flat, A, sz_inner)
end

@inline function flattento!(A::AbsArr{<:Any, L}, B::AbsArr{<:AbsArr{<:Any,M},N}) where {L,M,N}
    L == add(Val(M), Val(N)) || throw(ArgumentError("ndims(A) != inner_ndims(B)"))
    sz_inner = inner_size(B)
    inner_size(A) == sz_inner || throw(DimensionMismatch("inner_size(A) != inner_size(B)"))
    unsafe_flattento!(A, B, sz_inner)
end

function unsafe_flattento!(A::AbsArr, B::AbsArr{<:AbsArr{<:Any,M}}, sz_inner::Dims{M}) where {M}
    len_inner = prod(sz_inner)
    from = firstindex(A)
    for b in B
        copyto!(A, from, b, firstindex(b), len_inner)
        from += len_inner
    end
    @assert from  == len_inner * length(B) + 1
    A
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
# TODO: Implement flatview on generic nested arrays via new `FlatView`, using
# deepgetindex to implement getindex, etc.
@inline flatview(A::AbsArr{<:AbsArr}) = flatten(A)


"""
    inner_eltype(A::AbstractArray{<:AbstractArray{T}})
    inner_eltype(::Type{<:AbstractArray{<:AbstractArray{T}}})

Returns `T`, the element type of the element arrays of `A`.
"""
function inner_eltype end

@inline inner_eltype(::Type{<:AbsArr{<:AbsArr{T}}}) where {T} = T
@inline inner_eltype(A::AbsArr) = inner_eltype(typeof(A))


"""
    inner_ndims(A::AbstractArray{<:AbstractArray{T,N}})
    inner_ndims(::Type{<:AbstractArray{<:AbstractArray{T,N}}})

Returns `N`, the dimensionality of the element arrays of `A`.
"""
function inner_ndims end

@inline inner_ndims(::Type{<:AbsArr{<:AbsArr{<:Any,N}}}) where {N} = N
@inline inner_ndims(A::AbsArr) = inner_ndims(typeof(A))


"""
    inner_size(A::AbstractArray{<:AbstractArray}[, d])

Returns the size of the element arrays of `A`.
Fails if the elements of A are not of equal size.
"""
function inner_size end

function inner_size(A::AbsArr{<:AbsArr{<:Any,M}}) where {M}
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

@inline function inner_size(A::AbsArr, d::Integer)
    sz = inner_size(A)
    d <= length(sz) ? sz[d] : 0 # TODO offset arrays
end


"""
    inner_length(A::AbstractArray{<:AbstractArray})

Returns the common length of the element arrays of `A`.
Fails if the element arrays do not have equal size.
"""
function inner_length end

@inline inner_length(A::AbsArr{<:AbsArr}) = prod(inner_size(A))


"""
    inner_axes(A::AbstractArray{<:AbstractArray}[, d])

Returns the common length of the element arrays of `A`.
Fails if the element arrays do not have equal axes.
"""
function inner_axes end

function inner_axes(A::AbsArr{<:AbsArr})
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

@inline function inner_axes(A::AbsArr, d::Integer)
    ax = inner_axes(A)
    d <= length(ax) ? ax[d] : Base.OneTo(1) # TODO offset arrays
end
