struct NestedView{M,T,N,S,L} <: AbstractArray{T,N}
    slices::S
    function NestedView{M}(parent::AbsArr{<:Any,L}) where {M,L}
        check_nestedarray_parameters(Val(M), typeof(parent))
        N = L - M
        inner = ntuple(_ -> True(), Val(M))
        outer = ntuple(_ -> False(), Val(N))
        slices = Slices(parent, inner..., outer...)
        T = eltype(slices)
        new{unstatic(M),T,N,typeof(slices),IndexStyle(slices)}(slices)
    end
end


####
#### Core Array Interface
####

const SlowNestedView{M,T,N,S} = NestedView{M,T,N,S,IndexCartesian()}
const FastNestedView{M,T,N,S} = NestedView{M,T,N,S,IndexLinear()}

@inline Base.size(A::NestedView) = size(A.slices)
@inline Base.size(A::NestedView, d::Integer) = size(A.slices, d)

@propagate_inbounds function Base.getindex(A::SlowNestedView{<:Any,<:Any,N}, I::Vararg{Any,N}) where {N}
    getindex(A.slices, I...)
end
@propagate_inbounds Base.getindex(A::FastNestedView, i::Int) = getindex(A.slices, i)

@inline Base.getindex(A::SlowNestedView, ::Colon) = _getindex_colon(A)
@inline Base.getindex(A::FastNestedView, ::Colon) = _getindex_colon(A)
# To resolve method amiguities
@inline Base.getindex(A::SlowNestedView{<:Any,<:Any,1}, ::Colon) = _getindex_colon(A)
@inline Base.getindex(A::FastNestedView{<:Any,<:Any,1}, ::Colon) = _getindex_colon(A)
function _getindex_colon(A::NestedView{M}) where {M}
    NestedView{M}(reshape(copy(parent(A)), Val(M + 1)))
end

@propagate_inbounds function Base.setindex!(A::SlowNestedView{<:Any,<:Any,N}, v, I::Vararg{Any,N}) where {N}
    setindex!(A.slices, v, I...)
    return A
end
@propagate_inbounds function Base.setindex!(A::FastNestedView, v, i::Int)
    setindex!(A.slices, v, i)
    return A
end

@inline Base.IndexStyle(::Type{<:FastNestedView}) = IndexLinear()
@inline Base.IndexStyle(::Type{<:SlowNestedView}) = IndexCartesian()

@inline function Base.similar(A::NestedView{M}, T::Type{<:AbsArr{V}}, dims::Dims{N}) where {V,M,N}
    ndims(T) == M || throw(ArgumentError("ndims(T) must equal innerdims(A)"))
    NestedView{M}(similar(parent(A), eltype(T), innersize(A)..., dims...))
end

@inline Base.axes(A::NestedView) = axes(A.slices)


####
#### Misc
####


@inline function Base.:(==)(A::NestedView, B::NestedView)
    size(A) == size(B) && innersize(A) == innersize(B) && parent(A) == parent(B)
end

@inline Base.parent(A::NestedView) = parent(A.slices)

# much faster than default copyto!
@inline function Base.copyto!(dest::NestedView, src::NestedView)
    copyto!(dest.slices, src.slices)
    return dest
end
@inline Base.copyto!(dest::AbsNestedArr, src::NestedView) = copyto!(dest, src.slices)
@inline function Base.copyto!(dest::NestedView, src::AbsNestedArr)
    copyto!(dest.slices, src)
    return dest
end


@inline function Base.resize!(A::NestedView{<:Any,<:Any,N}, dims::NTuple{N,Integer}) where {N}
    resize!(parent(A), innersize(A)..., dims...)
    return A
end
@inline Base.resize!(A::NestedView, dims...) = resize!(A, dims)


function Base.reshape(A::NestedView{M}, ::Val{N}) where {M,N}
    newparent = reshape(parent(A), Val(M+N))
    NestedView{M}(newparent)
end

function Base.reshape(A::NestedView{M,<:Any}, dims::Dims{N}) where {M,N}
    newparent = reshape(parent(A), innersize(A)..., dims...)
    NestedView{M}(newparent)
end

function Base.append!(dest::NestedView, src::NestedView)
    if ndims(dest) != ndims(src) || innersize(dest) != innersize(src)
        throw(ArgumentError("Both `ndims` and `innersize` of dest & src must match"))
    end
    append!(parent(dest), parent(src))
    return dest
end

function Base.append!(dest::NestedView, src::AbsArr{<:AbsArr})
    if ndims(dest) != ndims(src) || innersize(dest) != innersize(src)
        throw(ArgumentError("Both `ndims` and `innersize` of dest & src must match"))
    end
    for x in src
        append!(parent(dest), x)
    end
    return dest
end



const NestedVector{M,T,S,L} = NestedView{M,T,1,S,L}


@inline function Base.push!(A::NestedVector, x)
    innersize(A) == size(x) || throw(DimensionMismatch("innersize(A) != size(x)"))
    append!(parent(A), x)
    return A
end

# TODO
@inline function Base.push!(A::NestedVector{0}, x::AbsArr{<:Any,0}) where {M}
    innersize(A) == size(x) || throw(DimensionMismatch("innersize(A) != size(x)"))
    append!(parent(A), x[])
    return A
end


####
#### SpecialArrays functions
####

"""
    nestedview(A::AbstractArray{M+N}, MorN::Integer; inner::StaticOrBool = static(true))

View array `A` as an either a `N`-dimensional array of `M`-dimensional arrays or
a `M`-dimensional array of `N`-dimensional arrays, as determined by `inner`.
See also: [StaticNumbers.jl](https://github.com/perrutquist/StaticNumbers.jl).
"""
@inline function nestedview(A::AbsArr{<:Any,L}, MorN::Integer; inner::StaticOrBool = static(true)) where {L}
    M = unstatic(inner) ? MorN : L - MorN
    NestedView{M}(A)
end

"""
    flatview(A::NestedView{M,T,N})

Returns the array of dimensionality `M + N` wrapped by `A`.
"""
@inline flatview(A::NestedView) = parent(A)

@inline inneraxes(A::NestedView) = inneraxes(A.slices)

@inline innersize(A::NestedView) = innersize(A.slices)



####
#### 3rd Party
####

@inline function UnsafeArrays.unsafe_uview(A::NestedView{M}) where {M}
    NestedView{M}(uview(parent(A)))
end


####
#### Util
####

# Need to unpack our zero-dimensional views first
@inline _maybe_unsqueeze(x::AbsArr{<:Any,0}) = x[]
@inline _maybe_unsqueeze(x) = x
# TODO
@inline _maybe_wrap(A::NestedView{M}, B::AbsArr{<:Any,M}) where {M} = B
@inline _maybe_wrap(A::NestedView{M}, B::AbsArr) where {M} = NestedView{M}(B)
