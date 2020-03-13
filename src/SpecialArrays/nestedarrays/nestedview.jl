#struct NestedView{M,T,N,S<:AbsArr{T,M},L} <: AbstractArray{T,N}
struct NestedView{M,T,N,S,L} <: AbstractArray{T,N}
    slices::S
    function NestedView{M}(parent::AbsArr{<:Any,L}) where {M,L}
        check_nestedarray_parameters(Val(M), typeof(parent))
        N = L - M
        inner = ntuple(_ -> True(), Val(M))
        outer = ntuple(_ -> False(), Val(N))
        slices = Slices(parent, inner..., outer...)
        T = eltype(slices)
        new{M,T,N,typeof(slices),IndexStyle(slices)}(slices)
    end
end


####
#### Core Array Interface
####

const SlowNestedView{M,T,N,S} = NestedView{M,T,N,S,IndexCartesian()}
const FastNestedView{M,T,N,S} = NestedView{M,T,N,S,IndexLinear()}

@inline Base.size(A::NestedView) = size(A.slices)


# TODO _maybe_wrap ?
@propagate_inbounds function Base.getindex(A::SlowNestedView{<:Any,<:Any,N}, I::Vararg{Any,N}) where {N}
    #_maybe_wrap(A, getindex(A.slices, I...))
    getindex(A.slices, I...)
end
@propagate_inbounds Base.getindex(A::FastNestedView, i::Int) = getindex(A.slices, i)
#@propagate_inbounds function Base.getindex(A::FastNestedView, i::Int)
#    #_maybe_wrap(A, getindex(A.slices, i))
#    getindex(A.slices, i)
#end


@inline Base.getindex(A::SlowNestedView, ::Colon) = _getindex_colon(A)
@inline Base.getindex(A::FastNestedView, ::Colon) = _getindex_colon(A)
# To resolve method amiguities
@inline Base.getindex(A::SlowNestedView{<:Any,<:Any,1}, ::Colon) = _getindex_colon(A)
@inline Base.getindex(A::FastNestedView{<:Any,<:Any,1}, ::Colon) = _getindex_colon(A)

function _getindex_colon(A::NestedView{M}) where {M}
    NestedView{M}(reshape(copy(parent(A)), Val(M + 1)))
end


#@inline function Base.getindex(A::FastNestedView{M}, c::Colon) where {M}
#    NestedView{M}(reshape(copy(parent(A)), Val(M + 1)))
#end

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


@inline function Base.similar(A::NestedView, T::Type{<:AbsArr}, dims::Dims)
    NestedView{ndims(T)}(similar(parent(A), eltype(T), innersize(A)..., dims...))
end

@inline Base.axes(A::NestedView) = axes(A.slices)


####
#### Misc
####

@inline function Base.:(==)(A::NestedView, B::NestedView)
    size(A) == size(B) && innersize(A) == innersize(B) && parent(A) == parent(B)
end

@inline Base.parent(A::NestedView) = parent(A.slices)

@inline Base.copyto!(dest::NestedView, src::NestedView) = copyto!(dest.slices, src.slices)
@inline Base.copyto!(dest::AbsArr{<:AbsArr}, src::NestedView) = nest!(dest, parent(src))
@inline Base.copyto!(dest::NestedView, src::AbsArr) = flatten!(parent(dest), src)

#function Base.deepcopy(A::NestedView{M,T,N,P}) where {M,T,N,P}
#    NestedView{M,T,N,P}(deepcopy(A.parent))
#end


@inline function Base.resize!(A::NestedView{<:Any,<:Any,N}, dims::NTuple{N,Integer}) where {N}
    resize!(parent(A), innersize(A)..., dims...)
    return A
end
@inline Base.resize!(A::NestedView, dims...) = resize!(A, dims)

#@propagate_inbounds function Base.view(A::NestedView{M,<:Any,N}, I::Vararg{Any, N}) where {M,N}
#    @boundscheck checkbounds(A, I...)
#    @inbounds NestedView{M}(view(A.parent, _flat_indices(A, I)...))
#end


function Base.reshape(A::NestedView{M}, ::Val{N}) where {M,N}
    newparent = reshape(parent(A), Val(M+N))
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

#function Base.append!(::NestedView{<:Any,<:Any,0}, ::AbsArr{<:AbsArr})
#    throw(ArgumentError("Cannot append to a zero-dimensional Slice"))
#end


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
    innerview(A::AbstractArray{M+N}, M::Integer)

View array `A` as an `N`-dimensional array of `M`-dimensional arrays.
See also: [`outerview`](@ref).
"""
innerview(A::AbsArr, M::Integer) = NestedView{M}(A)


"""
    outerview(A::AbstractArray{M+N}, N::Integer)

View array `A` as an `N`-dimensional array of `M`-dimensional arrays.
See also: [`innerview`](@ref).
"""
outerview(A::AbsArr, N::Integer) = NestedView{ndims(A) - N}(A)

"""
    flatview(A::NestedView{M,T,N,P}) --> Array{eltype(T),M+N}

Returns the array of dimensionality `M + N` wrapped by `A`. The shape of
the result may be freely changed without breaking the inner consistency of `A`.
"""
@inline flatview(A::NestedView) = parent(A)

@inline innersize(A::NestedView) = innersize(A.slices)

@inline inneraxes(A::NestedView) = inneraxes(A.slices)


####
#### 3rd Party
####

@inline function UnsafeArrays.unsafe_uview(A::NestedView{M}) where {M}
    NestedView{M}(uview(A.slices))
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
