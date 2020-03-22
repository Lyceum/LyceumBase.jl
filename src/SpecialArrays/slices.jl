struct Slices{T,N,M,P<:AbsArr,A} <: AbstractArray{T,N}
    parent::P
    alongs::A
    function Slices{T,N,M,P,A}(parent, alongs) where {T,N,M,P<:AbsArr,A}
        check_slices_parameters(T, Val(N), Val(M), P, A)
        new(parent, alongs)
    end
end

@inline function Slices(parent::AbsArr{<:Any,L}, alongs::NTuple{L,SBool}, inaxes::NTuple{M,Any}, outaxes::NTuple{N,Any}) where {L,M,N}
    ax = axes(parent)
    I = ntuple(i -> first(outaxes[i]), Val(N))
    J = static_merge(alongs, ntuple(i -> Base.Slice(inaxes[i]), Val(M)), I)
    T = viewtype(parent, J...)
    Slices{T,N,M,typeof(parent),typeof(alongs)}(parent, alongs)
end

@inline function Slices(parent::AbsArr{<:Any,L}, alongs::NTuple{L,SBool}) where {L}
    paxes = axes(parent)
    inaxes = static_filter(STrue(), alongs, paxes)
    outaxes = static_filter(SFalse(), alongs, paxes)
    Slices(parent, alongs, inaxes, outaxes)
end

function Slices(parent::AbsArr, alongs::TupleN{SBool})
    throw(ArgumentError("length(alongs) != ndims(parent)"))
end

@inline function Slices(parent::AbsArr{<:Any,L}, alongs::Vararg{SBool,L}) where {L}
    Slices(parent, alongs)
end

@inline function Slices(parent::AbsArr{<:Any,L}, alongs::TupleN{StaticOrInt}) where {L}
    Slices(parent, ntuple(dim -> static_in(dim, alongs), Val(L)))
end

@inline function Slices(parent::AbsArr{<:Any,L}, alongs::Vararg{StaticOrInt}) where {L}
    Slices(parent, alongs)
end#


####
#### Core Array Interface
####

# TODO
Base.parent(S::Slices) = error("Base.parent")

@inline Base.axes(S::Slices) = static_filter(SFalse(), S.alongs, axes(S.parent))
@inline Base.size(S::Slices) = static_filter(SFalse(), S.alongs, size(S.parent))

@inline inneraxes(S::Slices) = static_filter(STrue(), S.alongs, axes(S.parent))
@inline innersize(S::Slices) = static_filter(STrue(), S.alongs, size(S.parent))

@pure function parentindices(S::Slices{<:Any,N,M}, I::NTuple{N,Any}) where {N,M}
    inaxes = inneraxes(S)
    slices = ntuple(i -> (Base.@_inline_meta; Base.Slice(inaxes[i])), Val(M))
    static_merge(S.alongs, slices, I)
end

# standard cartesian indexing
@propagate_inbounds function Base.getindex(S::Slices{<:Any,N}, I::Vararg{Int,N}) where {N}
    J = parentindices(S, I)
    view(S.parent, J...)
end
@propagate_inbounds function Base.setindex!(S::Slices{<:Any,N}, v, I::Vararg{Int,N}) where {N}
    J = parentindices(S, I)
    setindex!(S.parent, v, J...)
    return S
end


# for I::Vararg{SliceIdx}, we might be able to just forward the indices to the parent array
const SliceIdx = Union{Colon, Real, AbstractArray}

#@propagate_inbounds function Base.getindex(S::Slices, I::SliceIdx...)
@propagate_inbounds function Base.getindex(S::Slices{<:Any,N}, I::Vararg{SliceIdx,N}) where {N}
    _getindex(_maybe_reshape(S, Base.index_ndims(I...)), I)
end

_maybe_reshape(A::AbstractArray, ::NTuple{1, Bool}) = reshape(A, Val(1))
_maybe_reshape(A::AbstractArray{<:Any,1}, ::NTuple{1, Bool}) = reshape(A, Val(1))
_maybe_reshape(A::AbstractArray{<:Any,N}, ::NTuple{N, Bool}) where {N} = A
_maybe_reshape(A::AbstractArray, ::NTuple{N, Bool}) where {N} = reshape(A, Val(N))

@propagate_inbounds function _getindex(S::Slices{<:Any,N}, I::NTuple{N,SliceIdx}) where {N}
    # If ndims(S) == length(I) after reshaping, we can forward the indices to the
    # parent array and drop the corresponding values in S.alongs.
    J = parentindices(S, I)
    _maybe_reslice(S, J, reslice(S.alongs, J))
end

function _maybe_reslice(S::Slices, J, newalongs::Tuple)
    Slices(view(S.parent, J...), newalongs)
end
function _maybe_reslice(S::Slices{<:Any,<:Any,M}, J, ::NTuple{M,SBool}) where {M}
    view(S.parent, J...)
end

# add/drop non-sliced dimensions (i.e. alongs[dim] == SFalse()) to match J
@inline function reslice(alongs::NTuple{L,SBool}, J::NTuple{L,Any}) where {L}
    (_reslice1(first(alongs), first(J))..., reslice(tail(alongs), tail(J))...)
end
reslice(::Tuple{}, ::Tuple{}) = ()
@inline _reslice1(::STrue, ::Any) = (static(true), ) # keep inner dimension
@inline _reslice1(::SFalse, j::Any) = _reslicefalse(j)
@inline _reslicefalse(::Real) = () # drop this dimension
@inline _reslicefalse(::Colon) = (static(false), ) # keep this dimension
@inline function _reslicefalse(::AbstractArray{<:Any,N}) where {N}
    ntuple(_ -> static(false), Val(N))
end

# Fall back to Cartesian indexing if:
#   1) ndims(I) != length(I)
#   2) _maybe_reshape returned did not return a Slices
@propagate_inbounds _getindex(S::Slices, I) = getindex(Indexer(A), I...)
@propagate_inbounds _getindex(A::AbstractArray, I) = getindex(Indexer(A), I...)

Base.IndexStyle(::Type{<:Slices}) = IndexCartesian()

## TODO always shove inner dims to front?
#function Base.similar(S::Slices{<:Any,<:Any,M1}, ::Type{<:AbsArr{V,M2}}, dims::Dims) where {M1,V,M2}
#    M1 == M2 || throw(ArgumentError("ndims(T) must equal ndims(S) or unspecified"))
#    _similar(S, V, dims)
#end
#Base.similar(S::Slices, ::Type{<:AbsArr{V}}, dims::Dims) where {V} = _similar(S, V, dims)
#function Base.similar(::Slices, ::Type{<:AbsArr}, ::Dims)
#    throw(ArgumentError("$T has no element type"))
#end
#
#Base.similar(S::Slices, T::Type, dims::Dims) = _similar(S, T, dims)
#
#function _similar(S::Slices{<:Any,<:Any,M}, ::Type{V}, dims::Dims{N}) where {M,V,N}
#    newparent = similar(parent(S), V, innersize(S)..., dims...)
#    newalongs = (ntuple(_ -> static(true), Val(M))..., ntuple(_ -> static(false), Val(N))...)
#    Slices(newparent, newalongs)
#end
#
#@inline Base.axes(S::Slices) = static_filter(map(static_not, S.alongs), axes(parent(S)))

Base.reshape(S::Slices{<:Any,N}, ::Val{N}) where {N} = S
@generated function Base.reshape(S::Slices{<:Any,N,M,<:Any,A}, ::Val{NewN}) where {N,M,A,NewN}
    if NewN > N
        newalongs = (Tuple(a() for a in A.parameters)..., ntuple(_ -> static(false), NewN - N)...)
        quote
            @_inline_meta
            newparent = reshape(S.parent, Val($(M + NewN)))
            Slices(newparent, $newalongs)
        end
    else
        return :(@_inline_meta; reshape(S, Base.rdims(Val($NewN), axes(S))))
    end
end


#
#
#####
##### Misc
#####
#
## TODO special case when leading slices (e.g. NestedView) / support prepend!/append!/push!
#
## TODO
##@inline Base.:(==)(A::Slices, B::Slices) = A.alongs == B.alongs && parent(A) == parent(B)
#
#
#@inline Base.parent(S::Slices) = S.parent
#
#@inline Base.dataids(S::Slices) = Base.dataids(parent(S))
#
#
## TODO copyto with absarr
#function Base.copyto!(dest::Slices{<:Any,N,M,<:Any,A}, src::Slices{<:Any,N,M,<:Any,A}) where {N,M,A}
#    axes(parent(dest)) == axes(parent(src)) || throwdm(axes(parent(dest)), axes(parent(src)))
#    copyto!(parent(dest), parent(src))
#    return dest
#end
#
#function Base.copyto!(dest::AbsArr, src::Slices)
#    destinds, srcinds = LinearIndices(dest), LinearIndices(src)
#    isempty(srcinds) || (checkbounds(Bool, destinds, first(srcinds)) && checkbounds(Bool, destinds, last(srcinds))) ||
#        throw(BoundsError(dest, srcinds))
#    @inbounds for i in srcinds
#        # TODO can't copyto! because collect(::Slices) will try to copy to
#        # but below currently allocates
#        #copyto!(dest[i], src[i])
#        dest[i] = src[i]
#    end
#    return dest
#end
#
#
#
Base.copy(S::Slices) = Slices(copy(S.parent), S.alongs)
#
#

#
#
#####
##### SpecialArrays functions
#####
#
#@inline flatten(S::Slices) = copy(parent(S))
#
#@inline flatview(S::Slices) = parent(S)
#
#@inline innersize(S::Slices) = static_filter(S.alongs, size(parent(S)))
#
#@inline inneraxes(S::Slices) = static_filter(S.alongs, axes(parent(S)))
#
#slicedims(S::Slices) = slicedims(S.alongs)
#function slicedims(alongs::NTuple{N,SBool}) where {N}
#    static_filter(alongs, ntuple(identity, static(N)))
#end
#
#
#####
##### 3rd Party
#####
#
#@inline UnsafeArrays.unsafe_uview(S::Slices) = Slices(uview(parent(S)), S.alongs)
#
#
#####
##### Util
#####
#
#@noinline function throwdm(axdest, axsrc)
#    throw(DimensionMismatch("destination axes $axdest are not compatible with source axes $axsrc"))
#end
#
#
#@inline _maybe_unsqueeze(S::Slices{<:Any,<:Any,0}, v::AbsArr{<:Any,0}) = v[]
#@inline _maybe_unsqueeze(S::Slices, v) = v

@generated function check_slices_parameters(::Type{T}, ::Val{N}, ::Val{M}, ::Type{P}, ::Type{A}) where {T,N,M,P,A}
    if !(N isa Int && M isa Int)
        return :(throw(ArgumentError("Slices paramters N and M must be of type Int")))
    elseif !(A <: NTuple{M+N,SBool})
        return :(throw(ArgumentError("Slices parameter A should be of type NTuple{M+N,$SBool}")))
    elseif N < 0 || M < 0 || ndims(P) != N + M || sum(unwrap, A.parameters) != M
        return :(throw(ArgumentError("Dimension mismatch in Slices parameters"))) # got N=$N, M=$M, ndims(P)=$(ndims(P)), and sum(A)=$(sum(A))")))
    else
        return nothing
    end
    error("Internal error. Please file a bug")
end

