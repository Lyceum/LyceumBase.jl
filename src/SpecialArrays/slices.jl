struct Slices{T,N,M,P<:AbsArr,A} <: AbstractArray{T,N}
    parent::P
    alongs::A
    function Slices{T,N,M,P,A}(parent, alongs) where {T,N,M,P<:AbsArr,A}
        check_slices_parameters(T, Val(N), Val(M), P, A)
        new(parent, alongs)
    end
end

#alongs(::Type{<:Slices{<:Any,<:Any,<:Any,<:Any,A}}) where {A} = A
#alongs(S::Slices) = alongs(typeof(S))

@generated function check_slices_parameters(::Type{T}, ::Val{N}, ::Val{M}, ::Type{P}, ::Type{A}) where {T,N,M,P,A}
    if !(N isa Int && M isa Int)
        return :(throw(ArgumentError("Slices paramters N and M must be of type Int")))
    elseif !(A <: NTuple{M+N,SBool})
        return :(throw(ArgumentError("Slices parameter A should be a NTuple{M+N,$SBool}")))
    elseif N < 0 || M < 0 || ndims(P) != N + M || sum(map(unwrap, A.parameters)) != M
        return :(throw(ArgumentError("Dimension mismatch in Slices parameters"))) # got N=$N, M=$M, ndims(P)=$(ndims(P)), and sum(A)=$(sum(A))")))
    else
        return nothing
    end
    error("Internal error. Please file a bug")
end

function Slices(parent::AbsArr{<:Any,L}, alongs::NTuple{L,SBool}) where {L}
    M = sum(map(unstatic, alongs))
    N = L - M
    parentaxes = axes(parent)
    I = ntuple(i -> (Base.@_inline_meta; _slice_or_firstindex(alongs[i], parentaxes[i])), Val(L))
    T = viewtype(parent, I)
    Slices{T,N,M,typeof(parent),typeof(alongs)}(parent, alongs)
end
@pure _slice_or_firstindex(::STrue, ax) = Colon()
@inline _slice_or_firstindex(::SFalse, ax) = first(ax)

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
end


####
#### Core Array Interface
####

# TODO
Base.parent(S::Slices) = error("Base.parent")

static_map(f, xs::NTuple{N,StaticOrVal}) where {N} = ntuple(i -> f(xs[i]), Val(N))

@inline function Base.size(S::Slices{<:Any,N}) where {N}
    #static_filter(ntuple(i -> static_not(S.alongs[i]), Val(L)), size(S.parent))
    static_filter(static_map(static_not, S.alongs), size(S.parent))
end

#_maybe_reshape(A::AbstractArray, ::NTuple{1, Bool}) = reshape(A, Val(1))
#_maybe_reshape(A::AbstractArray{<:Any,1}, ::NTuple{1, Bool}) = reshape(A, Val(1))
#_maybe_reshape(A::AbstractArray{<:Any,N}, ::NTuple{N, Bool}) where {N} = A
#_maybe_reshape(A::AbstractArray, ::NTuple{N, Bool}) where {N} = reshape(A, Val(N))
#
#Idx = Union{Colon, Real, AbstractArray}

# cases
# 1. N == ndims(I)
# 2. N > ndims(I)
# 3. N < ndims(I)

#function Base.getindex(S::Slices{<:Any,N}, I::Vararg{Idx,N}) where {N}

@propagate_inbounds function Base.getindex(S::Slices{<:Any,N}, I::Vararg{Int,N}) where {N}
    J = _parentindices(S, I)
    view(S.parent, J...)
end

@propagate_inbounds function Base.setindex!(S::Slices{<:Any,N}, v, I::Vararg{Int,N}) where {N}
    J = _parentindices(S, I)
    setindex!(S.parent, v, J...)
    return S
end

function _parentindices(S::Slices{<:Any,N,M}, I::NTuple{N,Any}) where {N,M}
    static_merge(S.alongs, ntuple(_ -> Colon(), Val(M)), I)
end

#@propagate_inbounds function Base.getindex(S::Slices, I::Idx...)
#    _getindex(_maybe_reshape(S, Base.index_ndims(I...)), I...)
#end

#@propagate_inbounds _getindex(S::Slices, I...) = getindex(Indexer(S), I...)
#
#function _getindex(S::Slices{<:Any,N}, I::Vararg{Idx,N}) where {N}
#    J = _parentindices(S, I...)
#    return J
#    al = reslice(S.alongs, I)
#    al = map(unstatic, al)
#    println("old: $(sum(S.alongs)) $(length(S.alongs))")
#    println("new: $(sum(al)) $(length(al))")
#end

#ScalarIndex = Union{Real, AbstractArray{<:Any, 0}}
#ScalarIndex = Real


#function reslice(alongs::Tuple{STrue, Vararg{Any}}, I::Tuple)
#    (static(true), reslice(tail(alongs), I)...)
#end
#function reslice(alongs::Tuple{SFalse, Vararg{Any}}, I::Tuple)
#    (_reslice_false(first(I))..., reslice(tail(alongs), tail(I))...)
#end
#function reslice(alongs::Tuple{SFalse, Vararg{Any}}, I::Tuple{})
#    ()
#end
#reslice(::Tuple{}, ::Tuple{}) = ()
#
## drop trailing
#reslice(::Tuple, ::Tuple{}) = ()
## add trailing
#reslice(::Tuple{}, I::Tuple) = _reslice_trailing(I...)
#_reslice_trailing(::ScalarIndex, I...) = _reslice_trailing(I...)
#_reslice_trailing(::Colon, I...) = (static(false), _reslice_trailing(I...)...)
#function _reslice_trailing(::AbstractArray{N}, I...) where {N}
#    (ntuple(_ -> static(false), Val(N))..., _reslice_trailing(I...)...)
#end
#_reslice_trailing() = ()
#
#_reslice_false(::ScalarIndex) = () # drop this dimension
#_reslice_false(::Colon) = (static(false), ) # keep this dimension
#function _reslice_false(::AbstractArray{<:Any,N}) where {N}
#    ntuple(_ -> static(false), Val(N))
#end

#function _reslice(alongs::NTuple{N,SBool}, J::NTuple{N,Any}) where {N}
#    __reslice(Base.indefirst(J), first(alongs), tail(J), tail(alongs))
#end
#__reslice(
#_reslice(::Tuple{}, ::Tuple{}) = ()
#_reslice(::NTuple{N,Any}, ::Tuple{}) where {N} = ntuple(_ -> static(false), Val(N))
#
#@inline _reslice_one(J, alongs::SBool) = __reslice_one(Base.index_dimsum(J), alongs)
#__reslice_one(::Tuple{}, ::SFalse) = ()
#__reslice_one(::NTuple{1,Bool}, ::SFalse) = (static(false), )
#__reslice_one(::NTuple{1,Bool}, ::STrue) = (static(true), )

#
## Now we can reaxis without worrying about mismatched axes/indices
#@inline _reaxis(axs::Tuple{}, idxs::Tuple{}) = ()
## Scalars are dropped
#@inline _reaxis(axs::Tuple, idxs::Tuple{ScalarIndex, Vararg{Any}}) = _reaxis(tail(axs), tail(idxs))
## Colon passes straight through
#@inline _reaxis(axs::Tuple, idxs::Tuple{Colon, Vararg{Any}}) = (axs[1], _reaxis(tail(axs), tail(idxs))...)
## But arrays can add or change dimensions and accompanying axis names
#@inline _reaxis(axs::Tuple, idxs::Tuple{AbstractArray, Vararg{Any}}) =
#    (_new_axes(axs[1], idxs[1])..., _reaxis(tail(axs), tail(idxs))...)

#reaxis(A::AxisArray, I::Idx...) = _reaxis(make_axes_match(axes(A), I), I)
## Linear indexing
#reaxis(A::AxisArray{<:Any,1}, I::AbstractArray{Int}) = _new_axes(A.axes[1], I)
#reaxis(A::AxisArray, I::AbstractArray{Int}) = default_axes(I)
#reaxis(A::AxisArray{<:Any,1}, I::Real) = ()
#reaxis(A::AxisArray, I::Real) = ()
#reaxis(A::AxisArray{<:Any,1}, I::Colon) = _new_axes(A.axes[1], Base.axes(A, 1))
#reaxis(A::AxisArray, I::Colon) = default_axes(Base.OneTo(length(A)))
#reaxis(A::AxisArray{<:Any,1}, I::AbstractArray{Bool}) = _new_axes(A.axes[1], findall(I))
#reaxis(A::AxisArray, I::AbstractArray{Bool}) = default_axes(findall(I))
#
## Ensure the number of axes matches the number of indexing dimensions
#@inline function make_axes_match(axs, idxs)
#    nidxs = Base.index_ndims(idxs...)
#    ntuple(i->(Base.@_inline_meta; _default_axis(i > length(axs) ? Base.OneTo(1) : axs[i], i)), length(nidxs))
#end
#

#@inline function _getindex(l::IndexStyle, A::AbstractArray, I::Union{Real, AbstractArray}...)
#    @info "here"
#    @boundscheck checkbounds(A, I...)
#    return _unsafe_getindex(l, Base._maybe_reshape(l, A, I...), I...)
#end
#
#function _unsafe_getindex(::IndexStyle, A::AbstractArray, I::Vararg{Union{Real, AbstractArray}, N}) where N
#    @info "there" typeof(A) size(A) length(I)
#    # This is specifically not inlined to prevent excessive allocations in type unstable code
#    shape = Base.index_shape(I...)
#    dest = similar(A, shape)
#    map(Base.unsafe_length, axes(dest)) == map(Base.unsafe_length, shape) || Base.throw_checksize_error(dest, shape)
#    @info size(dest) typeof(dest)
#    #Base._unsafe_getindex!(dest, A, I...) # usually a generated function, don't allow it to impact inference result
#    return dest
#
#    # This is specifically not inlined to prevent excessive allocations in type unstable code
#    #shape = Base.index_shape(I...)
#    #dest = similar(A, shape)
#    #map(Base.unsafe_length, axes(dest)) == map(Base.unsafe_length, shape) || Base.throw_checksize_error(dest, shape)
#    #_unsafe_getindex!(dest, A, I...) # usually a generated function, don't allow it to impact inference result
#    #dest, I
#    #return dest
#end
#
## a single element
#@propagate_inbounds function _getindex(S::Slices{<:Any,N}, J::NTuple{N,Any}, ::Tuple{}) where {N}
#    view(S.parent, J...)
#end
#
#@propagate_inbounds function _getindex(S::Slices{<:Any,N}, J::NTuple{N,Any}, ::Tuple) where {N}
#    Slice(view(S.parent, I...), _reslice(S.alongs, J))
#end
#
#@inline index_dimsum(i1, I...) = (index_dimsum(I...)...,)
#@inline index_dimsum(::Colon, I...) = (true, index_dimsum(I...)...)
#@inline index_dimsum(::AbstractArray{Bool}, I...) = (true, index_dimsum(I...)...)
#
#
#
#
#
#@propagate_inbounds function Base.setindex!(S::Slices{<:Any,N}, v, I::Vararg{Any,N}) where {N}
#    J = parentindices(S, I...)
#    IDims = Base.index_dimsum(I...)
#    #@info "IJ" I J size(parent(S)) size(v) size(first(v)) size(view(S.parent, J...))
#    _setindex!(S, v, I, J, IDims)
#end
#@propagate_inbounds function _setindex!(S::Slices, v, I, J::Tuple, ::NTuple{0,Bool})
#    setindex!(parent(S), v, J...)
#end
#@propagate_inbounds function _setindex!(S::Slices, v, I, J::Tuple, ::NTuple{N,Bool}) where {N}
#    #@info "IJ" I J size(parent(S)) size(v)
#    S2 = getindex(S, I...)
#    setindex!(S2, v, I...)
#    S
#    #Slices(view(parent(S), J...), _reslice(J, S.alongs))
#end
##@propagate_inbounds function _getindex(S::Slices{<:Any,N}, J::Tuple, ::NTuple{N,Bool}) where {N}
##    Slices(view(parent(S), J...), S.alongs)
##end
#
##@propagate_inbounds function _setindex!(S::Slices, v, I...)
##    setindex!(parent(S), _maybe_unsqueeze(S, v), parentindices(S, I...)...)
##    return S
##end
##@propagate_inbounds _setindex!(S::Slices, v, ::Colon) = (copyto!(S, v); S)

#
#
#
## TODO support IndexLinear()?
#Base.IndexStyle(::Type{<:Slices}) = IndexCartesian()
#
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


#Base.reshape(S::Slices{<:Any,N}, ::Val{N}) where {N} = S
#Base.reshape(S::Slices, ::Val{N}) where {N} = _reshape(S, Val(N))
#@generated function _reshape(S::Slices{<:Any,N,M,<:Any,A}, ::Val{NewN}) where {N,M,A,NewN}
#    L = N + M
#    diff = NewN - N
#    if NewN > N
#        alongs = Tuple(a() for a in A.parameters)..., ntuple(_ -> static(false), NewN - N)...)
#        I = Expr(:tuple)
#        quote
#            @_inline_meta
#            parent = reshape(S.parent, Val($(M + NewN)))
#            #parentaxes = axes(parent)
#            #T = viewtype(parent, $I...)
#            #T = typeof(view(parent, $I...))
#            #Slices{T,$NewN,$M,typeof(parent),$(typeof(alongs))}(parent, $alongs)
#
#            Slices(parent, $alongs)
#            #slices(newparent, $newalongs, Val($NewN), Val($M))
#        end
#    else
#        return :(@_inline_meta; reshape(S, Base.rdims(Val($NewN), axes(S))))
#    end
#end


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