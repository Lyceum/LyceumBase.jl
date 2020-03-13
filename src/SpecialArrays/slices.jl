# TODO _inner_axes size etc
# TODO LyceumBase --
export True
export False


abstract type TypedBool end

struct True <: TypedBool end

struct False <: TypedBool end


@inline untyped(::True) = true
@inline untyped(::False) = false

@inline not(::False) = True()
@inline not(::True) = False()

const TupleN{T,N} = NTuple{N,T}
const AbsArr{T,N} = AbstractArray{T,N}

# --------

struct Slices{T,N,P,A} <: AbstractArray{T,N}
    parent::P
    alongs::A
    function Slices{T,N,P,A}(parent, alongs) where {T,N,P,A}
        # TODO check type parameters
        new(parent, alongs)
    end
end

@propagate_inbounds function Slices(parent::P, alongs::A) where {P,A}
    T = viewtype(parent, map(_slice_or_firstindex, alongs, axes(parent)))
    N = bool_sum(map(not, alongs)...)
    Slices{T,N,P,A}(parent, alongs)
end

@propagate_inbounds Slices(parent, alongs::TypedBool...) = Slices(parent, alongs)

@propagate_inbounds function Slices(parent::AbsArr{<:Any,N}, alongs::Dims) where {N}
    alongs = map(dim -> dim in alongs ? True() : False(), ntuple(identity, Val(N)))
    Slices(parent, alongs)
end

@propagate_inbounds function Slices(parent, alongs::Int...)
    Slices(parent, alongs)
end

@propagate_inbounds function Slices(parent::AbsArr{<:Any,N}) where {N}
    Slices(parent, ntuple(_ -> False(), Val(N)))
end


@inline Base.size(S::Slices) = map(length, axes(S))

@propagate_inbounds function Base.getindex(S::Slices{<:Any,N}, I::Vararg{Any,N}) where {N}
    view(parent(S), parentindices(S, I)...)
end

@propagate_inbounds function Base.setindex!(S::Slices{<:Any,N}, v, I::Vararg{Any,N}) where {N}
    setindex!(parent(S), _maybe_unsqueeze(S, v), parentindices(S, I)...)
    return S
end

Base.IndexStyle(::Type{<:Slices}) = IndexCartesian()

@inline Base.axes(S::Slices) = bool_filter(map(not, S.alongs), axes(parent(S)))

@inline Base.parent(S::Slices) = S.parent

@inline Base.dataids(S::Slices) = Base.dataids(parent(S))


@inline UnsafeArrays.unsafe_uview(S::Slices) = Slices(uview(parent(S)), S.alongs)


@propagate_inbounds function Base.parentindices(S::Slices, I...)
    parentindices(S, I)
end

@propagate_inbounds function Base.parentindices(S::Slices, i::Int)
    parentindices(S, Base._ind2sub(axes(S), i))
end

@propagate_inbounds function Base.parentindices(S::Slices, I::CartesianIndex)
    parentindices(S, Tuple(I))
end

@inline function Base.parentindices(S::Slices{<:Any,N}, I::NTuple{N}) where {N}
    # TODO can't checkbounds here if parent(S) has non-standard indexing e.g. AxisArrays
    # results in "ArgumentError: unable to check bounds for indices of type Symbol"
    #@boundscheck checkbounds(S, I...)
    _parentindices(axes(parent(S)), map(not, S.alongs), I)
end


@inline function _parentindices(parentaxes::Tuple, alongs::Tuple, I::Tuple)
    if untyped(first(alongs))
        (first(I), _parentindices(tail(parentaxes), tail(alongs), tail(I))...)
    else
        (Base.Slice(first(parentaxes)), _parentindices(tail(parentaxes), tail(alongs), I)...)
    end
end

_parentindices(parentaxes::Tuple{}, alongs::Tuple{}, I::Tuple{}) = ()



@inline function _slice_or_firstindex(switch::TypedBool, ax)
    untyped(switch) ? Base.Slice(ax) : first(ax)
end

@inline _maybe_unsqueeze(S::Slices{<:AbsArr{<:Any,0}}, v::AbsArr{<:Any,0}) = v[]
@inline _maybe_unsqueeze(S::Slices, v) = v


## TODO move back to SpecialArrays
#@propagate_inbounds viewtype(A::AbsArr, I...) = viewtype(A, I)
#@propagate_inbounds function viewtype(A::AbsArr, I::Tuple)
#    T = viewtype(typeof(A), typeof(I))
#    _viewtype(A, I, T, Val(isconcretetype(T)))
#end
#
#@inline function viewtype(::Type{A}, ::Type{I}) where {A<:AbsArr,I<:Tuple}
#    Core.Compiler.return_type(view, _view_signature(A, I))
#end
#@pure _view_signature(::Type{A}, ::Type{I}) where {A,I<:Tuple} = Tuple{A, I.parameters...}
#
#_viewtype(A, I, T, ::Val{true}) = T
#@propagate_inbounds _viewtype(A, I, T, ::Val{false}) = typeof(view(A, I...))


@inline bool_sum(xs::TypedBool...) = length(_bool_sum(xs...))
@inline _bool_sum(::True, xs::TypedBool...) = (True(), _bool_sum(xs...)...)
@inline _bool_sum(::False, xs::TypedBool...) = (_bool_sum(xs...)...,)
_bool_sum() = ()

@inline function bool_filter(switches::Tuple, xs::Tuple)
    (bool_filter1(first(switches), first(xs))..., bool_filter(tail(switches), tail(xs))...)
end
bool_filter(::Tuple{}, ::Tuple{}) = ()
@inline bool_filter1(::True, x) = (x, )
@inline bool_filter1(::False, ::Any) = ()
# --
