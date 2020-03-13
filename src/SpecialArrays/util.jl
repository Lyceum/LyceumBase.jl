@pure _ncolons(::Val{N}) where N = ntuple(_ -> Colon(), Val(N))
@inline function ncolons(::Val{N}) where N
    isa(N, Int) || throw(ArgumentError("Expected an Int, got N = $N"))
    _ncolons(Val(N))
end

const LTuple{L} = Tuple{Vararg{<:Any,L}}

@inline function front_tuple(x::LTuple{L}, ::Val{M}) where {L,M}
    ntuple(i -> x[i], Val(M))
end

@inline function back_tuple(x::LTuple{L}, ::Val{N}) where {L,N}
    M = L - N
    ntuple(i -> x[i + M], Val(N))
end

@inline function split_tuple(x::LTuple{L}, ::Val{M}) where {L,M}
    N = L - M
    front_tuple(x, Val(M)), back_tuple(x, Val(N))
end

@inline front(x::LTuple{L}) where {L} = front_tuple(x, Val(L - 1))
@inline frontlast(x::LTuple{L}) where {L} = front(x), last(x)

@inline back(x::LTuple{L}) where {L} = back_tuple(x, Val(L - 1))
@inline firstback(x::LTuple{L}) where {L} = first(x), back(x)

@pure _tuple_length(T::Type{<:Tuple}) = length(T.parameters)
@pure _tuple_length(T::Tuple) = length(typeof(T))
tuple_length(T::Tuple) = _tuple_length(T)