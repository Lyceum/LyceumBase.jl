@pure _ncolons(::Val{N}) where N = ntuple(_ -> Colon(), Val(N))
@inline function ncolons(::Val{N}) where N
    isa(N, Int) || throw(ArgumentError("Expected an Int, got N = $N"))
    _ncolons(Val(N))
end


@inline function front_tuple(x::NTuple{L}, ::Val{M}) where {L,M}
    ntuple(i -> x[i], Val(M))
end

@inline function back_tuple(x::NTuple{L}, ::Val{N}) where {L,N}
    M = L - N
    ntuple(i -> x[i + M], Val(N))
end

@inline function split_tuple(x::NTuple{L}, ::Val{M}) where {L,M}
    N = L - M
    front_tuple(x, Val(M)), back_tuple(x, Val(N))
end

@inline frontlast(x::NTuple{L}) where {L} = front_tuple(x, Val(L - 1)), last(x)

@inline firstback(x::NTuple{L}) where {L} = first(x), back_tuple(x, Val(L - 1))

@pure _tuple_length(T::Type{<:Tuple}) = length(T.parameters)
@pure _tuple_length(T::Tuple) = length(typeof(T))
tuple_length(T::Tuple) = _tuple_length(T)
