### To base
using Base: @propagate_inbounds, @pure, front, last

@pure _add(::Val{M}, ::Val{N}) where {M,N} = M + N
@inline function add(::Val{M}, ::Val{N}) where {M,N}
    isa(M, Int) && isa(N, Int) || throw(ArgumentError("Expected M and N to be Ints, got M = $M, N = $N"))
    _add(Val(M), Val(N))
end

@pure _sub(::Val{M}, ::Val{N}) where {M,N} = M - N
@inline function sub(::Val{M}, ::Val{N}) where {M,N}
    isa(M, Int) && isa(N, Int) || throw(ArgumentError("Expected M and N to be Ints, got M = $M, N = $N"))
    _sub(Val(M), Val(N))
end

@pure _ncolons(::Val{N}) where N = ntuple(_ -> Colon(), Val(N))
@inline function ncolons(::Val{N}) where N
    isa(N, Int) || throw(ArgumentError("Expected an Int, got N = $N"))
    _ncolons(Val(N))
end


@inline function front_tuple(x::NTuple{L}, ::Val{M}) where {L,M}
    ntuple(i -> x[i], Val(M))
end

@inline function back_tuple(x::NTuple{L}, ::Val{N}) where {L,N}
    M = sub(Val(L), Val(N))
    ntuple(i -> x[i + M], Val(N))
end

@inline function split_tuple(x::NTuple{L}, ::Val{M}) where {L,M}
    N = sub(Val(L), Val(M))
    (front_tuple(x, Val(M)), back_tuple(x, Val(N)))
end

@inline function frontlast(x::NTuple{L}) where {L}
    front_tuple(x, Val(sub(Val(L), Val(1)))), last(x)
end

@inline function firstback(x::NTuple{L}) where {L}
    first(x), back_tuple(x, Val(sub(Val(L), Val(1))))
end


@pure _tuple_length(T::Type{<:Tuple}) = length(T.parameters)
@pure _tuple_length(T::Tuple) = length(typeof(T))
tuple_length(T::Tuple) = _tuple_length(T)
