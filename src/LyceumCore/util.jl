const LTuple{L} = Tuple{Vararg{<:Any,L}}

@inline ncolons(n::Integer) = ntuple(_ -> Colon(), n)

@inline front(t::Tuple, m::Integer) = ntuple(i -> t[i], m)
@inline front(t::LTuple{L}) where {L} = front(t, static(L - 1))

@inline function tail(t::LTuple{L}, n::Integer) where {L}
    m = @stat static(L) - n
    ntuple(i -> t[i + m], n)
end
@inline tail(t::LTuple{L}) where {L} = tail(t, static(L - 1))

@inline function split(t::LTuple{L}, m::Integer) where {L}
    n = @stat static(L) - m
    front(t, m), tail(t, n)
end

# TODO needed or use StaticArrays?
@pure _tuple_length(T::Type{<:Tuple}) = length(T.parameters)
@pure _tuple_length(T::Tuple) = length(typeof(T))
tuple_length(T::Tuple) = _tuple_length(T)
