@inline ncolons(n::Integer) = ntuple(_ -> Colon(), n)

@inline front(t::Tuple, m::Integer) = ntuple(i -> t[i], m)
@inline front(t::Tuple, ::StaticOrVal{M}) where {M} = ntuple(i -> t[i], Val(M))
@inline front(t::LTuple{L}) where {L} = front(t, Val(L - 1))

@inline function tail(t::Tuple, n::Integer)
    m = length(t) - n
    ntuple(i -> t[i + m], n)
end
@inline function tail(t::LTuple{L}, ::StaticOrVal{N}) where {L,N}
    M = L - N
    ntuple(i -> t[i + M], Val(N))
end
@inline tail(t::LTuple{L}) where {L} = tail(t, Val(L - 1))

@inline function tuplesplit(t::LTuple{L}, ::StaticOrVal{M}) where {L,M}
    front(t, Val(L)), tail(t, Val(L - M))
end
@inline function tuplesplit(t::LTuple{L}, m::Integer) where {L}
    front(t, Val(L)), tail(t, L - m)
end