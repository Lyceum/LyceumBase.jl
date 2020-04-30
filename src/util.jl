const Maybe{T} = Union{T,Nothing}
const TupleN{T,N} = NTuple{N,T}

const AbsArr{T,N} = AbstractArray{T,N}
const AbsMat{T} = AbstractMatrix{T}
const AbsVec{T} = AbstractVector{T}

macro mustimplement(sig)
    :($(esc(sig)) = error("must implement ", $(string(sig))))
end

argerror(s::AbstractString) = throw(ArgumentError(s))
