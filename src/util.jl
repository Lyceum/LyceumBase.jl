const Maybe{T} = Union{T,Nothing}
const TupleN{T,N} = NTuple{N,T}

const AbsArr{T,N} = AbstractArray{T,N}
const AbsMat{T} = AbstractMatrix{T}
const AbsVec{T} = AbstractVector{T}

macro mustimplement(sig)
    :($(esc(sig)) = error("must implement ", $(string(sig))))
end

@noinline argerror(msg::AbstractString) = throw(ArgumentError(msg))
@noinline dimerror(msg::AbstractString) = throw(DimensionMismatch(msg))
@noinline internalerror(msg::AbstractString) = error("Internal error: $msg\nPlease file a bug.")
@noinline internalerror() = error("Internal error. Please file a bug.")
