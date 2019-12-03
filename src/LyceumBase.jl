module LyceumBase

using Shapes

# Environment interface
export
    AbstractEnv,
    EnvSpaces,

    statespace,
    getstate!,
    getstate,

    observationspace,
    getobs!,
    getobs,

    actionspace,
    getaction!,
    setaction!,
    getaction,

    rewardspace,
    getreward,

    evaluationspace,
    geteval,

    reset!,
    randreset!,
    step!,
    isdone,
    sharedmemory_envs,
    timestep,
    effective_timestep,
    spaces



const Maybe{T} = Union{T,Nothing}
const Many{T} = Tuple{Vararg{T}}
const TupleN{T,N} = NTuple{N,T}

const AbsArr{T,N} = AbstractArray{T,N}
const AbsMat{T} = AbstractMatrix{T}
const AbsVec{T} = AbstractVector{T}

const RealArr{N} = AbstractArray{<:Real,N}
const RealMat = AbstractMatrix{<:Real}
const RealVec = AbstractVector{<:Real}



macro mustimplement(sig)
    fname = sig.args[1]
    arg1 = sig.args[2]
    if isa(arg1, Expr)
        arg1 = arg1.args[1]
    end
    :($(esc(sig)) = error(typeof($(esc(arg1))), " must implement ", $(Expr(:quote, fname))))
end

# Interfaces
include("abstractenvironment.jl")

include("Tools/Tools.jl")

end # module
