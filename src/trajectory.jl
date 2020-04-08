####
#### Trajectory
####

@auto_hash_equals struct Trajectory{SS<:AbsVec,OO<:AbsVec,AA<:AbsVec,RR<:AbsVec,ST,OT}
    S::SS
    O::OO
    A::AA
    R::RR
    sT::ST
    oT::OT
    done::Bool
    @inline function Trajectory{SS,OO,AA,RR,ST,OT}(
        S,
        O,
        A,
        R,
        sT,
        oT,
        done,
    ) where {SS,OO,AA,RR,ST,OT}
        τ = new(S, O, A, R, sT, oT, done)
        _check_consistency(τ)
        return τ
    end
end

@inline function Trajectory(
    S::SS,
    O::OO,
    A::AA,
    R::RR,
    sT::ST,
    oT::OT,
    done,
) where {SS,OO,AA,RR,ST,OT}
    Trajectory{SS,OO,AA,RR,ST,OT}(S, O, A, R, sT, oT, done)
end

@inline function _check_consistency(τ::Trajectory)
    if !(length(τ.S) == length(τ.O) == length(τ.A) == length(τ.R))
        throw(DimensionMismatch("Lengths of S, O, A, and R don't match"))
    end
    return τ
end

Base.length(τ::Trajectory) = (_check_consistency(τ); length(τ.A))


####
#### TrajectoryBuffer
####

mutable struct TrajectoryBuffer{T<:Trajectory,SS<:AbsVec,OO<:AbsVec,AA<:AbsVec,RR<:AbsVec,ST,OT} <:
               AbsVec{T}
    S::SS
    O::OO
    A::AA
    R::RR
    sT::ST
    oT::OT
    done::Vector{Bool}
    offsets::Vector{Int}
    len::Int
    @inline function TrajectoryBuffer{T,SS,OO,AA,RR,ST,OT}(
        S,
        O,
        A,
        R,
        sT,
        oT,
        done,
        offsets,
        len,
    ) where {T,SS,OO,AA,RR,ST,OT}
        B = new(S, O, A, R, sT, oT, done, offsets, len)
        _check_consistency(B)
        return B
    end
end

@inline function TrajectoryBuffer(
    S::SS,
    O::OO,
    A::AA,
    R::RR,
    sT::ST,
    oT::OT,
    done,
    offsets,
    len,
) where {SS,OO,AA,RR,ST,OT}
    I = 1:1
    T = Trajectory{_getindextype(S, I),_getindextype(O, I),_getindextype(A, I),_getindextype(R, I)}
    TrajectoryBuffer{T,SS,OO,AA,RR,ST,OT}(S, O, A, R, sT, oT, done, offsets, len)
end

function TrajectoryBuffer(
    env::AbstractEnvironment;
    dtype::Maybe{DataType} = nothing,
    sizehint::Integer = 1024,
)
    sizehint > 0 || throw(ArgumentError("sizehint must be ≥ 0"))
    sp = dtype === nothing ? spaces(env) : adapt(dtype, spaces(env))
    TrajectoryBuffer(
        asvec(ElasticArray(undef, sp.statespace, sizehint)),
        asvec(ElasticArray(undef, sp.observationspace, sizehint)),
        asvec(ElasticArray(undef, sp.actionspace, sizehint)),
        Array(undef, sp.rewardspace, sizehint),
        asvec(ElasticArray(undef, sp.statespace, 0)),
        asvec(ElasticArray(undef, sp.observationspace, 0)),
        Bool[],
        Int[0],
        0,
    )
end

function _check_consistency(B::TrajectoryBuffer)
    if !(length(B.S) == length(B.O) == length(B.A) == length(B.R))
        throw(DimensionMismatch("Lengths of S, O, A, and R do not match"))
    end
    if !(length(B.sT) == length(B.oT) == length(B.done))
        throw(DimensionMismatch("Lengths of sT, oT, and done do not match"))
    end
    length(B.S) < nsamples(B) && error("invalid dimensions")
    if !(B.offsets[1] == 0 && all(i -> B.offsets[i-1] < B.offsets[i], 2:length(B)))
        throw(DimensionMismatch("Invalid offsets"))
    end
    length(B) < 0 && throw(DomainError("length must be > 0"))
    return B
end


@inline Base.length(B::TrajectoryBuffer) = B.len

@inline Base.size(B::TrajectoryBuffer) = (length(B),)

@propagate_inbounds function Base.getindex(B::TrajectoryBuffer, i::Int)
    range = (B.offsets[i]+1):B.offsets[i+1]
    Trajectory(B.S[range], B.O[range], B.A[range], B.R[range], B.sT[i], B.oT[i], B.done[i])
end

Base.IndexStyle(::Type{<:TrajectoryBuffer}) = IndexLinear()

Base.empty!(B::TrajectoryBuffer) = B.len = 0

# TODO HasLength/HasShape
function Base.append!(B::TrajectoryBuffer, iter)
    offset = B.offsets[length(B)+1]
    _sizehint!(B, nsamples(B) + sum(length, iter), length(B) + length(iter))

    for τ::Trajectory in iter
        B.len += 1
        len_τ = length(τ)

        copyto!(B.S, offset + firstindex(B.S), τ.S, firstindex(τ.S), len_τ)
        copyto!(B.O, offset + firstindex(B.O), τ.O, firstindex(τ.O), len_τ)
        copyto!(B.A, offset + firstindex(B.A), τ.A, firstindex(τ.A), len_τ)
        copyto!(B.R, offset + firstindex(B.R), τ.R, firstindex(τ.R), len_τ)
        B.sT[length(B)] = τ.sT
        B.oT[length(B)] = τ.oT
        B.done[length(B)] = τ.done
        B.offsets[length(B)+1] = offset + len_τ

        offset += len_τ
    end

    _check_consistency(B)
    return B
end


function truncate!(B::TrajectoryBuffer)
    resize!(B.S, nsamples(B))
    resize!(B.O, nsamples(B))
    resize!(B.A, nsamples(B))
    resize!(B.R, nsamples(B))
    resize!(B.sT, length(B))
    resize!(B.oT, length(B))
    resize!(B.done, length(B))
    resize!(B.offsets, length(B) + 1)
    return B
end

@inline nsamples(B::TrajectoryBuffer) = B.offsets[length(B)+1]
@inline function nsamples(B::Trajectory, i::Integer)
    @boundscheck checkbounds(B, i)
    B.offsets[i+1] - B.offsets[i]
end

function _resize!(B::TrajectoryBuffer, nsamples::Int, ntrajectories::Int)
    resize!(B.S, nsamples)
    resize!(B.O, nsamples)
    resize!(B.A, nsamples)
    resize!(B.R, nsamples)
    resize!(B.sT, ntrajectories)
    resize!(B.oT, ntrajectories)
    resize!(B.done, ntrajectories)
    resize!(B.offsets, ntrajectories + 1)
    _check_consistency(B)
    return B
end

function _sizehint!(B::TrajectoryBuffer, nsamples::Int, ntrajectories::Int)
    _check_consistency(B)
    if length(B.S) < nsamples
        l = nextpow(2, nsamples)
        resize!(B.S, l)
        resize!(B.O, l)
        resize!(B.A, l)
        resize!(B.R, l)
    end
    if length(B.offsets) < ntrajectories + 2
        l = nextpow(2, ntrajectories + 2)
        resize!(B.sT, l)
        resize!(B.oT, l)
        resize!(B.done, l)
        resize!(B.offsets, l)
    end
    return B
end

"""
    $(TYPEDSIGNATURES)

Rollout the actions computed by `policy!` on `env`, starting at whatever state `env` is currently
in, for `Hmax` timesteps or until `isdone(st, ot, env)` returns `true` and store the resultant
trajectory in `B`. `policy!` should be a function of the form `policy!(at, ot)` which computes an
action given the current observation `ot` and stores it in `at`.

See also: [`sample!`](@ref), [`sample`](@ref).
"""
function rollout!(policy!, B::TrajectoryBuffer, env::AbstractEnvironment, Hmax::Integer)
    _rollout!(policy!, B, env, convert(Int, Hmax))
    truncate!(B)
    return B
end

function _rollout!(
    @specialize(policy!),
    B::TrajectoryBuffer,
    env::AbstractEnvironment,
    Hmax::Int,
    stopcb = () -> false,
)
    Hmax > 0 || throw(ArgumentError("Hmax must be > 0"))

    # pre-allocate for one additional trajectory with a length of Hmax + 1.
    # "+1" because of the terminal state/observation
    _sizehint!(B, nsamples(B) + Hmax + 1, length(B) + 1)
    offset = B.offsets[length(B)+1]
    @unpack S, O, A, R = B

    t::Int = 1
    done::Bool = false
    # get the initial state/observation
    st = S[offset+t]::SubArray
    ot = O[offset+t]::SubArray
    getstate!(st, env)
    getobservation!(ot, env)
    while true
        stopcb() && return 0 # Abandon the current rollout

        # Get the policy's action for (st, ot, at)
        at = A[offset+t]::SubArray
        getaction!(at, env) # Fill at with env's current action for convenience
        policy!(at, ot)
        R[offset+t] = getreward(st, at, ot, env)

        setaction!(env, at)
        step!(env)
        t += 1

        st = S[offset+t]::SubArray
        ot = O[offset+t]::SubArray
        getstate!(st, env)
        getobservation!(ot, env)

        done = isdone(st, ot, env)

        # Break *after* we've hit Hmax or isdone returns true. If isdone never returns true,
        # then this loop will run for Hmax + 1 iterations.
        (t > Hmax || done) && break
    end

    B.len += 1
    B.sT[length(B)] = S[offset+t]
    B.oT[length(B)] = O[offset+t]
    B.done[length(B)] = done
    B.offsets[length(B)+1] = B.offsets[length(B)] + t - 1

    return t - 1
end



function collate(Bs::AbsVec{<:TrajectoryBuffer}, env::AbstractEnvironment, nsamples::Integer)
    collate!(TrajectoryBuffer(env), Bs, nsamples)
end

function collate!(dest::TrajectoryBuffer, Bs::AbsVec{<:TrajectoryBuffer}, nsamples::Integer)
    _sizehint!(dest, nsamples, sum(length, Bs))
    empty!(dest)

    togo = nsamples
    for B in Bs, τ in B
        if togo == 0
            break
        elseif togo < length(τ)
            τ′ = Trajectory(
                view(τ.S, 1:togo),
                view(τ.O, 1:togo),
                view(τ.A, 1:togo),
                view(τ.R, 1:togo),
                τ.S[togo+1],
                τ.O[togo+1],
                false,
            )
        else
            τ′ = τ
        end
        push!(dest, τ′)
        togo -= length(τ′)
    end

    truncate!(dest)

    return dest
end


function asvec(A::AbsArr{<:Any,L}) where {L}
    alongs = (ntuple(_ -> True(), Val(L - 1))..., False())
    SlicedArray(A, alongs)
end

_getindextype(A::AbstractArray, I::Tuple) = typeof(A[I...])
_getindextype(A::AbstractArray, I...) = _getindextype(A, I)
