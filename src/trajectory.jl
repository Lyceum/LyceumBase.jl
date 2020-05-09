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
        S::SS,
        O::OO,
        A::AA,
        R::RR,
        sT::ST,
        oT::OT,
        done::Bool,
    ) where {SS<:AbsVec,OO<:AbsVec,AA<:AbsVec,RR<:AbsVec,ST,OT}
        τ = new(S, O, A, R, sT, oT, done)
        checkrep(τ)
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

@inline function checkrep(τ::Trajectory)
    if !(length(τ.S) == length(τ.O) == length(τ.A) == length(τ.R))
        throw(DimensionMismatch("Lengths of S, O, A, and R don't match"))
    end
    return τ
end

Base.length(τ::Trajectory) = (checkrep(τ); length(τ.A))


####
#### TrajectoryBuffer
####

const VoA = AbstractVector{<:AbstractArray}
@auto_hash_equals struct TrajectoryBuffer{SS<:VoA,OO<:VoA,AA<:VoA,RR<:AbstractVector{<:Real}}
    S::SS
    O::OO
    A::AA
    R::RR
    sT::SS
    oT::OO
    done::Vector{Bool}
    offsets::Vector{Int}
    function TrajectoryBuffer(
        S::SS,
        O::OO,
        A::AA,
        R::RR,
        sT::SS,
        oT::OO,
        done::Vector{Bool},
        offsets::Vector{Int},
    ) where {SS<:VoA,OO<:VoA,AA<:VoA,RR<:AbstractVector{<:Real}}
        Base.require_one_based_indexing(S, O, A, R, sT, oT)
        B = new{SS,OO,AA,RR}(S, O, A, R, sT, oT, done, offsets)
        checkrep(B)
        return B
    end
end

function TrajectoryBuffer(
    env::AbstractEnvironment;
    dtype::Maybe{DataType} = nothing,
    sizehint::Integer = 1024,
)
    sizehint > 0 || argerror("sizehint must be > 0")
    sp = dtype === nothing ? spaces(env) : adapt(dtype, spaces(env))
    sizehint_traj = max(1, div(sizehint, 10)) # a heuristic
    TrajectoryBuffer(
        asvec(ElasticArray(undef, sp.statespace, sizehint)),
        asvec(ElasticArray(undef, sp.observationspace, sizehint)),
        asvec(ElasticArray(undef, sp.actionspace, sizehint)),
        Array(undef, sp.rewardspace, sizehint),
        asvec(ElasticArray(undef, sp.statespace, sizehint_traj)),
        asvec(ElasticArray(undef, sp.observationspace, sizehint_traj)),
        Vector{Bool}(undef, sizehint_traj),
        Int[0],
    )
end

asvec(A::AbstractArray{<:Any,L}) where {L} = SpecialArrays.slice(A, Val(L-1))

@noinline function checkrep(B::TrajectoryBuffer)
    if !(length(B.S) == length(B.O) == length(B.A) == length(B.R))
        throw(DimensionMismatch("Lengths of S, O, A, and R do not match"))
    end
    if !(length(B.sT) == length(B.oT) == length(B.done)) # TODO compare against offsets
        throw(DimensionMismatch("Lengths of sT, oT, and done do not match"))
    end
    length(B.S) < nsamples(B) && error("invalid dimensions")
    if !(B.offsets[1] == 0 && all(i -> B.offsets[i-1] < B.offsets[i], 2:ntrajectories(B)))
        throw(DimensionMismatch("Invalid offsets"))
    end
    ntrajectories(B) < 0 && throw(DomainError("length must be > 0"))
    return B
end


ntrajectories(B::TrajectoryBuffer) = length(B.offsets) - 1

nsamples(B::TrajectoryBuffer) = B.offsets[ntrajectories(B)+1]
@inline function nsamples(B::TrajectoryBuffer, i::Integer)
    @boundscheck checkbounds(Base.OneTo(ntrajectories(B)), i)
    B.offsets[i+1] - B.offsets[i]
end

Base.empty!(B::TrajectoryBuffer) = resize!(B.offsets, 1)

function truncate!(B::TrajectoryBuffer)
    nsamp = nsamples(B)
    ntraj = ntrajectories(B)
    resize!(B.S, nsamp)
    resize!(B.O, nsamp)
    resize!(B.A, nsamp)
    resize!(B.R, nsamp)
    resize!(B.sT, ntraj)
    resize!(B.oT, ntraj)
    resize!(B.done, ntraj)
    resize!(B.offsets, ntraj + 1)
    checkrep(B)
    return B
end


"""
    $(SIGNATURES)

Rollout the actions computed by `policy!` on `env`, starting at whatever state `env` is currently
in, for `Hmax` timesteps or until `isdone(st, ot, env)` returns `true` and store the resultant
trajectory in `B`. `policy!` should be a function of the form `policy!(at, ot)` which computes an
action given the current observation `ot` and stores it in `at`.

See also: [`TrajectoryBuffer`](@ref), [`rollout`](@ref), [`sample`](@ref), [`sample!`](@ref)
"""
function rollout!(policy!, B::TrajectoryBuffer, env::AbstractEnvironment, Hmax::Integer)
    _rollout!(policy!, B, env, convert(Int, Hmax))
    truncate!(B)
    return B
end

function rollout(policy!, env::AbstractEnvironment, Hmax::Integer)
    rollout!(policy!, TrajectoryBuffer(env, sizehint = Hmax), env, Hmax)
end

function _rollout!(
    @specialize(policy!),
    B::TrajectoryBuffer,
    env::AbstractEnvironment,
    Hmax::Int,
    stopcb = () -> false,
)
    Hmax > 0 || argerror("Hmax must be > 0")

    @unpack S, O, A, R = B

    # pre-allocate for one additional trajectory with a length of Hmax + 1.
    # "+1" because of the terminal state/observation
    _preallocate_traj!(B, Hmax + 1)
    offset = B.offsets[end]

    t::Int = 1
    done::Bool = false

    # get the initial state/observation
    st = S[offset+t]
    ot = O[offset+t]
    getstate!(st, env)
    getobservation!(ot, env)

    while true
        # Abandon the current rollout. Used internally for multithreaded sampling.
        stopcb() && return 0

        at = A[offset+t]
        getaction!(at, env) # Fill at with env's current action for convenience
        policy!(at, ot)
        R[offset+t] = getreward(st, at, ot, env)

        setaction!(env, at)
        step!(env)
        t += 1

        # Get the next state, which may be a terminal state.
        st = S[offset+t]
        ot = O[offset+t]
        getstate!(st, env)
        getobservation!(ot, env)

        done = isdone(st, ot, env)

        # Break *after* we've hit Hmax or isdone returns true. If isdone never returns true,
        # then this loop will run for Hmax + 1 iterations.
        (t > Hmax || done) && break
    end

    push!(B.offsets, B.offsets[end] + t - 1)
    B.sT[ntrajectories(B)] = S[offset+t]
    B.oT[ntrajectories(B)] = O[offset+t]
    B.done[ntrajectories(B)] = done

    return t - 1
end

function _preallocate_traj!(B::TrajectoryBuffer, H::Int)
    nsamp = nsamples(B) + H
    ntraj = ntrajectories(B) + 1
    if length(B.S) < nsamp
        l = nextpow(2, nsamp)
        resize!(B.S, l)
        resize!(B.O, l)
        resize!(B.A, l)
        resize!(B.R, l)
    end
    if length(B.offsets) < ntraj + 2
        l = nextpow(2, ntraj + 2)
        resize!(B.sT, l)
        resize!(B.oT, l)
        resize!(B.done, l)
    end
    checkrep(B)
    return B
end


function collate!(dest::TrajectoryBuffer, Bs::AbstractVector{<:TrajectoryBuffer}, n::Integer)
    n >= 0 || argerror("n must be ≥ 0")
    ns = isempty(Bs) ? 0 : sum(nsamples, Bs)
    n <= ns || argerror("n is greater than the number of available samples")

    @unpack S, O, A, R, sT, oT, done, offsets = dest

    if isempty(Bs) || ns == 0
        resize!(S, 0)
        resize!(O, 0)
        resize!(A, 0)
        resize!(R, 0)
        resize!(sT, 0)
        resize!(oT, 0)
        resize!(done, 0)
        resize!(offsets, 1)
        return dest
    end

    ntraj = s = 0
    for B in Bs, i in eachindex(B.offsets)[1:end-1]
        s += B.offsets[i+1] - B.offsets[i]
        ntraj += 1
        s >= n && break
    end

    resize!(S, n)
    resize!(O, n)
    resize!(A, n)
    resize!(R, n)
    resize!(sT, ntraj)
    resize!(oT, ntraj)
    resize!(done, ntraj)
    resize!(offsets, ntraj + 1)

    togo = n
    doffs_samp = doffs_traj = 1
    for B in Bs
        soffs_samp = 1
        for soffs_traj = eachindex(B.offsets)[1:end-1]
            len = B.offsets[soffs_traj+1] - B.offsets[soffs_traj]
            if togo < len
                copyto!(S, doffs_samp, B.S, soffs_samp, togo)
                copyto!(O, doffs_samp, B.O, soffs_samp, togo)
                copyto!(A, doffs_samp, B.A, soffs_samp, togo)
                copyto!(R, doffs_samp, B.R, soffs_samp, togo)
                sT[doffs_traj] = B.S[soffs_samp + togo]
                oT[doffs_traj] = B.O[soffs_samp + togo]
                done[doffs_traj] = false
                offsets[doffs_traj+1] = offsets[doffs_traj] + togo
            else
                copyto!(S, doffs_samp, B.S, soffs_samp, len)
                copyto!(O, doffs_samp, B.O, soffs_samp, len)
                copyto!(A, doffs_samp, B.A, soffs_samp, len)
                copyto!(R, doffs_samp, B.R, soffs_samp, len)
                sT[doffs_traj] = B.sT[soffs_traj]
                oT[doffs_traj] = B.oT[soffs_traj]
                done[doffs_traj] = B.done[soffs_traj]
                offsets[doffs_traj+1] = offsets[doffs_traj] + len

                soffs_samp += len
                doffs_samp += len
                doffs_traj += 1
                togo -= len
            end
            if togo == 0
                checkrep(dest)
                return dest
            end
        end
    end

    # This shouldn't be reached
    internalerror()
end


function StructArrays.StructArray(B::TrajectoryBuffer)
    S = BatchedVector(B.S, B.offsets)
    O = BatchedVector(B.O, B.offsets)
    A = BatchedVector(B.A, B.offsets)
    R = BatchedVector(B.R, B.offsets)
    sT = B.sT
    oT = B.oT
    done = B.done
    T = Trajectory{eltype(S),eltype(O),eltype(A),eltype(R),eltype(sT),eltype(oT)}
    return StructArray{T}((S, O, A, R, sT, oT, done))
end
