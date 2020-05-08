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
    @inline function TrajectoryBuffer{T,SS,OO,AA,RR,ST,OT}(
        S,
        O,
        A,
        R,
        sT,
        oT,
        done,
        offsets,
    ) where {T,SS,OO,AA,RR,ST,OT}
        B = new(S, O, A, R, sT, oT, done, offsets)
        checkrep(B)
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
) where {SS,OO,AA,RR,ST,OT}
    I = 1:1
    T = Trajectory{_getindextype(S, I),_getindextype(O, I),_getindextype(A, I),_getindextype(R, I)}
    TrajectoryBuffer{T,SS,OO,AA,RR,ST,OT}(S, O, A, R, sT, oT, done, offsets)
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
    )
end

function checkrep(B::TrajectoryBuffer)
    if !(length(B.S) == length(B.O) == length(B.A) == length(B.R))
        throw(DimensionMismatch("Lengths of S, O, A, and R do not match"))
    end
    if !(length(B.sT) == length(B.oT) == length(B.done)) # TODO compare against offsets
        throw(DimensionMismatch("Lengths of sT, oT, and done do not match"))
    end
    length(B.S) < nsamples(B) && error("invalid dimensions")
    if !(B.offsets[1] == 0 && all(i -> B.offsets[i-1] < B.offsets[i], 2:length(B)))
        throw(DimensionMismatch("Invalid offsets"))
    end
    length(B) < 0 && throw(DomainError("length must be > 0"))
    return B
end


Base.length(B::TrajectoryBuffer) = length(B.offsets) - 1

Base.size(B::TrajectoryBuffer) = (length(B), )

@propagate_inbounds function Base.getindex(B::TrajectoryBuffer, i::Int)
    range = (B.offsets[i]+1):B.offsets[i+1]
    Trajectory(B.S[range], B.O[range], B.A[range], B.R[range], B.sT[i], B.oT[i], B.done[i])
end

Base.IndexStyle(::Type{<:TrajectoryBuffer}) = IndexLinear()

Base.empty!(B::TrajectoryBuffer) = _resize!(B, 0, 0)

# TODO HasLength/HasShape
function Base.append!(B::TrajectoryBuffer, iter)
    i = lastindex(B.offsets)
    offset = B.offsets[i]
    _sizehint!(B, nsamples(B) + sum(length, iter), length(B) + length(iter))

    for τ::Trajectory in iter
        len_τ = length(τ)

        push!(B.offsets, offset + len_τ)
        copyto!(B.S, offset + firstindex(B.S), τ.S, firstindex(τ.S), len_τ)
        copyto!(B.O, offset + firstindex(B.O), τ.O, firstindex(τ.O), len_τ)
        copyto!(B.A, offset + firstindex(B.A), τ.A, firstindex(τ.A), len_τ)
        copyto!(B.R, offset + firstindex(B.R), τ.R, firstindex(τ.R), len_τ)
        B.sT[length(B)] = τ.sT
        B.oT[length(B)] = τ.oT
        B.done[length(B)] = τ.done

        offset += len_τ
    end

    checkrep(B)
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
    checkrep(B)
    return B
end

function _sizehint!(B::TrajectoryBuffer, nsamples::Int, ntrajectories::Int)
    checkrep(B)
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
    offset = B.offsets[end]
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

    push!(B.offsets, B.offsets[end] + t - 1)
    B.sT[length(B)] = S[offset+t]
    B.oT[length(B)] = O[offset+t]
    B.done[length(B)] = done

    return t - 1
end



function collate(Bs::AbstractVector{<:TrajectoryBuffer}, env::AbstractEnvironment)
    collate!(TrajectoryBuffer(env, sizehint = sum(nsamples, Bs)), Bs)
end

function collate!(dest::TrajectoryBuffer, Bs::AbstractVector{<:TrajectoryBuffer}, nsamples::Integer)
    ntraj = s = 0
    for B in Bs, i in eachindex(B.offsets)[1:end-1]
        s += B.offsets[i+1] - B.offsets[i]
        ntraj += 1
        s >= nsamples && break
    end

    @unpack S, O, A, R, sT, oT, done, offsets = dest
    resize!(S, nsamples)
    resize!(O, nsamples)
    resize!(A, nsamples)
    resize!(R, nsamples)
    resize!(sT, ntraj)
    resize!(oT, ntraj)
    resize!(done, ntraj)
    resize!(offsets, ntraj + 1)

    togo = nsamples
    doffs_samp = doffs_traj = 1
    for B in Bs
        soffs_samp = 1
        for soffs_traj = eachindex(B.offsets)[1:end-1]
            len = B.offsets[soffs_traj+1] - B.offsets[soffs_traj]
            if togo == 0
                checkrep(dest)
                return dest
            elseif togo < len
                copyto!(S, doffs_samp, B.S, soffs_samp, togo)
                copyto!(O, doffs_samp, B.O, soffs_samp, togo)
                copyto!(A, doffs_samp, B.A, soffs_samp, togo)
                copyto!(R, doffs_samp, B.R, soffs_samp, togo)
                sT[doffs_traj] = B.S[soffs_samp + togo]
                oT[doffs_traj] = B.O[soffs_samp + togo]
                done[doffs_traj] = false
                offsets[doffs_traj+1] = offsets[doffs_traj] + togo

                checkrep(dest)
                return dest
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
        end
    end

    checkrep(dest)

    return dest
end

function collate!(dest::TrajectoryBuffer, Bs::AbstractVector{<:TrajectoryBuffer})
    @unpack S, O, A, R, sT, oT, done, offsets = dest
    nsamp = sum(nsamples, Bs)
    ntraj = sum(length, Bs)
    resize!(S, nsamp)
    resize!(O, nsamp)
    resize!(A, nsamp)
    resize!(R, nsamp)
    resize!(sT, ntraj)
    resize!(oT, ntraj)
    resize!(done, ntraj)
    resize!(offsets, ntraj + 1)

    offs_samp = offs_traj = 1
    for B in Bs
        nsamp = nsamples(B)
        ntraj = length(B)
        copyto!(S, offs_samp, B.S, 1, nsamp)
        copyto!(O, offs_samp, B.O, 1, nsamp)
        copyto!(A, offs_samp, B.A, 1, nsamp)
        copyto!(R, offs_samp, B.R, 1, nsamp)
        copyto!(sT, offs_traj, B.sT, 1, ntraj)
        copyto!(oT, offs_traj, B.oT, 1, ntraj)
        copyto!(done, offs_traj, B.done, 1, ntraj)
        for i = eachindex(B.offsets)[1:end-1]
            l = B.offsets[i+1] - B.offsets[i]
            offsets[offs_traj + i] = offsets[offs_traj + i - 1] + l
        end
        offs_samp += nsamp
        offs_traj += ntraj
    end

    checkrep(dest)

    return dest
end

function truncate!(B::TrajectoryBuffer, n::Integer)
    nextra = nsamples(B) - n
    if nextra > 0
        ntraj = findfirst(offs -> offs >= n, B.offsets) - 1
        if B.offsets[ntraj+1] > n
            B.sT[ntraj] = B.S[n + 1]
            B.oT[ntraj] = B.O[n + 1]
            B.done[ntraj] = false
            B.offsets[ntraj + 1] = n
        end

        resize!(B.S, n)
        resize!(B.O, n)
        resize!(B.A, n)
        resize!(B.R, n)
        resize!(B.sT, ntraj)
        resize!(B.oT, ntraj)
        resize!(B.done, ntraj)
        resize!(B.offsets, ntraj + 1)
    end
    return B
end


function asvec(A::AbstractArray{<:Any,L}) where {L}
    alongs = (ntuple(_ -> True(), Val(L - 1))..., False())
    SlicedArray(A, alongs)
end

_getindextype(A::AbstractArray, I::Tuple) = typeof(A[I...])
_getindextype(A::AbstractArray, I...) = _getindextype(A, I)
