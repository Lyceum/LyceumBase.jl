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

struct TrajectoryBuffer{T<:Trajectory,SS<:AbsVec,OO<:AbsVec,AA<:AbsVec,RR<:AbsVec,ST<:AbsVec,OT<:AbsVec} <: AbsVec{T}
    S::SS
    O::OO
    A::AA
    R::RR
    sT::ST
    oT::OT
    done::Vector{Bool}
    offsets::Vector{Int}

    @inline function TrajectoryBuffer(
        S::SS,
        O::OO,
        A::AA,
        R::RR,
        sT::ST,
        oT::OT,
        done::AbsVec{Bool},
        offsets::AbsVec{<:Integer},
    ) where {SS<:AbsVec,OO<:AbsVec,AA<:AbsVec,RR<:AbsVec,ST<:AbsVec,OT<:AbsVec}
        Base.require_one_based_indexing(S, O, A, R, sT, oT, done, offsets)
        I = 1:1
        T = Trajectory{_getindextype(S, I),_getindextype(O, I),_getindextype(A, I),_getindextype(R, I)}
        B = new{T,SS,OO,AA,RR,ST,OT}(S, O, A, R, sT, oT, done, offsets)
        _check_consistency(B)
        return B
    end
end

_getindextype(A::AbstractArray, I::Tuple) = @inbounds typeof(A[I...])
_getindextype(A::AbstractArray, I...) = _getindextype(A, I)

function TrajectoryBuffer(
    env::AbstractEnvironment;
    dtype::Maybe{DataType} = nothing,
    sizehint::Integer = 1,
)
    sizehint >= 0 || throw(ArgumentError("sizehint must be ≥ 0"))
    sp = dtype === nothing ? spaces(env) : adapt(dtype, spaces(env))
    TrajectoryBuffer(
        asvec(ElasticArray(undef, sp.statespace, sizehint)),
        asvec(ElasticArray(undef, sp.observationspace, sizehint)),
        asvec(ElasticArray(undef, sp.actionspace, sizehint)),
        # TODO once support for non-scalar rewards is defined.
        Array(undef, sp.rewardspace, sizehint),
        asvec(ElasticArray(undef, sp.statespace, 0)),
        asvec(ElasticArray(undef, sp.observationspace, 0)),
        Bool[],
        Int[1], # TODO firstindex?
    )
end

asvec(A::AbstractVector) = A
function asvec(A::AbstractArray{<:Any,L}) where {L}
    alongs = (ntuple(_ -> True(), Val(L - 1))..., False())
    SlicedArray(A, alongs)
end

# TODO remove once battle tested
function _check_consistency(B::TrajectoryBuffer)
    if !(length(B.S) == length(B.O) == length(B.A) == length(B.R))
        internalerror("Lengths of S, O, A, and R do not match.")
    end
    #if !(length(B.sT) == length(B.oT) == length(B.done) == length(B.offsets) - 1) # TODO
    if !(length(B.sT) == length(B.oT) == length(B.done))
        internalerror("Lengths of sT, oT, and done do not match")
    end
    offs = B.offsets
    if !(length(offs) > 0 && offs[1] == 1 && all(i -> offs[i-1] < offs[i], eachindex(offs)[2:end]))
        internalerror("Invalid offsets")
    end
    return nothing
end


@inline Base.length(B::TrajectoryBuffer) = length(B.offsets) - 1

@inline Base.size(B::TrajectoryBuffer) = (length(B), )

@propagate_inbounds function Base.getindex(B::TrajectoryBuffer, i::Int)
    r = B.offsets[i]:(B.offsets[i+1] - 1)
    Trajectory(B.S[r], B.O[r], B.A[r], B.R[r], B.sT[i], B.oT[i], B.done[i])
end

Base.IndexStyle(::Type{<:TrajectoryBuffer}) = IndexLinear()


function Base.push!(dest::TrajectoryBuffer, src::Trajectory)
    append!(dest.S, src.S)
    append!(dest.O, src.O)
    append!(dest.A, src.A)
    append!(dest.R, src.R)
    push!(dest.oT, src.oT)
    push!(dest.sT, src.sT)
    push!(dest.done, src.done)
    push!(dest.offsets, last(dest.offsets) + length(src))
    _check_consistency(dest)
    return dest
end

function Base.append!(B::TrajectoryBuffer, iter::TrajectoryBuffer)
    append!(B.S, iter.S)
    append!(B.O, iter.O)
    append!(B.A, iter.A)
    append!(B.R, iter.R)
    append!(B.oT, iter.oT)
    append!(B.sT, iter.sT)
    append!(B.done, iter.done)

    n = length(B.offsets)
    resize!(B.offsets, n + length(iter.offsets) - 1)
    for i = eachindex(iter.offsets)[1:end-1]
        B.offsets[n+i] = B.offsets[n+i-1] + iter.offsets[i+1] - iter.offsets[i]
    end
    _check_consistency(B)
    return B
end

function Base.append!(B::TrajectoryBuffer, iter)
    for item in iter
        push!(B, item)
    end
    return B
end




function Base.empty!(B::TrajectoryBuffer)
    _resize!(B, 0, 0)
    return B
end

function _softempty!(B::TrajectoryBuffer)
    resize!(B.offsets, 1)
    return B
end


@inline nsamples(B::TrajectoryBuffer) = B.offsets[end] - 1


function _resize!(B::TrajectoryBuffer, nsamples::Integer, length::Integer)
    resize!(B.S, nsamples)
    resize!(B.O, nsamples)
    resize!(B.A, nsamples)
    resize!(B.R, nsamples)
    resize!(B.sT, length)
    resize!(B.oT, length)
    resize!(B.done, length)
    resize!(B.offsets, length + 1)
    #_check_consistency(B)
    return B
end

function _sizehint!(B::TrajectoryBuffer, ns::Integer, l::Integer)
    if ns > nsamples(B)
        resize!(B.S, ns)
        resize!(B.O, ns)
        resize!(B.A, ns)
        resize!(B.R, ns)
    end
    if l > length(B)
        resize!(B.sT, l)
        resize!(B.oT, l)
        resize!(B.done, l)
        sizehint!(B.offsets, l)
    end
    _check_consistency(B)
    return B
end

_truncate!(B::TrajectoryBuffer) = _resize!(B, nsamples(B), length(B))

function _preallocate_trajectory!(B::TrajectoryBuffer, H::Integer)
    _sizehint!(B, nsamples(B) + H, length(B) + 1)
    #B.offsets[end] = B.offsets[end - 1] + H
    push!(B.offsets, B.offsets[end] + H)
    return B.offsets[end - 1]
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
    _truncate!(B)
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
    offset = _preallocate_trajectory!(B, Hmax + 1)

    @unpack S, O, A, R = B

    t::Int = 0
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

    B.sT[length(B)] = S[offset+t]
    B.oT[length(B)] = O[offset+t]
    B.done[length(B)] = done
    B.offsets[end] = offset + t
    _check_consistency(B)

    return t
end



function collate(Bs::AbsVec{<:TrajectoryBuffer}, env::AbstractEnvironment, nsamples::Integer)
    collate!(TrajectoryBuffer(env), Bs, nsamples)
end

function collate!(dest::TrajectoryBuffer, Bs::AbsVec{<:TrajectoryBuffer}, n::Integer)
    empty!(dest)
    #_sizehint!(dest, sum(nsamples, Bs), sum(length, Bs))

    for B in Bs
        #append!(parent(dest.S), parent(B.S))
        #append!(parent(dest.O), parent(B.O))
        #append!(parent(dest.A), parent(B.A))
        #append!(parent(dest.R), parent(B.R))
        #copyto!(parent(dest.S), 1, parent(B.S), 1, length(parent(B.S)))
        #copyto!(parent(dest.O), 1, parent(B.O), 1, length(parent(B.O)))
        #copyto!(parent(dest.A), 1, parent(B.A), 1, length(parent(B.A)))
        #copyto!(parent(dest.R), 1, parent(B.R), 1, length(parent(B.R)))
        #push!(dest, τ)
        #append!(dest, B)
        #nsamples(dest) >= n && break
    end
    #togo = n
    #for B in Bs, τ in B
    #    if togo == 0
    #        break
    #    elseif togo < length(τ)
    #        τ′ = Trajectory(
    #            view(τ.S, 1:togo),
    #            view(τ.O, 1:togo),
    #            view(τ.A, 1:togo),
    #            view(τ.R, 1:togo),
    #            τ.S[togo+1],
    #            τ.O[togo+1],
    #            false,
    #        )
    #    else
    #        τ′ = τ
    #    end
    #    push!(dest, τ′)
    #    togo -= length(τ′)
    #end

    #return dest
end
