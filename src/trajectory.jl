@auto_hash_equals struct Trajectory{SS<:AbsVec,OO<:AbsVec,AA<:AbsVec,RR<:AbsVec,ST,OT}
    S::SS
    O::OO
    A::AA
    R::RR
    sT::ST
    oT::OT
    done::Bool
    @inline function Trajectory{SS,OO,AA,RR,ST,OT}(S, O, A, R, sT, oT, done) where {SS,OO,AA,RR,ST,OT}
        τ = new(S, O, A, R, sT, oT, done)
        _check_consistency(τ)
        return τ
    end
end

@inline function Trajectory(S::SS, O::OO, A::AA, R::RR, sT::ST, oT::OT, done) where {SS,OO,AA,RR,ST,OT}
    Trajectory{SS,OO,AA,RR,ST,OT}(S, O, A, R, sT, oT, done)
end

@inline function _check_consistency(τ::Trajectory)
    if !(length(τ.S) == length(τ.O) == length(τ.A) == length(τ.R))
        throw(DimensionMismatch("Lengths of S, O, A, and R don't match"))
    end
    return τ
end

Base.length(τ::Trajectory) = (_check_consistency(τ); length(τ.A))



mutable struct TrajectoryVector{T<:Trajectory,SS<:AbsVec,OO<:AbsVec,AA<:AbsVec,RR<:AbsVec,ST,OT} <: AbsVec{T}
    S::SS
    O::OO
    A::AA
    R::RR
    sT::ST
    oT::OT
    done::Vector{Bool}
    offsets::Vector{Int}
    len::Int
    @inline function TrajectoryVector{T,SS,OO,AA,RR,ST,OT}(
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
        V = new(S, O, A, R, sT, oT, done, offsets, len)
        _check_consistency(V)
        return V
    end
end

@inline function TrajectoryVector(
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
    TrajectoryVector{T,SS,OO,AA,RR,ST,OT}(S, O, A, R, sT, oT, done, offsets, len)
end

function TrajectoryVector(env::AbstractEnvironment; sizehint::Integer = 1024)
    sizehint > 0 || throw(ArgumentError("sizehint must be ≥ 0"))
    TrajectoryVector(
        asvec(ElasticArray(undef, statespace(env), sizehint)),
        asvec(ElasticArray(undef, obsspace(env), sizehint)),
        asvec(ElasticArray(undef, actionspace(env), sizehint)),
        Array(undef, rewardspace(env), sizehint),
        asvec(ElasticArray(undef, statespace(env), 0)),
        asvec(ElasticArray(undef, obsspace(env), 0)),
        Bool[],
        Int[0],
        0,
    )
end

@inline function _check_consistency(V::TrajectoryVector)
    if !(length(V.S) == length(V.O) == length(V.A) == length(V.R))
        throw(DimensionMismatch("Lengths of S, O, A, and R do not match"))
    end
    if !(length(V.sT) == length(V.oT) == length(V.done))
        throw(DimensionMismatch("Lengths of sT, oT, and done do not match"))
    end
    length(V.S) < ntimesteps(V) && error("invalid dimensions")
    if !(V.offsets[1] == 0 && all(i -> V.offsets[i-1] < V.offsets[i], 2:length(V)))
        throw(DimensionMismatch("Invalid offsets"))
    end
    length(V) < 0 && throw(DomainError("length must be > 0"))
    return V
end


@inline Base.length(V::TrajectoryVector) = V.len

@inline Base.size(V::TrajectoryVector) = (length(V),)

@propagate_inbounds function Base.getindex(V::TrajectoryVector, i::Int)
    from, to = V.offsets[i] + 1, V.offsets[i + 1]
    range = from:to
    Trajectory(V.S[range], V.O[range], V.A[range], V.R[range], V.sT[i], V.oT[i], V.done[i])
end

Base.IndexStyle(::Type{<:TrajectoryVector}) = IndexLinear()

Base.empty!(V::TrajectoryVector) = V.len = 0

# TODO HasLength/HasShape
function Base.append!(V::TrajectoryVector, iter)
    offset = V.offsets[length(V) + 1]
    _sizehint!(V, ntimesteps(V) + sum(length, iter), length(V) + length(iter))

    for τ::Trajectory in iter
        V.len += 1
        len_τ = length(τ)

        copyto!(V.S, offset + firstindex(V.S), τ.S, firstindex(τ.S), len_τ)
        copyto!(V.O, offset + firstindex(V.O), τ.O, firstindex(τ.O), len_τ)
        copyto!(V.A, offset + firstindex(V.A), τ.A, firstindex(τ.A), len_τ)
        copyto!(V.R, offset + firstindex(V.R), τ.R, firstindex(τ.R), len_τ)
        V.sT[length(V)] = τ.sT
        V.oT[length(V)] = τ.oT
        V.done[length(V)] = τ.done
        V.offsets[length(V) + 1] = offset + len_τ

        offset += len_τ
    end
    _check_consistency(V)
    return V
end

function finish!(V::TrajectoryVector)
    resize!(V.S, ntimesteps(V))
    resize!(V.O, ntimesteps(V))
    resize!(V.A, ntimesteps(V))
    resize!(V.R, ntimesteps(V))
    resize!(V.sT, length(V))
    resize!(V.oT, length(V))
    resize!(V.done, length(V))
    resize!(V.offsets, length(V) + 1)
    return V
end

function _resize!(V::TrajectoryVector, ntimesteps::Int, ntrajectories::Int)
    resize!(V.S, ntimesteps)
    resize!(V.O, ntimesteps)
    resize!(V.A, ntimesteps)
    resize!(V.R, ntimesteps)
    resize!(V.sT, ntrajectories)
    resize!(V.oT, ntrajectories)
    resize!(V.done, ntrajectories)
    resize!(V.offsets, ntrajectories + 1)
    _check_consistency(V)
    return V
end

function _sizehint!(V::TrajectoryVector, ntimesteps::Int, ntrajectories::Int)
    _check_consistency(V)
    if length(V.S) < ntimesteps
        l = nextpow(2, ntimesteps)
        resize!(V.S, l)
        resize!(V.O, l)
        resize!(V.A, l)
        resize!(V.R, l)
    end
    if length(V.offsets) < ntrajectories + 1
        l = nextpow(2, ntrajectories + 1)
        resize!(V.sT, l)
        resize!(V.oT, l)
        resize!(V.done, l)
        resize!(V.offsets, l)
    end
    return V
end

@inline ntimesteps(V::TrajectoryVector) = V.offsets[length(V) + 1]

function rollout!(policy!::P, V::TrajectoryVector, env::AbstractEnvironment, Hmax::Int) where {P}
    Hmax > 0 || throw(ArgumentError("Hmax must be > 0"))

    _sizehint!(V, ntimesteps(V) + Hmax, length(V) + 1)
    offset = V.offsets[length(V) + 1]
    @unpack S, O, A, R = V

    t::Int = 0
    done::Bool = false
    while t < Hmax && !done
        t += 1
        st = S[offset + t]::SubArray
        ot = O[offset + t]::SubArray
        at = A[offset + t]::SubArray

        getstate!(st, env)
        getobs!(ot, env)
        policy!(at, st, ot)
        R[offset+t] = getreward(st, at, ot, env)

        setaction!(env, at)
        step!(env)
        done = isdone(st, ot, env)
    end

    V.len += 1
    getstate!(V.sT[length(V)], env)
    getobs!(V.oT[length(V)], env)
    V.offsets[length(V)+1] = V.offsets[length(V)] + t

    return V
end


function collate(
    Bs::TupleN{T},
    env::AbstractEnvironment,
    ntimesteps::Integer,
) where {T<:TrajectoryVector}
    collate!(TrajectoryVector(env), Bs, env, ntimesteps)
end

function collate!(
    V::T,
    Bs::TupleN{T},
    env::AbstractEnvironment,
    ntimesteps::Integer,
) where {T<:TrajectoryVector}
    _sizehint!(V, ntimesteps, sum(length, Bs))
    togo = ntimesteps
    for B in Bs, τ in B
        if togo < length(τ)
            push!(
                V,
                Trajectory(
                    view(τ.S, 1:togo),
                    view(τ.O, 1:togo),
                    view(τ.A, 1:togo),
                    view(τ.R, 1:togo),
                    τ.S[togo + 1],
                    τ.O[togo + 1],
                    false,
                ),
            )
            break
        else
            push!(V, τ)
        end
        togo -= length(τ)
    end

    return V
end


#### TODO

function asvec(A::AbsArr{<:Any,L}) where {L}
    alongs = (ntuple(_ -> True(), Val(L - 1))..., False())
    SlicedArray(A, alongs)
end

_getindextype(A::AbstractArray, I::Tuple) = typeof(A[I...])
_getindextype(A::AbstractArray, I...) = _getindextype(A, I)
