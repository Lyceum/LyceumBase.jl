struct Trajectory{SS<:AbsVec,OO<:AbsVec,AA<:AbsVec,RR<:AbsVec}
    S::SS
    O::OO
    A::AA
    R::RR
    done::Bool
    @inline function Trajectory{SS,OO,AA,RR}(S, O, A, R, done) where {SS,OO,AA,RR}
        _check_consistency(new(S, O, A, R, done))
    end
end

@inline function Trajectory(S::SS, O::OO, A::AA, R::RR, done) where {SS,OO,AA,RR}
    Trajectory{SS,OO,AA,RR}(S, O, A, R, done)
end

@inline function _check_consistency(τ::Trajectory)
    if !(length(τ.S) == length(τ.O) == length(τ.A) + 1 == length(τ.R) + 1)
        throw(DimensionMismatch("Not satisfied: length(S) == length(O) == length(A) + 1 == length(R) + 1"))
    end
    return τ
end

Base.length(τ::Trajectory) = length(_check_consistency(τ).A)

function Base.:(==)(τ1::Trajectory, τ2::Trajectory)
    τ1.done === τ2.done && τ1.R == τ2.R && τ1.S == τ2.S && τ1.O == τ2.O && τ1.A == τ2.A
end



mutable struct TrajectoryVector{T<:Trajectory,SS<:AbsVec,OO<:AbsVec,AA<:AbsVec,RR<:AbsVec} <:
               AbsVec{T}
    S::SS
    O::OO
    A::AA
    R::RR
    done::Vector{Bool}
    offsets::Vector{Int}
    len::Int
    @inline function TrajectoryVector{T,SS,OO,AA,RR}(
        S,
        O,
        A,
        R,
        done,
        offsets,
        len,
    ) where {T,SS,OO,AA,RR}
        _check_consistency(new(S, O, A, R, done, offsets, len))
    end
end

@inline function TrajectoryVector(
    S::SS,
    O::OO,
    A::AA,
    R::RR,
    done,
    offsets,
    len,
) where {SS,OO,AA,RR}
    I = 1:1
    T = Trajectory{_getindextype(S, I),_getindextype(O, I),_getindextype(A, I),_getindextype(R, I)}
    TrajectoryVector{T,SS,OO,AA,RR}(S, O, A, R, done, offsets, len)
end

function TrajectoryVector(env::AbstractEnvironment; sizehint::Integer = 1024)
    sizehint > 0 || throw(ArgumentError("sizehint must be ≥ 0"))
    TrajectoryVector(
        asvec(ElasticArray(undef, statespace(env), sizehint)),
        asvec(ElasticArray(undef, obsspace(env), sizehint)),
        asvec(ElasticArray(undef, actionspace(env), sizehint)),
        Array(undef, rewardspace(env), sizehint),
        Bool[],
        Int[0],
        0,
    )
end

@inline function _check_consistency(V::TrajectoryVector)
    length(V.S) != length(V.O) && error("length(S) != length(O)")
    length(V.A) != length(V.R) && error("length(A) != length(R)")
    if length(V.S) < ntimesteps(V) + length(V) || length(V.A) < ntimesteps(V)
        error("invalid dimensions")
    end
    if !(V.offsets[1] == 0 && all(i -> V.offsets[i-1] < V.offsets[i], 2:length(V)))
        throw(DimensionMismatch("Invalid offsets"))
    end
    length(V) < 0 && throw(DomainError("length must be > 0"))
    return V
end


@inline Base.length(V::TrajectoryVector) = V.len

@inline Base.size(V::TrajectoryVector) = (length(V),)

@propagate_inbounds function Base.getindex(V::TrajectoryVector, i::Int)
    range_SO, range_AR = _ranges(V, i)
    Trajectory(V.S[range_SO], V.O[range_SO], V.A[range_AR], V.R[range_AR], V.done[i])
end

Base.IndexStyle(::Type{<:TrajectoryVector}) = IndexLinear()

Base.empty!(V::TrajectoryVector) = V.len = 0

# TODO HasLength/HasShape
function Base.append!(V::TrajectoryVector, iter)
    offset_SO, offset_AR = _offsets(V, length(V) + 1)
    _sizehint!(V, ntimesteps(V) + sum(length, iter), length(V) + length(iter))

    for τ::Trajectory in iter
        len_τ = length(τ)
        copyto!(V.S, offset_SO + firstindex(V.S), τ.S, firstindex(τ.S), len_τ + 1)
        copyto!(V.O, offset_SO + firstindex(V.O), τ.O, firstindex(τ.O), len_τ + 1)
        copyto!(V.A, offset_AR + firstindex(V.A), τ.A, firstindex(τ.A), len_τ)
        copyto!(V.R, offset_AR + firstindex(V.R), τ.R, firstindex(τ.R), len_τ)

        V.len += 1
        V.done[length(V)] = τ.done
        offset_SO += len_τ + 1
        offset_AR += len_τ
        V.offsets[length(V)+1] = V.offsets[length(V)] + len_τ
    end
    return _check_consistency(V)
end

function finish!(V::TrajectoryVector)
    resize!(V.S, ntimesteps(V) + length(V))
    resize!(V.O, ntimesteps(V) + length(V))
    resize!(V.A, ntimesteps(V))
    resize!(V.R, ntimesteps(V))
    resize!(V.done, length(V))
    resize!(V.offsets, length(V) + 1)
    return V
end

function _resize!(V::TrajectoryVector, ntimesteps::Int, ntrajectories::Int)
    resize!(V.S, ntimesteps + ntrajectories)
    resize!(V.O, ntimesteps + ntrajectories)
    resize!(V.A, ntimesteps)
    resize!(V.R, ntimesteps)
    resize!(V.done, ntrajectories)
    resize!(V.offsets, ntrajectories + 1)
    _check_consistency(V)
    return V
end

function _sizehint!(V::TrajectoryVector, ntimesteps::Int, ntrajectories::Int)
    if length(V.S) < ntimesteps + ntrajectories
        l = nextpow(2, ntimesteps + ntrajectories)
        resize!(V.S, l)
        resize!(V.O, l)
    end
    length(V.A) < ntimesteps && (l = nextpow(2, ntimesteps); resize!(V.A, l); resize!(V.R, l))
    if length(V.offsets) < ntrajectories + 1
        l = nextpow(2, ntimesteps)
        resize!(V.offsets, l)
        resize!(V.done, l)
    end
    _check_consistency(V)
    return V
end

@inline function _offsets(V::TrajectoryVector, i::Int)
    @boundscheck checkbounds(Base.OneTo(length(V) + 1), i)
    offset_AR = @inbounds V.offsets[i]
    offset_SO = offset_AR + i - 1
    return offset_SO, offset_AR
end

@propagate_inbounds function _ranges(V::TrajectoryVector, i::Int)
    offset_SO, offset_AR = _offsets(V, i)
    to_SO, to_AR = _offsets(V, i + 1)
    (offset_SO+1):to_SO, (offset_AR+1):to_AR
end


@inline ntimesteps(V::TrajectoryVector) = V.offsets[length(V)+1]

function rollout!(policy!::P, V::TrajectoryVector, env::AbstractEnvironment, Hmax::Int) where {P}
    Hmax > 0 || throw(ArgumentError("Hmax must be > 0"))

    _sizehint!(V, ntimesteps(V) + Hmax, length(V) + 1)
    offset_SO, offset_AR = _offsets(V, length(V) + 1)
    @unpack S, O, A, R = V

    # TODO implement uviews
    st = S[offset_SO+1]::SubArray
    ot = O[offset_SO+1]::SubArray
    getstate!(st, env)
    getobs!(ot, env)

    t::Int = 1
    done::Bool = false
    while t <= Hmax && !done
        at = A[offset_AR+t]::SubArray
        policy!(at, st, ot)
        R[offset_AR+t] = getreward(st, at, ot, env)

        setaction!(env, at)
        step!(env)
        t += 1

        st = S[offset_SO+t]::SubArray
        ot = O[offset_SO+t]::SubArray
        getstate!(st, env)
        getobs!(ot, env)
        done = isdone(st, ot, env)
    end

    V.len += 1
    V.offsets[length(V)+1] = V.offsets[length(V)] + t - 1

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
                    view(τ.S, 1:togo+1),
                    view(τ.O, 1:togo+1),
                    view(τ.A, 1:togo),
                    view(τ.R, 1:togo),
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
