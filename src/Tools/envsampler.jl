struct TrajectoryBuffer{B1<:ElasticBuffer,B2<:ElasticBuffer}
    trajectory::B1
    terminal::B2
end

function TrajectoryBuffer(
    env::AbstractEnvironment;
    sizehint::Maybe{Integer} = nothing,
    dtype::Maybe{DataType} = nothing,
)
    sp = dtype === nothing ? spaces(env) : adapt(dtype, spaces(env))

    trajectory = ElasticBuffer(
        states = sp.statespace,
        observations = sp.obsspace,
        actions = sp.actionspace,
        rewards = sp.rewardspace,
        evaluations = sp.evalspace,
    )

    terminal = ElasticBuffer(
        states = sp.statespace,
        observations = sp.obsspace,
        dones = Bool,
        lengths = Int,
    )

    if sizehint !== nothing
        sizehint!(trajectory, sizehint)
        sizehint!(terminal, sizehint)
    end

    TrajectoryBuffer(trajectory, terminal)
end


struct EnvSampler{E<:AbstractEnvironment,B<:TrajectoryBuffer,BA}
    envs::Vector{E}
    bufs::Vector{B}
    batch::BA
end

function EnvSampler(
    env_tconstructor::Function;
    sizehint::Maybe{Integer} = nothing,
    dtype::Maybe{DataType} = nothing,
)
    envs = [e for e in env_tconstructor(Threads.nthreads())]
    bufs = [
        TrajectoryBuffer(first(envs), sizehint = sizehint, dtype = dtype)
        for _ = 1:Threads.nthreads()
    ]
    batch = _makebatch(first(envs), sizehint = sizehint, dtype = dtype)

    EnvSampler(envs, bufs, batch)
end


function sample!(
    actionfn!,
    resetfn!,
    sampler::EnvSampler,
    nsamples::Integer;
    Hmax::Integer = nsamples,
    nthreads::Integer = Threads.nthreads(),
    copy::Bool = false,
)
    nsamples > 0 || error("`nsamples` must be > 0")
    0 < Hmax <= nsamples || error("`Hmax` must be in range (0, `nsamples`]")
    if !(0 < nthreads <= Threads.nthreads())
        error("`nthreads` must be in range (0, Threads.nthreads()]")
    end

    atomiccount = Threads.Atomic{Int}(0)
    atomicidx = Threads.Atomic{Int}(1)
    _emptybufs!(sampler)

    if nthreads == 1
        # short circuit
        _sample!(actionfn!, resetfn!, sampler, nsamples, Hmax)
    else
        @sync for _ = 1:nthreads
            Threads.@spawn _threadsample!(
                actionfn!,
                resetfn!,
                sampler,
                nsamples,
                Hmax,
                atomiccount,
            )
        end
    end

    _collate!(sampler, nsamples, copy)
    _checkbatch(sampler.batch, nsamples)
    sampler.batch
end

function _sample!(actionfn!::F, resetfn!::G, sampler, nsamples, Hmax) where {F,G}
    env = first(sampler.envs)
    buf = first(sampler.bufs)
    traj = buf.trajectory
    term = buf.terminal

    resetfn!(env)
    trajlength = n = 0
    while n < nsamples
        done = _rolloutstep!(actionfn!, traj, env)
        trajlength += 1

        if done || trajlength == Hmax
            n += trajlength
            _terminate_trajectory!(term, env, trajlength, done)
            resetfn!(env)
            trajlength = 0
        end
    end
    nothing
end

function _threadsample!(
    actionfn!::F,
    resetfn!::G,
    sampler,
    nsamples,
    Hmax,
    atomiccount,
) where {F,G}
    env = sampler.envs[Threads.threadid()]
    buf = sampler.bufs[Threads.threadid()]
    traj = buf.trajectory
    term = buf.terminal

    resetfn!(env)
    trajlength = 0
    while true
        done = _rolloutstep!(actionfn!, traj, env)
        trajlength += 1

        if done || trajlength == Hmax
            Threads.atomic_add!(atomiccount, trajlength)
            _terminate_trajectory!(term, env, trajlength, done)
            if atomiccount[] >= nsamples
                break
            else
                resetfn!(env)
                trajlength = 0
            end
        end
    end
    nothing
end

function _rolloutstep!(actionfn!::F, traj::ElasticBuffer, env::AbstractEnvironment) where {F}
    grow!(traj)
    t = lastindex(traj)
    st = view(traj.states, :, t)
    ot = view(traj.observations, :, t)
    at = view(traj.actions, :, t)

    getstate!(st, env)
    getobs!(ot, env)
    getaction!(at, env)

    actionfn!(at, st, ot)
    setaction!(env, at)

    step!(env)

    traj.rewards[t] = getreward(st, at, ot, env)
    traj.evaluations[t] = geteval(st, at, ot, env)
    return isdone(st, at, ot, env)
end

function _terminate_trajectory!(
    term::ElasticBuffer,
    env::AbstractEnvironment,
    trajlength::Integer,
    done::Bool,
)
    grow!(term)
    i = lastindex(term)
    thistraj = view(term, i)
    getstate!(vec(thistraj.states), env) # TODO
    getobs!(vec(thistraj.observations), env) # TODO
    term.dones[i] = done
    term.lengths[i] = trajlength
    term
end

function _emptybufs!(sampler::EnvSampler)
    for buf in sampler.bufs
        empty!(buf.trajectory)
        empty!(buf.terminal)
    end
    nothing
end

function _makebatch(
    env::AbstractEnvironment;
    sizehint::Maybe{Integer} = nothing,
    dtype::Maybe{DataType} = nothing,
)
    sp = dtype === nothing ? spaces(env) : adapt(dtype, spaces(env))

    batch = (
        states = BatchedArray(sp.statespace),
        observations = BatchedArray(sp.obsspace),
        actions = BatchedArray(sp.actionspace),
        rewards = BatchedArray(sp.rewardspace),
        evaluations = BatchedArray(sp.evalspace),
        terminal_states = BatchedArray(sp.statespace),
        terminal_observations = BatchedArray(sp.obsspace),
        dones = Vector{Bool}(),
    )

    sizehint !== nothing && foreach(el -> sizehint!(el, sizehint), batch)

    batch
end

function _collate!(sampler::EnvSampler, N::Integer, copy::Bool)
    batch = copy ? map(deepcopy, sampler.batch) : sampler.batch # TODO
    for b in batch
        empty!(b)
        sizehint!(b, N)
    end

    count = 0
    togo = N - count
    for buf in sampler.bufs
        trajbuf = buf.trajectory
        termbuf = buf.terminal
        from = firstindex(trajbuf)

        for episode_idx in eachindex(termbuf)
            togo = N - count
            togo == 0 && return batch

            len = termbuf.lengths[episode_idx]
            if togo < len # we only want `N` samples
                len = togo
                # because we cropped this trajectory, it didn't actually terminate early
                termbuf.dones[episode_idx] = false
            end

            until = from + len
            to = until - 1
            count += len

            traj = view(trajbuf, from:to)
            term = view(termbuf, episode_idx)

            push!(batch.states, traj.states)
            push!(batch.observations, traj.observations)
            push!(batch.actions, traj.actions)
            push!(batch.rewards, traj.rewards)
            push!(batch.evaluations, traj.evaluations)
            push!(batch.terminal_states, reshape(term.states, (:, 1))) # TODO (:, 1)
            push!(batch.terminal_observations, reshape(term.observations, (:, 1))) #TODO (:, 1)
            push!(batch.dones, termbuf.dones[episode_idx])

            from = until
        end
    end
    batch
end

function _checkbatch(b, n)
    @assert n == nsamples(b.states)
    @assert n == nsamples(b.observations)
    @assert n == nsamples(b.actions)
    @assert n == nsamples(b.rewards)
    @assert n == nsamples(b.evaluations)
    @assert length(b.dones) == length(b.terminal_states) == length(b.terminal_observations)
    nothing
end
